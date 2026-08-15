"""Independent persisted-output verifier for completed Stage 7-D11 runs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Final

import torch

from .stage7d11_barline_meter_training import (
    EXPECTED_D10_REPOSITORY_SHA,
    FROZEN_D11_CONFIG,
    STAGE7D11_METRICS_SCHEMA,
    STAGE7D11_VERIFICATION_SCHEMA,
    STAGE7D11_VERSION,
    BarlineMetrics,
    MeterMetrics,
    Stage7D11TrainingError,
    acceptance_from_metrics,
    build_barline_refiner,
    build_meter_refiner,
    stage7d11_profile_fingerprint,
)
from .training_model import count_trainable_parameters, model_state_sha256
from .stage7d9_structure_refinement_contract import D9_ACCEPTANCE, EXPECTED_D7_STRUCTURE_STATE_SHA256

_HEX40: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
_HEX64: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_OPTIMIZER_STEPS_PER_REFINER: Final[int] = 2464
EXPECTED_TRAIN_RECORDS: Final[int] = 19_680
EXPECTED_VALIDATION_RECORDS: Final[int] = 2_448


@dataclass(frozen=True, slots=True)
class Stage7D11VerificationReceipt:
    run_id: str
    repository_sha: str
    profile_fingerprint: str
    d10_manifest_sha256: str
    d10_artifact_binding_sha256: str
    checkpoint_sha256: str
    metrics_sha256: str
    verification_sha256: str
    barline_state_sha256: str
    meter_state_sha256: str
    barline_optimizer_steps: int
    meter_optimizer_steps: int
    acceptance_passed: bool
    test_records: int
    test_opened: bool
    core_loaded: bool
    core_mutated: bool


def _fail(message: str) -> None:
    raise Stage7D11TrainingError(message)


def _canonical_json(payload: object) -> bytes:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Stage7D11TrainingError("D11 verification payload is not canonical JSON serializable") from exc


def _sha64(value: object, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        _fail(f"{name} must be canonical lowercase SHA-256")
    return value


def _git_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        _fail(f"{name} must be canonical 40-character Git SHA")
    return value


def _read_canonical_json(path: Path, name: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{name} must be a regular file")
    raw = path.read_bytes()
    if not 1 <= len(raw) <= 32 * 1024 * 1024:
        _fail(f"{name} byte length is outside D11 verification bounds")
    try:
        payload = json.loads(raw.decode("ascii"), parse_constant=lambda token: _fail(f"non-finite constant in {name}: {token}"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D11TrainingError(f"{name} is not valid ASCII JSON") from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        _fail(f"{name} must be canonical JSON object bytes")
    return payload, raw


def _single_hash_file(root: Path, prefix: str, suffix: str) -> tuple[Path, str]:
    matches = sorted(root.glob(f"{prefix}-*{suffix}"))
    if len(matches) != 1:
        _fail(f"D11 run must contain exactly one {prefix} artifact")
    path = matches[0]
    if path.is_symlink() or not path.is_file():
        _fail(f"D11 {prefix} artifact must be a regular file")
    stem = path.name[len(prefix) + 1 : -len(suffix)]
    expected = _sha64(stem, f"D11 {prefix} filename hash")
    actual = sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        _fail(f"D11 {prefix} content hash differs from filename")
    return path, actual


def _finite_metric(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{name} must be numeric")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        _fail(f"{name} must be in [0,1]")
    return number


def _expected_run_id(repository_sha: str, manifest_sha: str, binding_sha: str) -> str:
    identity = {
        "version": STAGE7D11_VERSION,
        "repository_sha": repository_sha,
        "profile_fingerprint": stage7d11_profile_fingerprint(FROZEN_D11_CONFIG),
        "d10_manifest_sha256": manifest_sha,
        "d10_artifact_binding_sha256": binding_sha,
    }
    return sha256(_canonical_json(identity)).hexdigest()


def verify_stage7d11_run(
    run_directory: str | Path,
    *,
    expected_repository_sha: str,
    expected_d10_manifest_sha256: str,
    expected_d10_artifact_binding_sha256: str,
) -> Stage7D11VerificationReceipt:
    """Reopen a completed D11 run without trusting the trainer return object."""

    root = Path(run_directory)
    if root.is_symlink() or not root.is_dir():
        _fail("D11 run_directory must be a regular directory")

    repository_sha = _git_sha(expected_repository_sha, "expected repository SHA")
    manifest_sha = _sha64(expected_d10_manifest_sha256, "expected D10 manifest SHA-256")
    binding_sha = _sha64(expected_d10_artifact_binding_sha256, "expected D10 artifact-binding SHA-256")
    run_id = _expected_run_id(repository_sha, manifest_sha, binding_sha)
    if root.name != run_id:
        _fail("D11 run directory identity mismatch")

    checkpoint_path, checkpoint_sha = _single_hash_file(root, "checkpoint", ".pt")
    metrics_path, metrics_sha = _single_hash_file(root, "metrics", ".json")
    verification_path, verification_sha = _single_hash_file(root, "verification", ".json")
    complete_path = root / "COMPLETE"
    if complete_path.is_symlink() or not complete_path.is_file():
        _fail("D11 COMPLETE marker is missing")

    expected_files = {checkpoint_path.name, metrics_path.name, verification_path.name, "COMPLETE"}
    actual_files = {item.name for item in root.iterdir()}
    if actual_files != expected_files:
        _fail("D11 run contains missing or unexpected top-level artifacts")

    expected_complete = (
        f"{verification_sha}  {verification_path.name}\n"
        f"{metrics_sha}  {metrics_path.name}\n"
        f"{checkpoint_sha}  {checkpoint_path.name}\n"
    ).encode("ascii")
    if complete_path.read_bytes() != expected_complete:
        _fail("D11 COMPLETE marker does not bind exact artifacts")

    metrics, _ = _read_canonical_json(metrics_path, "D11 metrics")
    verification, _ = _read_canonical_json(verification_path, "D11 verification")

    if metrics.get("schema_version") != STAGE7D11_METRICS_SCHEMA or metrics.get("stage7d11_version") != STAGE7D11_VERSION:
        _fail("D11 metrics schema/version mismatch")
    if verification.get("schema_version") != STAGE7D11_VERIFICATION_SCHEMA or verification.get("stage7d11_version") != STAGE7D11_VERSION:
        _fail("D11 verification schema/version mismatch")

    profile = stage7d11_profile_fingerprint(FROZEN_D11_CONFIG)
    for payload, name in ((metrics, "metrics"), (verification, "verification")):
        if payload.get("repository_sha") != repository_sha:
            _fail(f"D11 {name} repository SHA mismatch")
        if payload.get("profile_fingerprint") != profile:
            _fail(f"D11 {name} profile fingerprint mismatch")

    d10 = metrics.get("d10")
    if not isinstance(d10, dict):
        _fail("D11 metrics D10 binding is missing")
    if d10.get("repository_sha") != EXPECTED_D10_REPOSITORY_SHA or d10.get("manifest_sha256") != manifest_sha or d10.get("artifact_binding_sha256") != binding_sha or d10.get("roi_records") != 22_128 or d10.get("test_records") != 0:
        _fail("D11 metrics D10 binding mismatch")

    if verification.get("d10_manifest_sha256") != manifest_sha or verification.get("d10_artifact_binding_sha256") != binding_sha:
        _fail("D11 verification D10 binding mismatch")
    if verification.get("metrics_sha256") != metrics_sha or verification.get("checkpoint_sha256") != checkpoint_sha:
        _fail("D11 verification artifact hash binding mismatch")

    barline = metrics.get("barline")
    meter = metrics.get("meter")
    if not isinstance(barline, dict) or not isinstance(meter, dict):
        _fail("D11 refiner metrics are missing")
    if barline.get("task") != "barline" or meter.get("task") != "meter":
        _fail("D11 refiner task identity mismatch")

    barline_state = _sha64(barline.get("state_sha256"), "barline state SHA-256")
    meter_state = _sha64(meter.get("state_sha256"), "meter state SHA-256")
    if verification.get("barline_state_sha256") != barline_state or verification.get("meter_state_sha256") != meter_state:
        _fail("D11 verification state hash binding mismatch")

    barline_steps = barline.get("optimizer_steps")
    meter_steps = meter.get("optimizer_steps")
    if barline_steps != EXPECTED_OPTIMIZER_STEPS_PER_REFINER or meter_steps != EXPECTED_OPTIMIZER_STEPS_PER_REFINER:
        _fail("D11 metrics optimizer step count mismatch")
    if verification.get("barline_optimizer_steps") != barline_steps or verification.get("meter_optimizer_steps") != meter_steps:
        _fail("D11 verification optimizer step count mismatch")
    if verification.get("train_records") != EXPECTED_TRAIN_RECORDS or verification.get("validation_records") != EXPECTED_VALIDATION_RECORDS or verification.get("test_records") != 0 or verification.get("test_opened") is not False:
        _fail("D11 split/test verification mismatch")

    if metrics.get("sealed_test_split_opened") is not False:
        _fail("D11 metrics indicate TEST access")
    if metrics.get("accepted_d7_structure_state_sha256") != EXPECTED_D7_STRUCTURE_STATE_SHA256 or metrics.get("accepted_d7_structure_core_loaded") is not False:
        _fail("D11 metrics D7 core freeze evidence mismatch")
    if verification.get("accepted_d7_structure_core_loaded") is not False or verification.get("accepted_d7_structure_core_mutated") is not False:
        _fail("D11 verification indicates D7 core access or mutation")
    if verification.get("checkpoint_reload_verified") is not True or verification.get("repository_stable_during_run") is not True or verification.get("runtime_stable_during_run") is not True:
        _fail("D11 verification runtime/reload stability flags are incomplete")

    barline_validation = barline.get("validation_metrics")
    meter_validation = meter.get("validation_metrics")
    if not isinstance(barline_validation, dict) or not isinstance(meter_validation, dict):
        _fail("D11 validation metrics are missing")
    barline_metrics = BarlineMetrics(
        strict_dice=_finite_metric(barline_validation.get("strict_dice"), "barline strict Dice"),
        tolerant_f1_2px=_finite_metric(barline_validation.get("tolerant_f1_2px"), "barline tolerant F1"),
    )
    meter_metrics = MeterMetrics(
        macro_f1=_finite_metric(meter_validation.get("macro_f1"), "meter macro F1"),
        positive_localization_f1_2px=_finite_metric(meter_validation.get("positive_localization_f1_2px"), "meter localization F1"),
    )
    accepted = acceptance_from_metrics(barline_metrics, meter_metrics)
    if metrics.get("acceptance_thresholds") != {
        "max_total_new_trainable_parameters": D9_ACCEPTANCE.max_total_new_trainable_parameters,
        "barline_min_strict_dice_milli": D9_ACCEPTANCE.barline_min_strict_dice_milli,
        "barline_min_tolerant_f1_2px_milli": D9_ACCEPTANCE.barline_min_tolerant_f1_2px_milli,
        "meter_min_macro_f1_milli": D9_ACCEPTANCE.meter_min_macro_f1_milli,
        "meter_min_positive_localization_f1_2px_milli": D9_ACCEPTANCE.meter_min_positive_localization_f1_2px_milli,
        "test_records": D9_ACCEPTANCE.test_records,
        "core_model_mutation_allowed": D9_ACCEPTANCE.core_model_mutation_allowed,
    }:
        _fail("D11 frozen acceptance thresholds mismatch")
    if metrics.get("acceptance_passed") is not accepted or verification.get("acceptance_passed") is not accepted:
        _fail("D11 acceptance result does not reproduce from persisted metrics")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Stage7D11TrainingError("D11 persisted checkpoint cannot be safely loaded") from exc
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"barline_state_dict", "meter_state_dict"}:
        _fail("D11 persisted checkpoint root is invalid")
    barline_model = build_barline_refiner(FROZEN_D11_CONFIG)
    meter_model = build_meter_refiner(FROZEN_D11_CONFIG)
    try:
        barline_model.load_state_dict(checkpoint["barline_state_dict"], strict=True)
        meter_model.load_state_dict(checkpoint["meter_state_dict"], strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise Stage7D11TrainingError("D11 persisted checkpoint state cannot be strictly reloaded") from exc
    if model_state_sha256(barline_model) != barline_state or model_state_sha256(meter_model) != meter_state:
        _fail("D11 persisted checkpoint state hash mismatch")
    if count_trainable_parameters(barline_model) != barline.get("parameter_count") or count_trainable_parameters(meter_model) != meter.get("parameter_count"):
        _fail("D11 persisted parameter count mismatch")

    return Stage7D11VerificationReceipt(
        run_id=run_id,
        repository_sha=repository_sha,
        profile_fingerprint=profile,
        d10_manifest_sha256=manifest_sha,
        d10_artifact_binding_sha256=binding_sha,
        checkpoint_sha256=checkpoint_sha,
        metrics_sha256=metrics_sha,
        verification_sha256=verification_sha,
        barline_state_sha256=barline_state,
        meter_state_sha256=meter_state,
        barline_optimizer_steps=barline_steps,
        meter_optimizer_steps=meter_steps,
        acceptance_passed=accepted,
        test_records=0,
        test_opened=False,
        core_loaded=False,
        core_mutated=False,
    )
