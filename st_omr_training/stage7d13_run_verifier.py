"""Independent persisted-run verifier for Stage 7-D13 specialist training.

The verifier does not train models. It reopens run/checkpoint/metrics evidence,
loads checkpoint tensors with ``weights_only=True``, reconstructs all three frozen
models, recomputes state hashes, optimizer-count evidence and acceptance, and
fails closed on any drift. A successful verification still does not write
COMPLETE; the authoritative launcher owns that final persistence action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Final, Mapping

import torch

from .stage7d13_symbol_models import (
    SpecialistMetrics,
    acceptance_passed,
    build_symbol_model,
    model_profile_fingerprint,
)
from .stage7d13_symbol_training_contract import FROZEN_D13_CONFIG, SPECIALIST_CLASSES
from .stage7d13_training import STAGE7D13_TRAINING_VERSION, training_profile_fingerprint
from .stage7d13_verified_surface import (
    D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
    D13_DERIVATIVE_BUILD_ID,
    D13_DERIVATIVE_MANIFEST_SHA256,
    D13_EXPECTED_OPTIMIZER_STEPS,
    D13_EXPECTED_OPTIMIZER_STEPS_TOTAL,
    D13_RECORD_SPLIT_COUNTS,
)
from .training_model import (
    assert_model_finite,
    count_trainable_parameters,
    model_state_sha256,
    verify_torch_runtime,
)


STAGE7D13_RUN_VERIFIER_VERSION: Final[str] = "stage7d13-persisted-run-verifier-v1"
_EXPECTED_UNCOMPLETED_TOP: Final[frozenset[str]] = frozenset(
    {"checkpoint.pt", "metrics.json", "run.json"}
)
_MAX_JSON_BYTES: Final[int] = 64 * 1024 * 1024
_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")


class Stage7D13RunVerificationError(RuntimeError):
    """Raised when persisted D13 training evidence cannot be independently proven."""


def _fail(message: str) -> None:
    raise Stage7D13RunVerificationError(message)


def _canonical_json(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Stage7D13RunVerificationError("D13 verification payload is not canonical JSON") from exc


def _read_json(path: Path, name: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{name} must be regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= _MAX_JSON_BYTES:
        _fail(f"{name} byte length is outside verifier bound")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D13RunVerificationError(f"{name} is not valid ASCII JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != raw:
        _fail(f"{name} must be canonical JSON object bytes")
    return value, raw


def _sha64(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX for c in value):
        _fail(f"{name} must be lowercase SHA-256")
    return value


def _finite(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class Stage7D13RunVerificationReceipt:
    verifier_version: str
    run_id: str
    repository_sha: str
    training_profile_fingerprint: str
    checkpoint_sha256: str
    metrics_sha256: str
    run_sha256: str
    state_sha256: dict[str, str]
    optimizer_steps: dict[str, int]
    optimizer_steps_total: int
    best_epochs: dict[str, int]
    metrics: dict[str, dict[str, float]]
    specialist_acceptance: dict[str, bool]
    acceptance: bool
    test_opened: bool
    complete_marker_present: bool
    verification_passed: bool


def _metrics_row(name: str, row: object) -> tuple[SpecialistMetrics, bool, int, str]:
    if not isinstance(row, Mapping):
        _fail(f"{name} metrics row must be object")
    if row.get("model_fingerprint") != model_profile_fingerprint(name):
        _fail(f"{name} model fingerprint mismatch")
    expected_steps = D13_EXPECTED_OPTIMIZER_STEPS[name]
    if row.get("optimizer_steps") != expected_steps:
        _fail(f"{name} optimizer steps mismatch")
    best_epoch = row.get("best_epoch")
    if not isinstance(best_epoch, int) or isinstance(best_epoch, bool) or not 0 <= best_epoch <= FROZEN_D13_CONFIG.epochs:
        _fail(f"{name} best epoch is invalid")
    initial_loss = _finite(row.get("initial_validation_loss"), f"{name}.initial_validation_loss")
    final_loss = _finite(row.get("final_validation_loss"), f"{name}.final_validation_loss")
    history = row.get("history")
    if not isinstance(history, list) or len(history) != FROZEN_D13_CONFIG.epochs:
        _fail(f"{name} history length mismatch")
    losses = [initial_loss]
    for expected_epoch, epoch_row in enumerate(history, start=1):
        if not isinstance(epoch_row, Mapping) or epoch_row.get("epoch") != expected_epoch:
            _fail(f"{name} history epoch mismatch")
        losses.append(_finite(epoch_row.get("validation_loss"), f"{name}.history.validation_loss"))
        for field in ("train_loss", "center_f1_4px", "bbox_f1_iou50", "macro_class_f1"):
            _finite(epoch_row.get(field), f"{name}.history.{field}")
    minimum = min(losses)
    expected_best_epoch = min(index for index, value in enumerate(losses) if value == minimum)
    if best_epoch != expected_best_epoch:
        _fail(f"{name} best epoch does not match minimum validation loss")
    if not math.isclose(final_loss, minimum, rel_tol=0.0, abs_tol=1e-12):
        _fail(f"{name} final validation loss does not reproduce selected minimum")
    metrics_raw = row.get("metrics")
    if not isinstance(metrics_raw, Mapping):
        _fail(f"{name} final metrics missing")
    metrics = SpecialistMetrics(
        class_aware_center_f1_4px=_finite(
            metrics_raw.get("class_aware_center_f1_4px"),
            f"{name}.center_f1",
        ),
        class_aware_bbox_f1_iou50=_finite(
            metrics_raw.get("class_aware_bbox_f1_iou50"),
            f"{name}.bbox_f1",
        ),
        macro_class_f1=_finite(metrics_raw.get("macro_class_f1"), f"{name}.macro_f1"),
    )
    for value in asdict(metrics).values():
        if not 0.0 <= value <= 1.0:
            _fail(f"{name} final metric leaves [0,1]")
    accepted = acceptance_passed(name, metrics)
    if row.get("accepted") is not accepted:
        _fail(f"{name} persisted acceptance mismatch")
    state_sha = _sha64(row.get("final_state_sha256"), f"{name}.final_state_sha256")
    parameter_count = row.get("parameter_count")
    if not isinstance(parameter_count, int) or isinstance(parameter_count, bool) or parameter_count <= 0:
        _fail(f"{name} parameter count invalid")
    return metrics, accepted, best_epoch, state_sha


def verify_stage7d13_run(
    run_root: str | Path,
    *,
    require_complete: bool = False,
) -> Stage7D13RunVerificationReceipt:
    """Independently verify a persisted D13 run without training or mutation."""
    root = Path(run_root)
    if root.is_symlink() or not root.is_dir():
        _fail("D13 run root must be regular non-symlink directory")
    names = {p.name for p in root.iterdir()}
    expected = set(_EXPECTED_UNCOMPLETED_TOP)
    if require_complete:
        expected.update({"verification.json", "COMPLETE"})
    if names != expected:
        _fail("D13 run top-level surface differs from expected state")
    if (root / "COMPLETE").exists() != require_complete:
        _fail("D13 COMPLETE presence disagrees with verifier mode")

    verify_torch_runtime()
    run, run_raw = _read_json(root / "run.json", "D13 run.json")
    metrics_doc, metrics_raw = _read_json(root / "metrics.json", "D13 metrics.json")
    run_sha = sha256(run_raw).hexdigest()
    metrics_sha = sha256(metrics_raw).hexdigest()
    if run.get("version") != STAGE7D13_TRAINING_VERSION or metrics_doc.get("version") != STAGE7D13_TRAINING_VERSION:
        _fail("D13 training version mismatch")
    run_id = _sha64(run.get("run_id"), "run_id")
    if metrics_doc.get("run_id") != run_id:
        _fail("D13 run/metrics run_id mismatch")
    profile = training_profile_fingerprint()
    if run.get("training_profile_fingerprint") != profile or metrics_doc.get("training_profile_fingerprint") != profile:
        _fail("D13 training profile fingerprint mismatch")
    derivative = run.get("derivative")
    if not isinstance(derivative, Mapping):
        _fail("D13 derivative binding missing")
    expected_derivative = {
        "build_id": D13_DERIVATIVE_BUILD_ID,
        "manifest_sha256": D13_DERIVATIVE_MANIFEST_SHA256,
        "artifact_binding_sha256": D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
        "train_records": D13_RECORD_SPLIT_COUNTS["train"],
        "validation_records": D13_RECORD_SPLIT_COUNTS["validation"],
        "test_records": 0,
    }
    if dict(derivative) != expected_derivative:
        _fail("D13 persisted derivative binding mismatch")
    if run.get("test_opened") is not False:
        _fail("D13 run indicates TEST access")
    if run.get("complete_marker_written") is not False:
        _fail("D13 training runner claimed COMPLETE")
    if run.get("optimizer_steps") != D13_EXPECTED_OPTIMIZER_STEPS or run.get("optimizer_steps_total") != D13_EXPECTED_OPTIMIZER_STEPS_TOTAL:
        _fail("D13 run optimizer-step evidence mismatch")
    checkpoint_sha = _sha64(run.get("checkpoint_sha256"), "checkpoint_sha256")
    if run.get("metrics_sha256") != metrics_sha:
        _fail("D13 metrics SHA binding mismatch")

    checkpoint_path = root / "checkpoint.pt"
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        _fail("D13 checkpoint must be regular file")
    actual_checkpoint_sha = sha256(checkpoint_path.read_bytes()).hexdigest()
    if actual_checkpoint_sha != checkpoint_sha:
        _fail("D13 checkpoint SHA mismatch")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Stage7D13RunVerificationError("D13 checkpoint safe load failed") from exc
    if not isinstance(checkpoint, dict) or set(checkpoint) != set(SPECIALIST_CLASSES):
        _fail("D13 checkpoint specialist set mismatch")

    specialists = metrics_doc.get("specialists")
    if not isinstance(specialists, Mapping) or set(specialists) != set(SPECIALIST_CLASSES):
        _fail("D13 metrics specialist set mismatch")
    state_hashes: dict[str, str] = {}
    best_epochs: dict[str, int] = {}
    metric_values: dict[str, dict[str, float]] = {}
    accepted_map: dict[str, bool] = {}
    for name in SPECIALIST_CLASSES:
        metrics, accepted, best_epoch, expected_state_sha = _metrics_row(name, specialists[name])
        model = build_symbol_model(name)
        state = checkpoint[name]
        if not isinstance(state, dict):
            _fail(f"{name} checkpoint state must be dict")
        try:
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise Stage7D13RunVerificationError(f"{name} checkpoint state load failed") from exc
        assert_model_finite(model)
        state_sha = model_state_sha256(model)
        if state_sha != expected_state_sha:
            _fail(f"{name} checkpoint state hash mismatch")
        row = specialists[name]
        assert isinstance(row, Mapping)
        if row.get("parameter_count") != count_trainable_parameters(model):
            _fail(f"{name} parameter count does not reproduce")
        state_hashes[name] = state_sha
        best_epochs[name] = best_epoch
        metric_values[name] = asdict(metrics)
        accepted_map[name] = accepted

    acceptance = all(accepted_map.values())
    if metrics_doc.get("acceptance") is not acceptance:
        _fail("D13 overall persisted acceptance mismatch")

    if require_complete:
        verification_doc, verification_raw = _read_json(root / "verification.json", "D13 verification.json")
        if verification_doc.get("verifier_version") != STAGE7D13_RUN_VERIFIER_VERSION:
            _fail("D13 persisted verifier version mismatch")
        if verification_doc.get("run_sha256") != run_sha:
            _fail("D13 verification/run binding mismatch")
        complete_raw = _read_json(root / "COMPLETE", "D13 COMPLETE")[0]
        if complete_raw.get("run_id") != run_id or complete_raw.get("verification_sha256") != sha256(verification_raw).hexdigest():
            _fail("D13 COMPLETE binding mismatch")

    return Stage7D13RunVerificationReceipt(
        verifier_version=STAGE7D13_RUN_VERIFIER_VERSION,
        run_id=run_id,
        repository_sha=str(run.get("repository_sha")),
        training_profile_fingerprint=profile,
        checkpoint_sha256=checkpoint_sha,
        metrics_sha256=metrics_sha,
        run_sha256=run_sha,
        state_sha256=state_hashes,
        optimizer_steps=dict(D13_EXPECTED_OPTIMIZER_STEPS),
        optimizer_steps_total=D13_EXPECTED_OPTIMIZER_STEPS_TOTAL,
        best_epochs=best_epochs,
        metrics=metric_values,
        specialist_acceptance=accepted_map,
        acceptance=acceptance,
        test_opened=False,
        complete_marker_present=require_complete,
        verification_passed=True,
    )
