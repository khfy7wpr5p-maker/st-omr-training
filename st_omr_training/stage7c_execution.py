"""Authoritative Stage 7-C execution gate with source/runtime provenance verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import subprocess
from typing import Final

import torch

from .dataset_builder import SyntheticDatasetBuild
from .stage7c_dataset import STAGE7C_BASELINE_DATASET_CONFIG_FINGERPRINT
from .stage7c_profile import (
    STAGE7C_FROZEN_MODEL_CONFIG,
    STAGE7C_FROZEN_PREPROCESS_CONFIG,
    STAGE7C_FROZEN_RUN_CONFIG,
    STAGE7C_FROZEN_RUN_FINGERPRINT,
    STAGE7C_FROZEN_TRAINER_CONFIG,
)
from .training_model import (
    BaselineModelConfig,
    assert_model_finite,
    build_baseline_model,
    model_config_fingerprint,
    model_state_sha256,
    verify_torch_runtime,
)
from .training_run import BaselineRunResult, run_baseline_training


STAGE7C_EXECUTION_GATE_VERSION: Final[str] = "stage7c-authoritative-execution-v1"
EXPECTED_REPOSITORY_FULL_NAME: Final[str] = "khfy7wpr5p-maker/st-omr-training"
EXPECTED_REPOSITORY_ORIGINS: Final[frozenset[str]] = frozenset(
    {
        "https://github.com/khfy7wpr5p-maker/st-omr-training",
        "https://github.com/khfy7wpr5p-maker/st-omr-training.git",
        "git@github.com:khfy7wpr5p-maker/st-omr-training.git",
    }
)
REQUIRED_STAGE7C_RUNTIME: Final[dict[str, str]] = {
    "lxml": "6.1.1",
    "verovio": "6.2.1",
    "CairoSVG": "2.8.2",
    "Pillow": "12.3.0",
    "torch": "2.13.0+cpu",
}
_HEX = frozenset("0123456789abcdef")


class Stage7CExecutionError(RuntimeError):
    """Raised when authoritative Stage 7-C provenance cannot be established."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_sha(name: str, value: object, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in _HEX for character in value)
    ):
        raise Stage7CExecutionError(f"{name} is not a canonical lowercase SHA value")
    return value


def _run_git(repository_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repository_root), *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Stage7CExecutionError("unable to execute git provenance check") from exc
    if completed.returncode != 0:
        raise Stage7CExecutionError("git provenance check failed")
    return completed.stdout.strip()


def verify_repository_checkout(repository_root: str | Path) -> str:
    """Return exact HEAD only for a clean, explicit Git repository root."""

    if not isinstance(repository_root, (str, Path)):
        raise TypeError("repository_root must be str or pathlib.Path")
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise Stage7CExecutionError("repository_root is not a directory")

    top_level_text = _run_git(root, "rev-parse", "--show-toplevel")
    top_level = Path(top_level_text).resolve()
    if top_level != root:
        raise Stage7CExecutionError("repository_root must be the exact Git top-level directory")

    head = _require_sha("repository HEAD", _run_git(root, "rev-parse", "--verify", "HEAD"), 40)
    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise Stage7CExecutionError("repository checkout is not clean; authoritative run refused")
    return head


def verify_authoritative_repository(repository_root: str | Path) -> tuple[str, str]:
    """Bind evidence to the Git repository that contains this executing package source."""

    if not isinstance(repository_root, (str, Path)):
        raise TypeError("repository_root must be str or pathlib.Path")
    root = Path(repository_root).resolve()
    executing_root = Path(__file__).resolve().parents[1]
    if root != executing_root:
        raise Stage7CExecutionError(
            "authoritative repository root must be the source tree executing Stage 7-C"
        )
    head = verify_repository_checkout(root)
    origin = _run_git(root, "remote", "get-url", "origin")
    if origin not in EXPECTED_REPOSITORY_ORIGINS:
        raise Stage7CExecutionError(
            f"repository origin is not the expected {EXPECTED_REPOSITORY_FULL_NAME} remote"
        )
    return head, origin


def verify_stage7c_runtime() -> dict[str, str]:
    """Fail closed unless every Stage 7-C runtime dependency is exactly pinned."""

    verify_torch_runtime()
    actual: dict[str, str] = {}
    for distribution, expected in REQUIRED_STAGE7C_RUNTIME.items():
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise Stage7CExecutionError(
                f"required Stage 7-C runtime package is missing: {distribution}"
            ) from exc
        if version != expected:
            raise Stage7CExecutionError(
                f"Stage 7-C runtime mismatch for {distribution}: expected {expected}, got {version}"
            )
        actual[distribution] = version
    return actual


@dataclass(frozen=True, slots=True)
class VerifiedBaselineRunResult:
    result: BaselineRunResult
    verification_path: Path
    verification_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.result, BaselineRunResult):
            raise Stage7CExecutionError("result must be BaselineRunResult")
        if not isinstance(self.verification_path, Path) or not self.verification_path.is_file():
            raise Stage7CExecutionError("verification_path must reference a written file")
        _require_sha("verification_sha256", self.verification_sha256, 64)
        if sha256(self.verification_path.read_bytes()).hexdigest() != self.verification_sha256:
            raise Stage7CExecutionError("verification file hash mismatch")


def _model_config_from_evidence(model_payload: dict[str, object]) -> BaselineModelConfig:
    normalized = dict(model_payload)
    conv_channels = normalized.get("conv_channels")
    if not isinstance(conv_channels, list) or len(conv_channels) != 2:
        raise Stage7CExecutionError(
            "metrics evidence model conv_channels must be the canonical two-item JSON array"
        )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in conv_channels):
        raise Stage7CExecutionError("metrics evidence model conv_channels contains a non-integer")
    normalized["conv_channels"] = tuple(conv_channels)
    try:
        return BaselineModelConfig(**normalized)
    except (TypeError, ValueError) as exc:
        raise Stage7CExecutionError("metrics evidence model configuration was rejected") from exc


def _load_and_verify_checkpoint(
    checkpoint_path: Path,
    result: BaselineRunResult,
    evidence: dict[str, object],
) -> str:
    configuration = evidence.get("configuration")
    if not isinstance(configuration, dict):
        raise Stage7CExecutionError("metrics evidence is missing model configuration")
    model_payload = configuration.get("model")
    if not isinstance(model_payload, dict):
        raise Stage7CExecutionError("metrics evidence model configuration is invalid")
    model_config = _model_config_from_evidence(model_payload)
    if model_config != STAGE7C_FROZEN_MODEL_CONFIG:
        raise Stage7CExecutionError("checkpoint model config differs from frozen Stage 7-C profile")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Stage7CExecutionError("selected checkpoint cannot be safely reloaded") from exc
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "epoch",
        "model_fingerprint",
        "model_state_dict",
    }:
        raise Stage7CExecutionError("selected checkpoint has an unexpected structure")
    if checkpoint.get("epoch") != result.best_epoch:
        raise Stage7CExecutionError("selected checkpoint epoch differs from run evidence")
    expected_fingerprint = model_config_fingerprint(model_config)
    if checkpoint.get("model_fingerprint") != expected_fingerprint:
        raise Stage7CExecutionError("selected checkpoint model fingerprint mismatch")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise Stage7CExecutionError("selected checkpoint model state is invalid")

    model = build_baseline_model(model_config, seed=0)
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise Stage7CExecutionError("selected checkpoint model state cannot be loaded strictly") from exc
    assert_model_finite(model)
    state_sha = model_state_sha256(model)

    checkpoint_evidence = evidence.get("checkpoint")
    if not isinstance(checkpoint_evidence, dict):
        raise Stage7CExecutionError("metrics evidence is missing checkpoint provenance")
    if checkpoint_evidence.get("sha256") != result.checkpoint_sha256:
        raise Stage7CExecutionError("metrics evidence checkpoint file hash mismatch")
    if checkpoint_evidence.get("state_sha256") != state_sha:
        raise Stage7CExecutionError("reloaded checkpoint state hash differs from metrics evidence")
    if checkpoint_evidence.get("filename") != checkpoint_path.name:
        raise Stage7CExecutionError("metrics evidence checkpoint filename mismatch")
    return state_sha


def _verify_run_evidence(
    result: BaselineRunResult,
    build: SyntheticDatasetBuild,
    repository_sha: str,
    runtime_versions: dict[str, str],
) -> tuple[dict[str, object], str]:
    if build.config_fingerprint != STAGE7C_BASELINE_DATASET_CONFIG_FINGERPRINT:
        raise Stage7CExecutionError("run did not use the frozen Stage 7-C dataset profile")

    metrics_path = result.run_directory / f"metrics-{result.metrics_sha256}.json"
    checkpoint_path = result.run_directory / f"checkpoint-{result.checkpoint_sha256}.pt"
    complete_path = result.run_directory / "COMPLETE"
    if not metrics_path.is_file() or not checkpoint_path.is_file() or not complete_path.is_file():
        raise Stage7CExecutionError("completed run is missing required evidence artifacts")
    metrics_bytes = metrics_path.read_bytes()
    if sha256(metrics_bytes).hexdigest() != result.metrics_sha256:
        raise Stage7CExecutionError("metrics artifact hash mismatch at authoritative gate")
    if sha256(checkpoint_path.read_bytes()).hexdigest() != result.checkpoint_sha256:
        raise Stage7CExecutionError("checkpoint artifact hash mismatch at authoritative gate")
    expected_complete = f"{result.metrics_sha256}  {metrics_path.name}\n".encode("ascii")
    if complete_path.read_bytes() != expected_complete:
        raise Stage7CExecutionError("COMPLETE marker does not exactly bind the metrics artifact")

    try:
        evidence = json.loads(metrics_bytes.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7CExecutionError("metrics evidence is not valid canonical JSON text") from exc
    if not isinstance(evidence, dict):
        raise Stage7CExecutionError("metrics evidence root must be an object")
    if _canonical_json_bytes(evidence) != metrics_bytes:
        raise Stage7CExecutionError("metrics evidence is not canonical JSON")
    if evidence.get("repository_sha") != repository_sha:
        raise Stage7CExecutionError("metrics evidence repository SHA differs from verified checkout")
    dataset = evidence.get("dataset")
    if not isinstance(dataset, dict):
        raise Stage7CExecutionError("metrics evidence is missing dataset provenance")
    if dataset.get("build_id") != build.build_id or dataset.get("manifest_sha256") != build.manifest_sha256:
        raise Stage7CExecutionError("metrics evidence dataset provenance mismatch")
    if dataset.get("config_fingerprint") != STAGE7C_BASELINE_DATASET_CONFIG_FINGERPRINT:
        raise Stage7CExecutionError("metrics evidence dataset profile fingerprint mismatch")
    fingerprints = evidence.get("fingerprints")
    if not isinstance(fingerprints, dict) or fingerprints.get("run") != STAGE7C_FROZEN_RUN_FINGERPRINT:
        raise Stage7CExecutionError("metrics evidence run profile fingerprint mismatch")
    runtime = evidence.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("dependencies") != runtime_versions:
        raise Stage7CExecutionError("metrics evidence runtime provenance mismatch")
    if evidence.get("sealed_test_split_opened") is not False:
        raise Stage7CExecutionError("metrics evidence does not prove the sealed test split stayed closed")

    state_sha = _load_and_verify_checkpoint(checkpoint_path, result, evidence)
    return evidence, state_sha


def run_verified_baseline_training(
    build: object,
    dataset_root: str | Path,
    run_root: str | Path,
    repository_root: str | Path,
) -> VerifiedBaselineRunResult:
    """Run the one frozen Stage 7-C profile with source-bound clean provenance."""

    if not isinstance(build, SyntheticDatasetBuild):
        raise TypeError("build must be SyntheticDatasetBuild")
    if build.config_fingerprint != STAGE7C_BASELINE_DATASET_CONFIG_FINGERPRINT:
        raise Stage7CExecutionError("authoritative run requires the frozen Stage 7-C dataset profile")

    repository_sha, repository_origin = verify_authoritative_repository(repository_root)
    runtime_versions = verify_stage7c_runtime()

    result = run_baseline_training(
        build,
        dataset_root,
        run_root,
        repository_sha=repository_sha,
        run_config=STAGE7C_FROZEN_RUN_CONFIG,
        model_config=STAGE7C_FROZEN_MODEL_CONFIG,
        trainer_config=STAGE7C_FROZEN_TRAINER_CONFIG,
        preprocess_config=STAGE7C_FROZEN_PREPROCESS_CONFIG,
    )

    ending_sha, ending_origin = verify_authoritative_repository(repository_root)
    if ending_sha != repository_sha or ending_origin != repository_origin:
        raise Stage7CExecutionError("repository identity changed during Stage 7-C execution")
    ending_runtime = verify_stage7c_runtime()
    if ending_runtime != runtime_versions:
        raise Stage7CExecutionError("runtime dependency identity changed during Stage 7-C execution")

    _evidence, reloaded_state_sha = _verify_run_evidence(
        result,
        build,
        repository_sha,
        runtime_versions,
    )
    verification_payload = {
        "schema_version": "stage7c-authoritative-verification-v1",
        "execution_gate_version": STAGE7C_EXECUTION_GATE_VERSION,
        "repository_full_name": EXPECTED_REPOSITORY_FULL_NAME,
        "repository_origin": repository_origin,
        "repository_sha": repository_sha,
        "dataset_build_id": build.build_id,
        "dataset_config_fingerprint": build.config_fingerprint,
        "manifest_sha256": build.manifest_sha256,
        "run_profile_fingerprint": STAGE7C_FROZEN_RUN_FINGERPRINT,
        "metrics_sha256": result.metrics_sha256,
        "checkpoint_sha256": result.checkpoint_sha256,
        "checkpoint_state_sha256": reloaded_state_sha,
        "runtime": runtime_versions,
        "run_result": {
            "run_id": result.run_id,
            "best_epoch": result.best_epoch,
            "training_steps": result.training_steps,
            "untrained_validation_loss": result.untrained_validation_loss,
            "best_validation_loss": result.best_validation_loss,
            "prediction_metrics": asdict(result.prediction_metrics),
        },
        "frozen_stage7c_dataset_verified": True,
        "frozen_stage7c_training_profile_verified": True,
        "source_tree_bound_to_executing_package": True,
        "source_clean_before_and_after": True,
        "checkpoint_reloaded_strictly": True,
        "metrics_canonical_json_verified": True,
        "complete_marker_verified": True,
        "sealed_test_split_opened": False,
    }
    verification_bytes = _canonical_json_bytes(verification_payload)
    verification_sha = sha256(verification_bytes).hexdigest()
    verification_path = result.run_directory / f"VERIFIED-{verification_sha}.json"
    if verification_path.exists():
        raise Stage7CExecutionError("authoritative verification marker already exists")
    verification_path.write_bytes(verification_bytes)
    if sha256(verification_path.read_bytes()).hexdigest() != verification_sha:
        raise Stage7CExecutionError("authoritative verification marker changed after writing")

    final_sha, final_origin = verify_authoritative_repository(repository_root)
    if final_sha != repository_sha or final_origin != repository_origin:
        verification_path.unlink(missing_ok=True)
        raise Stage7CExecutionError("repository identity changed while final verification was written")

    return VerifiedBaselineRunResult(
        result=result,
        verification_path=verification_path,
        verification_sha256=verification_sha,
    )
