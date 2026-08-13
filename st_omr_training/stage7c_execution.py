"""Authoritative Stage 7-C execution gate with source/runtime provenance verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import subprocess
from typing import Final

from .dataset_builder import SyntheticDatasetBuild
from .training_data import InputPreprocessConfig
from .training_model import BaselineModelConfig, TrainerConfig, verify_torch_runtime
from .training_run import BaselineRunConfig, BaselineRunError, BaselineRunResult, run_baseline_training


STAGE7C_EXECUTION_GATE_VERSION: Final[str] = "stage7c-authoritative-execution-v1"
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


def _verify_run_evidence(
    result: BaselineRunResult,
    build: SyntheticDatasetBuild,
    repository_sha: str,
    runtime_versions: dict[str, str],
) -> dict[str, object]:
    metrics_path = result.run_directory / f"metrics-{result.metrics_sha256}.json"
    checkpoint_path = result.run_directory / f"checkpoint-{result.checkpoint_sha256}.pt"
    complete_path = result.run_directory / "COMPLETE"
    if not metrics_path.is_file() or not checkpoint_path.is_file() or not complete_path.is_file():
        raise Stage7CExecutionError("completed run is missing required evidence artifacts")
    if sha256(metrics_path.read_bytes()).hexdigest() != result.metrics_sha256:
        raise Stage7CExecutionError("metrics artifact hash mismatch at authoritative gate")
    if sha256(checkpoint_path.read_bytes()).hexdigest() != result.checkpoint_sha256:
        raise Stage7CExecutionError("checkpoint artifact hash mismatch at authoritative gate")

    try:
        evidence = json.loads(metrics_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7CExecutionError("metrics evidence is not valid canonical JSON text") from exc
    if not isinstance(evidence, dict):
        raise Stage7CExecutionError("metrics evidence root must be an object")
    if evidence.get("repository_sha") != repository_sha:
        raise Stage7CExecutionError("metrics evidence repository SHA differs from verified checkout")
    dataset = evidence.get("dataset")
    if not isinstance(dataset, dict):
        raise Stage7CExecutionError("metrics evidence is missing dataset provenance")
    if dataset.get("build_id") != build.build_id or dataset.get("manifest_sha256") != build.manifest_sha256:
        raise Stage7CExecutionError("metrics evidence dataset provenance mismatch")
    runtime = evidence.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("dependencies") != runtime_versions:
        raise Stage7CExecutionError("metrics evidence runtime provenance mismatch")
    if evidence.get("sealed_test_split_opened") is not False:
        raise Stage7CExecutionError("metrics evidence does not prove the sealed test split stayed closed")
    return evidence


def run_verified_baseline_training(
    build: object,
    dataset_root: str | Path,
    run_root: str | Path,
    repository_root: str | Path,
    *,
    run_config: BaselineRunConfig = BaselineRunConfig(),
    model_config: BaselineModelConfig = BaselineModelConfig(),
    trainer_config: TrainerConfig = TrainerConfig(),
    preprocess_config: InputPreprocessConfig = InputPreprocessConfig(),
) -> VerifiedBaselineRunResult:
    """Run Stage 7-C only after proving clean source/runtime provenance before and after."""

    if not isinstance(build, SyntheticDatasetBuild):
        raise TypeError("build must be SyntheticDatasetBuild")
    repository_sha = verify_repository_checkout(repository_root)
    runtime_versions = verify_stage7c_runtime()

    result = run_baseline_training(
        build,
        dataset_root,
        run_root,
        repository_sha=repository_sha,
        run_config=run_config,
        model_config=model_config,
        trainer_config=trainer_config,
        preprocess_config=preprocess_config,
    )

    ending_sha = verify_repository_checkout(repository_root)
    if ending_sha != repository_sha:
        raise Stage7CExecutionError("repository HEAD changed during Stage 7-C execution")
    ending_runtime = verify_stage7c_runtime()
    if ending_runtime != runtime_versions:
        raise Stage7CExecutionError("runtime dependency identity changed during Stage 7-C execution")

    _verify_run_evidence(result, build, repository_sha, runtime_versions)
    verification_payload = {
        "schema_version": "stage7c-authoritative-verification-v1",
        "execution_gate_version": STAGE7C_EXECUTION_GATE_VERSION,
        "repository_sha": repository_sha,
        "dataset_build_id": build.build_id,
        "manifest_sha256": build.manifest_sha256,
        "metrics_sha256": result.metrics_sha256,
        "checkpoint_sha256": result.checkpoint_sha256,
        "runtime": runtime_versions,
        "run_result": {
            "run_id": result.run_id,
            "best_epoch": result.best_epoch,
            "training_steps": result.training_steps,
            "untrained_validation_loss": result.untrained_validation_loss,
            "best_validation_loss": result.best_validation_loss,
            "prediction_metrics": asdict(result.prediction_metrics),
        },
        "source_clean_before_and_after": True,
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

    return VerifiedBaselineRunResult(
        result=result,
        verification_path=verification_path,
        verification_sha256=verification_sha,
    )
