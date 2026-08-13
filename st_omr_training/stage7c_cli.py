"""Command-line entrypoint for the one frozen authoritative Stage 7-C baseline run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .stage7c_dataset import (
    STAGE7C_BASELINE_DATASET_CONFIG_FINGERPRINT,
    build_and_persist_stage7c_baseline_dataset,
)
from .stage7c_execution import (
    run_verified_baseline_training,
    verify_authoritative_repository,
    verify_stage7c_runtime,
)
from .training_run import ProgressCallback


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_workspace_path(workspace: str | Path, repository_root: str | Path) -> Path:
    if not isinstance(workspace, (str, Path)) or not isinstance(repository_root, (str, Path)):
        raise TypeError("workspace and repository_root must be str or pathlib.Path")
    root = Path(repository_root).resolve()
    target = Path(workspace).expanduser().resolve()
    if target == root or root in target.parents:
        raise ValueError("Stage 7-C workspace must be outside the Git repository")
    if target.exists():
        raise FileExistsError("Stage 7-C workspace must not already exist")
    return target


def run_frozen_stage7c_baseline(
    workspace: str | Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Build the frozen dataset and execute the authoritative Stage 7-C baseline profile."""

    repository_root = _repository_root()
    target = validate_workspace_path(workspace, repository_root)
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    if progress is not None:
        progress({"event": "stage7c_started"})
    initial_repository_sha, initial_repository_origin = verify_authoritative_repository(
        repository_root
    )
    if progress is not None:
        progress(
            {
                "event": "repository_verified",
                "repository_sha": initial_repository_sha,
            }
        )
    runtime = verify_stage7c_runtime()
    if progress is not None:
        progress({"event": "runtime_verified", "runtime": runtime})

    target.mkdir(parents=True, exist_ok=False)
    dataset_root = target / "dataset"
    run_root = target / "training-runs"
    if progress is not None:
        progress({"event": "dataset_build_started"})
    build = build_and_persist_stage7c_baseline_dataset(
        dataset_root,
        progress=progress,
    )
    if build.config_fingerprint != STAGE7C_BASELINE_DATASET_CONFIG_FINGERPRINT:
        raise RuntimeError("frozen Stage 7-C dataset identity mismatch")
    if progress is not None:
        progress(
            {
                "event": "dataset_build_completed",
                "dataset_build_id": build.build_id,
                "manifest_sha256": build.manifest_sha256,
            }
        )

    verified = run_verified_baseline_training(
        build,
        dataset_root,
        run_root,
        repository_root,
        progress=progress,
    )
    if verified.result.repository_sha != initial_repository_sha:
        raise RuntimeError("repository SHA changed between Stage 7-C preparation and training")

    summary = {
        "schema_version": "stage7c-cli-summary-v1",
        "repository_origin": initial_repository_origin,
        "repository_sha": verified.result.repository_sha,
        "dataset_build_id": verified.result.dataset_build_id,
        "manifest_sha256": verified.result.manifest_sha256,
        "dataset_config_fingerprint": build.config_fingerprint,
        "runtime": runtime,
        "run_id": verified.result.run_id,
        "untrained_validation_loss": verified.result.untrained_validation_loss,
        "best_validation_loss": verified.result.best_validation_loss,
        "best_epoch": verified.result.best_epoch,
        "training_steps": verified.result.training_steps,
        "prediction_metrics": {
            "token_error_rate": verified.result.prediction_metrics.token_error_rate,
            "exact_sequence_accuracy": verified.result.prediction_metrics.exact_sequence_accuracy,
            "detokenization_success_rate": verified.result.prediction_metrics.detokenization_success_rate,
            "semantic_validity_rate": verified.result.prediction_metrics.semantic_validity_rate,
            "musicxml_regeneration_validity_rate": (
                verified.result.prediction_metrics.musicxml_regeneration_validity_rate
            ),
            "validation_samples": verified.result.prediction_metrics.validation_samples,
            "valid_semantic_predictions": (
                verified.result.prediction_metrics.valid_semantic_predictions
            ),
        },
        "checkpoint_sha256": verified.result.checkpoint_sha256,
        "metrics_sha256": verified.result.metrics_sha256,
        "verification_sha256": verified.verification_sha256,
        "verification_file": verified.verification_path.name,
    }
    if progress is not None:
        progress({"event": "stage7c_completed", "run_id": verified.result.run_id})
    return summary


def _write_progress(payload: dict[str, object]) -> None:
    print(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        file=sys.stderr,
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen authoritative Stage 7-C ST-OMR synthetic baseline.",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="fresh output directory outside the Git repository",
    )
    arguments = parser.parse_args(argv)
    summary = run_frozen_stage7c_baseline(
        arguments.workspace,
        progress=_write_progress,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
