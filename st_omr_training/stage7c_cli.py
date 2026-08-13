"""Command-line entrypoint for the one frozen authoritative Stage 7-C baseline run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .stage7c_dataset import (
    STAGE7C_BASELINE_DATASET_CONFIG_FINGERPRINT,
    build_and_persist_stage7c_baseline_dataset,
)
from .stage7c_execution import (
    run_verified_baseline_training,
    verify_repository_checkout,
    verify_stage7c_runtime,
)


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


def run_frozen_stage7c_baseline(workspace: str | Path) -> dict[str, object]:
    """Build the frozen dataset and execute the authoritative Stage 7-C baseline profile."""

    repository_root = _repository_root()
    target = validate_workspace_path(workspace, repository_root)
    initial_repository_sha = verify_repository_checkout(repository_root)
    runtime = verify_stage7c_runtime()

    target.mkdir(parents=True, exist_ok=False)
    dataset_root = target / "dataset"
    run_root = target / "training-runs"
    build = build_and_persist_stage7c_baseline_dataset(dataset_root)
    if build.config_fingerprint != STAGE7C_BASELINE_DATASET_CONFIG_FINGERPRINT:
        raise RuntimeError("frozen Stage 7-C dataset identity mismatch")

    verified = run_verified_baseline_training(
        build,
        dataset_root,
        run_root,
        repository_root,
    )
    if verified.result.repository_sha != initial_repository_sha:
        raise RuntimeError("repository SHA changed between Stage 7-C preparation and training")

    return {
        "schema_version": "stage7c-cli-summary-v1",
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
    summary = run_frozen_stage7c_baseline(arguments.workspace)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
