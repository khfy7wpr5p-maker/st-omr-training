"""Bounded execution wrapper for Meter V4-1 learned numerator specialist."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Final

from .meter_v4_1_numerator_specialist import (
    FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1,
    METER_V4_1_NUMERATOR_SPECIALIST,
    MeterV4_1Error,
    config_fingerprint_v4_1,
    decision_v4_1,
    load_crop_tensor_v4_1,
    summarize_predictions_v4_1,
    train_fold_v4_1,
    verify_parent_artifact_v4_1,
)
from .training_model import TORCH_PINNED_VERSION


RESULT_SCHEMA_V4_1: Final[str] = "st-omr-meter-v4-1-learned-numerator-specialist-result-v1"


def _fail(message: str) -> None:
    raise MeterV4_1Error(message)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _hex64(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        _fail(f"{name} must be canonical lowercase SHA-256")
    return value


def _git_sha40(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        _fail("git_commit_sha must be canonical lowercase SHA-1")
    return value


def repository_binding_v4_1(git_commit_sha: str) -> str:
    git_commit_sha = _git_sha40(git_commit_sha)
    return sha256(("git-commit-sha1:" + git_commit_sha).encode("ascii")).hexdigest()


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _fold_payload(result) -> dict[str, object]:
    return {
        "fold": result.fold,
        "train_record_ids": list(result.train_record_ids),
        "holdout_record_ids": list(result.holdout_record_ids),
        "final_loss": result.final_loss,
        "model_state_sha256": result.model_state_sha256,
        "optimizer_steps": result.optimizer_steps,
        "predictions": [
            {
                "record_id": row.record_id,
                "family_id": row.family_id,
                "fold": row.fold,
                "true": row.true_class,
                "pred": row.predicted_class,
                "logits": {
                    "2": row.logits[0],
                    "3": row.logits[1],
                    "4": row.logits[2],
                },
                "probabilities": {
                    "2": row.probabilities[0],
                    "3": row.probabilities[1],
                    "4": row.probabilities[2],
                },
            }
            for row in result.predictions
        ],
    }


def run_meter_v4_1_numerator_specialist(
    *,
    parent_v4_0_root: str | Path,
    output_root: str | Path,
    git_commit_sha: str,
    repository_sha: str,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run exact fixed V4-1 family-disjoint OOF training and write immutable evidence."""
    git_commit_sha = _git_sha40(git_commit_sha)
    repository_sha = _hex64("repository_sha", repository_sha)
    expected_binding = repository_binding_v4_1(git_commit_sha)
    if repository_sha != expected_binding:
        _fail("repository_sha does not bind the supplied exact Git commit")
    output = Path(output_root)
    if output.exists() or output.is_symlink():
        _fail("V4-1 output root must be fresh")
    temporary = output.with_name(f".{output.name}.part")
    if temporary.exists() or temporary.is_symlink():
        _fail("V4-1 temporary output root already exists")

    _progress(progress, "STAGE 1/5 verify exact V4-0 parent")
    parent = verify_parent_artifact_v4_1(parent_v4_0_root)
    crops = {
        record.record_id: load_crop_tensor_v4_1(parent, record)
        for record in parent.records
    }
    _progress(progress, "STAGE 2/5 loaded 27 verified numerator crops")

    fold_results = []
    determinism_rows = []
    for fold in (0, 1, 2):
        _progress(progress, f"FOLD {fold + 1}/3 primary fixed-epoch training")
        primary = train_fold_v4_1(parent.records, crops, heldout_fold=fold)
        _progress(progress, f"FOLD {fold + 1}/3 deterministic repeat")
        repeated = train_fold_v4_1(parent.records, crops, heldout_fold=fold)
        primary_payload = _fold_payload(primary)
        repeated_payload = _fold_payload(repeated)
        if _canonical_json(primary_payload) != _canonical_json(repeated_payload):
            _fail(f"V4-1 deterministic repeat mismatch in fold {fold}")
        fold_results.append(primary)
        determinism_rows.append(
            {
                "fold": fold,
                "repeat_pass": True,
                "model_state_sha256": primary.model_state_sha256,
                "final_loss": primary.final_loss,
            }
        )

    _progress(progress, "STAGE 3/5 aggregate 27 OOF predictions")
    predictions = tuple(
        prediction
        for fold_result in fold_results
        for prediction in fold_result.predictions
    )
    summary = summarize_predictions_v4_1(predictions)
    decision = decision_v4_1(summary)

    _progress(progress, "STAGE 4/5 build canonical evidence")
    result = {
        "schema": RESULT_SCHEMA_V4_1,
        "experiment": METER_V4_1_NUMERATOR_SPECIALIST,
        "git_commit_sha": git_commit_sha,
        "repository_sha": repository_sha,
        "repository_binding_formula": "sha256('git-commit-sha1:' + git_commit_sha)",
        "parent_v4_0": {
            "result_sha256": parent.result_sha256,
            "repository_binding": parent.repository_binding,
            "decision": parent.result["decision"],
            "oof_summary": parent.result["oof_summary"],
        },
        "configuration": asdict(FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1),
        "configuration_fingerprint": config_fingerprint_v4_1(),
        "torch_version": TORCH_PINNED_VERSION,
        "data_surface": {
            "records": 27,
            "families": 27,
            "classes": {"2": 9, "3": 9, "4": 9},
            "inherited_folds": 3,
            "teacher_adaptation_validation_evaluated": False,
            "teacher_adaptation_validation_images_decoded": 0,
            "d10_opened": False,
            "test_opened": False,
        },
        "records": [
            {
                "record_id": record.record_id,
                "family_id": record.family_id,
                "numerator_class": record.numerator_class,
                "fold": record.fold,
                "crop_png_sha256": record.crop_png_sha256,
            }
            for record in parent.records
        ],
        "fold_results": [_fold_payload(row) for row in fold_results],
        "determinism": {
            "repeat_count_per_fold": 2,
            "repeat_pass": True,
            "folds": determinism_rows,
        },
        "oof_summary": {
            "record_count": summary.record_count,
            "accuracy": summary.accuracy,
            "macro_f1": summary.macro_f1,
            "per_class_recall": dict(summary.per_class_recall),
            "confusion": [list(row) for row in summary.confusion],
        },
        "decision": {
            "name": decision.decision,
            "strong_signal": decision.strong_signal,
            "reasons": list(decision.reasons),
        },
        "total_optimizer_steps": sum(row.optimizer_steps for row in fold_results) * 2,
        "checkpoint_promotion_authorized": False,
        "runtime_connected": False,
        "resolver_connected": False,
        "production_promotion_authorized": False,
    }

    temporary.mkdir(parents=True)
    raw = _canonical_json(result)
    (temporary / "result.json").write_bytes(raw)
    result_sha = sha256(raw).hexdigest()
    (temporary / "COMPLETE").write_bytes(f"{result_sha}  result.json\n".encode("ascii"))
    temporary.replace(output)
    _progress(progress, "STAGE 5/5 result + COMPLETE written")
    return result
