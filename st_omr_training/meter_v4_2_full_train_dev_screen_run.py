"""Bounded execution wrapper for Meter V4-2 full-train candidate + dev screen."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Final

import torch

from .meter_v4_1_numerator_specialist import (
    FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1,
    config_fingerprint_v4_1,
    load_crop_tensor_v4_1,
    verify_parent_artifact_v4_1,
)
from .meter_v4_2_full_train_dev_screen import (
    FINAL_SEED_V4_2,
    METER_V4_2_FULL_TRAIN_DEV_SCREEN,
    MeterV4_2Error,
    dev_decision_v4_2,
    evaluate_validation_positives_v4_2,
    select_validation_positives_v4_2,
    train_full_candidate_v4_2,
)
from .training_model import TORCH_PINNED_VERSION


RESULT_SCHEMA_V4_2: Final[str] = "st-omr-meter-v4-2-full-train-dev-screen-result-v1"
EXPECTED_V4_1_RESULT_SHA256: Final[str] = "41c9ff92992628c86387582ff7c0395f2e74006b481660791cfaee66c7629f18"
_MAX_JSON_BYTES: Final[int] = 4 * 1024 * 1024


def _fail(message: str) -> None:
    raise MeterV4_2Error(message)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _hex64(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        _fail(f"{name} must be canonical lowercase SHA-256")
    return value


def repository_binding_v4_2(git_commit_sha: str) -> str:
    if not isinstance(git_commit_sha, str) or len(git_commit_sha) != 40 or any(ch not in "0123456789abcdef" for ch in git_commit_sha):
        _fail("git commit SHA must be canonical lowercase SHA-1")
    return sha256(("git-commit-sha1:" + git_commit_sha).encode("ascii")).hexdigest()


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _verify_v4_1_result(root: Path) -> dict[str, object]:
    if not root.is_dir() or root.is_symlink():
        _fail("V4-1 result root must be regular directory")
    result_path = root / "result.json"
    complete_path = root / "COMPLETE"
    if not result_path.is_file() or result_path.is_symlink() or result_path.stat().st_size <= 0 or result_path.stat().st_size > _MAX_JSON_BYTES:
        _fail("V4-1 result.json is missing or outside bounds")
    raw = result_path.read_bytes()
    result_sha = sha256(raw).hexdigest()
    if result_sha != EXPECTED_V4_1_RESULT_SHA256:
        _fail("V4-1 result SHA differs from accepted external evidence")
    expected_complete = f"{result_sha}  result.json\n".encode("ascii")
    if not complete_path.is_file() or complete_path.is_symlink() or complete_path.read_bytes() != expected_complete:
        _fail("V4-1 COMPLETE does not bind exact accepted result")
    try:
        result = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeterV4_2Error("V4-1 result is not canonical ASCII JSON") from exc
    if not isinstance(result, dict):
        _fail("V4-1 result root must be object")
    decision = result.get("decision")
    summary = result.get("oof_summary")
    determinism = result.get("determinism")
    surface = result.get("data_surface")
    if not isinstance(decision, Mapping) or decision.get("name") != "LEARNED_NUMERATOR_SIGNAL_STRONG" or decision.get("strong_signal") is not True or decision.get("reasons") != []:
        _fail("V4-1 accepted decision missing")
    if not isinstance(summary, Mapping) or summary.get("record_count") != 27 or summary.get("accuracy") != 1.0 or summary.get("macro_f1") != 1.0:
        _fail("V4-1 accepted OOF summary changed")
    if summary.get("per_class_recall") != {"2": 1.0, "3": 1.0, "4": 1.0} or summary.get("confusion") != [[9,0,0],[0,9,0],[0,0,9]]:
        _fail("V4-1 accepted class metrics changed")
    if not isinstance(determinism, Mapping) or determinism.get("repeat_pass") is not True or determinism.get("repeat_count_per_fold") != 2:
        _fail("V4-1 deterministic repeat evidence missing")
    required_surface = {
        "records": 27,
        "families": 27,
        "classes": {"2": 9, "3": 9, "4": 9},
        "teacher_adaptation_validation_evaluated": False,
        "teacher_adaptation_validation_images_decoded": 0,
        "d10_opened": False,
        "test_opened": False,
    }
    if not isinstance(surface, Mapping):
        _fail("V4-1 data surface missing")
    for key, expected in required_surface.items():
        if surface.get(key) != expected:
            _fail(f"V4-1 safety field {key} changed")
    if result.get("checkpoint_promotion_authorized") is not False or result.get("runtime_connected") is not False or result.get("resolver_connected") is not False or result.get("production_promotion_authorized") is not False:
        _fail("V4-1 crossed forbidden promotion/runtime boundary")
    return result


def run_meter_v4_2_full_train_dev_screen(
    *,
    parent_v4_0_root: str | Path,
    parent_v4_1_root: str | Path,
    pilot_path: str | Path,
    choices_path: str | Path,
    permission_path: str | Path,
    privacy_path: str | Path,
    output_root: str | Path,
    git_commit_sha: str,
    repository_sha: str,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    repository_sha = _hex64("repository_sha", repository_sha)
    if repository_sha != repository_binding_v4_2(git_commit_sha):
        _fail("repository SHA-256 binding does not match git commit")
    output = Path(output_root)
    temporary = output.with_name(f".{output.name}.part")
    if output.exists() or output.is_symlink() or temporary.exists() or temporary.is_symlink():
        _fail("V4-2 output and temporary roots must both be fresh")

    _progress(progress, "STAGE 1/6 verify accepted V4-0 and V4-1 parents")
    v4_1_result = _verify_v4_1_result(Path(parent_v4_1_root))
    parent = verify_parent_artifact_v4_1(parent_v4_0_root)
    crops = {row.record_id: load_crop_tensor_v4_1(parent, row) for row in parent.records}

    _progress(progress, "STAGE 2/6 full 27-family primary training")
    primary = train_full_candidate_v4_2(parent, crops)
    _progress(progress, "STAGE 3/6 deterministic full-training repeat")
    repeated = train_full_candidate_v4_2(parent, crops)
    repeat_pass = (
        primary.model_state_sha256 == repeated.model_state_sha256
        and primary.final_loss == repeated.final_loss
        and primary.optimizer_steps == repeated.optimizer_steps == 160
    )
    if not repeat_pass:
        _fail("V4-2 full candidate deterministic repeat mismatch")

    # Contamination barrier: Teacher Gold adaptation-validation is not opened until the
    # full candidate state and deterministic repeat are already fixed.
    _progress(progress, "STAGE 4/6 open exact 9-family development-validation screen")
    selected, source_provenance = select_validation_positives_v4_2(
        pilot_path=pilot_path,
        choices_path=choices_path,
        permission_path=permission_path,
        privacy_path=privacy_path,
    )
    predictions, summary = evaluate_validation_positives_v4_2(primary.model, selected)
    decision = dev_decision_v4_2(summary, deterministic_repeat_pass=repeat_pass)

    _progress(progress, "STAGE 5/6 write bounded development candidate artifact")
    temporary.mkdir(parents=True)
    checkpoint_path = temporary / "numerator-specialist-development-candidate.pt"
    torch.save(
        {
            "schema": "st-omr-meter-v4-2-development-candidate-v1",
            "experiment": METER_V4_2_FULL_TRAIN_DEV_SCREEN,
            "repository_sha": repository_sha,
            "parent_v4_0_result_sha256": parent.result_sha256,
            "parent_v4_1_result_sha256": EXPECTED_V4_1_RESULT_SHA256,
            "model_state_sha256": primary.model_state_sha256,
            "config_fingerprint_v4_1": config_fingerprint_v4_1(),
            "seed": FINAL_SEED_V4_2,
            "state_dict": {name: tensor.detach().cpu() for name, tensor in primary.model.state_dict().items()},
        },
        checkpoint_path,
    )
    checkpoint_sha = sha256(checkpoint_path.read_bytes()).hexdigest()

    result = {
        "schema": RESULT_SCHEMA_V4_2,
        "experiment": METER_V4_2_FULL_TRAIN_DEV_SCREEN,
        "git_commit_sha": git_commit_sha,
        "repository_sha": repository_sha,
        "torch_version": TORCH_PINNED_VERSION,
        "parent_v4_0": {"result_sha256": parent.result_sha256, "repository_binding": parent.repository_binding},
        "parent_v4_1": {"result_sha256": EXPECTED_V4_1_RESULT_SHA256, "decision": v4_1_result["decision"], "oof_summary": v4_1_result["oof_summary"]},
        "configuration": asdict(FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1),
        "configuration_fingerprint": config_fingerprint_v4_1(),
        "full_train": {
            "records": 27,
            "families": 27,
            "classes": {"2": 9, "3": 9, "4": 9},
            "augmented_views": 243,
            "seed": FINAL_SEED_V4_2,
            "optimizer_steps_per_repeat": primary.optimizer_steps,
            "repeat_count": 2,
            "total_optimizer_steps": primary.optimizer_steps + repeated.optimizer_steps,
            "final_loss": primary.final_loss,
            "model_state_sha256": primary.model_state_sha256,
            "deterministic_repeat_pass": repeat_pass,
        },
        "development_validation": {
            "record_count": summary.record_count,
            "families": 9,
            "classes": {"2": 3, "3": 3, "4": 3},
            "accuracy": summary.accuracy,
            "macro_f1": summary.macro_f1,
            "per_class_recall": dict(summary.per_class_recall),
            "confusion": [list(row) for row in summary.confusion],
            "predictions": [
                {
                    "record_id": row.record_id,
                    "family_id": row.family_id,
                    "true": row.true_class,
                    "pred": row.predicted_class,
                    "logits": {"2": row.logits[0], "3": row.logits[1], "4": row.logits[2]},
                    "probabilities": {"2": row.probabilities[0], "3": row.probabilities[1], "4": row.probabilities[2]},
                }
                for row in predictions
            ],
            "source_provenance": source_provenance,
            "used_for_training": False,
            "used_for_tuning": False,
            "final_independent_holdout": False,
        },
        "candidate_checkpoint": {
            "filename": checkpoint_path.name,
            "sha256": checkpoint_sha,
            "model_state_sha256": primary.model_state_sha256,
            "development_candidate_authorized": bool(decision["accepted_for_shadow_planning"]),
            "production_candidate_authorized": False,
        },
        "decision": decision,
        "safety": {
            "none_tasks_used": 0,
            "development_validation_used": True,
            "development_validation_used_for_training": False,
            "development_validation_images_decoded": 9,
            "d10_opened": False,
            "test_opened": False,
            "fresh_independent_holdout_required": True,
            "runtime_connected": False,
            "resolver_connected": False,
            "production_promotion_authorized": False,
        },
    }
    raw = _canonical_json(result)
    (temporary / "result.json").write_bytes(raw)
    result_sha = sha256(raw).hexdigest()
    (temporary / "COMPLETE").write_bytes(f"{result_sha}  result.json\n".encode("ascii"))
    temporary.replace(output)
    _progress(progress, "STAGE 6/6 result + checkpoint + COMPLETE written")
    return result
