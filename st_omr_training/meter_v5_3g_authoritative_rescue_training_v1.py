"""Meter V5-3G authoritative TRAIN materialization and single rescue execution.

This stage binds the CI-green V5-3F tensor harness to the exact V5 TRAIN and
historical TRAIN frozen-feature surfaces. It materializes only same-specialist
frozen-negative rows, runs exactly one preregistered rescue fit for 2-AI and
3-AI, verifies frozen-state isolation, and writes separate rescue artifacts.

Historical validation, immutable First-30, V5 reserve, V5 validation,
FINAL_HOLDOUT, threshold tuning, runtime wiring, and production promotion are
outside this module.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Final, Mapping

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2n_frozen_feature_transfer_audit_v1 as v52n
from . import meter_v5_3e_rescue_training_preregistration_v1 as v53e
from . import meter_v5_3f_rescue_training_execution_harness_v1 as v53f


SCHEMA: Final[str] = "st-omr-meter-v5-3g-authoritative-rescue-training-v1"
RESCUE_ARTIFACT_SCHEMA: Final[str] = "st-omr-meter-v5-3g-rescue-artifact-v1"
V53F_HEAD_SHA: Final[str] = "7ed41f2872058ac5e3e52df756b9098a1d60052d"
V53F_MODULE_BLOB_SHA: Final[str] = "908b5b7f83fc5a5358261b7dc04ab606ee66e063"
V53F_DOC_BLOB_SHA: Final[str] = "164f84b3dc89230024c8a62ef189204adfe4ebed"
APPROVAL_TOKEN: Final[str] = "V5_3G_SINGLE_AUTHORITATIVE_RESCUE_TRAIN_RUN_APPROVED"
REPORT_NAME: Final[str] = "v5_3g_authoritative_rescue_training_report.json"
ARTIFACT_DIR_NAME: Final[str] = "v5_3g_authoritative_rescue_artifacts"
TEMP_ARTIFACT_DIR_NAME: Final[str] = ".v5_3g_authoritative_rescue_artifacts.tmp"

ProgressCallback = Callable[[int, int, str], None]


class MeterV5_3GError(RuntimeError):
    """Raised when V5-3G departs from its exact authorized execution contract."""


def _fail(message: str) -> None:
    raise MeterV5_3GError(message)


def prerequisite_contract() -> dict[str, object]:
    return {
        "v5_3f_head_sha": V53F_HEAD_SHA,
        "v5_3f_module_blob_sha": V53F_MODULE_BLOB_SHA,
        "v5_3f_doc_blob_sha": V53F_DOC_BLOB_SHA,
        "v5_3f_schema": v53f.SCHEMA,
        "recipe_id": v53e.RECIPE_ID,
        "candidate_configuration_count": v53e.CANDIDATE_CONFIGURATION_COUNT,
        "fixed_optimizer_steps": v53e.FIXED_OPTIMIZER_STEPS,
        "rescue_threshold": v53e.RESCUE_THRESHOLD,
    }


def safety_boundary() -> dict[str, object]:
    return {
        "single_authoritative_train_entry": True,
        "data_surfaces": ("v5_train", "historical_train"),
        "frozen_feature_extractor_reused": "v5-2n",
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "frozen_checkpoint_read": True,
        "frozen_checkpoint_write": False,
        "frozen_model_mutation_allowed": False,
        "trainable_surface": "new-rescue-parameters-only",
        "digit4_loaded": False,
        "digit4_frozen": True,
        "rescue_artifact_write": True,
        "original_specialist_checkpoint_replacement": False,
        "threshold_tuning": False,
        "hyperparameter_sweep": False,
        "automatic_second_configuration": False,
        "historical_validation_opened": False,
        "first30_opened": False,
        "v5_reserve_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "resolver_wiring": False,
        "production_promotion": False,
        "colab_execution_wrapper_present": False,
    }


def execution_contract() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "prerequisite": prerequisite_contract(),
        "v5_3f_execution": v53f.execution_contract(),
        "authoritative_group_counts": {
            digit: dict(counts)
            for digit, counts in v53e.EXPECTED_TRAIN_GROUP_COUNTS.items()
        },
        "approval_token_required": True,
        "one_shot_non_overwriting": True,
        "exact_sha_colab_wrapper_required_for_external_execution": True,
        **safety_boundary(),
    }


def _tensor_fingerprint(tensor, *, name: str) -> str:
    torch, _nn = v52b._import_torch()
    if not isinstance(tensor, torch.Tensor):
        _fail(f"{name} must be a torch.Tensor")
    cpu = tensor.detach().cpu().contiguous()
    if cpu.ndim != 2 or cpu.shape[1] != v53e.FEATURE_DIM or cpu.shape[0] <= 0:
        _fail(f"{name} feature shape changed: {tuple(cpu.shape)}")
    if not bool(torch.isfinite(cpu).all().item()):
        _fail(f"{name} contains non-finite values")
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(str(tuple(cpu.shape)).encode("ascii"))
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _frozen_probabilities_from_features(model, features, *, digit: str):
    torch, _nn = v52b._import_torch()
    if digit not in v53e.RESCUE_SPECIALISTS:
        _fail("V5-3G frozen probabilities are defined only for 2-AI and 3-AI")
    x = features.detach().cpu().to(dtype=torch.float32)
    if x.ndim != 2 or x.shape[1] != v53e.FEATURE_DIM:
        _fail(f"{digit}-AI frozen feature shape changed")
    model.eval()
    with torch.no_grad():
        logits = model.head(x).squeeze(1)
        probabilities = torch.sigmoid(logits).cpu()
    if probabilities.ndim != 1 or probabilities.numel() != x.shape[0]:
        _fail(f"{digit}-AI frozen probability shape changed")
    if not bool(torch.isfinite(probabilities).all().item()):
        _fail(f"{digit}-AI frozen probabilities became non-finite")
    return probabilities


def _validate_binary_targets(targets, *, name: str, expected_count: int):
    torch, _nn = v52b._import_torch()
    if not isinstance(targets, torch.Tensor):
        _fail(f"{name} targets must be a torch.Tensor")
    y = targets.detach().cpu().to(dtype=torch.float32).reshape(-1)
    if y.numel() != expected_count:
        _fail(f"{name} target count changed: {y.numel()} != {expected_count}")
    if not bool(torch.isfinite(y).all().item()):
        _fail(f"{name} targets contain non-finite values")
    if not bool(((y == 0.0) | (y == 1.0)).all().item()):
        _fail(f"{name} targets are not binary")
    return y


def _materialize_frozen_negative_groups_v1(
    *,
    digit: str,
    model,
    v5_features,
    v5_targets,
    historical_features,
    historical_targets,
    enforce_preregistered_counts: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    """Select only frozen-negative TRAIN rows into the four preregistered groups."""
    if digit not in v53e.RESCUE_SPECIALISTS:
        _fail("V5-3G supports only 2-AI and 3-AI")
    if type(enforce_preregistered_counts) is not bool:
        _fail("enforce_preregistered_counts must be bool")
    torch, _nn = v52b._import_torch()

    x_v5 = v5_features.detach().cpu().to(dtype=torch.float32)
    x_hist = historical_features.detach().cpu().to(dtype=torch.float32)
    if x_v5.ndim != 2 or x_v5.shape != (v52n.EXPECTED_V5_COUNT, v53e.FEATURE_DIM):
        if enforce_preregistered_counts:
            _fail(f"{digit}-AI V5 TRAIN feature shape changed: {tuple(x_v5.shape)}")
        if x_v5.ndim != 2 or x_v5.shape[1] != v53e.FEATURE_DIM or x_v5.shape[0] <= 0:
            _fail(f"{digit}-AI V5 TRAIN feature shape invalid: {tuple(x_v5.shape)}")
    if x_hist.ndim != 2 or x_hist.shape != (v52n.EXPECTED_HISTORICAL_COUNT, v53e.FEATURE_DIM):
        if enforce_preregistered_counts:
            _fail(f"{digit}-AI historical TRAIN feature shape changed: {tuple(x_hist.shape)}")
        if x_hist.ndim != 2 or x_hist.shape[1] != v53e.FEATURE_DIM or x_hist.shape[0] <= 0:
            _fail(f"{digit}-AI historical TRAIN feature shape invalid: {tuple(x_hist.shape)}")
    if not bool(torch.isfinite(x_v5).all().item()) or not bool(torch.isfinite(x_hist).all().item()):
        _fail(f"{digit}-AI TRAIN features contain non-finite values")

    y_v5 = _validate_binary_targets(
        v5_targets,
        name=f"{digit}-AI V5 TRAIN",
        expected_count=int(x_v5.shape[0]),
    )
    y_hist = _validate_binary_targets(
        historical_targets,
        name=f"{digit}-AI historical TRAIN",
        expected_count=int(x_hist.shape[0]),
    )
    if enforce_preregistered_counts:
        if int(y_v5.sum().item()) != v52n.EXPECTED_V5_POSITIVE:
            _fail(f"{digit}-AI V5 positive count changed")
        expected_hist_positive = v52n.EXPECTED_HISTORICAL_LABEL_COUNTS[digit]
        if int(y_hist.sum().item()) != expected_hist_positive:
            _fail(f"{digit}-AI historical positive count changed")

    v5_probability = _frozen_probabilities_from_features(model, x_v5, digit=digit)
    historical_probability = _frozen_probabilities_from_features(model, x_hist, digit=digit)
    threshold = v52b.FROZEN_THRESHOLDS[digit]
    v5_negative = v5_probability < threshold
    historical_negative = historical_probability < threshold

    groups: dict[str, object] = {
        "v5_frozen_false_negative_positive": x_v5[v5_negative & (y_v5 == 1.0)].clone(),
        "v5_frozen_true_negative": x_v5[v5_negative & (y_v5 == 0.0)].clone(),
        "historical_frozen_false_negative_positive": x_hist[
            historical_negative & (y_hist == 1.0)
        ].clone(),
        "historical_frozen_true_negative": x_hist[
            historical_negative & (y_hist == 0.0)
        ].clone(),
    }
    if tuple(groups) != v53e.TRAIN_GROUPS:
        _fail("V5-3G materialized group order changed")

    counts = {name: int(groups[name].shape[0]) for name in v53e.TRAIN_GROUPS}
    if enforce_preregistered_counts and counts != v53e.EXPECTED_TRAIN_GROUP_COUNTS[digit]:
        _fail(f"{digit}-AI frozen-negative group counts changed: {counts}")

    fingerprints = {
        name: _tensor_fingerprint(groups[name], name=f"{digit}:{name}")
        for name in v53e.TRAIN_GROUPS
    }
    evidence = {
        "digit": digit,
        "frozen_threshold": threshold,
        "frozen_threshold_unchanged": True,
        "group_counts": counts,
        "group_fingerprints": fingerprints,
        "v5_frozen_negative_count": int(v5_negative.sum().item()),
        "historical_frozen_negative_count": int(historical_negative.sum().item()),
        "eligible_rows_only": "same-specialist-frozen-negative",
        "labels_derived_from_frozen_group_identity": True,
    }
    return groups, evidence


def _artifact_path(directory: Path, digit: str) -> Path:
    if digit not in v53e.RESCUE_SPECIALISTS:
        _fail("unknown V5-3G rescue specialist")
    return directory / f"digit_{digit}_rescue.pt"


def _save_rescue_artifact(
    *,
    model,
    path: Path,
    digit: str,
    source_checkpoint_sha256: str,
    slot_manifest_sha256: str,
    materialization: Mapping[str, object],
    execution: Mapping[str, object],
) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    if path.exists():
        _fail(f"refusing to overwrite rescue artifact: {path}")
    state_fingerprint = v53f._state_fingerprint(model)
    metadata = {
        "schema": RESCUE_ARTIFACT_SCHEMA,
        "role": f"digit-{digit}-v5-3g-rescue",
        "v5_3f_head_sha": V53F_HEAD_SHA,
        "recipe_id": v53e.RECIPE_ID,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "slot_manifest_sha256": slot_manifest_sha256,
        "rescue_threshold": v53e.RESCUE_THRESHOLD,
        "frozen_threshold": v52b.FROZEN_THRESHOLDS[digit],
        "parameter_count": v53e.PARAMETERS_PER_RESCUE,
        "trainable_surface": "new-rescue-parameters-only",
        "digit4_frozen": True,
        "group_counts": dict(materialization["group_counts"]),
        "group_fingerprints": dict(materialization["group_fingerprints"]),
        "optimizer_steps": execution["optimizer_steps"],
        "initial_state_fingerprint": execution["initial_state_fingerprint"],
        "final_state_fingerprint": execution["final_state_fingerprint"],
        "state_fingerprint": state_fingerprint,
        "historical_validation_opened": False,
        "first30_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "runtime_authority_changed": False,
        "production_promotion": False,
    }
    payload = {
        "metadata": metadata,
        "rescue_state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    artifact_sha = v52b._sha_file(path)

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise MeterV5_3GError(f"cannot reload V5-3G {digit}-AI rescue artifact") from exc
    if not isinstance(loaded, Mapping):
        _fail(f"{digit}-AI rescue artifact payload must be a mapping")
    loaded_metadata = loaded.get("metadata")
    loaded_state = loaded.get("rescue_state_dict")
    if loaded_metadata != metadata or not isinstance(loaded_state, Mapping):
        _fail(f"{digit}-AI rescue artifact metadata/state changed")
    reloaded = v53f._build_rescue_model_v1()
    reloaded.load_state_dict(dict(loaded_state), strict=True)
    if v53f._state_fingerprint(reloaded) != state_fingerprint:
        _fail(f"{digit}-AI rescue artifact state fingerprint mismatch")
    return {
        "artifact_path": str(path),
        "artifact_sha256": artifact_sha,
        "state_fingerprint": state_fingerprint,
        "reload_verified": True,
    }


def _preflight_outputs(root: Path) -> tuple[Path, Path, Path]:
    ann = root / v51.ANNOTATIONS_DIR
    report = ann / REPORT_NAME
    artifact_dir = ann / ARTIFACT_DIR_NAME
    temporary_dir = ann / TEMP_ARTIFACT_DIR_NAME
    if report.exists() or artifact_dir.exists() or temporary_dir.exists():
        _fail("refusing to overwrite or rerun existing V5-3G execution evidence")
    return report, artifact_dir, temporary_dir


def run_authoritative_rescue_training_v1(
    data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    confirmation: str,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Execute the one authorized TRAIN-only rescue fit and stop before TRAIN gating."""
    if confirmation != APPROVAL_TOKEN:
        _fail("exact V5-3G approval token is required before data access")

    root = Path(data_root)
    report_path, artifact_dir, temporary_dir = _preflight_outputs(root)

    models = v52n._frozen_models(
        digit2_frozen=Path(digit2_frozen),
        digit3_frozen=Path(digit3_frozen),
    )
    frozen_before = {
        digit: v53f._state_fingerprint(models[digit]) for digit in v53e.RESCUE_SPECIALISTS
    }
    source_checkpoint_sha = {
        "2": v52b.DIGIT2_SHA256,
        "3": v52b.DIGIT3_SHA256,
    }

    manifest_path, _rows, v5_features, v5_targets, _metrics = v52n._v5_surface(
        root, models
    )
    historical_features, historical_targets = v52n._historical_surface(
        m4a_root=Path(m4a_root),
        d10_root=Path(d10_root),
        models=models,
        progress=progress,
    )
    slot_manifest_sha = v52b._sha_file(manifest_path)

    per_specialist: dict[str, dict[str, object]] = {}
    trained_models: dict[str, object] = {}
    for digit in v53e.RESCUE_SPECIALISTS:
        groups, materialization = _materialize_frozen_negative_groups_v1(
            digit=digit,
            model=models[digit],
            v5_features=v5_features[digit],
            v5_targets=v5_targets[digit],
            historical_features=historical_features[digit],
            historical_targets=historical_targets[digit],
            enforce_preregistered_counts=True,
        )
        rescue, execution = v53f.execute_rescue_tensor_harness_v1(
            digit=digit,
            features_by_group=groups,
            approval_token=v53f.APPROVAL_TOKEN,
            enforce_preregistered_counts=True,
        )
        if execution.get("authoritative_dataset_execution") is not False:
            _fail("V5-3F evidence unexpectedly claims authoritative execution")
        frozen_after_digit = v53f._state_fingerprint(models[digit])
        if frozen_after_digit != frozen_before[digit]:
            _fail(f"{digit}-AI frozen specialist changed during rescue execution")
        trained_models[digit] = rescue
        per_specialist[digit] = {
            "materialization": materialization,
            "execution": {
                **dict(execution),
                "authoritative_dataset_execution": True,
                "authoritative_wrapper_schema": SCHEMA,
            },
            "frozen_state_before": frozen_before[digit],
            "frozen_state_after": frozen_after_digit,
            "frozen_state_bit_identical": True,
        }

    frozen_after = {
        digit: v53f._state_fingerprint(models[digit]) for digit in v53e.RESCUE_SPECIALISTS
    }
    if frozen_after != frozen_before:
        _fail("frozen specialist state changed across V5-3G execution")

    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        for digit in v53e.RESCUE_SPECIALISTS:
            saved = _save_rescue_artifact(
                model=trained_models[digit],
                path=_artifact_path(temporary_dir, digit),
                digit=digit,
                source_checkpoint_sha256=source_checkpoint_sha[digit],
                slot_manifest_sha256=slot_manifest_sha,
                materialization=per_specialist[digit]["materialization"],
                execution=per_specialist[digit]["execution"],
            )
            per_specialist[digit]["artifact"] = saved
        temporary_dir.replace(artifact_dir)
    except Exception:
        # Preserve partial temporary evidence for forensic inspection; rerun remains blocked.
        raise

    for digit in v53e.RESCUE_SPECIALISTS:
        final_path = _artifact_path(artifact_dir, digit)
        per_specialist[digit]["artifact"]["artifact_path"] = str(final_path)
        per_specialist[digit]["artifact"]["artifact_sha256"] = v52b._sha_file(final_path)

    report: dict[str, object] = {
        "schema": SCHEMA,
        "approval_token_verified": True,
        "v5_3f_head_sha": V53F_HEAD_SHA,
        "slot_manifest_sha256": slot_manifest_sha,
        "source_checkpoint_sha256": source_checkpoint_sha,
        "implementation_contract": execution_contract(),
        "authoritative_train_tensor_materialization": True,
        "single_authoritative_execution_completed": True,
        "candidate_configuration_count": 1,
        "per_specialist": per_specialist,
        "numerical_integrity_gate": {"gate": "PASS", "reasons": []},
        "frozen_state_isolation_gate": {"gate": "PASS", "reasons": []},
        "train_performance_gate_executed": False,
        "historical_validation_retention_executed": False,
        "first30_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "runtime_authority_changed": False,
        "production_promotion": False,
    }
    v51._atomic_write_json(report_path, report)
    return report


def train_performance_gate_executed_by_this_module() -> bool:
    return False


def protected_evaluation_access_allowed() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False


def future_gate_order() -> tuple[str, ...]:
    return (
        "v5_3g_exact_ci_green_sha",
        "exact_sha_external_execution_wrapper",
        "single_authoritative_execution_receipt",
        "train_v5_f1_and_frozen_correct_retention",
        "historical_validation_retention_at_frozen_thresholds",
        "immutable_v5_first30_diagnostic",
        "separately_authorized_v5_validation",
        "separately_authorized_final_holdout",
    )
