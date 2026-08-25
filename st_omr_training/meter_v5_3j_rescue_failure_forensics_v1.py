"""Meter V5-3J read-only rescue failure forensics.

V5-3J is opened only because V5-3I returned HOLD. It reuses the exact frozen
2-AI/3-AI TRAIN surfaces and exact V5-3G rescue artifacts to describe why the
acceptance gate failed. It performs no fitting, threshold search, checkpoint
write, protected validation access, or production wiring.

Historical Validation, immutable First-30, V5 reserve/validation and
FINAL_HOLDOUT remain closed regardless of the diagnostic result.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Final, Mapping

from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2n_frozen_feature_transfer_audit_v1 as v52n
from . import meter_v5_3e_rescue_training_preregistration_v1 as v53e
from . import meter_v5_3f_rescue_training_execution_harness_v1 as v53f
from . import meter_v5_3g_authoritative_rescue_training_v1 as v53g
from . import meter_v5_3i_train_acceptance_gate_v1 as v53i

SCHEMA: Final[str] = "st-omr-meter-v5-3j-rescue-failure-forensics-v1"
REPORT_NAME: Final[str] = "v5_3j_rescue_failure_forensics_v1.json"
V53I_HEAD_SHA: Final[str] = "88c7acc551fa2b00b1f877f6a839704d58825adb"
V53I_MODULE_BLOB_SHA: Final[str] = "abb5f1ae4c42b0c5f3ae26b80f2a467f47582197"
EXPECTED_V53I_REPORT_SHA256: Final[str] = (
    "448b807086bc9ee66d090fdf173ce54e3c5e2a133e60cf6ae0a791aed2717434"
)
EXPECTED_HOLD_REASONS: Final[tuple[str, ...]] = (
    "2-AI historical TRAIN frozen-correct regression count != 0",
    "3-AI V5 TRAIN F1 != 1.0",
    "3-AI V5 TRAIN false negatives != 0",
    "3-AI historical TRAIN frozen-correct regression count != 0",
)
EXPECTED_ACCEPTANCE_WITNESS: Final[dict[str, dict[str, object]]] = {
    "2": {"v5_f1": 1.0, "v5_fp": 0, "v5_fn": 0, "v5_regressions": 0, "historical_regressions": 5307},
    "3": {"v5_f1": 0.0, "v5_fp": 0, "v5_fn": 90, "v5_regressions": 0, "historical_regressions": 15775},
}

ProgressCallback = Callable[[int, int, str], None]


class MeterV5_3JError(RuntimeError):
    """Raised when V5-3J departs from the read-only forensic contract."""


def _fail(message: str) -> None:
    raise MeterV5_3JError(message)


def safety_boundary() -> dict[str, object]:
    return {
        "training": False,
        "autograd_grad_used": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_write": False,
        "rescue_artifact_write": False,
        "frozen_model_mutation_allowed": False,
        "threshold_tuning": False,
        "threshold_sweep": False,
        "hyperparameter_sweep": False,
        "automatic_second_configuration": False,
        "architecture_change_authorized": False,
        "retraining_authorized": False,
        "v5_adaptation_train_read": True,
        "historical_train_read": True,
        "historical_validation_opened": False,
        "first30_opened": False,
        "v5_reserve_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_loaded": False,
        "digit4_frozen": True,
        "resolver_wiring": False,
        "runtime_authority_changed": False,
        "production_promotion": False,
        "forensics_report_write_only": True,
    }


def forensic_contract() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "prerequisite_v5_3i_head": V53I_HEAD_SHA,
        "prerequisite_v5_3i_module_blob": V53I_MODULE_BLOB_SHA,
        "bound_v5_3i_report_sha256": EXPECTED_V53I_REPORT_SHA256,
        "required_v5_3i_decision": "HOLD",
        "bound_hold_reasons": list(EXPECTED_HOLD_REASONS),
        "rescue_threshold": v53e.RESCUE_THRESHOLD,
        "frozen_thresholds": {"2": v52b.FROZEN_THRESHOLDS["2"], "3": v52b.FROZEN_THRESHOLDS["3"]},
        "descriptive_outputs": (
            "fixed-threshold crossing counts",
            "fixed quantiles",
            "within-domain positive-over-negative rank fraction",
            "cross-domain V5-positive-over-historical-negative rank fraction",
            "correction/regression counts",
        ),
        "no_threshold_selection_from_diagnostics": True,
        **safety_boundary(),
    }


def _sha_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        _fail(f"expected regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_bound_json(path: Path, *, expected_sha256: str, label: str) -> dict[str, object]:
    actual = _sha_file(path)
    if actual != expected_sha256:
        _fail(f"{label} SHA changed: {actual}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeterV5_3JError(f"invalid {label} JSON") from exc
    if not isinstance(payload, dict):
        _fail(f"{label} JSON must be an object")
    return payload


def _validate_v53i_hold_receipt(report: Mapping[str, object]) -> None:
    expected_scalars = {
        "schema": v53i.SCHEMA,
        "decision": "HOLD",
        "frozen_state_bit_identical": True,
        "only_rescue_parameters_changed": True,
        "historical_validation_retention_executed": False,
        "first30_opened": False,
        "v5_reserve_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "retraining_authorized": False,
    }
    for key, expected in expected_scalars.items():
        if report.get(key) != expected:
            _fail(f"V5-3I HOLD receipt field changed: {key}")
    if report.get("acceptance_reasons") != list(EXPECTED_HOLD_REASONS):
        _fail("V5-3I HOLD reasons changed")

    bound = report.get("bound_evidence")
    if not isinstance(bound, Mapping):
        _fail("V5-3I bound evidence missing")
    if bound.get("v5_3g_head_sha") != v53i.V53G_HEAD_SHA:
        _fail("V5-3I V5-3G binding changed")
    if bound.get("v5_3h_wrapper_head_sha") != v53i.V53H_WRAPPER_HEAD_SHA:
        _fail("V5-3I V5-3H binding changed")
    if bound.get("v5_3g_report_sha256") != v53i.EXPECTED_V53G_REPORT_SHA256:
        _fail("V5-3I V5-3G report binding changed")
    if bound.get("v5_3h_envelope_sha256") != v53i.EXPECTED_V53H_ENVELOPE_SHA256:
        _fail("V5-3I envelope binding changed")
    if bound.get("rescue_artifact_sha256") != v53i.EXPECTED_RESCUE_ARTIFACT_SHA256:
        _fail("V5-3I rescue artifact binding changed")

    per_specialist = report.get("per_specialist")
    if not isinstance(per_specialist, Mapping) or set(per_specialist) != {"2", "3"}:
        _fail("V5-3I per-specialist HOLD evidence changed")
    for digit in ("2", "3"):
        item = per_specialist.get(digit)
        if not isinstance(item, Mapping):
            _fail(f"{digit}-AI V5-3I evidence missing")
        v5 = item.get("v5_train")
        hist = item.get("historical_train")
        if not isinstance(v5, Mapping) or not isinstance(hist, Mapping):
            _fail(f"{digit}-AI V5-3I domain evidence missing")
        metrics = v5.get("combined_metrics")
        if not isinstance(metrics, Mapping):
            _fail(f"{digit}-AI V5-3I V5 metrics missing")
        witness = EXPECTED_ACCEPTANCE_WITNESS[digit]
        observed = {
            "v5_f1": metrics.get("f1"),
            "v5_fp": metrics.get("fp"),
            "v5_fn": metrics.get("fn"),
            "v5_regressions": v5.get("frozen_correct_regression_count"),
            "historical_regressions": hist.get("frozen_correct_regression_count"),
        }
        if observed != witness:
            _fail(f"{digit}-AI V5-3I HOLD witness changed: {observed}")


def _finite_vector(values, *, label: str):
    torch, _nn = v52b._import_torch()
    if not isinstance(values, torch.Tensor):
        _fail(f"{label} must be a torch.Tensor")
    x = values.detach().cpu().to(dtype=torch.float64).reshape(-1)
    if x.numel() <= 0:
        _fail(f"{label} must not be empty")
    if not bool(torch.isfinite(x).all().item()):
        _fail(f"{label} contains non-finite values")
    return x


def _probability_distribution(values, *, label: str) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    x = _finite_vector(values, label=label)
    if bool(((x < 0.0) | (x > 1.0)).any().item()):
        _fail(f"{label} is outside [0,1]")
    q = torch.tensor([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99], dtype=torch.float64)
    values_q = torch.quantile(x, q)
    return {
        "count": int(x.numel()),
        "min": float(x.min().item()), "p01": float(values_q[0].item()), "p05": float(values_q[1].item()),
        "p10": float(values_q[2].item()), "p25": float(values_q[3].item()), "median": float(values_q[4].item()),
        "p75": float(values_q[5].item()), "p90": float(values_q[6].item()), "p95": float(values_q[7].item()),
        "p99": float(values_q[8].item()), "max": float(x.max().item()), "mean": float(x.mean().item()),
        "std_population": float(x.std(unbiased=False).item()),
    }


def _pairwise_rank_fraction(positive_scores, negative_scores, *, label: str) -> float:
    """Fraction of positive/negative pairs ordered positive>negative; ties count 0.5."""
    torch, _nn = v52b._import_torch()
    pos = _finite_vector(positive_scores, label=f"{label}:positive")
    neg = _finite_vector(negative_scores, label=f"{label}:negative")
    neg_sorted, _indices = torch.sort(neg)
    lower = torch.searchsorted(neg_sorted, pos, right=False).to(dtype=torch.float64)
    upper = torch.searchsorted(neg_sorted, pos, right=True).to(dtype=torch.float64)
    wins = lower + 0.5 * (upper - lower)
    fraction = float((wins / float(neg.numel())).mean().item())
    if not math.isfinite(fraction) or not (0.0 <= fraction <= 1.0):
        _fail(f"{label} rank fraction invalid")
    return fraction


def _score_group_diagnostics(*, frozen_probability, rescue_probability, targets, frozen_threshold: float, rescue_threshold: float, label: str) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    frozen = _finite_vector(frozen_probability, label=f"{label}:frozen")
    rescue = _finite_vector(rescue_probability, label=f"{label}:rescue")
    y = _finite_vector(targets, label=f"{label}:targets")
    if frozen.numel() != rescue.numel() or frozen.numel() != y.numel():
        _fail(f"{label} cardinality mismatch")
    if bool(((frozen < 0.0) | (frozen > 1.0)).any().item()) or bool(((rescue < 0.0) | (rescue > 1.0)).any().item()):
        _fail(f"{label} probability outside [0,1]")
    if not bool(((y == 0.0) | (y == 1.0)).all().item()):
        _fail(f"{label} targets are not binary")

    truth = y == 1.0
    eligible = frozen < float(frozen_threshold)
    eligible_positive = eligible & truth
    eligible_negative = eligible & ~truth
    if int(eligible_positive.sum().item()) <= 0 or int(eligible_negative.sum().item()) <= 0:
        _fail(f"{label} requires non-empty frozen-negative positive and negative groups")

    rescue_positive = rescue >= float(rescue_threshold)
    corrected_positive = eligible_positive & rescue_positive
    remaining_false_negative = eligible_positive & ~rescue_positive
    regressed_true_negative = eligible_negative & rescue_positive
    preserved_true_negative = eligible_negative & ~rescue_positive
    pos_scores = rescue[eligible_positive]
    neg_scores = rescue[eligible_negative]
    pos_dist = _probability_distribution(pos_scores, label=f"{label}:eligible-positive")
    neg_dist = _probability_distribution(neg_scores, label=f"{label}:eligible-negative")
    return {
        "count": int(frozen.numel()),
        "frozen_threshold": float(frozen_threshold), "rescue_threshold": float(rescue_threshold),
        "rescue_eligible_count": int(eligible.sum().item()),
        "eligible_positive_count": int(eligible_positive.sum().item()),
        "eligible_negative_count": int(eligible_negative.sum().item()),
        "eligible_positive_rescue_above_threshold": int(corrected_positive.sum().item()),
        "eligible_positive_rescue_below_threshold": int(remaining_false_negative.sum().item()),
        "eligible_negative_rescue_above_threshold": int(regressed_true_negative.sum().item()),
        "eligible_negative_rescue_below_threshold": int(preserved_true_negative.sum().item()),
        "eligible_positive_rescue_probability": pos_dist,
        "eligible_negative_rescue_probability": neg_dist,
        "positive_mean_minus_negative_mean": float(pos_dist["mean"]) - float(neg_dist["mean"]),
        "positive_median_minus_negative_median": float(pos_dist["median"]) - float(neg_dist["median"]),
        "positive_over_negative_rank_fraction": _pairwise_rank_fraction(pos_scores, neg_scores, label=f"{label}:within-domain"),
    }


def _domain_forensics(*, digit: str, frozen_model, rescue_model, features, targets, label: str):
    torch, _nn = v52b._import_torch()
    frozen_probability = v53g._frozen_probabilities_from_features(frozen_model, features, digit=digit)
    rescue_probability = v53i._probabilities_from_rescue(rescue_model, features, digit=digit)
    evidence = _score_group_diagnostics(
        frozen_probability=frozen_probability, rescue_probability=rescue_probability, targets=targets,
        frozen_threshold=v52b.FROZEN_THRESHOLDS[digit], rescue_threshold=v53e.RESCUE_THRESHOLD, label=label,
    )
    y = targets.detach().cpu().to(dtype=torch.float32).reshape(-1) == 1.0
    eligible = frozen_probability < v52b.FROZEN_THRESHOLDS[digit]
    return evidence, rescue_probability[eligible & y].clone(), rescue_probability[eligible & ~y].clone()


def _failure_signature(*, digit: str, v5: Mapping[str, object], historical: Mapping[str, object], v5_positive_scores, historical_negative_scores) -> dict[str, object]:
    v5_positive_count = int(v5["eligible_positive_count"])
    v5_corrected = int(v5["eligible_positive_rescue_above_threshold"])
    hist_regressions = int(historical["eligible_negative_rescue_above_threshold"])
    if v5_corrected == v5_positive_count and hist_regressions > 0:
        signature = "V5_RECOVERED_HISTORICAL_TN_COLLAPSE"
    elif v5_corrected == 0 and hist_regressions > 0:
        signature = "V5_POSITIVE_NOT_RECOVERED_HISTORICAL_TN_COLLAPSE"
    elif v5_corrected < v5_positive_count and hist_regressions > 0:
        signature = "PARTIAL_V5_RECOVERY_HISTORICAL_TN_COLLAPSE"
    elif v5_corrected < v5_positive_count:
        signature = "V5_RECOVERY_FAILURE_WITHOUT_HISTORICAL_TN_COLLAPSE"
    else:
        signature = "NO_FAILURE_SIGNATURE"

    cross_rank = _pairwise_rank_fraction(v5_positive_scores, historical_negative_scores, label=f"{digit}-AI:cross-domain-v5-positive-vs-historical-negative")
    v5_pos_dist = _probability_distribution(v5_positive_scores, label=f"{digit}-AI:cross-v5-positive")
    hist_neg_dist = _probability_distribution(historical_negative_scores, label=f"{digit}-AI:cross-historical-negative")
    return {
        "signature": signature,
        "v5_positive_recovery_fraction": v5_corrected / v5_positive_count,
        "historical_true_negative_regression_count": hist_regressions,
        "cross_domain_v5_positive_over_historical_negative_rank_fraction": cross_rank,
        "cross_domain_v5_positive_mean_minus_historical_negative_mean": float(v5_pos_dist["mean"]) - float(hist_neg_dist["mean"]),
        "cross_domain_v5_positive_median_minus_historical_negative_median": float(v5_pos_dist["median"]) - float(hist_neg_dist["median"]),
        "score_ordering_conflict_observed": bool(cross_rank < 0.5),
        "fixed_threshold_separates_required_groups": bool(v5_corrected == v5_positive_count and hist_regressions == 0),
        "interpretation_scope": "descriptive TRAIN-only evidence; no threshold, architecture, or retraining authorization",
    }


def run_rescue_failure_forensics_v1(v5_data_root: str | Path, *, m4a_root: str | Path, d10_root: str | Path, digit2_frozen: str | Path, digit3_frozen: str | Path, v53g_report: str | Path, v53h_envelope: str | Path, v53i_report: str | Path, rescue_artifact_dir: str | Path, progress: ProgressCallback | None = None) -> dict[str, object]:
    """Describe the V5-3I HOLD on TRAIN-only surfaces without fitting."""
    root = Path(v5_data_root)
    hold_report = _read_bound_json(Path(v53i_report), expected_sha256=EXPECTED_V53I_REPORT_SHA256, label="V5-3I HOLD report")
    _validate_v53i_hold_receipt(hold_report)

    v53g_payload = v53i._read_json_bound(Path(v53g_report), expected_sha256=v53i.EXPECTED_V53G_REPORT_SHA256, label="V5-3G report")
    v53h_payload = v53i._read_json_bound(Path(v53h_envelope), expected_sha256=v53i.EXPECTED_V53H_ENVELOPE_SHA256, label="V5-3H envelope")
    v53i._validate_execution_receipt(report=v53g_payload, envelope=v53h_payload)

    frozen_models = v52n._frozen_models(digit2_frozen=Path(digit2_frozen), digit3_frozen=Path(digit3_frozen))
    frozen_state_before = {digit: v53f._state_fingerprint(frozen_models[digit]) for digit in ("2", "3")}
    manifest_path, _rows, v5_features, v5_targets, _frozen_v5_metrics = v52n._v5_surface(root, frozen_models)
    actual_slot_manifest_sha = v52b._sha_file(manifest_path)
    if actual_slot_manifest_sha != v53g_payload.get("slot_manifest_sha256"):
        _fail("V5 slot manifest no longer matches V5-3G receipt")

    historical_features, historical_targets = v52n._historical_surface(m4a_root=Path(m4a_root), d10_root=Path(d10_root), models=frozen_models, progress=progress)
    per_specialist_receipt = v53g_payload.get("per_specialist")
    if not isinstance(per_specialist_receipt, Mapping):
        _fail("V5-3G per-specialist receipt missing")

    rescue_models: dict[str, object] = {}
    rescue_state_before: dict[str, str] = {}
    for digit in ("2", "3"):
        item = per_specialist_receipt.get(digit)
        if not isinstance(item, Mapping):
            _fail(f"{digit}-AI V5-3G receipt missing")
        item_with_slot = dict(item)
        item_with_slot["_slot_manifest_sha256"] = actual_slot_manifest_sha
        rescue_models[digit] = v53i._load_rescue_artifact(Path(rescue_artifact_dir) / f"digit_{digit}_rescue.pt", digit=digit, report_item=item_with_slot)
        rescue_state_before[digit] = v53f._state_fingerprint(rescue_models[digit])

    per_specialist: dict[str, object] = {}
    for digit in ("2", "3"):
        item = per_specialist_receipt.get(digit)
        if not isinstance(item, Mapping):
            _fail(f"{digit}-AI V5-3G receipt missing")
        groups, group_evidence = v53g._materialize_frozen_negative_groups_v1(
            digit=digit, model=frozen_models[digit], v5_features=v5_features[digit], v5_targets=v5_targets[digit],
            historical_features=historical_features[digit], historical_targets=historical_targets[digit], enforce_preregistered_counts=True,
        )
        del groups
        materialization = item.get("materialization")
        if not isinstance(materialization, Mapping):
            _fail(f"{digit}-AI materialization receipt missing")
        if group_evidence.get("group_counts") != materialization.get("group_counts") or group_evidence.get("group_fingerprints") != materialization.get("group_fingerprints"):
            _fail(f"{digit}-AI rematerialized group identity changed")

        v5_diag, v5_positive_scores, _v5_negative_scores = _domain_forensics(
            digit=digit, frozen_model=frozen_models[digit], rescue_model=rescue_models[digit],
            features=v5_features[digit], targets=v5_targets[digit], label=f"{digit}-AI:V5-TRAIN",
        )
        hist_diag, _hist_positive_scores, hist_negative_scores = _domain_forensics(
            digit=digit, frozen_model=frozen_models[digit], rescue_model=rescue_models[digit],
            features=historical_features[digit], targets=historical_targets[digit], label=f"{digit}-AI:HISTORICAL-TRAIN",
        )
        acceptance_item = hold_report["per_specialist"][digit]
        expected_v5_reg = acceptance_item["v5_train"]["frozen_correct_regression_count"]
        expected_hist_reg = acceptance_item["historical_train"]["frozen_correct_regression_count"]
        if v5_diag["eligible_negative_rescue_above_threshold"] != expected_v5_reg:
            _fail(f"{digit}-AI V5 regression witness did not reproduce V5-3I")
        if hist_diag["eligible_negative_rescue_above_threshold"] != expected_hist_reg:
            _fail(f"{digit}-AI historical regression witness did not reproduce V5-3I")

        per_specialist[digit] = {
            "v5_train": v5_diag,
            "historical_train": hist_diag,
            "failure_signature": _failure_signature(
                digit=digit, v5=v5_diag, historical=hist_diag,
                v5_positive_scores=v5_positive_scores, historical_negative_scores=hist_negative_scores,
            ),
            "v5_3i_acceptance_witness_reproduced": True,
            "group_identity_reverified": True,
            "rescue_artifact_sha256": v53i.EXPECTED_RESCUE_ARTIFACT_SHA256[digit],
        }

    frozen_state_after = {digit: v53f._state_fingerprint(frozen_models[digit]) for digit in ("2", "3")}
    rescue_state_after = {digit: v53f._state_fingerprint(rescue_models[digit]) for digit in ("2", "3")}
    if frozen_state_after != frozen_state_before:
        _fail("V5-3J mutated a frozen specialist")
    if rescue_state_after != rescue_state_before:
        _fail("V5-3J mutated a rescue artifact in memory")

    return {
        "schema": SCHEMA,
        "question": "why_did_v5_3i_train_acceptance_hold",
        "forensic_contract": forensic_contract(),
        "bound_evidence": {
            "v5_3i_head_sha": V53I_HEAD_SHA,
            "v5_3i_module_blob_sha": V53I_MODULE_BLOB_SHA,
            "v5_3i_report_sha256": EXPECTED_V53I_REPORT_SHA256,
            "v5_3g_report_sha256": v53i.EXPECTED_V53G_REPORT_SHA256,
            "v5_3h_envelope_sha256": v53i.EXPECTED_V53H_ENVELOPE_SHA256,
            "rescue_artifact_sha256": dict(v53i.EXPECTED_RESCUE_ARTIFACT_SHA256),
            "slot_manifest_sha256": actual_slot_manifest_sha,
        },
        "v5_3i_decision_reproduced": "HOLD",
        "per_specialist": per_specialist,
        "frozen_state_bit_identical": True,
        "rescue_state_bit_identical_during_forensics": True,
        "diagnosis_scope": "TRAIN-only descriptive forensics",
        "repair_recipe_selected": False,
        "retraining_authorized": False,
        "historical_validation_opened": False,
        "first30_opened": False,
        "v5_reserve_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        **safety_boundary(),
    }


def retraining_allowed_after_forensics() -> bool:
    return False


def threshold_tuning_allowed() -> bool:
    return False


def historical_validation_access_allowed() -> bool:
    return False


def first30_access_allowed() -> bool:
    return False


def v5_validation_access_allowed() -> bool:
    return False


def final_holdout_access_allowed() -> bool:
    return False


def future_gate_order() -> tuple[str, ...]:
    return (
        "v5_3j_train_only_failure_forensics",
        "separately_preregistered_repair_hypothesis_if_supported",
        "separately_authorized_single_repair_execution_if_approved",
        "new_train_acceptance_gate",
        "historical_validation_retention_only_after_train_acceptance_pass",
    )
