"""Meter V5-3K TRAIN-only feature/domain-shift forensics.

V5-3K is opened only after the completed V5-3J rescue-failure forensic report.
It uses the exact frozen 2-AI/3-AI TRAIN surfaces and exact V5-3G rescue
artifacts to describe where V5 and historical domains diverge in the frozen
64D representation and in the rescue network's fixed 8D hidden representation.

This stage is descriptive only. It performs no fitting, no threshold search,
no parameter update, no checkpoint/artifact mutation, and no protected
validation access. Historical Validation, immutable First-30, V5 reserve,
V5 validation and FINAL_HOLDOUT remain closed.
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
from . import meter_v5_3j_rescue_failure_forensics_v1 as v53j


SCHEMA: Final[str] = "st-omr-meter-v5-3k-feature-domain-shift-forensics-v1"
REPORT_NAME: Final[str] = "v5_3k_feature_domain_shift_forensics_v1.json"
V53J_FINAL_HEAD_SHA: Final[str] = "08b2458cf6fa4aee3e5f32d1aefbe637cdbd01ec"
V53J_IMPLEMENTATION_HEAD_SHA: Final[str] = "c978b14fba23f91c60f06d2166bb23e87856d8d6"
V53J_MODULE_BLOB_SHA: Final[str] = "092a32504ffee9b9aafa74ddefea1c2aeb831e56"
EXPECTED_V53J_REPORT_SHA256: Final[str] = (
    "7a49d29e0d7257be7c59d499ab3d9ab575d369a7473b0b5298ea62aa80c7d37f"
)
EXPECTED_FAILURE_SIGNATURES: Final[dict[str, str]] = {
    "2": "V5_RECOVERED_HISTORICAL_TN_COLLAPSE",
    "3": "V5_POSITIVE_NOT_RECOVERED_HISTORICAL_TN_COLLAPSE",
}
EXPECTED_HISTORICAL_TN_REGRESSIONS: Final[dict[str, int]] = {
    "2": 5307,
    "3": 15775,
}
EXPECTED_V5_POSITIVE_RECOVERY_FRACTION: Final[dict[str, float]] = {
    "2": 1.0,
    "3": 0.0,
}
TOP_FEATURE_DIMENSIONS: Final[int] = 10
TOP_HIDDEN_DIMENSIONS: Final[int] = 8
STANDARDIZATION_EPSILON: Final[float] = 1e-12

ProgressCallback = Callable[[int, int, str], None]


class MeterV5_3KError(RuntimeError):
    """Raised when V5-3K departs from the read-only forensic contract."""


def _fail(message: str) -> None:
    raise MeterV5_3KError(message)


def safety_boundary() -> dict[str, object]:
    return {
        "training": False,
        "fitting": False,
        "autograd_grad_used": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_write": False,
        "rescue_artifact_write": False,
        "frozen_model_mutation_allowed": False,
        "rescue_model_mutation_allowed": False,
        "threshold_tuning": False,
        "threshold_sweep": False,
        "hyperparameter_sweep": False,
        "automatic_second_configuration": False,
        "architecture_change_authorized": False,
        "repair_recipe_selected": False,
        "retraining_authorized": False,
        "v5_adaptation_train_read": True,
        "historical_train_read": True,
        "frozen_checkpoint_read": True,
        "rescue_artifact_read": True,
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
        "prerequisite_v5_3j_final_head": V53J_FINAL_HEAD_SHA,
        "prerequisite_v5_3j_implementation_head": V53J_IMPLEMENTATION_HEAD_SHA,
        "prerequisite_v5_3j_module_blob": V53J_MODULE_BLOB_SHA,
        "bound_v5_3j_report_sha256": EXPECTED_V53J_REPORT_SHA256,
        "fixed_rescue_threshold": v53e.RESCUE_THRESHOLD,
        "fixed_frozen_thresholds": {
            "2": v52b.FROZEN_THRESHOLDS["2"],
            "3": v52b.FROZEN_THRESHOLDS["3"],
        },
        "frozen_feature_dimension": v53e.FEATURE_DIM,
        "rescue_hidden_dimension": v53e.HIDDEN_WIDTH,
        "top_feature_dimensions_reported": TOP_FEATURE_DIMENSIONS,
        "top_hidden_dimensions_reported": TOP_HIDDEN_DIMENSIONS,
        "descriptive_comparisons": (
            "same-label V5 TN vs historical TN",
            "same-label V5 FN-positive vs historical FN-positive",
            "critical V5 positive vs historical TN",
            "historical fixed-threshold hard TN vs preserved TN",
        ),
        "no_classifier_fit": True,
        "no_pca_fit": True,
        "no_threshold_selection": True,
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
        raise MeterV5_3KError(f"invalid {label} JSON") from exc
    if not isinstance(payload, dict):
        _fail(f"{label} JSON must be an object")
    return payload


def _validate_v53j_report(report: Mapping[str, object]) -> None:
    expected_scalars = {
        "schema": v53j.SCHEMA,
        "v5_3i_decision_reproduced": "HOLD",
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
    }
    for key, expected in expected_scalars.items():
        if report.get(key) != expected:
            _fail(f"V5-3J report field changed: {key}")

    bound = report.get("bound_evidence")
    if not isinstance(bound, Mapping):
        _fail("V5-3J bound evidence missing")
    if bound.get("v5_3i_head_sha") != v53j.V53I_HEAD_SHA:
        _fail("V5-3J V5-3I binding changed")
    if bound.get("v5_3i_report_sha256") != v53j.EXPECTED_V53I_REPORT_SHA256:
        _fail("V5-3J V5-3I report binding changed")
    if bound.get("v5_3g_report_sha256") != v53i.EXPECTED_V53G_REPORT_SHA256:
        _fail("V5-3J V5-3G report binding changed")
    if bound.get("v5_3h_envelope_sha256") != v53i.EXPECTED_V53H_ENVELOPE_SHA256:
        _fail("V5-3J V5-3H envelope binding changed")
    if bound.get("rescue_artifact_sha256") != v53i.EXPECTED_RESCUE_ARTIFACT_SHA256:
        _fail("V5-3J rescue artifact binding changed")

    per_specialist = report.get("per_specialist")
    if not isinstance(per_specialist, Mapping) or set(per_specialist) != {"2", "3"}:
        _fail("V5-3J per-specialist evidence changed")
    for digit in ("2", "3"):
        item = per_specialist.get(digit)
        if not isinstance(item, Mapping):
            _fail(f"{digit}-AI V5-3J evidence missing")
        signature = item.get("failure_signature")
        if not isinstance(signature, Mapping):
            _fail(f"{digit}-AI V5-3J failure signature missing")
        if signature.get("signature") != EXPECTED_FAILURE_SIGNATURES[digit]:
            _fail(f"{digit}-AI V5-3J failure signature changed")
        if signature.get("historical_true_negative_regression_count") != EXPECTED_HISTORICAL_TN_REGRESSIONS[digit]:
            _fail(f"{digit}-AI V5-3J historical regression witness changed")
        if signature.get("v5_positive_recovery_fraction") != EXPECTED_V5_POSITIVE_RECOVERY_FRACTION[digit]:
            _fail(f"{digit}-AI V5-3J V5 recovery witness changed")
        if item.get("v5_3i_acceptance_witness_reproduced") is not True:
            _fail(f"{digit}-AI V5-3J acceptance witness was not reproduced")
        if item.get("group_identity_reverified") is not True:
            _fail(f"{digit}-AI V5-3J group identity was not reverified")


def _finite_matrix(values, *, expected_dim: int, label: str):
    torch, _nn = v52b._import_torch()
    if not isinstance(values, torch.Tensor):
        _fail(f"{label} must be a torch.Tensor")
    matrix = values.detach().cpu().to(dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[1] != expected_dim or matrix.shape[0] <= 0:
        _fail(f"{label} shape changed: {tuple(matrix.shape)}")
    if not bool(torch.isfinite(matrix).all().item()):
        _fail(f"{label} contains non-finite values")
    return matrix


def _matrix_summary(values, *, expected_dim: int, label: str) -> dict[str, object]:
    matrix = _finite_matrix(values, expected_dim=expected_dim, label=label)
    mean = matrix.mean(dim=0)
    std = matrix.std(dim=0, unbiased=False)
    return {
        "count": int(matrix.shape[0]),
        "dimension": int(matrix.shape[1]),
        "mean": [float(x) for x in mean.tolist()],
        "std_population": [float(x) for x in std.tolist()],
        "mean_l2_norm": float(mean.square().sum().sqrt().item()),
        "mean_absolute_std": float(std.abs().mean().item()),
    }


def _centroid_geometry(a, b, *, expected_dim: int, label: str) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    left = _finite_matrix(a, expected_dim=expected_dim, label=f"{label}:a")
    right = _finite_matrix(b, expected_dim=expected_dim, label=f"{label}:b")
    mean_left = left.mean(dim=0)
    mean_right = right.mean(dim=0)
    delta = mean_left - mean_right
    centroid_distance = float(delta.square().sum().sqrt().item())
    left_rms = float(((left - mean_left).square().sum(dim=1).mean()).sqrt().item())
    right_rms = float(((right - mean_right).square().sum(dim=1).mean()).sqrt().item())
    denom = left_rms + right_rms + STANDARDIZATION_EPSILON
    left_norm = float(mean_left.square().sum().sqrt().item())
    right_norm = float(mean_right.square().sum().sqrt().item())
    cosine = None
    if left_norm > STANDARDIZATION_EPSILON and right_norm > STANDARDIZATION_EPSILON:
        cosine = float((mean_left @ mean_right).item() / (left_norm * right_norm))
        cosine = max(-1.0, min(1.0, cosine))
    return {
        "dimension": expected_dim,
        "centroid_l2_distance": centroid_distance,
        "a_within_centroid_rms": left_rms,
        "b_within_centroid_rms": right_rms,
        "centroid_distance_over_sum_within_rms": centroid_distance / denom,
        "centroid_cosine_similarity": cosine,
    }


def _standardized_mean_shift(a, b, *, expected_dim: int, top_k: int, label: str) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    left = _finite_matrix(a, expected_dim=expected_dim, label=f"{label}:a")
    right = _finite_matrix(b, expected_dim=expected_dim, label=f"{label}:b")
    mean_left = left.mean(dim=0)
    mean_right = right.mean(dim=0)
    std_left = left.std(dim=0, unbiased=False)
    std_right = right.std(dim=0, unbiased=False)
    pooled = torch.sqrt((std_left.square() + std_right.square()) / 2.0 + STANDARDIZATION_EPSILON)
    signed_effect = (mean_left - mean_right) / pooled
    records = []
    for index in range(expected_dim):
        records.append(
            {
                "dimension": index,
                "a_mean": float(mean_left[index].item()),
                "b_mean": float(mean_right[index].item()),
                "signed_standardized_mean_shift": float(signed_effect[index].item()),
                "absolute_standardized_mean_shift": float(abs(signed_effect[index].item())),
                "pooled_population_scale": float(pooled[index].item()),
            }
        )
    records.sort(key=lambda row: (-float(row["absolute_standardized_mean_shift"]), int(row["dimension"])))
    top = records[:top_k]
    finite_effects = [float(abs(value)) for value in signed_effect.tolist() if math.isfinite(float(value))]
    if len(finite_effects) != expected_dim:
        _fail(f"{label} standardized shift contains non-finite values")
    return {
        "dimension": expected_dim,
        "top_k": top_k,
        "max_absolute_standardized_mean_shift": max(finite_effects),
        "mean_absolute_standardized_mean_shift": sum(finite_effects) / len(finite_effects),
        "top_dimensions": top,
    }


def _hidden_activations(rescue_model, features, *, label: str):
    torch, _nn = v52b._import_torch()
    x = _finite_matrix(features, expected_dim=v53e.FEATURE_DIM, label=f"{label}:features").to(dtype=torch.float32)
    if not hasattr(rescue_model, "hidden") or not hasattr(rescue_model, "activation") or not hasattr(rescue_model, "output"):
        _fail(f"{label} rescue model shape changed")
    with torch.no_grad():
        hidden = rescue_model.activation(rescue_model.hidden(x)).detach().cpu().to(dtype=torch.float64)
    if hidden.ndim != 2 or hidden.shape != (x.shape[0], v53e.HIDDEN_WIDTH):
        _fail(f"{label} hidden activation shape changed: {tuple(hidden.shape)}")
    if not bool(torch.isfinite(hidden).all().item()):
        _fail(f"{label} hidden activations are non-finite")
    return hidden


def _rescue_probabilities(rescue_model, features, *, digit: str, label: str):
    torch, _nn = v52b._import_torch()
    x = _finite_matrix(features, expected_dim=v53e.FEATURE_DIM, label=f"{label}:features").to(dtype=torch.float32)
    probabilities = v53i._probabilities_from_rescue(rescue_model, x, digit=digit)
    p = probabilities.detach().cpu().to(dtype=torch.float64).reshape(-1)
    if p.shape[0] != x.shape[0] or not bool(torch.isfinite(p).all().item()):
        _fail(f"{label} rescue probability shape/non-finite failure")
    if bool(((p < 0.0) | (p > 1.0)).any().item()):
        _fail(f"{label} rescue probabilities outside [0,1]")
    return p


def _output_gap_decomposition(rescue_model, a_hidden, b_hidden, *, label: str) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    left = _finite_matrix(a_hidden, expected_dim=v53e.HIDDEN_WIDTH, label=f"{label}:a-hidden")
    right = _finite_matrix(b_hidden, expected_dim=v53e.HIDDEN_WIDTH, label=f"{label}:b-hidden")
    weight = rescue_model.output.weight.detach().cpu().to(dtype=torch.float64).reshape(-1)
    bias = rescue_model.output.bias.detach().cpu().to(dtype=torch.float64).reshape(-1)
    if weight.numel() != v53e.HIDDEN_WIDTH or bias.numel() != 1:
        _fail(f"{label} output-layer shape changed")
    delta_hidden = left.mean(dim=0) - right.mean(dim=0)
    contributions = weight * delta_hidden
    a_mean_logit = float((weight @ left.mean(dim=0) + bias[0]).item())
    b_mean_logit = float((weight @ right.mean(dim=0) + bias[0]).item())
    rows = [
        {
            "hidden_dimension": index,
            "output_weight": float(weight[index].item()),
            "mean_activation_gap": float(delta_hidden[index].item()),
            "logit_gap_contribution": float(contributions[index].item()),
            "absolute_logit_gap_contribution": float(abs(contributions[index].item())),
        }
        for index in range(v53e.HIDDEN_WIDTH)
    ]
    rows.sort(key=lambda row: (-float(row["absolute_logit_gap_contribution"]), int(row["hidden_dimension"])))
    contribution_sum = float(contributions.sum().item())
    direct_gap = a_mean_logit - b_mean_logit
    if not math.isclose(contribution_sum, direct_gap, rel_tol=1e-10, abs_tol=1e-10):
        _fail(f"{label} output decomposition did not close")
    return {
        "a_mean_logit": a_mean_logit,
        "b_mean_logit": b_mean_logit,
        "mean_logit_gap_a_minus_b": direct_gap,
        "output_bias": float(bias[0].item()),
        "contribution_sum": contribution_sum,
        "hidden_dimension_contributions": rows,
    }


def _pair_diagnostics(
    *,
    a_features,
    b_features,
    a_hidden,
    b_hidden,
    rescue_model,
    label: str,
    include_output_decomposition: bool,
) -> dict[str, object]:
    result = {
        "feature_64d": {
            "a_summary": _matrix_summary(a_features, expected_dim=v53e.FEATURE_DIM, label=f"{label}:feature-a"),
            "b_summary": _matrix_summary(b_features, expected_dim=v53e.FEATURE_DIM, label=f"{label}:feature-b"),
            "geometry": _centroid_geometry(a_features, b_features, expected_dim=v53e.FEATURE_DIM, label=f"{label}:feature-geometry"),
            "standardized_mean_shift": _standardized_mean_shift(
                a_features,
                b_features,
                expected_dim=v53e.FEATURE_DIM,
                top_k=TOP_FEATURE_DIMENSIONS,
                label=f"{label}:feature-shift",
            ),
        },
        "hidden_8d": {
            "a_summary": _matrix_summary(a_hidden, expected_dim=v53e.HIDDEN_WIDTH, label=f"{label}:hidden-a"),
            "b_summary": _matrix_summary(b_hidden, expected_dim=v53e.HIDDEN_WIDTH, label=f"{label}:hidden-b"),
            "geometry": _centroid_geometry(a_hidden, b_hidden, expected_dim=v53e.HIDDEN_WIDTH, label=f"{label}:hidden-geometry"),
            "standardized_mean_shift": _standardized_mean_shift(
                a_hidden,
                b_hidden,
                expected_dim=v53e.HIDDEN_WIDTH,
                top_k=TOP_HIDDEN_DIMENSIONS,
                label=f"{label}:hidden-shift",
            ),
        },
    }
    if include_output_decomposition:
        result["output_logit_gap_decomposition"] = _output_gap_decomposition(
            rescue_model,
            a_hidden,
            b_hidden,
            label=f"{label}:output",
        )
    return result


def _validate_group_identity(
    *,
    digit: str,
    group_evidence: Mapping[str, object],
    v53g_item: Mapping[str, object],
) -> None:
    materialization = v53g_item.get("materialization")
    if not isinstance(materialization, Mapping):
        _fail(f"{digit}-AI V5-3G materialization receipt missing")
    if group_evidence.get("group_counts") != materialization.get("group_counts"):
        _fail(f"{digit}-AI rematerialized group counts changed")
    if group_evidence.get("group_fingerprints") != materialization.get("group_fingerprints"):
        _fail(f"{digit}-AI rematerialized group fingerprints changed")


def run_feature_domain_shift_forensics_v1(
    v5_data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    v53g_report: str | Path,
    v53h_envelope: str | Path,
    v53j_report: str | Path,
    rescue_artifact_dir: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Describe frozen-feature and rescue-hidden domain shift on TRAIN only."""
    root = Path(v5_data_root)
    j_report = _read_bound_json(
        Path(v53j_report),
        expected_sha256=EXPECTED_V53J_REPORT_SHA256,
        label="V5-3J forensic report",
    )
    _validate_v53j_report(j_report)

    g_report = v53i._read_json_bound(
        Path(v53g_report),
        expected_sha256=v53i.EXPECTED_V53G_REPORT_SHA256,
        label="V5-3G report",
    )
    h_envelope = v53i._read_json_bound(
        Path(v53h_envelope),
        expected_sha256=v53i.EXPECTED_V53H_ENVELOPE_SHA256,
        label="V5-3H envelope",
    )
    v53i._validate_execution_receipt(report=g_report, envelope=h_envelope)

    frozen_models = v52n._frozen_models(
        digit2_frozen=Path(digit2_frozen),
        digit3_frozen=Path(digit3_frozen),
    )
    frozen_before = {digit: v53f._state_fingerprint(frozen_models[digit]) for digit in ("2", "3")}

    manifest_path, _rows, v5_features, v5_targets, _frozen_v5_metrics = v52n._v5_surface(root, frozen_models)
    actual_slot_manifest_sha = v52b._sha_file(manifest_path)
    if actual_slot_manifest_sha != g_report.get("slot_manifest_sha256"):
        _fail("V5 slot manifest no longer matches V5-3G receipt")

    historical_features, historical_targets = v52n._historical_surface(
        m4a_root=Path(m4a_root),
        d10_root=Path(d10_root),
        models=frozen_models,
        progress=progress,
    )

    per_g = g_report.get("per_specialist")
    if not isinstance(per_g, Mapping):
        _fail("V5-3G per-specialist receipt missing")

    rescue_models: dict[str, object] = {}
    rescue_before: dict[str, str] = {}
    for digit in ("2", "3"):
        item = per_g.get(digit)
        if not isinstance(item, Mapping):
            _fail(f"{digit}-AI V5-3G receipt missing")
        item_with_slot = dict(item)
        item_with_slot["_slot_manifest_sha256"] = actual_slot_manifest_sha
        rescue_models[digit] = v53i._load_rescue_artifact(
            Path(rescue_artifact_dir) / f"digit_{digit}_rescue.pt",
            digit=digit,
            report_item=item_with_slot,
        )
        rescue_before[digit] = v53f._state_fingerprint(rescue_models[digit])

    per_specialist: dict[str, object] = {}
    j_per = j_report.get("per_specialist")
    if not isinstance(j_per, Mapping):
        _fail("V5-3J per-specialist evidence missing")

    for digit in ("2", "3"):
        g_item = per_g.get(digit)
        if not isinstance(g_item, Mapping):
            _fail(f"{digit}-AI V5-3G receipt missing")
        groups, group_evidence = v53g._materialize_frozen_negative_groups_v1(
            digit=digit,
            model=frozen_models[digit],
            v5_features=v5_features[digit],
            v5_targets=v5_targets[digit],
            historical_features=historical_features[digit],
            historical_targets=historical_targets[digit],
            enforce_preregistered_counts=True,
        )
        _validate_group_identity(digit=digit, group_evidence=group_evidence, v53g_item=g_item)

        v5_pos = groups["v5_frozen_false_negative_positive"]
        v5_tn = groups["v5_frozen_true_negative"]
        hist_pos = groups["historical_frozen_false_negative_positive"]
        hist_tn = groups["historical_frozen_true_negative"]

        model = rescue_models[digit]
        v5_pos_hidden = _hidden_activations(model, v5_pos, label=f"{digit}:v5-pos")
        v5_tn_hidden = _hidden_activations(model, v5_tn, label=f"{digit}:v5-tn")
        hist_pos_hidden = _hidden_activations(model, hist_pos, label=f"{digit}:hist-pos")
        hist_tn_hidden = _hidden_activations(model, hist_tn, label=f"{digit}:hist-tn")

        hist_tn_probability = _rescue_probabilities(model, hist_tn, digit=digit, label=f"{digit}:hist-tn")
        hard_mask = hist_tn_probability >= float(v53e.RESCUE_THRESHOLD)
        preserved_mask = ~hard_mask
        hard_count = int(hard_mask.sum().item())
        preserved_count = int(preserved_mask.sum().item())
        if hard_count != EXPECTED_HISTORICAL_TN_REGRESSIONS[digit]:
            _fail(f"{digit}-AI fixed-threshold hard historical TN count changed: {hard_count}")
        if preserved_count <= 0:
            _fail(f"{digit}-AI has no preserved historical TN rows")

        hard_hist_tn = hist_tn[hard_mask]
        preserved_hist_tn = hist_tn[preserved_mask]
        hard_hist_hidden = hist_tn_hidden[hard_mask]
        preserved_hist_hidden = hist_tn_hidden[preserved_mask]

        v5_pos_probability = _rescue_probabilities(model, v5_pos, digit=digit, label=f"{digit}:v5-pos")
        recovered_fraction = float((v5_pos_probability >= float(v53e.RESCUE_THRESHOLD)).to(dtype=v5_pos_probability.dtype).mean().item())
        if recovered_fraction != EXPECTED_V5_POSITIVE_RECOVERY_FRACTION[digit]:
            _fail(f"{digit}-AI V5 positive recovery fraction changed")

        j_item = j_per.get(digit)
        if not isinstance(j_item, Mapping):
            _fail(f"{digit}-AI V5-3J evidence missing")
        j_signature = j_item.get("failure_signature")
        if not isinstance(j_signature, Mapping):
            _fail(f"{digit}-AI V5-3J signature missing")

        per_specialist[digit] = {
            "v5_3j_failure_signature": j_signature.get("signature"),
            "v5_3j_cross_domain_rank_fraction": j_signature.get("cross_domain_v5_positive_over_historical_negative_rank_fraction"),
            "fixed_threshold_witness": {
                "rescue_threshold": v53e.RESCUE_THRESHOLD,
                "v5_positive_count": int(v5_pos.shape[0]),
                "v5_positive_recovery_fraction": recovered_fraction,
                "historical_tn_count": int(hist_tn.shape[0]),
                "historical_hard_tn_count": hard_count,
                "historical_preserved_tn_count": preserved_count,
            },
            "group_identity_reverified": True,
            "same_label_negative_domain_shift": _pair_diagnostics(
                a_features=v5_tn,
                b_features=hist_tn,
                a_hidden=v5_tn_hidden,
                b_hidden=hist_tn_hidden,
                rescue_model=model,
                label=f"{digit}:v5-tn-vs-hist-tn",
                include_output_decomposition=True,
            ),
            "same_label_positive_domain_shift": _pair_diagnostics(
                a_features=v5_pos,
                b_features=hist_pos,
                a_hidden=v5_pos_hidden,
                b_hidden=hist_pos_hidden,
                rescue_model=model,
                label=f"{digit}:v5-pos-vs-hist-pos",
                include_output_decomposition=True,
            ),
            "critical_v5_positive_vs_historical_tn": _pair_diagnostics(
                a_features=v5_pos,
                b_features=hist_tn,
                a_hidden=v5_pos_hidden,
                b_hidden=hist_tn_hidden,
                rescue_model=model,
                label=f"{digit}:v5-pos-vs-hist-tn",
                include_output_decomposition=True,
            ),
            "historical_hard_tn_subpopulation": _pair_diagnostics(
                a_features=hard_hist_tn,
                b_features=preserved_hist_tn,
                a_hidden=hard_hist_hidden,
                b_hidden=preserved_hist_hidden,
                rescue_model=model,
                label=f"{digit}:hard-hist-tn-vs-preserved-hist-tn",
                include_output_decomposition=True,
            ),
        }

    frozen_after = {digit: v53f._state_fingerprint(frozen_models[digit]) for digit in ("2", "3")}
    rescue_after = {digit: v53f._state_fingerprint(rescue_models[digit]) for digit in ("2", "3")}
    if frozen_after != frozen_before:
        _fail("V5-3K mutated a frozen specialist")
    if rescue_after != rescue_before:
        _fail("V5-3K mutated a rescue artifact in memory")

    return {
        "schema": SCHEMA,
        "question": "where_do_v5_and_historical_train_domains_diverge_in_frozen_and_rescue_representations",
        "forensic_contract": forensic_contract(),
        "bound_evidence": {
            "v5_3j_final_head_sha": V53J_FINAL_HEAD_SHA,
            "v5_3j_implementation_head_sha": V53J_IMPLEMENTATION_HEAD_SHA,
            "v5_3j_module_blob_sha": V53J_MODULE_BLOB_SHA,
            "v5_3j_report_sha256": EXPECTED_V53J_REPORT_SHA256,
            "v5_3g_report_sha256": v53i.EXPECTED_V53G_REPORT_SHA256,
            "v5_3h_envelope_sha256": v53i.EXPECTED_V53H_ENVELOPE_SHA256,
            "rescue_artifact_sha256": dict(v53i.EXPECTED_RESCUE_ARTIFACT_SHA256),
            "slot_manifest_sha256": actual_slot_manifest_sha,
        },
        "per_specialist": per_specialist,
        "frozen_state_bit_identical": True,
        "rescue_state_bit_identical_during_forensics": True,
        "diagnosis_scope": "TRAIN-only frozen-64D and rescue-hidden-8D descriptive forensics",
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
        "v5_3k_train_only_feature_domain_shift_forensics",
        "separately_preregistered_digit_specific_repair_hypothesis_if_supported",
        "separately_authorized_single_repair_execution_if_approved",
        "new_train_acceptance_gate",
        "historical_validation_retention_only_after_train_acceptance_pass",
    )
