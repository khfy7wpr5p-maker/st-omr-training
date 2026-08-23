"""Meter V5-2R TRAIN-only class/margin/gradient contribution audit.

Read-only follow-up to V5-2Q. It uses only the already-open V5 adaptation TRAIN
surface and historical M4A TRAIN surface, exact frozen 2-AI/3-AI checkpoints,
and the exact retained V5-2P HOLD candidate checkpoints.

No training, autograd, backward, optimizer step, checkpoint mutation, threshold
change, new objective, new solver, or repair selection is authorized here.
Historical VALIDATION, First-30, V5 VALIDATION, and FINAL_HOLDOUT stay closed.
Only aggregate TRAIN evidence is emitted; no per-example identities are written.
"""

import math
from pathlib import Path
from typing import Callable, Final, Mapping

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2n_frozen_feature_transfer_audit_v1 as v52n
from . import meter_v5_2p_fixed_bias_head_repair_v1 as v52p
from . import meter_v5_2p_numerical_evidence_guard_v1 as v52p_guard
from . import meter_v5_2q_historical_positive_margin_audit_v1 as v52q


SCHEMA: Final[str] = "st-omr-meter-v5-2r-train-class-margin-gradient-audit-v1"
REPORT_NAME: Final[str] = "v5_2r_train_class_margin_gradient_audit_v1.json"
EXPECTED_FEATURE_DIM: Final[int] = 64
EXPECTED_V5_COUNT: Final[int] = 540
EXPECTED_HISTORICAL_COUNT: Final[int] = 26_964
GROUPS: Final[tuple[str, ...]] = (
    "v5_positive",
    "v5_negative",
    "historical_positive",
    "historical_negative",
)
ProgressCallback = Callable[[int, int, str], None]


class MeterV5_2RError(RuntimeError):
    """Raised when the TRAIN-only V5-2R audit cannot prove its safety contract."""


def _fail(message: str) -> None:
    raise MeterV5_2RError(message)


def safety_boundary() -> dict[str, object]:
    """Declare the exact read-only authority of V5-2R."""
    return {
        "training": False,
        "autograd_grad_used": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_read": True,
        "checkpoint_write": False,
        "candidate_checkpoint_mutation": False,
        "evidence_report_write": True,
        "objective_changed": False,
        "new_objective_selected": False,
        "solver_settings_changed": False,
        "new_solver_selected": False,
        "domain_weights_changed": False,
        "threshold_tuning": False,
        "bias_tuning": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "reserve_v5_train_opened": False,
        "historical_validation_opened": False,
        "historical_validation_retention_report_read": False,
        "historical_validation_error_examples_read": False,
        "historical_validation_example_identities_emitted": False,
        "first30_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "per_example_rows_emitted": False,
        "repair_selected": False,
        "repair_training_authorized": False,
        "production_promotion": False,
    }


def _finite_tensor(tensor, *, name: str) -> None:
    torch, _nn = v52b._import_torch()
    if tensor.numel() == 0:
        _fail(f"empty tensor: {name}")
    if not bool(torch.isfinite(tensor).all().item()):
        _fail(f"non-finite tensor: {name}")


def _threshold_logit(threshold: float) -> float:
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        _fail(f"invalid frozen threshold: {threshold}")
    return math.log(threshold / (1.0 - threshold))


def _as_surface(features, targets, *, name: str):
    torch, _nn = v52b._import_torch()
    x = features.detach().cpu().to(dtype=torch.float64)
    y = targets.detach().cpu().to(dtype=torch.float64).reshape(-1)
    if x.ndim != 2 or x.shape[1] != EXPECTED_FEATURE_DIM:
        _fail(f"{name} feature shape changed: {tuple(x.shape)}")
    if len(x) != len(y):
        _fail(f"{name} feature/target cardinality mismatch")
    _finite_tensor(x, name=f"{name}-features")
    _finite_tensor(y, name=f"{name}-targets")
    if not bool(((y == 0.0) | (y == 1.0)).all().item()):
        _fail(f"{name} targets are not binary")
    return x, y


def _as_weight(weight, *, name: str):
    torch, _nn = v52b._import_torch()
    w = weight.detach().cpu().to(dtype=torch.float64).reshape(-1)
    if w.numel() != EXPECTED_FEATURE_DIM:
        _fail(f"{name} dimension changed: {w.numel()}")
    _finite_tensor(w, name=name)
    return w


def _cosine_or_none(a, b) -> float | None:
    torch, _nn = v52b._import_torch()
    x = a.detach().cpu().to(dtype=torch.float64).reshape(-1)
    y = b.detach().cpu().to(dtype=torch.float64).reshape(-1)
    if x.shape != y.shape or x.numel() == 0:
        _fail("cosine vector shape mismatch")
    _finite_tensor(x, name="cosine-a")
    _finite_tensor(y, name="cosine-b")
    nx = float(torch.linalg.vector_norm(x).item())
    ny = float(torch.linalg.vector_norm(y).item())
    if nx == 0.0 or ny == 0.0:
        return None
    value = float(torch.dot(x, y).item()) / (nx * ny)
    return min(1.0, max(-1.0, value))


def _angle_degrees_or_none(a, b) -> float | None:
    cosine = _cosine_or_none(a, b)
    if cosine is None:
        return None
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def _prediction_from_logits(logits, *, threshold_logit: float):
    return logits >= threshold_logit


def _historical_positive_transition_matrix(
    *,
    features,
    targets,
    frozen_weight,
    candidate_weight,
    frozen_bias: float,
    threshold: float,
) -> dict[str, int]:
    x, y = _as_surface(features, targets, name="historical-transition")
    w0 = _as_weight(frozen_weight, name="frozen-head-weight")
    w1 = _as_weight(candidate_weight, name="candidate-head-weight")
    if not math.isfinite(frozen_bias):
        _fail("frozen bias is non-finite")
    boundary = _threshold_logit(threshold)
    pos_x = x[y == 1.0]
    if pos_x.numel() == 0:
        _fail("historical positive transition matrix requires positives")
    frozen_correct = _prediction_from_logits(pos_x @ w0 + frozen_bias, threshold_logit=boundary)
    candidate_correct = _prediction_from_logits(
        pos_x @ w1 + frozen_bias, threshold_logit=boundary
    )
    correct_to_correct = int((frozen_correct & candidate_correct).sum().item())
    correct_to_wrong = int((frozen_correct & ~candidate_correct).sum().item())
    wrong_to_correct = int((~frozen_correct & candidate_correct).sum().item())
    wrong_to_wrong = int((~frozen_correct & ~candidate_correct).sum().item())
    total = correct_to_correct + correct_to_wrong + wrong_to_correct + wrong_to_wrong
    if total != int(pos_x.shape[0]):
        _fail("historical positive transition accounting mismatch")
    return {
        "positive_count": int(pos_x.shape[0]),
        "correct_to_correct": correct_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_correct": wrong_to_correct,
        "wrong_to_wrong": wrong_to_wrong,
    }


def _negative_margin_distribution(
    *,
    features,
    targets,
    weight,
    frozen_bias: float,
    threshold: float,
    name: str,
) -> dict[str, object]:
    x, y = _as_surface(features, targets, name=name)
    w = _as_weight(weight, name=f"{name}-weight")
    if not math.isfinite(frozen_bias):
        _fail("frozen bias is non-finite")
    boundary = _threshold_logit(threshold)
    neg_x = x[y == 0.0]
    if neg_x.numel() == 0:
        _fail(f"{name} requires negative examples")
    margins = boundary - (neg_x @ w + frozen_bias)
    return v52q._quantile_summary(margins, name=f"{name}-negative-margin")


def _head_geometry(*, frozen_weight, candidate_weight) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    w0 = _as_weight(frozen_weight, name="frozen-head-weight")
    w1 = _as_weight(candidate_weight, name="candidate-head-weight")
    delta = w1 - w0
    frozen_norm = float(torch.linalg.vector_norm(w0).item())
    candidate_norm = float(torch.linalg.vector_norm(w1).item())
    delta_norm = float(torch.linalg.vector_norm(delta).item())
    if frozen_norm == 0.0:
        delta_over_frozen = None
        candidate_over_frozen = None
    else:
        delta_over_frozen = delta_norm / frozen_norm
        candidate_over_frozen = candidate_norm / frozen_norm
    return {
        "frozen_weight_l2": frozen_norm,
        "candidate_weight_l2": candidate_norm,
        "delta_weight_l2": delta_norm,
        "delta_over_frozen_l2": delta_over_frozen,
        "candidate_over_frozen_l2": candidate_over_frozen,
        "frozen_candidate_cosine": _cosine_or_none(w0, w1),
        "head_angle_change_degrees": _angle_degrees_or_none(w0, w1),
        "delta_weight_max_abs": float(torch.max(torch.abs(delta)).item()),
    }


def _group_mean_bce_and_gradient(*, features, targets, weight, bias: float, name: str):
    """Closed-form BCE-with-logits mean and analytic d(mean BCE)/d(weight)."""
    torch, _nn = v52b._import_torch()
    x, y = _as_surface(features, targets, name=name)
    w = _as_weight(weight, name=f"{name}-weight")
    if not math.isfinite(bias):
        _fail("bias is non-finite")
    logits = x @ w + bias
    _finite_tensor(logits, name=f"{name}-logits")
    losses = torch.logaddexp(torch.zeros_like(logits), logits) - y * logits
    residual = torch.sigmoid(logits) - y
    gradient = x.transpose(0, 1) @ residual / float(x.shape[0])
    _finite_tensor(losses, name=f"{name}-bce")
    _finite_tensor(gradient, name=f"{name}-analytic-gradient")
    return float(losses.mean().item()), gradient


def _group_surfaces(hist_x, hist_y, v5_x, v5_y):
    return {
        "v5_positive": (v5_x[v5_y == 1.0], v5_y[v5_y == 1.0], "v5", 0.5),
        "v5_negative": (v5_x[v5_y == 0.0], v5_y[v5_y == 0.0], "v5", 0.5),
        "historical_positive": (
            hist_x[hist_y == 1.0],
            hist_y[hist_y == 1.0],
            "historical",
            0.5,
        ),
        "historical_negative": (
            hist_x[hist_y == 0.0],
            hist_y[hist_y == 0.0],
            "historical",
            0.5,
        ),
    }


def _gradient_summary(gradient, *, coefficient: float) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    g = gradient.detach().cpu().to(dtype=torch.float64).reshape(-1)
    _finite_tensor(g, name="gradient-summary")
    if not math.isfinite(coefficient) or coefficient < 0.0:
        _fail("invalid objective coefficient")
    return {
        "l2_norm": float(torch.linalg.vector_norm(g).item()),
        "inf_norm": float(torch.max(torch.abs(g)).item()),
        "mean_abs": float(torch.mean(torch.abs(g)).item()),
        "objective_coefficient": coefficient,
        "weighted_objective_gradient_l2_norm": float(
            torch.linalg.vector_norm(g * coefficient).item()
        ),
        "weighted_objective_gradient_inf_norm": float(
            torch.max(torch.abs(g * coefficient)).item()
        ),
    }


def _pairwise_gradient_cosines(gradients: Mapping[str, object]) -> dict[str, object]:
    matrix: dict[str, object] = {}
    for left in GROUPS:
        row: dict[str, object] = {}
        for right in GROUPS:
            row[right] = _cosine_or_none(gradients[left], gradients[right])
        matrix[left] = row
    return matrix


def _contribution_state(
    *,
    hist_x,
    hist_y,
    v5_x,
    v5_y,
    weight,
    bias: float,
) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    groups = _group_surfaces(hist_x, hist_y, v5_x, v5_y)
    domain_sizes = {"v5": int(v5_x.shape[0]), "historical": int(hist_x.shape[0])}
    gradients: dict[str, object] = {}
    group_report: dict[str, object] = {}
    total_weighted_bce = 0.0
    total_weighted_gradient = torch.zeros(EXPECTED_FEATURE_DIM, dtype=torch.float64)

    for group_name in GROUPS:
        xg, yg, domain, domain_weight = groups[group_name]
        if xg.shape[0] == 0:
            _fail(f"empty group: {group_name}")
        mean_bce, gradient = _group_mean_bce_and_gradient(
            features=xg,
            targets=yg,
            weight=weight,
            bias=bias,
            name=group_name,
        )
        empirical_frequency = float(xg.shape[0]) / float(domain_sizes[domain])
        coefficient = domain_weight * empirical_frequency
        total_weighted_bce += coefficient * mean_bce
        total_weighted_gradient = total_weighted_gradient + coefficient * gradient
        gradients[group_name] = gradient
        group_report[group_name] = {
            "count": int(xg.shape[0]),
            "domain": domain,
            "domain_weight_carried_from_v5_2p": domain_weight,
            "empirical_frequency_within_domain": empirical_frequency,
            "objective_coefficient": coefficient,
            "mean_bce": mean_bce,
            "weighted_objective_bce_contribution": coefficient * mean_bce,
            "analytic_mean_gradient": _gradient_summary(
                gradient, coefficient=coefficient
            ),
        }

    coefficient_sum = sum(
        float(group_report[name]["objective_coefficient"]) for name in GROUPS
    )
    if abs(coefficient_sum - 1.0) > 1e-12:
        _fail(f"group objective coefficients do not sum to 1: {coefficient_sum}")
    _finite_tensor(total_weighted_gradient, name="total-weighted-objective-gradient")

    return {
        "groups": group_report,
        "gradient_conflict_cosine_matrix": _pairwise_gradient_cosines(gradients),
        "objective_reconstruction": {
            "coefficient_sum": coefficient_sum,
            "total_weighted_bce": total_weighted_bce,
            "total_weighted_gradient_l2_norm": float(
                torch.linalg.vector_norm(total_weighted_gradient).item()
            ),
            "total_weighted_gradient_inf_norm": float(
                torch.max(torch.abs(total_weighted_gradient)).item()
            ),
        },
    }


def class_margin_gradient_audit_metrics_v1(
    *,
    historical_features,
    historical_targets,
    v5_features,
    v5_targets,
    frozen_weight,
    candidate_weight,
    frozen_bias: float,
    threshold: float,
) -> dict[str, object]:
    """Pure aggregate TRAIN-only class, margin, weight, BCE, and gradient evidence."""
    hist_x, hist_y = _as_surface(
        historical_features, historical_targets, name="historical"
    )
    v5_x, v5_y = _as_surface(v5_features, v5_targets, name="v5")
    w0 = _as_weight(frozen_weight, name="frozen-head-weight")
    w1 = _as_weight(candidate_weight, name="candidate-head-weight")
    if not math.isfinite(frozen_bias):
        _fail("frozen bias is non-finite")

    class_balance = {
        "v5_positive_count": int((v5_y == 1.0).sum().item()),
        "v5_negative_count": int((v5_y == 0.0).sum().item()),
        "historical_positive_count": int((hist_y == 1.0).sum().item()),
        "historical_negative_count": int((hist_y == 0.0).sum().item()),
        "v5_positive_fraction": float((v5_y == 1.0).to(dtype=hist_x.dtype).mean().item()),
        "historical_positive_fraction": float(
            (hist_y == 1.0).to(dtype=hist_x.dtype).mean().item()
        ),
    }

    transition = _historical_positive_transition_matrix(
        features=hist_x,
        targets=hist_y,
        frozen_weight=w0,
        candidate_weight=w1,
        frozen_bias=frozen_bias,
        threshold=threshold,
    )

    negative_margins = {
        "historical": {
            "frozen": _negative_margin_distribution(
                features=hist_x,
                targets=hist_y,
                weight=w0,
                frozen_bias=frozen_bias,
                threshold=threshold,
                name="historical-frozen",
            ),
            "candidate": _negative_margin_distribution(
                features=hist_x,
                targets=hist_y,
                weight=w1,
                frozen_bias=frozen_bias,
                threshold=threshold,
                name="historical-candidate",
            ),
        },
        "v5": {
            "frozen": _negative_margin_distribution(
                features=v5_x,
                targets=v5_y,
                weight=w0,
                frozen_bias=frozen_bias,
                threshold=threshold,
                name="v5-frozen",
            ),
            "candidate": _negative_margin_distribution(
                features=v5_x,
                targets=v5_y,
                weight=w1,
                frozen_bias=frozen_bias,
                threshold=threshold,
                name="v5-candidate",
            ),
        },
    }

    return {
        "threshold": float(threshold),
        "threshold_logit": _threshold_logit(threshold),
        "class_balance": class_balance,
        "historical_positive_transition_matrix": transition,
        "negative_margin_distribution": negative_margins,
        "head_geometry": _head_geometry(
            frozen_weight=w0,
            candidate_weight=w1,
        ),
        "gradient_and_bce_at_frozen_head": _contribution_state(
            hist_x=hist_x,
            hist_y=hist_y,
            v5_x=v5_x,
            v5_y=v5_y,
            weight=w0,
            bias=frozen_bias,
        ),
        "gradient_and_bce_at_candidate_head": _contribution_state(
            hist_x=hist_x,
            hist_y=hist_y,
            v5_x=v5_x,
            v5_y=v5_y,
            weight=w1,
            bias=frozen_bias,
        ),
        "descriptive_only": True,
        "mechanism_selected": False,
        "repair_selected": False,
        "new_objective_selected": False,
    }


def _load_exact_v5_2p_evidence(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Reuse V5-2Q's exact candidate binding without opening retention evidence."""
    return v52q._load_exact_v5_2p_evidence(root)


def run_train_class_margin_gradient_audit_v1(
    v5_data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run V5-2R on aggregate open TRAIN surfaces only."""
    root = Path(v5_data_root)
    ann = root / v51.ANNOTATIONS_DIR
    report_path = ann / REPORT_NAME
    if report_path.exists():
        _fail(f"refusing to overwrite V5-2R evidence: {report_path}")

    training, numerical = _load_exact_v5_2p_evidence(root)
    frozen_models = v52n._frozen_models(
        digit2_frozen=Path(digit2_frozen),
        digit3_frozen=Path(digit3_frozen),
    )
    manifest_path, _rows, v5_features, v5_targets, _metrics = v52n._v5_surface(
        root, frozen_models
    )
    historical_features, historical_targets = v52n._historical_surface(
        m4a_root=Path(m4a_root),
        d10_root=Path(d10_root),
        models=frozen_models,
        progress=progress,
    )

    expected_candidate_sha = {
        "2": v52q.DIGIT2_CANDIDATE_SHA256,
        "3": v52q.DIGIT3_CANDIDATE_SHA256,
    }
    per_specialist: dict[str, object] = {}

    for digit in ("2", "3"):
        item = training["candidates"][digit]
        candidate_path = Path(item["candidate_path"])
        if v52b._sha_file(candidate_path) != expected_candidate_sha[digit]:
            _fail(f"V5-2P {digit}-AI candidate file SHA changed")
        candidate_model = v52p._load_candidate(
            candidate_path,
            digit=digit,
            training_report=training,
        )
        frozen_model = frozen_models[digit]
        invariants = v52q.verify_candidate_frozen_surface_v1(
            frozen_model=frozen_model,
            candidate_model=candidate_model,
        )
        frozen_bias = float(
            frozen_model.head.bias.detach().cpu().reshape(-1)[0].item()
        )
        metrics = class_margin_gradient_audit_metrics_v1(
            historical_features=historical_features[digit],
            historical_targets=historical_targets[digit],
            v5_features=v5_features[digit],
            v5_targets=v5_targets[digit],
            frozen_weight=frozen_model.head.weight.detach().cpu().reshape(-1),
            candidate_weight=candidate_model.head.weight.detach().cpu().reshape(-1),
            frozen_bias=frozen_bias,
            threshold=v52b.FROZEN_THRESHOLDS[digit],
        )
        per_specialist[digit] = {
            "candidate_sha256": expected_candidate_sha[digit],
            "candidate_state_invariants": invariants,
            **metrics,
        }

    report: dict[str, object] = {
        "schema": SCHEMA,
        "question": (
            "is_v5_2p_train_geometry_consistent_with_class_imbalance_"
            "head_weight_growth_or_their_combination"
        ),
        "analysis_surface": "aggregate-open-train-only",
        "slot_manifest_sha256": v52b._sha_file(manifest_path),
        "v5_2p_training_report_sha256": v52b._sha_file(
            ann / v52p.TRAINING_REPORT_NAME
        ),
        "v5_2p_numerical_report_sha256": v52b._sha_file(
            ann / v52p_guard.REPORT_NAME
        ),
        "v5_adaptation_train_slot_count": EXPECTED_V5_COUNT,
        "m4a_historical_train_record_count": EXPECTED_HISTORICAL_COUNT,
        "feature_dim": EXPECTED_FEATURE_DIM,
        "candidate_checkpoint_sha256": expected_candidate_sha,
        "frozen_checkpoint_sha256": {
            "2": v52b.DIGIT2_SHA256,
            "3": v52b.DIGIT3_SHA256,
        },
        "v5_2p_domain_weights_carried_forward_for_evidence_only": {
            "v5": 0.5,
            "historical": 0.5,
        },
        "numerical_integrity_gate_carried_forward": numerical[
            "numerical_integrity_gate"
        ]["gate"],
        "historical_validation_hold_result_used_for_model_design": False,
        "historical_validation_examples_opened": False,
        "historical_validation_example_identities_opened": False,
        "historical_retention_report_read": False,
        "per_example_output": False,
        "per_specialist": per_specialist,
        **safety_boundary(),
    }
    v51._atomic_write_json(report_path, report)
    return report
