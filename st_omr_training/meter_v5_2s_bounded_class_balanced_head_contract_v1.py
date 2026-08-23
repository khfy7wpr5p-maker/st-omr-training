"""Preregistered, training-disabled Meter V5-2S repair contract.

V5-2R showed that the V5-2P equal-domain objective still assigned roughly
89 percent of its empirical coefficient to negative examples and allowed the
64D head vector to grow by more than 100x while rotating by roughly 86 degrees.

This module selects one later repair objective without executing it:

* four TRAIN-only domain/class groups receive equal 0.25 BCE coefficients;
* the frozen head is the centre of a deterministic proximal penalty;
* the penalty is scaled so any non-increasing final objective is confined to a
  15-degree frozen-head trust region;
* backbone, head bias, thresholds, 4-AI, and all closed data surfaces remain
  frozen.

There is deliberately no optimizer, autograd, backward call, checkpoint write,
or training entry point in this contract stage.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Final, Mapping

from . import meter_v5_2p_fixed_bias_head_repair_v1 as v52p


SCHEMA: Final[str] = "st-omr-meter-v5-2s-bounded-class-balanced-head-contract-v1"
V52R_IMPLEMENTATION_HEAD: Final[str] = "85c0b0083792e8b9ec60ee632cfc7015e885d548"
V52R_REPORT_SHA256: Final[str] = (
    "2c374189c285232eb79c7a8ca331d9a53b60286b36dde18d5dec559b14f58dc7"
)
V52R_ENVELOPE_SHA256: Final[str] = (
    "6a80aa1536722720f6a3a85d93d363e356830eb6ff711d0b41c4af5c45080226"
)

GROUPS: Final[tuple[str, ...]] = (
    "v5_positive",
    "v5_negative",
    "historical_positive",
    "historical_negative",
)
GROUP_LABELS: Final[dict[str, float]] = {
    "v5_positive": 1.0,
    "v5_negative": 0.0,
    "historical_positive": 1.0,
    "historical_negative": 0.0,
}
GROUP_WEIGHT: Final[float] = 0.25
EXPECTED_FEATURE_DIM: Final[int] = 64
MAX_HEAD_ANGLE_DEGREES: Final[float] = 15.0
OBJECTIVE_TOLERANCE: Final[float] = 1e-10
GEOMETRY_TOLERANCE: Final[float] = 1e-10


class MeterV5_2SError(RuntimeError):
    """Raised when the preregistered V5-2S contract cannot be proven."""


def _fail(message: str) -> None:
    raise MeterV5_2SError(message)


def safety_boundary() -> dict[str, object]:
    return {
        "contract_stage_only": True,
        "objective_preregistered": True,
        "training_authorized": False,
        "training_executed": False,
        "autograd_grad_used": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_write": False,
        "trainable_surface_if_later_authorized": "head.weight-only-64-parameters",
        "frozen_backbone": True,
        "frozen_head_bias": True,
        "runtime_threshold_tuning": False,
        "alternative_threshold_evaluated": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "reserve_v5_train_opened": False,
        "historical_validation_opened": False,
        "first30_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "automatic_second_configuration": False,
        "hyperparameter_sweep": False,
        "production_promotion": False,
    }


def prerequisite_evidence_contract() -> dict[str, object]:
    """Bind the later experiment to the exact completed V5-2R evidence."""
    return {
        "v5_2r_implementation_head": V52R_IMPLEMENTATION_HEAD,
        "v5_2r_report_sha256": V52R_REPORT_SHA256,
        "v5_2r_execution_envelope_sha256": V52R_ENVELOPE_SHA256,
        "evidence_surface": "v5-and-historical-train-only",
        "historical_retention_examples_used_for_design": False,
        "v5_validation_used_for_design": False,
        "final_holdout_used_for_design": False,
    }


def objective_contract() -> dict[str, object]:
    return {
        "formula": (
            "0.25*mean(BCE(V5+))+0.25*mean(BCE(V5-))+"
            "0.25*mean(BCE(HIST+))+0.25*mean(BCE(HIST-))+"
            "0.5*lambda*||w-w0||2^2"
        ),
        "group_weights": {name: GROUP_WEIGHT for name in GROUPS},
        "group_weight_sum": 1.0,
        "class_balanced_within_each_domain": True,
        "domain_weight_sum": {"v5": 0.5, "historical": 0.5},
        "frozen_reference": "exact-source-head-weight-w0",
        "maximum_head_angle_degrees": MAX_HEAD_ANGLE_DEGREES,
        "trust_radius_formula": "R=sin(15deg)*||w0||2",
        "proximal_lambda_formula": "lambda=2*L_balanced(w0)/R^2",
        "selection_method": "fixed-analytic-no-sweep-no-validation-selection",
        "head_bias_trainable": False,
        "backbone_trainable": False,
        "threshold_trainable": False,
    }


def solver_contract() -> dict[str, object]:
    """Carry forward one solver configuration without authorizing execution."""
    return {
        "execution_authorized": False,
        "optimizer_if_later_authorized": "LBFGS",
        "dtype": "float64-head-optimization-copy-back-float32",
        "lr": v52p.LBFGS_LR,
        "max_iter": v52p.LBFGS_MAX_ITER,
        "max_eval": v52p.LBFGS_MAX_EVAL,
        "history_size": v52p.LBFGS_HISTORY_SIZE,
        "tolerance_grad": v52p.LBFGS_TOLERANCE_GRAD,
        "tolerance_change": v52p.LBFGS_TOLERANCE_CHANGE,
        "line_search_fn": v52p.LBFGS_LINE_SEARCH,
        "initialization": "exact-frozen-head-weight",
        "candidate_selection": "single-final-solver-state-no-sweep",
        "automatic_second_configuration": False,
    }


def gate_order() -> tuple[str, ...]:
    return (
        "numerical_integrity_and_geometry",
        "historical_retention_v3",
        "immutable_v5_first30_diagnostic",
        "separately_authorized_v5_validation",
        "separately_authorized_final_holdout",
    )


def verify_exact_v5_2r_evidence(
    *, report_path: str | Path, envelope_path: str | Path
) -> dict[str, str]:
    report = Path(report_path)
    envelope = Path(envelope_path)
    if not report.is_file() or not envelope.is_file():
        _fail("exact V5-2R report/envelope missing")
    report_sha = hashlib.sha256(report.read_bytes()).hexdigest()
    envelope_sha = hashlib.sha256(envelope.read_bytes()).hexdigest()
    if report_sha != V52R_REPORT_SHA256:
        _fail("V5-2R report SHA256 mismatch")
    if envelope_sha != V52R_ENVELOPE_SHA256:
        _fail("V5-2R execution envelope SHA256 mismatch")
    return {"report_sha256": report_sha, "envelope_sha256": envelope_sha}


def _finite_values(values, *, name: str) -> tuple[float, ...]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    elif hasattr(values, "reshape") and hasattr(values, "tolist"):
        values = values.reshape(-1).tolist()
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise MeterV5_2SError(f"{name} must be a flat numeric sequence") from exc
    if not vector:
        _fail(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in vector):
        _fail(f"{name} contains non-finite values")
    return vector


def _weight_vector(weight, *, name: str) -> tuple[float, ...]:
    vector = _finite_values(weight, name=name)
    if len(vector) != EXPECTED_FEATURE_DIM:
        _fail(f"{name} must contain exactly {EXPECTED_FEATURE_DIM} values")
    return vector


def _validate_group(*, name: str, logits, targets) -> None:
    if name not in GROUP_LABELS:
        _fail(f"unexpected objective group: {name}")
    flat_logits = _finite_values(logits, name=f"{name} logits")
    flat_targets = _finite_values(targets, name=f"{name} targets")
    if len(flat_logits) != len(flat_targets):
        _fail(f"{name} objective surface is empty or misaligned")
    if any(target != GROUP_LABELS[name] for target in flat_targets):
        _fail(f"{name} contains the wrong class label")


def balanced_four_group_bce_v1(
    *, group_logits: Mapping[str, object], group_targets: Mapping[str, object]
) -> tuple[float, dict[str, float]]:
    """Return the fixed 0.25-per-group BCE without fitting any parameter."""
    if set(group_logits) != set(GROUPS) or set(group_targets) != set(GROUPS):
        _fail("four-group objective keys changed")
    losses: dict[str, float] = {}
    total = 0.0
    for name in GROUPS:
        logits = _finite_values(group_logits[name], name=f"{name} logits")
        targets = _finite_values(group_targets[name], name=f"{name} targets")
        _validate_group(name=name, logits=logits, targets=targets)
        loss = sum(
            max(logit, 0.0)
            - logit * target
            + math.log1p(math.exp(-abs(logit)))
            for logit, target in zip(logits, targets)
        ) / len(logits)
        losses[name] = loss
        total += GROUP_WEIGHT * loss
    if not math.isfinite(total):
        _fail("balanced four-group objective became non-finite")
    return total, losses


def derive_proximal_contract_v1(
    *, frozen_weight, initial_balanced_bce: float
) -> dict[str, float]:
    """Derive the unique penalty scale from TRAIN-only initial loss and w0."""
    w0 = _weight_vector(frozen_weight, name="frozen head weight")
    frozen_norm = math.sqrt(sum(value * value for value in w0))
    initial = float(initial_balanced_bce)
    if not math.isfinite(frozen_norm) or frozen_norm <= 0.0:
        _fail("frozen head weight norm must be finite and positive")
    if not math.isfinite(initial) or initial <= 0.0:
        _fail("initial balanced BCE must be finite and positive")
    radius_ratio = math.sin(math.radians(MAX_HEAD_ANGLE_DEGREES))
    radius = radius_ratio * frozen_norm
    proximal_lambda = 2.0 * initial / (radius * radius)
    penalty_at_radius = 0.5 * proximal_lambda * radius * radius
    return {
        "frozen_head_weight_l2": frozen_norm,
        "maximum_head_angle_degrees": MAX_HEAD_ANGLE_DEGREES,
        "trust_radius_over_frozen_l2": radius_ratio,
        "trust_radius_l2": radius,
        "proximal_lambda": proximal_lambda,
        "penalty_at_trust_radius": penalty_at_radius,
        "initial_balanced_bce": initial,
        "minimum_candidate_over_frozen_l2": 1.0 - radius_ratio,
        "maximum_candidate_over_frozen_l2": 1.0 + radius_ratio,
    }


def proximal_penalty_v1(
    *, candidate_weight, frozen_weight, proximal_lambda: float
) -> float:
    candidate = _weight_vector(candidate_weight, name="candidate head weight")
    frozen = _weight_vector(frozen_weight, name="frozen head weight")
    coefficient = float(proximal_lambda)
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        _fail("proximal lambda must be finite and positive")
    value = 0.5 * coefficient * sum(
        (after - before) ** 2 for after, before in zip(candidate, frozen)
    )
    if not math.isfinite(value) or value < 0.0:
        _fail("proximal penalty became invalid")
    return value


def bounded_class_balanced_objective_v1(
    *,
    group_logits: Mapping[str, object],
    group_targets: Mapping[str, object],
    candidate_weight,
    frozen_weight,
    initial_balanced_bce: float,
):
    """Evaluate the preregistered objective without taking a gradient step."""
    balanced, group_losses = balanced_four_group_bce_v1(
        group_logits=group_logits, group_targets=group_targets
    )
    proximal = derive_proximal_contract_v1(
        frozen_weight=frozen_weight,
        initial_balanced_bce=initial_balanced_bce,
    )
    penalty = proximal_penalty_v1(
        candidate_weight=candidate_weight,
        frozen_weight=frozen_weight,
        proximal_lambda=proximal["proximal_lambda"],
    )
    total = balanced + penalty
    if not math.isfinite(total):
        _fail("bounded class-balanced objective became non-finite")
    return total, balanced, penalty, group_losses, proximal


def geometry_evidence_v1(*, frozen_weight, candidate_weight) -> dict[str, object]:
    frozen = _weight_vector(frozen_weight, name="frozen head weight")
    candidate = _weight_vector(candidate_weight, name="candidate head weight")
    frozen_norm = math.sqrt(sum(value * value for value in frozen))
    candidate_norm = math.sqrt(sum(value * value for value in candidate))
    delta_norm = math.sqrt(
        sum((after - before) ** 2 for after, before in zip(candidate, frozen))
    )
    if frozen_norm <= 0.0 or candidate_norm <= 0.0:
        _fail("head geometry requires positive frozen/candidate norms")
    cosine = sum(before * after for before, after in zip(frozen, candidate)) / (
        frozen_norm * candidate_norm
    )
    cosine = max(-1.0, min(1.0, cosine))
    angle = math.degrees(math.acos(cosine))
    radius_ratio = math.sin(math.radians(MAX_HEAD_ANGLE_DEGREES))
    radius = radius_ratio * frozen_norm
    within_radius = delta_norm <= radius + GEOMETRY_TOLERANCE
    within_angle = angle <= MAX_HEAD_ANGLE_DEGREES + GEOMETRY_TOLERANCE
    return {
        "frozen_head_weight_l2": frozen_norm,
        "candidate_head_weight_l2": candidate_norm,
        "candidate_over_frozen_l2": candidate_norm / frozen_norm,
        "delta_weight_l2": delta_norm,
        "delta_over_frozen_l2": delta_norm / frozen_norm,
        "frozen_candidate_cosine": cosine,
        "head_angle_change_degrees": angle,
        "trust_radius_l2": radius,
        "trust_radius_over_frozen_l2": radius_ratio,
        "within_trust_radius": within_radius,
        "within_angle_limit": within_angle,
        "gate": "PASS" if within_radius and within_angle else "HOLD",
    }


def verify_final_candidate_v1(
    *,
    frozen_weight,
    candidate_weight,
    initial_total_objective: float,
    final_total_objective: float,
) -> dict[str, object]:
    initial = float(initial_total_objective)
    final = float(final_total_objective)
    if not math.isfinite(initial) or not math.isfinite(final):
        _fail("initial/final V5-2S objective must be finite")
    if final > initial + OBJECTIVE_TOLERANCE:
        _fail("final V5-2S objective increased")
    geometry = geometry_evidence_v1(
        frozen_weight=frozen_weight, candidate_weight=candidate_weight
    )
    if geometry["gate"] != "PASS":
        _fail("final V5-2S head escaped the preregistered geometry bound")
    return {
        "finite_non_increasing_objective": True,
        "initial_total_objective": initial,
        "final_total_objective": final,
        "geometry": geometry,
        "gate": "PASS",
    }


def training_entry_point_available() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False
