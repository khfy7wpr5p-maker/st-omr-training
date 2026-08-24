"""Training-disabled Meter V5-3D gated rescue architecture contract.

V5-3C verified that robust shared linear heads exist on the exact TRAIN
surfaces, but those witnesses rotate the mature heads by about 87 degrees and
require changes 20x to 26x larger than the frozen head norms. V5-3D therefore
keeps every frozen specialist tensor authoritative and defines a small,
separate nonlinear rescue candidate that can only add a positive decision when
the corresponding frozen specialist is negative.

This module is declarative. It contains no model implementation, optimizer,
autograd, training entry point, checkpoint write, validation access, or runtime
promotion path.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Final, Mapping


SCHEMA: Final[str] = "st-omr-meter-v5-3d-gated-rescue-architecture-contract-v1"
V53C_IMPLEMENTATION_HEAD: Final[str] = "61361612abfce132994abaca742c855f91305b44"
V53C_HARNESS_HEAD: Final[str] = "74bbdba45b08ee4ca350b487627b792cc5255806"
V53C_REPORT_SHA256: Final[str] = (
    "630b202f5369f12ea2562c81799613b81a295d77da08f9b3dd94a8fb1d801389"
)
V53C_ENVELOPE_SHA256: Final[str] = (
    "d682a31809702373df312b828a8452440517e480c55ad69fdf9633f35e5f436d"
)

RESCUE_SPECIALISTS: Final[tuple[str, ...]] = ("2", "3")
FROZEN_THRESHOLDS: Final[dict[str, float]] = {"2": 0.48, "3": 0.60, "4": 0.47}
FEATURE_DIM: Final[int] = 64
HIDDEN_WIDTH: Final[int] = 8
RESCUE_THRESHOLD: Final[float] = 0.50
ACTIVATION: Final[str] = "tanh"
PARAMETERS_PER_RESCUE: Final[int] = (
    FEATURE_DIM * HIDDEN_WIDTH + HIDDEN_WIDTH + HIDDEN_WIDTH + 1
)
TOTAL_RESCUE_PARAMETERS: Final[int] = len(RESCUE_SPECIALISTS) * PARAMETERS_PER_RESCUE

TRAIN_GROUPS: Final[tuple[str, ...]] = (
    "v5_frozen_false_negative_positive",
    "v5_frozen_true_negative",
    "historical_frozen_false_negative_positive",
    "historical_frozen_true_negative",
)
TRAIN_GROUP_WEIGHT: Final[float] = 0.25
EXPECTED_TRAIN_GROUP_COUNTS: Final[dict[str, dict[str, int]]] = {
    "2": {
        "v5_frozen_false_negative_positive": 90,
        "v5_frozen_true_negative": 450,
        "historical_frozen_false_negative_positive": 14,
        "historical_frozen_true_negative": 25254,
    },
    "3": {
        "v5_frozen_false_negative_positive": 90,
        "v5_frozen_true_negative": 450,
        "historical_frozen_false_negative_positive": 12,
        "historical_frozen_true_negative": 25364,
    },
}

V53C_GEOMETRY_EVIDENCE: Final[dict[str, dict[str, float]]] = {
    "2": {
        "delta_over_frozen_l2": 20.2917658975717,
        "frozen_candidate_cosine": 0.05072397191256342,
        "head_angle_change_degrees": 87.0924827769758,
    },
    "3": {
        "delta_over_frozen_l2": 25.71342751662423,
        "frozen_candidate_cosine": 0.0417679887967989,
        "head_angle_change_degrees": 87.60617414808654,
    },
}


class MeterV5_3DContractError(RuntimeError):
    """Raised when the frozen V5-3D architecture boundary is violated."""


def _fail(message: str) -> None:
    raise MeterV5_3DContractError(message)


def prerequisite_evidence_contract() -> dict[str, object]:
    return {
        "v5_3c_implementation_head": V53C_IMPLEMENTATION_HEAD,
        "v5_3c_harness_head": V53C_HARNESS_HEAD,
        "v5_3c_report_sha256": V53C_REPORT_SHA256,
        "v5_3c_execution_envelope_sha256": V53C_ENVELOPE_SHA256,
        "diagnostic_witness_gate": "PASS",
        "geometry": {digit: dict(values) for digit, values in V53C_GEOMETRY_EVIDENCE.items()},
        "shared_linear_head_feasible_on_train": True,
        "shared_linear_head_selected_for_repair": False,
        "shared_linear_head_lane_closed_by_safety_policy": True,
        "shared_linear_infeasibility_claimed": False,
        "generalization_proven": False,
    }


def topology_contract() -> dict[str, object]:
    return {
        "specialists": RESCUE_SPECIALISTS,
        "independent_per_specialist": True,
        "input": "exact-frozen-64d-backbone-feature",
        "feature_dim": FEATURE_DIM,
        "hidden_width": HIDDEN_WIDTH,
        "activation": ACTIVATION,
        "output": "single-rescue-logit",
        "topology": "Linear(64,8)->tanh->Linear(8,1)->sigmoid",
        "parameters_per_rescue": PARAMETERS_PER_RESCUE,
        "total_trainable_parameters_if_later_authorized": TOTAL_RESCUE_PARAMETERS,
        "separate_state_namespace": True,
        "frozen_checkpoint_replacement": False,
        "architecture_sweep": False,
        "hidden_width_sweep": False,
        "activation_sweep": False,
    }


def decision_contract() -> dict[str, object]:
    return {
        "frozen_thresholds": dict(FROZEN_THRESHOLDS),
        "rescue_threshold": RESCUE_THRESHOLD,
        "rescue_threshold_tuned": False,
        "rescue_eligibility": "same-specialist-frozen-decision-is-negative",
        "frozen_positive_can_be_demoted": False,
        "missing_or_unverified_rescue": "use-frozen-decision",
        "digit4_rescue_allowed": False,
        "multiple_digit_hits": "preserve-existing-ambiguous-result",
        "specialist_priority_added": False,
        "current_runtime_authority": "frozen-specialists-only",
        "rescue_runtime_enabled": False,
    }


def train_surface_contract() -> dict[str, object]:
    return {
        "future_execution_authorized": False,
        "eligible_rows": "frozen-negative-only",
        "groups": TRAIN_GROUPS,
        "group_weights": {name: TRAIN_GROUP_WEIGHT for name in TRAIN_GROUPS},
        "group_weight_sum": sum(TRAIN_GROUP_WEIGHT for _ in TRAIN_GROUPS),
        "expected_group_counts": {
            digit: dict(counts) for digit, counts in EXPECTED_TRAIN_GROUP_COUNTS.items()
        },
        "data_surfaces": ("v5_train", "historical_train"),
        "historical_validation_used": False,
        "first30_used": False,
        "v5_reserve_used": False,
        "v5_validation_used": False,
        "final_holdout_used": False,
        "optimizer_selected": False,
        "automatic_second_configuration": False,
        "hyperparameter_sweep": False,
    }


def safety_boundary() -> dict[str, object]:
    return {
        "architecture_contract_only": True,
        "training_authorized": False,
        "training_executed": False,
        "autograd_grad_used": False,
        "backward": False,
        "optimizer_steps": 0,
        "model_parameter_mutation": False,
        "candidate_checkpoint_write": False,
        "rescue_artifact_write": False,
        "frozen_backbone": True,
        "frozen_head_weight": True,
        "frozen_head_bias": True,
        "threshold_tuning": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "historical_retention_executed": False,
        "first30_opened": False,
        "v5_reserve_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "production_promotion": False,
    }


def future_gate_order() -> tuple[str, ...]:
    return (
        "separate_fixed_training_recipe_and_exact_ci_green_sha",
        "single_candidate_numerical_and_state_isolation",
        "train_v5_f1_and_frozen_correct_retention",
        "historical_validation_retention_at_frozen_thresholds",
        "immutable_v5_first30_diagnostic",
        "separately_authorized_v5_validation",
        "separately_authorized_final_holdout",
    )


def _probability(value: object, *, name: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise MeterV5_3DContractError(f"{name} must be numeric") from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        _fail(f"{name} must be finite and inside [0,1]")
    return probability


def shadow_candidate_decision_v1(
    *,
    digit: str,
    frozen_probability: object,
    rescue_probability: object | None,
    rescue_artifact_verified: bool,
) -> dict[str, object]:
    """Evaluate the preregistered shadow rule without changing production authority."""
    if digit not in RESCUE_SPECIALISTS:
        _fail("rescue shadow decisions are defined only for 2-AI and 3-AI")
    if type(rescue_artifact_verified) is not bool:
        _fail("rescue artifact verification flag must be bool")
    frozen = _probability(frozen_probability, name="frozen probability")
    frozen_decision = frozen >= FROZEN_THRESHOLDS[digit]
    eligible = not frozen_decision
    rescue_evaluated = eligible and rescue_artifact_verified
    rescue_decision = False
    rescue: float | None = None
    if rescue_evaluated:
        if rescue_probability is None:
            _fail("verified eligible rescue requires a rescue probability")
        rescue = _probability(rescue_probability, name="rescue probability")
        rescue_decision = rescue >= RESCUE_THRESHOLD
    candidate_decision = frozen_decision or rescue_decision
    return {
        "digit": digit,
        "frozen_probability": frozen,
        "frozen_threshold": FROZEN_THRESHOLDS[digit],
        "frozen_decision": frozen_decision,
        "rescue_eligible": eligible,
        "rescue_artifact_verified": rescue_artifact_verified,
        "rescue_evaluated": rescue_evaluated,
        "rescue_probability": rescue,
        "rescue_threshold": RESCUE_THRESHOLD,
        "rescue_decision": rescue_decision,
        "shadow_candidate_decision": candidate_decision,
        "production_decision": frozen_decision,
        "production_authority_changed": False,
    }


def frozen_production_decision_v1(*, digit: str, frozen_probability: object) -> bool:
    """Return the only production-authorized decision in this contract stage."""
    if digit not in FROZEN_THRESHOLDS:
        _fail("unknown frozen Meter specialist")
    frozen = _probability(frozen_probability, name="frozen probability")
    return frozen >= FROZEN_THRESHOLDS[digit]


def verify_exact_v5_3c_evidence(
    *, report_path: str | Path, envelope_path: str | Path
) -> dict[str, str]:
    report = Path(report_path)
    envelope = Path(envelope_path)
    if not report.is_file() or not envelope.is_file():
        _fail("exact V5-3C report/envelope missing")
    report_sha = hashlib.sha256(report.read_bytes()).hexdigest()
    envelope_sha = hashlib.sha256(envelope.read_bytes()).hexdigest()
    if report_sha != V53C_REPORT_SHA256:
        _fail("V5-3C report SHA256 mismatch")
    if envelope_sha != V53C_ENVELOPE_SHA256:
        _fail("V5-3C execution envelope SHA256 mismatch")
    return {"report_sha256": report_sha, "envelope_sha256": envelope_sha}


def validate_architecture_contract_v1(
    *, observed_group_counts: Mapping[str, Mapping[str, int]] | None = None
) -> dict[str, object]:
    if PARAMETERS_PER_RESCUE != 529 or TOTAL_RESCUE_PARAMETERS != 1058:
        _fail("fixed rescue parameter count changed")
    if FEATURE_DIM != 64 or HIDDEN_WIDTH != 8 or ACTIVATION != "tanh":
        _fail("fixed rescue topology changed")
    if RESCUE_THRESHOLD != 0.50:
        _fail("fixed rescue threshold changed")
    if set(RESCUE_SPECIALISTS) != {"2", "3"} or "4" in RESCUE_SPECIALISTS:
        _fail("rescue specialist boundary changed")
    if not math.isclose(
        sum(TRAIN_GROUP_WEIGHT for _ in TRAIN_GROUPS), 1.0, rel_tol=0.0, abs_tol=1e-15
    ):
        _fail("TRAIN group weights no longer sum to one")
    if observed_group_counts is not None:
        normalized = {
            str(digit): {str(name): int(value) for name, value in counts.items()}
            for digit, counts in observed_group_counts.items()
        }
        if normalized != EXPECTED_TRAIN_GROUP_COUNTS:
            _fail("frozen-negative TRAIN group counts changed")
    return {
        "schema": SCHEMA,
        "gate": "PASS",
        "parameters_per_rescue": PARAMETERS_PER_RESCUE,
        "total_rescue_parameters": TOTAL_RESCUE_PARAMETERS,
        "training_authorized": False,
        "production_promotion": False,
    }


def training_entry_point_available() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False
