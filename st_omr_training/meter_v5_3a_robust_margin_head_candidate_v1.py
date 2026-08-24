"""Single fixed robust-margin candidate fit after V5-2Z.

V5-2Z proved that the hard TRAIN decision constraints do not require the
enormous per-coordinate changes seen in V5-2Y, but its min-Linf witness spread
change across the head, rotated it by about 88 degrees, and left active rows at
the 1e-4 diagnostic margin.  V5-3A therefore fits one deployment candidate per
specialist with a fixed robust margin and a lexicographic parameter objective:
first minimize total absolute head-weight change (L1), then, without degrading
that primary optimum, minimize the largest absolute component change (Linf).

The fit is TRAIN-only.  Bias, thresholds, backbone, 4-AI, crop geometry, and
all validation surfaces remain frozen/closed.  A candidate checkpoint is only
written after independent float64 and float32-copy verification.  Historical
retention remains a separate mandatory gate.
"""
from __future__ import annotations

import hashlib
import math
from importlib import metadata
from pathlib import Path
from typing import Callable, Final, Mapping

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2n_frozen_feature_transfer_audit_v1 as v52n
from . import meter_v5_2p_fixed_bias_head_repair_v1 as v52p
from . import meter_v5_2r_train_class_margin_gradient_audit_v1 as v52r
from . import meter_v5_2t_bounded_class_balanced_head_repair_v1 as v52t
from . import meter_v5_2x_minimum_functional_logit_drift_audit_v1 as v52x
from . import meter_v5_2y_lexicographic_parameter_stability_audit_v1 as v52y
from . import meter_v5_2z_minimum_parameter_change_audit_v1 as v52z


SCHEMA: Final[str] = "st-omr-meter-v5-3a-robust-margin-head-candidate-v1"
REPORT_NAME: Final[str] = "v5_3a_robust_margin_head_candidate_report_v1.json"
CANDIDATE_DIR_NAME: Final[str] = "v5_3a_robust_margin_head_candidates"
TEMP_CANDIDATE_DIR_NAME: Final[str] = ".v5_3a_robust_margin_head_candidates.tmp"
APPROVAL_TOKEN: Final[str] = "V5_3A_SINGLE_ROBUST_MARGIN_HEAD_FIT_APPROVED"
V52Z_IMPLEMENTATION_HEAD: Final[str] = (
    "040e1d80fcbb09f6cac7b43e15fd34567c3f7dad"
)
V52Z_REPORT_SHA256: Final[str] = (
    "39fd82009f1bbef66877d0e65ad9719f7ecff9adc67f2c6d1a6a6e1a163ab8e4"
)
V52Z_EXECUTION_ENVELOPE_SHA256: Final[str] = (
    "fa3adc4f96fcf1d3109b43750b0958a3267fa57075a9c6ff061ad30b42864e12"
)
EXPECTED_FEATURE_DIM: Final[int] = 64
EXPECTED_V5_COUNT: Final[int] = 540
EXPECTED_HISTORICAL_COUNT: Final[int] = 26_964
ROBUST_DECISION_MARGIN: Final[float] = 0.25
SOLVER_MARGIN_BUFFER: Final[float] = 1e-4
MINIMUM_HISTORICAL_MARGIN: Final[float] = 1e-4
PRIMARY_L1_ABSOLUTE_SLACK: Final[float] = 1e-6
WITNESS_TOLERANCE: Final[float] = 1e-7
IDENTITY_TOLERANCE: Final[float] = 1e-9
FLOAT32_MARGIN_TOLERANCE: Final[float] = 2e-5
SOLVER_METHOD: Final[str] = "highs-ds"
EXPECTED_SCIPY_VERSION: Final[str] = "1.18.0"
SOLVER_PRIMAL_FEASIBILITY_TOLERANCE: Final[float] = 1e-9
SOLVER_DUAL_FEASIBILITY_TOLERANCE: Final[float] = 1e-9
ProgressCallback = Callable[[int, int, str], None]


class MeterV5_3AError(RuntimeError):
    """Raised when V5-3A departs from its preregistered contract."""


def _fail(message: str) -> None:
    raise MeterV5_3AError(message)


def safety_boundary() -> dict[str, object]:
    return {
        "single_fixed_candidate_fit": True,
        "linear_head_candidate_fit_authorized": True,
        "candidate_checkpoint_write_authorized": True,
        "candidate_parameter_surface": "head.weight-only-64",
        "frozen_backbone": True,
        "frozen_head_bias": True,
        "autograd_grad_used": False,
        "backward": False,
        "optimizer_steps": 0,
        "automatic_second_configuration": False,
        "hyperparameter_sweep": False,
        "runtime_threshold_tuning": False,
        "alternative_threshold_evaluated": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "reserve_v5_train_opened": False,
        "historical_validation_opened": False,
        "historical_retention_executed_by_this_module": False,
        "first30_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "production_promotion": False,
    }


def solver_contract() -> dict[str, object]:
    actual_version = metadata.version("scipy")
    return {
        "library": "scipy.optimize.linprog",
        "expected_library_version": EXPECTED_SCIPY_VERSION,
        "actual_library_version": actual_version,
        "library_version_matches_expected": actual_version == EXPECTED_SCIPY_VERSION,
        "method": SOLVER_METHOD,
        "presolve": False,
        "primal_feasibility_tolerance": SOLVER_PRIMAL_FEASIBILITY_TOLERANCE,
        "dual_feasibility_tolerance": SOLVER_DUAL_FEASIBILITY_TOLERANCE,
        "robust_decision_margin": ROBUST_DECISION_MARGIN,
        "solver_margin_buffer": SOLVER_MARGIN_BUFFER,
        "minimum_historical_margin": MINIMUM_HISTORICAL_MARGIN,
        "primary_l1_absolute_slack": PRIMARY_L1_ABSOLUTE_SLACK,
        "witness_tolerance": WITNESS_TOLERANCE,
        "float32_margin_tolerance": FLOAT32_MARGIN_TOLERANCE,
        "primary_objective": "minimum_total_absolute_delta_weight_l1",
        "secondary_objective": "minimum_max_absolute_delta_weight_linf",
        "primary_objective_fixed_before_secondary": True,
        "historical_margin_policy": (
            "at-least-max(1e-4,min(frozen-signed-margin,0.25))"
        ),
        "v5_margin_policy": "at-least-0.25-logit-after-float32-copy",
        "weight_l2_minimized": False,
        "historical_logit_drift_optimized": False,
        "automatic_second_configuration": False,
        "solver_sweep": False,
        "threshold_search": False,
        "bias_search": False,
        "optimal_status_wording": "SOLVER_REPORTED_OPTIMAL_NOT_FORMAL_PROOF",
        "infeasible_status_wording": (
            "SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF"
        ),
    }


def _numpy_modules():
    try:
        import numpy as np
        from scipy.optimize import linprog
    except Exception as exc:  # pragma: no cover - runtime guard
        raise MeterV5_3AError("NumPy/SciPy LP runtime is unavailable") from exc
    return np, linprog


def _to_numpy(value, *, name: str):
    return v52z._to_numpy(value, name=name)


def _as_surface(features, targets, *, name: str):
    return v52z._as_surface(features, targets, name=name)


def _linprog_options() -> dict[str, object]:
    return {
        "presolve": False,
        "primal_feasibility_tolerance": SOLVER_PRIMAL_FEASIBILITY_TOLERANCE,
        "dual_feasibility_tolerance": SOLVER_DUAL_FEASIBILITY_TOLERANCE,
    }


def _solver_stage_claim(result, *, stage: str) -> str:
    status = int(result.status)
    if status == 0 and bool(result.success):
        return f"{stage}_WITNESS_PENDING_VERIFICATION"
    if status == 2:
        return f"{stage}_SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF"
    return f"{stage}_UNPROVEN_SOLVER_DID_NOT_RETURN_A_USABLE_STATUS"


def _decision_surface_v1(
    *, historical_x, historical_y, v5_x, v5_y, frozen_weight, frozen_bias, boundary
):
    np, _linprog = _numpy_modules()
    frozen_hist_logit = historical_x @ frozen_weight + frozen_bias
    frozen_v5_logit = v5_x @ frozen_weight + frozen_bias
    frozen_hist_prediction = frozen_hist_logit >= boundary
    frozen_v5_prediction = frozen_v5_logit >= boundary
    hist_correct = frozen_hist_prediction == (historical_y == 1.0)
    if not bool(hist_correct.any()):
        _fail("frozen specialist has no correct historical decisions")
    retained_x = historical_x[hist_correct]
    retained_y = historical_y[hist_correct]
    retained_sign = 2.0 * retained_y - 1.0
    v5_sign = 2.0 * v5_y - 1.0
    retained_frozen_margin = retained_sign * (
        frozen_hist_logit[hist_correct] - boundary
    )
    retained_required_margin = np.maximum(
        MINIMUM_HISTORICAL_MARGIN,
        np.minimum(retained_frozen_margin, ROBUST_DECISION_MARGIN),
    )
    retained_solver_margin = retained_required_margin + SOLVER_MARGIN_BUFFER
    v5_required_margin = np.full(
        v5_x.shape[0], ROBUST_DECISION_MARGIN, dtype=np.float64
    )
    v5_solver_margin = v5_required_margin + SOLVER_MARGIN_BUFFER

    v5_frozen_margin = v5_sign * (frozen_v5_logit - boundary)
    v5_rows = -(v5_sign[:, None] * v5_x)
    v5_bounds = v5_frozen_margin - v5_solver_margin
    retained_rows = -(retained_sign[:, None] * retained_x)
    retained_bounds = retained_frozen_margin - retained_solver_margin
    return {
        "decision_rows": np.concatenate((v5_rows, retained_rows), axis=0),
        "decision_bounds": np.concatenate((v5_bounds, retained_bounds), axis=0),
        "frozen_hist_logit": frozen_hist_logit,
        "frozen_v5_logit": frozen_v5_logit,
        "frozen_hist_prediction": frozen_hist_prediction,
        "frozen_v5_prediction": frozen_v5_prediction,
        "hist_correct": hist_correct,
        "retained_required_margin": retained_required_margin,
        "retained_solver_margin": retained_solver_margin,
        "v5_required_margin": v5_required_margin,
        "v5_solver_margin": v5_solver_margin,
        "retained_sign": retained_sign,
        "v5_sign": v5_sign,
    }


def solve_robust_margin_minimum_total_change_v1(
    *,
    historical_features,
    historical_targets,
    v5_features,
    v5_targets,
    frozen_weight,
    frozen_bias: float,
    threshold: float,
) -> tuple[dict[str, object], object | None]:
    """Run the fixed primary L1 and secondary Linf LPs."""
    np, linprog = _numpy_modules()
    hist_x, hist_y = _as_surface(
        historical_features, historical_targets, name="historical"
    )
    v5_x, v5_y = _as_surface(v5_features, v5_targets, name="v5")
    w0 = _to_numpy(frozen_weight, name="frozen weight").reshape(-1)
    if w0.shape != (EXPECTED_FEATURE_DIM,):
        _fail(f"frozen weight shape changed: {w0.shape}")
    bias = float(frozen_bias)
    if not math.isfinite(bias):
        _fail("frozen bias is non-finite")
    boundary = v52r._threshold_logit(float(threshold))
    surface = _decision_surface_v1(
        historical_x=hist_x,
        historical_y=hist_y,
        v5_x=v5_x,
        v5_y=v5_y,
        frozen_weight=w0,
        frozen_bias=bias,
        boundary=boundary,
    )
    decision_rows = surface["decision_rows"]
    decision_bounds = surface["decision_bounds"]
    dimension = EXPECTED_FEATURE_DIM

    # Primary variables: delta[64], absolute-value auxiliaries u[64].
    primary_variable_count = 2 * dimension
    primary_objective = np.concatenate(
        (np.zeros(dimension, dtype=np.float64), np.ones(dimension, dtype=np.float64))
    )
    primary_decision_rows = np.zeros(
        (decision_rows.shape[0], primary_variable_count), dtype=np.float64
    )
    primary_decision_rows[:, :dimension] = decision_rows
    upper_abs_rows = np.zeros((dimension, primary_variable_count), dtype=np.float64)
    upper_abs_rows[:, :dimension] = np.eye(dimension, dtype=np.float64)
    upper_abs_rows[:, dimension:] = -np.eye(dimension, dtype=np.float64)
    lower_abs_rows = np.zeros((dimension, primary_variable_count), dtype=np.float64)
    lower_abs_rows[:, :dimension] = -np.eye(dimension, dtype=np.float64)
    lower_abs_rows[:, dimension:] = -np.eye(dimension, dtype=np.float64)
    primary_a = np.concatenate(
        (primary_decision_rows, upper_abs_rows, lower_abs_rows), axis=0
    )
    primary_b = np.concatenate(
        (decision_bounds, np.zeros(2 * dimension, dtype=np.float64))
    )
    primary = linprog(
        primary_objective,
        A_ub=primary_a,
        b_ub=primary_b,
        bounds=[(None, None)] * dimension + [(0.0, None)] * dimension,
        method=SOLVER_METHOD,
        options=_linprog_options(),
    )
    primary_claim = _solver_stage_claim(primary, stage="PRIMARY")
    report: dict[str, object] = {
        "candidate_claim": primary_claim,
        "robust_candidate_witness_verified": False,
        "primary_status": int(primary.status),
        "primary_success": bool(primary.success),
        "primary_iterations": int(getattr(primary, "nit", 0)),
        "primary_message": str(primary.message),
        "primary_optimality_claim": (
            "SOLVER_REPORTED_OPTIMAL_NOT_FORMAL_PROOF"
            if primary_claim == "PRIMARY_WITNESS_PENDING_VERIFICATION"
            else "OPTIMALITY_NOT_PROVEN"
        ),
        "secondary_optimality_claim": "OPTIMALITY_NOT_PROVEN",
        "v5_constraint_count": int(v5_x.shape[0]),
        "historical_all_count": int(hist_x.shape[0]),
        "historical_frozen_correct_count": int(np.sum(surface["hist_correct"])),
        "historical_frozen_wrong_count": int(np.sum(~surface["hist_correct"])),
        "fixed_bias": bias,
        "frozen_threshold": float(threshold),
        "frozen_threshold_logit": boundary,
        "robust_decision_margin": ROBUST_DECISION_MARGIN,
        "solver_margin_buffer": SOLVER_MARGIN_BUFFER,
        "candidate_weight_values_emitted": False,
        "candidate_checkpoint_written": False,
        "repair_candidate_selected": False,
    }
    if primary_claim != "PRIMARY_WITNESS_PENDING_VERIFICATION":
        return report, None

    primary_solution = np.asarray(primary.x, dtype=np.float64)
    if not bool(np.isfinite(primary_solution).all()):
        _fail("primary LP witness is non-finite")
    primary_delta = primary_solution[:dimension]
    primary_l1_solver = float(primary.fun)
    primary_l1_recomputed = float(np.sum(np.abs(primary_delta)))
    primary_l1_error = abs(primary_l1_solver - primary_l1_recomputed)
    primary_decision_violations = int(
        np.sum(
            decision_rows @ primary_delta
            > decision_bounds + WITNESS_TOLERANCE
        )
    )
    primary_auxiliary_violations = int(
        np.sum(
            np.abs(primary_delta)
            > primary_solution[dimension:] + WITNESS_TOLERANCE
        )
    )
    report.update(
        {
            "primary_decision_constraint_violations": (
                primary_decision_violations
            ),
            "primary_auxiliary_bound_violations": primary_auxiliary_violations,
        }
    )
    if (
        primary_l1_error > WITNESS_TOLERANCE
        or primary_decision_violations != 0
        or primary_auxiliary_violations != 0
    ):
        report["candidate_claim"] = "PRIMARY_UNPROVEN_WITNESS_RESIDUAL_FAILED"
        return report, None
    primary_l1_cap = primary_l1_solver + PRIMARY_L1_ABSOLUTE_SLACK

    # Secondary variables: delta[64], u[64], max absolute component r.
    secondary_variable_count = 2 * dimension + 1
    secondary_objective = np.zeros(secondary_variable_count, dtype=np.float64)
    secondary_objective[-1] = 1.0
    secondary_decision_rows = np.zeros(
        (decision_rows.shape[0], secondary_variable_count), dtype=np.float64
    )
    secondary_decision_rows[:, :dimension] = decision_rows
    secondary_upper_abs = np.zeros(
        (dimension, secondary_variable_count), dtype=np.float64
    )
    secondary_upper_abs[:, :dimension] = np.eye(dimension, dtype=np.float64)
    secondary_upper_abs[:, dimension : 2 * dimension] = -np.eye(
        dimension, dtype=np.float64
    )
    secondary_lower_abs = np.zeros(
        (dimension, secondary_variable_count), dtype=np.float64
    )
    secondary_lower_abs[:, :dimension] = -np.eye(dimension, dtype=np.float64)
    secondary_lower_abs[:, dimension : 2 * dimension] = -np.eye(
        dimension, dtype=np.float64
    )
    l1_cap_row = np.zeros((1, secondary_variable_count), dtype=np.float64)
    l1_cap_row[0, dimension : 2 * dimension] = 1.0
    upper_linf_rows = np.zeros(
        (dimension, secondary_variable_count), dtype=np.float64
    )
    upper_linf_rows[:, :dimension] = np.eye(dimension, dtype=np.float64)
    upper_linf_rows[:, -1] = -1.0
    lower_linf_rows = np.zeros(
        (dimension, secondary_variable_count), dtype=np.float64
    )
    lower_linf_rows[:, :dimension] = -np.eye(dimension, dtype=np.float64)
    lower_linf_rows[:, -1] = -1.0
    secondary_a = np.concatenate(
        (
            secondary_decision_rows,
            secondary_upper_abs,
            secondary_lower_abs,
            l1_cap_row,
            upper_linf_rows,
            lower_linf_rows,
        ),
        axis=0,
    )
    secondary_b = np.concatenate(
        (
            decision_bounds,
            np.zeros(2 * dimension, dtype=np.float64),
            np.asarray([primary_l1_cap], dtype=np.float64),
            np.zeros(2 * dimension, dtype=np.float64),
        )
    )
    secondary = linprog(
        secondary_objective,
        A_ub=secondary_a,
        b_ub=secondary_b,
        bounds=(
            [(None, None)] * dimension
            + [(0.0, None)] * dimension
            + [(0.0, None)]
        ),
        method=SOLVER_METHOD,
        options=_linprog_options(),
    )
    secondary_claim = _solver_stage_claim(secondary, stage="SECONDARY")
    report.update(
        {
            "minimum_delta_weight_l1": primary_l1_solver,
            "independently_recomputed_primary_delta_weight_l1": (
                primary_l1_recomputed
            ),
            "primary_objective_recomputation_absolute_error": primary_l1_error,
            "primary_l1_cap": primary_l1_cap,
            "secondary_status": int(secondary.status),
            "secondary_success": bool(secondary.success),
            "secondary_iterations": int(getattr(secondary, "nit", 0)),
            "secondary_message": str(secondary.message),
            "secondary_optimality_claim": (
                "SOLVER_REPORTED_OPTIMAL_NOT_FORMAL_PROOF"
                if secondary_claim == "SECONDARY_WITNESS_PENDING_VERIFICATION"
                else "OPTIMALITY_NOT_PROVEN"
            ),
        }
    )
    if secondary_claim != "SECONDARY_WITNESS_PENDING_VERIFICATION":
        report["candidate_claim"] = secondary_claim
        return report, None

    solution = np.asarray(secondary.x, dtype=np.float64)
    if not bool(np.isfinite(solution).all()):
        _fail("secondary LP witness is non-finite")
    delta = solution[:dimension]
    candidate = w0 + delta
    solver_linf = float(secondary.fun)
    recomputed_l1 = float(np.sum(np.abs(delta)))
    recomputed_linf = float(np.max(np.abs(delta)))
    l1_cap_violations = int(recomputed_l1 > primary_l1_cap + WITNESS_TOLERANCE)
    primary_lower_bound_conflict = int(
        recomputed_l1 < primary_l1_solver - WITNESS_TOLERANCE
    )
    parameter_bound_violations = int(
        np.sum(np.abs(delta) > solver_linf + WITNESS_TOLERANCE)
    )

    diagnostic_hist_logit = hist_x @ candidate + bias
    diagnostic_v5_logit = v5_x @ candidate + bias
    direct_hist_delta = hist_x @ delta
    recomputed_hist_delta = diagnostic_hist_logit - surface["frozen_hist_logit"]
    identity_error = float(np.max(np.abs(direct_hist_delta - recomputed_hist_delta)))
    v5_margin = surface["v5_sign"] * (diagnostic_v5_logit - boundary)
    retained_margin = surface["retained_sign"] * (
        diagnostic_hist_logit[surface["hist_correct"]] - boundary
    )
    v5_violations = int(
        np.sum(v5_margin < surface["v5_required_margin"] - WITNESS_TOLERANCE)
    )
    v5_solver_margin_violations = int(
        np.sum(v5_margin < surface["v5_solver_margin"] - WITNESS_TOLERANCE)
    )
    historical_violations = int(
        np.sum(
            retained_margin
            < surface["retained_required_margin"] - WITNESS_TOLERANCE
        )
    )
    historical_solver_margin_violations = int(
        np.sum(
            retained_margin
            < surface["retained_solver_margin"] - WITNESS_TOLERANCE
        )
    )
    diagnostic_hist_prediction = diagnostic_hist_logit >= boundary
    diagnostic_v5_prediction = diagnostic_v5_logit >= boundary
    report.update(
        {
            "minimum_delta_weight_linf": solver_linf,
            "independently_recomputed_final_delta_weight_l1": recomputed_l1,
            "independently_recomputed_delta_weight_linf": recomputed_linf,
            "secondary_objective_recomputation_absolute_error": abs(
                solver_linf - recomputed_linf
            ),
            "primary_l1_cap_violations": l1_cap_violations,
            "primary_lower_bound_conflicts": primary_lower_bound_conflict,
            "parameter_bound_violations": parameter_bound_violations,
            "minimum_v5_signed_decision_margin": float(np.min(v5_margin)),
            "minimum_historical_retained_signed_decision_margin": float(
                np.min(retained_margin)
            ),
            "v5_constraint_violations": v5_violations,
            "v5_solver_margin_constraint_violations": (
                v5_solver_margin_violations
            ),
            "historical_margin_constraint_violations": historical_violations,
            "historical_solver_margin_constraint_violations": (
                historical_solver_margin_violations
            ),
            "functional_delta_identity_max_abs_error": identity_error,
            "functional_delta_identity_verified": identity_error <= IDENTITY_TOLERANCE,
            "historical_logit_drift": v52x._quantile_summary(
                direct_hist_delta, name="historical logit drift"
            ),
            "historical_absolute_logit_drift": v52x._quantile_summary(
                np.abs(direct_hist_delta), name="historical absolute logit drift"
            ),
            "weight_geometry": v52y._weight_geometry(
                frozen_weight=w0, diagnostic_weight=candidate
            ),
            "frozen_v5_train_metrics": v52x._classification_metrics(
                prediction=surface["frozen_v5_prediction"], targets=v5_y
            ),
            "diagnostic_v5_train_metrics": v52x._classification_metrics(
                prediction=diagnostic_v5_prediction, targets=v5_y
            ),
            "frozen_historical_train_metrics": v52x._classification_metrics(
                prediction=surface["frozen_hist_prediction"], targets=hist_y
            ),
            "diagnostic_historical_train_metrics": v52x._classification_metrics(
                prediction=diagnostic_hist_prediction, targets=hist_y
            ),
            "v5_transition_counts": v52x._transition_counts(
                frozen_prediction=surface["frozen_v5_prediction"],
                diagnostic_prediction=diagnostic_v5_prediction,
                targets=v5_y,
            ),
            "historical_transition_counts": v52x._transition_counts(
                frozen_prediction=surface["frozen_hist_prediction"],
                diagnostic_prediction=diagnostic_hist_prediction,
                targets=hist_y,
            ),
        }
    )
    verification_ok = (
        v5_violations == 0
        and v5_solver_margin_violations == 0
        and historical_violations == 0
        and historical_solver_margin_violations == 0
        and l1_cap_violations == 0
        and primary_lower_bound_conflict == 0
        and parameter_bound_violations == 0
        and identity_error <= IDENTITY_TOLERANCE
        and abs(solver_linf - recomputed_linf) <= WITNESS_TOLERANCE
        and report["historical_transition_counts"]["correct_to_wrong"] == 0
        and report["diagnostic_v5_train_metrics"]["f1"] == 1.0
    )
    if not verification_ok:
        report["candidate_claim"] = "SECONDARY_UNPROVEN_WITNESS_RESIDUAL_FAILED"
        return report, None
    report["candidate_claim"] = "CANDIDATE_WITNESS_VERIFIED"
    report["robust_candidate_witness_verified"] = True
    return report, candidate


def path_diagnosis_v1(result: Mapping[str, object]) -> dict[str, object]:
    verified = result.get("candidate_claim") == "CANDIDATE_WITNESS_VERIFIED"
    return {
        "status": (
            "ROBUST_MARGIN_MINIMUM_TOTAL_CHANGE_CANDIDATE_VERIFIED_ON_TRAIN"
            if verified
            else "ROBUST_MARGIN_HEAD_CANDIDATE_NOT_VERIFIED"
        ),
        "robust_margin_candidate_verified_on_train": verified,
        "repair_candidate_selected": verified,
        "historical_retention_required_next": verified,
        "historical_validation_preservation_proven": False,
        "generalization_proven": False,
        "production_promotion_authorized": False,
        "threshold_or_bias_change_authorized": False,
        "architecture_change_authorized": False,
    }


def _read_exact_v5_2z_evidence_v1(
    *, report_path: Path, envelope_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    if not report_path.is_file() or not envelope_path.is_file():
        _fail("exact V5-2Z report/envelope missing")
    report_bytes = report_path.read_bytes()
    envelope_bytes = envelope_path.read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != V52Z_REPORT_SHA256:
        _fail("V5-2Z report SHA256 mismatch")
    if hashlib.sha256(envelope_bytes).hexdigest() != V52Z_EXECUTION_ENVELOPE_SHA256:
        _fail("V5-2Z execution envelope SHA256 mismatch")
    report = v52b._read_json(report_path)
    envelope = v52b._read_json(envelope_path)
    if report.get("schema") != v52z.SCHEMA:
        _fail("V5-2Z report schema mismatch")
    for key, expected in (
        ("model_training", False),
        ("autograd_grad_used", False),
        ("backward", False),
        ("optimizer_steps", 0),
        ("diagnostic_minimum_parameter_witness_fit", True),
        ("diagnostic_witness_persisted", False),
        ("diagnostic_witness_values_emitted", False),
        ("candidate_checkpoint_write", False),
        ("model_parameter_mutation", False),
        ("threshold_tuning", False),
        ("bias_selection", False),
        ("historical_validation_opened", False),
        ("first30_opened", False),
        ("v5_validation_opened", False),
        ("final_holdout_locked", True),
        ("digit4_frozen", True),
        ("repair_selected", False),
    ):
        if report.get(key) != expected:
            _fail(f"V5-2Z safety boundary changed: {key}")
    specialists = report.get("per_specialist")
    if not isinstance(specialists, Mapping):
        _fail("V5-2Z specialist evidence missing")
    for digit in ("2", "3"):
        item = specialists.get(digit)
        if not isinstance(item, Mapping):
            _fail(f"V5-2Z {digit}-AI evidence missing")
        result = item.get("minimum_parameter_change")
        if not isinstance(result, Mapping):
            _fail(f"V5-2Z {digit}-AI result missing")
        if result.get("witness_claim") != "WITNESS_VERIFIED":
            _fail(f"V5-2Z {digit}-AI witness is not verified")
        if result.get("minimum_parameter_change_witness_verified") is not True:
            _fail(f"V5-2Z {digit}-AI witness verification missing")
        if result.get("historical_transition_counts", {}).get("correct_to_wrong") != 0:
            _fail(f"V5-2Z {digit}-AI historical decision evidence changed")
        if result.get("diagnostic_v5_train_metrics", {}).get("f1") != 1.0:
            _fail(f"V5-2Z {digit}-AI V5 TRAIN evidence changed")
        if result.get("witness_values_emitted") is not False:
            _fail(f"V5-2Z {digit}-AI emitted witness values")
        if result.get("witness_persisted") is not False:
            _fail(f"V5-2Z {digit}-AI persisted witness values")
    if envelope.get("expected_head") != V52Z_IMPLEMENTATION_HEAD:
        _fail("V5-2Z execution HEAD mismatch")
    if envelope.get("actual_head_before_run") != V52Z_IMPLEMENTATION_HEAD:
        _fail("V5-2Z pre-run HEAD mismatch")
    if envelope.get("actual_head_after_run") != V52Z_IMPLEMENTATION_HEAD:
        _fail("V5-2Z post-run HEAD mismatch")
    if envelope.get("audit_report_sha256") != V52Z_REPORT_SHA256:
        _fail("V5-2Z report binding mismatch")
    return report, envelope


def _verify_float32_copy_v1(
    *,
    candidate_weight,
    frozen_weight,
    frozen_bias: float,
    threshold: float,
    historical_features,
    historical_targets,
    v5_features,
    v5_targets,
) -> dict[str, object]:
    np, _linprog = _numpy_modules()
    hist_x, hist_y = _as_surface(
        historical_features, historical_targets, name="historical"
    )
    v5_x, v5_y = _as_surface(v5_features, v5_targets, name="v5")
    w0 = _to_numpy(frozen_weight, name="frozen weight").reshape(-1)
    candidate32 = np.asarray(candidate_weight, dtype=np.float32).astype(np.float64)
    boundary = v52r._threshold_logit(float(threshold))
    surface = _decision_surface_v1(
        historical_x=hist_x,
        historical_y=hist_y,
        v5_x=v5_x,
        v5_y=v5_y,
        frozen_weight=w0,
        frozen_bias=float(frozen_bias),
        boundary=boundary,
    )
    hist_logit = hist_x @ candidate32 + float(frozen_bias)
    v5_logit = v5_x @ candidate32 + float(frozen_bias)
    v5_margin = surface["v5_sign"] * (v5_logit - boundary)
    hist_margin = surface["retained_sign"] * (
        hist_logit[surface["hist_correct"]] - boundary
    )
    v5_violations = int(
        np.sum(v5_margin < ROBUST_DECISION_MARGIN - FLOAT32_MARGIN_TOLERANCE)
    )
    hist_violations = int(
        np.sum(
            hist_margin
            < surface["retained_required_margin"] - FLOAT32_MARGIN_TOLERANCE
        )
    )
    hist_prediction = hist_logit >= boundary
    v5_prediction = v5_logit >= boundary
    transitions = v52x._transition_counts(
        frozen_prediction=surface["frozen_hist_prediction"],
        diagnostic_prediction=hist_prediction,
        targets=hist_y,
    )
    metrics = v52x._classification_metrics(prediction=v5_prediction, targets=v5_y)
    passed = (
        v5_violations == 0
        and hist_violations == 0
        and transitions["correct_to_wrong"] == 0
        and metrics["f1"] == 1.0
    )
    return {
        "gate": "PASS" if passed else "HOLD",
        "v5_margin_violations": v5_violations,
        "historical_margin_violations": hist_violations,
        "minimum_v5_signed_decision_margin": float(np.min(v5_margin)),
        "minimum_historical_retained_signed_decision_margin": float(
            np.min(hist_margin)
        ),
        "historical_transition_counts": transitions,
        "v5_train_metrics": metrics,
        "float32_copy_back": True,
    }


def _verify_runtime_torch_copy_v1(
    *,
    model,
    frozen_state: Mapping[str, object],
    threshold: float,
    historical_features,
    historical_targets,
    v5_features,
    v5_targets,
) -> dict[str, object]:
    """Verify margins using the actual float32 tensor operations used at runtime."""
    torch, _nn = v52b._import_torch()
    x_hist = historical_features.detach().cpu().to(dtype=model.head.weight.dtype)
    y_hist = historical_targets.detach().cpu().reshape(-1).to(dtype=torch.bool)
    x_v5 = v5_features.detach().cpu().to(dtype=model.head.weight.dtype)
    y_v5 = v5_targets.detach().cpu().reshape(-1).to(dtype=torch.bool)
    frozen_weight = frozen_state["head.weight"].detach().cpu().reshape(-1)
    frozen_bias = frozen_state["head.bias"].detach().cpu().reshape(-1)[0]
    candidate_weight = model.head.weight.detach().cpu().reshape(-1)
    candidate_bias = model.head.bias.detach().cpu().reshape(-1)[0]
    boundary = torch.tensor(
        v52r._threshold_logit(float(threshold)), dtype=x_hist.dtype
    )
    with torch.no_grad():
        frozen_hist_logit = x_hist @ frozen_weight + frozen_bias
        candidate_hist_logit = x_hist @ candidate_weight + candidate_bias
        candidate_v5_logit = x_v5 @ candidate_weight + candidate_bias
    frozen_hist_prediction = frozen_hist_logit >= boundary
    candidate_hist_prediction = candidate_hist_logit >= boundary
    candidate_v5_prediction = candidate_v5_logit >= boundary
    hist_correct = frozen_hist_prediction == y_hist
    retained_sign = 2.0 * y_hist[hist_correct].to(dtype=x_hist.dtype) - 1.0
    v5_sign = 2.0 * y_v5.to(dtype=x_v5.dtype) - 1.0
    frozen_margin = retained_sign * (frozen_hist_logit[hist_correct] - boundary)
    required_hist_margin = torch.maximum(
        torch.full_like(frozen_margin, MINIMUM_HISTORICAL_MARGIN),
        torch.minimum(
            frozen_margin,
            torch.full_like(frozen_margin, ROBUST_DECISION_MARGIN),
        ),
    )
    candidate_hist_margin = retained_sign * (
        candidate_hist_logit[hist_correct] - boundary
    )
    candidate_v5_margin = v5_sign * (candidate_v5_logit - boundary)
    v5_violations = int(
        (
            candidate_v5_margin
            < ROBUST_DECISION_MARGIN - FLOAT32_MARGIN_TOLERANCE
        ).sum().item()
    )
    hist_violations = int(
        (
            candidate_hist_margin
            < required_hist_margin - FLOAT32_MARGIN_TOLERANCE
        ).sum().item()
    )
    correct_to_wrong = int(
        (hist_correct & (candidate_hist_prediction != y_hist)).sum().item()
    )
    v5_correct = int((candidate_v5_prediction == y_v5).sum().item())
    passed = (
        v5_violations == 0
        and hist_violations == 0
        and correct_to_wrong == 0
        and v5_correct == int(y_v5.numel())
    )
    return {
        "gate": "PASS" if passed else "HOLD",
        "v5_margin_violations": v5_violations,
        "historical_margin_violations": hist_violations,
        "historical_frozen_correct_to_wrong": correct_to_wrong,
        "v5_correct_count": v5_correct,
        "v5_count": int(y_v5.numel()),
        "minimum_v5_signed_decision_margin": float(
            torch.min(candidate_v5_margin).item()
        ),
        "minimum_historical_retained_signed_decision_margin": float(
            torch.min(candidate_hist_margin).item()
        ),
        "actual_runtime_float32_tensor_path": True,
    }


def _candidate_path(directory: Path, digit: str) -> Path:
    return directory / f"digit{digit}_v5_3a_robust_margin_candidate.pt"


def _save_candidate_v1(
    *, model, path: Path, digit: str, manifest_sha: str, evidence: Mapping[str, object]
) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    fingerprint = v52t._state_fingerprint_without_numpy_v1(model)
    source_sha = v52b.DIGIT2_SHA256 if digit == "2" else v52b.DIGIT3_SHA256
    payload = {
        "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "metadata": {
            "schema": SCHEMA,
            "role": f"digit-{digit}-v5-3a-robust-margin-candidate",
            "source_checkpoint_sha256": source_sha,
            "slot_manifest_sha256": manifest_sha,
            "candidate_parameter_surface": "head.weight-only-64",
            "head_bias_frozen": True,
            "backbone_frozen": True,
            "threshold": v52b.FROZEN_THRESHOLDS[digit],
            "threshold_tuned": False,
            "state_fingerprint": fingerprint,
            "solver_contract": solver_contract(),
            "fit_evidence": dict(evidence),
            "historical_retention_executed": False,
            "first30_opened": False,
            "v5_validation_opened": False,
            "final_holdout_locked": True,
        },
    }
    torch.save(payload, path)
    return {
        "candidate_path": str(path),
        "candidate_sha256": v52b._sha_file(path),
        "state_fingerprint": fingerprint,
    }


def _reload_candidate_v1(
    path: Path, *, digit: str, manifest_sha: str, frozen_state: Mapping[str, object]
) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise MeterV5_3AError(f"cannot reload V5-3A {digit}-AI candidate") from exc
    if not isinstance(payload, Mapping):
        _fail(f"{digit}-AI candidate payload is not a mapping")
    state = payload.get("model_state_dict")
    meta = payload.get("metadata")
    if not isinstance(state, Mapping) or not isinstance(meta, Mapping):
        _fail(f"{digit}-AI candidate state/metadata missing")
    expected_source = v52b.DIGIT2_SHA256 if digit == "2" else v52b.DIGIT3_SHA256
    expected = {
        "schema": SCHEMA,
        "role": f"digit-{digit}-v5-3a-robust-margin-candidate",
        "source_checkpoint_sha256": expected_source,
        "slot_manifest_sha256": manifest_sha,
        "candidate_parameter_surface": "head.weight-only-64",
        "head_bias_frozen": True,
        "backbone_frozen": True,
        "threshold": v52b.FROZEN_THRESHOLDS[digit],
        "threshold_tuned": False,
        "historical_retention_executed": False,
        "first30_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            _fail(f"{digit}-AI candidate metadata changed: {key}")
    model = v52b._build_digit_model().cpu()
    model.load_state_dict(dict(state), strict=True)
    fingerprint = v52t._state_fingerprint_without_numpy_v1(model)
    if meta.get("state_fingerprint") != fingerprint:
        _fail(f"{digit}-AI candidate fingerprint mismatch")
    try:
        invariants = v52p._verify_only_head_weight_changed(model, frozen_state)
    except v52p.MeterV5_2PError as exc:
        raise MeterV5_3AError(f"{digit}-AI reloaded state integrity failed") from exc
    if invariants.get("only_head_weight_changed") is not True:
        _fail(f"{digit}-AI reloaded candidate changed an illegal tensor")
    return {"reload_verified": True, "state_invariants": invariants}


def fit_robust_margin_head_candidates_v1(
    data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    v5_2z_report: str | Path,
    v5_2z_envelope: str | Path,
    confirmation: str,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Fit one fixed candidate per specialist and stop before retention."""
    if confirmation != APPROVAL_TOKEN:
        _fail("exact V5-3A approval token missing")
    root = Path(data_root)
    ann = root / v51.ANNOTATIONS_DIR
    report_path = ann / REPORT_NAME
    candidate_dir = ann / CANDIDATE_DIR_NAME
    temporary_dir = ann / TEMP_CANDIDATE_DIR_NAME
    if report_path.exists() or candidate_dir.exists() or temporary_dir.exists():
        _fail("refusing to overwrite/rerun V5-3A evidence")

    prior_report, _prior_envelope = _read_exact_v5_2z_evidence_v1(
        report_path=Path(v5_2z_report), envelope_path=Path(v5_2z_envelope)
    )
    models = v52n._frozen_models(
        digit2_frozen=Path(digit2_frozen), digit3_frozen=Path(digit3_frozen)
    )
    manifest_path, _rows, v5_features, v5_targets, _metrics = v52n._v5_surface(
        root, models
    )
    manifest_sha = v52b._sha_file(manifest_path)
    if prior_report.get("slot_manifest_sha256") != manifest_sha:
        _fail("V5 TRAIN slot manifest changed after V5-2Z")
    historical_features, historical_targets = v52n._historical_surface(
        m4a_root=Path(m4a_root),
        d10_root=Path(d10_root),
        models=models,
        progress=progress,
    )

    per_specialist: dict[str, dict[str, object]] = {}
    candidate_weights: dict[str, object] = {}
    frozen_states: dict[str, Mapping[str, object]] = {}
    all_verified = True
    for digit in ("2", "3"):
        model = models[digit]
        frozen_states[digit] = v52p._frozen_state_snapshot(model)
        frozen_weight = model.head.weight.detach().cpu().reshape(-1)
        frozen_bias = float(model.head.bias.detach().cpu().reshape(-1)[0].item())
        result, candidate = solve_robust_margin_minimum_total_change_v1(
            historical_features=historical_features[digit],
            historical_targets=historical_targets[digit],
            v5_features=v5_features[digit],
            v5_targets=v5_targets[digit],
            frozen_weight=frozen_weight,
            frozen_bias=frozen_bias,
            threshold=v52b.FROZEN_THRESHOLDS[digit],
        )
        prior_result = prior_report["per_specialist"][digit][
            "minimum_parameter_change"
        ]
        if result.get("v5_constraint_count") != EXPECTED_V5_COUNT:
            _fail(f"{digit}-AI V5 TRAIN count changed")
        if result.get("historical_all_count") != EXPECTED_HISTORICAL_COUNT:
            _fail(f"{digit}-AI historical TRAIN count changed")
        if result.get("historical_frozen_correct_count") != prior_result.get(
            "historical_frozen_correct_count"
        ):
            _fail(f"{digit}-AI frozen-correct historical surface changed")
        if result.get("fixed_bias") != prior_result.get("fixed_bias"):
            _fail(f"{digit}-AI frozen bias changed after V5-2Z")
        if result.get("frozen_threshold") != prior_result.get(
            "frozen_threshold"
        ):
            _fail(f"{digit}-AI frozen threshold changed after V5-2Z")
        diagnosis = path_diagnosis_v1(result)
        per_specialist[digit] = {
            "fit": result,
            "path_diagnosis": diagnosis,
            "threshold": v52b.FROZEN_THRESHOLDS[digit],
            "threshold_unchanged": True,
        }
        if candidate is None or result.get("candidate_claim") != (
            "CANDIDATE_WITNESS_VERIFIED"
        ):
            all_verified = False
            continue
        float32_gate = _verify_float32_copy_v1(
            candidate_weight=candidate,
            frozen_weight=frozen_weight,
            frozen_bias=frozen_bias,
            threshold=v52b.FROZEN_THRESHOLDS[digit],
            historical_features=historical_features[digit],
            historical_targets=historical_targets[digit],
            v5_features=v5_features[digit],
            v5_targets=v5_targets[digit],
        )
        per_specialist[digit]["float32_copy_gate"] = float32_gate
        if float32_gate["gate"] != "PASS":
            all_verified = False
            continue
        candidate_weights[digit] = candidate

    contract = solver_contract()
    if contract["library_version_matches_expected"] is not True:
        _fail("SciPy runtime version does not match the pinned solver contract")
    gate = "PASS" if all_verified and len(candidate_weights) == 2 else "HOLD"
    report: dict[str, object] = {
        "schema": SCHEMA,
        "approval_token_verified": True,
        "exact_v5_2z_binding": {
            "implementation_head": V52Z_IMPLEMENTATION_HEAD,
            "report_sha256": V52Z_REPORT_SHA256,
            "execution_envelope_sha256": V52Z_EXECUTION_ENVELOPE_SHA256,
        },
        "slot_manifest_sha256": manifest_sha,
        "v5_train_slot_count": EXPECTED_V5_COUNT,
        "historical_train_record_count": EXPECTED_HISTORICAL_COUNT,
        "feature_dim": EXPECTED_FEATURE_DIM,
        "solver_contract": contract,
        "per_specialist": per_specialist,
        "candidate_selection_gate": gate,
        "candidate_checkpoint_written": False,
        "linear_program_candidate_fit_executed": True,
        "gradient_based_model_training_executed": False,
        "model_parameter_mutation_executed": False,
        "historical_preservation_claimed": False,
        **safety_boundary(),
    }
    if gate != "PASS":
        v51._atomic_write_json(report_path, report)
        return report

    torch, _nn = v52b._import_torch()
    for digit in ("2", "3"):
        model = models[digit]
        candidate = candidate_weights[digit]
        with torch.no_grad():
            weight32 = torch.as_tensor(
                candidate, dtype=model.head.weight.dtype
            ).reshape_as(model.head.weight)
            model.head.weight.copy_(weight32)
        try:
            invariants = v52p._verify_only_head_weight_changed(
                model, frozen_states[digit]
            )
        except v52p.MeterV5_2PError as exc:
            raise MeterV5_3AError(f"{digit}-AI state integrity failed") from exc
        if invariants.get("only_head_weight_changed") is not True:
            _fail(f"{digit}-AI changed an illegal state tensor")
        per_specialist[digit]["state_invariants"] = invariants
        runtime_gate = _verify_runtime_torch_copy_v1(
            model=model,
            frozen_state=frozen_states[digit],
            threshold=v52b.FROZEN_THRESHOLDS[digit],
            historical_features=historical_features[digit],
            historical_targets=historical_targets[digit],
            v5_features=v5_features[digit],
            v5_targets=v5_targets[digit],
        )
        if runtime_gate["gate"] != "PASS":
            _fail(f"{digit}-AI actual float32 runtime margin gate HOLD")
        per_specialist[digit]["runtime_float32_gate"] = runtime_gate

    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        for digit in ("2", "3"):
            saved = _save_candidate_v1(
                model=models[digit],
                path=_candidate_path(temporary_dir, digit),
                digit=digit,
                manifest_sha=manifest_sha,
                evidence={
                    "fit": per_specialist[digit]["fit"],
                    "float32_copy_gate": per_specialist[digit]["float32_copy_gate"],
                    "runtime_float32_gate": per_specialist[digit][
                        "runtime_float32_gate"
                    ],
                    "state_invariants": per_specialist[digit]["state_invariants"],
                },
            )
            per_specialist[digit]["candidate"] = saved
            reload_evidence = _reload_candidate_v1(
                _candidate_path(temporary_dir, digit),
                digit=digit,
                manifest_sha=manifest_sha,
                frozen_state=frozen_states[digit],
            )
            per_specialist[digit]["candidate"].update(reload_evidence)
        temporary_dir.replace(candidate_dir)
    except Exception:
        if temporary_dir.exists():
            for child in temporary_dir.iterdir():
                child.unlink()
            temporary_dir.rmdir()
        raise
    for digit in ("2", "3"):
        final_path = _candidate_path(candidate_dir, digit)
        per_specialist[digit]["candidate"]["candidate_path"] = str(final_path)
        per_specialist[digit]["candidate"]["candidate_sha256"] = v52b._sha_file(
            final_path
        )
        per_specialist[digit]["fit"]["candidate_checkpoint_written"] = True
        per_specialist[digit]["fit"]["repair_candidate_selected"] = True
    report["candidate_checkpoint_written"] = True
    report["model_parameter_mutation_executed"] = True
    v51._atomic_write_json(report_path, report)
    return report


def historical_retention_executed_by_this_module() -> bool:
    return False


def validation_opened_by_this_module() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False
