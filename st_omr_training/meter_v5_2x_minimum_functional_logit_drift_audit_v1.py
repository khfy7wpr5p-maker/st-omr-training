"""TRAIN-only minimum functional-logit-drift audit after V5-2W.

V5-2W verified that a shared 64D head weight exists for each specialist at
the frozen bias and frozen runtime threshold.  Its zero objective did not,
however, minimize witness norm or functional disturbance.  V5-2X therefore
asks one safety-relevant question before any further repair training:

What is the smallest possible maximum absolute logit drift over every
historical TRAIN example, while classifying all V5 TRAIN examples correctly
and preserving every historical TRAIN decision the frozen head gets right?

The answer is obtained with one preregistered linear program.  A temporary
diagnostic delta-weight witness is verified in memory but never emitted,
persisted, copied into a model, or selected as a repair.  The source bias and
runtime threshold remain frozen.  Historical validation, First-30, V5 VAL,
and FINAL_HOLDOUT remain closed.  Solver-reported optimality or infeasibility
is not promoted to a formal mathematical proof.
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
from . import meter_v5_2r_train_class_margin_gradient_audit_v1 as v52r
from . import meter_v5_2w_shared_affine_head_feasibility_audit_v1 as v52w


SCHEMA: Final[str] = "st-omr-meter-v5-2x-minimum-functional-logit-drift-audit-v1"
REPORT_NAME: Final[str] = "v5_2x_minimum_functional_logit_drift_audit_v1.json"
V52W_IMPLEMENTATION_HEAD: Final[str] = "bdd82204182e3d5043a64907de7e0f0394089a20"
V52W_REPORT_SHA256: Final[str] = (
    "0fdcff6a9114eec08a2f3c512de1336cf4af96be4ab82cf168d14fa2e77f4095"
)
V52W_EXECUTION_ENVELOPE_SHA256: Final[str] = (
    "4b86e81547d9d22aee60353e57eac7d28fd9ab8ed6988f7ec629241c6a723d54"
)
EXPECTED_FEATURE_DIM: Final[int] = 64
EXPECTED_V5_COUNT: Final[int] = 540
EXPECTED_HISTORICAL_COUNT: Final[int] = 26_964
DECISION_MARGIN: Final[float] = 1e-4
WITNESS_TOLERANCE: Final[float] = 1e-7
IDENTITY_TOLERANCE: Final[float] = 1e-9
SOLVER_METHOD: Final[str] = "highs-ds"
EXPECTED_SCIPY_VERSION: Final[str] = "1.18.0"
SOLVER_PRIMAL_FEASIBILITY_TOLERANCE: Final[float] = 1e-9
SOLVER_DUAL_FEASIBILITY_TOLERANCE: Final[float] = 1e-9
ProgressCallback = Callable[[int, int, str], None]


class MeterV5_2XError(RuntimeError):
    """Raised when V5-2X cannot preserve its fail-closed audit contract."""


def _fail(message: str) -> None:
    raise MeterV5_2XError(message)


def safety_boundary() -> dict[str, object]:
    """Declare exactly what the diagnostic LP is and is not allowed to do."""
    return {
        "model_training": False,
        "autograd_grad_used": False,
        "backward": False,
        "optimizer_steps": 0,
        "diagnostic_linear_program_solve": True,
        "diagnostic_witness_fit": True,
        "diagnostic_witness_persisted": False,
        "diagnostic_witness_values_emitted": False,
        "classifier_fit_for_deployment": False,
        "checkpoint_read": True,
        "candidate_checkpoint_write": False,
        "model_parameter_mutation": False,
        "threshold_tuning": False,
        "alternative_threshold_evaluated": False,
        "bias_selection": False,
        "historical_validation_opened": False,
        "historical_validation_report_read": False,
        "historical_validation_error_examples_read": False,
        "first30_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "per_example_rows_emitted": False,
        "repair_selected": False,
        "repair_training_authorized": False,
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
        "decision_margin": DECISION_MARGIN,
        "witness_verification_tolerance": WITNESS_TOLERANCE,
        "functional_identity_tolerance": IDENTITY_TOLERANCE,
        "objective": "minimum_max_absolute_historical_train_logit_drift",
        "historical_drift_objective_minimized": True,
        "weight_norm_minimized": False,
        "weight_norm_not_minimal": True,
        "automatic_second_solver": False,
        "secondary_lexicographic_solve": False,
        "solver_sweep": False,
        "threshold_search": False,
        "bias_search": False,
        "infeasible_status_wording": "SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF",
        "optimal_status_wording": "SOLVER_REPORTED_OPTIMAL_NOT_FORMAL_PROOF",
    }


def _numpy_modules():
    try:
        import numpy as np
        from scipy.optimize import linprog
    except Exception as exc:  # pragma: no cover - runtime guard
        raise MeterV5_2XError("NumPy/SciPy LP runtime is unavailable") from exc
    return np, linprog


def _to_numpy(value, *, name: str):
    np, _linprog = _numpy_modules()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MeterV5_2XError(f"{name} is not a numeric array") from exc
    if array.size == 0 or not bool(np.isfinite(array).all()):
        _fail(f"{name} is empty or non-finite")
    return array


def _as_surface(features, targets, *, name: str):
    np, _linprog = _numpy_modules()
    x = _to_numpy(features, name=f"{name} features")
    y = _to_numpy(targets, name=f"{name} targets").reshape(-1)
    if x.ndim != 2 or x.shape[1] != EXPECTED_FEATURE_DIM:
        _fail(f"{name} feature shape changed: {x.shape}")
    if x.shape[0] != y.shape[0]:
        _fail(f"{name} feature/target cardinality mismatch")
    if not bool(np.isin(y, (0.0, 1.0)).all()):
        _fail(f"{name} targets are not binary")
    if not bool((y == 0.0).any()) or not bool((y == 1.0).any()):
        _fail(f"{name} must contain both classes")
    return x, y


def _linprog_options() -> dict[str, object]:
    return {
        "presolve": False,
        "primal_feasibility_tolerance": SOLVER_PRIMAL_FEASIBILITY_TOLERANCE,
        "dual_feasibility_tolerance": SOLVER_DUAL_FEASIBILITY_TOLERANCE,
    }


def _solver_claims(result) -> tuple[str, str]:
    status = int(result.status)
    if status == 0 and bool(result.success):
        return (
            "WITNESS_PENDING_VERIFICATION",
            "SOLVER_REPORTED_OPTIMAL_NOT_FORMAL_PROOF",
        )
    if status == 2:
        return (
            "SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF",
            "OPTIMALITY_NOT_PROVEN",
        )
    return (
        "UNPROVEN_SOLVER_DID_NOT_RETURN_A_USABLE_STATUS",
        "OPTIMALITY_NOT_PROVEN",
    )


def _quantile_summary(values, *, name: str) -> dict[str, float | int]:
    np, _linprog = _numpy_modules()
    array = _to_numpy(values, name=name).reshape(-1)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std_population": float(np.std(array)),
    }


def _transition_counts(*, frozen_prediction, diagnostic_prediction, targets):
    np, _linprog = _numpy_modules()
    p0 = np.asarray(frozen_prediction, dtype=bool).reshape(-1)
    p1 = np.asarray(diagnostic_prediction, dtype=bool).reshape(-1)
    y = np.asarray(targets, dtype=bool).reshape(-1)
    if p0.size == 0 or p0.shape != p1.shape or p0.shape != y.shape:
        _fail("prediction transition surface mismatch")
    c0 = p0 == y
    c1 = p1 == y
    result = {
        "count": int(y.size),
        "correct_to_correct": int(np.sum(c0 & c1)),
        "correct_to_wrong": int(np.sum(c0 & ~c1)),
        "wrong_to_correct": int(np.sum(~c0 & c1)),
        "wrong_to_wrong": int(np.sum(~c0 & ~c1)),
        "below_to_above_threshold": int(np.sum(~p0 & p1)),
        "above_to_below_threshold": int(np.sum(p0 & ~p1)),
        "frozen_predicted_positive": int(np.sum(p0)),
        "diagnostic_predicted_positive": int(np.sum(p1)),
    }
    if sum(
        result[key]
        for key in (
            "correct_to_correct",
            "correct_to_wrong",
            "wrong_to_correct",
            "wrong_to_wrong",
        )
    ) != result["count"]:
        _fail("prediction transition accounting mismatch")
    return result


def _classification_metrics(*, prediction, targets) -> dict[str, float | int]:
    np, _linprog = _numpy_modules()
    predicted = np.asarray(prediction, dtype=bool).reshape(-1)
    target = np.asarray(targets, dtype=bool).reshape(-1)
    if predicted.shape != target.shape or target.size == 0:
        _fail("classification metric surface mismatch")
    tp = int(np.sum(predicted & target))
    fp = int(np.sum(predicted & ~target))
    fn = int(np.sum(~predicted & target))
    tn = int(np.sum(~predicted & ~target))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / target.size,
    }


def _weight_geometry(*, frozen_weight, diagnostic_weight) -> dict[str, object]:
    np, _linprog = _numpy_modules()
    w0 = np.asarray(frozen_weight, dtype=np.float64).reshape(-1)
    w1 = np.asarray(diagnostic_weight, dtype=np.float64).reshape(-1)
    delta = w1 - w0
    norm0 = float(np.linalg.norm(w0))
    norm1 = float(np.linalg.norm(w1))
    delta_norm = float(np.linalg.norm(delta))
    denominator = norm0 * norm1
    cosine = None
    angle = None
    if denominator > 0.0:
        cosine = float(np.clip(np.dot(w0, w1) / denominator, -1.0, 1.0))
        angle = float(math.degrees(math.acos(cosine)))
    return {
        "frozen_weight_l2": norm0,
        "diagnostic_weight_l2": norm1,
        "delta_weight_l1": float(np.linalg.norm(delta, ord=1)),
        "delta_weight_l2": delta_norm,
        "delta_over_frozen_l2": delta_norm / norm0 if norm0 > 0.0 else None,
        "frozen_diagnostic_cosine": cosine,
        "head_angle_change_degrees": angle,
        "descriptive_only": True,
        "weight_norm_minimized": False,
        "weight_norm_not_minimal": True,
    }


def minimum_functional_logit_drift_v1(
    *,
    historical_features,
    historical_targets,
    v5_features,
    v5_targets,
    frozen_weight,
    frozen_bias: float,
    threshold: float,
) -> dict[str, object]:
    """Solve and independently verify the single V5-2X minimax LP."""
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

    frozen_hist_logit = hist_x @ w0 + bias
    frozen_v5_logit = v5_x @ w0 + bias
    frozen_hist_prediction = frozen_hist_logit >= boundary
    frozen_v5_prediction = frozen_v5_logit >= boundary
    hist_correct = frozen_hist_prediction == (hist_y == 1.0)
    retained_x = hist_x[hist_correct]
    retained_y = hist_y[hist_correct]
    if retained_x.shape[0] == 0:
        _fail("frozen specialist has no correct historical decisions")

    # Variables are [delta_weight_0..63, max_abs_historical_logit_drift].
    variable_count = EXPECTED_FEATURE_DIM + 1
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[-1] = 1.0

    def hard_decision_rows(x, y):
        sign = 2.0 * y - 1.0
        frozen_margin = sign * (x @ w0 + bias - boundary)
        rows = np.zeros((x.shape[0], variable_count), dtype=np.float64)
        rows[:, :-1] = -(sign[:, None] * x)
        bounds = frozen_margin - DECISION_MARGIN
        return rows, bounds

    v5_rows, v5_bounds = hard_decision_rows(v5_x, v5_y)
    retained_rows, retained_bounds = hard_decision_rows(retained_x, retained_y)
    upper_drift_rows = np.zeros((hist_x.shape[0], variable_count), dtype=np.float64)
    upper_drift_rows[:, :-1] = hist_x
    upper_drift_rows[:, -1] = -1.0
    lower_drift_rows = np.zeros((hist_x.shape[0], variable_count), dtype=np.float64)
    lower_drift_rows[:, :-1] = -hist_x
    lower_drift_rows[:, -1] = -1.0
    a_ub = np.concatenate(
        (v5_rows, retained_rows, upper_drift_rows, lower_drift_rows), axis=0
    )
    b_ub = np.concatenate(
        (
            v5_bounds,
            retained_bounds,
            np.zeros(hist_x.shape[0], dtype=np.float64),
            np.zeros(hist_x.shape[0], dtype=np.float64),
        )
    )
    constraint_count = int(a_ub.shape[0])
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * EXPECTED_FEATURE_DIM + [(0.0, None)],
        method=SOLVER_METHOD,
        options=_linprog_options(),
    )
    # The dense constraint matrix is deliberately specialist-local.  Release
    # it before aggregate verification so 2-AI and 3-AI do not overlap in RAM.
    del a_ub, b_ub, v5_rows, retained_rows, upper_drift_rows, lower_drift_rows
    witness_claim, optimality_claim = _solver_claims(result)
    report: dict[str, object] = {
        "method": SOLVER_METHOD,
        "status": int(result.status),
        "success": bool(result.success),
        "iterations": int(getattr(result, "nit", 0)),
        "message": str(result.message),
        "witness_claim": witness_claim,
        "optimality_claim": optimality_claim,
        "functional_drift_witness_verified": False,
        "constraint_count": constraint_count,
        "variable_count": variable_count,
        "v5_constraint_count": int(v5_x.shape[0]),
        "historical_retention_constraint_count": int(retained_x.shape[0]),
        "historical_absolute_drift_constraint_count": int(2 * hist_x.shape[0]),
        "historical_all_count": int(hist_x.shape[0]),
        "historical_frozen_correct_count": int(np.sum(hist_correct)),
        "historical_frozen_wrong_count": int(np.sum(~hist_correct)),
        "fixed_bias": bias,
        "frozen_threshold": float(threshold),
        "frozen_threshold_logit": boundary,
        "required_signed_decision_margin": DECISION_MARGIN,
        "objective": "minimum_max_absolute_historical_train_logit_drift",
        "functional_drift_objective_minimized": True,
        "weight_norm_minimized": False,
        "weight_norm_not_minimal": True,
        "witness_values_emitted": False,
        "witness_persisted": False,
        "repair_selected": False,
        "repair_training_authorized": False,
    }
    if witness_claim != "WITNESS_PENDING_VERIFICATION":
        return report

    solution = np.asarray(result.x, dtype=np.float64)
    delta = solution[:-1]
    objective_bound = float(solution[-1])
    solver_objective = float(result.fun)
    diagnostic_weight = w0 + delta
    diagnostic_hist_logit = hist_x @ diagnostic_weight + bias
    diagnostic_v5_logit = v5_x @ diagnostic_weight + bias
    direct_hist_delta = hist_x @ delta
    recomputed_hist_delta = diagnostic_hist_logit - frozen_hist_logit
    identity_error = float(np.max(np.abs(direct_hist_delta - recomputed_hist_delta)))
    if not bool(np.isfinite(solution).all()):
        _fail("minimum-drift LP witness is non-finite")
    if not bool(np.isfinite(diagnostic_hist_logit).all()):
        _fail("minimum-drift historical logits are non-finite")
    if not bool(np.isfinite(diagnostic_v5_logit).all()):
        _fail("minimum-drift V5 logits are non-finite")

    v5_sign = 2.0 * v5_y - 1.0
    retained_sign = 2.0 * retained_y - 1.0
    v5_margin = v5_sign * (diagnostic_v5_logit - boundary)
    retained_margin = retained_sign * (
        diagnostic_hist_logit[hist_correct] - boundary
    )
    max_abs_drift = float(np.max(np.abs(direct_hist_delta)))
    objective_recomputation_error = abs(objective_bound - max_abs_drift)
    solver_objective_identity_error = abs(solver_objective - objective_bound)
    v5_violations = int(np.sum(v5_margin < DECISION_MARGIN - WITNESS_TOLERANCE))
    retained_violations = int(
        np.sum(retained_margin < DECISION_MARGIN - WITNESS_TOLERANCE)
    )
    drift_violations = int(
        np.sum(np.abs(direct_hist_delta) > objective_bound + WITNESS_TOLERANCE)
    )
    identity_verified = identity_error <= IDENTITY_TOLERANCE
    diagnostic_hist_prediction = diagnostic_hist_logit >= boundary
    diagnostic_v5_prediction = diagnostic_v5_logit >= boundary
    report.update(
        {
            "minimum_max_absolute_historical_logit_drift": objective_bound,
            "solver_objective_value": solver_objective,
            "independently_recomputed_max_absolute_historical_logit_drift": max_abs_drift,
            "objective_recomputation_absolute_error": objective_recomputation_error,
            "solver_objective_identity_absolute_error": solver_objective_identity_error,
            "historical_logit_drift": _quantile_summary(
                direct_hist_delta, name="historical logit drift"
            ),
            "historical_absolute_logit_drift": _quantile_summary(
                np.abs(direct_hist_delta), name="historical absolute logit drift"
            ),
            "minimum_v5_signed_decision_margin": float(np.min(v5_margin)),
            "minimum_historical_retained_signed_decision_margin": float(
                np.min(retained_margin)
            ),
            "v5_constraint_violations": v5_violations,
            "historical_retention_constraint_violations": retained_violations,
            "historical_drift_bound_violations": drift_violations,
            "functional_delta_identity_max_abs_error": identity_error,
            "functional_delta_identity_verified": identity_verified,
            "weight_geometry": _weight_geometry(
                frozen_weight=w0, diagnostic_weight=diagnostic_weight
            ),
            "frozen_v5_train_metrics": _classification_metrics(
                prediction=frozen_v5_prediction, targets=v5_y
            ),
            "diagnostic_v5_train_metrics": _classification_metrics(
                prediction=diagnostic_v5_prediction, targets=v5_y
            ),
            "frozen_historical_train_metrics": _classification_metrics(
                prediction=frozen_hist_prediction, targets=hist_y
            ),
            "diagnostic_historical_train_metrics": _classification_metrics(
                prediction=diagnostic_hist_prediction, targets=hist_y
            ),
            "v5_transition_counts": _transition_counts(
                frozen_prediction=frozen_v5_prediction,
                diagnostic_prediction=diagnostic_v5_prediction,
                targets=v5_y,
            ),
            "historical_transition_counts": _transition_counts(
                frozen_prediction=frozen_hist_prediction,
                diagnostic_prediction=diagnostic_hist_prediction,
                targets=hist_y,
            ),
        }
    )
    verification_ok = (
        objective_bound >= -WITNESS_TOLERANCE
        and v5_violations == 0
        and retained_violations == 0
        and drift_violations == 0
        and identity_verified
        and objective_recomputation_error <= WITNESS_TOLERANCE
        and solver_objective_identity_error <= WITNESS_TOLERANCE
    )
    if not verification_ok:
        report["witness_claim"] = "UNPROVEN_WITNESS_RESIDUAL_FAILED"
        report["optimality_claim"] = "OPTIMALITY_NOT_PROVEN"
        return report
    report["witness_claim"] = "WITNESS_VERIFIED"
    report["functional_drift_witness_verified"] = True
    return report


def path_diagnosis_v1(result: Mapping[str, object]) -> dict[str, object]:
    verified = result.get("witness_claim") == "WITNESS_VERIFIED"
    solver_infeasible = (
        result.get("witness_claim")
        == "SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF"
    )
    if verified:
        status = "MINIMUM_FUNCTIONAL_DRIFT_WITNESS_VERIFIED_ON_TRAIN"
    elif solver_infeasible:
        status = "EVIDENCE_CONFLICT_V5_2W_FEASIBLE_BUT_V5_2X_SOLVER_REPORTED_INFEASIBLE"
    else:
        status = "UNPROVEN_SOLVER_OR_RESIDUAL_EVIDENCE_GAP"
    return {
        "status": status,
        "v5_2w_frozen_runtime_shared_head_feasibility_bound": True,
        "minimum_functional_drift_witness_verified_on_train": verified,
        "minimum_functional_drift_solver_reported_optimal": bool(
            result.get("optimality_claim")
            == "SOLVER_REPORTED_OPTIMAL_NOT_FORMAL_PROOF"
        ),
        "formal_optimality_proof_claimed": False,
        "formal_infeasibility_proof_claimed": False,
        "weight_norm_minimized": False,
        "generalization_proven": False,
        "historical_validation_preservation_proven": False,
        "representation_failure_proven": False,
        "threshold_or_bias_change_authorized": False,
        "repair_selected": False,
        "repair_training_authorized": False,
    }


def _read_exact_v5_2w_evidence_v1(
    *, report_path: Path, envelope_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    if not report_path.is_file() or not envelope_path.is_file():
        _fail("exact V5-2W report/envelope missing")
    report_bytes = report_path.read_bytes()
    envelope_bytes = envelope_path.read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != V52W_REPORT_SHA256:
        _fail("V5-2W report SHA256 mismatch")
    if hashlib.sha256(envelope_bytes).hexdigest() != V52W_EXECUTION_ENVELOPE_SHA256:
        _fail("V5-2W execution envelope SHA256 mismatch")
    report = v52b._read_json(report_path)
    envelope = v52b._read_json(envelope_path)
    if report.get("schema") != v52w.SCHEMA:
        _fail("V5-2W report schema mismatch")
    if report.get("repair_selected") is not False:
        _fail("V5-2W unexpectedly selected a repair")
    if report.get("first30_opened") is not False:
        _fail("V5-2W unexpectedly opened First-30")
    if report.get("v5_validation_opened") is not False:
        _fail("V5-2W unexpectedly opened V5 validation")
    if report.get("final_holdout_locked") is not True:
        _fail("V5-2W FINAL_HOLDOUT lock changed")
    for key, expected in (
        ("model_training", False),
        ("autograd_grad_used", False),
        ("backward", False),
        ("optimizer_steps", 0),
        ("diagnostic_witness_persisted", False),
        ("diagnostic_witness_values_emitted", False),
        ("candidate_checkpoint_write", False),
        ("model_parameter_mutation", False),
        ("threshold_tuning", False),
        ("bias_selection", False),
        ("historical_validation_opened", False),
    ):
        if report.get(key) != expected:
            _fail(f"V5-2W safety boundary changed: {key}")
    specialists = report.get("per_specialist")
    if not isinstance(specialists, Mapping):
        _fail("V5-2W per-specialist evidence missing")
    for digit in ("2", "3"):
        item = specialists.get(digit)
        if not isinstance(item, Mapping):
            _fail(f"V5-2W {digit}-AI evidence missing")
        fixed = item.get("fixed_runtime_feasibility")
        if not isinstance(fixed, Mapping):
            _fail(f"V5-2W {digit}-AI fixed-runtime evidence missing")
        if fixed.get("feasibility_claim") != "WITNESS_VERIFIED":
            _fail(f"V5-2W {digit}-AI shared-head feasibility is not verified")
        if fixed.get("witness_persisted") is not False:
            _fail(f"V5-2W {digit}-AI unexpectedly persisted a witness")
        if fixed.get("witness_values_emitted") is not False:
            _fail(f"V5-2W {digit}-AI unexpectedly emitted witness values")
    if envelope.get("expected_head") != V52W_IMPLEMENTATION_HEAD:
        _fail("V5-2W execution HEAD mismatch")
    if envelope.get("actual_head_before_run") != V52W_IMPLEMENTATION_HEAD:
        _fail("V5-2W pre-run HEAD mismatch")
    if envelope.get("actual_head_after_run") != V52W_IMPLEMENTATION_HEAD:
        _fail("V5-2W post-run HEAD mismatch")
    if envelope.get("audit_report_sha256") != V52W_REPORT_SHA256:
        _fail("V5-2W execution report binding mismatch")
    return report, envelope


def run_minimum_functional_logit_drift_audit_v1(
    data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    v5_2w_report: str | Path,
    v5_2w_envelope: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run the aggregate minimax functional-drift audit on frozen TRAIN features."""
    root = Path(data_root)
    output = root / v51.ANNOTATIONS_DIR / REPORT_NAME
    if output.exists():
        _fail("refusing to overwrite/rerun V5-2X evidence")
    prior_report, _prior_envelope = _read_exact_v5_2w_evidence_v1(
        report_path=Path(v5_2w_report),
        envelope_path=Path(v5_2w_envelope),
    )
    frozen_models = v52n._frozen_models(
        digit2_frozen=Path(digit2_frozen),
        digit3_frozen=Path(digit3_frozen),
    )
    manifest_path, _rows, v5_features, v5_targets, _metrics = v52n._v5_surface(
        root, frozen_models
    )
    manifest_sha = v52b._sha_file(manifest_path)
    if prior_report.get("slot_manifest_sha256") != manifest_sha:
        _fail("V5 TRAIN slot manifest changed after V5-2W")
    historical_features, historical_targets = v52n._historical_surface(
        m4a_root=Path(m4a_root),
        d10_root=Path(d10_root),
        models=frozen_models,
        progress=progress,
    )
    per_specialist: dict[str, object] = {}
    for digit in ("2", "3"):
        frozen = frozen_models[digit]
        result = minimum_functional_logit_drift_v1(
            historical_features=historical_features[digit],
            historical_targets=historical_targets[digit],
            v5_features=v5_features[digit],
            v5_targets=v5_targets[digit],
            frozen_weight=frozen.head.weight.detach().cpu().reshape(-1),
            frozen_bias=float(frozen.head.bias.detach().cpu().reshape(-1)[0].item()),
            threshold=v52b.FROZEN_THRESHOLDS[digit],
        )
        if result["v5_constraint_count"] != EXPECTED_V5_COUNT:
            _fail(f"{digit}-AI V5 TRAIN count changed")
        if result["historical_all_count"] != EXPECTED_HISTORICAL_COUNT:
            _fail(f"{digit}-AI historical TRAIN count changed")
        per_specialist[digit] = {
            "minimum_functional_logit_drift": result,
            "path_diagnosis": path_diagnosis_v1(result),
        }
    solver = solver_contract()
    if solver["library_version_matches_expected"] is not True:
        _fail("SciPy runtime version does not match the pinned solver contract")
    report: dict[str, object] = {
        "schema": SCHEMA,
        "question": (
            "what_is_the_minimum_possible_maximum_absolute_historical_train_"
            "logit_drift_under_all_v5_and_frozen_correct_historical_decision_"
            "constraints"
        ),
        "analysis_surface": "aggregate-v5-and-historical-train-only",
        "slot_manifest_sha256": manifest_sha,
        "v5_train_slot_count": EXPECTED_V5_COUNT,
        "historical_train_record_count": EXPECTED_HISTORICAL_COUNT,
        "feature_dim": EXPECTED_FEATURE_DIM,
        "exact_v5_2w_binding": {
            "implementation_head": V52W_IMPLEMENTATION_HEAD,
            "report_sha256": V52W_REPORT_SHA256,
            "execution_envelope_sha256": V52W_EXECUTION_ENVELOPE_SHA256,
        },
        "constraint_policy": {
            "v5_train": "all-ground-truth-decisions-required",
            "historical_train_frozen_correct": "all-decisions-preserved",
            "historical_train_all": "maximum-absolute-logit-drift-minimized",
            "historical_validation_examples_used": False,
        },
        "solver_contract": solver,
        "per_specialist": per_specialist,
        **safety_boundary(),
    }
    v51._atomic_write_json(output, report)
    return report


def validation_opened_by_this_module() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False
