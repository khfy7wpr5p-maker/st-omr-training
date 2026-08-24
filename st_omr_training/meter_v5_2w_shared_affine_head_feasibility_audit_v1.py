"""TRAIN-only shared affine-head feasibility audit after V5-2V.

V5-2V showed that a small 64D head-weight movement can create a large,
same-sign functional drift on V5 positives and historical negatives.  V5-2W
therefore asks a narrower question before any further repair training:

Can one linear head recover every V5 TRAIN decision while preserving every
historical TRAIN decision that the frozen specialist already gets right?

Two deterministic linear-program diagnostics are reported separately:

* frozen runtime: head weight is free, but bias and threshold stay frozen;
* free affine: weight and an effective intercept are free, only to test whether
  the frozen feature representation contains any shared affine separator.

The solver witnesses are verified in memory and never emitted or persisted.
This is diagnostic fitting, not model training: no model parameter, checkpoint,
threshold, or bias is changed.  Historical validation, First-30, V5 VAL, and
FINAL_HOLDOUT remain closed.  A solver status that merely reports apparent
infeasibility is not upgraded into a formal mathematical proof.
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
from . import meter_v5_2v_functional_logit_drift_audit_v1 as v52v


SCHEMA: Final[str] = "st-omr-meter-v5-2w-shared-affine-head-feasibility-audit-v1"
REPORT_NAME: Final[str] = "v5_2w_shared_affine_head_feasibility_audit_v1.json"
V52V_IMPLEMENTATION_HEAD: Final[str] = "b1db7923e91cec534fcfd95afad7f8b4ef87607b"
V52V_REPORT_SHA256: Final[str] = (
    "1ecc6b6600e0f01c1eeb4e8530d2184800dd470d2b66344c52a28a79d170bd3a"
)
V52V_EXECUTION_ENVELOPE_SHA256: Final[str] = (
    "f8c87b5ecec00f5a4e2cbaf5f1f07bb599f85dac41af8b989686c6f33f03ca4d"
)
EXPECTED_FEATURE_DIM: Final[int] = 64
EXPECTED_V5_COUNT: Final[int] = 540
EXPECTED_HISTORICAL_COUNT: Final[int] = 26_964
DECISION_MARGIN: Final[float] = 1e-4
WITNESS_TOLERANCE: Final[float] = 1e-7
SOLVER_METHOD: Final[str] = "highs-ds"
EXPECTED_SCIPY_VERSION: Final[str] = "1.18.0"
SOLVER_PRIMAL_FEASIBILITY_TOLERANCE: Final[float] = 1e-9
SOLVER_DUAL_FEASIBILITY_TOLERANCE: Final[float] = 1e-9
ProgressCallback = Callable[[int, int, str], None]


class MeterV5_2WError(RuntimeError):
    """Raised when V5-2W cannot preserve its fail-closed audit contract."""


def _fail(message: str) -> None:
    raise MeterV5_2WError(message)


def safety_boundary() -> dict[str, object]:
    """Declare diagnostic authority without disguising the affine LP fit."""
    return {
        "model_training": False,
        "autograd_grad_used": False,
        "backward": False,
        "optimizer_steps": 0,
        "diagnostic_linear_program_solve": True,
        "diagnostic_affine_witness_fit": True,
        "diagnostic_witness_persisted": False,
        "diagnostic_witness_values_emitted": False,
        "classifier_fit_for_deployment": False,
        "checkpoint_read": True,
        "candidate_checkpoint_write": False,
        "model_parameter_mutation": False,
        "threshold_tuning": False,
        "alternative_threshold_evaluated": False,
        "bias_selection": False,
        "free_intercept_diagnostic_only": True,
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
        "fixed_runtime_decision_margin": DECISION_MARGIN,
        "witness_verification_tolerance": WITNESS_TOLERANCE,
        "automatic_second_solver": False,
        "solver_sweep": False,
        "threshold_search": False,
        "bias_search": False,
        "infeasible_status_wording": "SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF",
    }


def _numpy_modules():
    try:
        import numpy as np
        from scipy.optimize import linprog
    except Exception as exc:  # pragma: no cover - exercised by runtime guard
        raise MeterV5_2WError("NumPy/SciPy LP runtime is unavailable") from exc
    return np, linprog


def _to_numpy(value, *, name: str):
    np, _linprog = _numpy_modules()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MeterV5_2WError(f"{name} is not a numeric array") from exc
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


def _solver_summary(result) -> dict[str, object]:
    status = int(result.status)
    if status == 0 and bool(result.success):
        claim = "WITNESS_PENDING_VERIFICATION"
    elif status == 2:
        claim = "SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF"
    else:
        claim = "UNPROVEN_SOLVER_DID_NOT_RETURN_A_USABLE_STATUS"
    return {
        "method": SOLVER_METHOD,
        "status": status,
        "success": bool(result.success),
        "iterations": int(getattr(result, "nit", 0)),
        "message": str(result.message),
        "feasibility_claim": claim,
    }


def _linprog_options() -> dict[str, object]:
    return {
        "presolve": False,
        "primal_feasibility_tolerance": SOLVER_PRIMAL_FEASIBILITY_TOLERANCE,
        "dual_feasibility_tolerance": SOLVER_DUAL_FEASIBILITY_TOLERANCE,
    }


def fixed_runtime_feasibility_v1(
    *, features, targets, fixed_bias: float, threshold: float
) -> dict[str, object]:
    """Find and verify a weight-only witness at the frozen bias/threshold."""
    np, linprog = _numpy_modules()
    x, y = _as_surface(features, targets, name="fixed-runtime joint surface")
    bias = float(fixed_bias)
    if not math.isfinite(bias):
        _fail("fixed runtime bias is non-finite")
    boundary = v52r._threshold_logit(float(threshold))
    sign = 2.0 * y - 1.0
    # sign * (x @ weight + bias - boundary) >= DECISION_MARGIN
    a_ub = -(sign[:, None] * x)
    b_ub = sign * (bias - boundary) - DECISION_MARGIN
    result = linprog(
        np.zeros(x.shape[1], dtype=np.float64),
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * x.shape[1],
        method=SOLVER_METHOD,
        options=_linprog_options(),
    )
    report = _solver_summary(result)
    report.update(
        {
            "constraint_count": int(x.shape[0]),
            "variable_count": int(x.shape[1]),
            "fixed_bias": bias,
            "frozen_threshold": float(threshold),
            "frozen_threshold_logit": boundary,
            "required_signed_decision_margin": DECISION_MARGIN,
            "feasible_witness_verified": False,
            "witness_values_emitted": False,
            "witness_persisted": False,
        }
    )
    if report["feasibility_claim"] != "WITNESS_PENDING_VERIFICATION":
        return report
    weight = np.asarray(result.x, dtype=np.float64)
    signed_margin = sign * (x @ weight + bias - boundary)
    minimum_margin = float(np.min(signed_margin))
    violations = int(
        np.sum(signed_margin < DECISION_MARGIN - WITNESS_TOLERANCE)
    )
    if not bool(np.isfinite(weight).all()) or not bool(np.isfinite(signed_margin).all()):
        _fail("fixed-runtime LP witness is non-finite")
    report.update(
        {
            "minimum_signed_decision_margin": minimum_margin,
            "witness_constraint_violations": violations,
            "witness_weight_l1": float(np.linalg.norm(weight, ord=1)),
            "witness_weight_l2": float(np.linalg.norm(weight, ord=2)),
        }
    )
    if violations != 0:
        report["feasibility_claim"] = "UNPROVEN_WITNESS_RESIDUAL_FAILED"
        return report
    report["feasibility_claim"] = "WITNESS_VERIFIED"
    report["feasible_witness_verified"] = True
    return report


def free_affine_feasibility_v1(*, features, targets) -> dict[str, object]:
    """Find and verify a diagnostic affine witness independent of calibration."""
    np, linprog = _numpy_modules()
    x, y = _as_surface(features, targets, name="free-affine joint surface")
    sign = 2.0 * y - 1.0
    augmented = np.concatenate(
        (x, np.ones((x.shape[0], 1), dtype=np.float64)), axis=1
    )
    # A positive unit margin is WLOG for a strictly separable free affine head.
    a_ub = -(sign[:, None] * augmented)
    b_ub = -np.ones(x.shape[0], dtype=np.float64)
    result = linprog(
        np.zeros(augmented.shape[1], dtype=np.float64),
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * augmented.shape[1],
        method=SOLVER_METHOD,
        options=_linprog_options(),
    )
    report = _solver_summary(result)
    report.update(
        {
            "constraint_count": int(x.shape[0]),
            "variable_count": int(augmented.shape[1]),
            "required_normalized_signed_margin": 1.0,
            "diagnostic_intercept_free": True,
            "diagnostic_intercept_selected_for_runtime": False,
            "feasible_witness_verified": False,
            "witness_values_emitted": False,
            "witness_persisted": False,
        }
    )
    if report["feasibility_claim"] != "WITNESS_PENDING_VERIFICATION":
        return report
    vector = np.asarray(result.x, dtype=np.float64)
    signed_margin = sign * (augmented @ vector)
    minimum_margin = float(np.min(signed_margin))
    violations = int(np.sum(signed_margin < 1.0 - WITNESS_TOLERANCE))
    if not bool(np.isfinite(vector).all()) or not bool(np.isfinite(signed_margin).all()):
        _fail("free-affine LP witness is non-finite")
    report.update(
        {
            "minimum_normalized_signed_margin": minimum_margin,
            "witness_constraint_violations": violations,
            "witness_weight_l1": float(np.linalg.norm(vector[:-1], ord=1)),
            "witness_weight_l2": float(np.linalg.norm(vector[:-1], ord=2)),
            "witness_absolute_effective_intercept": float(abs(vector[-1])),
        }
    )
    if violations != 0:
        report["feasibility_claim"] = "UNPROVEN_WITNESS_RESIDUAL_FAILED"
        return report
    report["feasibility_claim"] = "WITNESS_VERIFIED"
    report["feasible_witness_verified"] = True
    return report


def joint_path_diagnosis_v1(
    *, fixed_runtime: Mapping[str, object], free_affine: Mapping[str, object]
) -> dict[str, object]:
    fixed_claim = fixed_runtime.get("feasibility_claim")
    free_claim = free_affine.get("feasibility_claim")
    fixed_verified = fixed_claim == "WITNESS_VERIFIED"
    free_verified = free_claim == "WITNESS_VERIFIED"
    fixed_infeasible = fixed_claim == "SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF"
    free_infeasible = free_claim == "SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF"
    if fixed_verified:
        status = "FROZEN_RUNTIME_SHARED_HEAD_FEASIBLE_ON_TRAIN"
    elif fixed_infeasible and free_verified:
        status = "FREE_AFFINE_ONLY_FEASIBLE_ON_TRAIN"
    elif fixed_infeasible and free_infeasible:
        status = "NO_SHARED_AFFINE_WITNESS_SOLVER_REPORTED_INFEASIBLE"
    else:
        status = "UNPROVEN_SOLVER_EVIDENCE_GAP"
    return {
        "status": status,
        "frozen_runtime_shared_head_feasibility_proven": fixed_verified,
        "free_affine_shared_head_feasibility_proven": free_verified,
        "frozen_runtime_infeasibility_formally_proven": False,
        "free_affine_infeasibility_formally_proven": False,
        "representation_failure_proven": False,
        "threshold_or_bias_change_authorized": False,
        "repair_selected": False,
        "repair_training_authorized": False,
    }


def shared_head_feasibility_metrics_v1(
    *,
    historical_features,
    historical_targets,
    v5_features,
    v5_targets,
    frozen_weight,
    frozen_bias: float,
    threshold: float,
) -> dict[str, object]:
    """Build the no-regression joint TRAIN constraint surface and solve it."""
    np, _linprog = _numpy_modules()
    hist_x, hist_y = _as_surface(
        historical_features, historical_targets, name="historical"
    )
    v5_x, v5_y = _as_surface(v5_features, v5_targets, name="v5")
    weight = _to_numpy(frozen_weight, name="frozen weight").reshape(-1)
    if weight.shape != (EXPECTED_FEATURE_DIM,):
        _fail(f"frozen weight shape changed: {weight.shape}")
    bias = float(frozen_bias)
    if not math.isfinite(bias):
        _fail("frozen bias is non-finite")
    boundary = v52r._threshold_logit(float(threshold))
    hist_prediction = (hist_x @ weight + bias) >= boundary
    hist_correct = hist_prediction == (hist_y == 1.0)
    retained_x = hist_x[hist_correct]
    retained_y = hist_y[hist_correct]
    if retained_x.shape[0] == 0:
        _fail("frozen specialist has no correct historical decisions")
    joint_x = np.concatenate((v5_x, retained_x), axis=0)
    joint_y = np.concatenate((v5_y, retained_y), axis=0)
    fixed = fixed_runtime_feasibility_v1(
        features=joint_x,
        targets=joint_y,
        fixed_bias=bias,
        threshold=threshold,
    )
    free = free_affine_feasibility_v1(features=joint_x, targets=joint_y)
    return {
        "question": (
            "can_one_shared_head_classify_all_v5_train_examples_and_preserve_"
            "all_historical_train_decisions_that_the_frozen_head_gets_right"
        ),
        "constraint_policy": {
            "v5_train": "all-ground-truth-decisions-required",
            "historical_train": "all-frozen-correct-ground-truth-decisions-required",
            "historical_frozen_wrong": "unconstrained-no-new-regression-possible",
            "validation_examples_used": False,
        },
        "surface_counts": {
            "v5_all": int(v5_x.shape[0]),
            "v5_positive": int(np.sum(v5_y == 1.0)),
            "v5_negative": int(np.sum(v5_y == 0.0)),
            "historical_all": int(hist_x.shape[0]),
            "historical_frozen_correct": int(np.sum(hist_correct)),
            "historical_frozen_correct_positive": int(
                np.sum(hist_correct & (hist_y == 1.0))
            ),
            "historical_frozen_correct_negative": int(
                np.sum(hist_correct & (hist_y == 0.0))
            ),
            "historical_frozen_wrong": int(np.sum(~hist_correct)),
            "joint_constraints": int(joint_x.shape[0]),
        },
        "fixed_runtime_feasibility": fixed,
        "free_affine_feasibility": free,
        "joint_path_diagnosis": joint_path_diagnosis_v1(
            fixed_runtime=fixed,
            free_affine=free,
        ),
    }


def _read_exact_v5_2v_evidence_v1(
    *, report_path: Path, envelope_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    if not report_path.is_file() or not envelope_path.is_file():
        _fail("exact V5-2V report/envelope missing")
    report_bytes = report_path.read_bytes()
    envelope_bytes = envelope_path.read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != V52V_REPORT_SHA256:
        _fail("V5-2V report SHA256 mismatch")
    if hashlib.sha256(envelope_bytes).hexdigest() != V52V_EXECUTION_ENVELOPE_SHA256:
        _fail("V5-2V execution envelope SHA256 mismatch")
    report = v52b._read_json(report_path)
    envelope = v52b._read_json(envelope_path)
    if report.get("schema") != v52v.SCHEMA:
        _fail("V5-2V report schema mismatch")
    if report.get("repair_selected") is not False:
        _fail("V5-2V unexpectedly selected a repair")
    if report.get("first30_opened") is not False:
        _fail("V5-2V unexpectedly opened First-30")
    if report.get("v5_validation_opened") is not False:
        _fail("V5-2V unexpectedly opened V5 validation")
    if report.get("final_holdout_locked") is not True:
        _fail("V5-2V FINAL_HOLDOUT lock changed")
    if envelope.get("expected_head") != V52V_IMPLEMENTATION_HEAD:
        _fail("V5-2V execution HEAD mismatch")
    if envelope.get("audit_report_sha256") != V52V_REPORT_SHA256:
        _fail("V5-2V execution report binding mismatch")
    return report, envelope


def run_shared_affine_head_feasibility_audit_v1(
    data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    v5_2v_report: str | Path,
    v5_2v_envelope: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run the aggregate no-regression feasibility audit on frozen TRAIN features."""
    root = Path(data_root)
    output = root / v51.ANNOTATIONS_DIR / REPORT_NAME
    if output.exists():
        _fail("refusing to overwrite/rerun V5-2W evidence")
    prior_report, _prior_envelope = _read_exact_v5_2v_evidence_v1(
        report_path=Path(v5_2v_report),
        envelope_path=Path(v5_2v_envelope),
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
        _fail("V5 TRAIN slot manifest changed after V5-2V")
    historical_features, historical_targets = v52n._historical_surface(
        m4a_root=Path(m4a_root),
        d10_root=Path(d10_root),
        models=frozen_models,
        progress=progress,
    )
    per_specialist: dict[str, object] = {}
    for digit in ("2", "3"):
        frozen = frozen_models[digit]
        metrics = shared_head_feasibility_metrics_v1(
            historical_features=historical_features[digit],
            historical_targets=historical_targets[digit],
            v5_features=v5_features[digit],
            v5_targets=v5_targets[digit],
            frozen_weight=frozen.head.weight.detach().cpu().reshape(-1),
            frozen_bias=float(frozen.head.bias.detach().cpu().reshape(-1)[0].item()),
            threshold=v52b.FROZEN_THRESHOLDS[digit],
        )
        counts = metrics["surface_counts"]
        if counts["v5_all"] != EXPECTED_V5_COUNT:
            _fail(f"{digit}-AI V5 TRAIN count changed")
        if counts["historical_all"] != EXPECTED_HISTORICAL_COUNT:
            _fail(f"{digit}-AI historical TRAIN count changed")
        per_specialist[digit] = metrics
    solver = solver_contract()
    if solver["library_version_matches_expected"] is not True:
        _fail("SciPy runtime version does not match the pinned solver contract")
    report: dict[str, object] = {
        "schema": SCHEMA,
        "analysis_surface": "aggregate-v5-and-historical-train-only",
        "slot_manifest_sha256": manifest_sha,
        "v5_train_slot_count": EXPECTED_V5_COUNT,
        "historical_train_record_count": EXPECTED_HISTORICAL_COUNT,
        "feature_dim": EXPECTED_FEATURE_DIM,
        "exact_v5_2v_binding": {
            "implementation_head": V52V_IMPLEMENTATION_HEAD,
            "report_sha256": V52V_REPORT_SHA256,
            "execution_envelope_sha256": V52V_EXECUTION_ENVELOPE_SHA256,
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
