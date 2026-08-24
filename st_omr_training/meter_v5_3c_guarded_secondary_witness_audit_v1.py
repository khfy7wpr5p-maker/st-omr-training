"""Diagnostic-only V5-3C guarded secondary witness audit.

V5-3A's 3-AI secondary solve satisfied every decision, margin, parameter,
identity, transition, and metric check.  Its only failed predicate was a
1.2146432482040836e-7 excess over the fixed primary L1 cap, just
2.146432482040836e-8 beyond the unchanged 1e-7 witness tolerance.

This stage does not loosen a tolerance or change the scientific problem.  It
reuses the exact V5-3A primary optimum, keeps the external L1 acceptance cap
unchanged, tightens the solver-facing cap by five witness tolerances, and
normalizes that single secondary constraint to RHS one.  It produces evidence
only: no candidate checkpoint, model mutation, retention, or validation.
"""
from __future__ import annotations

import hashlib
import json
import math
from importlib import metadata
from pathlib import Path
from typing import Callable, Final, Mapping

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2n_frozen_feature_transfer_audit_v1 as v52n
from . import meter_v5_2p_fixed_bias_head_repair_v1 as v52p
from . import meter_v5_2x_minimum_functional_logit_drift_audit_v1 as v52x
from . import meter_v5_2y_lexicographic_parameter_stability_audit_v1 as v52y
from . import meter_v5_3a_robust_margin_head_candidate_v1 as v53a


SCHEMA: Final[str] = "st-omr-meter-v5-3c-guarded-secondary-witness-audit-v1"
REPORT_NAME: Final[str] = "v5_3c_guarded_secondary_witness_audit_v1.json"
APPROVAL_TOKEN: Final[str] = "V5_3C_SINGLE_GUARDED_SECONDARY_AUDIT_APPROVED"
V53A_SOURCE_IMPLEMENTATION_HEAD: Final[str] = (
    "cdc6683a556c16b00e7b154fca8e89ba5dd848b7"
)
V53A_SOURCE_HARNESS_HEAD: Final[str] = (
    "c2d5f1652adac52387e33b9d2f33078f864f980b"
)
V53A_REPORT_SHA256: Final[str] = (
    "a483173353b9e425a4a3eb8d177376c15a7c5fa1d13c62689356c04b3fffd92e"
)
V53A_RECOVERY_ENVELOPE_SHA256: Final[str] = (
    "6514983e886c9ba41398f2a0c1888d3088455ab612cd6ad91614bcd8d7db4d40"
)
OBSERVED_3AI_CAP_EXCESS: Final[float] = 1.2146432482040836e-7
INTERNAL_CAP_GUARD_MULTIPLIER: Final[float] = 5.0
INTERNAL_CAP_GUARD: Final[float] = (
    INTERNAL_CAP_GUARD_MULTIPLIER * v53a.WITNESS_TOLERANCE
)

EXPECTED_FEATURE_DIM = v53a.EXPECTED_FEATURE_DIM
EXPECTED_V5_COUNT = v53a.EXPECTED_V5_COUNT
EXPECTED_HISTORICAL_COUNT = v53a.EXPECTED_HISTORICAL_COUNT
ROBUST_DECISION_MARGIN = v53a.ROBUST_DECISION_MARGIN
SOLVER_MARGIN_BUFFER = v53a.SOLVER_MARGIN_BUFFER
PRIMARY_L1_ABSOLUTE_SLACK = v53a.PRIMARY_L1_ABSOLUTE_SLACK
WITNESS_TOLERANCE = v53a.WITNESS_TOLERANCE
IDENTITY_TOLERANCE = v53a.IDENTITY_TOLERANCE
ProgressCallback = Callable[[int, int, str], None]


class MeterV5_3CError(RuntimeError):
    """Raised when V5-3C departs from its fixed audit contract."""


def _fail(message: str) -> None:
    raise MeterV5_3CError(message)


def safety_boundary() -> dict[str, object]:
    return {
        "training": False,
        "secondary_linear_program_witness_fit": True,
        "primary_linear_program_rerun": False,
        "candidate_checkpoint_write_authorized": False,
        "candidate_checkpoint_written": False,
        "model_parameter_mutation_executed": False,
        "autograd_grad_used": False,
        "backward": False,
        "optimizer_steps": 0,
        "frozen_backbone": True,
        "frozen_head_bias": True,
        "runtime_threshold_tuning": False,
        "alternative_threshold_evaluated": False,
        "solver_sweep": False,
        "fallback_solver": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "historical_retention_executed": False,
        "historical_validation_opened": False,
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
        "expected_library_version": v53a.EXPECTED_SCIPY_VERSION,
        "actual_library_version": actual_version,
        "library_version_matches_expected": (
            actual_version == v53a.EXPECTED_SCIPY_VERSION
        ),
        "method": v53a.SOLVER_METHOD,
        "presolve": False,
        "primal_feasibility_tolerance": (
            v53a.SOLVER_PRIMAL_FEASIBILITY_TOLERANCE
        ),
        "dual_feasibility_tolerance": v53a.SOLVER_DUAL_FEASIBILITY_TOLERANCE,
        "witness_tolerance": WITNESS_TOLERANCE,
        "identity_tolerance": IDENTITY_TOLERANCE,
        "primary_l1_absolute_slack": PRIMARY_L1_ABSOLUTE_SLACK,
        "internal_cap_guard": INTERNAL_CAP_GUARD,
        "internal_cap_guard_in_witness_tolerances": (
            INTERNAL_CAP_GUARD_MULTIPLIER
        ),
        "secondary_l1_cap_row_normalized_to_rhs_one": True,
        "external_acceptance_cap_unchanged": True,
        "primary_optimum_reused_from_exact_v5_3a": True,
        "primary_lp_rerun": False,
        "solver_sweep": False,
        "fallback_solver": False,
        "tolerance_changed": False,
        "objective_changed": False,
        "margin_changed": False,
        "threshold_or_bias_changed": False,
        "candidate_checkpoint_write": False,
    }


def _read_json(path: Path, *, name: str) -> dict[str, object]:
    if not path.is_file():
        _fail(f"{name} missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MeterV5_3CError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        _fail(f"{name} root is not an object")
    return value


def _read_exact_v5_3a_evidence_v1(
    *, report_path: Path, recovery_envelope_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    if v52b._sha_file(report_path) != V53A_REPORT_SHA256:
        _fail("V5-3A report SHA256 mismatch")
    if v52b._sha_file(recovery_envelope_path) != V53A_RECOVERY_ENVELOPE_SHA256:
        _fail("V5-3A recovery envelope SHA256 mismatch")
    report = _read_json(report_path, name="V5-3A report")
    envelope = _read_json(
        recovery_envelope_path, name="V5-3A recovery envelope"
    )
    if report.get("schema") != v53a.SCHEMA:
        _fail("V5-3A report schema mismatch")
    if report.get("candidate_selection_gate") != "HOLD":
        _fail("V5-3A report is not the exact HOLD execution")
    if report.get("candidate_checkpoint_written") is not False:
        _fail("V5-3A HOLD report claims candidate checkpoint write")
    if report.get("model_parameter_mutation_executed") is not False:
        _fail("V5-3A HOLD report claims model parameter mutation")
    digit3 = report.get("per_specialist", {}).get("3", {}).get("fit", {})
    if digit3.get("candidate_claim") != (
        "SECONDARY_UNPROVEN_WITNESS_RESIDUAL_FAILED"
    ):
        _fail("V5-3A 3-AI residual claim changed")
    if digit3.get("primary_l1_cap_violations") != 1:
        _fail("V5-3A 3-AI sole L1-cap violation changed")
    if envelope.get("source_report_sha256_before") != V53A_REPORT_SHA256:
        _fail("recovery envelope report binding mismatch")
    if envelope.get("source_report_sha256_after") != V53A_REPORT_SHA256:
        _fail("recovery envelope did not preserve the source report")
    if envelope.get("candidate_fit_reexecuted") is not False:
        _fail("recovery envelope claims candidate fit rerun")
    if envelope.get("candidate_checkpoint_written") is not False:
        _fail("recovery envelope claims candidate checkpoint write")
    return report, envelope


def solve_guarded_secondary_witness_v1(
    *,
    historical_features,
    historical_targets,
    v5_features,
    v5_targets,
    frozen_weight,
    frozen_bias: float,
    threshold: float,
    exact_primary_l1_optimum: float,
    exact_external_l1_cap: float,
) -> tuple[dict[str, object], object | None]:
    """Solve the single normalized, guarded secondary LP and verify it."""
    np, linprog = v53a._numpy_modules()
    hist_x, hist_y = v53a._as_surface(
        historical_features, historical_targets, name="historical"
    )
    v5_x, v5_y = v53a._as_surface(v5_features, v5_targets, name="v5")
    w0 = v53a._to_numpy(frozen_weight, name="frozen weight").reshape(-1)
    if w0.shape != (EXPECTED_FEATURE_DIM,):
        _fail(f"frozen weight shape changed: {w0.shape}")
    bias = float(frozen_bias)
    primary_l1 = float(exact_primary_l1_optimum)
    external_cap = float(exact_external_l1_cap)
    for name, value in (
        ("frozen bias", bias),
        ("exact primary L1 optimum", primary_l1),
        ("exact external L1 cap", external_cap),
    ):
        if not math.isfinite(value):
            _fail(f"{name} is non-finite")
    if primary_l1 < 0.0:
        _fail("exact primary L1 optimum is negative")
    expected_cap = primary_l1 + PRIMARY_L1_ABSOLUTE_SLACK
    if abs(external_cap - expected_cap) > 1e-12:
        _fail("exact external L1 cap does not match V5-3A")
    internal_cap = external_cap - INTERNAL_CAP_GUARD
    if internal_cap <= primary_l1 or internal_cap <= 0.0:
        _fail("exact external L1 cap leaves no guarded secondary surface")

    boundary = v53a.v52r._threshold_logit(float(threshold))
    surface = v53a._decision_surface_v1(
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
    variable_count = 2 * dimension + 1
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[-1] = 1.0

    decision_a = np.zeros((decision_rows.shape[0], variable_count), dtype=np.float64)
    decision_a[:, :dimension] = decision_rows
    upper_abs = np.zeros((dimension, variable_count), dtype=np.float64)
    upper_abs[:, :dimension] = np.eye(dimension, dtype=np.float64)
    upper_abs[:, dimension : 2 * dimension] = -np.eye(
        dimension, dtype=np.float64
    )
    lower_abs = np.zeros((dimension, variable_count), dtype=np.float64)
    lower_abs[:, :dimension] = -np.eye(dimension, dtype=np.float64)
    lower_abs[:, dimension : 2 * dimension] = -np.eye(
        dimension, dtype=np.float64
    )
    normalized_l1_cap = np.zeros((1, variable_count), dtype=np.float64)
    normalized_l1_cap[0, dimension : 2 * dimension] = 1.0 / internal_cap
    upper_linf = np.zeros((dimension, variable_count), dtype=np.float64)
    upper_linf[:, :dimension] = np.eye(dimension, dtype=np.float64)
    upper_linf[:, -1] = -1.0
    lower_linf = np.zeros((dimension, variable_count), dtype=np.float64)
    lower_linf[:, :dimension] = -np.eye(dimension, dtype=np.float64)
    lower_linf[:, -1] = -1.0
    a_ub = np.concatenate(
        (decision_a, upper_abs, lower_abs, normalized_l1_cap, upper_linf, lower_linf),
        axis=0,
    )
    b_ub = np.concatenate(
        (
            decision_bounds,
            np.zeros(2 * dimension, dtype=np.float64),
            np.ones(1, dtype=np.float64),
            np.zeros(2 * dimension, dtype=np.float64),
        )
    )
    solved = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=(
            [(None, None)] * dimension
            + [(0.0, None)] * dimension
            + [(0.0, None)]
        ),
        method=v53a.SOLVER_METHOD,
        options=v53a._linprog_options(),
    )
    report: dict[str, object] = {
        "witness_claim": v53a._solver_stage_claim(solved, stage="GUARDED_SECONDARY"),
        "witness_verified": False,
        "status": int(solved.status),
        "success": bool(solved.success),
        "iterations": int(getattr(solved, "nit", 0)),
        "message": str(solved.message),
        "optimality_claim": (
            "SOLVER_REPORTED_OPTIMAL_NOT_FORMAL_PROOF"
            if int(solved.status) == 0 and bool(solved.success)
            else "OPTIMALITY_NOT_PROVEN"
        ),
        "exact_primary_l1_optimum": primary_l1,
        "external_l1_acceptance_cap": external_cap,
        "internal_guarded_l1_cap": internal_cap,
        "internal_cap_guard": INTERNAL_CAP_GUARD,
        "normalized_l1_cap_rhs": 1.0,
        "v5_constraint_count": int(v5_x.shape[0]),
        "historical_all_count": int(hist_x.shape[0]),
        "historical_frozen_correct_count": int(np.sum(surface["hist_correct"])),
        "fixed_bias": bias,
        "frozen_threshold": float(threshold),
        "witness_weight_values_emitted": False,
        "candidate_checkpoint_written": False,
    }
    if not (int(solved.status) == 0 and bool(solved.success)):
        return report, None

    solution = np.asarray(solved.x, dtype=np.float64)
    if not bool(np.isfinite(solution).all()):
        _fail("guarded secondary witness is non-finite")
    delta = solution[:dimension]
    witness = w0 + delta
    solver_linf = float(solved.fun)
    recomputed_l1 = float(np.sum(np.abs(delta)))
    recomputed_linf = float(np.max(np.abs(delta)))
    external_cap_excess = recomputed_l1 - external_cap
    internal_cap_excess = recomputed_l1 - internal_cap
    external_cap_violations = int(external_cap_excess > WITNESS_TOLERANCE)
    internal_cap_violations = int(internal_cap_excess > WITNESS_TOLERANCE)
    lower_bound_conflicts = int(
        recomputed_l1 < primary_l1 - WITNESS_TOLERANCE
    )
    parameter_bound_violations = int(
        np.sum(np.abs(delta) > solver_linf + WITNESS_TOLERANCE)
    )

    diagnostic_hist_logit = hist_x @ witness + bias
    diagnostic_v5_logit = v5_x @ witness + bias
    direct_hist_delta = hist_x @ delta
    recomputed_hist_delta = diagnostic_hist_logit - surface["frozen_hist_logit"]
    identity_error = float(
        np.max(np.abs(direct_hist_delta - recomputed_hist_delta))
    )
    v5_margin = surface["v5_sign"] * (diagnostic_v5_logit - boundary)
    retained_margin = surface["retained_sign"] * (
        diagnostic_hist_logit[surface["hist_correct"]] - boundary
    )
    v5_violations = int(
        np.sum(v5_margin < surface["v5_required_margin"] - WITNESS_TOLERANCE)
    )
    v5_solver_violations = int(
        np.sum(v5_margin < surface["v5_solver_margin"] - WITNESS_TOLERANCE)
    )
    historical_violations = int(
        np.sum(
            retained_margin
            < surface["retained_required_margin"] - WITNESS_TOLERANCE
        )
    )
    historical_solver_violations = int(
        np.sum(
            retained_margin
            < surface["retained_solver_margin"] - WITNESS_TOLERANCE
        )
    )
    diagnostic_hist_prediction = diagnostic_hist_logit >= boundary
    diagnostic_v5_prediction = diagnostic_v5_logit >= boundary
    transitions = v52x._transition_counts(
        frozen_prediction=surface["frozen_hist_prediction"],
        diagnostic_prediction=diagnostic_hist_prediction,
        targets=hist_y,
    )
    v5_metrics = v52x._classification_metrics(
        prediction=diagnostic_v5_prediction, targets=v5_y
    )
    report.update(
        {
            "minimum_delta_weight_linf": solver_linf,
            "independently_recomputed_delta_weight_l1": recomputed_l1,
            "independently_recomputed_delta_weight_linf": recomputed_linf,
            "secondary_objective_recomputation_absolute_error": abs(
                solver_linf - recomputed_linf
            ),
            "external_l1_cap_excess": external_cap_excess,
            "internal_guarded_l1_cap_excess": internal_cap_excess,
            "external_l1_cap_violations": external_cap_violations,
            "internal_guarded_l1_cap_violations": internal_cap_violations,
            "primary_lower_bound_conflicts": lower_bound_conflicts,
            "parameter_bound_violations": parameter_bound_violations,
            "minimum_v5_signed_decision_margin": float(np.min(v5_margin)),
            "minimum_historical_retained_signed_decision_margin": float(
                np.min(retained_margin)
            ),
            "v5_constraint_violations": v5_violations,
            "v5_solver_margin_constraint_violations": v5_solver_violations,
            "historical_margin_constraint_violations": historical_violations,
            "historical_solver_margin_constraint_violations": (
                historical_solver_violations
            ),
            "functional_delta_identity_max_abs_error": identity_error,
            "functional_delta_identity_verified": (
                identity_error <= IDENTITY_TOLERANCE
            ),
            "historical_absolute_logit_drift": v52x._quantile_summary(
                np.abs(direct_hist_delta), name="historical absolute logit drift"
            ),
            "weight_geometry": v52y._weight_geometry(
                frozen_weight=w0, diagnostic_weight=witness
            ),
            "diagnostic_v5_train_metrics": v5_metrics,
            "diagnostic_historical_train_metrics": v52x._classification_metrics(
                prediction=diagnostic_hist_prediction, targets=hist_y
            ),
            "historical_transition_counts": transitions,
        }
    )
    verified = (
        external_cap_violations == 0
        and internal_cap_violations == 0
        and lower_bound_conflicts == 0
        and parameter_bound_violations == 0
        and v5_violations == 0
        and v5_solver_violations == 0
        and historical_violations == 0
        and historical_solver_violations == 0
        and identity_error <= IDENTITY_TOLERANCE
        and abs(solver_linf - recomputed_linf) <= WITNESS_TOLERANCE
        and transitions["correct_to_wrong"] == 0
        and v5_metrics["f1"] == 1.0
    )
    if not verified:
        report["witness_claim"] = "GUARDED_SECONDARY_WITNESS_RESIDUAL_FAILED"
        return report, None
    report["witness_claim"] = "GUARDED_SECONDARY_WITNESS_VERIFIED"
    report["witness_verified"] = True
    return report, witness


def run_guarded_secondary_witness_audit_v1(
    data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    v5_3a_report: str | Path,
    v5_3a_recovery_envelope: str | Path,
    confirmation: str,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run one exact diagnostic witness audit without publishing a model."""
    if confirmation != APPROVAL_TOKEN:
        _fail("exact V5-3C approval token missing")
    root = Path(data_root)
    report_path = root / v51.ANNOTATIONS_DIR / REPORT_NAME
    if report_path.exists():
        _fail(f"refusing overwrite/rerun: {report_path}")
    prior, _recovery = _read_exact_v5_3a_evidence_v1(
        report_path=Path(v5_3a_report),
        recovery_envelope_path=Path(v5_3a_recovery_envelope),
    )
    models = v52n._frozen_models(
        digit2_frozen=Path(digit2_frozen), digit3_frozen=Path(digit3_frozen)
    )
    manifest_path, _rows, v5_features, v5_targets, _metrics = v52n._v5_surface(
        root, models
    )
    manifest_sha = v52b._sha_file(manifest_path)
    if prior.get("slot_manifest_sha256") != manifest_sha:
        _fail("V5 TRAIN slot manifest changed after V5-3A")
    historical_features, historical_targets = v52n._historical_surface(
        m4a_root=Path(m4a_root),
        d10_root=Path(d10_root),
        models=models,
        progress=progress,
    )
    per_specialist: dict[str, dict[str, object]] = {}
    all_verified = True
    for digit in ("2", "3"):
        prior_fit = prior["per_specialist"][digit]["fit"]
        model = models[digit]
        frozen_weight = model.head.weight.detach().cpu().reshape(-1)
        frozen_bias = float(model.head.bias.detach().cpu().reshape(-1)[0].item())
        result, witness = solve_guarded_secondary_witness_v1(
            historical_features=historical_features[digit],
            historical_targets=historical_targets[digit],
            v5_features=v5_features[digit],
            v5_targets=v5_targets[digit],
            frozen_weight=frozen_weight,
            frozen_bias=frozen_bias,
            threshold=v52b.FROZEN_THRESHOLDS[digit],
            exact_primary_l1_optimum=prior_fit["minimum_delta_weight_l1"],
            exact_external_l1_cap=prior_fit["primary_l1_cap"],
        )
        specialist: dict[str, object] = {"guarded_secondary": result}
        if witness is None or result.get("witness_claim") != (
            "GUARDED_SECONDARY_WITNESS_VERIFIED"
        ):
            all_verified = False
        else:
            float32_gate = v53a._verify_float32_copy_v1(
                candidate_weight=witness,
                frozen_weight=frozen_weight,
                frozen_bias=frozen_bias,
                threshold=v52b.FROZEN_THRESHOLDS[digit],
                historical_features=historical_features[digit],
                historical_targets=historical_targets[digit],
                v5_features=v5_features[digit],
                v5_targets=v5_targets[digit],
            )
            specialist["float32_copy_gate"] = float32_gate
            if float32_gate.get("gate") != "PASS":
                all_verified = False
        per_specialist[digit] = specialist

    contract = solver_contract()
    if contract["library_version_matches_expected"] is not True:
        _fail("SciPy runtime version does not match the pinned solver contract")
    report: dict[str, object] = {
        "schema": SCHEMA,
        "exact_v5_3a_binding": {
            "source_implementation_head": V53A_SOURCE_IMPLEMENTATION_HEAD,
            "source_harness_head": V53A_SOURCE_HARNESS_HEAD,
            "report_sha256": V53A_REPORT_SHA256,
            "recovery_envelope_sha256": V53A_RECOVERY_ENVELOPE_SHA256,
        },
        "observed_v5_3a_3ai_cap_excess": OBSERVED_3AI_CAP_EXCESS,
        "solver_contract": contract,
        "slot_manifest_sha256": manifest_sha,
        "per_specialist": per_specialist,
        "diagnostic_witness_gate": "PASS" if all_verified else "HOLD",
        "witness_weight_values_emitted": False,
        **safety_boundary(),
    }
    v51._atomic_write_json(report_path, report)
    return report


def historical_retention_executed_by_this_module() -> bool:
    return False


def validation_opened_by_this_module() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False
