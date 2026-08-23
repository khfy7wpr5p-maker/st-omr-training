"""Numerical evidence hardening for the preregistered Meter V5-2P repair.

This module does not change the V5-2P architecture, objective, solver settings,
data surfaces, thresholds, or performance-gate order. It instruments the exact
existing LBFGS solve, records termination state, independently recomputes the
final objective gradient, verifies float64->float32 copy-back, and stops before
historical retention. Numerical integrity is a safety check only; convergence
status is evidence and does not create a new performance gate. Historical
retention remains the only evidence that can establish source-domain
preservation at the unchanged runtime thresholds.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Final, Mapping

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2n_frozen_feature_transfer_audit_v1 as v52n
from . import meter_v5_2p_fixed_bias_head_repair_v1 as v52p


SCHEMA: Final[str] = "st-omr-meter-v5-2p-numerical-evidence-guard-v1"
REPORT_NAME: Final[str] = "v5_2p_numerical_evidence_guard_v1.json"
LOSS_NON_INCREASE_TOLERANCE: Final[float] = 1e-10
ProgressCallback = Callable[[int, int, str], None]

_REQUIRED_SPECIALIST_EVIDENCE: Final[tuple[str, ...]] = (
    "trainable_parameter_count",
    "only_head_weight_changed",
    "backbone_bit_identical",
    "head_bias_bit_identical",
    "threshold_unchanged",
    "solver_final_loss_finite",
    "solver_final_loss_not_above_initial",
    "float32_copy_back_bit_exact",
    "float32_copy_back_loss_finite",
    "float32_copy_back_loss_not_above_initial",
    "lbfgs_termination",
)
_REQUIRED_CONVERGENCE_EVIDENCE: Final[tuple[str, ...]] = (
    "final_gradient_inf_norm",
    "final_gradient_l2_norm",
    "final_gradient_finite",
    "gradient_tolerance_met",
    "n_iter",
    "func_evals",
    "closure_evaluations",
    "iteration_limit_reached",
    "evaluation_limit_reached",
    "termination_reason_exposed",
    "termination_evidence_class",
    "convergence_proven",
    "convergence_claim",
)


class MeterV5_2PNumericalEvidenceError(RuntimeError):
    """Raised when V5-2P numerical evidence cannot be established safely."""


def _fail(message: str) -> None:
    raise MeterV5_2PNumericalEvidenceError(message)


def evidence_contract() -> dict[str, object]:
    """Declare that this layer is evidence-only and does not alter V5-2P."""
    return {
        "architecture_changed": False,
        "objective_changed": False,
        "solver_settings_changed": False,
        "data_surfaces_changed": False,
        "thresholds_changed": False,
        "performance_gate_order_changed": False,
        "v5_2p_objective_contract": v52p.objective_contract(),
        "v5_2p_solver_contract": v52p.solver_contract(),
        "v5_2p_performance_gate_order": list(v52p.gate_order()),
        "instrumentation_only": True,
        "numerical_integrity_is_safety_gate_only": True,
        "convergence_evidence_is_performance_gate": False,
        "convergence_unproven_creates_integrity_hold": False,
        "retention_executed_by_this_module": False,
        "first30_executed_by_this_module": False,
    }


def _termination_evidence_v1(
    *,
    n_iter: int,
    func_evals: int,
    closure_evaluations: int,
    final_gradient_inf_norm: float,
    final_gradient_l2_norm: float,
) -> dict[str, object]:
    if n_iter < 0 or func_evals < 0 or closure_evaluations < 0:
        _fail("negative LBFGS state counter")

    gradient_finite = (
        math.isfinite(final_gradient_inf_norm)
        and math.isfinite(final_gradient_l2_norm)
        and final_gradient_inf_norm >= 0.0
        and final_gradient_l2_norm >= 0.0
    )
    gradient_tolerance_met = (
        gradient_finite and final_gradient_inf_norm <= v52p.LBFGS_TOLERANCE_GRAD
    )
    iteration_limit_reached = n_iter >= v52p.LBFGS_MAX_ITER
    evaluation_limit_reached = func_evals >= v52p.LBFGS_MAX_EVAL
    terminated_before_limits = not iteration_limit_reached and not evaluation_limit_reached

    if not gradient_finite:
        evidence_class = "FINAL_GRADIENT_NONFINITE_CONVERGENCE_NOT_PROVEN"
        convergence_proven = False
    elif gradient_tolerance_met:
        evidence_class = "PROVEN_FINAL_GRADIENT_TOLERANCE"
        convergence_proven = True
    elif iteration_limit_reached:
        evidence_class = "MAX_ITER_REACHED_TERMINATION_REASON_NOT_EXPOSED_CONVERGENCE_NOT_PROVEN"
        convergence_proven = False
    elif evaluation_limit_reached:
        evidence_class = "MAX_EVAL_REACHED_TERMINATION_REASON_NOT_EXPOSED_CONVERGENCE_NOT_PROVEN"
        convergence_proven = False
    else:
        # torch.optim.LBFGS does not expose a stable public termination-reason
        # field. Do not infer tolerance_change convergence from private state.
        evidence_class = "TERMINATED_BEFORE_LIMIT_REASON_NOT_EXPOSED_CONVERGENCE_NOT_PROVEN"
        convergence_proven = False

    return {
        "n_iter": int(n_iter),
        "func_evals": int(func_evals),
        "closure_evaluations": int(closure_evaluations),
        "closure_evaluations_match_func_evals": int(closure_evaluations) == int(func_evals),
        "max_iter": v52p.LBFGS_MAX_ITER,
        "max_eval": v52p.LBFGS_MAX_EVAL,
        "tolerance_grad": v52p.LBFGS_TOLERANCE_GRAD,
        "tolerance_change": v52p.LBFGS_TOLERANCE_CHANGE,
        "final_gradient_inf_norm": float(final_gradient_inf_norm),
        "final_gradient_l2_norm": float(final_gradient_l2_norm),
        "final_gradient_finite": gradient_finite,
        "gradient_tolerance_met": gradient_tolerance_met,
        "iteration_limit_reached": iteration_limit_reached,
        "evaluation_limit_reached": evaluation_limit_reached,
        "terminated_before_limits": terminated_before_limits,
        "termination_reason_exposed": False,
        "termination_reason_exposed_by_torch_lbfgs": False,
        "termination_evidence_class": evidence_class,
        "convergence_proven": convergence_proven,
        "convergence_claim": "PROVEN" if convergence_proven else "UNPROVEN",
    }


def _convergence_evidence_v1(
    per_specialist: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for digit in ("2", "3"):
        item = per_specialist.get(digit)
        termination = item.get("lbfgs_termination") if isinstance(item, Mapping) else None
        if not isinstance(termination, Mapping):
            result[digit] = {
                "evidence_present": False,
                "convergence_proven": False,
                "convergence_claim": "UNPROVEN",
                "termination_reason_exposed": False,
                "termination_evidence_class": "MISSING_CONVERGENCE_EVIDENCE",
            }
            continue
        result[digit] = {
            "evidence_present": all(name in termination for name in _REQUIRED_CONVERGENCE_EVIDENCE),
            **{name: termination.get(name) for name in termination},
        }
    return result


def _numerical_integrity_gate_v1(
    per_specialist: Mapping[str, Mapping[str, object]],
    *,
    observed_lbfgs_solves: int = 2,
) -> dict[str, object]:
    reasons: list[str] = []
    if observed_lbfgs_solves != 2:
        reasons.append(f"LBFGS_CAPTURE_COUNT_MISMATCH:{observed_lbfgs_solves}")

    for digit in ("2", "3"):
        item = per_specialist.get(digit)
        if not isinstance(item, Mapping):
            reasons.append(f"{digit}-AI_EVIDENCE_MISSING")
            continue

        missing = [name for name in _REQUIRED_SPECIALIST_EVIDENCE if name not in item]
        if missing:
            reasons.append(f"{digit}-AI_REQUIRED_EVIDENCE_MISSING:{','.join(missing)}")

        termination = item.get("lbfgs_termination")
        if not isinstance(termination, Mapping):
            reasons.append(f"{digit}-AI_CONVERGENCE_EVIDENCE_MISSING")
        else:
            missing_termination = [
                name for name in _REQUIRED_CONVERGENCE_EVIDENCE if name not in termination
            ]
            if missing_termination:
                reasons.append(
                    f"{digit}-AI_REQUIRED_CONVERGENCE_EVIDENCE_MISSING:"
                    f"{','.join(missing_termination)}"
                )
            if termination.get("final_gradient_finite") is not True:
                reasons.append(f"{digit}-AI_FINAL_GRADIENT_NONFINITE")

        if item.get("trainable_parameter_count") != v52p.EXPECTED_FEATURE_DIM:
            reasons.append(f"{digit}-AI_TRAINABLE_PARAMETER_COUNT_CHANGED")
        if item.get("only_head_weight_changed") is not True:
            reasons.append(f"{digit}-AI_ILLEGAL_STATE_MUTATION")
        if item.get("backbone_bit_identical") is not True:
            reasons.append(f"{digit}-AI_BACKBONE_NOT_BIT_IDENTICAL")
        if item.get("head_bias_bit_identical") is not True:
            reasons.append(f"{digit}-AI_HEAD_BIAS_NOT_BIT_IDENTICAL")
        if item.get("threshold_unchanged") is not True:
            reasons.append(f"{digit}-AI_THRESHOLD_CHANGED")
        if item.get("solver_final_loss_finite") is not True:
            reasons.append(f"{digit}-AI_SOLVER_FINAL_LOSS_NONFINITE")
        if item.get("solver_final_loss_not_above_initial") is not True:
            reasons.append(f"{digit}-AI_SOLVER_FINAL_LOSS_INCREASED")
        if item.get("float32_copy_back_bit_exact") is not True:
            reasons.append(f"{digit}-AI_FLOAT32_COPY_BACK_NOT_EXACT")
        if item.get("float32_copy_back_loss_finite") is not True:
            reasons.append(f"{digit}-AI_FLOAT32_COPY_BACK_LOSS_NONFINITE")
        if item.get("float32_copy_back_loss_not_above_initial") is not True:
            reasons.append(f"{digit}-AI_FLOAT32_COPY_BACK_LOSS_INCREASED")

    return {
        "gate": "PASS" if not reasons else "HOLD",
        "reasons": reasons,
        "historical_retention_authorized_after_separate_review": not reasons,
        "historical_preservation_claimed": False,
    }


def _guard_decision_v1(
    per_specialist: Mapping[str, Mapping[str, object]],
    *,
    observed_lbfgs_solves: int = 2,
) -> dict[str, object]:
    """Separate safety integrity from non-gating convergence evidence."""
    integrity = _numerical_integrity_gate_v1(
        per_specialist,
        observed_lbfgs_solves=observed_lbfgs_solves,
    )
    convergence = _convergence_evidence_v1(per_specialist)
    return {
        "numerical_integrity_gate": integrity,
        "convergence_evidence": convergence,
        # Compatibility aliases intentionally reflect integrity only. An
        # UNPROVEN convergence claim does not add a HOLD reason.
        "gate": integrity["gate"],
        "reasons": integrity["reasons"],
        "historical_retention_authorized_after_separate_review": integrity[
            "historical_retention_authorized_after_separate_review"
        ],
        "historical_preservation_claimed": False,
    }


def _capture_lbfgs_steps(torch):
    """Install transparent LBFGS.step instrumentation and return capture state."""
    original_step = torch.optim.LBFGS.step
    captures: list[dict[str, object]] = []

    def instrumented_step(optimizer, closure):
        result = original_step(optimizer, closure)
        groups = optimizer.param_groups
        if len(groups) != 1 or len(groups[0].get("params", [])) != 1:
            _fail("V5-2P LBFGS trainable surface changed during instrumentation")
        parameter = groups[0]["params"][0]
        if int(parameter.numel()) != v52p.EXPECTED_FEATURE_DIM:
            _fail("V5-2P LBFGS parameter count changed during instrumentation")
        state = optimizer.state.get(parameter, {})
        captures.append({
            "final_weight_float64": parameter.detach().cpu().clone(),
            "n_iter": int(state.get("n_iter", 0)),
            "func_evals": int(state.get("func_evals", 0)),
            "optimizer_state_keys": sorted(str(key) for key in state.keys()),
        })
        return result

    torch.optim.LBFGS.step = instrumented_step
    return original_step, captures


def _restore_lbfgs_step(torch, original_step) -> None:
    torch.optim.LBFGS.step = original_step


def _evaluate_weight_state_v1(
    *,
    torch,
    digit: str,
    captured: Mapping[str, object],
    fit: Mapping[str, object],
    frozen_model,
    candidate_model,
    v5_features,
    v5_targets,
    historical_features,
    historical_targets,
) -> dict[str, object]:
    final_weight64 = captured.get("final_weight_float64")
    if final_weight64 is None or tuple(final_weight64.shape) != (v52p.EXPECTED_FEATURE_DIM,):
        _fail(f"{digit}-AI captured float64 weight shape changed")
    if final_weight64.dtype != torch.float64:
        _fail(f"{digit}-AI LBFGS weight is not float64")

    x_v5 = v5_features.detach().cpu().to(dtype=torch.float64)
    y_v5 = v5_targets.detach().cpu().to(dtype=torch.float64).reshape(-1)
    x_hist = historical_features.detach().cpu().to(dtype=torch.float64)
    y_hist = historical_targets.detach().cpu().to(dtype=torch.float64).reshape(-1)
    frozen_bias = float(frozen_model.head.bias.detach().cpu().reshape(-1)[0].item())

    weight_for_grad = torch.nn.Parameter(final_weight64.detach().clone())
    total, v5_loss, historical_loss = v52p._balanced_domain_bce_v1(
        v5_logits=x_v5 @ weight_for_grad + frozen_bias,
        v5_targets=y_v5,
        historical_logits=x_hist @ weight_for_grad + frozen_bias,
        historical_targets=y_hist,
    )
    total.backward()
    if weight_for_grad.grad is None:
        _fail(f"{digit}-AI final objective gradient missing")
    final_grad_l2 = float(torch.linalg.vector_norm(weight_for_grad.grad).item())
    final_grad_inf = float(torch.max(torch.abs(weight_for_grad.grad)).item())
    final_gradient_finite = bool(torch.isfinite(weight_for_grad.grad).all().item())

    reevaluated_total = float(total.detach().item())
    fit_final = float(fit.get("final_total_loss"))
    initial_total = float(fit.get("initial_total_loss"))
    if not all(math.isfinite(value) for value in (reevaluated_total, fit_final, initial_total)):
        _fail(f"{digit}-AI solver loss evidence non-finite")
    if not math.isclose(reevaluated_total, fit_final, rel_tol=1e-10, abs_tol=1e-12):
        _fail(f"{digit}-AI final float64 objective does not reproduce fit report")
    solver_not_above_initial = reevaluated_total <= initial_total + LOSS_NON_INCREASE_TOLERANCE
    if not solver_not_above_initial:
        _fail(f"{digit}-AI final float64 objective increased above initial")

    candidate_weight32 = candidate_model.head.weight.detach().cpu().reshape(-1)
    if candidate_weight32.dtype != torch.float32:
        _fail(f"{digit}-AI candidate head.weight is not float32")
    expected_copy32 = final_weight64.to(dtype=torch.float32)
    copy_back_bit_exact = torch.equal(candidate_weight32, expected_copy32)
    if not copy_back_bit_exact:
        _fail(f"{digit}-AI float32 copy-back differs from exact cast")

    quantization_delta64 = expected_copy32.to(dtype=torch.float64) - final_weight64
    copy_weight64 = expected_copy32.to(dtype=torch.float64)
    with torch.no_grad():
        copy_total, copy_v5, copy_hist = v52p._balanced_domain_bce_v1(
            v5_logits=x_v5 @ copy_weight64 + frozen_bias,
            v5_targets=y_v5,
            historical_logits=x_hist @ copy_weight64 + frozen_bias,
            historical_targets=y_hist,
        )
    copy_total_value = float(copy_total.item())
    if not math.isfinite(copy_total_value):
        _fail(f"{digit}-AI float32 copy-back objective non-finite")
    copy_not_above_initial = copy_total_value <= initial_total + LOSS_NON_INCREASE_TOLERANCE
    if not copy_not_above_initial:
        _fail(f"{digit}-AI float32 copy-back objective increased above initial")

    frozen_state = frozen_model.state_dict()
    candidate_state = candidate_model.state_dict()
    if set(frozen_state) != set(candidate_state):
        _fail(f"{digit}-AI candidate/frozen state keys differ")
    changed_keys = [
        name for name in sorted(candidate_state)
        if not torch.equal(
            frozen_state[name].detach().cpu(),
            candidate_state[name].detach().cpu(),
        )
    ]
    illegal = [name for name in changed_keys if name != "head.weight"]
    backbone_bit_identical = all(
        torch.equal(
            frozen_state[name].detach().cpu(),
            candidate_state[name].detach().cpu(),
        )
        for name in candidate_state if name.startswith("features.")
    )
    head_bias_bit_identical = torch.equal(
        frozen_state["head.bias"].detach().cpu(),
        candidate_state["head.bias"].detach().cpu(),
    )
    if illegal or not backbone_bit_identical or not head_bias_bit_identical:
        _fail(f"{digit}-AI frozen-state integrity failed")

    termination = _termination_evidence_v1(
        n_iter=int(captured.get("n_iter", 0)),
        func_evals=int(captured.get("func_evals", 0)),
        closure_evaluations=int(fit.get("closure_evaluations", 0)),
        final_gradient_inf_norm=final_grad_inf,
        final_gradient_l2_norm=final_grad_l2,
    )
    if termination["final_gradient_finite"] != final_gradient_finite:
        _fail(f"{digit}-AI final gradient finiteness evidence mismatch")

    return {
        "trainable_surface": "head.weight-only-64-parameters",
        "trainable_parameter_count": int(candidate_weight32.numel()),
        "changed_state_keys": changed_keys,
        "only_head_weight_changed": not illegal,
        "backbone_bit_identical": backbone_bit_identical,
        "head_bias_bit_identical": head_bias_bit_identical,
        "threshold": v52b.FROZEN_THRESHOLDS[digit],
        "threshold_unchanged": True,
        "solver_final_loss_finite": math.isfinite(reevaluated_total),
        "solver_final_loss_not_above_initial": solver_not_above_initial,
        "initial_total_loss": initial_total,
        "solver_final_total_loss_reported": fit_final,
        "solver_final_total_loss_reevaluated_float64": reevaluated_total,
        "solver_final_v5_mean_bce_reevaluated_float64": float(v5_loss.detach().item()),
        "solver_final_historical_mean_bce_reevaluated_float64": float(historical_loss.detach().item()),
        "float32_copy_back_bit_exact": copy_back_bit_exact,
        "float32_copy_back_weight_max_abs_error": float(torch.max(torch.abs(quantization_delta64)).item()),
        "float32_copy_back_weight_l2_error": float(torch.linalg.vector_norm(quantization_delta64).item()),
        "float32_copy_back_total_loss": copy_total_value,
        "float32_copy_back_v5_mean_bce": float(copy_v5.item()),
        "float32_copy_back_historical_mean_bce": float(copy_hist.item()),
        "float32_copy_back_loss_delta_vs_solver_float64": copy_total_value - reevaluated_total,
        "float32_copy_back_loss_finite": math.isfinite(copy_total_value),
        "float32_copy_back_loss_not_above_initial": copy_not_above_initial,
        "lbfgs_termination": termination,
        "optimizer_state_keys_observed": list(captured.get("optimizer_state_keys", [])),
    }


def _base_report_v1(
    *,
    ann: Path,
    per_specialist: Mapping[str, Mapping[str, object]],
    decision: Mapping[str, object],
    observed_lbfgs_solves: int,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "base_training_schema": v52p.SCHEMA,
        "base_training_report_sha256": v52b._sha_file(ann / v52p.TRAINING_REPORT_NAME),
        "evidence_contract": evidence_contract(),
        "per_specialist": dict(per_specialist),
        "numerical_integrity_gate": decision["numerical_integrity_gate"],
        "convergence_evidence": decision["convergence_evidence"],
        "numerical_guard": decision,
        "observed_lbfgs_solves": observed_lbfgs_solves,
        "expected_lbfgs_solves": 2,
        "evidence_autograd_used_for_final_gradient_only": True,
        "evidence_optimizer_steps": 0,
        "training_architecture_changed": False,
        "objective_changed": False,
        "solver_settings_changed": False,
        "thresholds_changed": False,
        "historical_retention_executed": False,
        "first30_diagnostic_executed": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "production_promotion_authorized": False,
        "historical_preservation_claimed": False,
    }


def train_with_numerical_evidence_guard_v1(
    data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    confirmation: str,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run the exact V5-2P solve once, then stop after numerical evidence.

    This function deliberately does not run historical retention or first-30.
    A later call may proceed only after the numerical integrity report is
    reviewed. An UNPROVEN convergence claim alone does not block retention.
    """
    root = Path(data_root)
    ann = root / v51.ANNOTATIONS_DIR
    evidence_path = ann / REPORT_NAME
    if evidence_path.exists():
        _fail(f"refusing to overwrite V5-2P numerical evidence: {evidence_path}")

    torch, _nn = v52b._import_torch()
    original_step, captures = _capture_lbfgs_steps(torch)
    try:
        training = v52p.train_fixed_bias_head_repair_v1(
            root,
            m4a_root=m4a_root,
            d10_root=d10_root,
            digit2_frozen=digit2_frozen,
            digit3_frozen=digit3_frozen,
            confirmation=confirmation,
            progress=progress,
        )
    finally:
        _restore_lbfgs_step(torch, original_step)

    if len(captures) != 2:
        decision = _guard_decision_v1({}, observed_lbfgs_solves=len(captures))
        report = _base_report_v1(
            ann=ann,
            per_specialist={},
            decision=decision,
            observed_lbfgs_solves=len(captures),
        )
        v51._atomic_write_json(evidence_path, report)
        return report

    frozen_models = v52n._frozen_models(
        digit2_frozen=Path(digit2_frozen),
        digit3_frozen=Path(digit3_frozen),
    )
    _manifest_path, _rows, v5_features, v5_targets, _metrics = v52n._v5_surface(
        root, frozen_models
    )
    historical_features, historical_targets = v52n._historical_surface(
        m4a_root=Path(m4a_root),
        d10_root=Path(d10_root),
        models=frozen_models,
        progress=progress,
    )

    per_specialist: dict[str, dict[str, object]] = {}
    for index, digit in enumerate(("2", "3")):
        candidate_path = Path(training["candidates"][digit]["candidate_path"])
        candidate = v52p._load_candidate(
            candidate_path,
            digit=digit,
            training_report=training,
        )
        per_specialist[digit] = _evaluate_weight_state_v1(
            torch=torch,
            digit=digit,
            captured=captures[index],
            fit=training["candidates"][digit]["fit"],
            frozen_model=frozen_models[digit],
            candidate_model=candidate,
            v5_features=v5_features[digit],
            v5_targets=v5_targets[digit],
            historical_features=historical_features[digit],
            historical_targets=historical_targets[digit],
        )

    decision = _guard_decision_v1(per_specialist, observed_lbfgs_solves=len(captures))
    report = _base_report_v1(
        ann=ann,
        per_specialist=per_specialist,
        decision=decision,
        observed_lbfgs_solves=len(captures),
    )
    v51._atomic_write_json(evidence_path, report)
    return report


def historical_retention_executed_by_this_module() -> bool:
    return False


def validation_opened_by_this_module() -> bool:
    return False


def final_holdout_locked() -> bool:
    return True


def production_promotion_allowed() -> bool:
    return False
