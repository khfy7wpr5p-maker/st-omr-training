"""TRAIN-only functional logit drift audit for the rejected V5-2T heads.

V5-2T proved that a small weight-space angle does not preserve behavior: its
bounded candidates passed numerical/geometry checks and then failed Historical
Retention through large false-positive expansion.  V5-2V measures how the
small head-weight delta is amplified by frozen features into logit, margin, and
threshold-crossing changes on the already-open V5 TRAIN and historical TRAIN
surfaces.

The audit is descriptive and read-only.  It performs no fitting, autograd,
backward, optimizer step, threshold/bias selection, checkpoint mutation, or
per-example emission.  Historical validation error examples, First-30, V5 VAL,
and FINAL_HOLDOUT remain closed.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Callable, Final, Mapping

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2n_frozen_feature_transfer_audit_v1 as v52n
from . import meter_v5_2p_fixed_bias_head_repair_v1 as v52p
from . import meter_v5_2q_historical_positive_margin_audit_v1 as v52q
from . import meter_v5_2r_train_class_margin_gradient_audit_v1 as v52r
from . import meter_v5_2s_bounded_class_balanced_head_contract_v1 as v52s
from . import meter_v5_2t_bounded_class_balanced_head_repair_v1 as v52t
from . import meter_v5_2u_v5_2t_historical_retention_v1 as v52u


SCHEMA: Final[str] = "st-omr-meter-v5-2v-functional-logit-drift-audit-v1"
REPORT_NAME: Final[str] = "v5_2v_functional_logit_drift_audit_v1.json"
V52U_IMPLEMENTATION_HEAD: Final[str] = "55c56671fef326a96909e169ee440a22986ff71b"
V52U_RETENTION_REPORT_SHA256: Final[str] = (
    "6f072c99e4d6d60681a5c4739aecdb520327b1788b87c94f85c02583b343366f"
)
V52U_EXECUTION_ENVELOPE_SHA256: Final[str] = (
    "87fb3230d694798096e3ce501cfaf681c96ac92fcb6dc7fc510cf23d891a9135"
)
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


class MeterV5_2VError(RuntimeError):
    """Raised when V5-2V cannot prove its exact read-only audit contract."""


def _fail(message: str) -> None:
    raise MeterV5_2VError(message)


def safety_boundary() -> dict[str, object]:
    return {
        "training": False,
        "autograd_grad_used": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_read": True,
        "checkpoint_write": False,
        "candidate_checkpoint_mutation": False,
        "evidence_report_write": True,
        "objective_selected": False,
        "solver_selected": False,
        "classifier_fit": False,
        "threshold_tuning": False,
        "bias_tuning": False,
        "historical_validation_opened": False,
        "historical_validation_retention_report_read": True,
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


def _read_exact_evidence_v1(
    *,
    training_report: Path,
    training_envelope: Path,
    retention_report: Path,
    retention_envelope: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    training, _envelope = v52u._read_exact_execution_evidence_v1(
        training_report=training_report,
        execution_envelope=training_envelope,
    )
    if not retention_report.is_file() or not retention_envelope.is_file():
        _fail("exact V5-2U report/envelope missing")
    report_bytes = retention_report.read_bytes()
    envelope_bytes = retention_envelope.read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != V52U_RETENTION_REPORT_SHA256:
        _fail("V5-2U retention report SHA256 mismatch")
    if hashlib.sha256(envelope_bytes).hexdigest() != V52U_EXECUTION_ENVELOPE_SHA256:
        _fail("V5-2U execution envelope SHA256 mismatch")
    retention = v52b._read_json(retention_report)
    envelope = v52b._read_json(retention_envelope)
    if retention.get("schema") != v52u.SCHEMA or retention.get("gate") != "HOLD":
        _fail("V5-2U exact HOLD result missing")
    binding = retention.get("exact_v5_2t_binding")
    if not isinstance(binding, Mapping):
        _fail("V5-2U exact V5-2T binding missing")
    if binding.get("candidate_checkpoint_sha256") != v52u.V52T_CANDIDATE_SHA256:
        _fail("V5-2U candidate binding changed")
    if retention.get("first30_opened") is not False:
        _fail("V5-2U unexpectedly opened First-30")
    if retention.get("v5_validation_opened") is not False:
        _fail("V5-2U unexpectedly opened V5 validation")
    if retention.get("final_holdout_locked") is not True:
        _fail("V5-2U FINAL_HOLDOUT lock changed")
    if envelope.get("expected_head") != V52U_IMPLEMENTATION_HEAD:
        _fail("V5-2U envelope HEAD mismatch")
    if envelope.get("retention_report_sha256") != V52U_RETENTION_REPORT_SHA256:
        _fail("V5-2U envelope report binding mismatch")
    if (
        envelope.get("gate") != "HOLD"
        or envelope.get("first30_authorized") is not False
    ):
        _fail("V5-2U envelope stop boundary changed")
    return training, retention


def _transition_counts(*, frozen_prediction, candidate_prediction, target) -> dict[str, int]:
    torch, _nn = v52b._import_torch()
    p0 = frozen_prediction.detach().cpu().to(dtype=torch.bool).reshape(-1)
    p1 = candidate_prediction.detach().cpu().to(dtype=torch.bool).reshape(-1)
    y = target.detach().cpu().to(dtype=torch.bool).reshape(-1)
    if p0.shape != p1.shape or p0.shape != y.shape or p0.numel() == 0:
        _fail("prediction transition surface mismatch")
    c0 = p0 == y
    c1 = p1 == y
    result = {
        "count": int(y.numel()),
        "correct_to_correct": int((c0 & c1).sum().item()),
        "correct_to_wrong": int((c0 & ~c1).sum().item()),
        "wrong_to_correct": int((~c0 & c1).sum().item()),
        "wrong_to_wrong": int((~c0 & ~c1).sum().item()),
        "below_to_above_threshold": int((~p0 & p1).sum().item()),
        "above_to_below_threshold": int((p0 & ~p1).sum().item()),
        "frozen_predicted_positive": int(p0.sum().item()),
        "candidate_predicted_positive": int(p1.sum().item()),
    }
    accounting_keys = (
        "correct_to_correct",
        "correct_to_wrong",
        "wrong_to_correct",
        "wrong_to_wrong",
    )
    if sum(result[key] for key in accounting_keys) != result["count"]:
        _fail("prediction transition accounting mismatch")
    return result


def _group_drift_metrics_v1(
    *,
    features,
    targets,
    frozen_weight,
    candidate_weight,
    bias: float,
    threshold: float,
    name: str,
) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    x, y = v52r._as_surface(features, targets, name=name)
    w0 = v52r._as_weight(frozen_weight, name=f"{name}-frozen-weight")
    w1 = v52r._as_weight(candidate_weight, name=f"{name}-candidate-weight")
    if not math.isfinite(bias):
        _fail(f"{name} bias is non-finite")
    boundary = v52r._threshold_logit(threshold)
    delta_weight = w1 - w0
    delta_weight_norm = float(torch.linalg.vector_norm(delta_weight).item())
    feature_norm = torch.linalg.vector_norm(x, dim=1)
    frozen_logit = x @ w0 + bias
    candidate_logit = x @ w1 + bias
    logit_delta = candidate_logit - frozen_logit
    direct_delta = x @ delta_weight
    if not torch.allclose(logit_delta, direct_delta, rtol=0.0, atol=1e-12):
        _fail(f"{name} functional delta identity failed")
    cauchy_bound = feature_norm * delta_weight_norm
    if bool((torch.abs(logit_delta) > cauchy_bound + 1e-10).any().item()):
        _fail(f"{name} Cauchy logit-drift bound violated")
    nonzero = cauchy_bound > 0.0
    utilization = torch.zeros_like(cauchy_bound)
    signed_alignment = torch.zeros_like(cauchy_bound)
    utilization[nonzero] = torch.abs(logit_delta[nonzero]) / cauchy_bound[nonzero]
    signed_alignment[nonzero] = logit_delta[nonzero] / cauchy_bound[nonzero]
    label_sign = 2.0 * y - 1.0
    frozen_margin = label_sign * (frozen_logit - boundary)
    candidate_margin = label_sign * (candidate_logit - boundary)
    margin_delta = candidate_margin - frozen_margin
    frozen_prediction = frozen_logit >= boundary
    candidate_prediction = candidate_logit >= boundary
    transitions = _transition_counts(
        frozen_prediction=frozen_prediction,
        candidate_prediction=candidate_prediction,
        target=y,
    )
    return {
        "count": int(x.shape[0]),
        "positive_label_count": int(y.sum().item()),
        "feature_l2": v52q._quantile_summary(feature_norm, name=f"{name}-feature-norm"),
        "frozen_logit": v52q._quantile_summary(frozen_logit, name=f"{name}-frozen-logit"),
        "candidate_logit": v52q._quantile_summary(candidate_logit, name=f"{name}-candidate-logit"),
        "logit_delta": v52q._quantile_summary(logit_delta, name=f"{name}-logit-delta"),
        "absolute_logit_delta": v52q._quantile_summary(
            torch.abs(logit_delta), name=f"{name}-absolute-logit-delta"
        ),
        "cauchy_absolute_logit_delta_bound": v52q._quantile_summary(
            cauchy_bound, name=f"{name}-cauchy-bound"
        ),
        "cauchy_bound_utilization": v52q._quantile_summary(
            utilization, name=f"{name}-bound-utilization"
        ),
        "signed_feature_alignment_with_delta_weight": v52q._quantile_summary(
            signed_alignment, name=f"{name}-signed-alignment"
        ),
        "frozen_signed_margin": v52q._quantile_summary(
            frozen_margin, name=f"{name}-frozen-margin"
        ),
        "candidate_signed_margin": v52q._quantile_summary(
            candidate_margin, name=f"{name}-candidate-margin"
        ),
        "signed_margin_delta": v52q._quantile_summary(
            margin_delta, name=f"{name}-margin-delta"
        ),
        "transition_counts": transitions,
        "fraction_logit_increased": float(
            (logit_delta > 0.0).to(dtype=torch.float64).mean().item()
        ),
        "fraction_absolute_logit_delta_ge_1": float(
            (torch.abs(logit_delta) >= 1.0).to(dtype=torch.float64).mean().item()
        ),
        "fraction_absolute_logit_delta_ge_2": float(
            (torch.abs(logit_delta) >= 2.0).to(dtype=torch.float64).mean().item()
        ),
        "functional_delta_identity_verified": True,
        "cauchy_bound_verified": True,
    }


def functional_logit_drift_metrics_v1(
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
    torch, _nn = v52b._import_torch()
    hist_x, hist_y = v52r._as_surface(
        historical_features, historical_targets, name="historical"
    )
    v5_x, v5_y = v52r._as_surface(v5_features, v5_targets, name="v5")
    w0 = v52r._as_weight(frozen_weight, name="frozen-head-weight")
    w1 = v52r._as_weight(candidate_weight, name="candidate-head-weight")
    groups = {
        "v5_positive": (v5_x[v5_y == 1.0], v5_y[v5_y == 1.0]),
        "v5_negative": (v5_x[v5_y == 0.0], v5_y[v5_y == 0.0]),
        "historical_positive": (hist_x[hist_y == 1.0], hist_y[hist_y == 1.0]),
        "historical_negative": (hist_x[hist_y == 0.0], hist_y[hist_y == 0.0]),
    }
    per_group = {
        name: _group_drift_metrics_v1(
            features=groups[name][0],
            targets=groups[name][1],
            frozen_weight=w0,
            candidate_weight=w1,
            bias=frozen_bias,
            threshold=threshold,
            name=name,
        )
        for name in GROUPS
    }
    geometry = v52s.geometry_evidence_v1(
        frozen_weight=w0,
        candidate_weight=w1,
    )
    hist_neg_crossings = per_group["historical_negative"]["transition_counts"][
        "correct_to_wrong"
    ]
    v5_pos_recoveries = per_group["v5_positive"]["transition_counts"][
        "wrong_to_correct"
    ]
    v5_positive_delta = groups["v5_positive"][0] @ (w1 - w0)
    historical_negative_delta = groups["historical_negative"][0] @ (w1 - w0)
    v5_mean = groups["v5_positive"][0].mean(dim=0)
    hist_neg_mean = groups["historical_negative"][0].mean(dim=0)
    return {
        "threshold": float(threshold),
        "threshold_logit": v52r._threshold_logit(threshold),
        "head_geometry": geometry,
        "per_group": per_group,
        "cross_domain_delta_relationship": {
            "v5_positive_mean_logit_delta": float(v5_positive_delta.mean().item()),
            "historical_negative_mean_logit_delta": float(
                historical_negative_delta.mean().item()
            ),
            "mean_feature_cosine_v5_positive_vs_historical_negative": (
                v52r._cosine_or_none(v5_mean, hist_neg_mean)
            ),
            "mean_delta_same_sign": bool(
                float(v5_positive_delta.mean().item())
                * float(historical_negative_delta.mean().item())
                > 0.0
            ),
        },
        "functional_retention_diagnosis": {
            "weight_geometry_gate": geometry["gate"],
            "historical_negative_correct_to_wrong": hist_neg_crossings,
            "v5_positive_wrong_to_correct": v5_pos_recoveries,
            "parameter_geometry_passed_but_historical_decisions_changed": bool(
                geometry["gate"] == "PASS" and hist_neg_crossings > 0
            ),
            "weight_space_bound_sufficient_for_decision_retention": bool(
                hist_neg_crossings == 0
            ),
            "shared_linear_head_feasibility_proven": False,
            "representation_failure_proven": False,
            "repair_selected": False,
        },
    }


def run_functional_logit_drift_audit_v1(
    data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    digit2_candidate: str | Path,
    digit3_candidate: str | Path,
    training_report: str | Path,
    training_envelope: str | Path,
    retention_report: str | Path,
    retention_envelope: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run aggregate functional-drift evidence on the two open TRAIN surfaces."""
    root = Path(data_root)
    output = root / v51.ANNOTATIONS_DIR / REPORT_NAME
    if output.exists():
        _fail("refusing to overwrite/rerun V5-2V evidence")
    training, retention = _read_exact_evidence_v1(
        training_report=Path(training_report),
        training_envelope=Path(training_envelope),
        retention_report=Path(retention_report),
        retention_envelope=Path(retention_envelope),
    )
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
    candidate_paths = {"2": Path(digit2_candidate), "3": Path(digit3_candidate)}
    per_specialist: dict[str, object] = {}
    manifest_sha = v52b._sha_file(manifest_path)
    if training.get("slot_manifest_sha256") != manifest_sha:
        _fail("V5 TRAIN slot manifest changed after V5-2T")
    for digit in ("2", "3"):
        if v52b._sha_file(candidate_paths[digit]) != v52u.V52T_CANDIDATE_SHA256[digit]:
            _fail(f"V5-2T {digit}-AI candidate file SHA changed")
        source_sha = v52b.DIGIT2_SHA256 if digit == "2" else v52b.DIGIT3_SHA256
        candidate = v52t._load_candidate(
            candidate_paths[digit],
            digit=digit,
            source_sha=source_sha,
            manifest_sha=manifest_sha,
        )
        frozen = frozen_models[digit]
        invariants = v52q.verify_candidate_frozen_surface_v1(
            frozen_model=frozen,
            candidate_model=candidate,
        )
        metrics = functional_logit_drift_metrics_v1(
            historical_features=historical_features[digit],
            historical_targets=historical_targets[digit],
            v5_features=v5_features[digit],
            v5_targets=v5_targets[digit],
            frozen_weight=frozen.head.weight.detach().cpu().reshape(-1),
            candidate_weight=candidate.head.weight.detach().cpu().reshape(-1),
            frozen_bias=float(frozen.head.bias.detach().cpu().reshape(-1)[0].item()),
            threshold=v52b.FROZEN_THRESHOLDS[digit],
        )
        per_specialist[digit] = {
            "candidate_sha256": v52u.V52T_CANDIDATE_SHA256[digit],
            "candidate_state_invariants": invariants,
            "v5_train_metrics_at_frozen_threshold": v52p._feature_metrics(
                candidate,
                v5_features[digit],
                v5_targets[digit],
                threshold=v52b.FROZEN_THRESHOLDS[digit],
            ),
            "historical_train_metrics_at_frozen_threshold": v52p._feature_metrics(
                candidate,
                historical_features[digit],
                historical_targets[digit],
                threshold=v52b.FROZEN_THRESHOLDS[digit],
            ),
            **metrics,
        }
    report: dict[str, object] = {
        "schema": SCHEMA,
        "question": (
            "how_did_a_small_v5_2t_head_weight_delta_create_large_train_logit_"
            "and_threshold_decision_drift"
        ),
        "analysis_surface": "aggregate-v5-and-historical-train-only",
        "slot_manifest_sha256": manifest_sha,
        "v5_train_slot_count": EXPECTED_V5_COUNT,
        "historical_train_record_count": EXPECTED_HISTORICAL_COUNT,
        "feature_dim": EXPECTED_FEATURE_DIM,
        "exact_v5_2t_binding": {
            "implementation_head": v52u.V52T_IMPLEMENTATION_HEAD,
            "training_report_sha256": v52u.V52T_TRAINING_REPORT_SHA256,
            "execution_envelope_sha256": v52u.V52T_EXECUTION_ENVELOPE_SHA256,
            "candidate_checkpoint_sha256": dict(v52u.V52T_CANDIDATE_SHA256),
        },
        "exact_v5_2u_hold_binding": {
            "implementation_head": V52U_IMPLEMENTATION_HEAD,
            "retention_report_sha256": V52U_RETENTION_REPORT_SHA256,
            "execution_envelope_sha256": V52U_EXECUTION_ENVELOPE_SHA256,
            "gate": retention["gate"],
        },
        "historical_validation_aggregate_hold_read_for_binding_only": True,
        "historical_validation_examples_opened": False,
        "historical_validation_error_examples_opened": False,
        "per_specialist": per_specialist,
        **safety_boundary(),
    }
    v51._atomic_write_json(output, report)
    return report


def validation_opened_by_this_module() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False
