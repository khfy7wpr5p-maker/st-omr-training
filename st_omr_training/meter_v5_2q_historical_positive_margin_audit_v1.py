"""Meter V5-2Q TRAIN-only historical-positive margin audit.

Read-only geometry audit over the already-open V5 adaptation TRAIN surface and
historical M4A TRAIN surface. It compares the exact frozen V5-2P source heads
with the exact retained V5-2P HOLD candidate heads without opening historical
validation, first-30, V5 validation, or FINAL_HOLDOUT.

No objective, solver, threshold, model parameter, crop, BBox, or data split is
selected or modified here. No per-example identifiers are emitted.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Final, Mapping

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2n_frozen_feature_transfer_audit_v1 as v52n
from . import meter_v5_2p_fixed_bias_head_repair_v1 as v52p
from . import meter_v5_2p_numerical_evidence_guard_v1 as v52p_guard


SCHEMA: Final[str] = "st-omr-meter-v5-2q-historical-positive-margin-audit-v1"
REPORT_NAME: Final[str] = "v5_2q_historical_positive_margin_audit_v1.json"
EXPECTED_FEATURE_DIM: Final[int] = 64
EXPECTED_V5_COUNT: Final[int] = 540
EXPECTED_HISTORICAL_COUNT: Final[int] = 26_964
DIGIT2_CANDIDATE_SHA256: Final[str] = (
    "369b7f610b1d9785368422f62669419868a8b86975a89f13b03c169fa6161616"
)
DIGIT3_CANDIDATE_SHA256: Final[str] = (
    "491602359a74c4829a205d60c194a00e63b54553823873f814980457667ac785"
)
QUANTILE_LEVELS: Final[tuple[float, ...]] = (
    0.0,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.0,
)
ProgressCallback = Callable[[int, int, str], None]


class MeterV5_2QError(RuntimeError):
    """Raised when the TRAIN-only V5-2Q audit boundary cannot be proven."""


def _fail(message: str) -> None:
    raise MeterV5_2QError(message)


def safety_boundary() -> dict[str, object]:
    """Declare the non-training, non-validation authority of this audit."""
    return {
        "training": False,
        "autograd_grad_used": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_read": True,
        "checkpoint_write": False,
        "candidate_checkpoint_mutation": False,
        "objective_changed": False,
        "solver_settings_changed": False,
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
        "repair_objective_selected": False,
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


def _quantile_summary(values, *, name: str) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    x = values.detach().cpu().to(dtype=torch.float64).reshape(-1)
    _finite_tensor(x, name=name)
    q = torch.tensor(QUANTILE_LEVELS, dtype=torch.float64)
    quantiles = torch.quantile(x, q)
    labels = (
        "min",
        "p01",
        "p05",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
    )
    return {
        "count": int(x.numel()),
        "mean": float(x.mean().item()),
        "std_population": float(x.std(unbiased=False).item()),
        **{label: float(value.item()) for label, value in zip(labels, quantiles)},
    }


def _rank_binned_shift_v1(frozen_margin, margin_delta) -> dict[str, object]:
    """Describe shift by frozen-margin rank without selecting a decision threshold."""
    torch, _nn = v52b._import_torch()
    frozen = frozen_margin.detach().cpu().to(dtype=torch.float64).reshape(-1)
    delta = margin_delta.detach().cpu().to(dtype=torch.float64).reshape(-1)
    if frozen.shape != delta.shape or frozen.numel() < 20:
        _fail("rank-bin audit requires aligned margin vectors with at least 20 rows")
    _finite_tensor(frozen, name="frozen-positive-margin-rank-source")
    _finite_tensor(delta, name="positive-margin-delta-rank-source")
    order = torch.argsort(frozen)
    n = int(frozen.numel())
    bins = (
        ("bottom_10pct", 0.00, 0.10),
        ("p10_to_p25", 0.10, 0.25),
        ("p25_to_p50", 0.25, 0.50),
        ("p50_to_p75", 0.50, 0.75),
        ("p75_to_p90", 0.75, 0.90),
        ("top_10pct", 0.90, 1.00),
    )
    result: dict[str, object] = {}
    for label, lo, hi in bins:
        start = int(math.floor(lo * n))
        stop = n if hi == 1.0 else int(math.floor(hi * n))
        stop = max(stop, start + 1)
        idx = order[start:stop]
        if idx.numel() == 0:
            _fail(f"empty rank bin: {label}")
        f = frozen[idx]
        d = delta[idx]
        result[label] = {
            "count": int(idx.numel()),
            "frozen_margin_mean": float(f.mean().item()),
            "margin_delta_mean": float(d.mean().item()),
            "margin_delta_median": float(torch.median(d).item()),
            "fraction_margin_decreased": float((d < 0.0).to(dtype=torch.float64).mean().item()),
        }
    return result


def _cosine_or_none(a, b) -> float | None:
    torch, _nn = v52b._import_torch()
    x = a.detach().cpu().to(dtype=torch.float64).reshape(-1)
    y = b.detach().cpu().to(dtype=torch.float64).reshape(-1)
    if x.shape != y.shape or x.numel() == 0:
        _fail("cosine vector shape mismatch")
    nx = float(torch.linalg.vector_norm(x).item())
    ny = float(torch.linalg.vector_norm(y).item())
    if not math.isfinite(nx) or not math.isfinite(ny):
        _fail("non-finite cosine norm")
    if nx == 0.0 or ny == 0.0:
        return None
    value = float(torch.dot(x, y).item()) / (nx * ny)
    return min(1.0, max(-1.0, value))


def verify_candidate_frozen_surface_v1(*, frozen_model, candidate_model) -> dict[str, object]:
    """Require V5-2P candidate differences to remain head.weight-only."""
    torch, _nn = v52b._import_torch()
    frozen_state = frozen_model.state_dict()
    candidate_state = candidate_model.state_dict()
    if set(frozen_state) != set(candidate_state):
        _fail("candidate/frozen state keys differ")
    changed = [
        name
        for name in sorted(frozen_state)
        if not torch.equal(
            frozen_state[name].detach().cpu(),
            candidate_state[name].detach().cpu(),
        )
    ]
    illegal = [name for name in changed if name != "head.weight"]
    if illegal:
        _fail(f"candidate contains non-head.weight mutation: {illegal}")
    if candidate_model.head.weight.numel() != EXPECTED_FEATURE_DIM:
        _fail("candidate head.weight parameter count changed")
    backbone_identical = all(
        torch.equal(
            frozen_state[name].detach().cpu(),
            candidate_state[name].detach().cpu(),
        )
        for name in frozen_state
        if name.startswith("features.")
    )
    bias_identical = torch.equal(
        frozen_state["head.bias"].detach().cpu(),
        candidate_state["head.bias"].detach().cpu(),
    )
    if not backbone_identical or not bias_identical:
        _fail("candidate frozen backbone/bias integrity failed")
    return {
        "changed_state_keys": changed,
        "only_head_weight_changed": not illegal,
        "backbone_bit_identical": backbone_identical,
        "head_bias_bit_identical": bias_identical,
        "head_weight_parameter_count": int(candidate_model.head.weight.numel()),
    }


def positive_margin_audit_metrics_v1(
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
    """Pure TRAIN-only descriptive margin geometry for one digit specialist."""
    torch, _nn = v52b._import_torch()
    hist_x = historical_features.detach().cpu().to(dtype=torch.float64)
    hist_y = historical_targets.detach().cpu().to(dtype=torch.float64).reshape(-1)
    v5_x = v5_features.detach().cpu().to(dtype=torch.float64)
    v5_y = v5_targets.detach().cpu().to(dtype=torch.float64).reshape(-1)
    w0 = frozen_weight.detach().cpu().to(dtype=torch.float64).reshape(-1)
    w1 = candidate_weight.detach().cpu().to(dtype=torch.float64).reshape(-1)

    for name, x, y in (("historical", hist_x, hist_y), ("v5", v5_x, v5_y)):
        if x.ndim != 2 or x.shape[1] != EXPECTED_FEATURE_DIM:
            _fail(f"{name} feature shape changed: {tuple(x.shape)}")
        if len(x) != len(y):
            _fail(f"{name} feature/target cardinality mismatch")
        _finite_tensor(x, name=f"{name}-features")
        if not bool(((y == 0.0) | (y == 1.0)).all().item()):
            _fail(f"{name} targets are not binary")
    if w0.numel() != EXPECTED_FEATURE_DIM or w1.numel() != EXPECTED_FEATURE_DIM:
        _fail("head weight dimension changed")
    if not math.isfinite(frozen_bias):
        _fail("frozen bias is non-finite")
    _finite_tensor(w0, name="frozen-head-weight")
    _finite_tensor(w1, name="candidate-head-weight")

    hist_pos = hist_x[hist_y == 1.0]
    v5_pos = v5_x[v5_y == 1.0]
    if hist_pos.numel() == 0 or v5_pos.numel() == 0:
        _fail("positive examples required on both TRAIN surfaces")

    boundary = _threshold_logit(threshold)
    delta_w = w1 - w0

    hist_frozen_margin = hist_pos @ w0 + frozen_bias - boundary
    hist_candidate_margin = hist_pos @ w1 + frozen_bias - boundary
    hist_delta = hist_candidate_margin - hist_frozen_margin

    v5_frozen_logits = v5_x @ w0 + frozen_bias
    v5_candidate_logits = v5_x @ w1 + frozen_bias
    v5_frozen_class_margin = torch.where(
        v5_y == 1.0,
        v5_frozen_logits - boundary,
        boundary - v5_frozen_logits,
    )
    v5_candidate_class_margin = torch.where(
        v5_y == 1.0,
        v5_candidate_logits - boundary,
        boundary - v5_candidate_logits,
    )
    v5_class_margin_delta = v5_candidate_class_margin - v5_frozen_class_margin
    v5_pos_frozen_margin = v5_pos @ w0 + frozen_bias - boundary
    v5_pos_candidate_margin = v5_pos @ w1 + frozen_bias - boundary
    v5_pos_delta = v5_pos_candidate_margin - v5_pos_frozen_margin

    for name, tensor in (
        ("historical-frozen-positive-margin", hist_frozen_margin),
        ("historical-candidate-positive-margin", hist_candidate_margin),
        ("historical-positive-margin-delta", hist_delta),
        ("v5-positive-margin-delta", v5_pos_delta),
        ("v5-classification-margin-delta", v5_class_margin_delta),
    ):
        _finite_tensor(tensor, name=name)

    hist_mean_feature = hist_pos.mean(dim=0)
    v5_pos_mean_feature = v5_pos.mean(dim=0)
    delta_norm = float(torch.linalg.vector_norm(delta_w).item())
    if not math.isfinite(delta_norm):
        _fail("head rotation delta norm non-finite")

    hist_below_frozen = int((hist_frozen_margin < 0.0).sum().item())
    hist_below_candidate = int((hist_candidate_margin < 0.0).sum().item())

    return {
        "threshold": float(threshold),
        "threshold_logit": boundary,
        "historical_positive_count": int(hist_pos.shape[0]),
        "v5_positive_count": int(v5_pos.shape[0]),
        "head_rotation": {
            "delta_weight_l2": delta_norm,
            "delta_weight_max_abs": float(torch.max(torch.abs(delta_w)).item()),
            "delta_weight_cosine_historical_positive_mean_feature": _cosine_or_none(
                delta_w, hist_mean_feature
            ),
            "delta_weight_cosine_v5_positive_mean_feature": _cosine_or_none(
                delta_w, v5_pos_mean_feature
            ),
        },
        "historical_positive_margin": {
            "frozen": _quantile_summary(hist_frozen_margin, name="historical-frozen-positive-margin"),
            "candidate": _quantile_summary(
                hist_candidate_margin, name="historical-candidate-positive-margin"
            ),
            "candidate_minus_frozen": _quantile_summary(
                hist_delta, name="historical-positive-margin-delta"
            ),
            "fraction_margin_decreased": float(
                (hist_delta < 0.0).to(dtype=torch.float64).mean().item()
            ),
            "fraction_margin_increased": float(
                (hist_delta > 0.0).to(dtype=torch.float64).mean().item()
            ),
            "frozen_below_threshold_count": hist_below_frozen,
            "candidate_below_threshold_count": hist_below_candidate,
            "additional_below_threshold_count": hist_below_candidate - hist_below_frozen,
            "shift_by_frozen_margin_rank": _rank_binned_shift_v1(
                hist_frozen_margin, hist_delta
            ),
        },
        "v5_train_margin": {
            "positive_frozen": _quantile_summary(
                v5_pos_frozen_margin, name="v5-positive-frozen-margin"
            ),
            "positive_candidate": _quantile_summary(
                v5_pos_candidate_margin, name="v5-positive-candidate-margin"
            ),
            "positive_candidate_minus_frozen": _quantile_summary(
                v5_pos_delta, name="v5-positive-margin-delta"
            ),
            "classification_margin_candidate_minus_frozen": _quantile_summary(
                v5_class_margin_delta, name="v5-classification-margin-delta"
            ),
            "fraction_classification_margin_improved": float(
                (v5_class_margin_delta > 0.0).to(dtype=torch.float64).mean().item()
            ),
        },
        "directional_relation": {
            "historical_positive_mean_logit_shift": float(hist_delta.mean().item()),
            "v5_positive_mean_logit_shift": float(v5_pos_delta.mean().item()),
            "v5_all_classification_margin_mean_change": float(v5_class_margin_delta.mean().item()),
            "v5_positive_gain_and_historical_positive_loss_have_opposite_sign": bool(
                float(v5_pos_delta.mean().item()) > 0.0
                and float(hist_delta.mean().item()) < 0.0
            ),
        },
        "descriptive_only": True,
        "numeric_pass_threshold_preregistered": False,
        "repair_objective_selected": False,
    }


def _load_exact_v5_2p_evidence(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    ann = root / v51.ANNOTATIONS_DIR
    training_path = ann / v52p.TRAINING_REPORT_NAME
    numerical_path = ann / v52p_guard.REPORT_NAME
    training = v52b._read_json(training_path)
    numerical = v52b._read_json(numerical_path)
    if training.get("schema") != v52p.SCHEMA:
        _fail("exact V5-2P training report missing/wrong schema")
    if numerical.get("schema") != v52p_guard.SCHEMA:
        _fail("exact V5-2P numerical evidence missing/wrong schema")
    integrity = numerical.get("numerical_integrity_gate")
    if not isinstance(integrity, Mapping) or integrity.get("gate") != "PASS":
        _fail("V5-2P numerical integrity must be PASS before V5-2Q")
    if numerical.get("historical_retention_executed") is not False:
        _fail("numerical report unexpectedly includes historical retention")
    expected = {"2": DIGIT2_CANDIDATE_SHA256, "3": DIGIT3_CANDIDATE_SHA256}
    candidates = training.get("candidates")
    if not isinstance(candidates, Mapping):
        _fail("V5-2P candidate evidence missing")
    for digit in ("2", "3"):
        item = candidates.get(digit)
        if not isinstance(item, Mapping) or item.get("candidate_sha256") != expected[digit]:
            _fail(f"V5-2P {digit}-AI candidate binding changed")
    return training, numerical


def run_historical_positive_margin_audit_v1(
    v5_data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run the aggregate TRAIN-only V5-2Q historical-positive margin audit."""
    root = Path(v5_data_root)
    ann = root / v51.ANNOTATIONS_DIR
    report_path = ann / REPORT_NAME
    if report_path.exists():
        _fail(f"refusing to overwrite V5-2Q evidence: {report_path}")

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
        "2": DIGIT2_CANDIDATE_SHA256,
        "3": DIGIT3_CANDIDATE_SHA256,
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
        invariants = verify_candidate_frozen_surface_v1(
            frozen_model=frozen_model,
            candidate_model=candidate_model,
        )
        frozen_bias = float(frozen_model.head.bias.detach().cpu().reshape(-1)[0].item())
        metrics = positive_margin_audit_metrics_v1(
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
        "question": "did_v5_2p_head_rotation_systematically_reduce_historical_train_positive_margin",
        "analysis_surface": "aggregate-open-train-only",
        "slot_manifest_sha256": v52b._sha_file(manifest_path),
        "v5_2p_training_report_sha256": v52b._sha_file(ann / v52p.TRAINING_REPORT_NAME),
        "v5_2p_numerical_report_sha256": v52b._sha_file(ann / v52p_guard.REPORT_NAME),
        "v5_adaptation_train_slot_count": EXPECTED_V5_COUNT,
        "m4a_historical_train_record_count": EXPECTED_HISTORICAL_COUNT,
        "feature_dim": EXPECTED_FEATURE_DIM,
        "candidate_checkpoint_sha256": expected_candidate_sha,
        "frozen_checkpoint_sha256": {
            "2": v52b.DIGIT2_SHA256,
            "3": v52b.DIGIT3_SHA256,
        },
        "numerical_integrity_gate_carried_forward": numerical["numerical_integrity_gate"]["gate"],
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