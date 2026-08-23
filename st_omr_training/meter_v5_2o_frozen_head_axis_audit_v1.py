"""Meter V5-2O frozen head-axis audit.

Read-only diagnostic over the exact V5-2N frozen 64D features. The existing
2-AI/3-AI linear-head directions are measured without fitting, gradients,
threshold changes, checkpoint writes, or new spatial semantics.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Final

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2n_frozen_feature_transfer_audit_v1 as v52n


SCHEMA: Final[str] = "st-omr-meter-v5-2o-frozen-head-axis-audit-v1"
REPORT_NAME: Final[str] = "v5_2o_frozen_head_axis_audit_v1.json"


class MeterV5_2OError(RuntimeError):
    """Raised whenever a V5-2O fail-closed invariant is violated."""


def _fail(message: str) -> None:
    raise MeterV5_2OError(message)


def safety_boundary() -> dict[str, object]:
    return {
        "training": False,
        "autograd_grad_used": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_read": True,
        "checkpoint_write": False,
        "image_read": True,
        "runtime_threshold_tuning": False,
        "alternative_threshold_evaluated": False,
        "bias_parameter_selected": False,
        "classifier_fit_performed": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "old_d11_glyph_window_reused": False,
        "reserve_v5_train_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "resolver_wiring": False,
        "production_promotion": False,
        "architecture_selected": False,
        "residual_topology_selected": False,
        "bias_only_repair_selected": False,
        "repair_training_authorized": False,
    }


def _finite_tensor(tensor, *, name: str) -> None:
    torch, _nn = v52b._import_torch()
    if tensor.numel() == 0:
        _fail(f"empty tensor: {name}")
    if not bool(torch.isfinite(tensor).all().item()):
        _fail(f"non-finite tensor: {name}")


def _scalar_stats(values) -> dict[str, float]:
    torch, _nn = v52b._import_torch()
    x = values.detach().cpu().to(dtype=torch.float64).reshape(-1)
    _finite_tensor(x, name="scalar-stats")
    q = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], dtype=torch.float64)
    quantiles = torch.quantile(x, q)
    return {
        "min": float(x.min().item()),
        "p10": float(quantiles[0].item()),
        "p25": float(quantiles[1].item()),
        "median": float(quantiles[2].item()),
        "p75": float(quantiles[3].item()),
        "p90": float(quantiles[4].item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
    }


def _rank_auc(positive_logits, negative_logits) -> float:
    """Exact Mann-Whitney rank AUC without allocating an O(P*N) matrix."""
    torch, _nn = v52b._import_torch()
    pos = positive_logits.detach().cpu().to(dtype=torch.float64).reshape(-1)
    neg = negative_logits.detach().cpu().to(dtype=torch.float64).reshape(-1)
    _finite_tensor(pos, name="positive-logits")
    _finite_tensor(neg, name="negative-logits")
    n_pos = int(pos.numel())
    n_neg = int(neg.numel())
    if n_pos == 0 or n_neg == 0:
        _fail("rank AUC requires both classes")

    items = [(float(value), 1) for value in pos.tolist()]
    items.extend((float(value), 0) for value in neg.tolist())
    items.sort(key=lambda item: item[0])

    positive_rank_sum = 0.0
    index = 0
    while index < len(items):
        end = index + 1
        score = items[index][0]
        while end < len(items) and items[end][0] == score:
            end += 1
        # Ranks are one-based. The block spans ranks index+1 through end.
        average_rank = ((index + 1) + end) / 2.0
        positive_in_block = sum(label for _value, label in items[index:end])
        positive_rank_sum += float(positive_in_block) * average_rank
        index = end

    auc = (
        positive_rank_sum - (n_pos * (n_pos + 1) / 2.0)
    ) / float(n_pos * n_neg)
    if not math.isfinite(auc) or not (0.0 <= auc <= 1.0):
        _fail(f"invalid rank AUC: {auc}")
    return auc


def _domain_head_metrics(*, features, targets, head_weight, head_bias: float, boundary_logit: float) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    x = features.detach().cpu().to(dtype=torch.float64)
    y = targets.detach().cpu().to(dtype=torch.float64).reshape(-1)
    w = head_weight.detach().cpu().to(dtype=torch.float64).reshape(-1)

    if x.ndim != 2 or x.shape[1] != v52n.EXPECTED_FEATURE_DIM:
        _fail(f"unexpected feature shape: {tuple(x.shape)}")
    if len(x) != len(y):
        _fail("feature/target cardinality mismatch")
    if w.numel() != v52n.EXPECTED_FEATURE_DIM:
        _fail("frozen head feature dimension changed")
    _finite_tensor(x, name="features")
    _finite_tensor(w, name="head-weight")
    if not bool(((y == 0.0) | (y == 1.0)).all().item()):
        _fail("targets are not binary")
    if not math.isfinite(float(head_bias)) or not math.isfinite(float(boundary_logit)):
        _fail("head bias/boundary must be finite")

    weight_norm = float(torch.linalg.vector_norm(w).item())
    if weight_norm <= 0.0:
        _fail("frozen head weight norm is zero")
    unit_w = w / weight_norm

    positive = x[y == 1.0]
    negative = x[y == 0.0]
    if len(positive) == 0 or len(negative) == 0:
        _fail("both binary classes are required")

    positive_centroid = positive.mean(dim=0)
    negative_centroid = negative.mean(dim=0)
    midpoint = (positive_centroid + negative_centroid) / 2.0
    delta = positive_centroid - negative_centroid

    logits = x @ w + float(head_bias)
    positive_logits = logits[y == 1.0]
    negative_logits = logits[y == 0.0]
    _finite_tensor(logits, name="frozen-head-logits")

    min_positive = float(positive_logits.min().item())
    max_negative = float(negative_logits.max().item())
    class_gap_logit_mean = float(positive_logits.mean().item() - negative_logits.mean().item())
    class_gap_axis = float(torch.dot(delta, unit_w).item())
    midpoint_axis = float(torch.dot(midpoint, unit_w).item())
    midpoint_logit = float(torch.dot(midpoint, w).item()) + float(head_bias)

    return {
        "count": int(len(x)),
        "positive_count": int(len(positive)),
        "negative_count": int(len(negative)),
        "positive_logit": _scalar_stats(positive_logits),
        "negative_logit": _scalar_stats(negative_logits),
        "rank_auc": _rank_auc(positive_logits, negative_logits),
        "strict_separation_gap_logit": min_positive - max_negative,
        "strictly_separable_under_same_head_direction": bool(min_positive > max_negative),
        "class_mean_logit_gap": class_gap_logit_mean,
        "class_gap_along_normalized_head_axis": class_gap_axis,
        "binary_midpoint_along_normalized_head_axis": midpoint_axis,
        "binary_midpoint_logit": midpoint_logit,
        "frozen_boundary_logit": float(boundary_logit),
        "min_positive_minus_frozen_boundary": min_positive - float(boundary_logit),
        "frozen_boundary_minus_max_negative": float(boundary_logit) - max_negative,
        "all_positive_below_frozen_boundary": bool((positive_logits < boundary_logit).all().item()),
        "all_positive_at_or_above_frozen_boundary": bool((positive_logits >= boundary_logit).all().item()),
        "all_negative_below_frozen_boundary": bool((negative_logits < boundary_logit).all().item()),
        "all_negative_at_or_above_frozen_boundary": bool((negative_logits >= boundary_logit).all().item()),
    }


def head_axis_transfer_metrics_v1(
    *,
    source_features,
    source_targets,
    v5_features,
    v5_targets,
    head_weight,
    head_bias: float,
    frozen_threshold: float,
) -> dict[str, object]:
    """Descriptive frozen-head ranking/translation metrics; performs no fitting."""
    if not isinstance(frozen_threshold, (int, float)) or isinstance(frozen_threshold, bool):
        _fail("frozen threshold must be numeric")
    threshold = float(frozen_threshold)
    if not math.isfinite(threshold) or not (0.0 < threshold < 1.0):
        _fail("frozen threshold must be inside (0,1)")
    boundary_logit = math.log(threshold / (1.0 - threshold))

    source = _domain_head_metrics(
        features=source_features,
        targets=source_targets,
        head_weight=head_weight,
        head_bias=head_bias,
        boundary_logit=boundary_logit,
    )
    v5 = _domain_head_metrics(
        features=v5_features,
        targets=v5_targets,
        head_weight=head_weight,
        head_bias=head_bias,
        boundary_logit=boundary_logit,
    )

    source_gap = float(source["class_gap_along_normalized_head_axis"])
    v5_gap = float(v5["class_gap_along_normalized_head_axis"])
    midpoint_shift = float(v5["binary_midpoint_along_normalized_head_axis"]) - float(
        source["binary_midpoint_along_normalized_head_axis"]
    )

    return {
        "frozen_threshold": threshold,
        "frozen_boundary_logit": boundary_logit,
        "historical_train": source,
        "v5_adaptation_train": v5,
        "source_to_v5": {
            "midpoint_shift_along_normalized_head_axis": midpoint_shift,
            "absolute_midpoint_shift_along_normalized_head_axis": abs(midpoint_shift),
            "v5_over_source_abs_class_gap_along_head": (
                abs(v5_gap) / abs(source_gap) if source_gap != 0.0 else None
            ),
            "abs_midpoint_shift_over_abs_v5_class_gap": (
                abs(midpoint_shift) / abs(v5_gap) if v5_gap != 0.0 else None
            ),
            "class_gap_direction_preserved_along_head": bool(source_gap * v5_gap > 0.0),
        },
        "same_frozen_head_direction_strictly_separates_v5_train": bool(
            v5["strictly_separable_under_same_head_direction"]
        ),
        "bias_or_threshold_selected": False,
        "classifier_fit_performed": False,
    }


def _verify_v52n_evidence(root: Path) -> Path:
    report_path = root / v51.ANNOTATIONS_DIR / v52n.REPORT_NAME
    if report_path.is_symlink() or not report_path.is_file():
        _fail(f"required V5-2N evidence missing/non-regular: {report_path}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail(f"cannot parse V5-2N evidence: {exc}")
    if not isinstance(payload, dict) or payload.get("schema") != v52n.SCHEMA:
        _fail("V5-2N evidence schema mismatch")
    if payload.get("v5_adaptation_train_slot_count") != v52n.EXPECTED_V5_COUNT:
        _fail("V5-2N V5 count changed")
    if payload.get("m4a_train_record_count") != v52n.EXPECTED_HISTORICAL_COUNT:
        _fail("V5-2N historical count changed")
    if payload.get("feature_dim") != v52n.EXPECTED_FEATURE_DIM:
        _fail("V5-2N feature dimension changed")
    expected_sha = {"2": v52b.DIGIT2_SHA256, "3": v52b.DIGIT3_SHA256}
    if payload.get("frozen_checkpoint_sha256") != expected_sha:
        _fail("V5-2N frozen checkpoint binding changed")
    for key, expected in {
        "fit_or_training_performed": False,
        "training": False,
        "autograd_grad_used": False,
        "backward": False,
        "checkpoint_write": False,
        "threshold_tuning": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "reserve_v5_train_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "repair_training_authorized": False,
    }.items():
        if payload.get(key) is not expected:
            _fail(f"V5-2N safety evidence changed: {key}")
    return report_path


def run_frozen_head_axis_audit_v1(
    v5_data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    progress: v52n.ProgressCallback | None = None,
) -> dict[str, object]:
    """Run V5-2O over exact frozen V5-2N feature surfaces."""
    root = Path(v5_data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    report_path = ann_dir / REPORT_NAME
    if report_path.exists():
        _fail(f"refusing to overwrite existing V5-2O evidence: {report_path}")

    v52n_report = _verify_v52n_evidence(root)
    models = v52n._frozen_models(
        digit2_frozen=Path(digit2_frozen),
        digit3_frozen=Path(digit3_frozen),
    )
    manifest_path, _rows, v5_features, v5_targets, frozen_metrics = v52n._v5_surface(root, models)
    source_features, source_targets = v52n._historical_surface(
        m4a_root=Path(m4a_root),
        d10_root=Path(d10_root),
        models=models,
        progress=progress,
    )

    per_specialist: dict[str, object] = {}
    for digit in ("2", "3"):
        model = models[digit]
        if model.head.bias is None:
            _fail(f"frozen {digit}-AI head bias missing")
        head_weight = model.head.weight.detach().cpu().reshape(-1)
        head_bias = float(model.head.bias.detach().cpu().reshape(-1)[0].item())
        metrics = head_axis_transfer_metrics_v1(
            source_features=source_features[digit],
            source_targets=source_targets[digit],
            v5_features=v5_features[digit],
            v5_targets=v5_targets[digit],
            head_weight=head_weight,
            head_bias=head_bias,
            frozen_threshold=v52b.FROZEN_THRESHOLDS[digit],
        )
        metrics["frozen_head_v5_confusion_at_unchanged_threshold"] = frozen_metrics[digit]
        per_specialist[digit] = metrics

    report: dict[str, object] = {
        "schema": SCHEMA,
        "question": "does_existing_frozen_head_direction_order_v5_and_how_large_is_source_to_v5_head_axis_shift",
        "v52n_report_sha256": v52b._sha_file(v52n_report),
        "slot_manifest_sha256": v52b._sha_file(manifest_path),
        "frozen_checkpoint_sha256": {
            "2": v52b.DIGIT2_SHA256,
            "3": v52b.DIGIT3_SHA256,
        },
        "v5_adaptation_train_slot_count": v52n.EXPECTED_V5_COUNT,
        "m4a_train_record_count": v52n.EXPECTED_HISTORICAL_COUNT,
        "feature_dim": v52n.EXPECTED_FEATURE_DIM,
        "classifier_fit_performed": False,
        "alternative_threshold_evaluated": False,
        "numeric_pass_threshold_preregistered": False,
        "per_specialist": per_specialist,
        **safety_boundary(),
    }
    v51._atomic_write_json(report_path, report)
    return report
