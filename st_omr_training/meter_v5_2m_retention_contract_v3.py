"""Meter V5-2M corrected historical retention-only contract.

V3 keeps the exact corrected V5-2C V2 frozen oracle and removes the inherited
absolute precision/recall floors that were tied to V1's invalid oracle. The
retention question is evaluated only as degradation from the exact frozen
baseline at unchanged thresholds.
"""
from __future__ import annotations

from typing import Final, Mapping

from . import meter_v5_2c_historical_retention_v2 as ret_v2


SCHEMA: Final[str] = "st-omr-meter-v5-2m-retention-contract-v3"
MAX_F1_DROP: Final[float] = 0.005
MAX_RECALL_DROP: Final[float] = 0.005
MAX_PRECISION_DROP: Final[float] = 0.005

EXPECTED_FROZEN_COUNTS: Final[dict[str, dict[str, int]]] = {
    digit: dict(counts) for digit, counts in ret_v2.EXPECTED_FROZEN_COUNTS.items()
}


def _metric_from_counts(counts: Mapping[str, object]) -> dict[str, float]:
    tp = int(counts["tp"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    tn = int(counts["tn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def corrected_frozen_metrics() -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for digit, counts in EXPECTED_FROZEN_COUNTS.items():
        metrics = _metric_from_counts(counts)
        result[digit] = {**dict(counts), **metrics}
    return result


def evaluate_retention_gate_v3(
    *,
    frozen_metrics: Mapping[str, Mapping[str, object]],
    candidate_metrics: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Evaluate prospective retention as baseline-relative degradation only."""
    reasons: list[str] = []
    per_digit: dict[str, dict[str, object]] = {}

    for digit in ("2", "3"):
        baseline = frozen_metrics[digit]
        candidate = candidate_metrics[digit]

        f1_drop = float(baseline["f1"]) - float(candidate["f1"])
        recall_drop = float(baseline["recall"]) - float(candidate["recall"])
        precision_drop = float(baseline["precision"]) - float(candidate["precision"])

        digit_reasons: list[str] = []
        if f1_drop > MAX_F1_DROP + 1e-12:
            digit_reasons.append("F1_DROP_GT_0.005")
        if recall_drop > MAX_RECALL_DROP + 1e-12:
            digit_reasons.append("RECALL_DROP_GT_0.005")
        if precision_drop > MAX_PRECISION_DROP + 1e-12:
            digit_reasons.append("PRECISION_DROP_GT_0.005")

        reasons.extend(f"{digit}-AI_{reason}" for reason in digit_reasons)
        per_digit[digit] = {
            "f1_drop": f1_drop,
            "recall_drop": recall_drop,
            "precision_drop": precision_drop,
            "candidate_precision": float(candidate["precision"]),
            "candidate_recall": float(candidate["recall"]),
            "reasons": digit_reasons,
        }

    return {
        "schema": SCHEMA,
        "gate": "PASS" if not reasons else "HOLD",
        "reasons": reasons,
        "per_digit": per_digit,
        "absolute_precision_floor_used": False,
        "absolute_recall_floor_used": False,
        "retention_only": True,
    }


def safety_boundary() -> dict[str, object]:
    return {
        "training": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_write": False,
        "threshold_tuning": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "reserve_v5_train_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "resolver_wiring": False,
        "production_promotion": False,
    }
