"""Meter V5-2G declarative repair-training preregistration boundary.

No training, checkpoint mutation, threshold tuning, image access, or spatial
operation is implemented here. This module only freezes the analytical replay
feasibility evidence produced by V5-2F.
"""
from __future__ import annotations

from typing import Final

SCHEMA: Final[str] = "st-omr-meter-v5-2g-repair-preregistration-v1"

FULL_HISTORICAL_PASS_RATIO: Final[float] = 49.93333333333333

ZERO_CROSSING_POS_WEIGHT_1: Final[dict[str, float]] = {
    "2": 7.289268597494998,
    "3": 9.468108115675737,
}
ZERO_CROSSING_POS_WEIGHT_5: Final[dict[str, float]] = {
    "2": 173.9494597881494,
    "3": 100.36788462067109,
}


def shared_single_pass_feasibility_boundary() -> dict[str, object]:
    """Return the evidence-bound shared replay interval without selecting it."""
    floor = max(ZERO_CROSSING_POS_WEIGHT_1.values())
    ceiling = FULL_HISTORICAL_PASS_RATIO
    return {
        "schema": SCHEMA,
        "diagnostic_positive_weight": 1.0,
        "strict_lower_bound_source_examples_per_v5_example": floor,
        "upper_bound_source_examples_per_v5_example": ceiling,
        "lower_bound_is_zero_crossing_not_safety_margin": True,
        "replay_ratio_selected": False,
        "positive_weight_selected": False,
        "sampling_strategy_selected": False,
    }


def frozen_pos_weight_5_is_single_pass_feasible() -> dict[str, bool]:
    return {
        digit: ratio <= FULL_HISTORICAL_PASS_RATIO
        for digit, ratio in ZERO_CROSSING_POS_WEIGHT_5.items()
    }


def repair_training_authorized() -> bool:
    return False


def safety_boundary() -> dict[str, object]:
    return {
        "training": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_write": False,
        "threshold_tuning": False,
        "replay_ratio_selected": False,
        "positive_weight_selected": False,
        "sampling_strategy_selected": False,
        "repair_training_authorized": False,
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
