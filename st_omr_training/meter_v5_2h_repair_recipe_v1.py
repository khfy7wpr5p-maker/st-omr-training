"""Exact, review-only Meter V5-2H repair recipe.

This module freezes the selected repair pilot configuration. It deliberately
contains no training loop, backward call, optimizer execution, checkpoint write,
image access, threshold tuning, or spatial/BBox logic.
"""
from __future__ import annotations

import math
from typing import Final

SCHEMA: Final[str] = "st-omr-meter-v5-2h-repair-recipe-v1"

ZERO_CROSSING_POS_WEIGHT_1: Final[dict[str, float]] = {
    "2": 7.289268597494998,
    "3": 9.468108115675737,
}
SAFETY_MULTIPLIER: Final[float] = 1.25
REPLAY_RATIO: Final[int] = math.ceil(
    SAFETY_MULTIPLIER * max(ZERO_CROSSING_POS_WEIGHT_1.values())
)

V5_ADAPTATION_SLOTS: Final[int] = 540
HISTORICAL_REPLAY_COUNT: Final[int] = REPLAY_RATIO * V5_ADAPTATION_SLOTS
COMBINED_EXAMPLE_COUNT: Final[int] = V5_ADAPTATION_SLOTS + HISTORICAL_REPLAY_COUNT

HISTORICAL_LABEL_COUNTS: Final[dict[str, int]] = {
    "2": 367,
    "3": 381,
    "4": 1537,
    "NONE": 4195,
}

POS_WEIGHT: Final[float] = 1.0
OPTIMIZER: Final[str] = "AdamW"
LEARNING_RATE: Final[float] = 1e-4
WEIGHT_DECAY: Final[float] = 1e-4
BATCH_SIZE: Final[int] = 64
EPOCHS: Final[int] = 1
SEED: Final[int] = 52023
EXPECTED_OPTIMIZER_STEPS: Final[int] = math.ceil(COMBINED_EXAMPLE_COUNT / BATCH_SIZE)
PREVIOUS_V5_ONLY_OPTIMIZER_STEPS: Final[int] = 12 * math.ceil(V5_ADAPTATION_SLOTS / 64)


def recipe() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "recipe_selected": True,
        "selection_rule": "ceil(1.25 * max(pos_weight_1_zero_crossing))",
        "safety_multiplier": SAFETY_MULTIPLIER,
        "source_examples_per_v5_example": REPLAY_RATIO,
        "v5_adaptation_slots": V5_ADAPTATION_SLOTS,
        "historical_replay_count": HISTORICAL_REPLAY_COUNT,
        "combined_example_count": COMBINED_EXAMPLE_COUNT,
        "historical_label_counts": dict(HISTORICAL_LABEL_COUNTS),
        "sampling_without_replacement": True,
        "sampling_stratification": "frozen_m4a_four_label_hamilton_allocation",
        "same_source_manifest_for_digit2_and_digit3": True,
        "first30_v5_diagnostic_zero_gradient": True,
        "positive_weight": POS_WEIGHT,
        "optimizer": OPTIMIZER,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "seed": SEED,
        "deterministic_shuffle": True,
        "fixed_final_epoch_only": True,
        "expected_optimizer_steps_if_later_authorized": EXPECTED_OPTIMIZER_STEPS,
        "previous_v5_only_optimizer_steps": PREVIOUS_V5_ONLY_OPTIMIZER_STEPS,
        "threshold_tuning": False,
        "digit4_frozen": True,
    }


def gates() -> dict[str, object]:
    return {
        "historical_retention_first": True,
        "historical_abs_f1_drop_max": 0.005,
        "historical_abs_recall_drop_max": 0.005,
        "historical_precision_min": 0.98,
        "historical_recall_min": 0.98,
        "v5_diagnostic_2_of_4_min": 8,
        "v5_diagnostic_3_of_4_min": 8,
        "v5_diagnostic_4_of_4_min": 9,
        "v5_denominator_exact4_min": 26,
        "automatic_second_configuration": False,
    }


def safety_boundary() -> dict[str, object]:
    return {
        "repair_training_authorized": False,
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
