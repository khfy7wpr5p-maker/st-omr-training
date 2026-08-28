"""Meter V5-4 unified numerator shadow architecture contract.

This module is declarative and deterministic only. It does not train a model,
open protected validation, mutate checkpoints, or grant runtime authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


SCHEMA: Final[str] = "st-omr-meter-v5-4-unified-numerator-shadow-contract-v1"
NUMERATOR_CLASSES: Final[tuple[str, ...]] = ("2", "3", "4")
TARGET_TRAINABLE_NUMERATOR_MODELS: Final[int] = 1
LEGACY_SPECIALISTS_REMAIN_CONTROLS: Final[bool] = True
SHADOW_ONLY: Final[bool] = True
V53K_EXTERNAL_REPORT_REQUIRED: Final[bool] = True


@dataclass(frozen=True)
class ShadowMeterCandidate:
    numerator: str
    denominator: str
    meter: str
    status: str = "SHADOW_ONLY"


def safety_boundary() -> dict[str, object]:
    return {
        "training": False,
        "fitting": False,
        "optimizer_steps": 0,
        "checkpoint_write": False,
        "model_mutation": False,
        "threshold_tuning": False,
        "crop_bbox_tuning": False,
        "historical_validation_opened": False,
        "first30_opened": False,
        "v5_reserve_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "resolver_wiring": False,
        "runtime_authority_changed": False,
        "production_promotion": False,
        "legacy_specialists_removed": False,
        "execution_recipe_selected": False,
        "v5_3k_external_report_required_before_training_preregistration": True,
    }


def architecture_contract() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "trainable_numerator_model_count": TARGET_TRAINABLE_NUMERATOR_MODELS,
        "numerator_classes": NUMERATOR_CLASSES,
        "legacy_specialists_remain_controls": LEGACY_SPECIALISTS_REMAIN_CONTROLS,
        "shadow_only": SHADOW_ONLY,
        "denominator_inside_numerator_classifier": False,
        "adapter_trainable": False,
        "meter_validator_trainable": False,
        "silent_meter_correction_allowed": False,
        "legacy_path_replacement_authorized": False,
        "v5_3k_external_report_required": V53K_EXTERNAL_REPORT_REQUIRED,
        "required_shadow_metrics": (
            "accuracy",
            "macro_f1",
            "recall_2",
            "recall_3",
            "recall_4",
            "confusion_2_to_3",
            "confusion_3_to_2",
            "abstention_or_review_rate",
            "historical_retention",
        ),
        **safety_boundary(),
    }


def compose_shadow_meter_candidate(
    numerator: str,
    denominator: str,
    *,
    numerator_admitted: bool,
    denominator_admitted: bool,
) -> ShadowMeterCandidate | None:
    """Compose only the current 2/4, 3/4, 4/4 shadow surface.

    The adapter never invents a denominator and never coerces unsupported input.
    """
    if not numerator_admitted or not denominator_admitted:
        return None
    if numerator not in NUMERATOR_CLASSES:
        return None
    if denominator != "4":
        return None
    return ShadowMeterCandidate(
        numerator=numerator,
        denominator=denominator,
        meter=f"{numerator}/{denominator}",
    )


def training_preregistration_allowed(*, v5_3k_external_report_bound: bool) -> bool:
    """V5-4 training design stays closed until V5-3K evidence is bound."""
    return bool(v5_3k_external_report_bound)


def production_promotion_allowed() -> bool:
    return False


def final_holdout_access_allowed() -> bool:
    return False


def future_gate_order() -> tuple[str, ...]:
    return (
        "complete_and_hash_bind_v5_3k_external_forensics",
        "preregister_one_fixed_unified_numerator_training_recipe",
        "train_one_shared_2_3_4_classifier_on_train_only_surface",
        "train_acceptance_and_historical_retention",
        "bounded_validation_if_prior_gates_pass",
        "shadow_adapter_and_meter_validator",
        "freeze_model_adapter_validator",
        "one_time_untouched_final_holdout",
    )
