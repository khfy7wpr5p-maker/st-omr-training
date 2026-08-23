"""Meter V5-2F read-only replay-balance audit.

This module reads only the completed V5-2E JSON evidence and analytically
computes source/V5 signed-logit pressure zero crossings. It does not read
images or checkpoints and performs no training, backward pass, optimizer step,
threshold tuning, or spatial derivation.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Final, Mapping

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2e_gradient_pressure_audit_v1 as v52e


REPLAY_BALANCE_SCHEMA: Final[str] = "st-omr-meter-v5-2f-replay-balance-audit-v1"
REPLAY_BALANCE_REPORT_NAME: Final[str] = "v5_2f_replay_balance_audit_v1.json"
EXPECTED_V5_COUNT: Final[int] = 540
EXPECTED_HISTORICAL_COUNT: Final[int] = 26_964
EXPECTED_WEIGHTS: Final[tuple[float, ...]] = (1.0, 5.0)
EXPECTED_ROOT_CAUSE_CLASS: Final[str] = (
    "V5_ONLY_DOMAIN_ADAPTATION_WITH_UNCONSTRAINED_SOURCE_FORGETTING"
)


def _fail(message: str) -> None:
    raise v52b.MeterV5_2BError(message)


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"V5-2F requires numeric field: {field}")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"V5-2F requires finite field: {field}")
    return result


def _verify_v52e_report(root: Path) -> tuple[Path, dict[str, object]]:
    path = root / v51.ANNOTATIONS_DIR / v52e.PRESSURE_REPORT_NAME
    report = v52b._read_json(path)

    expected = {
        "schema": v52e.PRESSURE_SCHEMA,
        "v5_adaptation_train_slot_count": EXPECTED_V5_COUNT,
        "m4a_train_record_count": EXPECTED_HISTORICAL_COUNT,
        "root_cause_class": EXPECTED_ROOT_CAUSE_CLASS,
        "dominant_mechanism": "UNRESOLVED",
        "pos_weight_5_unique_root_cause_supported": False,
        "repair_training_authorized": False,
        "replay_ratio_selected": False,
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
        "frozen_control_specialist": "4-AI",
        "resolver_wiring_authorized": False,
        "production_promotion_authorized": False,
    }
    for key, expected_value in expected.items():
        if report.get(key) != expected_value:
            _fail(f"V5-2E carried-forward evidence changed: {key}")

    observed_weights = report.get("counterfactual_positive_weights")
    if observed_weights != list(EXPECTED_WEIGHTS):
        _fail("V5-2E counterfactual positive weights changed")

    v5_profiles = report.get("v5_adaptation_train_pressure")
    historical_profiles = report.get("historical_m4a_train_pressure")
    if not isinstance(v5_profiles, Mapping) or not isinstance(historical_profiles, Mapping):
        _fail("V5-2E pressure profiles are missing")

    for digit in ("2", "3"):
        v5_digit = v5_profiles.get(digit)
        historical_digit = historical_profiles.get(digit)
        if not isinstance(v5_digit, Mapping) or not isinstance(historical_digit, Mapping):
            _fail(f"V5-2E {digit}-AI pressure profile missing")
        frozen_v5 = v5_digit.get("frozen")
        frozen_historical = historical_digit.get("frozen")
        if not isinstance(frozen_v5, Mapping) or not isinstance(frozen_historical, Mapping):
            _fail(f"V5-2E {digit}-AI frozen pressure profile missing")
        for weight in EXPECTED_WEIGHTS:
            key = f"pos_weight_{weight:g}"
            for surface_name, surface, expected_count in (
                ("v5", frozen_v5, EXPECTED_V5_COUNT),
                ("historical", frozen_historical, EXPECTED_HISTORICAL_COUNT),
            ):
                profile = surface.get(key)
                if not isinstance(profile, Mapping):
                    _fail(f"V5-2E {digit}-AI {surface_name} {key} profile missing")
                if profile.get("count") != expected_count:
                    _fail(f"V5-2E {digit}-AI {surface_name} count changed")
                observed_weight = _finite_number(
                    profile.get("positive_weight"),
                    field=f"{digit}.{surface_name}.{key}.positive_weight",
                )
                if observed_weight != weight:
                    _fail(f"V5-2E {digit}-AI {surface_name} positive weight changed")
                for field in ("positive_pressure_total", "negative_pressure_total"):
                    value = _finite_number(
                        profile.get(field),
                        field=f"{digit}.{surface_name}.{key}.{field}",
                    )
                    if value < 0.0:
                        _fail("pressure totals must be non-negative")

    return path, report


def _signed_pressure(profile: Mapping[str, object], *, expected_count: int) -> dict[str, float | int]:
    count = profile.get("count")
    if count != expected_count:
        _fail(f"pressure profile count changed: expected {expected_count}, got {count}")
    positive = _finite_number(profile.get("positive_pressure_total"), field="positive_pressure_total")
    negative = _finite_number(profile.get("negative_pressure_total"), field="negative_pressure_total")
    if positive < 0.0 or negative < 0.0:
        _fail("pressure totals must be non-negative")
    signed_total = negative - positive
    signed_mean = signed_total / expected_count
    return {
        "count": expected_count,
        "positive_pressure_total": positive,
        "negative_pressure_total": negative,
        "signed_total": signed_total,
        "signed_mean": signed_mean,
    }


def _balance_pair(
    v5_profile: Mapping[str, object],
    source_profile: Mapping[str, object],
    *,
    v5_count: int = EXPECTED_V5_COUNT,
    source_count: int = EXPECTED_HISTORICAL_COUNT,
) -> dict[str, object]:
    if v5_count <= 0 or source_count <= 0:
        raise ValueError("balance counts must be positive")
    v5 = _signed_pressure(v5_profile, expected_count=v5_count)
    source = _signed_pressure(source_profile, expected_count=source_count)
    v5_mean = float(v5["signed_mean"])
    source_mean = float(source["signed_mean"])
    opposing = bool(v5_mean * source_mean < 0.0)

    zero_ratio: float | None = None
    zero_examples: float | None = None
    zero_source_fraction: float | None = None
    if opposing and source_mean != 0.0:
        candidate = -v5_mean / source_mean
        if math.isfinite(candidate) and candidate > 0.0:
            zero_ratio = candidate
            zero_examples = candidate * v5_count
            zero_source_fraction = zero_examples / source_count

    full_source_ratio = source_count / v5_count
    combined_total_full_pass = float(v5["signed_total"]) + float(source["signed_total"])
    combined_mean_full_pass = combined_total_full_pass / (v5_count + source_count)

    return {
        "v5": v5,
        "historical_source": source,
        "domain_signed_pressures_oppose": opposing,
        "zero_crossing_source_examples_per_v5_example": zero_ratio,
        "zero_crossing_historical_examples_for_one_v5_pass": zero_examples,
        "zero_crossing_fraction_of_full_historical_train": zero_source_fraction,
        "full_source_pass_source_examples_per_v5_example": full_source_ratio,
        "full_source_pass_combined_signed_total": combined_total_full_pass,
        "full_source_pass_combined_signed_mean": combined_mean_full_pass,
        "zero_crossing_within_one_full_source_pass": (
            bool(zero_ratio <= full_source_ratio) if zero_ratio is not None else False
        ),
    }


def _cross_specialist_summary(
    per_digit: Mapping[str, Mapping[str, object]],
    *,
    weight: float,
) -> dict[str, object]:
    key = f"pos_weight_{weight:g}"
    ratios: dict[str, float] = {}
    within: dict[str, bool] = {}
    for digit in ("2", "3"):
        digit_payload = per_digit.get(digit)
        if not isinstance(digit_payload, Mapping):
            _fail(f"missing {digit}-AI replay balance result")
        item = digit_payload.get(key)
        if not isinstance(item, Mapping):
            _fail(f"missing {digit}-AI {key} replay balance result")
        ratio = item.get("zero_crossing_source_examples_per_v5_example")
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(float(ratio)):
            return {
                "both_specialists_finite": False,
                "zero_crossing_span_source_examples_per_v5_example": None,
                "both_within_one_full_source_pass": False,
            }
        ratios[digit] = float(ratio)
        within[digit] = bool(item.get("zero_crossing_within_one_full_source_pass"))
    return {
        "both_specialists_finite": True,
        "per_specialist_zero_crossing": ratios,
        "zero_crossing_span_source_examples_per_v5_example": {
            "min": min(ratios.values()),
            "max": max(ratios.values()),
        },
        "both_within_one_full_source_pass": all(within.values()),
    }


def run_replay_balance_audit_v1(v5_data_root: str | Path) -> dict[str, object]:
    root = Path(v5_data_root)
    v52e_path, v52e_report = _verify_v52e_report(root)
    v5_profiles = v52e_report["v5_adaptation_train_pressure"]
    historical_profiles = v52e_report["historical_m4a_train_pressure"]
    assert isinstance(v5_profiles, Mapping)
    assert isinstance(historical_profiles, Mapping)

    per_digit: dict[str, dict[str, object]] = {}
    for digit in ("2", "3"):
        per_digit[digit] = {}
        v5_digit = v5_profiles[digit]
        source_digit = historical_profiles[digit]
        assert isinstance(v5_digit, Mapping) and isinstance(source_digit, Mapping)
        v5_frozen = v5_digit["frozen"]
        source_frozen = source_digit["frozen"]
        assert isinstance(v5_frozen, Mapping) and isinstance(source_frozen, Mapping)
        for weight in EXPECTED_WEIGHTS:
            key = f"pos_weight_{weight:g}"
            v5_profile = v5_frozen[key]
            source_profile = source_frozen[key]
            assert isinstance(v5_profile, Mapping) and isinstance(source_profile, Mapping)
            per_digit[digit][key] = _balance_pair(v5_profile, source_profile)

    cross_specialist = {
        f"pos_weight_{weight:g}": _cross_specialist_summary(
            per_digit,
            weight=weight,
        )
        for weight in EXPECTED_WEIGHTS
    }

    report: dict[str, object] = {
        "schema": REPLAY_BALANCE_SCHEMA,
        "v5_2e_report_sha256": v52b._sha_file(v52e_path),
        "v5_adaptation_train_slot_count": EXPECTED_V5_COUNT,
        "m4a_train_record_count": EXPECTED_HISTORICAL_COUNT,
        "full_source_pass_source_examples_per_v5_example": (
            EXPECTED_HISTORICAL_COUNT / EXPECTED_V5_COUNT
        ),
        "diagnostic_positive_weights": list(EXPECTED_WEIGHTS),
        "signed_pressure_contract": (
            "signed_total=negative_pressure_total-positive_pressure_total; "
            "combined_mean(r)=v5_signed_mean+r*historical_signed_mean"
        ),
        "per_specialist": per_digit,
        "cross_specialist": cross_specialist,
        "interpretation": {
            "logit_level_only": True,
            "parameter_gradient_balance_proven": False,
            "training_trajectory_stability_proven": False,
            "automatic_training_recipe_selection": False,
        },
        "replay_ratio_selected": False,
        "positive_weight_selected": False,
        "sampling_strategy_selected": False,
        "repair_training_authorized": False,
        "training": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_read": False,
        "checkpoint_write": False,
        "image_read": False,
        "threshold_tuning": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "reserve_v5_train_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "frozen_control_specialist": "4-AI",
        "resolver_wiring_authorized": False,
        "production_promotion_authorized": False,
    }
    v51._atomic_write_json(
        root / v51.ANNOTATIONS_DIR / REPLAY_BALANCE_REPORT_NAME,
        report,
    )
    return report


def training_allowed_by_this_module() -> bool:
    return False


def validation_opened_by_this_module() -> bool:
    return False


def final_holdout_locked() -> bool:
    return True


def production_promotion_allowed() -> bool:
    return False
