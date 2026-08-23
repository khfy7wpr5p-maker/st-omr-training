"""Meter V5-2J read-only domain-normalized balance audit.

Consumes only existing V5-2F/V5-2I JSON evidence. No images, checkpoints,
training, backward pass, optimizer step, threshold tuning, or spatial work.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Final, Mapping

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2f_replay_balance_audit_v1 as v52f
from . import meter_v5_2i_repair_pilot_v1 as v52i


SCHEMA: Final[str] = "st-omr-meter-v5-2j-domain-normalized-balance-audit-v1"
REPORT_NAME: Final[str] = "v5_2j_domain_normalized_balance_audit_v1.json"
EXPECTED_POS_WEIGHT: Final[float] = 1.0
EXPECTED_RAW_MIX_RATIO: Final[float] = 12.0
EXPECTED_V52I_CANDIDATE_SHA: Final[dict[str, str]] = {
    "2": "6d7bc9d6593496a16d8ff18839766520dc1f04b90cfa34d8feb626a821cf6253",
    "3": "555ace1477abd6c9ab69c6cbea85aa4aa956f6099a571ab36dae94d4d60b1319",
}


def _fail(message: str) -> None:
    raise v52b.MeterV5_2BError(message)


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"V5-2J requires numeric field: {field}")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"V5-2J requires finite field: {field}")
    return result


def _verify_v52f(root: Path) -> tuple[Path, dict[str, object]]:
    path = root / v51.ANNOTATIONS_DIR / v52f.REPLAY_BALANCE_REPORT_NAME
    report = v52b._read_json(path)
    if report.get("schema") != v52f.REPLAY_BALANCE_SCHEMA:
        _fail("V5-2J requires exact V5-2F replay-balance evidence")
    if report.get("replay_ratio_selected") is not False:
        _fail("V5-2F must not have selected a replay ratio")
    if report.get("repair_training_authorized") is not False:
        _fail("V5-2F authorization boundary changed")
    if report.get("training") is not False or report.get("optimizer_steps") != 0:
        _fail("V5-2F must remain analytical only")
    return path, report


def _verify_v52i(root: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    ann = root / v51.ANNOTATIONS_DIR
    training_path = ann / v52i.TRAINING_REPORT_NAME
    final_path = ann / v52i.FINAL_REPORT_NAME
    training = v52b._read_json(training_path)
    final = v52b._read_json(final_path)
    if training.get("schema") != v52i.SCHEMA or final.get("schema") != v52i.SCHEMA:
        _fail("V5-2J requires exact V5-2I evidence")
    recipe = training.get("recipe")
    if not isinstance(recipe, Mapping):
        _fail("V5-2I training recipe missing")
    expected_recipe = {
        "source_examples_per_v5_example": 12,
        "positive_weight": 1.0,
        "v5_adaptation_slots": 540,
        "historical_replay_count": 6480,
        "combined_example_count": 7020,
        "epochs": 1,
        "batch_size": 64,
    }
    for key, expected in expected_recipe.items():
        if recipe.get(key) != expected:
            _fail(f"V5-2I carried-forward recipe changed: {key}")
    if final.get("overall_gate") != "HOLD":
        _fail("V5-2J is defined for the observed V5-2I HOLD")
    if final.get("historical_retention_gate") != "HOLD":
        _fail("V5-2I historical retention must be HOLD")
    if final.get("v5_diagnostic_executed") is not False:
        _fail("V5-2I diagnostic must remain NOT_RUN after retention HOLD")
    if final.get("v5_diagnostic_gate") != "NOT_RUN":
        _fail("V5-2I diagnostic gate identity changed")
    if final.get("candidate_sha256") != EXPECTED_V52I_CANDIDATE_SHA:
        _fail("V5-2I candidate identity changed")
    return training_path, final_path, training, final


def _signed_means_from_v52f(report: Mapping[str, object], *, digit: str) -> tuple[float, float]:
    per = report.get("per_specialist")
    if not isinstance(per, Mapping):
        _fail("V5-2F per-specialist evidence missing")
    item = per.get(digit)
    if not isinstance(item, Mapping):
        _fail(f"V5-2F {digit}-AI evidence missing")
    weight = item.get("pos_weight_1")
    if not isinstance(weight, Mapping):
        _fail(f"V5-2F {digit}-AI pos_weight=1 evidence missing")
    v5 = weight.get("v5")
    source = weight.get("historical_source")
    if not isinstance(v5, Mapping) or not isinstance(source, Mapping):
        _fail(f"V5-2F {digit}-AI signed surfaces missing")
    v5_mean = _finite(v5.get("signed_mean"), field=f"{digit}.v5.signed_mean")
    source_mean = _finite(source.get("signed_mean"), field=f"{digit}.source.signed_mean")
    if not (v5_mean < 0.0 and source_mean > 0.0):
        _fail(f"V5-2J requires opposing V5/source signed means for {digit}-AI")
    return v5_mean, source_mean


def _zero_crossing(v5_mean: float, source_mean: float) -> float:
    if not (math.isfinite(v5_mean) and math.isfinite(source_mean)) or source_mean <= 0.0:
        raise ValueError("invalid signed means")
    value = -v5_mean / source_mean
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("invalid zero crossing")
    return value


def _minimax_reference(
    *,
    v5_2: float,
    source_2: float,
    v5_3: float,
    source_3: float,
) -> dict[str, float]:
    z2 = _zero_crossing(v5_2, source_2)
    z3 = _zero_crossing(v5_3, source_3)
    lower, upper = sorted((z2, z3))

    # Inside the interval, the two residuals have opposite signs. The minimax
    # point equalizes their absolute signed residual magnitudes.
    candidate = -(v5_2 + v5_3) / (source_2 + source_3)
    if not math.isfinite(candidate):
        raise ValueError("non-finite minimax reference")
    candidate = min(max(candidate, lower), upper)
    r2 = v5_2 + candidate * source_2
    r3 = v5_3 + candidate * source_3
    return {
        "lambda_source": candidate,
        "residual_2": r2,
        "residual_3": r3,
        "max_absolute_residual": max(abs(r2), abs(r3)),
    }


def run_domain_normalized_balance_audit_v1(v5_data_root: str | Path) -> dict[str, object]:
    root = Path(v5_data_root)
    v52f_path, balance = _verify_v52f(root)
    v52i_training_path, v52i_final_path, _training, _final = _verify_v52i(root)

    signed: dict[str, dict[str, float]] = {}
    zeros: dict[str, float] = {}
    for digit in ("2", "3"):
        v5_mean, source_mean = _signed_means_from_v52f(balance, digit=digit)
        signed[digit] = {
            "v5_signed_mean": v5_mean,
            "historical_signed_mean": source_mean,
        }
        zeros[digit] = _zero_crossing(v5_mean, source_mean)

    lower = min(zeros.values())
    upper = max(zeros.values())
    minimax = _minimax_reference(
        v5_2=signed["2"]["v5_signed_mean"],
        source_2=signed["2"]["historical_signed_mean"],
        v5_3=signed["3"]["v5_signed_mean"],
        source_3=signed["3"]["historical_signed_mean"],
    )

    residual_at_12 = {
        digit: signed[digit]["v5_signed_mean"]
        + EXPECTED_RAW_MIX_RATIO * signed[digit]["historical_signed_mean"]
        for digit in ("2", "3")
    }

    report: dict[str, object] = {
        "schema": SCHEMA,
        "v5_2f_report_sha256": v52b._sha_file(v52f_path),
        "v5_2i_training_report_sha256": v52b._sha_file(v52i_training_path),
        "v5_2i_final_report_sha256": v52b._sha_file(v52i_final_path),
        "objective_contract": "mean(V5_BCE_w1)+lambda_source*mean(HISTORICAL_BCE_w1)",
        "positive_weight": EXPECTED_POS_WEIGHT,
        "lambda_is_domain_loss_coefficient_not_sample_ratio": True,
        "signed_means": signed,
        "per_specialist_zero_crossing_lambda_source": zeros,
        "cross_specialist_zero_crossing_interval": {
            "min": lower,
            "max": upper,
        },
        "minimax_reference": {
            **minimax,
            "reference_only": True,
            "training_setting_selected": False,
        },
        "prior_raw_12_to_1_mixture": {
            "equivalent_first_order_lambda_source": EXPECTED_RAW_MIX_RATIO,
            "above_both_zero_crossings": EXPECTED_RAW_MIX_RATIO > upper,
            "signed_residual_at_frozen_checkpoint": residual_at_12,
            "source_direction_for_both": all(value > 0.0 for value in residual_at_12.values()),
        },
        "interpretation": {
            "domain_count_dominance_removed_by_mean_normalization": True,
            "logit_level_first_order_reference_only": True,
            "parameter_gradient_balance_proven": False,
            "training_trajectory_stability_proven": False,
            "retention_pass_proven": False,
            "v5_learning_pass_proven": False,
        },
        "domain_weight_selected": False,
        "replay_ratio_selected": False,
        "learning_rate_selected": False,
        "epoch_count_selected": False,
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
        "digit4_frozen": True,
        "resolver_wiring": False,
        "production_promotion": False,
    }
    v51._atomic_write_json(root / v51.ANNOTATIONS_DIR / REPORT_NAME, report)
    return report


def training_allowed_by_this_module() -> bool:
    return False


def validation_opened_by_this_module() -> bool:
    return False


def final_holdout_locked() -> bool:
    return True


def production_promotion_allowed() -> bool:
    return False
