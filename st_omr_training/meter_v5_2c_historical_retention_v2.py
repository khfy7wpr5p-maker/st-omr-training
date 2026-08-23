"""Meter V5-2C historical retention audit V2.

Inference-only correction to V1's historical self-check oracle.  V2 deliberately
reuses V1's exact M4A/D10 pixel replay helpers and changes only the confusion
counts that must be reproduced by the SHA-bound frozen checkpoints before the
V5-2B candidates may be interpreted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2c_historical_retention_v1 as legacy


RETENTION_SCHEMA: Final[str] = "st-omr-meter-v5-2c-historical-retention-v2"
RETENTION_REPORT_NAME: Final[str] = "v5_2c_historical_retention_v2.json"

M4A_MANIFEST_SHA256: Final[str] = legacy.M4A_MANIFEST_SHA256
D10_MANIFEST_SHA256: Final[str] = legacy.D10_MANIFEST_SHA256
DIGIT2_CANDIDATE_SHA256: Final[str] = legacy.DIGIT2_CANDIDATE_SHA256
DIGIT3_CANDIDATE_SHA256: Final[str] = legacy.DIGIT3_CANDIDATE_SHA256
EXPECTED_VALIDATION_LABEL_COUNTS: Final[dict[str, int]] = dict(
    legacy.EXPECTED_VALIDATION_LABEL_COUNTS
)

# Exact original validation results for the SHA-bound checkpoints on the M4A
# validation surface.  V1 incorrectly substituted later summary/shadow counts.
EXPECTED_FROZEN_COUNTS: Final[dict[str, dict[str, int]]] = {
    "2": {"tp": 185, "fp": 30, "fn": 1, "tn": 3156},
    "3": {"tp": 203, "fp": 1, "fn": 1, "tn": 3167},
    "4": {"tp": 788, "fp": 46, "fn": 4, "tn": 2534},
}

MAX_F1_DROP: Final[float] = legacy.MAX_F1_DROP
MAX_RECALL_DROP: Final[float] = legacy.MAX_RECALL_DROP
MIN_CANDIDATE_PRECISION: Final[float] = legacy.MIN_CANDIDATE_PRECISION
MIN_CANDIDATE_RECALL: Final[float] = legacy.MIN_CANDIDATE_RECALL


def run_historical_retention_v2(
    v5_data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    digit4_frozen: str | Path,
    digit2_candidate: str | Path,
    digit3_candidate: str | Path,
    progress=None,
) -> dict[str, object]:
    """Replay exact historical inputs, self-check frozen models, then candidates."""
    root = Path(v5_data_root)
    m4a = Path(m4a_root)
    d10 = Path(d10_root)
    frozen_paths = {
        "2": Path(digit2_frozen),
        "3": Path(digit3_frozen),
        "4": Path(digit4_frozen),
    }
    candidate_paths = {
        "2": Path(digit2_candidate),
        "3": Path(digit3_candidate),
    }

    manifest_path, _slot_rows, _slot_audit = v52b.verify_slot_manifest_v1(root)
    training_report = v52b._read_json(
        root / v51.ANNOTATIONS_DIR / v52b.TRAINING_REPORT_NAME
    )
    candidates_report = training_report.get("candidates")
    if not isinstance(candidates_report, Mapping):
        raise v52b.MeterV5_2BError("V5-2B training report missing candidates")

    expected_candidate_sha = {
        "2": DIGIT2_CANDIDATE_SHA256,
        "3": DIGIT3_CANDIDATE_SHA256,
    }
    for digit in ("2", "3"):
        actual = v52b._sha_file(candidate_paths[digit])
        if actual != expected_candidate_sha[digit]:
            raise v52b.MeterV5_2BError(f"{digit}-AI candidate SHA changed")
        candidate_evidence = candidates_report.get(digit)
        if not isinstance(candidate_evidence, Mapping):
            raise v52b.MeterV5_2BError(
                f"V5-2B training report missing {digit}-AI candidate"
            )
        if candidate_evidence.get("candidate_sha256") != actual:
            raise v52b.MeterV5_2BError(
                f"{digit}-AI candidate differs from V5-2B training report"
            )

    frozen_expected_sha = {
        "2": v52b.DIGIT2_SHA256,
        "3": v52b.DIGIT3_SHA256,
        "4": v52b.DIGIT4_SHA256,
    }
    for digit in ("2", "3", "4"):
        if v52b._sha_file(frozen_paths[digit]) != frozen_expected_sha[digit]:
            raise v52b.MeterV5_2BError(f"frozen {digit}-AI SHA changed")

    # Reuse the V1 replay implementation exactly: no new crop/spatial semantics.
    validation, d10_meter = legacy._load_manifests(
        m4a_root=m4a,
        d10_root=d10,
    )
    images, labels = legacy._prepare_inputs(
        validation=validation,
        d10_meter=d10_meter,
        d10_root=d10,
        progress=progress,
    )

    frozen_metrics: dict[str, dict[str, object]] = {}
    for digit in ("2", "3", "4"):
        probabilities = legacy._probabilities(
            legacy._frozen_model(frozen_paths[digit], digit=digit),
            images,
            progress=progress,
            phase=f"frozen-{digit}-AI-v2-self-check",
        )
        metrics = legacy._binary_counts(
            probabilities,
            legacy._truth_tensor(labels, digit),
            v52b.FROZEN_THRESHOLDS[digit],
        )
        frozen_metrics[digit] = metrics
        expected = EXPECTED_FROZEN_COUNTS[digit]
        if any(
            metrics[key] != expected[key]
            for key in ("tp", "fp", "fn", "tn")
        ):
            raise v52b.MeterV5_2BError(
                "historical pixel-path reproduction failed for "
                f"{digit}-AI under corrected V2 oracle: {metrics}"
            )

    manifest_sha = v52b._sha_file(manifest_path)
    candidate_metrics: dict[str, dict[str, object]] = {}
    for digit in ("2", "3"):
        model = v52b._load_candidate_model(
            candidate_paths[digit],
            digit=digit,
            manifest_sha256=manifest_sha,
        )
        probabilities = legacy._probabilities(
            model,
            images,
            progress=progress,
            phase=f"candidate-{digit}-AI-v2",
        )
        candidate_metrics[digit] = legacy._binary_counts(
            probabilities,
            legacy._truth_tensor(labels, digit),
            v52b.FROZEN_THRESHOLDS[digit],
        )

    gate = legacy.evaluate_retention_gate_v1(
        frozen_metrics=frozen_metrics,
        candidate_metrics=candidate_metrics,
    )

    report: dict[str, object] = {
        "schema": RETENTION_SCHEMA,
        "supersedes_invalid_oracle_schema": legacy.RETENTION_SCHEMA,
        "v1_failure_interpreted_candidates": False,
        "m4a_manifest_sha256": M4A_MANIFEST_SHA256,
        "d10_manifest_sha256": D10_MANIFEST_SHA256,
        "validation_record_count": 3372,
        "validation_label_counts": dict(EXPECTED_VALIDATION_LABEL_COUNTS),
        "thresholds": {
            "2": v52b.FROZEN_THRESHOLDS["2"],
            "3": v52b.FROZEN_THRESHOLDS["3"],
            "4": v52b.FROZEN_THRESHOLDS["4"],
        },
        "frozen_checkpoint_sha256": dict(frozen_expected_sha),
        "candidate_sha256": dict(expected_candidate_sha),
        "historical_pixel_path_reproduced": True,
        "historical_oracle_source": {
            "2": "m4c5-2ai-train-hard-none-v1/result.json",
            "3": "m4c2-none75-negative-sampling-v1/result.json",
            "4": "m4c2-none75-negative-sampling-v1/result.json",
        },
        "frozen_metrics": frozen_metrics,
        "candidate_metrics": candidate_metrics,
        "retention_limits": {
            "max_f1_drop": MAX_F1_DROP,
            "max_recall_drop": MAX_RECALL_DROP,
            "min_candidate_precision": MIN_CANDIDATE_PRECISION,
            "min_candidate_recall": MIN_CANDIDATE_RECALL,
        },
        "gate": gate["gate"],
        "reasons": gate["reasons"],
        "per_digit_retention": gate["per_digit"],
        "validation_bbox_stage_authorized": gate["gate"] == "PASS",
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "optimizer_steps": 0,
        "threshold_tuning": False,
        "resolver_wiring_authorized": False,
        "production_promotion_authorized": False,
    }
    v51._atomic_write_json(
        root / v51.ANNOTATIONS_DIR / RETENTION_REPORT_NAME,
        report,
    )
    return report


def validation_opened_by_this_module() -> bool:
    return False


def final_holdout_locked() -> bool:
    return True


def production_promotion_allowed() -> bool:
    return False
