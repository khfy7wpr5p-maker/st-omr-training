"""Exact-float joint arbitration for frozen Meter V2 digit specialists.

Historical validation thresholds were applied to floating-point sigmoid
probabilities. This module preserves that decision boundary exactly instead of
rounding probabilities to milli-units before thresholding.

Exactly one passing specialist yields accepted visual digit evidence. Multiple
passing specialists remain AMBIGUOUS even when one score is larger. No passing
specialist yields no observation. Malformed evidence is REJECTED.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

from .meter_v2_deterministic_composer_v1 import (
    AMBIGUOUS,
    ACCEPTED,
    REJECTED,
    MeterBox,
    MeterDigitObservation,
)

FROZEN_DIGIT_THRESHOLDS: Final[dict[int, float]] = {
    2: 0.48,
    3: 0.60,
    4: 0.47,
}

R_SLOT_INVALID: Final[str] = "METER_DIGIT_SLOT_INVALID"
R_SLOT_SPECIALIST_CONFLICT: Final[str] = "METER_DIGIT_SLOT_SPECIALIST_CONFLICT"


@dataclass(frozen=True, slots=True)
class MeterDigitSlotProbabilities:
    slot_id: str
    bbox: MeterBox
    score_2: float
    score_3: float
    score_4: float


def _valid_probability(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _valid_box(box: object) -> bool:
    if not isinstance(box, MeterBox):
        return False
    values = (box.x0, box.y0, box.x1, box.y1)
    return (
        all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in values
        )
        and float(box.x0) < float(box.x1)
        and float(box.y0) < float(box.y1)
    )


def digit_observation_from_probabilities_v1(
    slot: MeterDigitSlotProbabilities,
) -> MeterDigitObservation | None:
    if not isinstance(slot, MeterDigitSlotProbabilities) or not slot.slot_id or not _valid_box(slot.bbox):
        return MeterDigitObservation(
            "invalid-slot",
            REJECTED,
            None,
            0,
            None,
            (R_SLOT_INVALID,),
        )

    scores = {
        2: slot.score_2,
        3: slot.score_3,
        4: slot.score_4,
    }
    if any(not _valid_probability(value) for value in scores.values()):
        return MeterDigitObservation(
            slot.slot_id,
            REJECTED,
            None,
            0,
            None,
            (R_SLOT_INVALID,),
        )

    passing = tuple(
        digit
        for digit in (2, 3, 4)
        if float(scores[digit]) >= FROZEN_DIGIT_THRESHOLDS[digit]
    )
    if not passing:
        return None
    if len(passing) > 1:
        confidence = max(float(scores[digit]) for digit in passing)
        return MeterDigitObservation(
            slot.slot_id,
            AMBIGUOUS,
            None,
            int(round(confidence * 1000.0)),
            slot.bbox,
            (R_SLOT_SPECIALIST_CONFLICT,),
        )

    digit = passing[0]
    return MeterDigitObservation(
        slot.slot_id,
        ACCEPTED,
        digit,
        int(round(float(scores[digit]) * 1000.0)),
        slot.bbox,
        (),
    )


def confidence_ranking_breaks_conflict() -> bool:
    return False


def resolver_connection_allowed() -> bool:
    return False
