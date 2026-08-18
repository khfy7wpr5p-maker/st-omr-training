"""Validation-only Meter V2 presence bridge from frozen M3-B evidence.

M3-B evaluated the authoritative 1,224 D10 Meter VALIDATION records with the
frozen D11 Meter checkpoint and defined visual presence score as
``1 - P(none)``. The frozen development rule selected the highest threshold
whose positive recall was at least 0.995; that threshold is 0.90.

This module freezes that already-produced shadow evidence. It is not a new
Presence model, does not load D11, does not tune from TEST, and does not
connect to the runtime Resolver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

from .meter_v2_deterministic_composer_v1 import (
    ACCEPTED,
    REJECTED,
    MeterPresenceObservation,
)

M3B_PRESENCE_EVIDENCE_STAGE: Final[str] = "M3-B-v4-target213-resume-v1"
M3B_PRESENCE_CACHE_SHA256: Final[str] = (
    "12f70dcdd15c377b85d57f585b59a03a2286f507a0bb7022c0a9ff26a6515ebd"
)
M3B_D11_CHECKPOINT_SHA256: Final[str] = (
    "cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3"
)
M3B_PRESENCE_THRESHOLD: Final[float] = 0.90
M3B_VALIDATION_TOTAL: Final[int] = 1_224
M3B_VALIDATION_POSITIVE: Final[int] = 591
M3B_VALIDATION_NONE: Final[int] = 633
M3B_TEST_RECORDS: Final[int] = 0

R_M3B_SCORE_INVALID: Final[str] = "METER_M3B_PRESENCE_SCORE_INVALID"


@dataclass(frozen=True, slots=True)
class M3BPresenceValidationEvidence:
    true_positive: int = 590
    false_positive: int = 8
    false_negative: int = 1
    true_negative: int = 625

    @property
    def precision(self) -> float:
        return self.true_positive / (self.true_positive + self.false_positive)

    @property
    def recall(self) -> float:
        return self.true_positive / (self.true_positive + self.false_negative)

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        return 2.0 * precision * recall / (precision + recall)

    @property
    def accuracy(self) -> float:
        total = self.true_positive + self.false_positive + self.false_negative + self.true_negative
        return (self.true_positive + self.true_negative) / total


M3B_PRESENCE_VALIDATION: Final[M3BPresenceValidationEvidence] = M3BPresenceValidationEvidence()


def presence_from_m3b_score_v1(presence_score: float) -> MeterPresenceObservation:
    """Collapse frozen M3-B ``1-P(none)`` evidence to present/absent.

    The decision compares the original floating-point score directly with 0.90;
    it is not rounded to milli-units before thresholding.
    """
    if (
        isinstance(presence_score, bool)
        or not isinstance(presence_score, (int, float))
        or not math.isfinite(float(presence_score))
        or not 0.0 <= float(presence_score) <= 1.0
    ):
        return MeterPresenceObservation(REJECTED, None, 0, (R_M3B_SCORE_INVALID,))

    score = float(presence_score)
    confidence_milli = max(0, min(1000, int(round(score * 1000.0))))
    return MeterPresenceObservation(
        ACCEPTED,
        score >= M3B_PRESENCE_THRESHOLD,
        confidence_milli,
        (),
    )


def presence_bridge_product_quality_accepted() -> bool:
    """M3-B is a strong shadow bridge, not a final product Presence acceptance."""
    return False


def resolver_connection_allowed() -> bool:
    return False
