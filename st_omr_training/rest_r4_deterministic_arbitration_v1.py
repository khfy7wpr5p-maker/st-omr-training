"""Deterministic shadow arbitration for Half/Quarter/Eighth Rest specialists.

This module freezes only class arbitration semantics. It does not load models,
access sealed TEST, connect the runtime Resolver, or authorize production use.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Mapping

REST_CLASS_ORDER: Final[tuple[str, ...]] = ("half", "quarter", "eighth")
REST_DURATION_QUARTER_LENGTHS: Final[dict[str, float]] = {
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
}

# Frozen value-specific verifier thresholds from admitted R4 evidence.
REST_VERIFIER_THRESHOLDS: Final[dict[str, float]] = {
    "half": 0.070689357817173,
    "quarter": 0.3782260715961457,
    "eighth": 0.5620679259300232,
}

REST_ACCEPTED: Final[str] = "ACCEPTED"
REST_AMBIGUOUS: Final[str] = "AMBIGUOUS"
REST_REJECTED: Final[str] = "REJECTED"


@dataclass(frozen=True, slots=True)
class RestArbitrationResult:
    status: str
    class_name: str | None
    duration_quarter_lengths: float | None
    passing_classes: tuple[str, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {REST_ACCEPTED, REST_AMBIGUOUS, REST_REJECTED}:
            raise ValueError("unsupported Rest arbitration status")
        if self.status == REST_ACCEPTED:
            if self.class_name not in REST_CLASS_ORDER:
                raise ValueError("accepted result requires one supported Rest class")
            if self.duration_quarter_lengths != REST_DURATION_QUARTER_LENGTHS[self.class_name]:
                raise ValueError("accepted Rest duration/class mismatch")
            if self.passing_classes != (self.class_name,):
                raise ValueError("accepted result requires exactly one passing class")
            if self.reasons:
                raise ValueError("accepted result cannot carry failure reasons")
        else:
            if self.class_name is not None or self.duration_quarter_lengths is not None:
                raise ValueError("non-accepted result cannot assign a Rest class/duration")
            if not self.reasons:
                raise ValueError("non-accepted result must explain why")


def _valid_unit(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def arbitrate_rest_scores(
    scores: Mapping[str, float],
    *,
    thresholds: Mapping[str, float] = REST_VERIFIER_THRESHOLDS,
    bbox_finite: bool = True,
) -> RestArbitrationResult:
    """Fail-closed deterministic arbitration over value-specific Rest scores.

    Rules:
    - malformed/non-finite evidence -> REJECTED;
    - no class at or above its frozen threshold -> AMBIGUOUS;
    - more than one class at or above threshold -> AMBIGUOUS;
    - exactly one passing class -> ACCEPTED;
    - confidence ranking never breaks a class conflict.
    """

    score_keys = tuple(scores.keys()) if isinstance(scores, Mapping) else ()
    threshold_keys = tuple(thresholds.keys()) if isinstance(thresholds, Mapping) else ()
    if set(score_keys) != set(REST_CLASS_ORDER) or set(threshold_keys) != set(REST_CLASS_ORDER):
        return RestArbitrationResult(
            REST_REJECTED, None, None, (), ("R_INPUT_SCHEMA",)
        )
    if bbox_finite is not True:
        return RestArbitrationResult(
            REST_REJECTED, None, None, (), ("R_INVALID_BBOX",)
        )
    for class_name in REST_CLASS_ORDER:
        if not _valid_unit(scores[class_name]) or not _valid_unit(thresholds[class_name]):
            return RestArbitrationResult(
                REST_REJECTED, None, None, (), ("R_NONFINITE_OR_RANGE",)
            )

    passing = tuple(
        class_name
        for class_name in REST_CLASS_ORDER
        if float(scores[class_name]) >= float(thresholds[class_name])
    )

    if not passing:
        return RestArbitrationResult(
            REST_AMBIGUOUS, None, None, (), ("R_NO_CLASS_ABOVE_THRESHOLD",)
        )
    if len(passing) > 1:
        return RestArbitrationResult(
            REST_AMBIGUOUS, None, None, passing, ("R_CLASS_CONFLICT",)
        )

    class_name = passing[0]
    return RestArbitrationResult(
        REST_ACCEPTED,
        class_name,
        REST_DURATION_QUARTER_LENGTHS[class_name],
        passing,
        (),
    )


def resolver_connection_allowed() -> bool:
    """This shadow contract never authorizes runtime Resolver wiring."""
    return False
