"""Shadow-only deterministic Meter V2 composer.

The learned boundary is deliberately small:

- a Meter presence observation says only whether a meter group is visually present;
- digit observations say only which supported digit was seen and where;
- this module deterministically composes the supported product baseline meters
  ``2/4 | 3/4 | 4/4`` or fails closed.

No model/checkpoint is loaded here.  No D10/D11 training module, optimizer,
sealed TEST split, or runtime Deterministic Resolver is imported or connected.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

METER_V2_DIGITS: Final[tuple[int, ...]] = (2, 3, 4)
METER_V2_CLASSES: Final[tuple[str, ...]] = ("none", "2/4", "3/4", "4/4")
OBSERVATION_STATUSES: Final[tuple[str, ...]] = ("accepted", "ambiguous", "rejected")

ACCEPTED: Final[str] = "accepted"
AMBIGUOUS: Final[str] = "ambiguous"
REJECTED: Final[str] = "rejected"

R_PRESENCE_REJECTED: Final[str] = "METER_PRESENCE_REJECTED"
R_PRESENCE_AMBIGUOUS: Final[str] = "METER_PRESENCE_AMBIGUOUS"
R_PRESENCE_CONFLICT: Final[str] = "METER_PRESENCE_DIGIT_CONFLICT"
R_DIGIT_REJECTED: Final[str] = "METER_DIGIT_REJECTED"
R_DIGIT_AMBIGUOUS: Final[str] = "METER_DIGIT_AMBIGUOUS"
R_NONFINITE: Final[str] = "METER_NONFINITE_OR_RANGE"
R_INVALID_BBOX: Final[str] = "METER_INVALID_BBOX"
R_UNSUPPORTED_DIGIT: Final[str] = "METER_UNSUPPORTED_DIGIT"
R_MISSING_DIGIT: Final[str] = "METER_MISSING_DIGIT"
R_DIGIT_COUNT_CONFLICT: Final[str] = "METER_DIGIT_COUNT_CONFLICT"
R_GEOMETRY_AMBIGUOUS: Final[str] = "METER_DIGIT_GEOMETRY_AMBIGUOUS"
R_UNSUPPORTED_COMPOSITION: Final[str] = "METER_UNSUPPORTED_COMPOSITION"


@dataclass(frozen=True, slots=True)
class MeterBox:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True, slots=True)
class MeterPresenceObservation:
    status: str
    present: bool | None
    confidence_milli: int
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MeterDigitObservation:
    observation_id: str
    status: str
    digit: int | None
    confidence_milli: int
    bbox: MeterBox | None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MeterCompositionResult:
    status: str
    meter_class: str | None
    numerator: int | None
    denominator: int | None
    digit_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in OBSERVATION_STATUSES:
            raise ValueError("unsupported Meter composition status")
        if self.status == ACCEPTED:
            if self.meter_class not in METER_V2_CLASSES:
                raise ValueError("accepted Meter result requires a supported class")
            if self.reasons:
                raise ValueError("accepted Meter result cannot carry failure reasons")
            if self.meter_class == "none":
                if (self.numerator, self.denominator, self.digit_ids) != (None, None, ()):
                    raise ValueError("none Meter result cannot assign digits")
            else:
                if self.numerator not in METER_V2_DIGITS or self.denominator != 4:
                    raise ValueError("accepted Meter digits/class mismatch")
                if self.meter_class != f"{self.numerator}/{self.denominator}":
                    raise ValueError("accepted Meter class does not match digits")
                if len(self.digit_ids) != 2:
                    raise ValueError("accepted visible Meter requires exactly two digit ids")
        else:
            if self.meter_class is not None or self.numerator is not None or self.denominator is not None:
                raise ValueError("non-accepted Meter result cannot assign musical meaning")
            if not self.reasons:
                raise ValueError("non-accepted Meter result must explain why")


def _valid_confidence(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= 1000
    )


def _valid_box(box: object) -> bool:
    if not isinstance(box, MeterBox):
        return False
    values = (box.x0, box.y0, box.x1, box.y1)
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in values
    ):
        return False
    return float(box.x0) < float(box.x1) and float(box.y0) < float(box.y1)


def _x_grouped(a: MeterBox, b: MeterBox) -> bool:
    """Require the two stacked digits to occupy one plausible horizontal group."""
    overlap = min(float(a.x1), float(b.x1)) - max(float(a.x0), float(b.x0))
    if overlap > 0:
        return True
    center_a = (float(a.x0) + float(a.x1)) / 2.0
    center_b = (float(b.x0) + float(b.x1)) / 2.0
    max_width = max(float(a.x1) - float(a.x0), float(b.x1) - float(b.x0))
    return abs(center_a - center_b) <= max_width


def compose_meter_v2(
    presence: MeterPresenceObservation,
    digits: tuple[MeterDigitObservation, ...],
) -> MeterCompositionResult:
    """Compose Meter evidence without confidence ranking or musical guessing.

    Frozen priority:
    1. malformed/rejected evidence -> REJECTED;
    2. ambiguous evidence -> AMBIGUOUS;
    3. explicit absence with no digits -> accepted ``none``;
    4. explicit presence requires exactly two accepted supported digits;
    5. geometry determines upper/lower role;
    6. only 2/4, 3/4, 4/4 are admitted by v1;
    7. every unresolved case fails closed.
    """
    if not isinstance(presence, MeterPresenceObservation) or not isinstance(digits, tuple):
        return MeterCompositionResult(REJECTED, None, None, None, (), (R_NONFINITE,))
    if presence.status not in OBSERVATION_STATUSES or not _valid_confidence(presence.confidence_milli):
        return MeterCompositionResult(REJECTED, None, None, None, (), (R_NONFINITE,))
    if presence.status == REJECTED:
        return MeterCompositionResult(REJECTED, None, None, None, (), (R_PRESENCE_REJECTED,))
    if presence.status == AMBIGUOUS:
        return MeterCompositionResult(AMBIGUOUS, None, None, None, (), (R_PRESENCE_AMBIGUOUS,))
    if not isinstance(presence.present, bool):
        return MeterCompositionResult(REJECTED, None, None, None, (), (R_NONFINITE,))

    seen_ids: set[str] = set()
    for item in digits:
        if not isinstance(item, MeterDigitObservation):
            return MeterCompositionResult(REJECTED, None, None, None, (), (R_NONFINITE,))
        if not item.observation_id or item.observation_id in seen_ids:
            return MeterCompositionResult(REJECTED, None, None, None, (), (R_DIGIT_COUNT_CONFLICT,))
        seen_ids.add(item.observation_id)
        if item.status not in OBSERVATION_STATUSES or not _valid_confidence(item.confidence_milli):
            return MeterCompositionResult(REJECTED, None, None, None, (), (R_NONFINITE,))
        if item.status == REJECTED:
            return MeterCompositionResult(REJECTED, None, None, None, (), (R_DIGIT_REJECTED,))
        if item.status == ACCEPTED:
            if item.digit not in METER_V2_DIGITS:
                return MeterCompositionResult(REJECTED, None, None, None, (), (R_UNSUPPORTED_DIGIT,))
            if not _valid_box(item.bbox):
                return MeterCompositionResult(REJECTED, None, None, None, (), (R_INVALID_BBOX,))

    if any(item.status == AMBIGUOUS for item in digits):
        return MeterCompositionResult(AMBIGUOUS, None, None, None, (), (R_DIGIT_AMBIGUOUS,))

    accepted_digits = tuple(item for item in digits if item.status == ACCEPTED)

    if not presence.present:
        if accepted_digits:
            return MeterCompositionResult(AMBIGUOUS, None, None, None, (), (R_PRESENCE_CONFLICT,))
        return MeterCompositionResult(ACCEPTED, "none", None, None, (), ())

    if len(accepted_digits) < 2:
        return MeterCompositionResult(AMBIGUOUS, None, None, None, (), (R_MISSING_DIGIT,))
    if len(accepted_digits) > 2:
        return MeterCompositionResult(AMBIGUOUS, None, None, None, (), (R_DIGIT_COUNT_CONFLICT,))

    a, b = accepted_digits
    assert a.bbox is not None and b.bbox is not None
    if not _x_grouped(a.bbox, b.bbox):
        return MeterCompositionResult(AMBIGUOUS, None, None, None, (), (R_GEOMETRY_AMBIGUOUS,))

    center_y_a = (float(a.bbox.y0) + float(a.bbox.y1)) / 2.0
    center_y_b = (float(b.bbox.y0) + float(b.bbox.y1)) / 2.0
    if center_y_a == center_y_b:
        return MeterCompositionResult(AMBIGUOUS, None, None, None, (), (R_GEOMETRY_AMBIGUOUS,))

    upper, lower = (a, b) if center_y_a < center_y_b else (b, a)
    numerator = int(upper.digit)  # validated above
    denominator = int(lower.digit)

    if denominator != 4 or numerator not in (2, 3, 4):
        return MeterCompositionResult(
            AMBIGUOUS, None, None, None, (), (R_UNSUPPORTED_COMPOSITION,)
        )

    meter_class = f"{numerator}/{denominator}"
    return MeterCompositionResult(
        ACCEPTED,
        meter_class,
        numerator,
        denominator,
        (upper.observation_id, lower.observation_id),
        (),
    )


def resolver_connection_allowed() -> bool:
    """This shadow package never authorizes runtime Resolver wiring."""
    return False
