"""Immutable canonical music primitives for ST-OMR training.

Stage 1-A intentionally contains no generator, MusicXML, rendering, dataset,
or model-training logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from math import gcd
from typing import Final


MIN_OCTAVE: Final[int] = 0
MAX_OCTAVE: Final[int] = 9
V1_ALTERS: Final[frozenset[int]] = frozenset({-1, 0, 1})


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_positive_index(name: str, value: object) -> int:
    if not _is_plain_int(value) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize_onset(value: object) -> Fraction:
    if _is_plain_int(value):
        result = Fraction(value, 1)
    elif isinstance(value, Fraction):
        result = value
    else:
        raise TypeError("onset must be an int or fractions.Fraction")
    if result < 0:
        raise ValueError("onset must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class RationalDuration:
    """Exact, strictly positive musical duration."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if not _is_plain_int(self.numerator) or not _is_plain_int(self.denominator):
            raise TypeError("duration numerator and denominator must be integers")
        if self.denominator == 0:
            raise ValueError("duration denominator must not be zero")
        if self.denominator < 0:
            numerator = -self.numerator
            denominator = -self.denominator
        else:
            numerator = self.numerator
            denominator = self.denominator
        if numerator <= 0:
            raise ValueError("duration must be strictly positive")
        common = gcd(numerator, denominator)
        object.__setattr__(self, "numerator", numerator // common)
        object.__setattr__(self, "denominator", denominator // common)

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


@dataclass(frozen=True, slots=True)
class Pitch:
    """Structured V1 pitch spelling."""

    step: str
    alter: int
    octave: int

    def __post_init__(self) -> None:
        if not isinstance(self.step, str) or len(self.step) != 1:
            raise ValueError("step must be one letter A through G")
        step = self.step.upper()
        if step not in "ABCDEFG":
            raise ValueError("step must be one letter A through G")
        if not _is_plain_int(self.alter) or self.alter not in V1_ALTERS:
            raise ValueError("V1 alter must be one of -1, 0, +1")
        if not _is_plain_int(self.octave) or not (MIN_OCTAVE <= self.octave <= MAX_OCTAVE):
            raise ValueError(f"octave must be an integer from {MIN_OCTAVE} through {MAX_OCTAVE}")
        object.__setattr__(self, "step", step)


class DisplayAccidental(str, Enum):
    NONE = "none"
    SHARP = "sharp"
    FLAT = "flat"
    NATURAL = "natural"


@dataclass(frozen=True, slots=True)
class NotationIntent:
    """Visible notation intent kept separate from sounding pitch."""

    display_accidental: DisplayAccidental = DisplayAccidental.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.display_accidental, DisplayAccidental):
            raise TypeError("display_accidental must be a DisplayAccidental")


@dataclass(frozen=True, slots=True)
class NoteEvent:
    onset: Fraction | int
    duration: RationalDuration
    pitch: Pitch
    notation_intent: NotationIntent = NotationIntent()
    voice: int = 1
    staff: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "onset", _normalize_onset(self.onset))
        if not isinstance(self.duration, RationalDuration):
            raise TypeError("duration must be RationalDuration")
        if not isinstance(self.pitch, Pitch):
            raise TypeError("pitch must be Pitch")
        if not isinstance(self.notation_intent, NotationIntent):
            raise TypeError("notation_intent must be NotationIntent")
        object.__setattr__(self, "voice", _require_positive_index("voice", self.voice))
        object.__setattr__(self, "staff", _require_positive_index("staff", self.staff))


@dataclass(frozen=True, slots=True)
class RestEvent:
    onset: Fraction | int
    duration: RationalDuration
    voice: int = 1
    staff: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "onset", _normalize_onset(self.onset))
        if not isinstance(self.duration, RationalDuration):
            raise TypeError("duration must be RationalDuration")
        object.__setattr__(self, "voice", _require_positive_index("voice", self.voice))
        object.__setattr__(self, "staff", _require_positive_index("staff", self.staff))


@dataclass(frozen=True, slots=True)
class ChordEvent:
    """A V1 chord is one event with one shared onset/duration/voice/staff."""

    onset: Fraction | int
    duration: RationalDuration
    notes: tuple[Pitch, ...]
    voice: int = 1
    staff: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "onset", _normalize_onset(self.onset))
        if not isinstance(self.duration, RationalDuration):
            raise TypeError("duration must be RationalDuration")
        if not isinstance(self.notes, tuple):
            raise TypeError("notes must be an immutable tuple of Pitch values")
        if not 2 <= len(self.notes) <= 4:
            raise ValueError("V1 chords must contain 2 through 4 pitches")
        if any(not isinstance(note, Pitch) for note in self.notes):
            raise TypeError("every chord note must be Pitch")
        if len(set(self.notes)) != len(self.notes):
            raise ValueError("duplicate pitches are not allowed in a chord")
        object.__setattr__(self, "voice", _require_positive_index("voice", self.voice))
        object.__setattr__(self, "staff", _require_positive_index("staff", self.staff))
