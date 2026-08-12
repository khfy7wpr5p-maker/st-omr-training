"""Immutable Stage 1-C score-structure objects for ST-OMR training.

This module adds score/part/measure/voice structure only. It does not generate
music, serialize MusicXML, render notation, create datasets, or train models.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Final

from .core import ChordEvent, NoteEvent, RestEvent


V1_TIME_SIGNATURES: Final[frozenset[tuple[int, int]]] = frozenset(
    {(2, 4), (3, 4), (4, 4)}
)
V1_INSTRUMENT_CLASS: Final[str] = "generic_treble_staff"


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_positive_int(name: str, value: object) -> int:
    if not _is_plain_int(value) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _normalize_nonnegative_fraction(name: str, value: object) -> Fraction:
    if _is_plain_int(value):
        result = Fraction(value, 1)
    elif isinstance(value, Fraction):
        result = value
    else:
        raise TypeError(f"{name} must be an int or fractions.Fraction")
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


class Clef(str, Enum):
    TREBLE = "treble"


@dataclass(frozen=True, slots=True)
class TimeSignature:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if not _is_plain_int(self.numerator) or not _is_plain_int(self.denominator):
            raise TypeError("time signature values must be integers")
        if (self.numerator, self.denominator) not in V1_TIME_SIGNATURES:
            raise ValueError("V1 time signature must be 2/4, 3/4, or 4/4")

    @property
    def capacity(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


MusicEvent = NoteEvent | RestEvent | ChordEvent


@dataclass(frozen=True, slots=True)
class Voice:
    voice_id: int
    events: tuple[MusicEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "voice_id", _require_positive_int("voice_id", self.voice_id))
        if not isinstance(self.events, tuple):
            raise TypeError("events must be an immutable tuple")
        if any(not isinstance(event, (NoteEvent, RestEvent, ChordEvent)) for event in self.events):
            raise TypeError("every voice event must be NoteEvent, RestEvent, or ChordEvent")


@dataclass(frozen=True, slots=True)
class Measure:
    number: int
    time_signature: TimeSignature
    voices: tuple[Voice, ...]
    key_signature: int = 0
    clef: Clef = Clef.TREBLE
    expected_duration: Fraction | int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "number", _require_positive_int("measure number", self.number))
        if not isinstance(self.time_signature, TimeSignature):
            raise TypeError("time_signature must be TimeSignature")
        if not isinstance(self.voices, tuple):
            raise TypeError("voices must be an immutable tuple")
        if any(not isinstance(voice, Voice) for voice in self.voices):
            raise TypeError("every measure voice must be Voice")
        if not _is_plain_int(self.key_signature):
            raise TypeError("key_signature must be an integer")
        if not isinstance(self.clef, Clef):
            raise TypeError("clef must be Clef")

        expected = (
            self.time_signature.capacity
            if self.expected_duration is None
            else _normalize_nonnegative_fraction("expected_duration", self.expected_duration)
        )
        if expected <= 0:
            raise ValueError("expected_duration must be strictly positive")
        object.__setattr__(self, "expected_duration", expected)


@dataclass(frozen=True, slots=True)
class Part:
    part_id: str
    measures: tuple[Measure, ...]
    instrument_class: str = V1_INSTRUMENT_CLASS
    staff_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "part_id", _require_nonempty_string("part_id", self.part_id))
        if not isinstance(self.measures, tuple):
            raise TypeError("measures must be an immutable tuple")
        if any(not isinstance(measure, Measure) for measure in self.measures):
            raise TypeError("every part measure must be Measure")
        object.__setattr__(
            self,
            "instrument_class",
            _require_nonempty_string("instrument_class", self.instrument_class),
        )
        object.__setattr__(self, "staff_count", _require_positive_int("staff_count", self.staff_count))


@dataclass(frozen=True, slots=True)
class Score:
    score_id: str
    schema_version: str
    generator_version: str
    seed: int
    provenance: tuple[tuple[str, str], ...]
    parts: tuple[Part, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "score_id", _require_nonempty_string("score_id", self.score_id))
        object.__setattr__(
            self,
            "schema_version",
            _require_nonempty_string("schema_version", self.schema_version),
        )
        object.__setattr__(
            self,
            "generator_version",
            _require_nonempty_string("generator_version", self.generator_version),
        )
        if not _is_plain_int(self.seed):
            raise TypeError("seed must be an integer")
        if not isinstance(self.provenance, tuple):
            raise TypeError("provenance must be an immutable tuple of key/value pairs")
        for item in self.provenance:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(isinstance(value, str) for value in item)
            ):
                raise TypeError("every provenance entry must be a two-string tuple")
        if not isinstance(self.parts, tuple):
            raise TypeError("parts must be an immutable tuple")
        if any(not isinstance(part, Part) for part in self.parts):
            raise TypeError("every score part must be Part")
