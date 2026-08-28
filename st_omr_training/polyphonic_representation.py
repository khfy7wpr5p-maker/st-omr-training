"""TR-POLY-05 versioned structured polyphonic score representation.

This module defines the additive V2 research representation only.  It does not
replace the frozen V1 tokenizer, parse MusicXML, tokenize model targets, train a
model, or access datasets.  TR-POLY-06 owns parsing and deterministic roundtrip.

Time is represented as exact fractions of a whole note.  Therefore a quarter
note is 1/4 and a 4/4 measure has capacity 1/1.  Onset is explicit rather than
being inferred from serialization order, so independent voices may share the
same onset without being confused with a chord.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from fractions import Fraction
from hashlib import sha256
import json
from math import gcd
from typing import Final


POLYPHONIC_REPRESENTATION_VERSION: Final[str] = "st-omr-polyphonic-representation-v2"
POLYPHONIC_TIME_UNIT: Final[str] = "whole-note-fraction"


class PolyphonicRepresentationError(ValueError):
    """Raised when a V2 score violates the frozen representation contract."""


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(name: str, value: object) -> int:
    if not _plain_int(value) or value <= 0:
        raise PolyphonicRepresentationError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if not _plain_int(value) or value < 0:
        raise PolyphonicRepresentationError(f"{name} must be a non-negative integer")
    return value


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolyphonicRepresentationError(f"{name} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True, order=True)
class ExactRational:
    """Canonical exact non-negative rational in whole-note units."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if not _plain_int(self.numerator) or not _plain_int(self.denominator):
            raise PolyphonicRepresentationError("rational numerator/denominator must be integers")
        if self.denominator == 0:
            raise PolyphonicRepresentationError("rational denominator must not be zero")
        numerator = self.numerator
        denominator = self.denominator
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        if numerator < 0:
            raise PolyphonicRepresentationError("rational value must be non-negative")
        common = gcd(numerator, denominator)
        object.__setattr__(self, "numerator", numerator // common)
        object.__setattr__(self, "denominator", denominator // common)

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    @property
    def is_zero(self) -> bool:
        return self.numerator == 0


class EventKind(str, Enum):
    NOTE = "note"
    REST = "rest"
    CHORD = "chord"


class DisplayAccidentalV2(str, Enum):
    NONE = "none"
    SHARP = "sharp"
    FLAT = "flat"
    NATURAL = "natural"
    DOUBLE_SHARP = "double-sharp"
    DOUBLE_FLAT = "double-flat"


class NoteType(str, Enum):
    MAXIMA = "maxima"
    LONG = "long"
    BREVE = "breve"
    WHOLE = "whole"
    HALF = "half"
    QUARTER = "quarter"
    EIGHTH = "eighth"
    N16 = "16th"
    N32 = "32nd"
    N64 = "64th"
    N128 = "128th"
    N256 = "256th"
    N512 = "512th"
    N1024 = "1024th"


class StemDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    NONE = "none"
    DOUBLE = "double"


class TieState(str, Enum):
    START = "start"
    STOP = "stop"


class BeamState(str, Enum):
    BEGIN = "begin"
    CONTINUE = "continue"
    END = "end"
    FORWARD_HOOK = "forward-hook"
    BACKWARD_HOOK = "backward-hook"


class TupletBoundary(str, Enum):
    START = "start"
    CONTINUE = "continue"
    STOP = "stop"


class BarlineLocation(str, Enum):
    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"


class BarlineStyle(str, Enum):
    REGULAR = "regular"
    DOTTED = "dotted"
    DASHED = "dashed"
    HEAVY = "heavy"
    LIGHT_LIGHT = "light-light"
    LIGHT_HEAVY = "light-heavy"
    HEAVY_LIGHT = "heavy-light"
    HEAVY_HEAVY = "heavy-heavy"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class PitchSpelling:
    step: str
    alter: int
    octave: int
    display_accidental: DisplayAccidentalV2 = DisplayAccidentalV2.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.step, str) or len(self.step) != 1 or self.step.upper() not in "ABCDEFG":
            raise PolyphonicRepresentationError("pitch step must be one letter A through G")
        if not _plain_int(self.alter) or not -2 <= self.alter <= 2:
            raise PolyphonicRepresentationError("V2 pitch alter must be an integer from -2 through +2")
        if not _plain_int(self.octave) or not 0 <= self.octave <= 9:
            raise PolyphonicRepresentationError("pitch octave must be an integer from 0 through 9")
        if not isinstance(self.display_accidental, DisplayAccidentalV2):
            raise PolyphonicRepresentationError("display_accidental must be DisplayAccidentalV2")
        object.__setattr__(self, "step", self.step.upper())


@dataclass(frozen=True, slots=True)
class NoteAtom:
    """One notehead inside a note/chord event.

    `staff_override` preserves cross-staff chord placement while `PolyEvent.voice`
    remains the logical voice identity.
    """

    atom_id: str
    pitch: PitchSpelling
    ties: tuple[TieState, ...] = ()
    staff_override: int | None = None

    def __post_init__(self) -> None:
        _text("atom_id", self.atom_id)
        if not isinstance(self.pitch, PitchSpelling):
            raise PolyphonicRepresentationError("pitch must be PitchSpelling")
        if not isinstance(self.ties, tuple) or any(not isinstance(item, TieState) for item in self.ties):
            raise PolyphonicRepresentationError("ties must be an immutable tuple of TieState values")
        if len(set(self.ties)) != len(self.ties):
            raise PolyphonicRepresentationError("duplicate tie state on note atom")
        if self.staff_override is not None:
            _positive_int("staff_override", self.staff_override)


@dataclass(frozen=True, slots=True)
class BeamMark:
    level: int
    state: BeamState

    def __post_init__(self) -> None:
        _positive_int("beam level", self.level)
        if not isinstance(self.state, BeamState):
            raise PolyphonicRepresentationError("beam state must be BeamState")


@dataclass(frozen=True, slots=True)
class TupletMark:
    number: int
    actual_notes: int
    normal_notes: int
    boundary: TupletBoundary

    def __post_init__(self) -> None:
        _positive_int("tuplet number", self.number)
        _positive_int("actual_notes", self.actual_notes)
        _positive_int("normal_notes", self.normal_notes)
        if not isinstance(self.boundary, TupletBoundary):
            raise PolyphonicRepresentationError("tuplet boundary must be TupletBoundary")


@dataclass(frozen=True, slots=True)
class GraceSpec:
    slash: bool | None = None

    def __post_init__(self) -> None:
        if self.slash is not None and not isinstance(self.slash, bool):
            raise PolyphonicRepresentationError("grace slash must be bool or None")


@dataclass(frozen=True, slots=True)
class PolyEvent:
    event_id: str
    kind: EventKind
    onset: ExactRational
    duration: ExactRational
    voice: int
    staff: int
    note_type: NoteType | None
    noteheads: tuple[NoteAtom, ...] = ()
    dots: int = 0
    stem: StemDirection | None = None
    beams: tuple[BeamMark, ...] = ()
    tuplets: tuple[TupletMark, ...] = ()
    grace: GraceSpec | None = None

    def __post_init__(self) -> None:
        _text("event_id", self.event_id)
        if not isinstance(self.kind, EventKind):
            raise PolyphonicRepresentationError("kind must be EventKind")
        if not isinstance(self.onset, ExactRational) or not isinstance(self.duration, ExactRational):
            raise PolyphonicRepresentationError("onset and duration must be ExactRational")
        _positive_int("voice", self.voice)
        _positive_int("staff", self.staff)
        if self.note_type is not None and not isinstance(self.note_type, NoteType):
            raise PolyphonicRepresentationError("note_type must be NoteType or None")
        if not isinstance(self.noteheads, tuple) or any(not isinstance(note, NoteAtom) for note in self.noteheads):
            raise PolyphonicRepresentationError("noteheads must be an immutable NoteAtom tuple")
        _nonnegative_int("dots", self.dots)
        if self.stem is not None and not isinstance(self.stem, StemDirection):
            raise PolyphonicRepresentationError("stem must be StemDirection or None")
        if not isinstance(self.beams, tuple) or any(not isinstance(mark, BeamMark) for mark in self.beams):
            raise PolyphonicRepresentationError("beams must be an immutable BeamMark tuple")
        if len({mark.level for mark in self.beams}) != len(self.beams):
            raise PolyphonicRepresentationError("an event may contain at most one beam mark per level")
        if not isinstance(self.tuplets, tuple) or any(not isinstance(mark, TupletMark) for mark in self.tuplets):
            raise PolyphonicRepresentationError("tuplets must be an immutable TupletMark tuple")
        if len({mark.number for mark in self.tuplets}) != len(self.tuplets):
            raise PolyphonicRepresentationError("an event may contain at most one tuplet mark per number")
        if self.grace is not None and not isinstance(self.grace, GraceSpec):
            raise PolyphonicRepresentationError("grace must be GraceSpec or None")

        if self.grace is None:
            if self.duration.is_zero:
                raise PolyphonicRepresentationError("non-grace event duration must be positive")
        elif not self.duration.is_zero:
            raise PolyphonicRepresentationError("grace events use zero semantic duration in V2")

        if self.kind is EventKind.REST:
            if self.noteheads:
                raise PolyphonicRepresentationError("rest must not contain noteheads")
            if self.grace is not None:
                raise PolyphonicRepresentationError("grace rest is unsupported in V2")
        elif self.kind is EventKind.NOTE:
            if len(self.noteheads) != 1:
                raise PolyphonicRepresentationError("note event must contain exactly one notehead")
        elif self.kind is EventKind.CHORD:
            if len(self.noteheads) < 2:
                raise PolyphonicRepresentationError("chord event must contain at least two noteheads")

        if self.kind is not EventKind.REST and self.note_type is None:
            raise PolyphonicRepresentationError("pitched V2 event requires visible note_type")
        if len({note.atom_id for note in self.noteheads}) != len(self.noteheads):
            raise PolyphonicRepresentationError("duplicate note atom id within event")

    @property
    def end(self) -> Fraction:
        return self.onset.fraction + self.duration.fraction

    @property
    def is_cross_staff(self) -> bool:
        return any(
            note.staff_override is not None and note.staff_override != self.staff
            for note in self.noteheads
        )


@dataclass(frozen=True, slots=True)
class ClefAssignment:
    staff: int
    sign: str
    line: int | None
    octave_change: int = 0

    def __post_init__(self) -> None:
        _positive_int("clef staff", self.staff)
        sign = _text("clef sign", self.sign).upper()
        if sign not in {"G", "F", "C", "TAB", "PERCUSSION", "NONE"}:
            raise PolyphonicRepresentationError("unsupported V2 clef sign")
        if self.line is not None:
            _positive_int("clef line", self.line)
        if not _plain_int(self.octave_change) or not -4 <= self.octave_change <= 4:
            raise PolyphonicRepresentationError("clef octave_change must be -4 through +4")
        object.__setattr__(self, "sign", sign)


@dataclass(frozen=True, slots=True)
class KeySignature:
    fifths: int
    mode: str | None = None

    def __post_init__(self) -> None:
        if not _plain_int(self.fifths) or not -7 <= self.fifths <= 7:
            raise PolyphonicRepresentationError("key fifths must be an integer from -7 through +7")
        if self.mode is not None:
            object.__setattr__(self, "mode", _text("key mode", self.mode).lower())


@dataclass(frozen=True, slots=True)
class TimeSignature:
    """Additive-capable time signature, e.g. beats=(3,2), beat_type=8 for 3+2/8."""

    beats: tuple[int, ...]
    beat_type: int

    def __post_init__(self) -> None:
        if not isinstance(self.beats, tuple) or not self.beats:
            raise PolyphonicRepresentationError("time beats must be a non-empty immutable tuple")
        for beat in self.beats:
            _positive_int("time beat", beat)
        _positive_int("beat_type", self.beat_type)

    @property
    def capacity(self) -> Fraction:
        return Fraction(sum(self.beats), self.beat_type)


@dataclass(frozen=True, slots=True)
class Barline:
    location: BarlineLocation
    style: BarlineStyle
    repeat_direction: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.location, BarlineLocation):
            raise PolyphonicRepresentationError("barline location must be BarlineLocation")
        if not isinstance(self.style, BarlineStyle):
            raise PolyphonicRepresentationError("barline style must be BarlineStyle")
        if self.repeat_direction not in {None, "forward", "backward"}:
            raise PolyphonicRepresentationError("repeat_direction must be forward, backward, or None")


@dataclass(frozen=True, slots=True)
class PolyMeasure:
    measure_index: int
    source_number: str
    time_signature: TimeSignature
    key_signature: KeySignature
    clefs: tuple[ClefAssignment, ...]
    events: tuple[PolyEvent, ...]
    barlines: tuple[Barline, ...] = ()

    def __post_init__(self) -> None:
        _positive_int("measure_index", self.measure_index)
        _text("source_number", self.source_number)
        if not isinstance(self.time_signature, TimeSignature):
            raise PolyphonicRepresentationError("time_signature must be TimeSignature")
        if not isinstance(self.key_signature, KeySignature):
            raise PolyphonicRepresentationError("key_signature must be KeySignature")
        if not isinstance(self.clefs, tuple) or not self.clefs or any(
            not isinstance(clef, ClefAssignment) for clef in self.clefs
        ):
            raise PolyphonicRepresentationError("clefs must be a non-empty immutable ClefAssignment tuple")
        if len({clef.staff for clef in self.clefs}) != len(self.clefs):
            raise PolyphonicRepresentationError("duplicate clef staff assignment in measure")
        if not isinstance(self.events, tuple) or any(not isinstance(event, PolyEvent) for event in self.events):
            raise PolyphonicRepresentationError("events must be an immutable PolyEvent tuple")
        if not isinstance(self.barlines, tuple) or any(not isinstance(item, Barline) for item in self.barlines):
            raise PolyphonicRepresentationError("barlines must be an immutable Barline tuple")
        if len({item.location for item in self.barlines}) != len(self.barlines):
            raise PolyphonicRepresentationError("duplicate barline location in measure")

        keys = tuple(
            (event.onset.fraction, event.voice, event.staff, event.event_id)
            for event in self.events
        )
        if keys != tuple(sorted(keys)):
            raise PolyphonicRepresentationError(
                "events must be in canonical onset/voice/staff/event_id order"
            )
        if len({event.event_id for event in self.events}) != len(self.events):
            raise PolyphonicRepresentationError("duplicate event id within measure")
        capacity = self.time_signature.capacity
        for event in self.events:
            if event.onset.fraction > capacity:
                raise PolyphonicRepresentationError("event onset exceeds measure capacity")
            if event.grace is None and event.end > capacity:
                raise PolyphonicRepresentationError("event end exceeds measure capacity")


@dataclass(frozen=True, slots=True)
class PolyPart:
    part_id: str
    staff_count: int
    measures: tuple[PolyMeasure, ...]

    def __post_init__(self) -> None:
        _text("part_id", self.part_id)
        _positive_int("staff_count", self.staff_count)
        if not isinstance(self.measures, tuple) or not self.measures or any(
            not isinstance(measure, PolyMeasure) for measure in self.measures
        ):
            raise PolyphonicRepresentationError("measures must be a non-empty immutable PolyMeasure tuple")
        if tuple(measure.measure_index for measure in self.measures) != tuple(
            range(1, len(self.measures) + 1)
        ):
            raise PolyphonicRepresentationError("measure_index must be sequential from 1")
        for measure in self.measures:
            for clef in measure.clefs:
                if clef.staff > self.staff_count:
                    raise PolyphonicRepresentationError("clef references staff outside part")
            for event in measure.events:
                if event.staff > self.staff_count:
                    raise PolyphonicRepresentationError("event references staff outside part")
                for note in event.noteheads:
                    if note.staff_override is not None and note.staff_override > self.staff_count:
                        raise PolyphonicRepresentationError("notehead staff_override references staff outside part")


@dataclass(frozen=True, slots=True)
class PolyScore:
    parts: tuple[PolyPart, ...]
    representation_version: str = POLYPHONIC_REPRESENTATION_VERSION

    def __post_init__(self) -> None:
        if self.representation_version != POLYPHONIC_REPRESENTATION_VERSION:
            raise PolyphonicRepresentationError("unsupported polyphonic representation version")
        if not isinstance(self.parts, tuple) or not self.parts or any(
            not isinstance(part, PolyPart) for part in self.parts
        ):
            raise PolyphonicRepresentationError("parts must be a non-empty immutable PolyPart tuple")
        if len({part.part_id for part in self.parts}) != len(self.parts):
            raise PolyphonicRepresentationError("duplicate part_id")

        event_ids: set[str] = set()
        atom_ids: set[str] = set()
        for part in self.parts:
            for measure in part.measures:
                for event in measure.events:
                    if event.event_id in event_ids:
                        raise PolyphonicRepresentationError("event_id must be globally unique")
                    event_ids.add(event.event_id)
                    for note in event.noteheads:
                        if note.atom_id in atom_ids:
                            raise PolyphonicRepresentationError("atom_id must be globally unique")
                        atom_ids.add(note.atom_id)

    def canonical_payload(self) -> dict[str, object]:
        return _jsonable(asdict(self))

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def canonical_sha256(self) -> str:
        return sha256(self.canonical_json().encode("ascii")).hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


REQUIRED_POLYPHONIC_SEMANTICS: Final[tuple[str, ...]] = (
    "measure",
    "staff",
    "voice",
    "pitch",
    "duration",
    "onset",
    "rest",
    "chord_grouping",
    "clef",
    "key_signature",
    "time_signature",
    "display_accidental",
    "beam",
    "tie",
    "tuplet",
    "grace",
    "barline",
    "cross_staff",
)
