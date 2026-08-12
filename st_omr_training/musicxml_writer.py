"""Deterministic MusicXML 4.0 writer for the supported ST-OMR V1 model.

This module serializes an independently validated canonical ``Score`` into
uncompressed MusicXML 4.0 ``score-partwise`` bytes. It intentionally contains
no XSD validation, renderer integration, dataset logic, or MusicXML importer.
"""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from math import lcm
from typing import Final
import xml.etree.ElementTree as ET

from .core import ChordEvent, DisplayAccidental, NoteEvent, RationalDuration, RestEvent
from .structure import Measure, Score
from .structure_validator import validate_score


MUSICXML_VERSION: Final[str] = "4.0"
MUSICXML_WRITER_VERSION: Final[str] = "st-musicxml-writer-v1"
MUSICXML_PART_ID: Final[str] = "P1"
MUSICXML_PART_NAME: Final[str] = "ST-OMR Synthetic"

_NOTE_TYPE_BY_DURATION: Final[dict[Fraction, str]] = {
    Fraction(1, 1): "whole",
    Fraction(1, 2): "half",
    Fraction(1, 4): "quarter",
    Fraction(1, 8): "eighth",
}
_ACCIDENTAL_TEXT: Final[dict[DisplayAccidental, str]] = {
    DisplayAccidental.SHARP: "sharp",
    DisplayAccidental.FLAT: "flat",
    DisplayAccidental.NATURAL: "natural",
}


class MusicXMLWriteError(RuntimeError):
    """Raised when a canonical score cannot safely enter MusicXML writing."""


def _require_valid_v1_score(score: object) -> Score:
    result = validate_score(score)
    if not result.is_valid:
        codes = ", ".join(issue.code for issue in result.issues)
        raise MusicXMLWriteError(
            f"score failed independent canonical validation before MusicXML writing: {codes}"
        )

    assert isinstance(score, Score)
    if score.parts[0].part_id != MUSICXML_PART_ID:
        raise MusicXMLWriteError(
            f"V1 MusicXML contract requires canonical part_id {MUSICXML_PART_ID!r}"
        )
    return score


def _event_duration(event: NoteEvent | RestEvent | ChordEvent) -> Fraction:
    duration = event.duration
    if not isinstance(duration, RationalDuration):
        raise MusicXMLWriteError("event duration is not RationalDuration")
    return duration.fraction


def compute_musicxml_divisions(score: object) -> int:
    """Return the smallest score-wide integer divisions value for V1 events.

    Canonical durations use whole-note fractions while MusicXML divisions are
    units per quarter note. For each duration ``d`` the quarter-unit value is
    ``4 * d``; the LCM of the reduced denominators is the exact divisions value.
    """

    canonical = _require_valid_v1_score(score)
    denominators: list[int] = []
    for part in canonical.parts:
        for measure in part.measures:
            for voice in measure.voices:
                for event in voice.events:
                    quarter_units = Fraction(4, 1) * _event_duration(event)
                    denominators.append(quarter_units.denominator)

    if not denominators:
        raise MusicXMLWriteError("validated score unexpectedly contains no events")

    divisions = 1
    for denominator in denominators:
        divisions = lcm(divisions, denominator)
    if divisions <= 0:
        raise MusicXMLWriteError("computed MusicXML divisions must be positive")
    return divisions


def _duration_units(duration: RationalDuration, divisions: int) -> int:
    value = duration.fraction * 4 * divisions
    if value.denominator != 1 or value.numerator <= 0:
        raise MusicXMLWriteError("duration does not map to exact positive MusicXML divisions")
    return value.numerator


def _note_type(duration: RationalDuration) -> str:
    try:
        return _NOTE_TYPE_BY_DURATION[duration.fraction]
    except KeyError as exc:
        raise MusicXMLWriteError("duration has no supported V1 MusicXML note type") from exc


def _append_pitch(parent: ET.Element, event: NoteEvent) -> None:
    pitch = ET.SubElement(parent, "pitch")
    ET.SubElement(pitch, "step").text = event.pitch.step
    if event.pitch.alter != 0:
        ET.SubElement(pitch, "alter").text = str(event.pitch.alter)
    ET.SubElement(pitch, "octave").text = str(event.pitch.octave)


def _append_accidental(parent: ET.Element, event: NoteEvent) -> None:
    accidental = event.notation_intent.display_accidental
    if accidental is DisplayAccidental.NONE:
        return
    try:
        text = _ACCIDENTAL_TEXT[accidental]
    except KeyError as exc:
        raise MusicXMLWriteError("unsupported V1 display accidental") from exc
    ET.SubElement(parent, "accidental").text = text


def _append_pitched_note(
    measure_element: ET.Element,
    event: NoteEvent,
    *,
    divisions: int,
    chord_continuation: bool,
) -> None:
    note = ET.SubElement(measure_element, "note")
    if chord_continuation:
        ET.SubElement(note, "chord")
    _append_pitch(note, event)
    ET.SubElement(note, "duration").text = str(_duration_units(event.duration, divisions))
    ET.SubElement(note, "voice").text = str(event.voice)
    ET.SubElement(note, "type").text = _note_type(event.duration)
    _append_accidental(note, event)
    ET.SubElement(note, "staff").text = str(event.staff)


def _append_rest(
    measure_element: ET.Element,
    event: RestEvent,
    *,
    divisions: int,
) -> None:
    note = ET.SubElement(measure_element, "note")
    ET.SubElement(note, "rest")
    ET.SubElement(note, "duration").text = str(_duration_units(event.duration, divisions))
    ET.SubElement(note, "voice").text = str(event.voice)
    ET.SubElement(note, "type").text = _note_type(event.duration)
    ET.SubElement(note, "staff").text = str(event.staff)


def _append_event(
    measure_element: ET.Element,
    event: NoteEvent | RestEvent | ChordEvent,
    *,
    divisions: int,
) -> None:
    if isinstance(event, NoteEvent):
        _append_pitched_note(
            measure_element,
            event,
            divisions=divisions,
            chord_continuation=False,
        )
        return
    if isinstance(event, RestEvent):
        _append_rest(measure_element, event, divisions=divisions)
        return
    if isinstance(event, ChordEvent):
        for index, member in enumerate(event.notes):
            _append_pitched_note(
                measure_element,
                member,
                divisions=divisions,
                chord_continuation=index > 0,
            )
        return
    raise MusicXMLWriteError("unsupported canonical event type")


def _append_time(attributes: ET.Element, measure: Measure) -> None:
    time = ET.SubElement(attributes, "time")
    ET.SubElement(time, "beats").text = str(measure.time_signature.numerator)
    ET.SubElement(time, "beat-type").text = str(measure.time_signature.denominator)


def _append_first_measure_attributes(
    measure_element: ET.Element,
    measure: Measure,
    *,
    divisions: int,
) -> None:
    attributes = ET.SubElement(measure_element, "attributes")
    ET.SubElement(attributes, "divisions").text = str(divisions)
    key = ET.SubElement(attributes, "key")
    ET.SubElement(key, "fifths").text = "0"
    _append_time(attributes, measure)
    clef = ET.SubElement(attributes, "clef")
    ET.SubElement(clef, "sign").text = "G"
    ET.SubElement(clef, "line").text = "2"


def _append_time_change_attributes(measure_element: ET.Element, measure: Measure) -> None:
    attributes = ET.SubElement(measure_element, "attributes")
    _append_time(attributes, measure)


def _build_tree(score: Score, divisions: int) -> ET.Element:
    root = ET.Element("score-partwise", {"version": MUSICXML_VERSION})

    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", {"id": MUSICXML_PART_ID})
    ET.SubElement(score_part, "part-name").text = MUSICXML_PART_NAME

    part = ET.SubElement(root, "part", {"id": MUSICXML_PART_ID})
    previous_signature: tuple[int, int] | None = None

    canonical_part = score.parts[0]
    for index, measure in enumerate(canonical_part.measures):
        measure_element = ET.SubElement(part, "measure", {"number": str(measure.number)})
        signature = (
            measure.time_signature.numerator,
            measure.time_signature.denominator,
        )
        if index == 0:
            _append_first_measure_attributes(
                measure_element,
                measure,
                divisions=divisions,
            )
        elif signature != previous_signature:
            _append_time_change_attributes(measure_element, measure)

        for event in measure.voices[0].events:
            _append_event(measure_element, event, divisions=divisions)
        previous_signature = signature

    return root


def write_musicxml(score: object) -> bytes:
    """Serialize one validated V1 canonical score to deterministic XML bytes."""

    canonical = _require_valid_v1_score(score)
    divisions = compute_musicxml_divisions(canonical)
    root = _build_tree(canonical, divisions)
    return ET.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
        short_empty_elements=True,
    )


def musicxml_sha256(data: object) -> str:
    """Return SHA-256 for already serialized MusicXML bytes."""

    if not isinstance(data, bytes):
        raise TypeError("MusicXML digest input must be bytes")
    return sha256(data).hexdigest()
