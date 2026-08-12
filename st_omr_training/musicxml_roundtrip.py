"""Supported-V1 semantic round-trip verification for ST-OMR MusicXML.

Stage 2-D intentionally does not provide a general MusicXML importer. It only
projects already Stage-2-C-valid ST-OMR V1 MusicXML into the frozen semantic
surface needed to compare it with the canonical in-memory score.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import xml.etree.ElementTree as ET

from .core import ChordEvent, DisplayAccidental, NoteEvent, RestEvent
from .musicxml_validator import validate_musicxml
from .musicxml_writer import MusicXMLWriteError, write_musicxml
from .structure import Score
from .structure_validator import validate_score
from .validator import ValidationIssue, ValidationResult


@dataclass(frozen=True, slots=True)
class SemanticPitchProjection:
    step: str
    alter: int
    octave: int
    display_accidental: DisplayAccidental


@dataclass(frozen=True, slots=True)
class SemanticEventProjection:
    event_type: str
    onset: Fraction
    duration: Fraction
    staff: int
    pitches: tuple[SemanticPitchProjection, ...]


@dataclass(frozen=True, slots=True)
class SemanticVoiceProjection:
    voice_id: int
    events: tuple[SemanticEventProjection, ...]


@dataclass(frozen=True, slots=True)
class SemanticMeasureProjection:
    number: int
    time_signature: tuple[int, int]
    key_signature: int
    clef: str
    voices: tuple[SemanticVoiceProjection, ...]


@dataclass(frozen=True, slots=True)
class SemanticPartProjection:
    part_id: str
    staff_count: int
    measures: tuple[SemanticMeasureProjection, ...]


@dataclass(frozen=True, slots=True)
class SemanticScoreProjection:
    parts: tuple[SemanticPartProjection, ...]


class SupportedV1RoundTripError(ValueError):
    """Raised when a value cannot enter the supported-V1 projection boundary."""

    def __init__(self, message: str, validation: ValidationResult | None = None) -> None:
        super().__init__(message)
        self.validation = validation or ValidationResult()


def _issue(code: str, path: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, path=path, message=message)


def _project_pitch(note: NoteEvent) -> SemanticPitchProjection:
    return SemanticPitchProjection(
        step=note.pitch.step,
        alter=note.pitch.alter,
        octave=note.pitch.octave,
        display_accidental=note.notation_intent.display_accidental,
    )


def project_score_semantics(score: object) -> SemanticScoreProjection:
    """Project a canonical V1 Score onto the frozen Stage-2-D comparison surface."""

    validation = validate_score(score)
    if not validation.is_valid:
        raise SupportedV1RoundTripError(
            "canonical score failed independent V1 validation",
            validation,
        )
    assert isinstance(score, Score)

    projected_parts: list[SemanticPartProjection] = []
    for part in score.parts:
        projected_measures: list[SemanticMeasureProjection] = []
        for measure in part.measures:
            projected_voices: list[SemanticVoiceProjection] = []
            for voice in measure.voices:
                projected_events: list[SemanticEventProjection] = []
                for event in voice.events:
                    if isinstance(event, NoteEvent):
                        event_type = "note"
                        pitches = (_project_pitch(event),)
                    elif isinstance(event, RestEvent):
                        event_type = "rest"
                        pitches = ()
                    elif isinstance(event, ChordEvent):
                        event_type = "chord"
                        pitches = tuple(_project_pitch(note) for note in event.notes)
                    else:
                        raise SupportedV1RoundTripError("unsupported canonical event type")

                    projected_events.append(
                        SemanticEventProjection(
                            event_type=event_type,
                            onset=event.onset,
                            duration=event.duration.fraction,
                            staff=event.staff,
                            pitches=pitches,
                        )
                    )
                projected_voices.append(
                    SemanticVoiceProjection(
                        voice_id=voice.voice_id,
                        events=tuple(projected_events),
                    )
                )

            projected_measures.append(
                SemanticMeasureProjection(
                    number=measure.number,
                    time_signature=(
                        measure.time_signature.numerator,
                        measure.time_signature.denominator,
                    ),
                    key_signature=measure.key_signature,
                    clef=measure.clef.value,
                    voices=tuple(projected_voices),
                )
            )

        projected_parts.append(
            SemanticPartProjection(
                part_id=part.part_id,
                staff_count=part.staff_count,
                measures=tuple(projected_measures),
            )
        )

    return SemanticScoreProjection(parts=tuple(projected_parts))


def _required_int(element: ET.Element, child_name: str) -> int:
    child = element.find(child_name)
    if child is None or child.text is None:
        raise SupportedV1RoundTripError(f"validated V1 element is missing {child_name}")
    try:
        return int(child.text)
    except ValueError as exc:
        raise SupportedV1RoundTripError(f"validated V1 integer is invalid: {child_name}") from exc


def _parse_pitch(note: ET.Element) -> SemanticPitchProjection:
    pitch = note.find("pitch")
    if pitch is None:
        raise SupportedV1RoundTripError("validated pitched note is missing pitch")
    step = pitch.findtext("step")
    octave_text = pitch.findtext("octave")
    if step is None or octave_text is None:
        raise SupportedV1RoundTripError("validated pitch is incomplete")
    alter_text = pitch.findtext("alter")
    accidental_text = note.findtext("accidental")
    try:
        accidental = (
            DisplayAccidental.NONE
            if accidental_text is None
            else DisplayAccidental(accidental_text)
        )
        return SemanticPitchProjection(
            step=step,
            alter=0 if alter_text is None else int(alter_text),
            octave=int(octave_text),
            display_accidental=accidental,
        )
    except (ValueError, TypeError) as exc:
        raise SupportedV1RoundTripError("validated V1 pitch projection failed") from exc


def _parse_note_record(note: ET.Element, divisions: int) -> dict[str, object]:
    duration_units = _required_int(note, "duration")
    duration = Fraction(duration_units, 4 * divisions)
    voice = _required_int(note, "voice")
    staff = _required_int(note, "staff")
    chord = note.find("chord") is not None
    is_rest = note.find("rest") is not None
    pitch = None if is_rest else _parse_pitch(note)
    return {
        "chord": chord,
        "rest": is_rest,
        "duration": duration,
        "voice": voice,
        "staff": staff,
        "pitch": pitch,
    }


def parse_supported_v1_musicxml_projection(data: object) -> SemanticScoreProjection:
    """Parse only Stage-2-C-valid ST-OMR V1 MusicXML into a semantic projection.

    Unsupported or noncanonical MusicXML is rejected by the independent Stage 2-C
    gates before this limited parser examines the document.
    """

    validation = validate_musicxml(data)
    if not validation.is_valid:
        raise SupportedV1RoundTripError(
            "MusicXML failed Stage 2-C XSD/semantic validation",
            validation,
        )
    assert isinstance(data, bytes)

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise SupportedV1RoundTripError("validated MusicXML could not be parsed") from exc

    part = root.find("part")
    if part is None:
        raise SupportedV1RoundTripError("validated V1 MusicXML is missing P1")

    measures = part.findall("measure")
    first_attributes = measures[0].find("attributes")
    if first_attributes is None:
        raise SupportedV1RoundTripError("validated V1 MusicXML is missing first attributes")
    divisions = _required_int(first_attributes, "divisions")
    first_time = first_attributes.find("time")
    if first_time is None:
        raise SupportedV1RoundTripError("validated V1 MusicXML is missing first time signature")
    active_time = (_required_int(first_time, "beats"), _required_int(first_time, "beat-type"))

    projected_measures: list[SemanticMeasureProjection] = []
    for measure in measures:
        attributes = measure.find("attributes")
        if attributes is not None:
            time = attributes.find("time")
            if time is not None:
                active_time = (_required_int(time, "beats"), _required_int(time, "beat-type"))

        notes = [child for child in measure if child.tag == "note"]
        projected_events: list[SemanticEventProjection] = []
        cursor = Fraction(0, 1)
        index = 0
        while index < len(notes):
            base = _parse_note_record(notes[index], divisions)
            if bool(base["chord"]):
                raise SupportedV1RoundTripError("validated V1 chord continuation has no base note")

            duration = base["duration"]
            assert isinstance(duration, Fraction)
            staff = base["staff"]
            voice = base["voice"]
            assert isinstance(staff, int) and isinstance(voice, int)

            if bool(base["rest"]):
                projected_events.append(
                    SemanticEventProjection(
                        event_type="rest",
                        onset=cursor,
                        duration=duration,
                        staff=staff,
                        pitches=(),
                    )
                )
                cursor += duration
                index += 1
                continue

            base_pitch = base["pitch"]
            assert isinstance(base_pitch, SemanticPitchProjection)
            pitches = [base_pitch]
            next_index = index + 1
            while next_index < len(notes):
                continuation = _parse_note_record(notes[next_index], divisions)
                if not bool(continuation["chord"]):
                    break
                continuation_pitch = continuation["pitch"]
                if not isinstance(continuation_pitch, SemanticPitchProjection):
                    raise SupportedV1RoundTripError("validated V1 chord member is not pitched")
                pitches.append(continuation_pitch)
                next_index += 1

            projected_events.append(
                SemanticEventProjection(
                    event_type="chord" if len(pitches) > 1 else "note",
                    onset=cursor,
                    duration=duration,
                    staff=staff,
                    pitches=tuple(pitches),
                )
            )
            cursor += duration
            index = next_index

        projected_measures.append(
            SemanticMeasureProjection(
                number=int(measure.attrib["number"]),
                time_signature=active_time,
                key_signature=0,
                clef="treble",
                voices=(SemanticVoiceProjection(voice_id=1, events=tuple(projected_events)),),
            )
        )

    return SemanticScoreProjection(
        parts=(
            SemanticPartProjection(
                part_id=part.attrib["id"],
                staff_count=1,
                measures=tuple(projected_measures),
            ),
        )
    )


def compare_semantic_projections(expected: object, actual: object) -> ValidationResult:
    """Compare every field frozen by the Stage-2-D round-trip contract."""

    if not isinstance(expected, SemanticScoreProjection) or not isinstance(actual, SemanticScoreProjection):
        return ValidationResult(
            (_issue("roundtrip.projection_type", "$", "both values must be SemanticScoreProjection"),)
        )

    issues: list[ValidationIssue] = []

    def check(code: str, path: str, left: object, right: object, label: str) -> None:
        if left != right:
            issues.append(_issue(code, path, f"{label} differs: expected {left!r}, got {right!r}"))

    check("roundtrip.part_count", "$.parts", len(expected.parts), len(actual.parts), "part count")
    for part_index, (left_part, right_part) in enumerate(zip(expected.parts, actual.parts)):
        part_path = f"$.parts[{part_index}]"
        check("roundtrip.part_id", f"{part_path}.part_id", left_part.part_id, right_part.part_id, "part id")
        check("roundtrip.staff_count", f"{part_path}.staff_count", left_part.staff_count, right_part.staff_count, "staff count")
        check("roundtrip.measure_count", f"{part_path}.measures", len(left_part.measures), len(right_part.measures), "measure count")
        for measure_index, (left_measure, right_measure) in enumerate(zip(left_part.measures, right_part.measures)):
            measure_path = f"{part_path}.measures[{measure_index}]"
            check("roundtrip.measure_number", f"{measure_path}.number", left_measure.number, right_measure.number, "measure number")
            check("roundtrip.time_signature", f"{measure_path}.time_signature", left_measure.time_signature, right_measure.time_signature, "time signature")
            check("roundtrip.key_signature", f"{measure_path}.key_signature", left_measure.key_signature, right_measure.key_signature, "key signature")
            check("roundtrip.clef", f"{measure_path}.clef", left_measure.clef, right_measure.clef, "clef")
            check("roundtrip.voice_count", f"{measure_path}.voices", len(left_measure.voices), len(right_measure.voices), "voice count")
            for voice_index, (left_voice, right_voice) in enumerate(zip(left_measure.voices, right_measure.voices)):
                voice_path = f"{measure_path}.voices[{voice_index}]"
                check("roundtrip.voice_id", f"{voice_path}.voice_id", left_voice.voice_id, right_voice.voice_id, "voice id")
                check("roundtrip.event_count", f"{voice_path}.events", len(left_voice.events), len(right_voice.events), "event count")
                for event_index, (left_event, right_event) in enumerate(zip(left_voice.events, right_voice.events)):
                    event_path = f"{voice_path}.events[{event_index}]"
                    check("roundtrip.event_type", f"{event_path}.event_type", left_event.event_type, right_event.event_type, "event type")
                    check("roundtrip.onset", f"{event_path}.onset", left_event.onset, right_event.onset, "onset")
                    check("roundtrip.duration", f"{event_path}.duration", left_event.duration, right_event.duration, "duration")
                    check("roundtrip.staff", f"{event_path}.staff", left_event.staff, right_event.staff, "staff")
                    check("roundtrip.pitch_count", f"{event_path}.pitches", len(left_event.pitches), len(right_event.pitches), "pitch/member count")
                    for pitch_index, (left_pitch, right_pitch) in enumerate(zip(left_event.pitches, right_event.pitches)):
                        pitch_path = f"{event_path}.pitches[{pitch_index}]"
                        check("roundtrip.pitch_step", f"{pitch_path}.step", left_pitch.step, right_pitch.step, "pitch step")
                        check("roundtrip.pitch_alter", f"{pitch_path}.alter", left_pitch.alter, right_pitch.alter, "pitch alter")
                        check("roundtrip.pitch_octave", f"{pitch_path}.octave", left_pitch.octave, right_pitch.octave, "pitch octave")
                        check(
                            "roundtrip.display_accidental",
                            f"{pitch_path}.display_accidental",
                            left_pitch.display_accidental,
                            right_pitch.display_accidental,
                            "visible accidental intent",
                        )

    return ValidationResult(tuple(issues))


def verify_supported_v1_round_trip(score: object) -> ValidationResult:
    """Writer -> Stage 2-C validation -> limited parser -> semantic comparison."""

    canonical_validation = validate_score(score)
    if not canonical_validation.is_valid:
        return ValidationResult(
            (
                _issue(
                    "roundtrip.canonical_invalid",
                    "$",
                    "canonical score failed independent V1 validation",
                ),
            )
            + canonical_validation.issues
        )

    try:
        expected = project_score_semantics(score)
        data = write_musicxml(score)
        actual = parse_supported_v1_musicxml_projection(data)
    except MusicXMLWriteError:
        return ValidationResult(
            (_issue("roundtrip.writer_rejected", "$", "Stage 2-B writer rejected canonical score"),)
        )
    except SupportedV1RoundTripError as exc:
        return ValidationResult(
            (
                _issue(
                    "roundtrip.musicxml_rejected",
                    "$",
                    "generated MusicXML failed the supported-V1 projection boundary",
                ),
            )
            + exc.validation.issues
        )

    return compare_semantic_projections(expected, actual)
