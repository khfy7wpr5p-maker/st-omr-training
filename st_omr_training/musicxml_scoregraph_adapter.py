"""Controlled MusicXML -> ScoreGraph V2 adapter.

This additive adapter reconstructs explicit onset/voice/staff semantics from
bounded score-partwise MusicXML. It does not change the frozen V1 importer or
tokenizer and it does not grant dataset admission/training authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import xml.etree.ElementTree as ET

from .musicxml_validator import MAX_MUSICXML_BYTES
from .polyphonic_representation import (
    BeamMark,
    BeamState,
    ClefAssignment,
    DisplayAccidentalV2,
    EventKind,
    ExactRational,
    GraceSpec,
    KeySignature,
    NoteAtom,
    NoteType,
    PitchSpelling,
    PolyEvent,
    PolyMeasure,
    PolyPart,
    PolyScore,
    StemDirection,
    TieState,
    TimeSignature,
)
from .scoregraph_v2 import (
    CapabilityOutcome,
    ScoreGraphCapability,
    ScoreGraphV2,
    SourceNavigationEvent,
    SourceNavigationKind,
)


class MusicXMLScoreGraphAdapterError(ValueError):
    """Raised when bounded MusicXML cannot enter the controlled adapter."""


def _plain_positive_int(text: str | None, label: str) -> int:
    if text is None:
        raise MusicXMLScoreGraphAdapterError(f"{label} is required")
    try:
        value = int(text)
    except ValueError as exc:
        raise MusicXMLScoreGraphAdapterError(f"{label} must be an integer") from exc
    if value <= 0:
        raise MusicXMLScoreGraphAdapterError(f"{label} must be positive")
    return value


def _duration(units: int, divisions: int) -> ExactRational:
    return ExactRational(units, divisions * 4)


def _note_type(note: ET.Element) -> NoteType | None:
    text = note.findtext("type")
    if text is None:
        return None
    try:
        return NoteType(text)
    except ValueError as exc:
        raise MusicXMLScoreGraphAdapterError(f"unsupported visible note type: {text}") from exc


def _pitch(note: ET.Element) -> PitchSpelling:
    pitch = note.find("pitch")
    if pitch is None:
        raise MusicXMLScoreGraphAdapterError("pitched note is missing pitch")
    step = pitch.findtext("step")
    octave_text = pitch.findtext("octave")
    if step is None or octave_text is None:
        raise MusicXMLScoreGraphAdapterError("pitch is incomplete")
    alter_text = pitch.findtext("alter")
    try:
        alter = 0 if alter_text is None else int(alter_text)
        octave = int(octave_text)
    except ValueError as exc:
        raise MusicXMLScoreGraphAdapterError("pitch alter/octave must be integer") from exc

    accidental_text = note.findtext("accidental")
    if accidental_text is None:
        accidental = DisplayAccidentalV2.NONE
    else:
        try:
            accidental = DisplayAccidentalV2(accidental_text)
        except ValueError as exc:
            raise MusicXMLScoreGraphAdapterError(
                f"unsupported display accidental: {accidental_text}"
            ) from exc
    return PitchSpelling(step, alter, octave, accidental)


def _ties(note: ET.Element) -> tuple[TieState, ...]:
    values: set[TieState] = set()
    for element in list(note.findall("tie")) + list(note.findall("./notations/tied")):
        raw = element.attrib.get("type")
        if raw in {"start", "stop"}:
            values.add(TieState(raw))
    return tuple(sorted(values, key=lambda item: item.value))


def _beams(note: ET.Element) -> tuple[BeamMark, ...]:
    result: list[BeamMark] = []
    for element in note.findall("beam"):
        number = _plain_positive_int(element.attrib.get("number", "1"), "beam number")
        raw = (element.text or "").strip()
        try:
            state = BeamState(raw)
        except ValueError as exc:
            raise MusicXMLScoreGraphAdapterError(f"unsupported beam state: {raw}") from exc
        result.append(BeamMark(number, state))
    return tuple(sorted(result, key=lambda item: item.level))


def _stem(note: ET.Element) -> StemDirection | None:
    raw = note.findtext("stem")
    if raw is None:
        return None
    try:
        return StemDirection(raw)
    except ValueError as exc:
        raise MusicXMLScoreGraphAdapterError(f"unsupported stem direction: {raw}") from exc


def _time_signature(attributes: ET.Element, active: TimeSignature | None) -> TimeSignature:
    element = attributes.find("time")
    if element is None:
        if active is None:
            raise MusicXMLScoreGraphAdapterError("first measure requires a time signature")
        return active
    beats_text = element.findtext("beats")
    beat_type = _plain_positive_int(element.findtext("beat-type"), "beat-type")
    if beats_text is None:
        raise MusicXMLScoreGraphAdapterError("time beats are required")
    try:
        beats = tuple(int(piece) for piece in beats_text.split("+"))
    except ValueError as exc:
        raise MusicXMLScoreGraphAdapterError("time beats must be positive integers") from exc
    if not beats or any(value <= 0 for value in beats):
        raise MusicXMLScoreGraphAdapterError("time beats must be positive")
    return TimeSignature(beats, beat_type)


def _key_signature(attributes: ET.Element, active: KeySignature | None) -> KeySignature:
    element = attributes.find("key")
    if element is None:
        return active if active is not None else KeySignature(0)
    fifths_text = element.findtext("fifths")
    if fifths_text is None:
        raise MusicXMLScoreGraphAdapterError("key fifths are required")
    try:
        fifths = int(fifths_text)
    except ValueError as exc:
        raise MusicXMLScoreGraphAdapterError("key fifths must be integer") from exc
    mode = element.findtext("mode")
    return KeySignature(fifths, mode)


def _declared_staff_count(part: ET.Element) -> int:
    maximum = 1
    for element in part.findall("./measure/attributes/staves"):
        if element.text is not None:
            maximum = max(maximum, _plain_positive_int(element.text, "staves"))
    for element in part.findall("./measure/attributes/clef"):
        raw = element.attrib.get("number")
        if raw is not None:
            maximum = max(maximum, _plain_positive_int(raw, "clef number"))
    for element in part.findall("./measure/note/staff"):
        if element.text is not None:
            maximum = max(maximum, _plain_positive_int(element.text, "staff"))
    return maximum


def _update_clefs(
    attributes: ET.Element,
    *,
    staff_count: int,
    active: dict[int, ClefAssignment],
) -> dict[int, ClefAssignment]:
    result = dict(active)
    clefs = attributes.findall("clef")
    for index, element in enumerate(clefs, start=1):
        staff = _plain_positive_int(element.attrib.get("number", str(index)), "clef number")
        sign = element.findtext("sign")
        if sign is None:
            raise MusicXMLScoreGraphAdapterError("clef sign is required")
        line_text = element.findtext("line")
        line = None if line_text is None else _plain_positive_int(line_text, "clef line")
        octave_text = element.findtext("clef-octave-change")
        try:
            octave_change = 0 if octave_text is None else int(octave_text)
        except ValueError as exc:
            raise MusicXMLScoreGraphAdapterError("clef octave change must be integer") from exc
        result[staff] = ClefAssignment(staff, sign, line, octave_change)

    if not result:
        raise MusicXMLScoreGraphAdapterError("first measure requires clef assignments")
    if set(result) != set(range(1, staff_count + 1)):
        raise MusicXMLScoreGraphAdapterError("clef assignments do not cover every staff")
    return result


@dataclass
class _EventBuilder:
    event_id: str
    onset: Fraction
    duration: Fraction
    voice: int
    staff: int
    note_type: NoteType | None
    atoms: list[NoteAtom]
    is_rest: bool
    dots: int
    stem: StemDirection | None
    beams: tuple[BeamMark, ...]
    grace: GraceSpec | None


def _parse_measure_events(
    measure: ET.Element,
    *,
    part_id: str,
    measure_index: int,
    divisions: int,
) -> tuple[tuple[PolyEvent, ...], tuple[SourceNavigationEvent, ...]]:
    cursor = Fraction(0, 1)
    builders: list[_EventBuilder] = []
    navigation: list[SourceNavigationEvent] = []
    last_base_index: int | None = None

    for sequence_index, child in enumerate(list(measure)):
        if child.tag in {"attributes", "barline", "direction", "print", "sound"}:
            continue

        if child.tag in {"backup", "forward"}:
            units = _plain_positive_int(child.findtext("duration"), f"{child.tag} duration")
            exact = _duration(units, divisions)
            delta = exact.fraction
            if child.tag == "backup":
                if delta > cursor:
                    raise MusicXMLScoreGraphAdapterError("backup moves before measure start")
                cursor -= delta
                kind = SourceNavigationKind.BACKUP
            else:
                cursor += delta
                kind = SourceNavigationKind.FORWARD
            navigation.append(
                SourceNavigationEvent(
                    kind=kind,
                    part_id=part_id,
                    measure_index=measure_index,
                    sequence_index=sequence_index,
                    duration=exact,
                )
            )
            last_base_index = None
            continue

        if child.tag != "note":
            raise MusicXMLScoreGraphAdapterError(
                f"unsupported measure child in controlled adapter: {child.tag}"
            )

        note = child
        is_chord = note.find("chord") is not None
        is_grace = note.find("grace") is not None
        is_rest = note.find("rest") is not None
        voice = _plain_positive_int(note.findtext("voice"), "voice")
        staff = _plain_positive_int(note.findtext("staff") or "1", "staff")
        if is_grace:
            duration_fraction = Fraction(0, 1)
        else:
            units = _plain_positive_int(note.findtext("duration"), "note duration")
            duration_fraction = _duration(units, divisions).fraction

        if is_chord:
            if last_base_index is None:
                raise MusicXMLScoreGraphAdapterError("chord continuation has no base note")
            base = builders[last_base_index]
            if is_rest:
                raise MusicXMLScoreGraphAdapterError("chord continuation cannot be rest")
            if base.is_rest or base.voice != voice or base.duration != duration_fraction:
                raise MusicXMLScoreGraphAdapterError("chord continuation differs from base event")
            atom = NoteAtom(
                atom_id=f"{part_id}:m{measure_index}:n{sequence_index}",
                pitch=_pitch(note),
                ties=_ties(note),
                staff_override=staff if staff != base.staff else None,
            )
            base.atoms.append(atom)
            continue

        onset = cursor
        note_type = _note_type(note)
        grace = None
        if is_grace:
            slash_raw = note.find("grace").attrib.get("slash")
            slash = None if slash_raw is None else slash_raw.lower() in {"yes", "true", "1"}
            grace = GraceSpec(slash=slash)
        atoms: list[NoteAtom] = []
        if not is_rest:
            atoms.append(
                NoteAtom(
                    atom_id=f"{part_id}:m{measure_index}:n{sequence_index}",
                    pitch=_pitch(note),
                    ties=_ties(note),
                )
            )
        builder = _EventBuilder(
            event_id=f"{part_id}:m{measure_index}:e{sequence_index}",
            onset=onset,
            duration=duration_fraction,
            voice=voice,
            staff=staff,
            note_type=note_type,
            atoms=atoms,
            is_rest=is_rest,
            dots=len(note.findall("dot")),
            stem=_stem(note),
            beams=_beams(note),
            grace=grace,
        )
        builders.append(builder)
        last_base_index = len(builders) - 1
        if not is_grace:
            cursor += duration_fraction

    events: list[PolyEvent] = []
    for item in builders:
        if item.is_rest:
            kind = EventKind.REST
        elif len(item.atoms) == 1:
            kind = EventKind.NOTE
        else:
            kind = EventKind.CHORD
        duration = (
            ExactRational(0, 1)
            if item.grace is not None
            else ExactRational(item.duration.numerator, item.duration.denominator)
        )
        events.append(
            PolyEvent(
                event_id=item.event_id,
                kind=kind,
                onset=ExactRational(item.onset.numerator, item.onset.denominator),
                duration=duration,
                voice=item.voice,
                staff=item.staff,
                note_type=item.note_type,
                noteheads=tuple(item.atoms),
                dots=item.dots,
                stem=item.stem,
                beams=item.beams,
                grace=item.grace,
            )
        )

    ordered_events = tuple(
        sorted(
            events,
            key=lambda event: (
                event.onset.fraction,
                event.voice,
                event.staff,
                event.event_id,
            ),
        )
    )
    return ordered_events, tuple(navigation)


def parse_musicxml_to_scoregraph(data: object) -> ScoreGraphV2:
    """Parse a bounded controlled MusicXML subset into additive ScoreGraph V2."""

    if not isinstance(data, bytes):
        raise MusicXMLScoreGraphAdapterError("MusicXML input must be bytes")
    if not data or len(data) > MAX_MUSICXML_BYTES:
        raise MusicXMLScoreGraphAdapterError("MusicXML input is empty or exceeds byte limit")
    if b"<!DOCTYPE" in data.upper():
        raise MusicXMLScoreGraphAdapterError("DOCTYPE is forbidden")

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise MusicXMLScoreGraphAdapterError("MusicXML is not well formed") from exc
    if root.tag != "score-partwise":
        raise MusicXMLScoreGraphAdapterError("controlled adapter requires score-partwise root")

    capabilities: list[ScoreGraphCapability] = []
    parts: list[PolyPart] = []
    all_navigation: list[SourceNavigationEvent] = []

    raw_parts = root.findall("part")
    if not raw_parts:
        raise MusicXMLScoreGraphAdapterError("score contains no parts")

    for part_index, raw_part in enumerate(raw_parts):
        part_id = raw_part.attrib.get("id")
        if not part_id:
            raise MusicXMLScoreGraphAdapterError("part id is required")
        staff_count = _declared_staff_count(raw_part)
        measures = raw_part.findall("measure")
        if not measures:
            raise MusicXMLScoreGraphAdapterError("part contains no measures")

        active_divisions: int | None = None
        active_time: TimeSignature | None = None
        active_key: KeySignature | None = None
        active_clefs: dict[int, ClefAssignment] = {}
        parsed_measures: list[PolyMeasure] = []

        for measure_offset, raw_measure in enumerate(measures, start=1):
            measure_path = f"$.parts[{part_index}].measures[{measure_offset - 1}]"
            source_number = raw_measure.attrib.get("number")
            if source_number is None or not source_number.strip():
                source_number = str(measure_offset)
                capabilities.append(
                    ScoreGraphCapability(
                        code="musicxml.measure_number_missing",
                        outcome=CapabilityOutcome.REVIEW_REQUIRED,
                        path=measure_path,
                    )
                )

            attributes = raw_measure.find("attributes")
            if attributes is not None:
                divisions_text = attributes.findtext("divisions")
                if divisions_text is not None:
                    active_divisions = _plain_positive_int(divisions_text, "divisions")
                active_time = _time_signature(attributes, active_time)
                active_key = _key_signature(attributes, active_key)
                active_clefs = _update_clefs(
                    attributes,
                    staff_count=staff_count,
                    active=active_clefs,
                )

            if active_divisions is None:
                raise MusicXMLScoreGraphAdapterError("divisions must be established before notes")
            if active_time is None:
                raise MusicXMLScoreGraphAdapterError("time signature must be established")
            if active_key is None:
                active_key = KeySignature(0)
            if not active_clefs:
                raise MusicXMLScoreGraphAdapterError("clefs must be established")

            events, navigation = _parse_measure_events(
                raw_measure,
                part_id=part_id,
                measure_index=measure_offset,
                divisions=active_divisions,
            )
            all_navigation.extend(navigation)
            parsed_measures.append(
                PolyMeasure(
                    measure_index=measure_offset,
                    source_number=source_number,
                    time_signature=active_time,
                    key_signature=active_key,
                    clefs=tuple(active_clefs[index] for index in sorted(active_clefs)),
                    events=events,
                )
            )

        parts.append(PolyPart(part_id, staff_count, tuple(parsed_measures)))

    navigation_tuple = tuple(
        sorted(
            all_navigation,
            key=lambda item: (
                item.part_id,
                item.measure_index,
                item.sequence_index,
                item.kind.value,
            ),
        )
    )
    capability_tuple = tuple(
        sorted(
            capabilities,
            key=lambda item: (item.code, item.path, item.outcome.value),
        )
    )
    return ScoreGraphV2(
        core=PolyScore(tuple(parts)),
        source_navigation=navigation_tuple,
        capabilities=capability_tuple,
    )
