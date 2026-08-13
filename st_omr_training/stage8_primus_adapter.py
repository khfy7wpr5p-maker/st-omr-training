"""Fail-closed Stage 8-3A PrIMuS semantic/MEI to V1 MusicXML adapter.

This adapter is intentionally narrow. It accepts only a monophonic PrIMuS-style
semantic stream that is already inside the frozen ST-OMR V1 musical surface,
corroborates it against a matching MEI representation, builds the existing
canonical ST music model, independently validates that model, and serializes it
through the existing deterministic MusicXML writer.

It does not admit data, prove image pairing or rights, persist real bytes, open
a sealed test split, load a model/checkpoint, or run training.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import re
import xml.etree.ElementTree as ET
from typing import Final

from .core import (
    DisplayAccidental,
    NoteEvent,
    NotationIntent,
    Pitch,
    RationalDuration,
    RestEvent,
)
from .generator import DEFAULT_SCHEMA_VERSION, MAX_MEASURE_COUNT
from .musicxml_roundtrip import verify_supported_v1_round_trip
from .musicxml_validator import MAX_MUSICXML_BYTES, validate_musicxml
from .musicxml_writer import write_musicxml
from .structure import Clef, Measure, Part, Score, TimeSignature, Voice
from .structure_validator import validate_score


PRIMUS_V1_ADAPTER_VERSION: Final[str] = "st-stage8-primus-v1-adapter-v1"
MAX_AUXILIARY_BYTES: Final[int] = MAX_MUSICXML_BYTES
MAX_SEMANTIC_TOKENS: Final[int] = 4096
_SUPPORTED_METERS: Final[frozenset[tuple[int, int]]] = frozenset({(2, 4), (3, 4), (4, 4)})
_NOTE_DURATION_BY_NAME: Final[dict[str, Fraction]] = {
    "whole": Fraction(1, 1),
    "half": Fraction(1, 2),
    "quarter": Fraction(1, 4),
    "eighth": Fraction(1, 8),
}
_REST_DURATION_BY_NAME: Final[dict[str, Fraction]] = {
    "half": Fraction(1, 2),
    "quarter": Fraction(1, 4),
    "eighth": Fraction(1, 8),
}
_MEI_DURATION_TO_FRACTION: Final[dict[int, Fraction]] = {
    1: Fraction(1, 1),
    2: Fraction(1, 2),
    4: Fraction(1, 4),
    8: Fraction(1, 8),
}
_MEI_ACCIDENTAL_TO_ALTER: Final[dict[str, int]] = {"s": 1, "f": -1, "n": 0}
_NOTE_TOKEN = re.compile(r"note-([A-G])([#b]?)([0-9])_(whole|half|quarter|eighth)\Z")
_REST_TOKEN = re.compile(r"rest-(half|quarter|eighth)\Z")
_METER_TOKEN = re.compile(r"timeSignature-([234])/4\Z")
_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_XML_SURFACE = (b"<!DOCTYPE", b"<!ENTITY")
_FORBIDDEN_MEI_ELEMENTS: Final[frozenset[str]] = frozenset(
    {
        "beam",
        "chord",
        "dot",
        "mRest",
        "mSpace",
        "multiRest",
        "space",
        "tie",
        "slur",
        "tuplet",
        "tupletSpan",
    }
)


class PrimusV1AdapterError(ValueError):
    """Raised when an auxiliary package cannot safely enter the V1 adapter."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def primus_v1_adapter_policy_fingerprint() -> str:
    payload = {
        "adapter_version": PRIMUS_V1_ADAPTER_VERSION,
        "canonical_schema_version": DEFAULT_SCHEMA_VERSION,
        "semantic_header": ["clef-G2", "keySignature-CM", "timeSignature-{2|3|4}/4"],
        "meters": [list(item) for item in sorted(_SUPPORTED_METERS)],
        "note_durations": sorted(_NOTE_DURATION_BY_NAME),
        "rest_durations": sorted(_REST_DURATION_BY_NAME),
        "semantic_events": ["note", "rest", "barline"],
        "mei_event_children": ["note", "rest"],
        "accidentals": {"flat": -1, "natural": 0, "sharp": 1},
        "accidental_state": "key0-reset-each-measure",
        "max_auxiliary_bytes": MAX_AUXILIARY_BYTES,
        "max_semantic_tokens": MAX_SEMANTIC_TOKENS,
        "max_measures": MAX_MEASURE_COUNT,
        "validation_chain": [
            "mei-semantic-exact-event-corroboration",
            "canonical-independent-validator",
            "deterministic-musicxml-writer",
            "musicxml-xsd-and-v1-semantics",
            "supported-v1-roundtrip",
        ],
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_bytes(name: str, value: object) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise PrimusV1AdapterError(f"{name} must be non-empty bytes")
    if len(value) > MAX_AUXILIARY_BYTES:
        raise PrimusV1AdapterError(f"{name} exceeds the Stage 8-3A byte limit")
    return value


def _reject_forbidden_xml_surface(data: bytes) -> None:
    upper = data.upper()
    if any(marker in upper for marker in _FORBIDDEN_XML_SURFACE):
        raise PrimusV1AdapterError("DTD/entity declarations are forbidden in auxiliary MEI")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children_named(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _all_named(root: ET.Element, name: str) -> list[ET.Element]:
    return [node for node in root.iter() if _local_name(node.tag) == name]


def _duration(value: Fraction) -> RationalDuration:
    return RationalDuration(value.numerator, value.denominator)


@dataclass(frozen=True, slots=True)
class _SemanticEvent:
    kind: str
    duration: Fraction
    pitch: tuple[str, int, int] | None = None


@dataclass(frozen=True, slots=True)
class _ParsedSemantic:
    meter: tuple[int, int]
    measures: tuple[tuple[_SemanticEvent, ...], ...]

    @property
    def note_count(self) -> int:
        return sum(event.kind == "note" for measure in self.measures for event in measure)

    @property
    def rest_count(self) -> int:
        return sum(event.kind == "rest" for measure in self.measures for event in measure)


def _parse_semantic(data: bytes) -> _ParsedSemantic:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PrimusV1AdapterError("semantic annotation must be UTF-8") from exc

    tokens = tuple(token for token in re.split(r"\s+", text.strip()) if token)
    if not 4 <= len(tokens) <= MAX_SEMANTIC_TOKENS:
        raise PrimusV1AdapterError("semantic token count is outside the frozen adapter bounds")
    if tokens[0] != "clef-G2" or tokens[1] != "keySignature-CM":
        raise PrimusV1AdapterError("semantic header must be exactly G2 clef and key signature C major")
    meter_match = _METER_TOKEN.fullmatch(tokens[2])
    if meter_match is None:
        raise PrimusV1AdapterError("semantic meter must be exactly 2/4, 3/4, or 4/4")
    meter = (int(meter_match.group(1)), 4)
    if meter not in _SUPPORTED_METERS:
        raise PrimusV1AdapterError("semantic meter is outside V1")
    if any(
        token.startswith(("clef-", "keySignature-", "timeSignature-"))
        for token in tokens[3:]
    ):
        raise PrimusV1AdapterError("mid-stream or duplicate semantic headers are not supported")

    measures: list[tuple[_SemanticEvent, ...]] = []
    current: list[_SemanticEvent] = []
    for token in tokens[3:]:
        if token == "barline":
            if not current:
                raise PrimusV1AdapterError("semantic barline cannot close an empty measure")
            measures.append(tuple(current))
            current = []
            continue

        note_match = _NOTE_TOKEN.fullmatch(token)
        if note_match is not None:
            step, accidental, octave_text, duration_name = note_match.groups()
            alter = {"": 0, "#": 1, "b": -1}[accidental]
            current.append(
                _SemanticEvent(
                    kind="note",
                    duration=_NOTE_DURATION_BY_NAME[duration_name],
                    pitch=(step, alter, int(octave_text)),
                )
            )
            continue

        rest_match = _REST_TOKEN.fullmatch(token)
        if rest_match is not None:
            current.append(
                _SemanticEvent(
                    kind="rest",
                    duration=_REST_DURATION_BY_NAME[rest_match.group(1)],
                )
            )
            continue

        raise PrimusV1AdapterError(f"unsupported semantic token: {token}")

    if current:
        measures.append(tuple(current))
    if not measures:
        raise PrimusV1AdapterError("semantic annotation contains no measures")
    if len(measures) > MAX_MEASURE_COUNT:
        raise PrimusV1AdapterError("semantic annotation exceeds the V1 measure bound")

    capacity = Fraction(*meter)
    for index, measure in enumerate(measures, start=1):
        total = sum((event.duration for event in measure), Fraction(0, 1))
        if total != capacity:
            raise PrimusV1AdapterError(
                f"semantic measure {index} does not exactly fill its {meter[0]}/{meter[1]} capacity"
            )
    return _ParsedSemantic(meter=meter, measures=tuple(measures))


def _mei_header(root: ET.Element) -> tuple[tuple[int, int], ET.Element]:
    score_defs = _all_named(root, "scoreDef")
    staff_defs = _all_named(root, "staffDef")
    if len(score_defs) != 1 or len(staff_defs) != 1:
        raise PrimusV1AdapterError("MEI must contain exactly one scoreDef and one staffDef")
    score_def = score_defs[0]
    staff_def = staff_defs[0]
    if staff_def.attrib.get("clef.shape") != "G" or staff_def.attrib.get("clef.line") != "2":
        raise PrimusV1AdapterError("MEI clef must be exactly G2")
    if score_def.attrib.get("key.sig") not in {None, "", "0"}:
        raise PrimusV1AdapterError("MEI key signature must be zero")
    if score_def.attrib.get("meter.sym") not in {None, ""}:
        raise PrimusV1AdapterError("MEI meter symbols are outside the first V1 adapter")
    try:
        meter = (int(score_def.attrib["meter.count"]), int(score_def.attrib["meter.unit"]))
    except (KeyError, ValueError) as exc:
        raise PrimusV1AdapterError("MEI must provide a numeric meter") from exc
    if meter not in _SUPPORTED_METERS:
        raise PrimusV1AdapterError("MEI meter is outside V1")
    return meter, staff_def


def _mei_explicit_accidental(note: ET.Element) -> int | None:
    values: list[str] = []
    for attribute in ("accid.ges", "accid"):
        value = note.attrib.get(attribute)
        if value is not None:
            values.append(value)
    for child in note:
        if _local_name(child.tag) == "accid":
            value = child.attrib.get("accid.ges") or child.attrib.get("accid")
            if value is None:
                raise PrimusV1AdapterError("MEI accid element is missing a supported accidental value")
            values.append(value)
        elif _local_name(child.tag) not in set():
            raise PrimusV1AdapterError("MEI note contains unsupported child notation")
    if not values:
        return None
    converted: list[int] = []
    for value in values:
        try:
            converted.append(_MEI_ACCIDENTAL_TO_ALTER[value])
        except KeyError as exc:
            raise PrimusV1AdapterError("MEI contains an unsupported accidental") from exc
    if len(set(converted)) != 1:
        raise PrimusV1AdapterError("MEI accidental attributes disagree")
    return converted[0]


def _extract_mei_measures(root: ET.Element, meter: tuple[int, int]) -> tuple[tuple[_SemanticEvent, ...], ...]:
    for node in root.iter():
        name = _local_name(node.tag)
        if name in _FORBIDDEN_MEI_ELEMENTS:
            raise PrimusV1AdapterError(f"MEI contains deferred notation: {name}")
        if node.attrib.get("dots") not in {None, "", "0"}:
            raise PrimusV1AdapterError("dotted MEI durations are outside V1")

    measures = _all_named(root, "measure")
    if not 1 <= len(measures) <= MAX_MEASURE_COUNT:
        raise PrimusV1AdapterError("MEI measure count is outside the V1 bound")

    parsed_measures: list[tuple[_SemanticEvent, ...]] = []
    for measure_index, measure in enumerate(measures, start=1):
        direct_staff = _children_named(measure, "staff")
        if len(direct_staff) != 1 or any(_local_name(child.tag) != "staff" for child in measure):
            raise PrimusV1AdapterError("each MEI measure must contain exactly one direct staff")
        staff = direct_staff[0]
        if staff.attrib.get("n") not in {None, "1"}:
            raise PrimusV1AdapterError("MEI staff must be staff 1")
        direct_layer = _children_named(staff, "layer")
        if len(direct_layer) != 1 or any(_local_name(child.tag) != "layer" for child in staff):
            raise PrimusV1AdapterError("each MEI staff must contain exactly one direct layer")
        layer = direct_layer[0]
        if layer.attrib.get("n") not in {None, "1"}:
            raise PrimusV1AdapterError("MEI layer must be voice 1")

        accidental_state: dict[tuple[str, int], int] = {}
        events: list[_SemanticEvent] = []
        for child in layer:
            name = _local_name(child.tag)
            if name not in {"note", "rest"}:
                raise PrimusV1AdapterError(f"unsupported direct MEI layer event: {name}")
            try:
                mei_dur = int(child.attrib["dur"])
                duration = _MEI_DURATION_TO_FRACTION[mei_dur]
            except (KeyError, ValueError) as exc:
                raise PrimusV1AdapterError("MEI event has an unsupported duration") from exc

            if name == "rest":
                if duration not in set(_REST_DURATION_BY_NAME.values()):
                    raise PrimusV1AdapterError("MEI rest duration is outside V1")
                if len(child):
                    raise PrimusV1AdapterError("MEI rest contains unsupported child notation")
                events.append(_SemanticEvent(kind="rest", duration=duration))
                continue

            pname = child.attrib.get("pname")
            octave_text = child.attrib.get("oct")
            if pname not in tuple("abcdefg") or octave_text is None or not octave_text.isdigit():
                raise PrimusV1AdapterError("MEI note pitch is outside the V1 spelling surface")
            octave = int(octave_text)
            if not 0 <= octave <= 9:
                raise PrimusV1AdapterError("MEI note octave is outside V1")
            step = pname.upper()
            position = (step, octave)
            explicit_alter = _mei_explicit_accidental(child)
            if explicit_alter is not None:
                accidental_state[position] = explicit_alter
            alter = accidental_state.get(position, 0)
            events.append(
                _SemanticEvent(
                    kind="note",
                    duration=duration,
                    pitch=(step, alter, octave),
                )
            )

        if not events:
            raise PrimusV1AdapterError(f"MEI measure {measure_index} is empty")
        total = sum((event.duration for event in events), Fraction(0, 1))
        if total != Fraction(*meter):
            raise PrimusV1AdapterError(
                f"MEI measure {measure_index} does not exactly fill its meter"
            )
        parsed_measures.append(tuple(events))
    return tuple(parsed_measures)


def _parse_mei(data: bytes) -> tuple[tuple[int, int], tuple[tuple[_SemanticEvent, ...], ...]]:
    _reject_forbidden_xml_surface(data)
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise PrimusV1AdapterError("MEI is not well-formed XML") from exc
    if _local_name(root.tag) != "mei":
        raise PrimusV1AdapterError("auxiliary XML root must be MEI")
    meter, _ = _mei_header(root)
    return meter, _extract_mei_measures(root, meter)


def _display_for_transition(previous_alter: int, target_alter: int) -> DisplayAccidental:
    if target_alter == previous_alter:
        return DisplayAccidental.NONE
    if target_alter == 1:
        return DisplayAccidental.SHARP
    if target_alter == -1:
        return DisplayAccidental.FLAT
    return DisplayAccidental.NATURAL


def _build_score(parsed: _ParsedSemantic, *, mei_sha: str, semantic_sha: str) -> Score:
    measures: list[Measure] = []
    for measure_number, semantic_measure in enumerate(parsed.measures, start=1):
        cursor = Fraction(0, 1)
        accidental_state: dict[tuple[str, int], int] = {}
        events: list[NoteEvent | RestEvent] = []
        for source_event in semantic_measure:
            duration = _duration(source_event.duration)
            if source_event.kind == "rest":
                events.append(RestEvent(cursor, duration, voice=1, staff=1))
            else:
                assert source_event.pitch is not None
                step, alter, octave = source_event.pitch
                position = (step, octave)
                previous = accidental_state.get(position, 0)
                display = _display_for_transition(previous, alter)
                accidental_state[position] = alter
                events.append(
                    NoteEvent(
                        cursor,
                        duration,
                        Pitch(step, alter, octave),
                        NotationIntent(display),
                        voice=1,
                        staff=1,
                    )
                )
            cursor += source_event.duration

        time_signature = TimeSignature(*parsed.meter)
        measures.append(
            Measure(
                number=measure_number,
                time_signature=time_signature,
                voices=(Voice(voice_id=1, events=tuple(events)),),
                key_signature=0,
                clef=Clef.TREBLE,
                expected_duration=time_signature.capacity,
            )
        )

    identity = sha256(
        _canonical_json_bytes(
            {
                "adapter_version": PRIMUS_V1_ADAPTER_VERSION,
                "mei_sha256": mei_sha,
                "semantic_sha256": semantic_sha,
            }
        )
    ).hexdigest()
    score_id = f"st-real-{identity}"
    return Score(
        score_id=score_id,
        schema_version=DEFAULT_SCHEMA_VERSION,
        generator_version=PRIMUS_V1_ADAPTER_VERSION,
        seed=0,
        provenance=(
            ("adapter_version", PRIMUS_V1_ADAPTER_VERSION),
            ("created_by_pipeline", "stage8-primus-v1-adapter"),
            ("mei_sha256", mei_sha),
            ("semantic_sha256", semantic_sha),
            ("source_id", score_id),
            ("source_type", "primus-auxiliary"),
        ),
        parts=(Part(part_id="P1", measures=tuple(measures)),),
    )


@dataclass(frozen=True, slots=True)
class PrimusV1AdapterEvidence:
    mei_sha256: str
    semantic_sha256: str
    musicxml_sha256: str
    score_id: str
    measure_count: int
    note_count: int
    rest_count: int
    policy_fingerprint: str
    adapter_version: str = PRIMUS_V1_ADAPTER_VERSION

    def __post_init__(self) -> None:
        for name in ("mei_sha256", "semantic_sha256", "musicxml_sha256", "policy_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(ch not in _HEX for ch in value):
                raise PrimusV1AdapterError(f"{name} must be lowercase SHA-256")
        if not isinstance(self.score_id, str) or not self.score_id.startswith("st-real-"):
            raise PrimusV1AdapterError("score_id must be a deterministic real-data adapter id")
        if not 1 <= self.measure_count <= MAX_MEASURE_COUNT:
            raise PrimusV1AdapterError("measure_count is outside the adapter bounds")
        if self.note_count < 0 or self.rest_count < 0 or self.note_count + self.rest_count < 1:
            raise PrimusV1AdapterError("adapter evidence requires at least one event")
        if self.policy_fingerprint != primus_v1_adapter_policy_fingerprint():
            raise PrimusV1AdapterError("adapter policy fingerprint mismatch")
        if self.adapter_version != PRIMUS_V1_ADAPTER_VERSION:
            raise PrimusV1AdapterError("adapter version mismatch")


def adapt_primus_v1_to_musicxml(
    *,
    mei_bytes: object,
    semantic_bytes: object,
) -> tuple[bytes, PrimusV1AdapterEvidence]:
    """Convert one corroborated supported-V1 auxiliary pair to deterministic MusicXML.

    A successful return is still not Stage 8 admission. Rights, image pairing,
    PNG preparation, Stage 8-0 metadata admission, Stage 8-1 exact-byte checks,
    duplicate/leakage vetoes, and the 40/10 handoff remain separate gates.
    """

    mei = _require_bytes("MEI", mei_bytes)
    semantic = _require_bytes("semantic annotation", semantic_bytes)
    parsed_semantic = _parse_semantic(semantic)
    mei_meter, parsed_mei_measures = _parse_mei(mei)

    if mei_meter != parsed_semantic.meter:
        raise PrimusV1AdapterError("MEI and semantic meters disagree")
    if parsed_mei_measures != parsed_semantic.measures:
        raise PrimusV1AdapterError("MEI and semantic event streams disagree")

    mei_sha = sha256(mei).hexdigest()
    semantic_sha = sha256(semantic).hexdigest()
    score = _build_score(parsed_semantic, mei_sha=mei_sha, semantic_sha=semantic_sha)

    canonical_validation = validate_score(score)
    if not canonical_validation.is_valid:
        codes = ", ".join(issue.code for issue in canonical_validation.issues)
        raise PrimusV1AdapterError(f"adapted canonical score failed validation: {codes}")

    round_trip = verify_supported_v1_round_trip(score)
    if not round_trip.is_valid:
        codes = ", ".join(issue.code for issue in round_trip.issues)
        raise PrimusV1AdapterError(f"adapted score failed supported-V1 round trip: {codes}")

    musicxml = write_musicxml(score)
    musicxml_validation = validate_musicxml(musicxml)
    if not musicxml_validation.is_valid:
        codes = ", ".join(issue.code for issue in musicxml_validation.issues)
        raise PrimusV1AdapterError(f"adapted MusicXML failed validation: {codes}")

    evidence = PrimusV1AdapterEvidence(
        mei_sha256=mei_sha,
        semantic_sha256=semantic_sha,
        musicxml_sha256=sha256(musicxml).hexdigest(),
        score_id=score.score_id,
        measure_count=len(parsed_semantic.measures),
        note_count=parsed_semantic.note_count,
        rest_count=parsed_semantic.rest_count,
        policy_fingerprint=primus_v1_adapter_policy_fingerprint(),
    )
    return musicxml, evidence
