"""TR-POLY-06 strict parser and lossless tokenizer for Polyphonic Representation V2.

The codec is intentionally additive. It does not alter the frozen V1 tokenizer.
It provides two deterministic round-trip surfaces for the V2 object model:

1. strict canonical JSON <-> ``PolyScore``;
2. a closed-vocabulary structured token stream <-> canonical V2 payload.

The structured token stream is not raw JSON text. Known semantic field names are
first-class key tokens; arbitrary text and integers are encoded with bounded
byte/digit sub-tokens. This keeps the vocabulary closed while preserving event
IDs, source measure labels and other open-text evidence exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Final, Iterable

from .polyphonic_representation import (
    Barline,
    BarlineLocation,
    BarlineStyle,
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
    PolyphonicRepresentationError,
    POLYPHONIC_REPRESENTATION_VERSION,
    StemDirection,
    TieState,
    TimeSignature,
    TupletBoundary,
    TupletMark,
)


POLYPHONIC_SERIALIZATION_VERSION: Final[str] = "st-omr-polyphonic-serialization-v1"
POLYPHONIC_TOKENIZER_VERSION: Final[str] = "st-omr-polyphonic-tokenizer-v1"
MAX_CANONICAL_JSON_BYTES: Final[int] = 4_000_000
MAX_TOKEN_COUNT: Final[int] = 12_000_000
MAX_NESTING_DEPTH: Final[int] = 64
MAX_TEXT_BYTES: Final[int] = 1_000_000


class PolyphonicSerializationError(ValueError):
    """Raised when V2 serialized evidence violates the frozen codec contract."""


_PAYLOAD_KEYS: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            "actual_notes",
            "alter",
            "atom_id",
            "barlines",
            "beams",
            "beat_type",
            "beats",
            "boundary",
            "clefs",
            "denominator",
            "display_accidental",
            "dots",
            "duration",
            "event_id",
            "events",
            "fifths",
            "grace",
            "kind",
            "key_signature",
            "level",
            "line",
            "location",
            "measure_index",
            "measures",
            "mode",
            "normal_notes",
            "note_type",
            "noteheads",
            "number",
            "numerator",
            "octave",
            "octave_change",
            "onset",
            "part_id",
            "parts",
            "pitch",
            "repeat_direction",
            "representation_version",
            "sign",
            "slash",
            "source_number",
            "staff",
            "staff_count",
            "staff_override",
            "state",
            "stem",
            "step",
            "style",
            "ties",
            "time_signature",
            "tuplets",
            "voice",
        }
    )
)

_CONTROL_TOKENS: Final[tuple[str, ...]] = (
    "PAD",
    "BOS",
    "EOS",
    "OBJ_START",
    "OBJ_END",
    "ARR_START",
    "ARR_END",
    "NULL",
    "TRUE",
    "FALSE",
    "INT_START",
    "INT_NEG",
    "INT_END",
    "TEXT_START",
    "TEXT_END",
)
_DIGIT_TOKENS: Final[tuple[str, ...]] = tuple(f"DIGIT_{value}" for value in range(10))
_BYTE_TOKENS: Final[tuple[str, ...]] = tuple(f"BYTE_{value:02X}" for value in range(256))
_KEY_TOKENS: Final[tuple[str, ...]] = tuple(f"KEY:{key}" for key in _PAYLOAD_KEYS)
TOKEN_VOCABULARY: Final[tuple[str, ...]] = (
    _CONTROL_TOKENS + _DIGIT_TOKENS + _BYTE_TOKENS + _KEY_TOKENS
)
if len(TOKEN_VOCABULARY) != len(set(TOKEN_VOCABULARY)):
    raise RuntimeError("TR-POLY-06 token vocabulary contains duplicates")
TOKEN_TO_ID = MappingProxyType({token: index for index, token in enumerate(TOKEN_VOCABULARY)})
ID_TO_TOKEN = MappingProxyType({index: token for token, index in TOKEN_TO_ID.items()})
PAD_TOKEN_ID: Final[int] = TOKEN_TO_ID["PAD"]
BOS_TOKEN_ID: Final[int] = TOKEN_TO_ID["BOS"]
EOS_TOKEN_ID: Final[int] = TOKEN_TO_ID["EOS"]
VOCABULARY_SIZE: Final[int] = len(TOKEN_VOCABULARY)


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PolyphonicSerializationError("payload is not canonical-JSON serializable") from exc


def tokenizer_fingerprint() -> str:
    payload = {
        "serialization_version": POLYPHONIC_SERIALIZATION_VERSION,
        "tokenizer_version": POLYPHONIC_TOKENIZER_VERSION,
        "representation_version": POLYPHONIC_REPRESENTATION_VERSION,
        "vocabulary": list(TOKEN_VOCABULARY),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _mapping(value: object, path: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PolyphonicSerializationError(f"{path} must be an object")
    supplied = set(value)
    if supplied != keys:
        missing = sorted(keys - supplied)
        unknown = sorted(supplied - keys)
        raise PolyphonicSerializationError(
            f"{path} has invalid keys; missing={missing}, unknown={unknown}"
        )
    if any(not isinstance(key, str) for key in value):
        raise PolyphonicSerializationError(f"{path} object keys must be text")
    return value


def _list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise PolyphonicSerializationError(f"{path} must be an array")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise PolyphonicSerializationError(f"{path} must be text")
    return value


def _plain_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise PolyphonicSerializationError(f"{path} must be a plain integer")
    return value


def _optional_int(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _plain_int(value, path)


def _optional_text(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _text(value, path)


def _optional_bool(value: object, path: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise PolyphonicSerializationError(f"{path} must be bool or null")
    return value


def _enum(enum_type: type, value: object, path: str):
    text = _text(value, path)
    try:
        return enum_type(text)
    except ValueError as exc:
        raise PolyphonicSerializationError(f"{path} has unsupported value {text!r}") from exc


def _rational(value: object, path: str) -> ExactRational:
    item = _mapping(value, path, {"numerator", "denominator"})
    return ExactRational(
        _plain_int(item["numerator"], f"{path}.numerator"),
        _plain_int(item["denominator"], f"{path}.denominator"),
    )


def _pitch(value: object, path: str) -> PitchSpelling:
    item = _mapping(value, path, {"step", "alter", "octave", "display_accidental"})
    return PitchSpelling(
        step=_text(item["step"], f"{path}.step"),
        alter=_plain_int(item["alter"], f"{path}.alter"),
        octave=_plain_int(item["octave"], f"{path}.octave"),
        display_accidental=_enum(
            DisplayAccidentalV2,
            item["display_accidental"],
            f"{path}.display_accidental",
        ),
    )


def _note_atom(value: object, path: str) -> NoteAtom:
    item = _mapping(value, path, {"atom_id", "pitch", "ties", "staff_override"})
    ties = tuple(
        _enum(TieState, tie, f"{path}.ties[{index}]")
        for index, tie in enumerate(_list(item["ties"], f"{path}.ties"))
    )
    return NoteAtom(
        atom_id=_text(item["atom_id"], f"{path}.atom_id"),
        pitch=_pitch(item["pitch"], f"{path}.pitch"),
        ties=ties,
        staff_override=_optional_int(item["staff_override"], f"{path}.staff_override"),
    )


def _beam(value: object, path: str) -> BeamMark:
    item = _mapping(value, path, {"level", "state"})
    return BeamMark(
        level=_plain_int(item["level"], f"{path}.level"),
        state=_enum(BeamState, item["state"], f"{path}.state"),
    )


def _tuplet(value: object, path: str) -> TupletMark:
    item = _mapping(
        value,
        path,
        {"number", "actual_notes", "normal_notes", "boundary"},
    )
    return TupletMark(
        number=_plain_int(item["number"], f"{path}.number"),
        actual_notes=_plain_int(item["actual_notes"], f"{path}.actual_notes"),
        normal_notes=_plain_int(item["normal_notes"], f"{path}.normal_notes"),
        boundary=_enum(TupletBoundary, item["boundary"], f"{path}.boundary"),
    )


def _grace(value: object, path: str) -> GraceSpec | None:
    if value is None:
        return None
    item = _mapping(value, path, {"slash"})
    return GraceSpec(slash=_optional_bool(item["slash"], f"{path}.slash"))


def _event(value: object, path: str) -> PolyEvent:
    item = _mapping(
        value,
        path,
        {
            "event_id",
            "kind",
            "onset",
            "duration",
            "voice",
            "staff",
            "note_type",
            "noteheads",
            "dots",
            "stem",
            "beams",
            "tuplets",
            "grace",
        },
    )
    note_type = None if item["note_type"] is None else _enum(
        NoteType, item["note_type"], f"{path}.note_type"
    )
    stem = None if item["stem"] is None else _enum(
        StemDirection, item["stem"], f"{path}.stem"
    )
    return PolyEvent(
        event_id=_text(item["event_id"], f"{path}.event_id"),
        kind=_enum(EventKind, item["kind"], f"{path}.kind"),
        onset=_rational(item["onset"], f"{path}.onset"),
        duration=_rational(item["duration"], f"{path}.duration"),
        voice=_plain_int(item["voice"], f"{path}.voice"),
        staff=_plain_int(item["staff"], f"{path}.staff"),
        note_type=note_type,
        noteheads=tuple(
            _note_atom(note, f"{path}.noteheads[{index}]")
            for index, note in enumerate(_list(item["noteheads"], f"{path}.noteheads"))
        ),
        dots=_plain_int(item["dots"], f"{path}.dots"),
        stem=stem,
        beams=tuple(
            _beam(mark, f"{path}.beams[{index}]")
            for index, mark in enumerate(_list(item["beams"], f"{path}.beams"))
        ),
        tuplets=tuple(
            _tuplet(mark, f"{path}.tuplets[{index}]")
            for index, mark in enumerate(_list(item["tuplets"], f"{path}.tuplets"))
        ),
        grace=_grace(item["grace"], f"{path}.grace"),
    )


def _clef(value: object, path: str) -> ClefAssignment:
    item = _mapping(value, path, {"staff", "sign", "line", "octave_change"})
    return ClefAssignment(
        staff=_plain_int(item["staff"], f"{path}.staff"),
        sign=_text(item["sign"], f"{path}.sign"),
        line=_optional_int(item["line"], f"{path}.line"),
        octave_change=_plain_int(item["octave_change"], f"{path}.octave_change"),
    )


def _key_signature(value: object, path: str) -> KeySignature:
    item = _mapping(value, path, {"fifths", "mode"})
    return KeySignature(
        fifths=_plain_int(item["fifths"], f"{path}.fifths"),
        mode=_optional_text(item["mode"], f"{path}.mode"),
    )


def _time_signature(value: object, path: str) -> TimeSignature:
    item = _mapping(value, path, {"beats", "beat_type"})
    return TimeSignature(
        beats=tuple(
            _plain_int(beat, f"{path}.beats[{index}]")
            for index, beat in enumerate(_list(item["beats"], f"{path}.beats"))
        ),
        beat_type=_plain_int(item["beat_type"], f"{path}.beat_type"),
    )


def _barline(value: object, path: str) -> Barline:
    item = _mapping(value, path, {"location", "style", "repeat_direction"})
    return Barline(
        location=_enum(BarlineLocation, item["location"], f"{path}.location"),
        style=_enum(BarlineStyle, item["style"], f"{path}.style"),
        repeat_direction=_optional_text(item["repeat_direction"], f"{path}.repeat_direction"),
    )


def _measure(value: object, path: str) -> PolyMeasure:
    item = _mapping(
        value,
        path,
        {
            "measure_index",
            "source_number",
            "time_signature",
            "key_signature",
            "clefs",
            "events",
            "barlines",
        },
    )
    return PolyMeasure(
        measure_index=_plain_int(item["measure_index"], f"{path}.measure_index"),
        source_number=_text(item["source_number"], f"{path}.source_number"),
        time_signature=_time_signature(item["time_signature"], f"{path}.time_signature"),
        key_signature=_key_signature(item["key_signature"], f"{path}.key_signature"),
        clefs=tuple(
            _clef(clef, f"{path}.clefs[{index}]")
            for index, clef in enumerate(_list(item["clefs"], f"{path}.clefs"))
        ),
        events=tuple(
            _event(event, f"{path}.events[{index}]")
            for index, event in enumerate(_list(item["events"], f"{path}.events"))
        ),
        barlines=tuple(
            _barline(barline, f"{path}.barlines[{index}]")
            for index, barline in enumerate(_list(item["barlines"], f"{path}.barlines"))
        ),
    )


def _part(value: object, path: str) -> PolyPart:
    item = _mapping(value, path, {"part_id", "staff_count", "measures"})
    return PolyPart(
        part_id=_text(item["part_id"], f"{path}.part_id"),
        staff_count=_plain_int(item["staff_count"], f"{path}.staff_count"),
        measures=tuple(
            _measure(measure, f"{path}.measures[{index}]")
            for index, measure in enumerate(_list(item["measures"], f"{path}.measures"))
        ),
    )


def parse_polyphonic_payload(payload: object) -> PolyScore:
    """Parse a strict V2 canonical payload into the frozen object model."""

    root = _mapping(payload, "$", {"parts", "representation_version"})
    version = _text(root["representation_version"], "$.representation_version")
    if version != POLYPHONIC_REPRESENTATION_VERSION:
        raise PolyphonicSerializationError("unsupported representation_version")
    try:
        return PolyScore(
            parts=tuple(
                _part(part, f"$.parts[{index}]")
                for index, part in enumerate(_list(root["parts"], "$.parts"))
            ),
            representation_version=version,
        )
    except PolyphonicRepresentationError as exc:
        raise PolyphonicSerializationError("payload violates Polyphonic Representation V2") from exc


def serialize_polyphonic_score(score: object) -> str:
    if not isinstance(score, PolyScore):
        raise PolyphonicSerializationError("score must be PolyScore")
    encoded = score.canonical_json().encode("ascii")
    if len(encoded) > MAX_CANONICAL_JSON_BYTES:
        raise PolyphonicSerializationError("canonical V2 JSON exceeds codec size limit")
    return encoded.decode("ascii")


def parse_canonical_polyphonic_json(data: object) -> PolyScore:
    """Parse only the exact canonical JSON emitted by V2.

    Equivalent-but-noncanonical JSON (whitespace, different key order, alternate
    escaping, or unknown fields) is rejected to preserve reproducible hashes.
    """

    if isinstance(data, bytes):
        if len(data) > MAX_CANONICAL_JSON_BYTES:
            raise PolyphonicSerializationError("canonical V2 JSON exceeds codec size limit")
        try:
            text = data.decode("ascii")
        except UnicodeDecodeError as exc:
            raise PolyphonicSerializationError("canonical V2 JSON must be ASCII") from exc
    elif isinstance(data, str):
        try:
            encoded = data.encode("ascii")
        except UnicodeEncodeError as exc:
            raise PolyphonicSerializationError("canonical V2 JSON must be ASCII") from exc
        if len(encoded) > MAX_CANONICAL_JSON_BYTES:
            raise PolyphonicSerializationError("canonical V2 JSON exceeds codec size limit")
        text = data
    else:
        raise PolyphonicSerializationError("canonical V2 JSON must be str or bytes")

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise PolyphonicSerializationError("invalid canonical V2 JSON") from exc
    score = parse_polyphonic_payload(payload)
    if score.canonical_json() != text:
        raise PolyphonicSerializationError("JSON is semantically valid but not canonical V2 serialization")
    return score


def _emit_integer(value: int, tokens: list[str]) -> None:
    tokens.append("INT_START")
    if value < 0:
        tokens.append("INT_NEG")
    for digit in str(abs(value)):
        tokens.append(f"DIGIT_{digit}")
    tokens.append("INT_END")


def _emit_text(value: str, tokens: list[str]) -> None:
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_TEXT_BYTES:
        raise PolyphonicSerializationError("text value exceeds tokenizer byte limit")
    tokens.append("TEXT_START")
    tokens.extend(f"BYTE_{byte:02X}" for byte in encoded)
    tokens.append("TEXT_END")


def _emit_value(value: object, tokens: list[str], depth: int) -> None:
    if depth > MAX_NESTING_DEPTH:
        raise PolyphonicSerializationError("payload exceeds tokenizer nesting limit")
    if value is None:
        tokens.append("NULL")
    elif value is True:
        tokens.append("TRUE")
    elif value is False:
        tokens.append("FALSE")
    elif isinstance(value, int) and not isinstance(value, bool):
        _emit_integer(value, tokens)
    elif isinstance(value, str):
        _emit_text(value, tokens)
    elif isinstance(value, list):
        tokens.append("ARR_START")
        for item in value:
            _emit_value(item, tokens, depth + 1)
        tokens.append("ARR_END")
    elif isinstance(value, dict):
        keys = sorted(value)
        if any(not isinstance(key, str) or key not in _PAYLOAD_KEYS for key in keys):
            raise PolyphonicSerializationError("payload contains an unsupported tokenizer key")
        tokens.append("OBJ_START")
        for key in keys:
            tokens.append(f"KEY:{key}")
            _emit_value(value[key], tokens, depth + 1)
        tokens.append("OBJ_END")
    else:
        raise PolyphonicSerializationError("payload contains unsupported tokenizer value type")
    if len(tokens) > MAX_TOKEN_COUNT:
        raise PolyphonicSerializationError("token stream exceeds codec token limit")


def encode_tokens(tokens: object, *, allow_pad: bool = False) -> tuple[int, ...]:
    if not isinstance(tokens, tuple) or any(not isinstance(token, str) for token in tokens):
        raise PolyphonicSerializationError("tokens must be an immutable tuple of strings")
    result: list[int] = []
    for token in tokens:
        token_id = TOKEN_TO_ID.get(token)
        if token_id is None:
            raise PolyphonicSerializationError(f"token outside frozen V2 vocabulary: {token!r}")
        if token == "PAD" and not allow_pad:
            raise PolyphonicSerializationError("PAD is batching-only")
        result.append(token_id)
    if len(result) > MAX_TOKEN_COUNT:
        raise PolyphonicSerializationError("token stream exceeds codec token limit")
    return tuple(result)


def decode_token_ids(token_ids: object, *, allow_pad: bool = False) -> tuple[str, ...]:
    if not isinstance(token_ids, tuple) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in token_ids
    ):
        raise PolyphonicSerializationError("token_ids must be an immutable tuple of plain integers")
    if len(token_ids) > MAX_TOKEN_COUNT:
        raise PolyphonicSerializationError("token stream exceeds codec token limit")
    tokens: list[str] = []
    for token_id in token_ids:
        token = ID_TO_TOKEN.get(token_id)
        if token is None:
            raise PolyphonicSerializationError("token id outside frozen V2 vocabulary")
        if token == "PAD" and not allow_pad:
            raise PolyphonicSerializationError("PAD is batching-only")
        tokens.append(token)
    return tuple(tokens)


@dataclass(frozen=True, slots=True)
class TokenizedPolyphonicTarget:
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    tokenizer_fingerprint: str
    representation_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, tuple) or len(self.tokens) < 3:
            raise PolyphonicSerializationError("tokens must be a non-empty immutable semantic sequence")
        if not isinstance(self.token_ids, tuple):
            raise PolyphonicSerializationError("token_ids must be an immutable tuple")
        if encode_tokens(self.tokens) != self.token_ids:
            raise PolyphonicSerializationError("tokens/token_ids mismatch")
        if self.tokens[0] != "BOS" or self.tokens[-1] != "EOS":
            raise PolyphonicSerializationError("semantic token target requires BOS/EOS")
        if self.tokenizer_fingerprint != tokenizer_fingerprint():
            raise PolyphonicSerializationError("tokenizer fingerprint mismatch")
        if (
            not isinstance(self.representation_sha256, str)
            or len(self.representation_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.representation_sha256)
        ):
            raise PolyphonicSerializationError("representation_sha256 must be lowercase SHA-256")


def tokenize_polyphonic_score(score: object) -> TokenizedPolyphonicTarget:
    if not isinstance(score, PolyScore):
        raise PolyphonicSerializationError("score must be PolyScore")
    serialize_polyphonic_score(score)  # applies canonical size gate
    tokens: list[str] = ["BOS"]
    _emit_value(score.canonical_payload(), tokens, 0)
    tokens.append("EOS")
    token_tuple = tuple(tokens)
    return TokenizedPolyphonicTarget(
        tokens=token_tuple,
        token_ids=encode_tokens(token_tuple),
        tokenizer_fingerprint=tokenizer_fingerprint(),
        representation_sha256=score.canonical_sha256(),
    )


@dataclass(slots=True)
class _TokenCursor:
    tokens: tuple[str, ...]
    index: int = 0

    def pop(self) -> str:
        if self.index >= len(self.tokens):
            raise PolyphonicSerializationError("unexpected end of V2 token stream")
        token = self.tokens[self.index]
        self.index += 1
        return token

    def peek(self) -> str:
        if self.index >= len(self.tokens):
            raise PolyphonicSerializationError("unexpected end of V2 token stream")
        return self.tokens[self.index]


def _parse_integer_tokens(cursor: _TokenCursor) -> int:
    negative = False
    if cursor.peek() == "INT_NEG":
        cursor.pop()
        negative = True
    digits: list[str] = []
    while True:
        token = cursor.pop()
        if token == "INT_END":
            break
        if not token.startswith("DIGIT_") or len(token) != 7 or token[-1] not in "0123456789":
            raise PolyphonicSerializationError("invalid integer token payload")
        digits.append(token[-1])
    if not digits:
        raise PolyphonicSerializationError("integer token payload is empty")
    if len(digits) > 1 and digits[0] == "0":
        raise PolyphonicSerializationError("integer token payload has noncanonical leading zero")
    if negative and digits == ["0"]:
        raise PolyphonicSerializationError("negative zero is not canonical")
    value = int("".join(digits))
    return -value if negative else value


def _parse_text_tokens(cursor: _TokenCursor) -> str:
    data = bytearray()
    while True:
        token = cursor.pop()
        if token == "TEXT_END":
            break
        if not token.startswith("BYTE_") or len(token) != 7:
            raise PolyphonicSerializationError("invalid text byte token")
        try:
            data.append(int(token[5:], 16))
        except ValueError as exc:
            raise PolyphonicSerializationError("invalid text byte token") from exc
        if len(data) > MAX_TEXT_BYTES:
            raise PolyphonicSerializationError("text value exceeds tokenizer byte limit")
    try:
        return bytes(data).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PolyphonicSerializationError("text token payload is not UTF-8") from exc


def _parse_value_tokens(cursor: _TokenCursor, depth: int) -> object:
    if depth > MAX_NESTING_DEPTH:
        raise PolyphonicSerializationError("token stream exceeds nesting limit")
    token = cursor.pop()
    if token == "NULL":
        return None
    if token == "TRUE":
        return True
    if token == "FALSE":
        return False
    if token == "INT_START":
        return _parse_integer_tokens(cursor)
    if token == "TEXT_START":
        return _parse_text_tokens(cursor)
    if token == "ARR_START":
        result: list[object] = []
        while cursor.peek() != "ARR_END":
            result.append(_parse_value_tokens(cursor, depth + 1))
        cursor.pop()
        return result
    if token == "OBJ_START":
        result: dict[str, object] = {}
        previous_key: str | None = None
        while cursor.peek() != "OBJ_END":
            key_token = cursor.pop()
            if not key_token.startswith("KEY:"):
                raise PolyphonicSerializationError("object token stream requires KEY token")
            key = key_token[4:]
            if key not in _PAYLOAD_KEYS:
                raise PolyphonicSerializationError("token stream contains unsupported key")
            if key in result:
                raise PolyphonicSerializationError("duplicate object key in token stream")
            if previous_key is not None and key <= previous_key:
                raise PolyphonicSerializationError("object keys must be strictly canonical-sorted")
            previous_key = key
            result[key] = _parse_value_tokens(cursor, depth + 1)
        cursor.pop()
        return result
    raise PolyphonicSerializationError(f"unexpected value token {token!r}")


def parse_polyphonic_tokens(tokens: object) -> PolyScore:
    if not isinstance(tokens, tuple) or any(not isinstance(token, str) for token in tokens):
        raise PolyphonicSerializationError("tokens must be an immutable tuple of strings")
    if len(tokens) > MAX_TOKEN_COUNT:
        raise PolyphonicSerializationError("token stream exceeds codec token limit")
    encode_tokens(tokens)  # vocabulary and PAD gate
    if len(tokens) < 3 or tokens[0] != "BOS" or tokens[-1] != "EOS":
        raise PolyphonicSerializationError("semantic token stream requires exactly one BOS/EOS envelope")
    if "BOS" in tokens[1:-1] or "EOS" in tokens[1:-1]:
        raise PolyphonicSerializationError("nested BOS/EOS tokens are invalid")
    cursor = _TokenCursor(tokens=tokens, index=1)
    payload = _parse_value_tokens(cursor, 0)
    if cursor.pop() != "EOS" or cursor.index != len(tokens):
        raise PolyphonicSerializationError("trailing tokens after V2 payload")
    return parse_polyphonic_payload(payload)


def detokenize_polyphonic_target(target: object) -> PolyScore:
    if not isinstance(target, TokenizedPolyphonicTarget):
        raise PolyphonicSerializationError("target must be TokenizedPolyphonicTarget")
    score = parse_polyphonic_tokens(target.tokens)
    if score.canonical_sha256() != target.representation_sha256:
        raise PolyphonicSerializationError("round-trip representation hash mismatch")
    return score


def detokenize_polyphonic_ids(token_ids: object) -> PolyScore:
    return parse_polyphonic_tokens(decode_token_ids(token_ids))


def validate_roundtrip(score: object) -> TokenizedPolyphonicTarget:
    """Fail closed unless JSON and token surfaces reconstruct the exact V2 score."""

    if not isinstance(score, PolyScore):
        raise PolyphonicSerializationError("score must be PolyScore")
    canonical = serialize_polyphonic_score(score)
    from_json = parse_canonical_polyphonic_json(canonical)
    if from_json != score or from_json.canonical_sha256() != score.canonical_sha256():
        raise PolyphonicSerializationError("canonical JSON round-trip changed V2 semantics")
    target = tokenize_polyphonic_score(score)
    from_tokens = detokenize_polyphonic_target(target)
    from_ids = detokenize_polyphonic_ids(target.token_ids)
    if from_tokens != score or from_ids != score:
        raise PolyphonicSerializationError("token round-trip changed V2 semantics")
    return target
