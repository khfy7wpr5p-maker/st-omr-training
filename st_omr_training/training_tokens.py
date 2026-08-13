"""Frozen Stage 7-B semantic tokenizer for the ST-OMR V1 target surface."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Final

from .core import DisplayAccidental
from .musicxml_roundtrip import (
    SemanticEventProjection,
    SemanticMeasureProjection,
    SemanticPartProjection,
    SemanticPitchProjection,
    SemanticScoreProjection,
    SemanticVoiceProjection,
    compare_semantic_projections,
    parse_supported_v1_musicxml_projection,
)

TOKENIZER_VERSION: Final[str] = "st-omr-semantic-tokenizer-v1"

# PAD intentionally owns id 0 so sequence-loss masking is stable and explicit.
TOKEN_VOCABULARY: Final[tuple[str, ...]] = (
    "PAD",
    "BOS",
    "EOS",
    "MEASURE_START",
    "MEASURE_END",
    "TS_2_4",
    "TS_3_4",
    "TS_4_4",
    "NOTE",
    "REST",
    "CHORD_2",
    "CHORD_3",
    "CHORD_4",
    "DUR_WHOLE",
    "DUR_HALF",
    "DUR_QUARTER",
    "DUR_EIGHTH",
    "STEP_A",
    "STEP_B",
    "STEP_C",
    "STEP_D",
    "STEP_E",
    "STEP_F",
    "STEP_G",
    "ALTER_M1",
    "ALTER_0",
    "ALTER_P1",
    "OCT_3",
    "OCT_4",
    "OCT_5",
    "OCT_6",
    "ACC_NONE",
    "ACC_SHARP",
    "ACC_FLAT",
    "ACC_NATURAL",
)

if len(TOKEN_VOCABULARY) != len(set(TOKEN_VOCABULARY)):
    raise RuntimeError("Stage 7-B token vocabulary contains duplicates")

TOKEN_TO_ID = MappingProxyType({token: index for index, token in enumerate(TOKEN_VOCABULARY)})
ID_TO_TOKEN = MappingProxyType({index: token for token, index in TOKEN_TO_ID.items()})
PAD_TOKEN_ID: Final[int] = TOKEN_TO_ID["PAD"]
BOS_TOKEN_ID: Final[int] = TOKEN_TO_ID["BOS"]
EOS_TOKEN_ID: Final[int] = TOKEN_TO_ID["EOS"]
VOCABULARY_SIZE: Final[int] = len(TOKEN_VOCABULARY)

_DURATION_TO_TOKEN = MappingProxyType(
    {
        Fraction(1, 1): "DUR_WHOLE",
        Fraction(1, 2): "DUR_HALF",
        Fraction(1, 4): "DUR_QUARTER",
        Fraction(1, 8): "DUR_EIGHTH",
    }
)
_TOKEN_TO_DURATION = MappingProxyType({value: key for key, value in _DURATION_TO_TOKEN.items()})
_TIME_TO_TOKEN = MappingProxyType({(2, 4): "TS_2_4", (3, 4): "TS_3_4", (4, 4): "TS_4_4"})
_TOKEN_TO_TIME = MappingProxyType({value: key for key, value in _TIME_TO_TOKEN.items()})
_ALTER_TO_TOKEN = MappingProxyType({-1: "ALTER_M1", 0: "ALTER_0", 1: "ALTER_P1"})
_TOKEN_TO_ALTER = MappingProxyType({value: key for key, value in _ALTER_TO_TOKEN.items()})
_ACC_TO_TOKEN = MappingProxyType(
    {
        DisplayAccidental.NONE: "ACC_NONE",
        DisplayAccidental.SHARP: "ACC_SHARP",
        DisplayAccidental.FLAT: "ACC_FLAT",
        DisplayAccidental.NATURAL: "ACC_NATURAL",
    }
)
_TOKEN_TO_ACC = MappingProxyType({value: key for key, value in _ACC_TO_TOKEN.items()})
_ALLOWED_STEPS = frozenset("ABCDEFG")
_ALLOWED_OCTAVES = frozenset({3, 4, 5, 6})


class TokenizationError(ValueError):
    """Raised when a target cannot cross the frozen Stage 7-B token boundary."""


@dataclass(frozen=True, slots=True)
class TokenizedTarget:
    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    tokenizer_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, tuple) or not self.tokens:
            raise TokenizationError("tokens must be a non-empty immutable tuple")
        if not isinstance(self.token_ids, tuple) or not self.token_ids:
            raise TokenizationError("token_ids must be a non-empty immutable tuple")
        if len(self.tokens) != len(self.token_ids):
            raise TokenizationError("tokens/token_ids length mismatch")
        if encode_tokens(self.tokens) != self.token_ids:
            raise TokenizationError("token ids do not match the frozen vocabulary")
        if self.tokenizer_fingerprint != tokenizer_fingerprint():
            raise TokenizationError("tokenizer fingerprint mismatch")


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def tokenizer_fingerprint() -> str:
    payload = {
        "tokenizer_version": TOKENIZER_VERSION,
        "vocabulary": list(TOKEN_VOCABULARY),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def encode_tokens(tokens: object, *, allow_pad: bool = False) -> tuple[int, ...]:
    if not isinstance(tokens, tuple) or any(not isinstance(token, str) for token in tokens):
        raise TokenizationError("tokens must be an immutable tuple of strings")
    result: list[int] = []
    for token in tokens:
        if token not in TOKEN_TO_ID:
            raise TokenizationError(f"unknown Stage 7-B token: {token!r}")
        if token == "PAD" and not allow_pad:
            raise TokenizationError("PAD is batching-only and cannot enter a semantic target")
        result.append(TOKEN_TO_ID[token])
    return tuple(result)


def decode_token_ids(token_ids: object, *, allow_pad: bool = False) -> tuple[str, ...]:
    if (
        not isinstance(token_ids, tuple)
        or any(not isinstance(item, int) or isinstance(item, bool) for item in token_ids)
    ):
        raise TokenizationError("token_ids must be an immutable tuple of plain integers")
    tokens: list[str] = []
    for item in token_ids:
        token = ID_TO_TOKEN.get(item)
        if token is None:
            raise TokenizationError(f"token id outside frozen vocabulary: {item!r}")
        if token == "PAD" and not allow_pad:
            raise TokenizationError("PAD is batching-only and cannot enter a semantic target")
        tokens.append(token)
    return tuple(tokens)


def _measure_capacity(time_signature: tuple[int, int]) -> Fraction:
    if time_signature not in _TIME_TO_TOKEN:
        raise TokenizationError("unsupported V1 time signature")
    numerator, denominator = time_signature
    return Fraction(numerator, denominator)


def _validate_pitch(pitch: object) -> SemanticPitchProjection:
    if not isinstance(pitch, SemanticPitchProjection):
        raise TokenizationError("pitch projection has the wrong type")
    if pitch.step not in _ALLOWED_STEPS:
        raise TokenizationError("pitch step is outside V1")
    if pitch.alter not in _ALTER_TO_TOKEN:
        raise TokenizationError("pitch alter is outside V1")
    if pitch.octave not in _ALLOWED_OCTAVES:
        raise TokenizationError("pitch octave is outside V1")
    if not isinstance(pitch.display_accidental, DisplayAccidental):
        raise TokenizationError("display accidental is outside V1")
    if pitch.display_accidental is DisplayAccidental.SHARP and pitch.alter != 1:
        raise TokenizationError("sharp display intent requires alter +1")
    if pitch.display_accidental is DisplayAccidental.FLAT and pitch.alter != -1:
        raise TokenizationError("flat display intent requires alter -1")
    if pitch.display_accidental is DisplayAccidental.NATURAL and pitch.alter != 0:
        raise TokenizationError("natural display intent requires alter 0")
    return pitch


def _validate_projection(projection: object) -> SemanticScoreProjection:
    if not isinstance(projection, SemanticScoreProjection):
        raise TokenizationError("target must be SemanticScoreProjection")
    if len(projection.parts) != 1:
        raise TokenizationError("V1 requires exactly one part")
    part = projection.parts[0]
    if not isinstance(part, SemanticPartProjection) or part.part_id != "P1" or part.staff_count != 1:
        raise TokenizationError("V1 part/staff identity is invalid")
    if not isinstance(part.measures, tuple) or not part.measures:
        raise TokenizationError("V1 projection requires at least one measure")

    for measure_index, measure in enumerate(part.measures, start=1):
        if not isinstance(measure, SemanticMeasureProjection):
            raise TokenizationError("measure projection has the wrong type")
        if measure.number != measure_index:
            raise TokenizationError("measure numbers must be sequential from 1")
        if measure.time_signature not in _TIME_TO_TOKEN:
            raise TokenizationError("unsupported V1 time signature")
        if measure.key_signature != 0 or measure.clef != "treble":
            raise TokenizationError("V1 key signature/clef invariant is invalid")
        if len(measure.voices) != 1:
            raise TokenizationError("V1 requires exactly one voice")
        voice = measure.voices[0]
        if not isinstance(voice, SemanticVoiceProjection) or voice.voice_id != 1:
            raise TokenizationError("V1 voice identity is invalid")

        cursor = Fraction(0, 1)
        for event in voice.events:
            if not isinstance(event, SemanticEventProjection):
                raise TokenizationError("event projection has the wrong type")
            if event.staff != 1:
                raise TokenizationError("V1 event staff must be 1")
            if event.onset != cursor:
                raise TokenizationError("event onset is not the deterministic V1 timeline")
            if event.duration not in _DURATION_TO_TOKEN:
                raise TokenizationError("event duration is outside V1")

            if event.event_type == "rest":
                if event.pitches:
                    raise TokenizationError("rest must not contain pitches")
            elif event.event_type == "note":
                if len(event.pitches) != 1:
                    raise TokenizationError("note must contain exactly one pitch")
                _validate_pitch(event.pitches[0])
            elif event.event_type == "chord":
                if not 2 <= len(event.pitches) <= 4:
                    raise TokenizationError("V1 chord must contain two through four pitches")
                observed: set[tuple[str, int, int]] = set()
                for pitch in event.pitches:
                    checked = _validate_pitch(pitch)
                    identity = (checked.step, checked.alter, checked.octave)
                    if identity in observed:
                        raise TokenizationError("V1 chord contains a duplicate pitch")
                    observed.add(identity)
            else:
                raise TokenizationError("unsupported V1 event type")
            cursor += event.duration

        if cursor != _measure_capacity(measure.time_signature):
            raise TokenizationError("measure duration does not exactly fill its V1 meter")
    return projection


def _pitch_tokens(pitch: SemanticPitchProjection) -> tuple[str, str, str, str]:
    checked = _validate_pitch(pitch)
    return (
        f"STEP_{checked.step}",
        _ALTER_TO_TOKEN[checked.alter],
        f"OCT_{checked.octave}",
        _ACC_TO_TOKEN[checked.display_accidental],
    )


def tokenize_projection(projection: object) -> tuple[str, ...]:
    checked = _validate_projection(projection)
    tokens: list[str] = ["BOS"]
    for measure in checked.parts[0].measures:
        tokens.extend(("MEASURE_START", _TIME_TO_TOKEN[measure.time_signature]))
        for event in measure.voices[0].events:
            if event.event_type == "rest":
                tokens.extend(("REST", _DURATION_TO_TOKEN[event.duration]))
            elif event.event_type == "note":
                tokens.extend(("NOTE", _DURATION_TO_TOKEN[event.duration]))
                tokens.extend(_pitch_tokens(event.pitches[0]))
            else:
                tokens.extend((f"CHORD_{len(event.pitches)}", _DURATION_TO_TOKEN[event.duration]))
                for pitch in event.pitches:
                    tokens.extend(_pitch_tokens(pitch))
        tokens.append("MEASURE_END")
    tokens.append("EOS")
    result = tuple(tokens)
    encode_tokens(result)
    return result


def _require_token(tokens: tuple[str, ...], index: int, expected: str | None = None) -> tuple[str, int]:
    if index >= len(tokens):
        raise TokenizationError("unexpected end of semantic token sequence")
    token = tokens[index]
    if token not in TOKEN_TO_ID or token == "PAD":
        raise TokenizationError("invalid token in semantic target")
    if expected is not None and token != expected:
        raise TokenizationError(f"expected {expected}, got {token}")
    return token, index + 1


def _parse_pitch(tokens: tuple[str, ...], index: int) -> tuple[SemanticPitchProjection, int]:
    step_token, index = _require_token(tokens, index)
    alter_token, index = _require_token(tokens, index)
    octave_token, index = _require_token(tokens, index)
    accidental_token, index = _require_token(tokens, index)

    if not step_token.startswith("STEP_") or step_token[5:] not in _ALLOWED_STEPS:
        raise TokenizationError("invalid pitch-step token")
    if alter_token not in _TOKEN_TO_ALTER:
        raise TokenizationError("invalid alter token")
    if not octave_token.startswith("OCT_"):
        raise TokenizationError("invalid octave token")
    try:
        octave = int(octave_token[4:])
    except ValueError as exc:
        raise TokenizationError("invalid octave token") from exc
    if octave not in _ALLOWED_OCTAVES:
        raise TokenizationError("octave token is outside V1")
    if accidental_token not in _TOKEN_TO_ACC:
        raise TokenizationError("invalid accidental token")

    pitch = SemanticPitchProjection(
        step=step_token[5:],
        alter=_TOKEN_TO_ALTER[alter_token],
        octave=octave,
        display_accidental=_TOKEN_TO_ACC[accidental_token],
    )
    _validate_pitch(pitch)
    return pitch, index


def detokenize_tokens(tokens: object) -> SemanticScoreProjection:
    if not isinstance(tokens, tuple) or any(not isinstance(token, str) for token in tokens):
        raise TokenizationError("tokens must be an immutable tuple of strings")
    if len(tokens) < 5:
        raise TokenizationError("semantic target is too short")
    encode_tokens(tokens)

    index = 0
    _token, index = _require_token(tokens, index, "BOS")
    measures: list[SemanticMeasureProjection] = []

    while index < len(tokens):
        token = tokens[index]
        if token == "EOS":
            index += 1
            break

        _token, index = _require_token(tokens, index, "MEASURE_START")
        meter_token, index = _require_token(tokens, index)
        time_signature = _TOKEN_TO_TIME.get(meter_token)
        if time_signature is None:
            raise TokenizationError("measure is missing a V1 meter token")

        events: list[SemanticEventProjection] = []
        cursor = Fraction(0, 1)
        while True:
            event_token, index = _require_token(tokens, index)
            if event_token == "MEASURE_END":
                break

            if event_token == "REST":
                pitch_count = 0
                event_type = "rest"
            elif event_token == "NOTE":
                pitch_count = 1
                event_type = "note"
            elif event_token in {"CHORD_2", "CHORD_3", "CHORD_4"}:
                pitch_count = int(event_token[-1])
                event_type = "chord"
            else:
                raise TokenizationError("unexpected token where an event type was required")

            duration_token, index = _require_token(tokens, index)
            duration = _TOKEN_TO_DURATION.get(duration_token)
            if duration is None:
                raise TokenizationError("event is missing a V1 duration token")

            pitches: list[SemanticPitchProjection] = []
            for _ in range(pitch_count):
                pitch, index = _parse_pitch(tokens, index)
                pitches.append(pitch)

            if event_type == "chord":
                identities = {(pitch.step, pitch.alter, pitch.octave) for pitch in pitches}
                if len(identities) != len(pitches):
                    raise TokenizationError("detokenized chord contains duplicate pitches")

            events.append(
                SemanticEventProjection(
                    event_type=event_type,
                    onset=cursor,
                    duration=duration,
                    staff=1,
                    pitches=tuple(pitches),
                )
            )
            cursor += duration

        if cursor != _measure_capacity(time_signature):
            raise TokenizationError("detokenized measure does not exactly fill its V1 meter")
        measures.append(
            SemanticMeasureProjection(
                number=len(measures) + 1,
                time_signature=time_signature,
                key_signature=0,
                clef="treble",
                voices=(SemanticVoiceProjection(voice_id=1, events=tuple(events)),),
            )
        )

    if index != len(tokens):
        raise TokenizationError("tokens appear after EOS")
    if not measures:
        raise TokenizationError("semantic target contains no measures")

    projection = SemanticScoreProjection(
        parts=(
            SemanticPartProjection(
                part_id="P1",
                staff_count=1,
                measures=tuple(measures),
            ),
        )
    )
    return _validate_projection(projection)


def tokenize_musicxml(data: object) -> TokenizedTarget:
    source_projection = parse_supported_v1_musicxml_projection(data)
    tokens = tokenize_projection(source_projection)
    reconstructed = detokenize_tokens(tokens)
    comparison = compare_semantic_projections(source_projection, reconstructed)
    if not comparison.is_valid:
        codes = ", ".join(issue.code for issue in comparison.issues)
        raise TokenizationError(f"tokenizer/detokenizer semantic round trip failed: {codes}")
    return TokenizedTarget(
        tokens=tokens,
        token_ids=encode_tokens(tokens),
        tokenizer_fingerprint=tokenizer_fingerprint(),
    )
