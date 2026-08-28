from __future__ import annotations

from dataclasses import replace

import pytest

from st_omr_training.polyphonic_representation import (
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
    StemDirection,
    TieState,
    TimeSignature,
    TupletBoundary,
    TupletMark,
)
from st_omr_training.polyphonic_serialization import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    ID_TO_TOKEN,
    PAD_TOKEN_ID,
    POLYPHONIC_SERIALIZATION_VERSION,
    POLYPHONIC_TOKENIZER_VERSION,
    TOKEN_TO_ID,
    TOKEN_VOCABULARY,
    TokenizedPolyphonicTarget,
    PolyphonicSerializationError,
    decode_token_ids,
    detokenize_polyphonic_ids,
    detokenize_polyphonic_target,
    encode_tokens,
    parse_canonical_polyphonic_json,
    parse_polyphonic_payload,
    parse_polyphonic_tokens,
    serialize_polyphonic_score,
    tokenize_polyphonic_score,
    tokenizer_fingerprint,
    validate_roundtrip,
)


def _pitch(step: str, octave: int, *, alter: int = 0, accidental=DisplayAccidentalV2.NONE):
    return PitchSpelling(step=step, alter=alter, octave=octave, display_accidental=accidental)


def _atom(atom_id: str, step: str, octave: int, *, staff_override=None, ties=()):
    return NoteAtom(
        atom_id=atom_id,
        pitch=_pitch(step, octave),
        ties=ties,
        staff_override=staff_override,
    )


def _score() -> PolyScore:
    grace = PolyEvent(
        event_id="e-grâce-00",
        kind=EventKind.NOTE,
        onset=ExactRational(0, 1),
        duration=ExactRational(0, 1),
        voice=1,
        staff=1,
        note_type=NoteType.EIGHTH,
        noteheads=(_atom("a-grâce-00", "D", 5),),
        stem=StemDirection.UP,
        grace=GraceSpec(slash=True),
    )
    chord = PolyEvent(
        event_id="e-chord-01",
        kind=EventKind.CHORD,
        onset=ExactRational(0, 1),
        duration=ExactRational(1, 8),
        voice=1,
        staff=1,
        note_type=NoteType.EIGHTH,
        noteheads=(
            _atom("a-chord-01a", "C", 4, ties=(TieState.START,)),
            _atom("a-chord-01b", "G", 4, staff_override=2),
        ),
        dots=1,
        stem=StemDirection.DOWN,
        beams=(BeamMark(level=1, state=BeamState.BEGIN),),
        tuplets=(
            TupletMark(
                number=1,
                actual_notes=3,
                normal_notes=2,
                boundary=TupletBoundary.START,
            ),
        ),
    )
    voice_two = PolyEvent(
        event_id="e-v2-02",
        kind=EventKind.NOTE,
        onset=ExactRational(0, 1),
        duration=ExactRational(1, 4),
        voice=2,
        staff=2,
        note_type=NoteType.QUARTER,
        noteheads=(
            NoteAtom(
                atom_id="a-v2-02",
                pitch=PitchSpelling(
                    step="F",
                    alter=1,
                    octave=3,
                    display_accidental=DisplayAccidentalV2.SHARP,
                ),
                ties=(TieState.STOP,),
            ),
        ),
        stem=StemDirection.UP,
    )
    rest = PolyEvent(
        event_id="e-rest-03",
        kind=EventKind.REST,
        onset=ExactRational(1, 8),
        duration=ExactRational(1, 8),
        voice=1,
        staff=2,
        note_type=NoteType.EIGHTH,
    )
    measure = PolyMeasure(
        measure_index=1,
        source_number="1A-ölçü",
        time_signature=TimeSignature(beats=(3, 2), beat_type=8),
        key_signature=KeySignature(fifths=1, mode="major"),
        clefs=(
            ClefAssignment(staff=1, sign="G", line=2),
            ClefAssignment(staff=2, sign="F", line=4),
        ),
        events=(grace, chord, voice_two, rest),
        barlines=(
            Barline(
                location=BarlineLocation.RIGHT,
                style=BarlineStyle.LIGHT_HEAVY,
                repeat_direction="backward",
            ),
        ),
    )
    return PolyScore(parts=(PolyPart(part_id="Piyano-α", staff_count=2, measures=(measure,)),))


def _text_tokens(text: str) -> tuple[str, ...]:
    return (
        "TEXT_START",
        *(f"BYTE_{value:02X}" for value in text.encode("utf-8")),
        "TEXT_END",
    )


def test_versions_and_vocabulary_are_frozen_and_closed() -> None:
    assert POLYPHONIC_SERIALIZATION_VERSION == "st-omr-polyphonic-serialization-v1"
    assert POLYPHONIC_TOKENIZER_VERSION == "st-omr-polyphonic-tokenizer-v1"
    assert len(TOKEN_VOCABULARY) == len(set(TOKEN_VOCABULARY))
    assert TOKEN_VOCABULARY[PAD_TOKEN_ID] == "PAD"
    assert TOKEN_VOCABULARY[BOS_TOKEN_ID] == "BOS"
    assert TOKEN_VOCABULARY[EOS_TOKEN_ID] == "EOS"
    assert all(ID_TO_TOKEN[index] == token for token, index in TOKEN_TO_ID.items())
    assert len(tokenizer_fingerprint()) == 64


def test_canonical_json_roundtrip_preserves_exact_score_and_hash() -> None:
    score = _score()
    canonical = serialize_polyphonic_score(score)

    assert canonical.isascii()
    restored = parse_canonical_polyphonic_json(canonical)
    restored_bytes = parse_canonical_polyphonic_json(canonical.encode("ascii"))

    assert restored == score
    assert restored_bytes == score
    assert restored.canonical_sha256() == score.canonical_sha256()
    assert serialize_polyphonic_score(restored) == canonical


def test_structured_token_roundtrip_preserves_polyphony_cross_staff_and_unicode() -> None:
    score = _score()
    target = tokenize_polyphonic_score(score)

    assert target.tokens[0] == "BOS"
    assert target.tokens[-1] == "EOS"
    assert "KEY:voice" in target.tokens
    assert "KEY:staff" in target.tokens
    assert "KEY:onset" in target.tokens
    assert "KEY:staff_override" in target.tokens
    assert "Piyano-α" not in target.tokens
    assert "e-grâce-00" not in target.tokens
    assert all(token in TOKEN_TO_ID for token in target.tokens)

    assert detokenize_polyphonic_target(target) == score
    assert detokenize_polyphonic_ids(target.token_ids) == score
    assert parse_polyphonic_tokens(target.tokens) == score
    assert decode_token_ids(target.token_ids) == target.tokens
    assert encode_tokens(target.tokens) == target.token_ids


def test_validate_roundtrip_checks_json_tokens_ids_and_hash() -> None:
    score = _score()
    target = validate_roundtrip(score)
    assert isinstance(target, TokenizedPolyphonicTarget)
    assert target.representation_sha256 == score.canonical_sha256()


def test_canonical_json_rejects_equivalent_but_noncanonical_text() -> None:
    canonical = serialize_polyphonic_score(_score())
    with pytest.raises(PolyphonicSerializationError, match="not canonical"):
        parse_canonical_polyphonic_json(" " + canonical)


def test_payload_rejects_unknown_key_and_wrong_scalar_types() -> None:
    payload = _score().canonical_payload()
    payload["unknown"] = 1
    with pytest.raises(PolyphonicSerializationError, match="invalid keys"):
        parse_polyphonic_payload(payload)

    payload = _score().canonical_payload()
    payload["parts"][0]["staff_count"] = True
    with pytest.raises(PolyphonicSerializationError, match="plain integer"):
        parse_polyphonic_payload(payload)


def test_payload_rejects_invalid_enum_without_coercion() -> None:
    payload = _score().canonical_payload()
    payload["parts"][0]["measures"][0]["events"][0]["kind"] = "unknown-kind"
    with pytest.raises(PolyphonicSerializationError, match="unsupported value"):
        parse_polyphonic_payload(payload)


def test_token_stream_rejects_noncanonical_object_key_order() -> None:
    tokens = (
        "BOS",
        "OBJ_START",
        "KEY:representation_version",
        *_text_tokens("st-omr-polyphonic-representation-v2"),
        "KEY:parts",
        "ARR_START",
        "ARR_END",
        "OBJ_END",
        "EOS",
    )
    with pytest.raises(PolyphonicSerializationError, match="canonical-sorted"):
        parse_polyphonic_tokens(tokens)


def test_token_stream_rejects_unknown_ids_pad_and_nested_envelopes() -> None:
    target = tokenize_polyphonic_score(_score())

    with pytest.raises(PolyphonicSerializationError, match="outside frozen"):
        decode_token_ids(target.token_ids[:-1] + (len(TOKEN_VOCABULARY) + 99,))
    with pytest.raises(PolyphonicSerializationError, match="PAD"):
        encode_tokens(("BOS", "PAD", "EOS"))
    with pytest.raises(PolyphonicSerializationError, match="nested BOS/EOS"):
        parse_polyphonic_tokens(("BOS", "BOS", "EOS"))


def test_detokenizer_detects_representation_hash_tampering() -> None:
    target = tokenize_polyphonic_score(_score())
    tampered = replace(target, representation_sha256="0" * 64)
    with pytest.raises(PolyphonicSerializationError, match="hash mismatch"):
        detokenize_polyphonic_target(tampered)


def test_tokenized_target_rejects_token_id_mismatch_and_fingerprint_drift() -> None:
    target = tokenize_polyphonic_score(_score())
    with pytest.raises(PolyphonicSerializationError, match="tokens/token_ids mismatch"):
        replace(target, token_ids=target.token_ids[:-1] + (BOS_TOKEN_ID,))
    with pytest.raises(PolyphonicSerializationError, match="fingerprint mismatch"):
        replace(target, tokenizer_fingerprint="0" * 64)


def test_integer_token_form_is_canonical() -> None:
    # Root key order is valid; malformed integer is reached inside representation_version
    # replacement payload only after a complete small object is parsed.
    tokens = (
        "BOS",
        "OBJ_START",
        "KEY:parts",
        "INT_START",
        "DIGIT_0",
        "DIGIT_1",
        "INT_END",
        "KEY:representation_version",
        *_text_tokens("st-omr-polyphonic-representation-v2"),
        "OBJ_END",
        "EOS",
    )
    with pytest.raises(PolyphonicSerializationError, match="leading zero"):
        parse_polyphonic_tokens(tokens)
