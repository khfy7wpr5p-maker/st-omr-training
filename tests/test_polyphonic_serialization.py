from __future__ import annotations

from dataclasses import replace
import unittest

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
        # Canonical V2 order is onset/voice/staff/event_id. Grace is zero-duration
        # but still obeys the same deterministic ordering surface.
        events=(chord, grace, voice_two, rest),
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


class PolyphonicSerializationTests(unittest.TestCase):
    def test_versions_and_vocabulary_are_frozen_and_closed(self) -> None:
        self.assertEqual(POLYPHONIC_SERIALIZATION_VERSION, "st-omr-polyphonic-serialization-v1")
        self.assertEqual(POLYPHONIC_TOKENIZER_VERSION, "st-omr-polyphonic-tokenizer-v1")
        self.assertEqual(len(TOKEN_VOCABULARY), len(set(TOKEN_VOCABULARY)))
        self.assertEqual(TOKEN_VOCABULARY[PAD_TOKEN_ID], "PAD")
        self.assertEqual(TOKEN_VOCABULARY[BOS_TOKEN_ID], "BOS")
        self.assertEqual(TOKEN_VOCABULARY[EOS_TOKEN_ID], "EOS")
        self.assertTrue(all(ID_TO_TOKEN[index] == token for token, index in TOKEN_TO_ID.items()))
        self.assertEqual(len(tokenizer_fingerprint()), 64)

    def test_canonical_json_roundtrip_preserves_exact_score_and_hash(self) -> None:
        score = _score()
        canonical = serialize_polyphonic_score(score)

        self.assertTrue(canonical.isascii())
        restored = parse_canonical_polyphonic_json(canonical)
        restored_bytes = parse_canonical_polyphonic_json(canonical.encode("ascii"))

        self.assertEqual(restored, score)
        self.assertEqual(restored_bytes, score)
        self.assertEqual(restored.canonical_sha256(), score.canonical_sha256())
        self.assertEqual(serialize_polyphonic_score(restored), canonical)

    def test_structured_token_roundtrip_preserves_polyphony_cross_staff_and_unicode(self) -> None:
        score = _score()
        target = tokenize_polyphonic_score(score)

        self.assertEqual(target.tokens[0], "BOS")
        self.assertEqual(target.tokens[-1], "EOS")
        self.assertIn("KEY:voice", target.tokens)
        self.assertIn("KEY:staff", target.tokens)
        self.assertIn("KEY:onset", target.tokens)
        self.assertIn("KEY:staff_override", target.tokens)
        self.assertNotIn("Piyano-α", target.tokens)
        self.assertNotIn("e-grâce-00", target.tokens)
        self.assertTrue(all(token in TOKEN_TO_ID for token in target.tokens))

        self.assertEqual(detokenize_polyphonic_target(target), score)
        self.assertEqual(detokenize_polyphonic_ids(target.token_ids), score)
        self.assertEqual(parse_polyphonic_tokens(target.tokens), score)
        self.assertEqual(decode_token_ids(target.token_ids), target.tokens)
        self.assertEqual(encode_tokens(target.tokens), target.token_ids)

    def test_validate_roundtrip_checks_json_tokens_ids_and_hash(self) -> None:
        score = _score()
        target = validate_roundtrip(score)
        self.assertIsInstance(target, TokenizedPolyphonicTarget)
        self.assertEqual(target.representation_sha256, score.canonical_sha256())

    def test_canonical_json_rejects_equivalent_but_noncanonical_text(self) -> None:
        canonical = serialize_polyphonic_score(_score())
        with self.assertRaisesRegex(PolyphonicSerializationError, "not canonical"):
            parse_canonical_polyphonic_json(" " + canonical)

    def test_payload_rejects_unknown_key_and_wrong_scalar_types(self) -> None:
        payload = _score().canonical_payload()
        payload["unknown"] = 1
        with self.assertRaisesRegex(PolyphonicSerializationError, "invalid keys"):
            parse_polyphonic_payload(payload)

        payload = _score().canonical_payload()
        payload["parts"][0]["staff_count"] = True
        with self.assertRaisesRegex(PolyphonicSerializationError, "plain integer"):
            parse_polyphonic_payload(payload)

    def test_payload_rejects_invalid_enum_without_coercion(self) -> None:
        payload = _score().canonical_payload()
        payload["parts"][0]["measures"][0]["events"][0]["kind"] = "unknown-kind"
        with self.assertRaisesRegex(PolyphonicSerializationError, "unsupported value"):
            parse_polyphonic_payload(payload)

    def test_token_stream_rejects_noncanonical_object_key_order(self) -> None:
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
        with self.assertRaisesRegex(PolyphonicSerializationError, "canonical-sorted"):
            parse_polyphonic_tokens(tokens)

    def test_token_stream_rejects_unknown_ids_pad_and_nested_envelopes(self) -> None:
        target = tokenize_polyphonic_score(_score())

        with self.assertRaisesRegex(PolyphonicSerializationError, "outside frozen"):
            decode_token_ids(target.token_ids[:-1] + (len(TOKEN_VOCABULARY) + 99,))
        with self.assertRaisesRegex(PolyphonicSerializationError, "PAD"):
            encode_tokens(("BOS", "PAD", "EOS"))
        with self.assertRaisesRegex(PolyphonicSerializationError, "nested BOS/EOS"):
            parse_polyphonic_tokens(("BOS", "BOS", "EOS"))

    def test_detokenizer_detects_representation_hash_tampering(self) -> None:
        target = tokenize_polyphonic_score(_score())
        tampered = replace(target, representation_sha256="0" * 64)
        with self.assertRaisesRegex(PolyphonicSerializationError, "hash mismatch"):
            detokenize_polyphonic_target(tampered)

    def test_tokenized_target_rejects_token_id_mismatch_and_fingerprint_drift(self) -> None:
        target = tokenize_polyphonic_score(_score())
        with self.assertRaisesRegex(PolyphonicSerializationError, "tokens/token_ids mismatch"):
            replace(target, token_ids=target.token_ids[:-1] + (BOS_TOKEN_ID,))
        with self.assertRaisesRegex(PolyphonicSerializationError, "fingerprint mismatch"):
            replace(target, tokenizer_fingerprint="0" * 64)

    def test_integer_token_form_is_canonical(self) -> None:
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
        with self.assertRaisesRegex(PolyphonicSerializationError, "leading zero"):
            parse_polyphonic_tokens(tokens)


if __name__ == "__main__":
    unittest.main()
