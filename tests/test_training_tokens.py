from __future__ import annotations

import unittest

from st_omr_training.generator import GeneratorConfig, generate_score
from st_omr_training.musicxml_roundtrip import (
    compare_semantic_projections,
    parse_supported_v1_musicxml_projection,
)
from st_omr_training.musicxml_writer import write_musicxml
from st_omr_training.training_tokens import (
    PAD_TOKEN_ID,
    TOKEN_VOCABULARY,
    TokenizationError,
    decode_token_ids,
    detokenize_tokens,
    encode_tokens,
    tokenize_musicxml,
    tokenizer_fingerprint,
)


class TokenVocabularyTests(unittest.TestCase):
    def test_vocabulary_is_exact_finite_and_unique(self) -> None:
        self.assertEqual(len(TOKEN_VOCABULARY), 35)
        self.assertEqual(len(set(TOKEN_VOCABULARY)), 35)
        self.assertEqual(TOKEN_VOCABULARY[PAD_TOKEN_ID], "PAD")
        self.assertEqual(
            set(TOKEN_VOCABULARY),
            {
                "PAD", "BOS", "EOS", "MEASURE_START", "MEASURE_END",
                "TS_2_4", "TS_3_4", "TS_4_4",
                "NOTE", "REST", "CHORD_2", "CHORD_3", "CHORD_4",
                "DUR_WHOLE", "DUR_HALF", "DUR_QUARTER", "DUR_EIGHTH",
                "STEP_A", "STEP_B", "STEP_C", "STEP_D", "STEP_E", "STEP_F", "STEP_G",
                "ALTER_M1", "ALTER_0", "ALTER_P1",
                "OCT_3", "OCT_4", "OCT_5", "OCT_6",
                "ACC_NONE", "ACC_SHARP", "ACC_FLAT", "ACC_NATURAL",
            },
        )

    def test_tokenizer_fingerprint_is_stable(self) -> None:
        self.assertEqual(tokenizer_fingerprint(), tokenizer_fingerprint())
        self.assertEqual(len(tokenizer_fingerprint()), 64)

    def test_pad_cannot_enter_semantic_target(self) -> None:
        with self.assertRaises(TokenizationError):
            encode_tokens(("BOS", "PAD", "EOS"))
        with self.assertRaises(TokenizationError):
            decode_token_ids((1, PAD_TOKEN_ID, 2))


class SemanticRoundTripTests(unittest.TestCase):
    def test_generated_musicxml_token_round_trip_is_exact(self) -> None:
        configs = (
            GeneratorConfig(measure_count=2),
            GeneratorConfig(measure_count=2, event_kinds=("note",)),
            GeneratorConfig(measure_count=2, event_kinds=("rest",)),
            GeneratorConfig(measure_count=2, event_kinds=("chord",)),
        )
        for index, config in enumerate(configs):
            with self.subTest(config=index):
                musicxml = write_musicxml(generate_score(config, 700 + index))
                source = parse_supported_v1_musicxml_projection(musicxml)
                tokenized = tokenize_musicxml(musicxml)
                reconstructed = detokenize_tokens(tokenized.tokens)
                self.assertTrue(compare_semantic_projections(source, reconstructed).is_valid)
                self.assertEqual(
                    decode_token_ids(tokenized.token_ids),
                    tokenized.tokens,
                )

    def test_underfilled_measure_is_rejected(self) -> None:
        tokens = (
            "BOS",
            "MEASURE_START",
            "TS_4_4",
            "REST",
            "DUR_QUARTER",
            "MEASURE_END",
            "EOS",
        )
        with self.assertRaises(TokenizationError):
            detokenize_tokens(tokens)

    def test_accidental_pitch_incoherence_is_rejected(self) -> None:
        tokens = (
            "BOS",
            "MEASURE_START",
            "TS_4_4",
            "NOTE",
            "DUR_WHOLE",
            "STEP_C",
            "ALTER_0",
            "OCT_4",
            "ACC_SHARP",
            "MEASURE_END",
            "EOS",
        )
        with self.assertRaises(TokenizationError):
            detokenize_tokens(tokens)

    def test_trailing_tokens_after_eos_are_rejected(self) -> None:
        tokens = (
            "BOS",
            "MEASURE_START",
            "TS_4_4",
            "REST",
            "DUR_WHOLE",
            "MEASURE_END",
            "EOS",
            "BOS",
        )
        with self.assertRaises(TokenizationError):
            detokenize_tokens(tokens)


if __name__ == "__main__":
    unittest.main()
