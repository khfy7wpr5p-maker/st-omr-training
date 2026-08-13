from __future__ import annotations

import unittest

from st_omr_training.decoding_constraints import SemanticDecodeConstraint
from st_omr_training.training_tokens import (
    TOKEN_TO_ID,
    decode_token_ids,
    detokenize_tokens,
)


class SemanticDecodeConstraintTests(unittest.TestCase):
    def _advance(self, constraint: SemanticDecodeConstraint, token: str) -> None:
        token_id = TOKEN_TO_ID[token]
        self.assertIn(token_id, constraint.allowed_token_ids())
        constraint.advance(token_id)

    def test_one_measure_rest_sequence_is_forced_to_valid_eos(self) -> None:
        constraint = SemanticDecodeConstraint(measure_count=1)
        generated = [TOKEN_TO_ID["BOS"]]
        for token in (
            "MEASURE_START",
            "TS_4_4",
            "REST",
            "DUR_WHOLE",
            "MEASURE_END",
            "EOS",
        ):
            self._advance(constraint, token)
            generated.append(TOKEN_TO_ID[token])

        self.assertTrue(constraint.is_complete)
        projection = detokenize_tokens(decode_token_ids(tuple(generated)))
        self.assertEqual(len(projection.parts[0].measures), 1)

    def test_measure_cannot_end_until_its_meter_is_exactly_filled(self) -> None:
        constraint = SemanticDecodeConstraint(measure_count=1)
        for token in ("MEASURE_START", "TS_4_4", "REST", "DUR_HALF"):
            self._advance(constraint, token)
        self.assertNotIn(TOKEN_TO_ID["MEASURE_END"], constraint.allowed_token_ids())

        for token in ("REST", "DUR_HALF", "MEASURE_END"):
            self._advance(constraint, token)
        self.assertEqual(constraint.allowed_token_ids(), (TOKEN_TO_ID["EOS"],))

    def test_duplicate_chord_pitch_is_excluded_before_octave_selection(self) -> None:
        constraint = SemanticDecodeConstraint(measure_count=1)
        for token in (
            "MEASURE_START",
            "TS_4_4",
            "CHORD_2",
            "DUR_WHOLE",
            "STEP_C",
            "ALTER_0",
            "OCT_4",
            "ACC_NONE",
            "STEP_C",
            "ALTER_0",
        ):
            self._advance(constraint, token)
        self.assertNotIn(TOKEN_TO_ID["OCT_4"], constraint.allowed_token_ids())
        self.assertIn(TOKEN_TO_ID["OCT_5"], constraint.allowed_token_ids())

    def test_out_of_grammar_advance_fails_closed(self) -> None:
        constraint = SemanticDecodeConstraint(measure_count=8)
        with self.assertRaises(ValueError):
            constraint.advance(TOKEN_TO_ID["EOS"])
        with self.assertRaises(TypeError):
            SemanticDecodeConstraint(measure_count=True)

    def test_eight_measure_maximum_length_surface_completes_below_ceiling(self) -> None:
        constraint = SemanticDecodeConstraint(measure_count=8)
        generated = [TOKEN_TO_ID["BOS"]]
        for _ in range(8):
            for token in ("MEASURE_START", "TS_4_4"):
                self._advance(constraint, token)
                generated.append(TOKEN_TO_ID[token])
            for _ in range(8):
                for token in ("CHORD_4", "DUR_EIGHTH"):
                    self._advance(constraint, token)
                    generated.append(TOKEN_TO_ID[token])
                for octave in ("OCT_3", "OCT_4", "OCT_5", "OCT_6"):
                    for token in ("STEP_C", "ALTER_0", octave, "ACC_NONE"):
                        self._advance(constraint, token)
                        generated.append(TOKEN_TO_ID[token])
            self._advance(constraint, "MEASURE_END")
            generated.append(TOKEN_TO_ID["MEASURE_END"])
        self._advance(constraint, "EOS")
        generated.append(TOKEN_TO_ID["EOS"])

        self.assertTrue(constraint.is_complete)
        self.assertLessEqual(len(generated), 1536)
        projection = detokenize_tokens(decode_token_ids(tuple(generated)))
        self.assertEqual(len(projection.parts[0].measures), 8)


if __name__ == "__main__":
    unittest.main()
