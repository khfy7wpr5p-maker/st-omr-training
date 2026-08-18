from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

from st_omr_training.m4_e3k_r2_scoring import (
    EXPECTED_INTERIOR_BOUNDARIES,
    EXPECTED_TRAIN_RECORDS,
    EXPECTED_TRAIN_SYSTEMS,
    MINIMUM_RECALL_AT_ONE_STAFF_SPACE,
    M4E3KR2ScoringError,
    score_e3k_r2_train,
)


class M4E3KR2ScoringTests(unittest.TestCase):
    def test_frozen_train_surface_and_gate(self) -> None:
        self.assertEqual(EXPECTED_TRAIN_RECORDS, 1230)
        self.assertEqual(EXPECTED_TRAIN_SYSTEMS, 2346)
        self.assertEqual(EXPECTED_INTERIOR_BOUNDARIES, 7494)
        self.assertEqual(MINIMUM_RECALL_AT_ONE_STAFF_SPACE, 0.98)

    def test_empty_verified_train_surface_fails_closed(self) -> None:
        with patch(
            "st_omr_training.m4_e3k_r2_scoring.load_verified_stage7d7_records",
            return_value=(),
        ):
            with self.assertRaises(M4E3KR2ScoringError):
                score_e3k_r2_train("unused-corpus", "unused-d6")

    def test_scoring_api_has_no_split_parameter(self) -> None:
        signature = inspect.signature(score_e3k_r2_train)
        self.assertEqual(tuple(signature.parameters), ("corpus_root", "d6_root"))

    def test_scoring_source_does_not_load_or_train_models(self) -> None:
        module = __import__(
            "st_omr_training.m4_e3k_r2_scoring",
            fromlist=["*"],
        )
        source = inspect.getsource(module).lower()
        for forbidden in (
            "import torch",
            "torch.",
            ".backward(",
            "load_state_dict",
            "optimizer.step",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
