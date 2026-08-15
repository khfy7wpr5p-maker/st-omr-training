from __future__ import annotations

import unittest

from st_omr_training.stage7d13_training import training_profile_fingerprint
from st_omr_training.stage7d13_verified_surface import (
    D13_EXPECTED_OPTIMIZER_STEPS,
    D13_EXPECTED_OPTIMIZER_STEPS_TOTAL,
    D13_RECORD_SPLIT_COUNTS,
)


class Stage7D13TrainingTests(unittest.TestCase):
    def test_training_profile_fingerprint_is_deterministic_sha256(self) -> None:
        first = training_profile_fingerprint()
        second = training_profile_fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertLessEqual(set(first), set("0123456789abcdef"))

    def test_verified_record_counts_imply_exact_frozen_steps(self) -> None:
        self.assertEqual(D13_RECORD_SPLIT_COUNTS, {"train": 9840, "validation": 1224})
        self.assertEqual(
            D13_EXPECTED_OPTIMIZER_STEPS,
            {"notehead": 6150, "rest": 6150, "accidental": 6150},
        )
        self.assertEqual(D13_EXPECTED_OPTIMIZER_STEPS_TOTAL, 18450)


if __name__ == "__main__":
    unittest.main()
