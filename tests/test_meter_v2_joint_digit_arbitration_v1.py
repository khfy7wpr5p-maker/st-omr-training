from __future__ import annotations

import unittest

from st_omr_training.meter_v2_deterministic_composer_v1 import ACCEPTED, AMBIGUOUS, MeterBox
from st_omr_training.meter_v2_joint_digit_arbitration_v1 import (
    FROZEN_DIGIT_THRESHOLDS,
    MeterDigitSlotProbabilities,
    confidence_ranking_breaks_conflict,
    digit_observation_from_probabilities_v1,
    resolver_connection_allowed,
)


class MeterV2JointDigitArbitrationV1Tests(unittest.TestCase):
    @staticmethod
    def _slot(s2: float, s3: float, s4: float) -> MeterDigitSlotProbabilities:
        return MeterDigitSlotProbabilities(
            slot_id="slot",
            bbox=MeterBox(20.0, 10.0, 40.0, 35.0),
            score_2=s2,
            score_3=s3,
            score_4=s4,
        )

    def test_exact_float_threshold_boundary(self) -> None:
        self.assertEqual(FROZEN_DIGIT_THRESHOLDS, {2: 0.48, 3: 0.60, 4: 0.47})
        self.assertIsNone(digit_observation_from_probabilities_v1(self._slot(0.479999, 0.1, 0.1)))
        accepted = digit_observation_from_probabilities_v1(self._slot(0.48, 0.1, 0.1))
        self.assertIsNotNone(accepted)
        assert accepted is not None
        self.assertEqual(accepted.status, ACCEPTED)
        self.assertEqual(accepted.digit, 2)

    def test_joint_conflict_fails_closed(self) -> None:
        result = digit_observation_from_probabilities_v1(self._slot(0.99, 0.1, 0.98))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.status, AMBIGUOUS)
        self.assertIsNone(result.digit)
        self.assertFalse(confidence_ranking_breaks_conflict())

    def test_representative_output_is_10_of_10_deterministic(self) -> None:
        slot = self._slot(0.1, 0.91, 0.2)
        reports = [digit_observation_from_probabilities_v1(slot) for _ in range(10)]
        self.assertEqual(len(set(reports)), 1)
        self.assertFalse(resolver_connection_allowed())


if __name__ == "__main__":
    unittest.main()
