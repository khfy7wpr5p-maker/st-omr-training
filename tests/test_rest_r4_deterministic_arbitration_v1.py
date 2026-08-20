from __future__ import annotations

import math
import unittest

from st_omr_training import rest_r4_deterministic_arbitration_v1 as arb
from st_omr_training import runtime_specialist_shadow_acceptance_v1 as shadow


class RestR4DeterministicArbitrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t = dict(arb.REST_VERIFIER_THRESHOLDS)

    def assert_case(self, scores, expected_status, expected_class=None, expected_duration=None, expected_passing=(), expected_reason=()) -> None:
        actual = arb.arbitrate_rest_scores(scores)
        self.assertEqual(
            actual,
            arb.RestArbitrationResult(
                expected_status,
                expected_class,
                expected_duration,
                expected_passing,
                expected_reason,
            ),
            msg=f"actual={actual!r} expected_status={expected_status!r}",
        )

    def test_frozen_thresholds_match_shadow_evidence(self) -> None:
        observed = {
            item.class_name: item.verifier_threshold
            for item in shadow.REST_R2_SHADOW.verifiers
        }
        self.assertEqual(observed, arb.REST_VERIFIER_THRESHOLDS)

    def test_R01_half_only_accepts_two_beats(self) -> None:
        self.assert_case(
            {"half": self.t["half"], "quarter": 0.0, "eighth": 0.0},
            arb.REST_ACCEPTED, "half", 2.0, ("half",), (),
        )

    def test_R02_quarter_only_accepts_one_beat(self) -> None:
        self.assert_case(
            {"half": 0.0, "quarter": self.t["quarter"], "eighth": 0.0},
            arb.REST_ACCEPTED, "quarter", 1.0, ("quarter",), (),
        )

    def test_R03_eighth_only_accepts_half_beat(self) -> None:
        self.assert_case(
            {"half": 0.0, "quarter": 0.0, "eighth": self.t["eighth"]},
            arb.REST_ACCEPTED, "eighth", 0.5, ("eighth",), (),
        )

    def test_R04_all_below_threshold_is_ambiguous(self) -> None:
        self.assert_case(
            {
                "half": self.t["half"] - 1e-6,
                "quarter": self.t["quarter"] - 1e-6,
                "eighth": self.t["eighth"] - 1e-6,
            },
            arb.REST_AMBIGUOUS, None, None, (), ("R_NO_CLASS_ABOVE_THRESHOLD",),
        )

    def test_R05_half_quarter_conflict_is_ambiguous(self) -> None:
        self.assert_case(
            {"half": 0.99, "quarter": 0.99, "eighth": 0.0},
            arb.REST_AMBIGUOUS, None, None, ("half", "quarter"), ("R_CLASS_CONFLICT",),
        )

    def test_R06_half_eighth_conflict_is_ambiguous(self) -> None:
        self.assert_case(
            {"half": 0.99, "quarter": 0.0, "eighth": 0.99},
            arb.REST_AMBIGUOUS, None, None, ("half", "eighth"), ("R_CLASS_CONFLICT",),
        )

    def test_R07_quarter_eighth_conflict_is_ambiguous(self) -> None:
        self.assert_case(
            {"half": 0.0, "quarter": 0.99, "eighth": 0.99},
            arb.REST_AMBIGUOUS, None, None, ("quarter", "eighth"), ("R_CLASS_CONFLICT",),
        )

    def test_R08_three_way_conflict_is_ambiguous(self) -> None:
        self.assert_case(
            {"half": 0.99, "quarter": 0.99, "eighth": 0.99},
            arb.REST_AMBIGUOUS,
            None,
            None,
            ("half", "quarter", "eighth"),
            ("R_CLASS_CONFLICT",),
        )

    def test_R09_higher_score_cannot_break_conflict(self) -> None:
        actual = arb.arbitrate_rest_scores({"half": 0.10, "quarter": 0.99, "eighth": 0.57})
        self.assertEqual(actual.status, arb.REST_AMBIGUOUS)
        self.assertEqual(actual.passing_classes, ("half", "quarter", "eighth"))
        self.assertEqual(actual.reasons, ("R_CLASS_CONFLICT",))
        self.assertIsNone(actual.class_name)

    def test_R10_nonfinite_score_is_rejected(self) -> None:
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                actual = arb.arbitrate_rest_scores(
                    {"half": bad, "quarter": 0.0, "eighth": 0.0}
                )
                self.assertEqual(actual.status, arb.REST_REJECTED)
                self.assertEqual(actual.reasons, ("R_NONFINITE_OR_RANGE",))

    def test_R11_invalid_bbox_is_rejected(self) -> None:
        actual = arb.arbitrate_rest_scores(
            {"half": 0.99, "quarter": 0.0, "eighth": 0.0},
            bbox_finite=False,
        )
        self.assertEqual(actual.status, arb.REST_REJECTED)
        self.assertEqual(actual.reasons, ("R_INVALID_BBOX",))
        self.assertIsNone(actual.class_name)

    def test_R12_missing_or_extra_class_is_rejected(self) -> None:
        bad_inputs = (
            {"half": 0.9, "quarter": 0.1},
            {"half": 0.9, "quarter": 0.1, "eighth": 0.1, "whole": 0.1},
        )
        for scores in bad_inputs:
            with self.subTest(scores=scores):
                actual = arb.arbitrate_rest_scores(scores)
                self.assertEqual(actual.status, arb.REST_REJECTED)
                self.assertEqual(actual.reasons, ("R_INPUT_SCHEMA",))

    def test_all_cases_are_10_of_10_deterministic(self) -> None:
        cases = (
            {"half": self.t["half"], "quarter": 0.0, "eighth": 0.0},
            {"half": 0.0, "quarter": self.t["quarter"], "eighth": 0.0},
            {"half": 0.0, "quarter": 0.0, "eighth": self.t["eighth"]},
            {"half": 0.0, "quarter": 0.0, "eighth": 0.0},
            {"half": 0.99, "quarter": 0.99, "eighth": 0.0},
            {"half": 0.99, "quarter": 0.0, "eighth": 0.99},
            {"half": 0.0, "quarter": 0.99, "eighth": 0.99},
            {"half": 0.99, "quarter": 0.99, "eighth": 0.99},
        )
        for scores in cases:
            with self.subTest(scores=scores):
                outputs = tuple(arb.arbitrate_rest_scores(scores) for _ in range(10))
                self.assertEqual(len(set(outputs)), 1)

    def test_shadow_contract_never_authorizes_runtime_resolver(self) -> None:
        self.assertFalse(arb.resolver_connection_allowed())
        self.assertFalse(shadow.resolver_connection_allowed())


if __name__ == "__main__":
    unittest.main()
