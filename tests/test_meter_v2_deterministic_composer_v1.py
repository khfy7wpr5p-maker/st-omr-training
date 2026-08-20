from __future__ import annotations

import inspect
import math
import unittest

from st_omr_training import meter_v2_deterministic_composer_v1 as meter


class MeterV2DeterministicComposerTests(unittest.TestCase):
    def p(self, present=True, status="accepted", confidence=950):
        return meter.MeterPresenceObservation(status, present, confidence)

    def d(self, oid, digit, x0=10, y0=10, x1=20, y1=20, status="accepted", confidence=950):
        return meter.MeterDigitObservation(
            oid,
            status,
            digit,
            confidence,
            meter.MeterBox(x0, y0, x1, y1),
        )

    def test_M01_explicit_absence_accepts_none(self) -> None:
        actual = meter.compose_meter_v2(self.p(False), ())
        self.assertEqual(actual, meter.MeterCompositionResult("accepted", "none", None, None, (), ()))

    def test_M02_two_over_four_accepts(self) -> None:
        actual = meter.compose_meter_v2(
            self.p(),
            (self.d("top", 2, y0=10, y1=20), self.d("bottom", 4, y0=25, y1=35)),
        )
        self.assertEqual(actual.meter_class, "2/4")
        self.assertEqual((actual.numerator, actual.denominator), (2, 4))
        self.assertEqual(actual.digit_ids, ("top", "bottom"))

    def test_M03_three_over_four_accepts(self) -> None:
        actual = meter.compose_meter_v2(
            self.p(),
            (self.d("bottom", 4, y0=25, y1=35), self.d("top", 3, y0=10, y1=20)),
        )
        self.assertEqual(actual.status, "accepted")
        self.assertEqual(actual.meter_class, "3/4")
        self.assertEqual(actual.digit_ids, ("top", "bottom"))

    def test_M04_four_over_four_accepts(self) -> None:
        actual = meter.compose_meter_v2(
            self.p(),
            (self.d("top", 4, y0=10, y1=20), self.d("bottom", 4, y0=25, y1=35)),
        )
        self.assertEqual(actual.meter_class, "4/4")

    def test_M05_present_without_digits_is_ambiguous(self) -> None:
        actual = meter.compose_meter_v2(self.p(), ())
        self.assertEqual(actual.status, "ambiguous")
        self.assertEqual(actual.reasons, (meter.R_MISSING_DIGIT,))

    def test_M06_one_digit_is_ambiguous(self) -> None:
        actual = meter.compose_meter_v2(self.p(), (self.d("top", 3),))
        self.assertEqual(actual.status, "ambiguous")
        self.assertEqual(actual.reasons, (meter.R_MISSING_DIGIT,))

    def test_M07_three_digits_are_ambiguous(self) -> None:
        actual = meter.compose_meter_v2(
            self.p(),
            (
                self.d("a", 2, y0=5, y1=12),
                self.d("b", 3, y0=15, y1=22),
                self.d("c", 4, y0=25, y1=32),
            ),
        )
        self.assertEqual(actual.status, "ambiguous")
        self.assertEqual(actual.reasons, (meter.R_DIGIT_COUNT_CONFLICT,))

    def test_M08_equal_vertical_centers_are_ambiguous(self) -> None:
        actual = meter.compose_meter_v2(
            self.p(),
            (self.d("a", 3, x0=10, y0=10, x1=20, y1=20), self.d("b", 4, x0=22, y0=10, x1=32, y1=20)),
        )
        self.assertEqual(actual.status, "ambiguous")
        self.assertEqual(actual.reasons, (meter.R_GEOMETRY_AMBIGUOUS,))

    def test_M09_horizontally_separate_digits_are_ambiguous(self) -> None:
        actual = meter.compose_meter_v2(
            self.p(),
            (self.d("top", 3, x0=0, y0=10, x1=10, y1=20), self.d("bottom", 4, x0=40, y0=25, x1=50, y1=35)),
        )
        self.assertEqual(actual.status, "ambiguous")
        self.assertEqual(actual.reasons, (meter.R_GEOMETRY_AMBIGUOUS,))

    def test_M10_absence_with_digit_evidence_is_ambiguous(self) -> None:
        actual = meter.compose_meter_v2(self.p(False), (self.d("digit", 4),))
        self.assertEqual(actual.status, "ambiguous")
        self.assertEqual(actual.reasons, (meter.R_PRESENCE_CONFLICT,))

    def test_M11_unsupported_visual_digit_is_rejected(self) -> None:
        actual = meter.compose_meter_v2(
            self.p(),
            (self.d("top", 6, y0=10, y1=20), self.d("bottom", 4, y0=25, y1=35)),
        )
        self.assertEqual(actual.status, "rejected")
        self.assertEqual(actual.reasons, (meter.R_UNSUPPORTED_DIGIT,))

    def test_M12_invalid_bbox_is_rejected(self) -> None:
        bad = meter.MeterDigitObservation(
            "bad", "accepted", 3, 950, meter.MeterBox(20, 10, 10, 20)
        )
        actual = meter.compose_meter_v2(self.p(), (bad, self.d("bottom", 4, y0=25, y1=35)))
        self.assertEqual(actual.status, "rejected")
        self.assertEqual(actual.reasons, (meter.R_INVALID_BBOX,))

    def test_M13_nonfinite_bbox_is_rejected(self) -> None:
        bad = meter.MeterDigitObservation(
            "bad", "accepted", 3, 950, meter.MeterBox(10, math.nan, 20, 20)
        )
        actual = meter.compose_meter_v2(self.p(), (bad, self.d("bottom", 4, y0=25, y1=35)))
        self.assertEqual(actual.status, "rejected")
        self.assertEqual(actual.reasons, (meter.R_INVALID_BBOX,))

    def test_M14_invalid_confidence_is_rejected(self) -> None:
        actual = meter.compose_meter_v2(
            meter.MeterPresenceObservation("accepted", True, 1001), ()
        )
        self.assertEqual(actual.status, "rejected")
        self.assertEqual(actual.reasons, (meter.R_NONFINITE,))

    def test_M15_ambiguous_presence_stays_ambiguous(self) -> None:
        actual = meter.compose_meter_v2(
            meter.MeterPresenceObservation("ambiguous", None, 500, ("visual_conflict",)),
            (),
        )
        self.assertEqual(actual.status, "ambiguous")
        self.assertEqual(actual.reasons, (meter.R_PRESENCE_AMBIGUOUS,))

    def test_M16_rejected_presence_stays_rejected(self) -> None:
        actual = meter.compose_meter_v2(
            meter.MeterPresenceObservation("rejected", None, 0, ("bad_input",)),
            (),
        )
        self.assertEqual(actual.status, "rejected")
        self.assertEqual(actual.reasons, (meter.R_PRESENCE_REJECTED,))

    def test_M17_ambiguous_digit_stays_ambiguous(self) -> None:
        uncertain = meter.MeterDigitObservation(
            "uncertain", "ambiguous", None, 500, None, ("digit_conflict",)
        )
        actual = meter.compose_meter_v2(self.p(), (uncertain,))
        self.assertEqual(actual.status, "ambiguous")
        self.assertEqual(actual.reasons, (meter.R_DIGIT_AMBIGUOUS,))

    def test_M18_supported_digits_but_unsupported_meter_is_ambiguous(self) -> None:
        actual = meter.compose_meter_v2(
            self.p(),
            (self.d("top", 4, y0=10, y1=20), self.d("bottom", 2, y0=25, y1=35)),
        )
        self.assertEqual(actual.status, "ambiguous")
        self.assertEqual(actual.reasons, (meter.R_UNSUPPORTED_COMPOSITION,))
        self.assertIsNone(actual.meter_class)

    def test_duplicate_digit_ids_reject(self) -> None:
        actual = meter.compose_meter_v2(
            self.p(),
            (self.d("same", 3, y0=10, y1=20), self.d("same", 4, y0=25, y1=35)),
        )
        self.assertEqual(actual.status, "rejected")
        self.assertEqual(actual.reasons, (meter.R_DIGIT_COUNT_CONFLICT,))

    def test_high_confidence_never_breaks_geometry_conflict(self) -> None:
        actual = meter.compose_meter_v2(
            self.p(),
            (self.d("top", 3, x0=0, y0=10, x1=10, y1=20, confidence=1000), self.d("bottom", 4, x0=50, y0=25, x1=60, y1=35, confidence=1000)),
        )
        self.assertEqual(actual.status, "ambiguous")
        self.assertIsNone(actual.meter_class)

    def test_acceptance_cases_are_10_of_10_deterministic(self) -> None:
        cases = (
            (self.p(False), ()),
            (self.p(), (self.d("t2", 2, y0=10, y1=20), self.d("b2", 4, y0=25, y1=35))),
            (self.p(), (self.d("t3", 3, y0=10, y1=20), self.d("b3", 4, y0=25, y1=35))),
            (self.p(), (self.d("t4", 4, y0=10, y1=20), self.d("b4", 4, y0=25, y1=35))),
            (self.p(), ()),
            (self.p(False), (self.d("conflict", 4),)),
        )
        for presence, digits in cases:
            with self.subTest(presence=presence, digits=digits):
                outputs = tuple(meter.compose_meter_v2(presence, digits) for _ in range(10))
                self.assertEqual(len(set(outputs)), 1)

    def test_shadow_package_cannot_authorize_training_test_or_resolver(self) -> None:
        self.assertFalse(meter.resolver_connection_allowed())
        source = inspect.getsource(meter).lower()
        for forbidden in (
            "from .stage7d10_",
            "from .stage7d11_",
            "import stage7d10_",
            "import stage7d11_",
            "runtime_deterministic_resolver_v1",
            "torch.optim",
            "torch.load(",
            ".backward(",
            "dataloader(",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
