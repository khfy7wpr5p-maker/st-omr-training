from __future__ import annotations

import inspect
import math
import unittest

from st_omr_training.meter_v2_deterministic_composer_v1 import ACCEPTED, REJECTED
import st_omr_training.meter_v2_presence_shadow_v1 as presence_shadow
from st_omr_training.meter_v2_presence_shadow_v1 import (
    M3B_D11_CHECKPOINT_SHA256,
    M3B_PRESENCE_CACHE_SHA256,
    M3B_PRESENCE_THRESHOLD,
    M3B_PRESENCE_VALIDATION,
    M3B_TEST_RECORDS,
    M3B_VALIDATION_NONE,
    M3B_VALIDATION_POSITIVE,
    M3B_VALIDATION_TOTAL,
    presence_bridge_product_quality_accepted,
    presence_from_m3b_score_v1,
    resolver_connection_allowed,
)


class MeterV2PresenceShadowV1Tests(unittest.TestCase):
    def test_frozen_identity_threshold_and_inventory(self) -> None:
        self.assertEqual(
            M3B_PRESENCE_CACHE_SHA256,
            "12f70dcdd15c377b85d57f585b59a03a2286f507a0bb7022c0a9ff26a6515ebd",
        )
        self.assertEqual(
            M3B_D11_CHECKPOINT_SHA256,
            "cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3",
        )
        self.assertEqual(M3B_PRESENCE_THRESHOLD, 0.90)
        self.assertEqual(M3B_VALIDATION_TOTAL, 1224)
        self.assertEqual(M3B_VALIDATION_POSITIVE, 591)
        self.assertEqual(M3B_VALIDATION_NONE, 633)
        self.assertEqual(M3B_TEST_RECORDS, 0)

    def test_full_validation_confusion_metrics_are_frozen(self) -> None:
        evidence = M3B_PRESENCE_VALIDATION
        self.assertEqual(
            (
                evidence.true_positive,
                evidence.false_positive,
                evidence.false_negative,
                evidence.true_negative,
            ),
            (590, 8, 1, 625),
        )
        self.assertAlmostEqual(evidence.recall, 590 / 591)
        self.assertAlmostEqual(evidence.precision, 590 / 598)
        self.assertAlmostEqual(evidence.f1, 0.992430613961312)
        self.assertAlmostEqual(evidence.accuracy, 1215 / 1224)

    def test_threshold_boundary_uses_original_float_not_milli_rounding(self) -> None:
        below = presence_from_m3b_score_v1(0.899999)
        at = presence_from_m3b_score_v1(0.90)
        above = presence_from_m3b_score_v1(0.999)
        self.assertEqual(below.status, ACCEPTED)
        self.assertIs(below.present, False)
        self.assertEqual(at.status, ACCEPTED)
        self.assertIs(at.present, True)
        self.assertEqual(above.status, ACCEPTED)
        self.assertIs(above.present, True)

    def test_invalid_scores_fail_closed(self) -> None:
        for value in (-0.1, 1.1, math.nan, math.inf, -math.inf, True, "0.9"):
            with self.subTest(value=value):
                result = presence_from_m3b_score_v1(value)  # type: ignore[arg-type]
                self.assertEqual(result.status, REJECTED)
                self.assertIsNone(result.present)

    def test_shadow_only_isolation(self) -> None:
        self.assertFalse(presence_bridge_product_quality_accepted())
        self.assertFalse(resolver_connection_allowed())
        source = inspect.getsource(presence_shadow)
        self.assertNotIn("import torch", source)
        self.assertNotIn("runtime_deterministic_resolver", source)
        self.assertNotIn("optimizer", source.lower())


if __name__ == "__main__":
    unittest.main()
