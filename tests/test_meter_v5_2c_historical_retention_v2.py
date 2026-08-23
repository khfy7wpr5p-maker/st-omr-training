import inspect
import unittest

from PIL import Image

from st_omr_training import meter_v5_2c_historical_retention_v1 as legacy
from st_omr_training import meter_v5_2c_historical_retention_v2 as m


class TestMeterV52CHistoricalRetentionV2(unittest.TestCase):
    def test_identity_surface_is_unchanged(self):
        self.assertEqual(m.M4A_MANIFEST_SHA256, legacy.M4A_MANIFEST_SHA256)
        self.assertEqual(m.D10_MANIFEST_SHA256, legacy.D10_MANIFEST_SHA256)
        self.assertEqual(m.DIGIT2_CANDIDATE_SHA256, legacy.DIGIT2_CANDIDATE_SHA256)
        self.assertEqual(m.DIGIT3_CANDIDATE_SHA256, legacy.DIGIT3_CANDIDATE_SHA256)
        self.assertEqual(
            m.EXPECTED_VALIDATION_LABEL_COUNTS,
            {"2": 186, "3": 204, "4": 792, "NONE": 2190},
        )

    def test_corrected_frozen_oracle_matches_original_experiment_results(self):
        self.assertEqual(
            m.EXPECTED_FROZEN_COUNTS,
            {
                "2": {"tp": 185, "fp": 30, "fn": 1, "tn": 3156},
                "3": {"tp": 203, "fp": 1, "fn": 1, "tn": 3167},
                "4": {"tp": 788, "fp": 46, "fn": 4, "tn": 2534},
            },
        )
        self.assertNotEqual(
            m.EXPECTED_FROZEN_COUNTS,
            legacy.EXPECTED_FROZEN_COUNTS,
        )

    def test_pixel_transform_is_reused_without_new_geometry(self):
        image = Image.new("L", (100, 100), 255)
        for x in range(30, 50):
            for y in range(20, 50):
                image.putpixel((x, y), 0)
        expected = legacy._historical_canvas_from_bbox(
            image,
            [30.0, 20.0, 50.0, 50.0],
        )
        self.assertEqual(expected.mode, "L")
        self.assertEqual(expected.size, (64, 64))
        self.assertEqual(expected.getpixel((32, 32)), 0)
        self.assertEqual(expected.getpixel((0, 0)), 255)

        source = inspect.getsource(m)
        self.assertIn("legacy._load_manifests", source)
        self.assertIn("legacy._prepare_inputs", source)
        self.assertNotIn("detect_multistaff_geometry", source)
        self.assertNotIn("nearest", source.lower())
        self.assertNotIn("tolerance", source.lower())

    def test_retention_rule_is_not_relaxed(self):
        self.assertEqual(m.MAX_F1_DROP, legacy.MAX_F1_DROP)
        self.assertEqual(m.MAX_RECALL_DROP, legacy.MAX_RECALL_DROP)
        self.assertEqual(
            m.MIN_CANDIDATE_PRECISION,
            legacy.MIN_CANDIDATE_PRECISION,
        )
        self.assertEqual(
            m.MIN_CANDIDATE_RECALL,
            legacy.MIN_CANDIDATE_RECALL,
        )

        frozen = {
            "2": {"f1": 0.9226932668, "recall": 0.9946236559},
            "3": {"f1": 0.9950980392, "recall": 0.9950980392},
        }
        candidate = {
            "2": {"f1": 0.9200, "recall": 0.9940, "precision": 0.9799},
            "3": {"f1": 0.9940, "recall": 0.9940, "precision": 0.9990},
        }
        result = legacy.evaluate_retention_gate_v1(
            frozen_metrics=frozen,
            candidate_metrics=candidate,
        )
        self.assertEqual(result["gate"], "HOLD")
        self.assertIn("2-AI_PRECISION_LT_0.98", result["reasons"])

    def test_safety_surface_stays_closed(self):
        self.assertFalse(m.validation_opened_by_this_module())
        self.assertTrue(m.final_holdout_locked())
        self.assertFalse(m.production_promotion_allowed())

        source = inspect.getsource(m).lower()
        self.assertNotIn("optimizer =", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("final_holdout/", source)
        self.assertNotIn("threshold_search", source)


if __name__ == "__main__":
    unittest.main()
