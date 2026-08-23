import inspect
import unittest

from PIL import Image

from st_omr_training import meter_v5_2c_historical_retention_v1 as m


class TestMeterV52CHistoricalRetentionV1(unittest.TestCase):
    def test_frozen_identities_and_counts(self):
        self.assertEqual(
            m.M4A_MANIFEST_SHA256,
            "ebda40dae10f0d6490df2c7728dab5cc2cc6f58b5420b198dfbb441a99ecebb9",
        )
        self.assertEqual(
            m.D10_MANIFEST_SHA256,
            "6927e1bcc5251257a983a306e2f1875c9515f97c6724a8fe9f24382c6ff30db4",
        )
        self.assertEqual(
            m.EXPECTED_VALIDATION_LABEL_COUNTS,
            {"2": 186, "3": 204, "4": 792, "NONE": 2190},
        )
        self.assertEqual(
            m.EXPECTED_FROZEN_COUNTS["2"],
            {"tp": 185, "fp": 4, "fn": 1, "tn": 3182},
        )
        self.assertEqual(
            m.EXPECTED_FROZEN_COUNTS["3"],
            {"tp": 203, "fp": 0, "fn": 1, "tn": 3168},
        )
        self.assertEqual(
            m.EXPECTED_FROZEN_COUNTS["4"],
            {"tp": 788, "fp": 23, "fn": 4, "tn": 2557},
        )

    def test_historical_crop_contract_no_upscale_and_centered(self):
        image = Image.new("L", (100, 100), 255)
        for x in range(30, 50):
            for y in range(20, 50):
                image.putpixel((x, y), 0)
        canvas = m._historical_canvas_from_bbox(image, [30.0, 20.0, 50.0, 50.0])
        self.assertEqual(canvas.mode, "L")
        self.assertEqual(canvas.size, (64, 64))
        # 20x30 crop is not upscaled; it is centered at x=22..41, y=17..46.
        self.assertEqual(canvas.getpixel((32, 32)), 0)
        self.assertEqual(canvas.getpixel((0, 0)), 255)

    def test_retention_gate_passes_small_drop(self):
        frozen = {
            "2": {"f1": 0.9866, "recall": 0.9946},
            "3": {"f1": 0.9975, "recall": 0.9951},
        }
        candidate = {
            "2": {"f1": 0.9840, "recall": 0.9920, "precision": 0.9810},
            "3": {"f1": 0.9950, "recall": 0.9930, "precision": 0.9970},
        }
        result = m.evaluate_retention_gate_v1(
            frozen_metrics=frozen,
            candidate_metrics=candidate,
        )
        self.assertEqual(result["gate"], "PASS")
        self.assertEqual(result["reasons"], [])

    def test_retention_gate_holds_excess_f1_drop(self):
        frozen = {
            "2": {"f1": 0.9866, "recall": 0.9946},
            "3": {"f1": 0.9975, "recall": 0.9951},
        }
        candidate = {
            "2": {"f1": 0.9700, "recall": 0.9900, "precision": 0.9900},
            "3": {"f1": 0.9970, "recall": 0.9950, "precision": 0.9990},
        }
        result = m.evaluate_retention_gate_v1(
            frozen_metrics=frozen,
            candidate_metrics=candidate,
        )
        self.assertEqual(result["gate"], "HOLD")
        self.assertIn("2-AI_F1_DROP_GT_0.005", result["reasons"])

    def test_retention_gate_holds_precision_floor(self):
        frozen = {
            "2": {"f1": 0.9866, "recall": 0.9946},
            "3": {"f1": 0.9975, "recall": 0.9951},
        }
        candidate = {
            "2": {"f1": 0.9840, "recall": 0.9940, "precision": 0.9700},
            "3": {"f1": 0.9970, "recall": 0.9950, "precision": 0.9990},
        }
        result = m.evaluate_retention_gate_v1(
            frozen_metrics=frozen,
            candidate_metrics=candidate,
        )
        self.assertEqual(result["gate"], "HOLD")
        self.assertIn("2-AI_PRECISION_LT_0.98", result["reasons"])

    def test_safety_surface_closed(self):
        self.assertFalse(m.validation_opened_by_this_module())
        self.assertTrue(m.final_holdout_locked())
        self.assertFalse(m.production_promotion_allowed())
        source = inspect.getsource(m).lower()
        self.assertNotIn("optimizer =", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("final_holdout/", source)


if __name__ == "__main__":
    unittest.main()
