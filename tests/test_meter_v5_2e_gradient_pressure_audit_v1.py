import inspect
import math
import unittest

from st_omr_training import meter_v5_2b_specialist_adaptation as v52b
from st_omr_training import meter_v5_2e_gradient_pressure_audit_v1 as m


class TestMeterV52EGradientPressureAuditV1(unittest.TestCase):
    def test_analytical_bce_logit_derivative_matches_contract(self):
        torch, _nn = v52b._import_torch()
        logits = torch.tensor([-10.0, -10.0, 0.0, 0.0], dtype=torch.float32)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        result = m._pressure_profile_from_logits(
            logits,
            labels,
            positive_weight=5.0,
        )
        p_neg10 = 1.0 / (1.0 + math.exp(10.0))
        expected_positive = abs(5.0 * (p_neg10 - 1.0)) + 2.5
        expected_negative = p_neg10 + 0.5
        self.assertAlmostEqual(result["positive_pressure_total"], expected_positive, places=9)
        self.assertAlmostEqual(result["negative_pressure_total"], expected_negative, places=9)
        self.assertAlmostEqual(
            result["positive_to_negative_pressure_ratio"],
            expected_positive / expected_negative,
            places=9,
        )
        self.assertEqual(result["positive_count"], 2)
        self.assertEqual(result["negative_count"], 2)

    def test_pos_weight_scales_only_positive_pressure(self):
        torch, _nn = v52b._import_torch()
        logits = torch.tensor([-8.0, -8.0], dtype=torch.float32)
        labels = torch.tensor([1.0, 0.0], dtype=torch.float32)
        w1 = m._pressure_profile_from_logits(logits, labels, positive_weight=1.0)
        w5 = m._pressure_profile_from_logits(logits, labels, positive_weight=5.0)
        self.assertAlmostEqual(
            w5["positive_pressure_total"],
            5.0 * w1["positive_pressure_total"],
            places=10,
        )
        self.assertAlmostEqual(
            w5["negative_pressure_total"],
            w1["negative_pressure_total"],
            places=12,
        )

    def test_counterfactual_weights_are_diagnostic_only(self):
        self.assertEqual(m.COUNTERFACTUAL_POS_WEIGHTS, (1.0, 5.0))
        self.assertEqual(m.DOMINANCE_RATIO_FLOOR, 100.0)
        self.assertFalse(m.training_allowed_by_this_module())
        self.assertFalse(m.validation_opened_by_this_module())
        self.assertTrue(m.final_holdout_locked())
        self.assertFalse(m.production_promotion_allowed())

    def test_no_training_or_new_spatial_authority(self):
        source = inspect.getsource(m).lower()
        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("detect_multistaff_geometry", source)
        self.assertNotIn("_slot_edges", source)
        self.assertNotIn("nearest-staff", source)
        self.assertNotIn("tolerance expansion", source)
        self.assertNotIn("final_holdout/", source)
        self.assertIn("ret_legacy._historical_canvas_from_bbox", source)
        self.assertIn('"repair_training_authorized": false', source)
        self.assertIn('"replay_ratio_selected": false', source)
        self.assertIn('"threshold_tuning": false', source)

    def test_dominance_requires_two_orders_of_magnitude(self):
        self.assertTrue(m._dominance({
            "positive_pressure_total": 100.0,
            "negative_pressure_total": 1.0,
            "positive_to_negative_pressure_ratio": 100.0,
        }))
        self.assertFalse(m._dominance({
            "positive_pressure_total": 99.0,
            "negative_pressure_total": 1.0,
            "positive_to_negative_pressure_ratio": 99.0,
        }))
        self.assertTrue(m._dominance({
            "positive_pressure_total": 1.0,
            "negative_pressure_total": 0.0,
            "positive_to_negative_pressure_ratio": None,
        }))


if __name__ == "__main__":
    unittest.main()
