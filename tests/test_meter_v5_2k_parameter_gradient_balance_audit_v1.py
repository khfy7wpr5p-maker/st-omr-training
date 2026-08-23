import inspect
import unittest

from st_omr_training import meter_v5_2k_parameter_gradient_balance_audit_v1 as m


class TestMeterV52KParameterGradientBalanceAuditV1(unittest.TestCase):
    def test_pair_metrics_exact_opposition(self):
        torch, _nn = m.v52b._import_torch()
        v5 = {"head.weight": torch.tensor([1.0, 0.0])}
        source = {"head.weight": torch.tensor([-1.0, 0.0])}
        result = m._pair_metrics(
            v5,
            source,
            names=["head.weight"],
            reference_lambdas={"one": 1.0},
        )
        self.assertTrue(result["gradient_conflict"])
        self.assertAlmostEqual(result["cosine_similarity"], -1.0, places=12)
        self.assertAlmostEqual(result["minimum_norm_lambda_source"], 1.0, places=12)
        self.assertAlmostEqual(result["minimum_combined_gradient_l2"], 0.0, places=12)
        self.assertAlmostEqual(
            result["references"]["one"]["combined_gradient_l2"], 0.0, places=12
        )

    def test_pair_metrics_nonconflict_clamps_lambda_to_zero(self):
        torch, _nn = m.v52b._import_torch()
        v5 = {"head.weight": torch.tensor([1.0, 0.0])}
        source = {"head.weight": torch.tensor([0.5, 0.5])}
        result = m._pair_metrics(
            v5,
            source,
            names=["head.weight"],
            reference_lambdas={"zero": 0.0},
        )
        self.assertFalse(result["gradient_conflict"])
        self.assertEqual(result["minimum_norm_lambda_source"], 0.0)

    def test_common_minimax_reference_stays_between_individual_minima(self):
        per_digit = {
            "2": {
                "v5_gradient_l2": 3.0,
                "historical_gradient_l2": 1.0,
                "dot_product": -2.0,
                "minimum_norm_lambda_source": 2.0,
            },
            "3": {
                "v5_gradient_l2": 5.0,
                "historical_gradient_l2": 1.0,
                "dot_product": -4.0,
                "minimum_norm_lambda_source": 4.0,
            },
        }
        result = m._common_minimax_reference(per_digit)
        self.assertGreaterEqual(result["lambda_source"], 2.0)
        self.assertLessEqual(result["lambda_source"], 4.0)
        self.assertTrue(result["reference_only"])
        self.assertFalse(result["training_setting_selected"])

    def test_exact_surface_constants(self):
        self.assertEqual(m.EXPECTED_V5_COUNT, 540)
        self.assertEqual(m.EXPECTED_HISTORICAL_COUNT, 26964)
        self.assertEqual(m.EXPECTED_V5_POSITIVE_PER_SPECIALIST, 90)
        self.assertEqual(m.POS_WEIGHT, 1.0)
        self.assertEqual(m.HISTORICAL_BATCH_SIZE, 256)

    def test_safety_source_has_no_optimizer_or_checkpoint_write(self):
        source = inspect.getsource(m)
        self.assertIn("torch.autograd.grad", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.save", source)
        self.assertNotIn("detect_multistaff_geometry", source)
        self.assertNotIn("normalize_raster_page", source)
        self.assertNotIn("_integer_crop_box", source)
        self.assertIn("ret_legacy._historical_canvas_from_bbox", source)

    def test_closed_surfaces(self):
        self.assertFalse(m.training_allowed_by_this_module())
        self.assertFalse(m.validation_opened_by_this_module())
        self.assertTrue(m.final_holdout_locked())
        self.assertFalse(m.production_promotion_allowed())


if __name__ == "__main__":
    unittest.main()
