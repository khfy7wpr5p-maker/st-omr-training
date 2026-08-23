import inspect
import math
import unittest

from st_omr_training import meter_v5_2l_projected_gradient_repair_v1 as m


class TestMeterV52LProjectedGradientRepairV1(unittest.TestCase):
    def test_exact_recipe_is_frozen(self):
        self.assertEqual(m.POS_WEIGHT, 1.0)
        self.assertEqual(m.LEARNING_RATE, 1e-4)
        self.assertEqual(m.V5_BATCH_SIZE, 64)
        self.assertEqual(m.EPOCHS, 12)
        self.assertEqual(m.MASTER_SEED, 52023)
        self.assertEqual(m.EXPECTED_V5_COUNT, 540)
        self.assertEqual(m.EXPECTED_V5_BATCHES_PER_EPOCH, 9)
        self.assertEqual(m.EXPECTED_UPDATES_PER_SPECIALIST, 108)

    def test_projection_coefficient_for_conflict(self):
        # g_v5 dot g_source = -6 and ||g_source||^2 = 9.
        coeff = m._projection_coefficient_from_scalars(-6.0, 9.0)
        self.assertAlmostEqual(coeff, 2.0 / 3.0)
        self.assertAlmostEqual(-6.0 + coeff * 9.0, 0.0)

    def test_projection_coefficient_for_non_conflict_is_zero(self):
        self.assertEqual(m._projection_coefficient_from_scalars(1.0, 9.0), 0.0)
        self.assertEqual(m._projection_coefficient_from_scalars(0.0, 9.0), 0.0)

    def test_projection_rejects_invalid_source_norm(self):
        for source_sq in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                m._projection_coefficient_from_scalars(-1.0, source_sq)

    def test_gate_order_is_retention_first(self):
        self.assertEqual(
            m.gate_order(),
            ("historical_retention", "v5_first30_diagnostic"),
        )

    def test_safety_boundary(self):
        s = m.safety_boundary()
        self.assertTrue(s["single_fixed_projected_repair_authorized"])
        self.assertFalse(s["automatic_second_configuration"])
        self.assertTrue(s["source_reference_recomputed_each_epoch"])
        self.assertTrue(s["direct_sgd_no_momentum"])
        self.assertEqual(s["weight_decay"], 0.0)
        self.assertFalse(s["gradient_clipping"])
        self.assertFalse(s["gradient_renormalization"])
        self.assertFalse(s["threshold_tuning"])
        self.assertFalse(s["new_bbox"])
        self.assertFalse(s["new_crop_geometry"])
        self.assertFalse(s["new_spatial_heuristic"])
        self.assertFalse(s["reserve_v5_train_opened"])
        self.assertFalse(s["v5_validation_opened"])
        self.assertTrue(s["final_holdout_locked"])
        self.assertTrue(s["digit4_frozen"])
        self.assertFalse(s["resolver_wiring"])
        self.assertFalse(s["production_promotion"])

    def test_wrong_approval_token_fails_before_data_access(self):
        with self.assertRaises(m.MeterV5_2LError):
            m.train_projected_repair_v1(
                "/does/not/exist",
                m4a_root="/does/not/exist",
                d10_root="/does/not/exist",
                digit2_frozen="/does/not/exist",
                digit3_frozen="/does/not/exist",
                confirmation="WRONG",
            )

    def test_no_optimizer_or_backward_path(self):
        source = inspect.getsource(m)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn(".backward(", source)
        self.assertIn("torch.autograd.grad", source)
        self.assertIn("parameter.add_", source)

    def test_no_new_spatial_derivation(self):
        source = inspect.getsource(m)
        self.assertNotIn("detect_multistaff_geometry", source)
        self.assertNotIn("normalize_raster_page", source)
        self.assertNotIn("_integer_crop_box", source)
        self.assertNotIn("derive_staff_relative_slots_v1(", source)
        self.assertIn("_historical_train_records", source)
        self.assertIn("_historical_gradients", source)

    def test_no_automatic_second_configuration_or_validation_open(self):
        self.assertFalse(m.automatic_second_configuration_allowed())
        self.assertFalse(m.validation_opened_by_this_module())
        self.assertTrue(m.final_holdout_locked())
        self.assertFalse(m.production_promotion_allowed())


if __name__ == "__main__":
    unittest.main()
