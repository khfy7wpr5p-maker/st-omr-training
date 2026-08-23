import inspect
import unittest

from st_omr_training import meter_v5_2h_repair_recipe_v1 as m


class TestMeterV52HRepairRecipeV1(unittest.TestCase):
    def test_replay_ratio_is_fixed_by_evidence_rule(self):
        self.assertEqual(m.REPLAY_RATIO, 12)
        self.assertGreater(
            m.REPLAY_RATIO,
            max(m.ZERO_CROSSING_POS_WEIGHT_1.values()),
        )
        self.assertEqual(m.HISTORICAL_REPLAY_COUNT, 6480)
        self.assertEqual(m.COMBINED_EXAMPLE_COUNT, 7020)

    def test_historical_stratified_allocation_is_exact(self):
        self.assertEqual(
            m.HISTORICAL_LABEL_COUNTS,
            {"2": 367, "3": 381, "4": 1537, "NONE": 4195},
        )
        self.assertEqual(sum(m.HISTORICAL_LABEL_COUNTS.values()), 6480)

    def test_optimizer_recipe_is_fixed_and_single_epoch(self):
        recipe = m.recipe()
        self.assertEqual(recipe["positive_weight"], 1.0)
        self.assertEqual(recipe["optimizer"], "AdamW")
        self.assertEqual(recipe["learning_rate"], 1e-4)
        self.assertEqual(recipe["weight_decay"], 1e-4)
        self.assertEqual(recipe["batch_size"], 64)
        self.assertEqual(recipe["epochs"], 1)
        self.assertEqual(recipe["seed"], 52023)
        self.assertEqual(recipe["expected_optimizer_steps_if_later_authorized"], 110)
        self.assertEqual(recipe["previous_v5_only_optimizer_steps"], 108)
        self.assertTrue(recipe["sampling_without_replacement"])
        self.assertTrue(recipe["same_source_manifest_for_digit2_and_digit3"])
        self.assertTrue(recipe["first30_v5_diagnostic_zero_gradient"])

    def test_gates_are_fail_closed_and_ordered(self):
        gates = m.gates()
        self.assertTrue(gates["historical_retention_first"])
        self.assertEqual(gates["historical_abs_f1_drop_max"], 0.005)
        self.assertEqual(gates["historical_abs_recall_drop_max"], 0.005)
        self.assertEqual(gates["historical_precision_min"], 0.98)
        self.assertEqual(gates["historical_recall_min"], 0.98)
        self.assertEqual(gates["v5_diagnostic_2_of_4_min"], 8)
        self.assertEqual(gates["v5_diagnostic_3_of_4_min"], 8)
        self.assertEqual(gates["v5_diagnostic_4_of_4_min"], 9)
        self.assertEqual(gates["v5_denominator_exact4_min"], 26)
        self.assertFalse(gates["automatic_second_configuration"])

    def test_no_gradient_or_spatial_authority_is_implemented(self):
        source = inspect.getsource(m).lower()
        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("image.open", source)
        safety = m.safety_boundary()
        self.assertFalse(safety["repair_training_authorized"])
        self.assertFalse(safety["training"])
        self.assertFalse(safety["backward"])
        self.assertEqual(safety["optimizer_steps"], 0)
        self.assertFalse(safety["checkpoint_write"])
        self.assertFalse(safety["threshold_tuning"])
        self.assertFalse(safety["new_bbox"])
        self.assertFalse(safety["new_crop_geometry"])
        self.assertFalse(safety["new_spatial_heuristic"])
        self.assertFalse(safety["reserve_v5_train_opened"])
        self.assertFalse(safety["v5_validation_opened"])
        self.assertTrue(safety["final_holdout_locked"])
        self.assertTrue(safety["digit4_frozen"])
        self.assertFalse(safety["resolver_wiring"])
        self.assertFalse(safety["production_promotion"])


if __name__ == "__main__":
    unittest.main()
