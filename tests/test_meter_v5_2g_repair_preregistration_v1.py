import inspect
import unittest

from st_omr_training import meter_v5_2g_repair_preregistration_v1 as m


class TestMeterV52GRepairPreregistrationV1(unittest.TestCase):
    def test_pos_weight_1_shared_boundary_is_evidence_bound_and_unselected(self):
        boundary = m.shared_single_pass_feasibility_boundary()
        self.assertEqual(
            boundary["strict_lower_bound_source_examples_per_v5_example"],
            m.ZERO_CROSSING_POS_WEIGHT_1["3"],
        )
        self.assertEqual(
            boundary["upper_bound_source_examples_per_v5_example"],
            m.FULL_HISTORICAL_PASS_RATIO,
        )
        self.assertGreater(
            boundary["upper_bound_source_examples_per_v5_example"],
            boundary["strict_lower_bound_source_examples_per_v5_example"],
        )
        self.assertTrue(boundary["lower_bound_is_zero_crossing_not_safety_margin"])
        self.assertFalse(boundary["replay_ratio_selected"])
        self.assertFalse(boundary["positive_weight_selected"])
        self.assertFalse(boundary["sampling_strategy_selected"])

    def test_pos_weight_5_cannot_balance_inside_one_full_historical_pass(self):
        feasible = m.frozen_pos_weight_5_is_single_pass_feasible()
        self.assertEqual(feasible, {"2": False, "3": False})
        for ratio in m.ZERO_CROSSING_POS_WEIGHT_5.values():
            self.assertGreater(ratio, m.FULL_HISTORICAL_PASS_RATIO)

    def test_no_training_or_spatial_authority_exists(self):
        source = inspect.getsource(m).lower()
        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("image.open", source)
        self.assertNotIn("bbox", source.replace('"new_bbox"', ""))
        self.assertFalse(m.repair_training_authorized())

    def test_safety_boundary_is_fail_closed(self):
        safety = m.safety_boundary()
        self.assertFalse(safety["training"])
        self.assertFalse(safety["backward"])
        self.assertEqual(safety["optimizer_steps"], 0)
        self.assertFalse(safety["checkpoint_write"])
        self.assertFalse(safety["threshold_tuning"])
        self.assertFalse(safety["replay_ratio_selected"])
        self.assertFalse(safety["positive_weight_selected"])
        self.assertFalse(safety["sampling_strategy_selected"])
        self.assertFalse(safety["repair_training_authorized"])
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
