import inspect
import unittest

from st_omr_training import meter_v5_2i_repair_pilot_v1 as m
from st_omr_training import meter_v5_2h_repair_recipe_v1 as recipe


class TestMeterV52IRepairPilotV1(unittest.TestCase):
    def test_exact_recipe_is_frozen(self):
        self.assertEqual(recipe.REPLAY_RATIO, 12)
        self.assertEqual(recipe.HISTORICAL_REPLAY_COUNT, 6480)
        self.assertEqual(recipe.COMBINED_EXAMPLE_COUNT, 7020)
        self.assertEqual(recipe.POS_WEIGHT, 1.0)
        self.assertEqual(recipe.OPTIMIZER, "AdamW")
        self.assertEqual(recipe.LEARNING_RATE, 1e-4)
        self.assertEqual(recipe.WEIGHT_DECAY, 1e-4)
        self.assertEqual(recipe.BATCH_SIZE, 64)
        self.assertEqual(recipe.EPOCHS, 1)
        self.assertEqual(recipe.SEED, 52023)
        self.assertEqual(recipe.EXPECTED_OPTIMIZER_STEPS, 110)

    def test_replay_allocation_is_exact(self):
        self.assertEqual(
            m.EXPECTED_HISTORICAL_LABEL_COUNTS,
            {"2": 367, "3": 381, "4": 1537, "NONE": 4195},
        )
        self.assertEqual(sum(m.EXPECTED_HISTORICAL_LABEL_COUNTS.values()), 6480)

    def test_deterministic_stratified_selection_without_replacement(self):
        rows = []
        supply = {"2": 400, "3": 410, "4": 1600, "NONE": 4300}
        for label, count in supply.items():
            for index in range(count):
                rows.append(
                    {
                        "split": "train",
                        "digit_label": label,
                        "source_record_id": f"{label}-{index:05d}",
                        "bbox": [index, index + 1, index + 2, index + 3],
                    }
                )
        first = m.select_historical_replay_v1(rows)
        second = m.select_historical_replay_v1(list(reversed(rows)))
        self.assertEqual(
            [m._canonical_row(row) for row in first],
            [m._canonical_row(row) for row in second],
        )
        self.assertEqual(len(first), 6480)
        self.assertEqual(len({m._canonical_row(row) for row in first}), 6480)

    def test_gate_order_is_retention_then_diagnostic(self):
        self.assertEqual(
            m.gate_order(),
            ("historical_retention", "v5_first30_diagnostic"),
        )

    def test_safety_boundary_keeps_closed_surfaces_closed(self):
        safety = m.safety_boundary()
        self.assertTrue(safety["single_fixed_repair_pilot_authorized"])
        self.assertFalse(safety["automatic_second_configuration"])
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

    def test_wrong_approval_token_fails_before_data_access(self):
        with self.assertRaises(m.MeterV5_2IError):
            m.train_exact_repair_pilot_v1(
                "/does/not/exist",
                m4a_root="/does/not/exist",
                d10_root="/does/not/exist",
                digit2_frozen="/does/not/exist",
                digit3_frozen="/does/not/exist",
                confirmation="WRONG",
            )

    def test_only_frozen_historical_crop_helper_is_reused(self):
        source = inspect.getsource(m)
        self.assertIn("ret_legacy._historical_canvas_from_bbox", source)
        self.assertNotIn("detect_multistaff_geometry", source)
        self.assertNotIn("normalize_raster_page", source)
        self.assertNotIn("_integer_crop_box", source)
        self.assertNotIn("derive_staff_relative_slots_v1(", source)

    def test_no_automatic_second_configuration_or_open_validation(self):
        self.assertFalse(m.automatic_second_configuration_allowed())
        self.assertFalse(m.validation_opened_by_this_module())
        self.assertTrue(m.final_holdout_locked())
        self.assertFalse(m.production_promotion_allowed())


if __name__ == "__main__":
    unittest.main()
