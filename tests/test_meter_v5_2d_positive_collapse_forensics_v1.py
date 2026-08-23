import inspect
import unittest

from st_omr_training import meter_v5_2b_specialist_adaptation as v52b
from st_omr_training import meter_v5_2d_positive_collapse_forensics_v1 as m


class TestMeterV52DPositiveCollapseForensicsV1(unittest.TestCase):
    def test_frozen_data_surfaces(self):
        self.assertEqual(
            m.EXPECTED_M4A_TRAIN_COUNTS,
            {"2": 1527, "3": 1587, "4": 6396, "NONE": 17454},
        )
        self.assertEqual(m.EXPECTED_M4A_TRAIN_TOTAL, 26964)
        self.assertEqual(m.V5_TRAIN_SLOT_TOTAL, 540)
        self.assertEqual(m.V5_POSITIVE_PER_SPECIALIST, 90)
        self.assertEqual(m.V5_NEGATIVE_PER_SPECIALIST, 450)
        self.assertEqual(m.EXPECTED_POS_WEIGHT, 5.0)

    def test_stats_are_deterministic(self):
        result = m._stats([4.0, 1.0, 3.0, 2.0])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 4.0)
        self.assertEqual(result["median"], 2.5)
        self.assertEqual(result["mean"], 2.5)

    def test_batch_reconstruction_sees_all_90_450_each_epoch(self):
        rows = []
        for index in range(540):
            rows.append(
                {
                    "label_digit2": "1" if index < 90 else "0",
                    "label_digit3": "1" if 90 <= index < 180 else "0",
                }
            )
        for digit in ("2", "3"):
            audit = m._batch_construction_audit(rows, digit=digit)
            self.assertTrue(audit["all_rows_seen_once_per_epoch"])
            self.assertEqual(audit["epochs"], 12)
            for epoch in audit["epoch_summaries"]:
                self.assertEqual(epoch["positive"], 90)
                self.assertEqual(epoch["negative"], 450)
                self.assertEqual(epoch["batch_count"], 9)

    def test_parameter_drift_reports_head_and_features(self):
        torch, _nn = v52b._import_torch()
        frozen = v52b._build_digit_model().cpu()
        candidate = v52b._build_digit_model().cpu()
        candidate.load_state_dict(frozen.state_dict(), strict=True)
        with torch.no_grad():
            candidate.head.bias.add_(0.5)
        result = m._parameter_drift(frozen, candidate)
        self.assertGreater(result["head_bias"]["delta_l2"], 0.0)
        self.assertEqual(result["head_weight"]["delta_l2"], 0.0)
        self.assertTrue(result["feature_tensor_names"])

    def test_no_new_spatial_or_training_authority(self):
        self.assertFalse(m.training_allowed_by_this_module())
        self.assertFalse(m.validation_opened_by_this_module())
        self.assertTrue(m.final_holdout_locked())
        self.assertFalse(m.production_promotion_allowed())

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

    def test_no_automatic_root_cause_assignment_literal(self):
        source = inspect.getsource(m)
        self.assertIn('"root_cause_conclusion": "NOT_ASSIGNED_AUTOMATICALLY"', source)
        self.assertIn('"training_repair_authorized": False', source)
        self.assertIn('"new_bbox_work_authorized": False', source)


if __name__ == "__main__":
    unittest.main()
