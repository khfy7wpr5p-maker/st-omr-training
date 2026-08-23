import unittest

from st_omr_training import meter_v5_2m_retention_contract_v3 as m


class TestMeterV52MRetentionContractV3(unittest.TestCase):
    def test_corrected_frozen_oracle_is_v2(self):
        self.assertEqual(
            m.EXPECTED_FROZEN_COUNTS,
            {
                "2": {"tp": 185, "fp": 30, "fn": 1, "tn": 3156},
                "3": {"tp": 203, "fp": 1, "fn": 1, "tn": 3167},
                "4": {"tp": 788, "fp": 46, "fn": 4, "tn": 2534},
            },
        )

    def test_2ai_frozen_precision_is_below_old_absolute_floor(self):
        frozen = m.corrected_frozen_metrics()
        self.assertAlmostEqual(frozen["2"]["precision"], 185 / 215)
        self.assertLess(frozen["2"]["precision"], 0.98)

    def test_exact_frozen_candidate_passes_retention(self):
        frozen = m.corrected_frozen_metrics()
        result = m.evaluate_retention_gate_v3(
            frozen_metrics=frozen,
            candidate_metrics={"2": frozen["2"], "3": frozen["3"]},
        )
        self.assertEqual(result["gate"], "PASS")
        self.assertEqual(result["reasons"], [])
        self.assertFalse(result["absolute_precision_floor_used"])
        self.assertFalse(result["absolute_recall_floor_used"])

    def test_precision_drop_is_relative(self):
        frozen = m.corrected_frozen_metrics()
        candidate = {
            "2": dict(frozen["2"]),
            "3": dict(frozen["3"]),
        }
        candidate["2"]["precision"] = float(frozen["2"]["precision"]) - 0.006
        result = m.evaluate_retention_gate_v3(
            frozen_metrics=frozen,
            candidate_metrics=candidate,
        )
        self.assertEqual(result["gate"], "HOLD")
        self.assertIn("2-AI_PRECISION_DROP_GT_0.005", result["reasons"])

    def test_v52l_result_remains_hold_under_corrected_gate(self):
        frozen = m.corrected_frozen_metrics()
        candidate = {
            "2": {
                "precision": 0.8625592417061612,
                "recall": 0.978494623655914,
                "f1": 0.9168765743073048,
            },
            "3": {
                "precision": 0.9949748743718593,
                "recall": 0.9705882352941176,
                "f1": 0.9826302729528535,
            },
        }
        result = m.evaluate_retention_gate_v3(
            frozen_metrics=frozen,
            candidate_metrics=candidate,
        )
        self.assertEqual(result["gate"], "HOLD")
        self.assertIn("2-AI_F1_DROP_GT_0.005", result["reasons"])
        self.assertIn("2-AI_RECALL_DROP_GT_0.005", result["reasons"])
        self.assertIn("3-AI_F1_DROP_GT_0.005", result["reasons"])
        self.assertIn("3-AI_RECALL_DROP_GT_0.005", result["reasons"])

    def test_safety_boundary_is_read_only(self):
        safety = m.safety_boundary()
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
