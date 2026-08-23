import inspect
import unittest

from st_omr_training import meter_v5_2f_replay_balance_audit_v1 as m


class TestMeterV52FReplayBalanceAuditV1(unittest.TestCase):
    @staticmethod
    def _profile(count, positive, negative, weight=1.0):
        return {
            "count": count,
            "positive_weight": weight,
            "positive_pressure_total": positive,
            "negative_pressure_total": negative,
        }

    def test_signed_pressure_uses_negative_minus_positive(self):
        result = m._signed_pressure(
            self._profile(10, positive=9.0, negative=1.0),
            expected_count=10,
        )
        self.assertEqual(result["signed_total"], -8.0)
        self.assertEqual(result["signed_mean"], -0.8)

    def test_zero_crossing_ratio_is_analytical_and_exact(self):
        # V5 signed mean = -0.8; source signed mean = +0.4.
        # The exact source/V5 example ratio is therefore 2.0.
        result = m._balance_pair(
            self._profile(10, positive=9.0, negative=1.0),
            self._profile(20, positive=4.0, negative=12.0),
            v5_count=10,
            source_count=20,
        )
        self.assertTrue(result["domain_signed_pressures_oppose"])
        self.assertAlmostEqual(
            result["zero_crossing_source_examples_per_v5_example"],
            2.0,
            places=12,
        )
        self.assertAlmostEqual(
            result["zero_crossing_historical_examples_for_one_v5_pass"],
            20.0,
            places=12,
        )
        self.assertAlmostEqual(
            result["zero_crossing_fraction_of_full_historical_train"],
            1.0,
            places=12,
        )
        self.assertTrue(result["zero_crossing_within_one_full_source_pass"])
        self.assertAlmostEqual(
            result["full_source_pass_combined_signed_total"],
            0.0,
            places=12,
        )

    def test_same_direction_pressure_has_no_finite_balance(self):
        result = m._balance_pair(
            self._profile(10, positive=9.0, negative=1.0),
            self._profile(20, positive=18.0, negative=2.0),
            v5_count=10,
            source_count=20,
        )
        self.assertFalse(result["domain_signed_pressures_oppose"])
        self.assertIsNone(result["zero_crossing_source_examples_per_v5_example"])
        self.assertIsNone(result["zero_crossing_historical_examples_for_one_v5_pass"])
        self.assertFalse(result["zero_crossing_within_one_full_source_pass"])

    def test_cross_specialist_summary_does_not_select_a_ratio(self):
        per_digit = {
            "2": {
                "pos_weight_1": {
                    "zero_crossing_source_examples_per_v5_example": 7.0,
                    "zero_crossing_within_one_full_source_pass": True,
                }
            },
            "3": {
                "pos_weight_1": {
                    "zero_crossing_source_examples_per_v5_example": 9.0,
                    "zero_crossing_within_one_full_source_pass": True,
                }
            },
        }
        summary = m._cross_specialist_summary(per_digit, weight=1.0)
        self.assertTrue(summary["both_specialists_finite"])
        self.assertEqual(
            summary["zero_crossing_span_source_examples_per_v5_example"],
            {"min": 7.0, "max": 9.0},
        )
        self.assertTrue(summary["both_within_one_full_source_pass"])

    def test_stage_is_report_only_and_keeps_all_authority_closed(self):
        self.assertEqual(m.EXPECTED_WEIGHTS, (1.0, 5.0))
        self.assertEqual(m.EXPECTED_V5_COUNT, 540)
        self.assertEqual(m.EXPECTED_HISTORICAL_COUNT, 26964)
        self.assertFalse(m.training_allowed_by_this_module())
        self.assertFalse(m.validation_opened_by_this_module())
        self.assertTrue(m.final_holdout_locked())
        self.assertFalse(m.production_promotion_allowed())

        source = inspect.getsource(m).lower()
        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("image.open", source)
        self.assertNotIn("detect_multistaff_geometry", source)
        self.assertNotIn("_slot_edges", source)
        self.assertNotIn("nearest-staff", source)
        self.assertNotIn("tolerance expansion", source)
        self.assertIn('"replay_ratio_selected": false', source)
        self.assertIn('"positive_weight_selected": false', source)
        self.assertIn('"repair_training_authorized": false', source)
        self.assertIn('"checkpoint_read": false', source)
        self.assertIn('"image_read": false', source)
        self.assertIn('"v5_validation_opened": false', source)
        self.assertIn('"final_holdout_locked": true', source)


if __name__ == "__main__":
    unittest.main()
