import inspect
import math
import unittest

from st_omr_training import meter_v5_2j_domain_normalized_balance_audit_v1 as m


class TestMeterV52JDomainNormalizedBalanceAuditV1(unittest.TestCase):
    def test_zero_crossing_math_matches_observed_v52f_values(self):
        z2 = m._zero_crossing(-0.15352672660965055, 0.02106202077152308)
        z3 = m._zero_crossing(-0.1666666659793411, 0.017602953403478973)
        self.assertAlmostEqual(z2, 7.289268597494998, places=12)
        self.assertAlmostEqual(z3, 9.468108115675737, places=12)

    def test_minimax_reference_is_inside_cross_specialist_interval(self):
        result = m._minimax_reference(
            v5_2=-0.15352672660965055,
            source_2=0.02106202077152308,
            v5_3=-0.1666666659793411,
            source_3=0.017602953403478973,
        )
        self.assertAlmostEqual(result["lambda_source"], 8.281226081770031, places=12)
        self.assertGreater(result["lambda_source"], 7.289268597494998)
        self.assertLess(result["lambda_source"], 9.468108115675737)
        self.assertAlmostEqual(
            abs(result["residual_2"]),
            abs(result["residual_3"]),
            places=12,
        )

    def test_previous_raw_12_to_1_is_source_direction_for_both(self):
        residual2 = -0.15352672660965055 + 12.0 * 0.02106202077152308
        residual3 = -0.1666666659793411 + 12.0 * 0.017602953403478973
        self.assertGreater(residual2, 0.0)
        self.assertGreater(residual3, 0.0)
        self.assertGreater(12.0, 9.468108115675737)

    def test_no_training_or_io_authority(self):
        source = inspect.getsource(m).lower()
        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("image.open", source)
        self.assertNotIn("locate_checkpoint", source)
        self.assertFalse(m.training_allowed_by_this_module())
        self.assertFalse(m.validation_opened_by_this_module())
        self.assertTrue(m.final_holdout_locked())
        self.assertFalse(m.production_promotion_allowed())

    def test_expected_v52i_candidate_identity_is_frozen(self):
        self.assertEqual(
            m.EXPECTED_V52I_CANDIDATE_SHA,
            {
                "2": "6d7bc9d6593496a16d8ff18839766520dc1f04b90cfa34d8feb626a821cf6253",
                "3": "555ace1477abd6c9ab69c6cbea85aa4aa956f6099a571ab36dae94d4d60b1319",
            },
        )

    def test_invalid_zero_crossing_rejected(self):
        with self.assertRaises(ValueError):
            m._zero_crossing(-1.0, 0.0)
        with self.assertRaises(ValueError):
            m._zero_crossing(float("nan"), 1.0)


if __name__ == "__main__":
    unittest.main()
