from pathlib import Path
import unittest

from st_omr_training import meter_v5_2b_specialist_adaptation as v52b
from st_omr_training import meter_v5_3j_rescue_failure_forensics_v1 as v53j


class TestMeterV53JRescueFailureForensics(unittest.TestCase):
    def test_v5_3j_contract_is_read_only_and_keeps_protected_surfaces_closed(self):
        boundary = v53j.safety_boundary()
        self.assertFalse(boundary["training"])
        self.assertFalse(boundary["backward"])
        self.assertEqual(boundary["optimizer_steps"], 0)
        self.assertFalse(boundary["checkpoint_write"])
        self.assertFalse(boundary["rescue_artifact_write"])
        self.assertFalse(boundary["threshold_tuning"])
        self.assertFalse(boundary["threshold_sweep"])
        self.assertFalse(boundary["automatic_second_configuration"])
        self.assertFalse(boundary["retraining_authorized"])
        self.assertFalse(boundary["historical_validation_opened"])
        self.assertFalse(boundary["first30_opened"])
        self.assertFalse(boundary["v5_reserve_opened"])
        self.assertFalse(boundary["v5_validation_opened"])
        self.assertTrue(boundary["final_holdout_locked"])
        self.assertFalse(boundary["digit4_loaded"])
        self.assertFalse(v53j.retraining_allowed_after_forensics())
        self.assertFalse(v53j.threshold_tuning_allowed())
        self.assertFalse(v53j.historical_validation_access_allowed())
        self.assertFalse(v53j.first30_access_allowed())
        self.assertFalse(v53j.v5_validation_access_allowed())
        self.assertFalse(v53j.final_holdout_access_allowed())

    def test_v5_3j_is_bound_to_exact_hold_receipt_and_witness(self):
        contract = v53j.forensic_contract()
        self.assertEqual(
            contract["prerequisite_v5_3i_head"],
            "88c7acc551fa2b00b1f877f6a839704d58825adb",
        )
        self.assertEqual(
            contract["prerequisite_v5_3i_module_blob"],
            "abb5f1ae4c42b0c5f3ae26b80f2a467f47582197",
        )
        self.assertEqual(
            contract["bound_v5_3i_report_sha256"],
            "448b807086bc9ee66d090fdf173ce54e3c5e2a133e60cf6ae0a791aed2717434",
        )
        self.assertEqual(contract["required_v5_3i_decision"], "HOLD")
        self.assertEqual(contract["bound_hold_reasons"], list(v53j.EXPECTED_HOLD_REASONS))
        self.assertEqual(v53j.EXPECTED_ACCEPTANCE_WITNESS["2"]["historical_regressions"], 5307)
        self.assertEqual(v53j.EXPECTED_ACCEPTANCE_WITNESS["3"]["v5_fn"], 90)
        self.assertEqual(v53j.EXPECTED_ACCEPTANCE_WITNESS["3"]["historical_regressions"], 15775)

    def test_probability_distribution_and_rank_fraction_are_descriptive_only(self):
        torch, _nn = v52b._import_torch()
        values = torch.tensor([0.1, 0.2, 0.8, 0.9], dtype=torch.float32)
        dist = v53j._probability_distribution(values, label="unit")
        self.assertEqual(dist["count"], 4)
        self.assertAlmostEqual(dist["min"], 0.1)
        self.assertAlmostEqual(dist["max"], 0.9)
        self.assertAlmostEqual(dist["mean"], 0.5)

        pos = torch.tensor([0.8, 0.9], dtype=torch.float32)
        neg = torch.tensor([0.1, 0.2], dtype=torch.float32)
        self.assertAlmostEqual(v53j._pairwise_rank_fraction(pos, neg, label="perfect"), 1.0)
        self.assertAlmostEqual(v53j._pairwise_rank_fraction(neg, pos, label="reverse"), 0.0)
        ties = torch.tensor([0.5], dtype=torch.float32)
        self.assertAlmostEqual(v53j._pairwise_rank_fraction(ties, ties, label="tie"), 0.5)

    def test_score_group_diagnostics_reports_corrections_and_regressions_at_fixed_threshold(self):
        torch, _nn = v52b._import_torch()
        frozen = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float32)
        rescue = torch.tensor([0.9, 0.8, 0.9, 0.1], dtype=torch.float32)
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
        evidence = v53j._score_group_diagnostics(
            frozen_probability=frozen,
            rescue_probability=rescue,
            targets=targets,
            frozen_threshold=0.48,
            rescue_threshold=0.50,
            label="unit",
        )
        self.assertEqual(evidence["eligible_positive_count"], 2)
        self.assertEqual(evidence["eligible_negative_count"], 2)
        self.assertEqual(evidence["eligible_positive_rescue_above_threshold"], 2)
        self.assertEqual(evidence["eligible_positive_rescue_below_threshold"], 0)
        self.assertEqual(evidence["eligible_negative_rescue_above_threshold"], 1)
        self.assertEqual(evidence["eligible_negative_rescue_below_threshold"], 1)
        self.assertAlmostEqual(evidence["positive_over_negative_rank_fraction"], 0.625)

    def test_failure_signature_does_not_select_repair_recipe(self):
        torch, _nn = v52b._import_torch()
        signature = v53j._failure_signature(
            digit="2",
            v5={"eligible_positive_count": 2, "eligible_positive_rescue_above_threshold": 2},
            historical={"eligible_negative_rescue_above_threshold": 1},
            v5_positive_scores=torch.tensor([0.8, 0.9]),
            historical_negative_scores=torch.tensor([0.1, 0.7]),
        )
        self.assertEqual(signature["signature"], "V5_RECOVERED_HISTORICAL_TN_COLLAPSE")
        self.assertEqual(signature["v5_positive_recovery_fraction"], 1.0)
        self.assertEqual(signature["historical_true_negative_regression_count"], 1)
        self.assertFalse(signature["fixed_threshold_separates_required_groups"])
        self.assertIn("no threshold", signature["interpretation_scope"])

    def test_v5_3j_source_contains_no_training_or_protected_gate_entry_points(self):
        source = Path(v53j.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "run_authoritative_rescue_training_v1(",
            "execute_rescue_tensor_harness_v1(",
            "torch.optim.",
            ".backward(",
            "optimizer.step(",
            "run_historical_retention_gate(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertIn('"threshold_tuning": False', source)
        self.assertIn('"historical_validation_opened": False', source)
        self.assertIn('"final_holdout_locked": True', source)


if __name__ == "__main__":
    unittest.main()
