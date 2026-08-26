from pathlib import Path
import unittest

from st_omr_training import meter_v5_2b_specialist_adaptation as v52b
from st_omr_training import meter_v5_3e_rescue_training_preregistration_v1 as v53e
from st_omr_training import meter_v5_3f_rescue_training_execution_harness_v1 as v53f
from st_omr_training import meter_v5_3i_train_acceptance_gate_v1 as v53i
from st_omr_training import meter_v5_3j_rescue_failure_forensics_v1 as v53j
from st_omr_training import meter_v5_3k_feature_domain_shift_forensics_v1 as v53k


class TestMeterV53KFeatureDomainShiftForensics(unittest.TestCase):
    def _v53j_fixture(self):
        return {
            "schema": v53j.SCHEMA,
            "v5_3i_decision_reproduced": "HOLD",
            "frozen_state_bit_identical": True,
            "rescue_state_bit_identical_during_forensics": True,
            "diagnosis_scope": "TRAIN-only descriptive forensics",
            "repair_recipe_selected": False,
            "retraining_authorized": False,
            "historical_validation_opened": False,
            "first30_opened": False,
            "v5_reserve_opened": False,
            "v5_validation_opened": False,
            "final_holdout_locked": True,
            "bound_evidence": {
                "v5_3i_head_sha": v53j.V53I_HEAD_SHA,
                "v5_3i_report_sha256": v53j.EXPECTED_V53I_REPORT_SHA256,
                "v5_3g_report_sha256": v53i.EXPECTED_V53G_REPORT_SHA256,
                "v5_3h_envelope_sha256": v53i.EXPECTED_V53H_ENVELOPE_SHA256,
                "rescue_artifact_sha256": dict(v53i.EXPECTED_RESCUE_ARTIFACT_SHA256),
            },
            "per_specialist": {
                "2": {
                    "failure_signature": {
                        "signature": v53k.EXPECTED_FAILURE_SIGNATURES["2"],
                        "historical_true_negative_regression_count": 5307,
                        "v5_positive_recovery_fraction": 1.0,
                    },
                    "v5_3i_acceptance_witness_reproduced": True,
                    "group_identity_reverified": True,
                },
                "3": {
                    "failure_signature": {
                        "signature": v53k.EXPECTED_FAILURE_SIGNATURES["3"],
                        "historical_true_negative_regression_count": 15775,
                        "v5_positive_recovery_fraction": 0.0,
                    },
                    "v5_3i_acceptance_witness_reproduced": True,
                    "group_identity_reverified": True,
                },
            },
        }

    def test_contract_is_read_only_and_protected_surfaces_remain_closed(self):
        boundary = v53k.safety_boundary()
        self.assertFalse(boundary["training"])
        self.assertFalse(boundary["fitting"])
        self.assertFalse(boundary["backward"])
        self.assertEqual(boundary["optimizer_steps"], 0)
        self.assertFalse(boundary["checkpoint_write"])
        self.assertFalse(boundary["rescue_artifact_write"])
        self.assertFalse(boundary["threshold_tuning"])
        self.assertFalse(boundary["threshold_sweep"])
        self.assertFalse(boundary["repair_recipe_selected"])
        self.assertFalse(boundary["retraining_authorized"])
        self.assertFalse(boundary["historical_validation_opened"])
        self.assertFalse(boundary["first30_opened"])
        self.assertFalse(boundary["v5_reserve_opened"])
        self.assertFalse(boundary["v5_validation_opened"])
        self.assertTrue(boundary["final_holdout_locked"])
        self.assertFalse(boundary["digit4_loaded"])
        self.assertFalse(v53k.retraining_allowed_after_forensics())
        self.assertFalse(v53k.threshold_tuning_allowed())
        self.assertFalse(v53k.historical_validation_access_allowed())
        self.assertFalse(v53k.first30_access_allowed())
        self.assertFalse(v53k.v5_validation_access_allowed())
        self.assertFalse(v53k.final_holdout_access_allowed())

    def test_contract_binds_exact_v53j_receipt_and_fixed_descriptive_dimensions(self):
        contract = v53k.forensic_contract()
        self.assertEqual(contract["prerequisite_v5_3j_final_head"], "08b2458cf6fa4aee3e5f32d1aefbe637cdbd01ec")
        self.assertEqual(contract["prerequisite_v5_3j_implementation_head"], "c978b14fba23f91c60f06d2166bb23e87856d8d6")
        self.assertEqual(contract["prerequisite_v5_3j_module_blob"], "092a32504ffee9b9aafa74ddefea1c2aeb831e56")
        self.assertEqual(contract["bound_v5_3j_report_sha256"], "7a49d29e0d7257be7c59d499ab3d9ab575d369a7473b0b5298ea62aa80c7d37f")
        self.assertEqual(contract["fixed_rescue_threshold"], 0.50)
        self.assertEqual(contract["frozen_feature_dimension"], 64)
        self.assertEqual(contract["rescue_hidden_dimension"], 8)
        self.assertEqual(contract["top_feature_dimensions_reported"], 10)
        self.assertEqual(contract["top_hidden_dimensions_reported"], 8)
        self.assertTrue(contract["no_classifier_fit"])
        self.assertTrue(contract["no_pca_fit"])
        self.assertTrue(contract["no_threshold_selection"])

    def test_v53j_semantic_receipt_is_admitted_and_tamper_fails_closed(self):
        receipt = self._v53j_fixture()
        v53k._validate_v53j_report(receipt)
        receipt["per_specialist"]["3"]["failure_signature"]["historical_true_negative_regression_count"] = 15774
        with self.assertRaises(v53k.MeterV5_3KError):
            v53k._validate_v53j_report(receipt)

    def test_standardized_shift_identifies_known_shifted_dimension(self):
        torch, _nn = v52b._import_torch()
        a = torch.tensor(
            [
                [10.0, 0.0, 1.0],
                [12.0, 1.0, 2.0],
                [11.0, -1.0, 3.0],
            ],
            dtype=torch.float32,
        )
        b = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 2.0],
                [-1.0, -1.0, 3.0],
            ],
            dtype=torch.float32,
        )
        shift = v53k._standardized_mean_shift(a, b, expected_dim=3, top_k=2, label="unit")
        self.assertEqual(shift["top_dimensions"][0]["dimension"], 0)
        self.assertGreater(shift["top_dimensions"][0]["signed_standardized_mean_shift"], 0.0)
        self.assertEqual(shift["top_k"], 2)

    def test_centroid_geometry_is_descriptive_and_finite(self):
        torch, _nn = v52b._import_torch()
        a = torch.tensor([[0.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
        b = torch.tensor([[3.0, 0.0], [3.0, 2.0]], dtype=torch.float32)
        geometry = v53k._centroid_geometry(a, b, expected_dim=2, label="unit")
        self.assertAlmostEqual(geometry["centroid_l2_distance"], 3.0, places=12)
        self.assertAlmostEqual(geometry["a_within_centroid_rms"], 1.0, places=12)
        self.assertAlmostEqual(geometry["b_within_centroid_rms"], 1.0, places=12)
        self.assertAlmostEqual(geometry["centroid_distance_over_sum_within_rms"], 1.5, places=9)

    def test_hidden_output_decomposition_closes_exactly(self):
        torch, _nn = v52b._import_torch()
        model = v53f._build_rescue_model_v1()
        a = torch.zeros((4, v53e.HIDDEN_WIDTH), dtype=torch.float64)
        b = torch.zeros((5, v53e.HIDDEN_WIDTH), dtype=torch.float64)
        a[:, 0] = 1.0
        result = v53k._output_gap_decomposition(model, a, b, label="unit")
        self.assertAlmostEqual(
            result["mean_logit_gap_a_minus_b"],
            result["contribution_sum"],
            places=12,
        )
        self.assertEqual(len(result["hidden_dimension_contributions"]), v53e.HIDDEN_WIDTH)

    def test_pair_diagnostics_has_64d_8d_and_optional_output_decomposition(self):
        torch, _nn = v52b._import_torch()
        model = v53f._build_rescue_model_v1()
        a_features = torch.zeros((3, v53e.FEATURE_DIM), dtype=torch.float32)
        b_features = torch.ones((4, v53e.FEATURE_DIM), dtype=torch.float32)
        a_hidden = v53k._hidden_activations(model, a_features, label="a")
        b_hidden = v53k._hidden_activations(model, b_features, label="b")
        result = v53k._pair_diagnostics(
            a_features=a_features,
            b_features=b_features,
            a_hidden=a_hidden,
            b_hidden=b_hidden,
            rescue_model=model,
            label="unit",
            include_output_decomposition=True,
        )
        self.assertEqual(result["feature_64d"]["a_summary"]["dimension"], 64)
        self.assertEqual(result["hidden_8d"]["a_summary"]["dimension"], 8)
        self.assertIn("output_logit_gap_decomposition", result)

    def test_future_gate_order_requires_separate_digit_specific_hypothesis(self):
        self.assertEqual(
            v53k.future_gate_order(),
            (
                "v5_3k_train_only_feature_domain_shift_forensics",
                "separately_preregistered_digit_specific_repair_hypothesis_if_supported",
                "separately_authorized_single_repair_execution_if_approved",
                "new_train_acceptance_gate",
                "historical_validation_retention_only_after_train_acceptance_pass",
            ),
        )

    def test_source_contains_no_training_or_protected_gate_entry_points(self):
        source = Path(v53k.__file__).read_text(encoding="utf-8")
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
        self.assertIn('"repair_recipe_selected": False', source)
        self.assertIn('"final_holdout_locked": True', source)


if __name__ == "__main__":
    unittest.main()
