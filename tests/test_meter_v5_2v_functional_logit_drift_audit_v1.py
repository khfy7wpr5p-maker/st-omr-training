from __future__ import annotations

import importlib.util
import inspect
import unittest

from st_omr_training import meter_v5_2v_functional_logit_drift_audit_v1 as v


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class TestMeterV52VFunctionalLogitDriftAuditV1(unittest.TestCase):
    def test_exact_hold_evidence_is_bound(self):
        self.assertEqual(
            v.V52U_IMPLEMENTATION_HEAD,
            "55c56671fef326a96909e169ee440a22986ff71b",
        )
        self.assertEqual(
            v.V52U_RETENTION_REPORT_SHA256,
            "6f072c99e4d6d60681a5c4739aecdb520327b1788b87c94f85c02583b343366f",
        )
        self.assertEqual(
            v.V52U_EXECUTION_ENVELOPE_SHA256,
            "87fb3230d694798096e3ce501cfaf681c96ac92fcb6dc7fc510cf23d891a9135",
        )

    def test_stage_is_read_only_and_emits_only_aggregate_train_evidence(self):
        boundary = v.safety_boundary()
        self.assertFalse(boundary["training"])
        self.assertFalse(boundary["autograd_grad_used"])
        self.assertFalse(boundary["backward"])
        self.assertEqual(boundary["optimizer_steps"], 0)
        self.assertFalse(boundary["checkpoint_write"])
        self.assertFalse(boundary["classifier_fit"])
        self.assertFalse(boundary["threshold_tuning"])
        self.assertFalse(boundary["bias_tuning"])
        self.assertFalse(boundary["historical_validation_opened"])
        self.assertTrue(boundary["historical_validation_retention_report_read"])
        self.assertFalse(boundary["historical_validation_error_examples_read"])
        self.assertFalse(boundary["per_example_rows_emitted"])
        self.assertFalse(boundary["first30_opened"])
        self.assertFalse(boundary["v5_validation_opened"])
        self.assertTrue(boundary["final_holdout_locked"])
        self.assertFalse(boundary["repair_selected"])
        self.assertFalse(v.validation_opened_by_this_module())
        self.assertFalse(v.production_promotion_allowed())

        source = inspect.getsource(v)
        for forbidden in (
            ".backward(",
            "torch.autograd",
            "torch.optim",
            "optimizer.step",
            "train_bounded_class_balanced_head_repair_v1(",
            "run_historical_retention_v1(",
            "run_first30_diagnostic_v1(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_small_weight_angle_can_create_large_functional_crossings(self):
        import torch

        def row(first, second):
            value = torch.zeros(64, dtype=torch.float32)
            value[0] = first
            value[1] = second
            return value

        frozen_weight = torch.zeros(64, dtype=torch.float32)
        frozen_weight[0] = 1.0
        candidate_weight = frozen_weight.clone()
        candidate_weight[1] = 0.1

        v5_features = torch.stack(
            (
                row(-0.05, 1.0),
                row(-0.20, 3.0),
                row(-1.00, 0.0),
                row(-0.80, -1.0),
            )
        )
        v5_targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        historical_features = torch.stack(
            (
                row(1.0, 0.0),
                row(0.8, -1.0),
                row(-0.05, 1.0),
                row(-0.20, 3.0),
            )
        )
        historical_targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

        result = v.functional_logit_drift_metrics_v1(
            historical_features=historical_features,
            historical_targets=historical_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
            frozen_weight=frozen_weight,
            candidate_weight=candidate_weight,
            frozen_bias=0.0,
            threshold=0.5,
        )

        self.assertEqual(result["head_geometry"]["gate"], "PASS")
        self.assertLess(result["head_geometry"]["head_angle_change_degrees"], 15.0)
        self.assertEqual(
            result["per_group"]["v5_positive"]["transition_counts"][
                "wrong_to_correct"
            ],
            2,
        )
        self.assertEqual(
            result["per_group"]["historical_negative"]["transition_counts"][
                "correct_to_wrong"
            ],
            2,
        )
        diagnosis = result["functional_retention_diagnosis"]
        self.assertTrue(
            diagnosis["parameter_geometry_passed_but_historical_decisions_changed"]
        )
        self.assertFalse(diagnosis["weight_space_bound_sufficient_for_decision_retention"])
        self.assertFalse(diagnosis["shared_linear_head_feasibility_proven"])
        self.assertFalse(diagnosis["representation_failure_proven"])
        self.assertFalse(diagnosis["repair_selected"])
        for group in v.GROUPS:
            self.assertTrue(
                result["per_group"][group]["functional_delta_identity_verified"]
            )
            self.assertTrue(result["per_group"][group]["cauchy_bound_verified"])
            self.assertLessEqual(
                result["per_group"][group]["cauchy_bound_utilization"]["max"],
                1.0 + 1e-10,
            )


if __name__ == "__main__":
    unittest.main()
