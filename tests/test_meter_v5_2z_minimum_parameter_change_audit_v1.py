from __future__ import annotations

import inspect
import unittest

import numpy as np

from st_omr_training import meter_v5_2z_minimum_parameter_change_audit_v1 as v


def _row(first: float, second: float = 0.0) -> np.ndarray:
    value = np.zeros(64, dtype=np.float64)
    value[0] = first
    value[1] = second
    return value


class TestMeterV52ZMinimumParameterChangeAuditV1(unittest.TestCase):
    def test_exact_v5_2y_execution_is_bound(self):
        self.assertEqual(
            v.V52Y_IMPLEMENTATION_HEAD,
            "18e23ed2c25e50db03f41db70259db3fd74e224a",
        )
        self.assertEqual(
            v.V52Y_REPORT_SHA256,
            "d9f7133d02a0875f09a79e0ecb53a5ae2f510e92164d14b38e171f1042655913",
        )
        self.assertEqual(
            v.V52Y_EXECUTION_ENVELOPE_SHA256,
            "b56ee42a865d61c4a19e5bc6038f5b9094b5d91b6135be2b39e5f8eecce43d10",
        )

    def test_stage_is_train_only_and_keeps_all_gates_closed(self):
        boundary = v.safety_boundary()
        self.assertFalse(boundary["model_training"])
        self.assertFalse(boundary["autograd_grad_used"])
        self.assertFalse(boundary["backward"])
        self.assertEqual(boundary["optimizer_steps"], 0)
        self.assertTrue(boundary["diagnostic_linear_program_solve"])
        self.assertTrue(boundary["diagnostic_minimum_parameter_witness_fit"])
        self.assertFalse(boundary["diagnostic_witness_persisted"])
        self.assertFalse(boundary["diagnostic_witness_values_emitted"])
        self.assertFalse(boundary["model_parameter_mutation"])
        self.assertFalse(boundary["candidate_checkpoint_write"])
        self.assertFalse(boundary["threshold_tuning"])
        self.assertFalse(boundary["bias_selection"])
        self.assertFalse(boundary["historical_validation_opened"])
        self.assertFalse(boundary["first30_opened"])
        self.assertFalse(boundary["v5_validation_opened"])
        self.assertTrue(boundary["final_holdout_locked"])
        self.assertTrue(boundary["digit4_frozen"])
        self.assertFalse(boundary["repair_selected"])

        source = inspect.getsource(v)
        for forbidden in (
            ".backward(",
            "torch.autograd",
            "torch.optim",
            "optimizer.step",
            "run_historical_retention_v1(",
            "run_first30_diagnostic_v1(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_solver_contract_has_one_unconditional_parameter_objective(self):
        contract = v.solver_contract()
        self.assertEqual(contract["expected_library_version"], "1.18.0")
        self.assertEqual(contract["method"], "highs-ds")
        self.assertFalse(contract["presolve"])
        self.assertEqual(contract["decision_margin"], 1e-4)
        self.assertEqual(
            contract["objective"],
            "minimum_max_absolute_delta_weight_under_decision_constraints",
        )
        self.assertTrue(contract["maximum_absolute_delta_weight_minimized"])
        self.assertFalse(contract["historical_logit_drift_constrained"])
        self.assertTrue(contract["historical_logit_drift_descriptive_only"])
        self.assertFalse(contract["weight_l1_minimized"])
        self.assertFalse(contract["weight_l2_minimized"])
        self.assertFalse(contract["automatic_second_solver"])
        self.assertFalse(contract["solver_sweep"])
        self.assertFalse(contract["threshold_search"])
        self.assertFalse(contract["bias_search"])

    def test_known_unconditional_minimum_is_verified_without_witness_values(self):
        historical_features = np.stack((_row(1.0), _row(-2.0)))
        historical_targets = np.asarray([1.0, 0.0], dtype=np.float64)
        v5_features = np.stack((_row(1.0), _row(-1.0)))
        v5_targets = np.asarray([1.0, 0.0], dtype=np.float64)

        result = v.minimum_parameter_change_v1(
            historical_features=historical_features,
            historical_targets=historical_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
            frozen_weight=np.zeros(64, dtype=np.float64),
            frozen_bias=0.0,
            threshold=0.5,
            v5_2y_conditional_minimum_max_abs_delta_weight=2.0 * v.DECISION_MARGIN,
        )

        self.assertEqual(result["witness_claim"], "WITNESS_VERIFIED")
        self.assertEqual(
            result["optimality_claim"],
            "SOLVER_REPORTED_OPTIMAL_NOT_FORMAL_PROOF",
        )
        self.assertTrue(result["minimum_parameter_change_witness_verified"])
        self.assertAlmostEqual(
            result["minimum_max_absolute_delta_weight"],
            v.DECISION_MARGIN,
            places=8,
        )
        self.assertAlmostEqual(
            result["independently_recomputed_max_absolute_delta_weight"],
            v.DECISION_MARGIN,
            places=8,
        )
        self.assertEqual(result["v5_constraint_violations"], 0)
        self.assertEqual(result["historical_retention_constraint_violations"], 0)
        self.assertEqual(result["parameter_bound_violations"], 0)
        self.assertTrue(result["conditional_upper_bound_consistency_verified"])
        self.assertEqual(result["historical_transition_counts"]["correct_to_wrong"], 0)
        self.assertEqual(result["v5_transition_counts"]["wrong_to_correct"], 1)
        self.assertFalse(result["historical_logit_drift_constrained"])
        self.assertFalse(result["witness_values_emitted"])
        self.assertFalse(result["witness_persisted"])
        self.assertFalse(result["repair_selected"])
        for forbidden_key in ("weight", "delta_weight", "witness", "candidate_weight"):
            self.assertNotIn(forbidden_key, result)

    def test_v5_2y_upper_bound_contradiction_is_fail_closed(self):
        historical_features = np.stack((_row(1.0), _row(-2.0)))
        historical_targets = np.asarray([1.0, 0.0], dtype=np.float64)
        v5_features = np.stack((_row(1.0), _row(-1.0)))
        v5_targets = np.asarray([1.0, 0.0], dtype=np.float64)

        result = v.minimum_parameter_change_v1(
            historical_features=historical_features,
            historical_targets=historical_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
            frozen_weight=np.zeros(64, dtype=np.float64),
            frozen_bias=0.0,
            threshold=0.5,
            v5_2y_conditional_minimum_max_abs_delta_weight=0.0,
        )

        self.assertEqual(result["witness_claim"], "UNPROVEN_WITNESS_RESIDUAL_FAILED")
        self.assertFalse(result["conditional_upper_bound_consistency_verified"])
        self.assertFalse(result["minimum_parameter_change_witness_verified"])
        diagnosis = v.path_diagnosis_v1(result)
        self.assertEqual(
            diagnosis["status"],
            "EVIDENCE_CONFLICT_V5_2Y_CONDITIONAL_UPPER_BOUND_BELOW_V5_2Z_RESULT",
        )

    def test_inconsistent_v5_constraints_are_fail_closed(self):
        historical_features = np.stack((_row(1.0), _row(-1.0)))
        historical_targets = np.asarray([1.0, 0.0], dtype=np.float64)
        v5_features = np.stack((_row(1.0), _row(1.0)))
        v5_targets = np.asarray([1.0, 0.0], dtype=np.float64)

        result = v.minimum_parameter_change_v1(
            historical_features=historical_features,
            historical_targets=historical_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
            frozen_weight=_row(1.0),
            frozen_bias=0.0,
            threshold=0.5,
            v5_2y_conditional_minimum_max_abs_delta_weight=1.0,
        )

        self.assertEqual(
            result["witness_claim"],
            "SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF",
        )
        self.assertEqual(result["optimality_claim"], "OPTIMALITY_NOT_PROVEN")
        self.assertFalse(result["minimum_parameter_change_witness_verified"])
        diagnosis = v.path_diagnosis_v1(result)
        self.assertEqual(
            diagnosis["status"],
            "EVIDENCE_CONFLICT_V5_2Y_WITNESS_BUT_V5_2Z_SOLVER_REPORTED_INFEASIBLE",
        )


if __name__ == "__main__":
    unittest.main()
