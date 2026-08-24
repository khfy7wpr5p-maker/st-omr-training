from __future__ import annotations

import inspect
import unittest

import numpy as np

from st_omr_training import meter_v5_2y_lexicographic_parameter_stability_audit_v1 as v


def _row(first: float, second: float = 0.0) -> np.ndarray:
    value = np.zeros(64, dtype=np.float64)
    value[0] = first
    value[1] = second
    return value


class TestMeterV52YLexicographicParameterStabilityAuditV1(unittest.TestCase):
    def test_exact_v5_2x_execution_is_bound(self):
        self.assertEqual(
            v.V52X_IMPLEMENTATION_HEAD,
            "c276517e07ee129d80b53bb906ead06c3094f0af",
        )
        self.assertEqual(
            v.V52X_REPORT_SHA256,
            "46dbba37c1ae88e6212afa1a1fef92ecfb92935a21425437c494c9a320aafc51",
        )
        self.assertEqual(
            v.V52X_EXECUTION_ENVELOPE_SHA256,
            "30aee74467349fd6649ea3c6bc27d2f7f669c7a8341da7ef131d2cafda171846",
        )

    def test_stage_is_train_only_and_keeps_all_gates_closed(self):
        boundary = v.safety_boundary()
        self.assertFalse(boundary["model_training"])
        self.assertFalse(boundary["autograd_grad_used"])
        self.assertFalse(boundary["backward"])
        self.assertEqual(boundary["optimizer_steps"], 0)
        self.assertTrue(boundary["diagnostic_linear_program_solve"])
        self.assertTrue(boundary["diagnostic_lexicographic_witness_fit"])
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

    def test_solver_contract_has_one_fixed_secondary_lp(self):
        contract = v.solver_contract()
        self.assertEqual(contract["expected_library_version"], "1.18.0")
        self.assertEqual(contract["method"], "highs-ds")
        self.assertFalse(contract["presolve"])
        self.assertEqual(contract["decision_margin"], 1e-4)
        self.assertEqual(contract["primary_drift_absolute_slack"], 1e-6)
        self.assertEqual(contract["objective"], "minimum_max_absolute_delta_weight")
        self.assertTrue(contract["maximum_absolute_delta_weight_minimized"])
        self.assertFalse(contract["primary_functional_drift_reoptimized"])
        self.assertEqual(contract["primary_optimum_consistency_tolerance"], 1e-7)
        self.assertFalse(contract["weight_l1_minimized"])
        self.assertFalse(contract["weight_l2_minimized"])
        self.assertFalse(contract["automatic_second_solver"])
        self.assertFalse(contract["solver_sweep"])
        self.assertFalse(contract["threshold_search"])
        self.assertFalse(contract["bias_search"])

    def test_known_secondary_minimum_is_verified_and_nullspace_is_removed(self):
        # Only feature 0 affects any constraint; feature 1 is a null-space
        # direction.  The secondary min-Linf LP must leave that direction at
        # zero and select delta[0] = margin without emitting either value.
        historical_features = np.stack((_row(1.0), _row(-2.0)))
        historical_targets = np.asarray([1.0, 0.0], dtype=np.float64)
        v5_features = np.stack((_row(1.0), _row(-1.0)))
        v5_targets = np.asarray([1.0, 0.0], dtype=np.float64)

        result = v.lexicographic_parameter_stability_v1(
            historical_features=historical_features,
            historical_targets=historical_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
            frozen_weight=np.zeros(64, dtype=np.float64),
            frozen_bias=0.0,
            threshold=0.5,
            primary_max_absolute_historical_logit_drift=2.0 * v.DECISION_MARGIN,
        )

        self.assertEqual(result["witness_claim"], "WITNESS_VERIFIED")
        self.assertEqual(
            result["secondary_optimality_claim"],
            "SOLVER_REPORTED_OPTIMAL_NOT_FORMAL_PROOF",
        )
        self.assertTrue(result["parameter_stability_witness_verified"])
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
        self.assertEqual(result["historical_drift_cap_violations"], 0)
        self.assertEqual(result["parameter_bound_violations"], 0)
        self.assertLessEqual(
            result["independently_recomputed_max_absolute_historical_logit_drift"],
            result["primary_drift_cap"] + v.WITNESS_TOLERANCE,
        )
        self.assertTrue(result["primary_optimum_consistency_verified"])
        self.assertEqual(result["historical_transition_counts"]["correct_to_wrong"], 0)
        self.assertEqual(result["v5_transition_counts"]["wrong_to_correct"], 1)
        self.assertFalse(result["witness_values_emitted"])
        self.assertFalse(result["witness_persisted"])
        self.assertTrue(result["maximum_absolute_delta_weight_minimized"])
        self.assertFalse(result["weight_l1_minimized"])
        self.assertFalse(result["weight_l2_minimized"])
        for forbidden_key in ("weight", "delta_weight", "witness", "candidate_weight"):
            self.assertNotIn(forbidden_key, result)

    def test_primary_cap_conflict_is_fail_closed(self):
        historical_features = np.stack((_row(1.0), _row(-2.0)))
        historical_targets = np.asarray([1.0, 0.0], dtype=np.float64)
        v5_features = np.stack((_row(1.0), _row(-1.0)))
        v5_targets = np.asarray([1.0, 0.0], dtype=np.float64)

        result = v.lexicographic_parameter_stability_v1(
            historical_features=historical_features,
            historical_targets=historical_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
            frozen_weight=np.zeros(64, dtype=np.float64),
            frozen_bias=0.0,
            threshold=0.5,
            primary_max_absolute_historical_logit_drift=0.0,
        )

        self.assertEqual(
            result["witness_claim"],
            "SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF",
        )
        self.assertEqual(result["secondary_optimality_claim"], "OPTIMALITY_NOT_PROVEN")
        self.assertFalse(result["parameter_stability_witness_verified"])
        self.assertFalse(result["repair_selected"])
        diagnosis = v.path_diagnosis_v1(result)
        self.assertEqual(
            diagnosis["status"],
            "EVIDENCE_CONFLICT_V5_2X_WITNESS_BUT_V5_2Y_SOLVER_REPORTED_INFEASIBLE",
        )
        self.assertFalse(diagnosis["deployment_stability_proven"])

    def test_primary_optimum_contradiction_is_not_labelled_verified(self):
        historical_features = np.stack((_row(1.0), _row(-2.0)))
        historical_targets = np.asarray([1.0, 0.0], dtype=np.float64)
        v5_features = np.stack((_row(1.0), _row(-1.0)))
        v5_targets = np.asarray([1.0, 0.0], dtype=np.float64)

        result = v.lexicographic_parameter_stability_v1(
            historical_features=historical_features,
            historical_targets=historical_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
            frozen_weight=np.zeros(64, dtype=np.float64),
            frozen_bias=0.0,
            threshold=0.5,
            primary_max_absolute_historical_logit_drift=3.0 * v.DECISION_MARGIN,
        )

        self.assertEqual(result["witness_claim"], "UNPROVEN_WITNESS_RESIDUAL_FAILED")
        self.assertFalse(result["primary_optimum_consistency_verified"])
        self.assertFalse(result["parameter_stability_witness_verified"])


if __name__ == "__main__":
    unittest.main()
