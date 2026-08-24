from __future__ import annotations

import inspect
import unittest

import numpy as np

from st_omr_training import meter_v5_2x_minimum_functional_logit_drift_audit_v1 as v


def _row(first: float) -> np.ndarray:
    value = np.zeros(64, dtype=np.float64)
    value[0] = first
    return value


class TestMeterV52XMinimumFunctionalLogitDriftAuditV1(unittest.TestCase):
    def test_exact_v5_2w_execution_is_bound(self):
        self.assertEqual(
            v.V52W_IMPLEMENTATION_HEAD,
            "bdd82204182e3d5043a64907de7e0f0394089a20",
        )
        self.assertEqual(
            v.V52W_REPORT_SHA256,
            "0fdcff6a9114eec08a2f3c512de1336cf4af96be4ab82cf168d14fa2e77f4095",
        )
        self.assertEqual(
            v.V52W_EXECUTION_ENVELOPE_SHA256,
            "4b86e81547d9d22aee60353e57eac7d28fd9ab8ed6988f7ec629241c6a723d54",
        )

    def test_stage_is_train_only_diagnostic_and_keeps_all_gates_closed(self):
        boundary = v.safety_boundary()
        self.assertFalse(boundary["model_training"])
        self.assertFalse(boundary["autograd_grad_used"])
        self.assertFalse(boundary["backward"])
        self.assertEqual(boundary["optimizer_steps"], 0)
        self.assertTrue(boundary["diagnostic_linear_program_solve"])
        self.assertTrue(boundary["diagnostic_witness_fit"])
        self.assertFalse(boundary["diagnostic_witness_persisted"])
        self.assertFalse(boundary["model_parameter_mutation"])
        self.assertFalse(boundary["candidate_checkpoint_write"])
        self.assertFalse(boundary["threshold_tuning"])
        self.assertFalse(boundary["bias_selection"])
        self.assertFalse(boundary["historical_validation_opened"])
        self.assertFalse(boundary["first30_opened"])
        self.assertFalse(boundary["v5_validation_opened"])
        self.assertTrue(boundary["final_holdout_locked"])
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

    def test_solver_contract_minimizes_one_functional_quantity_only(self):
        contract = v.solver_contract()
        self.assertEqual(contract["expected_library_version"], "1.18.0")
        self.assertEqual(contract["method"], "highs-ds")
        self.assertFalse(contract["presolve"])
        self.assertEqual(contract["decision_margin"], 1e-4)
        self.assertEqual(contract["objective"], "minimum_max_absolute_historical_train_logit_drift")
        self.assertTrue(contract["historical_drift_objective_minimized"])
        self.assertFalse(contract["weight_norm_minimized"])
        self.assertFalse(contract["automatic_second_solver"])
        self.assertFalse(contract["solver_sweep"])
        self.assertFalse(contract["threshold_search"])
        self.assertFalse(contract["bias_search"])

    def test_known_one_dimensional_minimum_is_verified_without_emitting_witness(self):
        # Frozen w=0 predicts both historical rows positive.  Only the positive
        # row is frozen-correct.  V5 requires w >= margin; the x=-2 historical
        # row makes the minimum max absolute drift exactly 2 * margin.
        historical_features = np.stack((_row(1.0), _row(-2.0)))
        historical_targets = np.asarray([1.0, 0.0], dtype=np.float64)
        v5_features = np.stack((_row(1.0), _row(-1.0)))
        v5_targets = np.asarray([1.0, 0.0], dtype=np.float64)

        result = v.minimum_functional_logit_drift_v1(
            historical_features=historical_features,
            historical_targets=historical_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
            frozen_weight=np.zeros(64, dtype=np.float64),
            frozen_bias=0.0,
            threshold=0.5,
        )

        self.assertEqual(result["witness_claim"], "WITNESS_VERIFIED")
        self.assertEqual(
            result["optimality_claim"],
            "SOLVER_REPORTED_OPTIMAL_NOT_FORMAL_PROOF",
        )
        self.assertTrue(result["functional_drift_witness_verified"])
        self.assertAlmostEqual(
            result["minimum_max_absolute_historical_logit_drift"],
            2.0 * v.DECISION_MARGIN,
            places=8,
        )
        self.assertEqual(result["v5_constraint_violations"], 0)
        self.assertEqual(result["historical_retention_constraint_violations"], 0)
        self.assertEqual(result["historical_drift_bound_violations"], 0)
        self.assertLessEqual(
            result["objective_recomputation_absolute_error"],
            v.WITNESS_TOLERANCE,
        )
        self.assertLessEqual(
            result["solver_objective_identity_absolute_error"],
            v.WITNESS_TOLERANCE,
        )
        self.assertEqual(
            result["historical_transition_counts"]["correct_to_wrong"],
            0,
        )
        self.assertEqual(
            result["v5_transition_counts"]["wrong_to_correct"],
            1,
        )
        self.assertTrue(result["functional_delta_identity_verified"])
        self.assertFalse(result["weight_norm_minimized"])
        self.assertTrue(result["weight_norm_not_minimal"])
        self.assertFalse(result["witness_values_emitted"])
        self.assertFalse(result["witness_persisted"])
        diagnosis = v.path_diagnosis_v1(result)
        self.assertTrue(
            diagnosis["v5_2w_frozen_runtime_shared_head_feasibility_bound"]
        )
        self.assertTrue(
            diagnosis["minimum_functional_drift_witness_verified_on_train"]
        )
        self.assertFalse(diagnosis["generalization_proven"])
        for forbidden_key in ("weight", "delta_weight", "witness", "candidate_weight"):
            self.assertNotIn(forbidden_key, result)

    def test_inconsistent_v5_constraints_are_fail_closed(self):
        historical_features = np.stack((_row(1.0), _row(-1.0)))
        historical_targets = np.asarray([1.0, 0.0], dtype=np.float64)
        v5_features = np.stack((_row(1.0), _row(1.0)))
        v5_targets = np.asarray([1.0, 0.0], dtype=np.float64)

        result = v.minimum_functional_logit_drift_v1(
            historical_features=historical_features,
            historical_targets=historical_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
            frozen_weight=_row(1.0),
            frozen_bias=0.0,
            threshold=0.5,
        )

        self.assertEqual(
            result["witness_claim"],
            "SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF",
        )
        self.assertEqual(
            result["optimality_claim"],
            "OPTIMALITY_NOT_PROVEN",
        )
        self.assertFalse(result["functional_drift_witness_verified"])
        self.assertFalse(result["repair_selected"])
        diagnosis = v.path_diagnosis_v1(result)
        self.assertEqual(
            diagnosis["status"],
            "EVIDENCE_CONFLICT_V5_2W_FEASIBLE_BUT_V5_2X_SOLVER_REPORTED_INFEASIBLE",
        )
        self.assertTrue(
            diagnosis["v5_2w_frozen_runtime_shared_head_feasibility_bound"]
        )
        self.assertFalse(
            diagnosis["minimum_functional_drift_witness_verified_on_train"]
        )


if __name__ == "__main__":
    unittest.main()
