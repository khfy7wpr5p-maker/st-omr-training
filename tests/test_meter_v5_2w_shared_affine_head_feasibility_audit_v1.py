from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from st_omr_training import meter_v5_2w_shared_affine_head_feasibility_audit_v1 as v


def _row(first: float, second: float = 0.0) -> np.ndarray:
    value = np.zeros(64, dtype=np.float64)
    value[0] = first
    value[1] = second
    return value


class TestMeterV52WSharedAffineHeadFeasibilityAuditV1(unittest.TestCase):
    def test_exact_v5_2v_evidence_is_bound(self):
        self.assertEqual(
            v.V52V_IMPLEMENTATION_HEAD,
            "b1db7923e91cec534fcfd95afad7f8b4ef87607b",
        )
        self.assertEqual(
            v.V52V_REPORT_SHA256,
            "1ecc6b6600e0f01c1eeb4e8530d2184800dd470d2b66344c52a28a79d170bd3a",
        )
        self.assertEqual(
            v.V52V_EXECUTION_ENVELOPE_SHA256,
            "f8c87b5ecec00f5a4e2cbaf5f1f07bb599f85dac41af8b989686c6f33f03ca4d",
        )

    def test_stage_is_train_only_diagnostic_and_keeps_closed_surfaces_locked(self):
        boundary = v.safety_boundary()
        self.assertFalse(boundary["model_training"])
        self.assertFalse(boundary["autograd_grad_used"])
        self.assertFalse(boundary["backward"])
        self.assertEqual(boundary["optimizer_steps"], 0)
        self.assertTrue(boundary["diagnostic_linear_program_solve"])
        self.assertTrue(boundary["diagnostic_affine_witness_fit"])
        self.assertFalse(boundary["diagnostic_witness_persisted"])
        self.assertFalse(boundary["candidate_checkpoint_write"])
        self.assertFalse(boundary["classifier_fit_for_deployment"])
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

    def test_solver_contract_is_single_pinned_fail_closed_configuration(self):
        contract = v.solver_contract()
        self.assertEqual(contract["expected_library_version"], "1.18.0")
        self.assertEqual(contract["method"], "highs-ds")
        self.assertFalse(contract["presolve"])
        self.assertEqual(contract["fixed_runtime_decision_margin"], 1e-4)
        self.assertEqual(contract["witness_verification_tolerance"], 1e-7)
        self.assertFalse(contract["automatic_second_solver"])
        self.assertFalse(contract["solver_sweep"])
        self.assertFalse(contract["threshold_search"])
        self.assertFalse(contract["bias_search"])

    def test_fixed_runtime_and_free_affine_witnesses_are_verified(self):
        features = np.stack(
            (
                _row(2.0),
                _row(1.0),
                _row(-1.0),
                _row(-2.0),
            )
        )
        targets = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float64)

        fixed = v.fixed_runtime_feasibility_v1(
            features=features,
            targets=targets,
            fixed_bias=0.0,
            threshold=0.5,
        )
        free = v.free_affine_feasibility_v1(features=features, targets=targets)

        self.assertEqual(fixed["feasibility_claim"], "WITNESS_VERIFIED")
        self.assertTrue(fixed["feasible_witness_verified"])
        self.assertGreaterEqual(
            fixed["minimum_signed_decision_margin"],
            v.DECISION_MARGIN - v.WITNESS_TOLERANCE,
        )
        self.assertEqual(free["feasibility_claim"], "WITNESS_VERIFIED")
        self.assertTrue(free["feasible_witness_verified"])
        self.assertGreaterEqual(
            free["minimum_normalized_signed_margin"],
            1.0 - v.WITNESS_TOLERANCE,
        )
        self.assertNotIn("weight", fixed)
        self.assertNotIn("weight", free)
        self.assertNotIn("intercept", free)

    def test_free_intercept_can_succeed_when_frozen_runtime_intercept_cannot(self):
        features = np.stack((_row(-1.0), _row(-2.0)))
        targets = np.asarray([1.0, 0.0], dtype=np.float64)
        threshold = 1.0 / (1.0 + math.exp(-1.0))

        fixed = v.fixed_runtime_feasibility_v1(
            features=features,
            targets=targets,
            fixed_bias=0.0,
            threshold=threshold,
        )
        free = v.free_affine_feasibility_v1(features=features, targets=targets)
        diagnosis = v.joint_path_diagnosis_v1(
            fixed_runtime=fixed,
            free_affine=free,
        )

        self.assertEqual(
            fixed["feasibility_claim"],
            "SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF",
        )
        self.assertEqual(free["feasibility_claim"], "WITNESS_VERIFIED")
        self.assertEqual(
            diagnosis["status"],
            "FREE_AFFINE_ONLY_FEASIBLE_ON_TRAIN",
        )
        self.assertFalse(diagnosis["frozen_runtime_shared_head_feasibility_proven"])
        self.assertTrue(diagnosis["free_affine_shared_head_feasibility_proven"])
        self.assertFalse(diagnosis["repair_selected"])

    def test_nonseparable_surface_is_fail_closed(self):
        features = np.stack(
            (
                _row(1.0, 1.0),
                _row(-1.0, -1.0),
                _row(1.0, -1.0),
                _row(-1.0, 1.0),
            )
        )
        targets = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float64)

        fixed = v.fixed_runtime_feasibility_v1(
            features=features,
            targets=targets,
            fixed_bias=0.0,
            threshold=0.5,
        )
        free = v.free_affine_feasibility_v1(features=features, targets=targets)
        diagnosis = v.joint_path_diagnosis_v1(
            fixed_runtime=fixed,
            free_affine=free,
        )

        self.assertEqual(
            fixed["feasibility_claim"],
            "SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF",
        )
        self.assertEqual(
            free["feasibility_claim"],
            "SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF",
        )
        self.assertEqual(
            diagnosis["status"],
            "NO_SHARED_AFFINE_WITNESS_SOLVER_REPORTED_INFEASIBLE",
        )
        self.assertFalse(diagnosis["representation_failure_proven"])
        self.assertFalse(diagnosis["repair_selected"])

    def test_joint_surface_preserves_only_frozen_correct_historical_decisions(self):
        historical_features = np.stack(
            (
                _row(2.0),
                _row(-2.0),
                _row(-3.0),
            )
        )
        historical_targets = np.asarray([1.0, 0.0, 1.0], dtype=np.float64)
        v5_features = np.stack((_row(1.5), _row(-1.5)))
        v5_targets = np.asarray([1.0, 0.0], dtype=np.float64)
        frozen_weight = _row(1.0)

        result = v.shared_head_feasibility_metrics_v1(
            historical_features=historical_features,
            historical_targets=historical_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
            frozen_weight=frozen_weight,
            frozen_bias=0.0,
            threshold=0.5,
        )

        self.assertEqual(result["surface_counts"]["v5_all"], 2)
        self.assertEqual(result["surface_counts"]["historical_all"], 3)
        self.assertEqual(result["surface_counts"]["historical_frozen_correct"], 2)
        self.assertEqual(result["surface_counts"]["joint_constraints"], 4)
        self.assertEqual(
            result["fixed_runtime_feasibility"]["feasibility_claim"],
            "WITNESS_VERIFIED",
        )
        self.assertEqual(
            result["joint_path_diagnosis"]["status"],
            "FROZEN_RUNTIME_SHARED_HEAD_FEASIBLE_ON_TRAIN",
        )


if __name__ == "__main__":
    unittest.main()
