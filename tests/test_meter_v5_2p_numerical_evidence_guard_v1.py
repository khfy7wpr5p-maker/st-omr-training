import unittest

from st_omr_training import meter_v5_2p_fixed_bias_head_repair_v1 as v52p
from st_omr_training import meter_v5_2p_numerical_evidence_guard_v1 as guard


class TestMeterV52PNumericalEvidenceGuardV1(unittest.TestCase):
    def test_evidence_layer_does_not_change_preregistered_contract(self):
        contract = guard.evidence_contract()
        self.assertFalse(contract["architecture_changed"])
        self.assertFalse(contract["objective_changed"])
        self.assertFalse(contract["solver_settings_changed"])
        self.assertFalse(contract["data_surfaces_changed"])
        self.assertFalse(contract["thresholds_changed"])
        self.assertFalse(contract["performance_gate_order_changed"])
        self.assertEqual(contract["v5_2p_objective_contract"], v52p.objective_contract())
        self.assertEqual(contract["v5_2p_solver_contract"], v52p.solver_contract())
        self.assertEqual(contract["v5_2p_performance_gate_order"], list(v52p.gate_order()))

    def test_gradient_tolerance_proves_convergence(self):
        evidence = guard._termination_evidence_v1(
            n_iter=7,
            func_evals=9,
            closure_evaluations=9,
            final_gradient_inf_norm=v52p.LBFGS_TOLERANCE_GRAD * 0.5,
            final_gradient_l2_norm=1e-9,
        )
        self.assertTrue(evidence["gradient_tolerance_met"])
        self.assertTrue(evidence["convergence_proven"])
        self.assertEqual(
            evidence["termination_evidence_class"],
            "PROVEN_FINAL_GRADIENT_TOLERANCE",
        )
        self.assertFalse(evidence["termination_reason_exposed_by_torch_lbfgs"])

    def test_unexposed_early_termination_is_not_invented_as_convergence(self):
        evidence = guard._termination_evidence_v1(
            n_iter=7,
            func_evals=9,
            closure_evaluations=9,
            final_gradient_inf_norm=v52p.LBFGS_TOLERANCE_GRAD * 10.0,
            final_gradient_l2_norm=1e-7,
        )
        self.assertFalse(evidence["gradient_tolerance_met"])
        self.assertTrue(evidence["terminated_before_limits"])
        self.assertFalse(evidence["convergence_proven"])
        self.assertEqual(
            evidence["termination_evidence_class"],
            "TERMINATED_BEFORE_LIMIT_REASON_NOT_EXPOSED_CONVERGENCE_NOT_PROVEN",
        )

    def test_max_iter_is_fail_closed(self):
        evidence = guard._termination_evidence_v1(
            n_iter=v52p.LBFGS_MAX_ITER,
            func_evals=100,
            closure_evaluations=100,
            final_gradient_inf_norm=1e-4,
            final_gradient_l2_norm=1e-3,
        )
        self.assertTrue(evidence["iteration_limit_reached"])
        self.assertFalse(evidence["convergence_proven"])
        self.assertEqual(
            evidence["termination_evidence_class"],
            "MAX_ITER_REACHED_CONVERGENCE_NOT_PROVEN",
        )

    def test_guard_requires_all_numerical_and_state_evidence(self):
        termination = {
            "convergence_proven": True,
        }
        specialist = {
            "trainable_parameter_count": 64,
            "only_head_weight_changed": True,
            "backbone_bit_identical": True,
            "head_bias_bit_identical": True,
            "threshold_unchanged": True,
            "solver_final_loss_finite": True,
            "solver_final_loss_not_above_initial": True,
            "float32_copy_back_bit_exact": True,
            "float32_copy_back_loss_finite": True,
            "float32_copy_back_loss_not_above_initial": True,
            "lbfgs_termination": termination,
        }
        decision = guard._guard_decision_v1({"2": specialist, "3": dict(specialist)})
        self.assertEqual(decision["gate"], "PASS")
        self.assertEqual(decision["reasons"], [])
        self.assertTrue(decision["historical_retention_authorized_after_separate_review"])

    def test_guard_holds_when_convergence_is_not_proven(self):
        specialist = {
            "trainable_parameter_count": 64,
            "only_head_weight_changed": True,
            "backbone_bit_identical": True,
            "head_bias_bit_identical": True,
            "threshold_unchanged": True,
            "solver_final_loss_finite": True,
            "solver_final_loss_not_above_initial": True,
            "float32_copy_back_bit_exact": True,
            "float32_copy_back_loss_finite": True,
            "float32_copy_back_loss_not_above_initial": True,
            "lbfgs_termination": {"convergence_proven": False},
        }
        decision = guard._guard_decision_v1({"2": specialist, "3": dict(specialist)})
        self.assertEqual(decision["gate"], "HOLD")
        self.assertIn("2-AI_LBFGS_CONVERGENCE_NOT_PROVEN", decision["reasons"])
        self.assertIn("3-AI_LBFGS_CONVERGENCE_NOT_PROVEN", decision["reasons"])
        self.assertFalse(decision["historical_retention_authorized_after_separate_review"])

    def test_evidence_module_does_not_run_performance_gates(self):
        self.assertFalse(guard.historical_retention_executed_by_this_module())
        self.assertFalse(guard.validation_opened_by_this_module())
        self.assertTrue(guard.final_holdout_locked())
        self.assertFalse(guard.production_promotion_allowed())


if __name__ == "__main__":
    unittest.main()
