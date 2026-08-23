import copy
import math
import unittest

from st_omr_training import meter_v5_2p_fixed_bias_head_repair_v1 as v52p
from st_omr_training import meter_v5_2p_numerical_evidence_guard_v1 as guard


class TestMeterV52PNumericalEvidenceGuardV1(unittest.TestCase):
    def _valid_termination(self, *, proven=False):
        return guard._termination_evidence_v1(
            n_iter=7,
            func_evals=9,
            closure_evaluations=9,
            final_gradient_inf_norm=(
                v52p.LBFGS_TOLERANCE_GRAD * 0.5 if proven else 1e-6
            ),
            final_gradient_l2_norm=(1e-9 if proven else 2e-6),
        )

    def _valid_specialist(self, *, termination=None):
        return {
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
            "lbfgs_termination": termination or self._valid_termination(proven=True),
        }

    def test_evidence_layer_does_not_change_preregistered_contract(self):
        contract = guard.evidence_contract()
        self.assertFalse(contract["architecture_changed"])
        self.assertFalse(contract["objective_changed"])
        self.assertFalse(contract["solver_settings_changed"])
        self.assertFalse(contract["data_surfaces_changed"])
        self.assertFalse(contract["thresholds_changed"])
        self.assertFalse(contract["performance_gate_order_changed"])
        self.assertTrue(contract["numerical_integrity_is_safety_gate_only"])
        self.assertFalse(contract["convergence_evidence_is_performance_gate"])
        self.assertFalse(contract["convergence_unproven_creates_integrity_hold"])
        self.assertEqual(contract["v5_2p_objective_contract"], v52p.objective_contract())
        self.assertEqual(contract["v5_2p_solver_contract"], v52p.solver_contract())
        self.assertEqual(contract["v5_2p_performance_gate_order"], list(v52p.gate_order()))

    def test_gradient_tolerance_proves_convergence(self):
        evidence = self._valid_termination(proven=True)
        self.assertTrue(evidence["final_gradient_finite"])
        self.assertTrue(evidence["gradient_tolerance_met"])
        self.assertTrue(evidence["convergence_proven"])
        self.assertEqual(evidence["convergence_claim"], "PROVEN")
        self.assertEqual(
            evidence["termination_evidence_class"],
            "PROVEN_FINAL_GRADIENT_TOLERANCE",
        )
        self.assertFalse(evidence["termination_reason_exposed"])
        self.assertFalse(evidence["termination_reason_exposed_by_torch_lbfgs"])

    def test_unexposed_early_termination_is_not_invented_as_convergence(self):
        evidence = self._valid_termination(proven=False)
        self.assertTrue(evidence["final_gradient_finite"])
        self.assertFalse(evidence["gradient_tolerance_met"])
        self.assertTrue(evidence["terminated_before_limits"])
        self.assertFalse(evidence["convergence_proven"])
        self.assertEqual(evidence["convergence_claim"], "UNPROVEN")
        self.assertFalse(evidence["termination_reason_exposed"])
        self.assertIn("NOT_EXPOSED", evidence["termination_evidence_class"])

    def test_max_iter_reports_unproven_without_becoming_integrity_hold(self):
        termination = guard._termination_evidence_v1(
            n_iter=v52p.LBFGS_MAX_ITER,
            func_evals=100,
            closure_evaluations=100,
            final_gradient_inf_norm=1e-4,
            final_gradient_l2_norm=1e-3,
        )
        self.assertTrue(termination["iteration_limit_reached"])
        self.assertFalse(termination["convergence_proven"])
        self.assertEqual(termination["convergence_claim"], "UNPROVEN")
        specialist = self._valid_specialist(termination=termination)
        decision = guard._guard_decision_v1({"2": specialist, "3": dict(specialist)})
        self.assertEqual(decision["numerical_integrity_gate"]["gate"], "PASS")
        self.assertEqual(decision["numerical_integrity_gate"]["reasons"], [])
        self.assertTrue(decision["historical_retention_authorized_after_separate_review"])

    def test_convergence_unproven_does_not_create_hold(self):
        termination = self._valid_termination(proven=False)
        specialist = self._valid_specialist(termination=termination)
        decision = guard._guard_decision_v1({"2": specialist, "3": dict(specialist)})

        self.assertEqual(decision["numerical_integrity_gate"]["gate"], "PASS")
        self.assertEqual(decision["numerical_integrity_gate"]["reasons"], [])
        self.assertTrue(decision["historical_retention_authorized_after_separate_review"])
        self.assertFalse(decision["historical_preservation_claimed"])
        for digit in ("2", "3"):
            convergence = decision["convergence_evidence"][digit]
            self.assertTrue(convergence["evidence_present"])
            self.assertEqual(convergence["convergence_claim"], "UNPROVEN")
            self.assertFalse(convergence["convergence_proven"])
            self.assertFalse(convergence["termination_reason_exposed"])

    def test_integrity_gate_holds_only_real_integrity_failures(self):
        cases = {
            "final_gradient_nonfinite": (
                "2-AI_FINAL_GRADIENT_NONFINITE",
                lambda item: item["lbfgs_termination"].update(
                    guard._termination_evidence_v1(
                        n_iter=1,
                        func_evals=1,
                        closure_evaluations=1,
                        final_gradient_inf_norm=math.inf,
                        final_gradient_l2_norm=math.inf,
                    )
                ),
            ),
            "final_loss_nonfinite": (
                "2-AI_SOLVER_FINAL_LOSS_NONFINITE",
                lambda item: item.__setitem__("solver_final_loss_finite", False),
            ),
            "final_loss_increased": (
                "2-AI_SOLVER_FINAL_LOSS_INCREASED",
                lambda item: item.__setitem__("solver_final_loss_not_above_initial", False),
            ),
            "copy_nonfinite": (
                "2-AI_FLOAT32_COPY_BACK_LOSS_NONFINITE",
                lambda item: item.__setitem__("float32_copy_back_loss_finite", False),
            ),
            "copy_increased": (
                "2-AI_FLOAT32_COPY_BACK_LOSS_INCREASED",
                lambda item: item.__setitem__("float32_copy_back_loss_not_above_initial", False),
            ),
            "copy_not_exact": (
                "2-AI_FLOAT32_COPY_BACK_NOT_EXACT",
                lambda item: item.__setitem__("float32_copy_back_bit_exact", False),
            ),
            "backbone_changed": (
                "2-AI_BACKBONE_NOT_BIT_IDENTICAL",
                lambda item: item.__setitem__("backbone_bit_identical", False),
            ),
            "bias_changed": (
                "2-AI_HEAD_BIAS_NOT_BIT_IDENTICAL",
                lambda item: item.__setitem__("head_bias_bit_identical", False),
            ),
            "threshold_changed": (
                "2-AI_THRESHOLD_CHANGED",
                lambda item: item.__setitem__("threshold_unchanged", False),
            ),
            "illegal_state": (
                "2-AI_ILLEGAL_STATE_MUTATION",
                lambda item: item.__setitem__("only_head_weight_changed", False),
            ),
            "wrong_parameter_count": (
                "2-AI_TRAINABLE_PARAMETER_COUNT_CHANGED",
                lambda item: item.__setitem__("trainable_parameter_count", 63),
            ),
        }
        for name, (reason, mutate) in cases.items():
            with self.subTest(name=name):
                two = self._valid_specialist()
                three = self._valid_specialist()
                mutate(two)
                decision = guard._guard_decision_v1({"2": two, "3": three})
                self.assertEqual(decision["numerical_integrity_gate"]["gate"], "HOLD")
                self.assertIn(reason, decision["numerical_integrity_gate"]["reasons"])
                self.assertFalse(decision["historical_retention_authorized_after_separate_review"])

    def test_missing_evidence_and_capture_count_fail_closed(self):
        two = self._valid_specialist()
        three = self._valid_specialist()
        del two["float32_copy_back_bit_exact"]
        del three["lbfgs_termination"]["func_evals"]
        decision = guard._guard_decision_v1(
            {"2": two, "3": three},
            observed_lbfgs_solves=1,
        )
        reasons = decision["numerical_integrity_gate"]["reasons"]
        self.assertEqual(decision["numerical_integrity_gate"]["gate"], "HOLD")
        self.assertIn("LBFGS_CAPTURE_COUNT_MISMATCH:1", reasons)
        self.assertTrue(any(reason.startswith("2-AI_REQUIRED_EVIDENCE_MISSING:") for reason in reasons))
        self.assertTrue(
            any(reason.startswith("3-AI_REQUIRED_CONVERGENCE_EVIDENCE_MISSING:") for reason in reasons)
        )

    def test_synthetic_lbfgs_capture_and_weight_evidence_integration(self):
        torch, nn = __import__("torch"), __import__("torch").nn

        original_step, captures = guard._capture_lbfgs_steps(torch)
        try:
            weight = nn.Parameter(torch.full((64,), 0.01, dtype=torch.float64))
            optimizer = torch.optim.LBFGS([weight], lr=0.1, max_iter=1, max_eval=1)

            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = torch.sum(weight * weight)
                loss.backward()
                return loss

            optimizer.step(closure)
            self.assertEqual(len(captures), 1)
            self.assertEqual(tuple(captures[0]["final_weight_float64"].shape), (64,))
        finally:
            guard._restore_lbfgs_step(torch, original_step)
        self.assertIs(torch.optim.LBFGS.step, original_step)

        class TinySpecialist(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(nn.Linear(2, 2))
                self.head = nn.Linear(64, 1)

        torch.manual_seed(22023)
        frozen = TinySpecialist()
        candidate = copy.deepcopy(frozen)
        final_weight64 = frozen.head.weight.detach().cpu().reshape(-1).to(torch.float64) + 1e-4
        with torch.no_grad():
            candidate.head.weight.copy_(final_weight64.to(torch.float32).reshape_as(candidate.head.weight))

        v5_features = torch.zeros((4, 64), dtype=torch.float32)
        hist_features = torch.zeros((4, 64), dtype=torch.float32)
        v5_features[:, 0] = torch.tensor([-1.0, 1.0, -0.5, 0.5])
        hist_features[:, 0] = torch.tensor([-0.75, 0.75, -0.25, 0.25])
        v5_targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
        hist_targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
        bias = float(frozen.head.bias.detach().cpu().reshape(-1)[0].item())
        with torch.no_grad():
            total, _v5, _hist = v52p._balanced_domain_bce_v1(
                v5_logits=v5_features.to(torch.float64) @ final_weight64 + bias,
                v5_targets=v5_targets.to(torch.float64),
                historical_logits=hist_features.to(torch.float64) @ final_weight64 + bias,
                historical_targets=hist_targets.to(torch.float64),
            )
        final_loss = float(total.item())
        captured = {
            "final_weight_float64": final_weight64,
            "n_iter": 3,
            "func_evals": 5,
            "optimizer_state_keys": ["func_evals", "n_iter"],
        }
        fit = {
            "initial_total_loss": final_loss + 1.0,
            "final_total_loss": final_loss,
            "closure_evaluations": 5,
        }

        evidence = guard._evaluate_weight_state_v1(
            torch=torch,
            digit="2",
            captured=captured,
            fit=fit,
            frozen_model=frozen,
            candidate_model=candidate,
            v5_features=v5_features,
            v5_targets=v5_targets,
            historical_features=hist_features,
            historical_targets=hist_targets,
        )
        self.assertEqual(evidence["trainable_parameter_count"], 64)
        self.assertTrue(evidence["float32_copy_back_bit_exact"])
        self.assertTrue(evidence["backbone_bit_identical"])
        self.assertTrue(evidence["head_bias_bit_identical"])
        self.assertTrue(evidence["lbfgs_termination"]["final_gradient_finite"])
        self.assertTrue(math.isfinite(evidence["lbfgs_termination"]["final_gradient_inf_norm"]))
        self.assertTrue(math.isfinite(evidence["lbfgs_termination"]["final_gradient_l2_norm"]))

        with self.subTest("loss increase"):
            bad_fit = dict(fit)
            bad_fit["initial_total_loss"] = final_loss - 1e-3
            with self.assertRaises(guard.MeterV5_2PNumericalEvidenceError):
                guard._evaluate_weight_state_v1(
                    torch=torch,
                    digit="2",
                    captured=captured,
                    fit=bad_fit,
                    frozen_model=frozen,
                    candidate_model=candidate,
                    v5_features=v5_features,
                    v5_targets=v5_targets,
                    historical_features=hist_features,
                    historical_targets=hist_targets,
                )

        with self.subTest("non-finite loss evidence"):
            bad_fit = dict(fit)
            bad_fit["final_total_loss"] = math.nan
            with self.assertRaises(guard.MeterV5_2PNumericalEvidenceError):
                guard._evaluate_weight_state_v1(
                    torch=torch,
                    digit="2",
                    captured=captured,
                    fit=bad_fit,
                    frozen_model=frozen,
                    candidate_model=candidate,
                    v5_features=v5_features,
                    v5_targets=v5_targets,
                    historical_features=hist_features,
                    historical_targets=hist_targets,
                )

        with self.subTest("copy-back mismatch"):
            bad_candidate = copy.deepcopy(candidate)
            with torch.no_grad():
                bad_candidate.head.weight[0, 0] += 1e-3
            with self.assertRaises(guard.MeterV5_2PNumericalEvidenceError):
                guard._evaluate_weight_state_v1(
                    torch=torch,
                    digit="2",
                    captured=captured,
                    fit=fit,
                    frozen_model=frozen,
                    candidate_model=bad_candidate,
                    v5_features=v5_features,
                    v5_targets=v5_targets,
                    historical_features=hist_features,
                    historical_targets=hist_targets,
                )

        for mutation in ("backbone", "bias"):
            with self.subTest(mutation=mutation):
                bad_candidate = copy.deepcopy(candidate)
                with torch.no_grad():
                    if mutation == "backbone":
                        bad_candidate.features[0].weight[0, 0] += 1e-3
                    else:
                        bad_candidate.head.bias[0] += 1e-3
                with self.assertRaises(guard.MeterV5_2PNumericalEvidenceError):
                    guard._evaluate_weight_state_v1(
                        torch=torch,
                        digit="2",
                        captured=captured,
                        fit=fit,
                        frozen_model=frozen,
                        candidate_model=bad_candidate,
                        v5_features=v5_features,
                        v5_targets=v5_targets,
                        historical_features=hist_features,
                        historical_targets=hist_targets,
                    )

    def test_evidence_module_does_not_run_performance_gates(self):
        self.assertFalse(guard.historical_retention_executed_by_this_module())
        self.assertFalse(guard.validation_opened_by_this_module())
        self.assertTrue(guard.final_holdout_locked())
        self.assertFalse(guard.production_promotion_allowed())


if __name__ == "__main__":
    unittest.main()
