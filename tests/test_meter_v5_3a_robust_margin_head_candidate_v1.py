from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from st_omr_training import meter_v5_3a_robust_margin_head_candidate_v1 as v


def _row(first: float, second: float = 0.0) -> np.ndarray:
    value = np.zeros(64, dtype=np.float64)
    value[0] = first
    value[1] = second
    return value


class TestMeterV53ARobustMarginHeadCandidateV1(unittest.TestCase):
    def test_exact_v5_2z_execution_is_bound(self):
        self.assertEqual(
            v.V52Z_IMPLEMENTATION_HEAD,
            "040e1d80fcbb09f6cac7b43e15fd34567c3f7dad",
        )
        self.assertEqual(
            v.V52Z_REPORT_SHA256,
            "39fd82009f1bbef66877d0e65ad9719f7ecff9adc67f2c6d1a6a6e1a163ab8e4",
        )
        self.assertEqual(
            v.V52Z_EXECUTION_ENVELOPE_SHA256,
            "fa3adc4f96fcf1d3109b43750b0958a3267fa57075a9c6ff061ad30b42864e12",
        )

    def test_contract_is_one_fixed_lexicographic_fit(self):
        contract = v.solver_contract()
        self.assertEqual(contract["expected_library_version"], "1.18.0")
        self.assertEqual(contract["method"], "highs-ds")
        self.assertFalse(contract["presolve"])
        self.assertEqual(contract["robust_decision_margin"], 0.25)
        self.assertEqual(contract["solver_margin_buffer"], 1e-4)
        self.assertEqual(
            contract["primary_objective"],
            "minimum_total_absolute_delta_weight_l1",
        )
        self.assertEqual(
            contract["secondary_objective"],
            "minimum_max_absolute_delta_weight_linf",
        )
        self.assertTrue(contract["primary_objective_fixed_before_secondary"])
        self.assertFalse(contract["weight_l2_minimized"])
        self.assertFalse(contract["historical_logit_drift_optimized"])
        self.assertFalse(contract["automatic_second_configuration"])
        self.assertFalse(contract["solver_sweep"])
        self.assertFalse(contract["threshold_search"])
        self.assertFalse(contract["bias_search"])

    def test_candidate_stage_keeps_all_closed_surfaces_locked(self):
        boundary = v.safety_boundary()
        self.assertTrue(boundary["linear_head_candidate_fit_authorized"])
        self.assertTrue(boundary["candidate_checkpoint_write_authorized"])
        self.assertEqual(boundary["candidate_parameter_surface"], "head.weight-only-64")
        self.assertTrue(boundary["frozen_backbone"])
        self.assertTrue(boundary["frozen_head_bias"])
        self.assertFalse(boundary["autograd_grad_used"])
        self.assertFalse(boundary["backward"])
        self.assertEqual(boundary["optimizer_steps"], 0)
        self.assertFalse(boundary["runtime_threshold_tuning"])
        self.assertFalse(boundary["historical_validation_opened"])
        self.assertFalse(boundary["first30_opened"])
        self.assertFalse(boundary["v5_validation_opened"])
        self.assertTrue(boundary["final_holdout_locked"])
        self.assertTrue(boundary["digit4_frozen"])
        self.assertFalse(boundary["production_promotion"])

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

    def test_known_robust_minimum_is_verified_in_two_lexicographic_stages(self):
        historical_features = np.stack((_row(1.0), _row(-1.0)))
        historical_targets = np.asarray([1.0, 0.0], dtype=np.float64)
        v5_features = np.stack((_row(1.0), _row(-1.0)))
        v5_targets = np.asarray([1.0, 0.0], dtype=np.float64)

        result, candidate = v.solve_robust_margin_minimum_total_change_v1(
            historical_features=historical_features,
            historical_targets=historical_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
            frozen_weight=np.zeros(64, dtype=np.float64),
            frozen_bias=0.0,
            threshold=0.5,
        )

        expected = v.ROBUST_DECISION_MARGIN + v.SOLVER_MARGIN_BUFFER
        self.assertEqual(result["candidate_claim"], "CANDIDATE_WITNESS_VERIFIED")
        self.assertEqual(
            result["primary_optimality_claim"],
            "SOLVER_REPORTED_OPTIMAL_NOT_FORMAL_PROOF",
        )
        self.assertEqual(
            result["secondary_optimality_claim"],
            "SOLVER_REPORTED_OPTIMAL_NOT_FORMAL_PROOF",
        )
        self.assertTrue(result["robust_candidate_witness_verified"])
        self.assertAlmostEqual(result["minimum_delta_weight_l1"], expected, places=7)
        self.assertAlmostEqual(result["minimum_delta_weight_linf"], expected, places=7)
        self.assertGreaterEqual(
            result["minimum_v5_signed_decision_margin"],
            v.ROBUST_DECISION_MARGIN - v.WITNESS_TOLERANCE,
        )
        self.assertEqual(result["v5_constraint_violations"], 0)
        self.assertEqual(result["v5_solver_margin_constraint_violations"], 0)
        self.assertEqual(result["historical_margin_constraint_violations"], 0)
        self.assertEqual(
            result["historical_solver_margin_constraint_violations"], 0
        )
        self.assertEqual(result["primary_decision_constraint_violations"], 0)
        self.assertEqual(result["primary_auxiliary_bound_violations"], 0)
        self.assertEqual(result["primary_l1_cap_violations"], 0)
        self.assertEqual(result["parameter_bound_violations"], 0)
        self.assertTrue(result["functional_delta_identity_verified"])
        self.assertEqual(result["historical_transition_counts"]["correct_to_wrong"], 0)
        self.assertEqual(result["diagnostic_v5_train_metrics"]["f1"], 1.0)
        self.assertEqual(candidate.shape, (64,))
        self.assertTrue(np.isfinite(candidate).all())
        self.assertFalse(result["candidate_weight_values_emitted"])
        for forbidden_key in ("weight", "delta_weight", "candidate_weight"):
            self.assertNotIn(forbidden_key, result)

    def test_inconsistent_v5_constraints_fail_closed_without_candidate(self):
        historical_features = np.stack((_row(1.0), _row(-1.0)))
        historical_targets = np.asarray([1.0, 0.0], dtype=np.float64)
        v5_features = np.stack((_row(1.0), _row(1.0)))
        v5_targets = np.asarray([1.0, 0.0], dtype=np.float64)

        result, candidate = v.solve_robust_margin_minimum_total_change_v1(
            historical_features=historical_features,
            historical_targets=historical_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
            frozen_weight=_row(1.0),
            frozen_bias=0.0,
            threshold=0.5,
        )

        self.assertIsNone(candidate)
        self.assertEqual(
            result["candidate_claim"],
            "PRIMARY_SOLVER_REPORTED_INFEASIBLE_NOT_FORMAL_PROOF",
        )
        self.assertFalse(result["robust_candidate_witness_verified"])
        self.assertFalse(v.path_diagnosis_v1(result)["repair_candidate_selected"])

    def test_float32_copy_gate_checks_operational_margin(self):
        historical_features = np.stack((_row(1.0), _row(-1.0)))
        historical_targets = np.asarray([1.0, 0.0], dtype=np.float64)
        v5_features = np.stack((_row(1.0), _row(-1.0)))
        v5_targets = np.asarray([1.0, 0.0], dtype=np.float64)
        candidate = _row(v.ROBUST_DECISION_MARGIN + v.SOLVER_MARGIN_BUFFER)

        passed = v._verify_float32_copy_v1(
            candidate_weight=candidate,
            frozen_weight=np.zeros(64, dtype=np.float64),
            frozen_bias=0.0,
            threshold=0.5,
            historical_features=historical_features,
            historical_targets=historical_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
        )
        held = v._verify_float32_copy_v1(
            candidate_weight=_row(0.2),
            frozen_weight=np.zeros(64, dtype=np.float64),
            frozen_bias=0.0,
            threshold=0.5,
            historical_features=historical_features,
            historical_targets=historical_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
        )

        self.assertEqual(passed["gate"], "PASS")
        self.assertEqual(passed["v5_margin_violations"], 0)
        self.assertEqual(held["gate"], "HOLD")
        self.assertGreater(held["v5_margin_violations"], 0)

        try:
            torch, _nn = v.v52b._import_torch()
        except Exception:
            self.skipTest("PyTorch runtime is unavailable")
        model = torch.nn.Module()
        model.head = torch.nn.Linear(64, 1)
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
        frozen_state = {
            "head.weight": model.head.weight.detach().clone(),
            "head.bias": model.head.bias.detach().clone(),
        }
        with torch.no_grad():
            model.head.weight.copy_(torch.as_tensor(candidate).reshape(1, 64))
        runtime = v._verify_runtime_torch_copy_v1(
            model=model,
            frozen_state=frozen_state,
            threshold=0.5,
            historical_features=torch.as_tensor(
                historical_features, dtype=torch.float32
            ),
            historical_targets=torch.as_tensor(historical_targets),
            v5_features=torch.as_tensor(v5_features, dtype=torch.float32),
            v5_targets=torch.as_tensor(v5_targets),
        )
        self.assertEqual(runtime["gate"], "PASS")
        self.assertTrue(runtime["actual_runtime_float32_tensor_path"])

    def test_execution_requires_token_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "annotations"
            annotations.mkdir()
            with self.assertRaisesRegex(v.MeterV5_3AError, "approval token"):
                v.fit_robust_margin_head_candidates_v1(
                    root,
                    m4a_root=root,
                    d10_root=root,
                    digit2_frozen=root / "d2.pt",
                    digit3_frozen=root / "d3.pt",
                    v5_2z_report=root / "z.json",
                    v5_2z_envelope=root / "e.json",
                    confirmation="WRONG",
                )
            (annotations / v.REPORT_NAME).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(v.MeterV5_3AError, "overwrite/rerun"):
                v.fit_robust_margin_head_candidates_v1(
                    root,
                    m4a_root=root,
                    d10_root=root,
                    digit2_frozen=root / "d2.pt",
                    digit3_frozen=root / "d3.pt",
                    v5_2z_report=root / "z.json",
                    v5_2z_envelope=root / "e.json",
                    confirmation=v.APPROVAL_TOKEN,
                )


if __name__ == "__main__":
    unittest.main()
