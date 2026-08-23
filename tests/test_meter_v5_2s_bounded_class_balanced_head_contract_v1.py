from __future__ import annotations

import inspect
import math
from pathlib import Path
import tempfile
import unittest

from st_omr_training import meter_v5_2s_bounded_class_balanced_head_contract_v1 as s


class TestMeterV52SBoundedClassBalancedHeadContractV1(unittest.TestCase):
    def _weights(self):
        frozen = [0.0] * 64
        frozen[0] = 4.0
        candidate = frozen.copy()
        return frozen, candidate

    def _groups(self):
        logits = {
            "v5_positive": [0.2, 1.0],
            "v5_negative": [-0.2, -1.0, -2.0],
            "historical_positive": [0.4, 1.4, 2.0],
            "historical_negative": [-0.4, -1.4],
        }
        targets = {
            "v5_positive": [1.0, 1.0],
            "v5_negative": [0.0, 0.0, 0.0],
            "historical_positive": [1.0, 1.0, 1.0],
            "historical_negative": [0.0, 0.0],
        }
        return logits, targets

    def test_stage_is_preregistered_but_training_disabled(self):
        safety = s.safety_boundary()
        self.assertTrue(safety["objective_preregistered"])
        self.assertFalse(safety["training_authorized"])
        self.assertFalse(safety["training_executed"])
        self.assertFalse(safety["autograd_grad_used"])
        self.assertFalse(safety["backward"])
        self.assertEqual(safety["optimizer_steps"], 0)
        self.assertFalse(safety["checkpoint_write"])
        self.assertTrue(safety["frozen_backbone"])
        self.assertTrue(safety["frozen_head_bias"])
        self.assertFalse(safety["historical_validation_opened"])
        self.assertFalse(safety["first30_opened"])
        self.assertFalse(safety["v5_validation_opened"])
        self.assertTrue(safety["final_holdout_locked"])
        self.assertTrue(safety["digit4_frozen"])
        self.assertFalse(s.training_entry_point_available())
        self.assertFalse(s.production_promotion_allowed())
        source = inspect.getsource(s)
        for forbidden in ("torch.optim", ".backward(", "torch.autograd", "torch.save"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_exact_v5_2r_evidence_is_bound(self):
        evidence = s.prerequisite_evidence_contract()
        self.assertEqual(
            evidence["v5_2r_implementation_head"],
            "85c0b0083792e8b9ec60ee632cfc7015e885d548",
        )
        self.assertEqual(
            evidence["v5_2r_report_sha256"],
            "2c374189c285232eb79c7a8ca331d9a53b60286b36dde18d5dec559b14f58dc7",
        )
        self.assertEqual(
            evidence["v5_2r_execution_envelope_sha256"],
            "6a80aa1536722720f6a3a85d93d363e356830eb6ff711d0b41c4af5c45080226",
        )
        self.assertFalse(evidence["historical_retention_examples_used_for_design"])

    def test_four_group_objective_has_fixed_equal_coefficients(self):
        contract = s.objective_contract()
        self.assertEqual(contract["group_weights"], {name: 0.25 for name in s.GROUPS})
        self.assertEqual(contract["group_weight_sum"], 1.0)
        self.assertEqual(contract["domain_weight_sum"], {"v5": 0.5, "historical": 0.5})
        self.assertEqual(contract["maximum_head_angle_degrees"], 15.0)
        self.assertIn("no-sweep", contract["selection_method"])

    def test_four_group_bce_is_invariant_to_group_duplication(self):
        logits, targets = self._groups()
        total1, losses1 = s.balanced_four_group_bce_v1(
            group_logits=logits, group_targets=targets
        )
        factors = {
            "v5_positive": 7,
            "v5_negative": 3,
            "historical_positive": 11,
            "historical_negative": 5,
        }
        logits2 = {name: value * factors[name] for name, value in logits.items()}
        targets2 = {name: value * factors[name] for name, value in targets.items()}
        total2, losses2 = s.balanced_four_group_bce_v1(
            group_logits=logits2, group_targets=targets2
        )
        self.assertAlmostEqual(total1, total2, places=12)
        for name in s.GROUPS:
            self.assertAlmostEqual(losses1[name], losses2[name], places=12)

    def test_proximal_scale_is_analytic_and_penalty_equals_initial_at_radius(self):
        frozen, _candidate = self._weights()
        initial = 0.75
        contract = s.derive_proximal_contract_v1(
            frozen_weight=frozen, initial_balanced_bce=initial
        )
        self.assertAlmostEqual(
            contract["trust_radius_over_frozen_l2"], math.sin(math.radians(15.0))
        )
        self.assertAlmostEqual(contract["penalty_at_trust_radius"], initial)
        self.assertAlmostEqual(
            contract["maximum_candidate_over_frozen_l2"],
            1.0 + math.sin(math.radians(15.0)),
        )
        boundary = frozen.copy()
        boundary[1] = contract["trust_radius_l2"]
        penalty = s.proximal_penalty_v1(
            candidate_weight=boundary,
            frozen_weight=frozen,
            proximal_lambda=contract["proximal_lambda"],
        )
        self.assertAlmostEqual(penalty, initial)
        with self.assertRaisesRegex(s.MeterV5_2SError, "finite and positive"):
            s.proximal_penalty_v1(
                candidate_weight=frozen,
                frozen_weight=frozen,
                proximal_lambda=-1.0,
            )

    def test_complete_objective_has_zero_proximal_penalty_at_frozen_head(self):
        logits, targets = self._groups()
        frozen, candidate = self._weights()
        balanced, _losses = s.balanced_four_group_bce_v1(
            group_logits=logits, group_targets=targets
        )
        total, balanced2, penalty, _group_losses, _proximal = (
            s.bounded_class_balanced_objective_v1(
                group_logits=logits,
                group_targets=targets,
                candidate_weight=candidate,
                frozen_weight=frozen,
                initial_balanced_bce=balanced,
            )
        )
        self.assertEqual(penalty, 0.0)
        self.assertEqual(total, balanced2)

    def test_geometry_gate_accepts_small_change_and_rejects_v5_2p_like_rotation(self):
        frozen, candidate = self._weights()
        candidate[1] = 0.25
        passed = s.geometry_evidence_v1(
            frozen_weight=frozen, candidate_weight=candidate
        )
        self.assertEqual(passed["gate"], "PASS")
        self.assertLess(passed["head_angle_change_degrees"], 15.0)

        boundary_angle = math.radians(15.0)
        tangent = frozen.copy()
        tangent[0] = 4.0 * math.cos(boundary_angle) ** 2
        tangent[1] = 4.0 * math.sin(boundary_angle) * math.cos(boundary_angle)
        boundary = s.geometry_evidence_v1(
            frozen_weight=frozen, candidate_weight=tangent
        )
        self.assertEqual(boundary["gate"], "PASS")
        self.assertAlmostEqual(boundary["head_angle_change_degrees"], 15.0)
        self.assertAlmostEqual(
            boundary["delta_over_frozen_l2"], math.sin(boundary_angle)
        )

        angle = math.radians(86.0)
        rotated = frozen.copy()
        rotated[0] = 4.0 * math.cos(angle)
        rotated[1] = 4.0 * math.sin(angle)
        held = s.geometry_evidence_v1(
            frozen_weight=frozen, candidate_weight=rotated
        )
        self.assertEqual(held["gate"], "HOLD")
        self.assertGreater(held["head_angle_change_degrees"], 15.0)

    def test_final_candidate_guard_requires_finite_non_increasing_and_bounded(self):
        frozen, candidate = self._weights()
        candidate[1] = 0.1
        result = s.verify_final_candidate_v1(
            frozen_weight=frozen,
            candidate_weight=candidate,
            initial_total_objective=1.0,
            final_total_objective=0.9,
        )
        self.assertEqual(result["gate"], "PASS")

        with self.assertRaisesRegex(s.MeterV5_2SError, "increased"):
            s.verify_final_candidate_v1(
                frozen_weight=frozen,
                candidate_weight=candidate,
                initial_total_objective=1.0,
                final_total_objective=1.1,
            )

        escaped = frozen.copy()
        escaped[1] = 4.0
        with self.assertRaisesRegex(s.MeterV5_2SError, "geometry bound"):
            s.verify_final_candidate_v1(
                frozen_weight=frozen,
                candidate_weight=escaped,
                initial_total_objective=1.0,
                final_total_objective=0.5,
            )

    def test_wrong_or_nonfinite_group_fails_closed(self):
        logits, targets = self._groups()
        targets["v5_positive"] = [0.0, 0.0]
        with self.assertRaisesRegex(s.MeterV5_2SError, "wrong class label"):
            s.balanced_four_group_bce_v1(
                group_logits=logits, group_targets=targets
            )

        logits, targets = self._groups()
        logits["historical_negative"][0] = float("nan")
        with self.assertRaisesRegex(s.MeterV5_2SError, "non-finite"):
            s.balanced_four_group_bce_v1(
                group_logits=logits, group_targets=targets
            )

    def test_exact_evidence_hash_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.json"
            envelope = root / "envelope.json"
            report.write_text("{}", encoding="utf-8")
            envelope.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(s.MeterV5_2SError, "report SHA256 mismatch"):
                s.verify_exact_v5_2r_evidence(
                    report_path=report, envelope_path=envelope
                )

    def test_solver_is_single_carried_forward_configuration(self):
        solver = s.solver_contract()
        self.assertFalse(solver["execution_authorized"])
        self.assertEqual(solver["optimizer_if_later_authorized"], "LBFGS")
        self.assertEqual(solver["max_iter"], 100)
        self.assertEqual(solver["max_eval"], 125)
        self.assertEqual(solver["line_search_fn"], "strong_wolfe")
        self.assertEqual(solver["candidate_selection"], "single-final-solver-state-no-sweep")
        self.assertFalse(solver["automatic_second_configuration"])

    def test_document_and_gate_order_are_explicit(self):
        path = Path("METER_V5_2S_BOUNDED_CLASS_BALANCED_HEAD_CONTRACT_V1.md")
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        self.assertIn("Each group receives exactly `0.25`", text)
        self.assertIn("lambda = 2 * L0 / R^2", text)
        self.assertIn("This contract stage contains no optimizer", text)
        self.assertEqual(
            s.gate_order(),
            (
                "numerical_integrity_and_geometry",
                "historical_retention_v3",
                "immutable_v5_first30_diagnostic",
                "separately_authorized_v5_validation",
                "separately_authorized_final_holdout",
            ),
        )


if __name__ == "__main__":
    unittest.main()
