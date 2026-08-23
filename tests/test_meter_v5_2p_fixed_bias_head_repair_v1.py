from __future__ import annotations

import math
import unittest
from pathlib import Path

from st_omr_training import meter_v5_2m_retention_contract_v3 as ret_v3
from st_omr_training import meter_v5_2p_fixed_bias_head_repair_v1 as repair


class TestMeterV52PFixedBiasHeadRepairV1(unittest.TestCase):
    def _torch(self):
        torch, _nn = repair.v52b._import_torch()
        return torch

    def test_contract_is_narrow_and_fail_closed(self) -> None:
        safety = repair.safety_boundary()
        self.assertTrue(safety["single_fixed_repair_authorized"])
        self.assertFalse(safety["automatic_second_configuration"])
        self.assertEqual(safety["trainable_surface"], "head.weight-only-64-parameters")
        self.assertTrue(safety["frozen_backbone"])
        self.assertTrue(safety["frozen_head_bias"])
        self.assertFalse(safety["runtime_threshold_tuning"])
        self.assertFalse(safety["alternative_threshold_evaluated"])
        self.assertFalse(safety["new_bbox"])
        self.assertFalse(safety["new_crop_geometry"])
        self.assertFalse(safety["new_spatial_heuristic"])
        self.assertFalse(safety["reserve_v5_train_opened"])
        self.assertFalse(safety["v5_validation_opened"])
        self.assertTrue(safety["final_holdout_locked"])
        self.assertTrue(safety["digit4_frozen"])
        self.assertFalse(safety["runtime_domain_routing"])
        self.assertFalse(safety["production_promotion"])
        self.assertEqual(
            repair.gate_order(),
            ("historical_retention_v3", "v5_first30_diagnostic"),
        )
        self.assertFalse(repair.production_promotion_allowed())
        self.assertFalse(repair.validation_opened_by_this_module())
        self.assertTrue(repair.final_holdout_locked())

    def test_objective_and_solver_are_single_fixed_configuration(self) -> None:
        objective = repair.objective_contract()
        self.assertEqual(
            objective,
            {
                "formula": "0.5*mean(V5_BCE_w1)+0.5*mean(HISTORICAL_BCE_w1)",
                "v5_domain_weight": 0.5,
                "historical_domain_weight": 0.5,
                "positive_weight": 1.0,
                "class_reweighting": False,
                "replay_ratio": None,
                "full_batch": True,
                "head_bias_trainable": False,
                "backbone_trainable": False,
            },
        )
        solver = repair.solver_contract()
        self.assertEqual(solver["optimizer"], "LBFGS")
        self.assertEqual(solver["lr"], 1.0)
        self.assertEqual(solver["max_iter"], 100)
        self.assertEqual(solver["max_eval"], 125)
        self.assertEqual(solver["history_size"], 20)
        self.assertEqual(solver["line_search_fn"], "strong_wolfe")
        self.assertEqual(solver["initialization"], "exact-frozen-head-weight")
        self.assertEqual(solver["checkpoint_selection"], "single-final-solver-state-no-sweep")
        self.assertEqual(solver["weight_decay"], 0.0)
        self.assertEqual(solver["momentum"], 0.0)

    def test_equal_domain_objective_is_invariant_to_within_domain_duplication(self) -> None:
        torch = self._torch()
        v5_logits = torch.tensor([-2.0, 1.5], dtype=torch.float64)
        v5_targets = torch.tensor([0.0, 1.0], dtype=torch.float64)
        hist_logits = torch.tensor([-3.0, -1.0, 2.0], dtype=torch.float64)
        hist_targets = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

        total1, v51, hist1 = repair._balanced_domain_bce_v1(
            v5_logits=v5_logits,
            v5_targets=v5_targets,
            historical_logits=hist_logits,
            historical_targets=hist_targets,
        )
        total2, v52, hist2 = repair._balanced_domain_bce_v1(
            v5_logits=v5_logits.repeat(7),
            v5_targets=v5_targets.repeat(7),
            historical_logits=hist_logits.repeat(11),
            historical_targets=hist_targets.repeat(11),
        )
        self.assertTrue(math.isclose(float(v51), float(v52), rel_tol=0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(hist1), float(hist2), rel_tol=0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(float(total1), float(total2), rel_tol=0.0, abs_tol=1e-12))
        self.assertTrue(
            math.isclose(
                float(total1),
                0.5 * float(v51) + 0.5 * float(hist1),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

    def test_state_guard_allows_only_head_weight_change(self) -> None:
        torch = self._torch()
        model = repair.v52b._build_digit_model().cpu()
        frozen = repair._frozen_state_snapshot(model)
        with torch.no_grad():
            model.head.weight.add_(0.01)
        result = repair._verify_only_head_weight_changed(model, frozen)
        self.assertEqual(result["changed_state_keys"], ["head.weight"])
        self.assertTrue(result["only_head_weight_changed"])
        self.assertTrue(result["backbone_bit_identical"])
        self.assertTrue(result["head_bias_bit_identical"])

    def test_state_guard_rejects_bias_or_backbone_mutation(self) -> None:
        torch = self._torch()
        model = repair.v52b._build_digit_model().cpu()
        frozen = repair._frozen_state_snapshot(model)
        with torch.no_grad():
            model.head.bias.add_(0.01)
        with self.assertRaisesRegex(repair.MeterV5_2PError, "frozen tensor changed"):
            repair._verify_only_head_weight_changed(model, frozen)

        model = repair.v52b._build_digit_model().cpu()
        frozen = repair._frozen_state_snapshot(model)
        with torch.no_grad():
            next(model.features.parameters()).add_(0.01)
        with self.assertRaisesRegex(repair.MeterV5_2PError, "frozen tensor changed"):
            repair._verify_only_head_weight_changed(model, frozen)

    def test_corrected_retention_contract_is_relative_only(self) -> None:
        frozen = {
            "2": {"f1": 0.92, "recall": 0.99, "precision": 0.86},
            "3": {"f1": 0.995, "recall": 0.995, "precision": 0.995},
        }
        candidate = {
            "2": {"f1": 0.919, "recall": 0.989, "precision": 0.859},
            "3": {"f1": 0.994, "recall": 0.994, "precision": 0.994},
        }
        result = ret_v3.evaluate_retention_gate_v3(
            frozen_metrics=frozen,
            candidate_metrics=candidate,
        )
        self.assertEqual(result["gate"], "PASS")
        self.assertFalse(result["absolute_precision_floor_used"])
        self.assertFalse(result["absolute_recall_floor_used"])

    def test_approval_token_is_explicit(self) -> None:
        self.assertEqual(repair.APPROVAL_TOKEN, "V5_2P_FIXED_BIAS_HEAD_REPAIR_APPROVED")
        self.assertEqual(repair.EXPECTED_V5_COUNT, 540)
        self.assertEqual(repair.EXPECTED_HISTORICAL_COUNT, 26_964)
        self.assertEqual(repair.EXPECTED_FEATURE_DIM, 64)

    def test_contract_document_exists(self) -> None:
        path = Path("METER_V5_2P_FIXED_BIAS_HEAD_REPAIR_V1.md")
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        self.assertIn("freeze the complete convolutional feature extractor", text)
        self.assertIn("freeze the existing scalar head bias", text)
        self.assertIn("0.5 * mean(BCE_w1(V5_adaptation_train))", text)
        self.assertIn("V5 VALIDATION opening", text)
        self.assertIn("There is no automatic second configuration", text)


if __name__ == "__main__":
    unittest.main()
