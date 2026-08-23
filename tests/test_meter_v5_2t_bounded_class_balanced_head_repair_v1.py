from __future__ import annotations

import copy
import importlib.util
import inspect
import math
from pathlib import Path
import tempfile
import unittest

from st_omr_training import meter_v5_2s_bounded_class_balanced_head_contract_v1 as s
from st_omr_training import meter_v5_2t_bounded_class_balanced_head_repair_v1 as t


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class TestMeterV52TBoundedClassBalancedHeadRepairV1(unittest.TestCase):
    def test_implementation_is_exactly_bound_and_stops_before_retention(self):
        contract = t.implementation_contract()
        self.assertEqual(contract["prerequisite"], s.prerequisite_evidence_contract())
        self.assertEqual(contract["objective"], s.objective_contract())
        self.assertTrue(contract["solver"]["execution_authorized"])
        self.assertTrue(contract["actual_data_execution_requires_exact_sha_colab_harness"])
        self.assertFalse(contract["colab_harness_present_in_this_stage"])
        self.assertFalse(contract["historical_retention_executed_by_this_module"])
        self.assertFalse(contract["historical_validation_opened"])
        self.assertFalse(contract["first30_opened"])
        self.assertFalse(contract["v5_validation_opened"])
        self.assertTrue(contract["final_holdout_locked"])
        self.assertTrue(contract["digit4_frozen"])
        self.assertFalse(t.historical_retention_executed_by_this_module())
        self.assertFalse(t.validation_opened_by_this_module())
        self.assertFalse(t.production_promotion_allowed())

    def test_single_configuration_and_exact_token_are_frozen(self):
        self.assertEqual(
            t.APPROVAL_TOKEN,
            "V5_2T_SINGLE_BOUNDED_CLASS_BALANCED_RUN_APPROVED",
        )
        safety = t.safety_boundary()
        self.assertTrue(safety["single_fixed_training_entry"])
        self.assertFalse(safety["automatic_second_configuration"])
        self.assertFalse(safety["hyperparameter_sweep"])
        self.assertEqual(safety["trainable_surface"], "head.weight-only-64-parameters")
        self.assertTrue(safety["frozen_backbone"])
        self.assertTrue(safety["frozen_head_bias"])
        self.assertEqual(t.EXPECTED_HISTORICAL_POSITIVE_COUNT, {"2": 1527, "3": 1587})

    def test_wrong_token_fails_before_any_path_access(self):
        with self.assertRaisesRegex(t.MeterV5_2TError, "approval token"):
            t.train_bounded_class_balanced_head_repair_v1(
                "/does/not/exist",
                m4a_root="/does/not/exist",
                d10_root="/does/not/exist",
                digit2_frozen="/does/not/exist",
                digit3_frozen="/does/not/exist",
                v5_2r_report="/does/not/exist",
                v5_2r_execution_envelope="/does/not/exist",
                confirmation="WRONG",
            )

    def test_source_has_no_retention_first30_or_validation_execution(self):
        source = inspect.getsource(t)
        for forbidden in (
            "run_historical_retention_gate",
            "evaluate_diagnostic_gate",
            "FINAL_HOLDOUT_150",
            "V5_VALIDATION",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_torch_objective_matches_pure_contract_and_ignores_group_duplication(self):
        import torch

        logits = {
            "v5_positive": torch.tensor([0.2, 1.0], dtype=torch.float64),
            "v5_negative": torch.tensor([-0.2, -1.0, -2.0], dtype=torch.float64),
            "historical_positive": torch.tensor([0.4, 1.4, 2.0], dtype=torch.float64),
            "historical_negative": torch.tensor([-0.4, -1.4], dtype=torch.float64),
        }
        targets = {
            "v5_positive": torch.ones(2, dtype=torch.float64),
            "v5_negative": torch.zeros(3, dtype=torch.float64),
            "historical_positive": torch.ones(3, dtype=torch.float64),
            "historical_negative": torch.zeros(2, dtype=torch.float64),
        }
        v5_logits = torch.cat((logits["v5_positive"], logits["v5_negative"]))
        v5_targets = torch.cat((targets["v5_positive"], targets["v5_negative"]))
        hist_logits = torch.cat(
            (logits["historical_positive"], logits["historical_negative"])
        )
        hist_targets = torch.cat(
            (targets["historical_positive"], targets["historical_negative"])
        )
        total, losses, counts = t._four_group_bce_torch_v1(
            v5_logits=v5_logits,
            v5_targets=v5_targets,
            historical_logits=hist_logits,
            historical_targets=hist_targets,
        )
        pure, pure_losses = s.balanced_four_group_bce_v1(
            group_logits={name: value.tolist() for name, value in logits.items()},
            group_targets={name: value.tolist() for name, value in targets.items()},
        )
        self.assertAlmostEqual(float(total.item()), pure, places=12)
        self.assertEqual(counts, {name: len(logits[name]) for name in s.GROUPS})
        for name in s.GROUPS:
            self.assertAlmostEqual(float(losses[name].item()), pure_losses[name], places=12)

        repeated_total, _losses, _counts = t._four_group_bce_torch_v1(
            v5_logits=torch.cat((logits["v5_positive"].repeat(7), logits["v5_negative"].repeat(3))),
            v5_targets=torch.cat((targets["v5_positive"].repeat(7), targets["v5_negative"].repeat(3))),
            historical_logits=torch.cat((logits["historical_positive"].repeat(11), logits["historical_negative"].repeat(5))),
            historical_targets=torch.cat((targets["historical_positive"].repeat(11), targets["historical_negative"].repeat(5))),
        )
        self.assertAlmostEqual(float(total.item()), float(repeated_total.item()), places=12)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_synthetic_solver_is_bounded_and_mutates_only_head_weight(self):
        import torch

        torch.manual_seed(22024)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        model = t.v52b._build_digit_model().cpu()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.weight.reshape(-1)[0] = 2.0
            model.head.bias.zero_()
        frozen_state = t.v52p._frozen_state_snapshot(model)
        frozen_fingerprint = t._state_fingerprint_without_numpy_v1(model)
        self.assertEqual(len(frozen_fingerprint), 64)
        self.assertEqual(
            frozen_fingerprint,
            t._state_fingerprint_without_numpy_v1(copy.deepcopy(model)),
        )

        def rows(values):
            result = []
            for first, second in values:
                row = torch.zeros(64, dtype=torch.float32)
                row[0] = first
                row[1] = second
                result.append(row)
            return result

        v5_positive = rows([(0.5, 1.0), (0.4, 0.8), (0.3, 1.2)])
        v5_negative = rows([(-0.5, -1.0), (-0.4, -0.8), (-0.3, -1.2)])
        hist_positive = rows([(1.0, 0.2), (0.8, 0.3), (0.7, 0.1)])
        hist_negative = rows([(-1.0, -0.2), (-0.8, -0.3), (-0.7, -0.1)])
        v5_features = torch.stack(v5_positive + v5_negative)
        v5_targets = torch.tensor([1.0] * 3 + [0.0] * 3)
        hist_features = torch.stack(hist_positive + hist_negative)
        hist_targets = torch.tensor([1.0] * 3 + [0.0] * 3)

        fit = t._fit_bounded_head_v1(
            model,
            v5_features=v5_features,
            v5_targets=v5_targets,
            historical_features=hist_features,
            historical_targets=hist_targets,
            enforce_preregistered_counts=False,
        )
        invariants = t.v52p._verify_only_head_weight_changed(model, frozen_state)
        self.assertNotEqual(frozen_fingerprint, t._state_fingerprint_without_numpy_v1(model))
        self.assertTrue(invariants["only_head_weight_changed"])
        self.assertTrue(invariants["backbone_bit_identical"])
        self.assertTrue(invariants["head_bias_bit_identical"])
        self.assertTrue(fit["finite_non_increasing_objective"])
        self.assertEqual(fit["geometry_float64"]["gate"], "PASS")
        self.assertEqual(fit["geometry_float32_copy_back"]["gate"], "PASS")
        self.assertLessEqual(
            fit["geometry_float32_copy_back"]["head_angle_change_degrees"], 15.0
        )
        self.assertLessEqual(
            fit["geometry_float32_copy_back"]["candidate_over_frozen_l2"],
            1.0 + math.sin(math.radians(15.0)) + 1e-9,
        )
        self.assertTrue(fit["float32_copy_back_bit_exact"])
        self.assertTrue(fit["lbfgs_termination"]["final_gradient_finite"])

        with tempfile.TemporaryDirectory() as temporary:
            candidate_path = Path(temporary) / "candidate.pt"
            source_sha = "a" * 64
            manifest_sha = "b" * 64
            saved = t._save_candidate(
                model=model,
                path=candidate_path,
                digit="2",
                source_sha=source_sha,
                manifest_sha=manifest_sha,
                fit=fit,
                invariants=invariants,
            )
            self.assertEqual(saved["candidate_sha256"], t.v52b._sha_file(candidate_path))
            reloaded = t._load_candidate(
                candidate_path,
                digit="2",
                source_sha=source_sha,
                manifest_sha=manifest_sha,
            )
            reload_invariants = t.v52p._verify_only_head_weight_changed(
                reloaded, frozen_state
            )
            self.assertEqual(reload_invariants, invariants)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
    def test_missing_class_fails_closed(self):
        import torch

        model = t.v52b._build_digit_model().cpu()
        features = torch.zeros((4, 64), dtype=torch.float32)
        all_positive = torch.ones(4, dtype=torch.float32)
        mixed = torch.tensor([1.0, 0.0, 1.0, 0.0])
        with self.assertRaisesRegex(t.MeterV5_2TError, "both classes"):
            t._fit_bounded_head_v1(
                model,
                v5_features=features,
                v5_targets=all_positive,
                historical_features=features,
                historical_targets=mixed,
                enforce_preregistered_counts=False,
            )


if __name__ == "__main__":
    unittest.main()
