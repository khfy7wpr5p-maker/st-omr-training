from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

from st_omr_training import meter_v5_3d_gated_rescue_architecture_contract_v1 as v


class TestMeterV53DGatedRescueArchitectureContractV1(unittest.TestCase):
    def test_exact_v5_3c_pass_evidence_is_bound_without_overclaim(self):
        evidence = v.prerequisite_evidence_contract()
        self.assertEqual(
            evidence["v5_3c_implementation_head"],
            "61361612abfce132994abaca742c855f91305b44",
        )
        self.assertEqual(
            evidence["v5_3c_harness_head"],
            "74bbdba45b08ee4ca350b487627b792cc5255806",
        )
        self.assertEqual(
            evidence["v5_3c_report_sha256"],
            "630b202f5369f12ea2562c81799613b81a295d77da08f9b3dd94a8fb1d801389",
        )
        self.assertEqual(
            evidence["v5_3c_execution_envelope_sha256"],
            "d682a31809702373df312b828a8452440517e480c55ad69fdf9633f35e5f436d",
        )
        self.assertTrue(evidence["shared_linear_head_feasible_on_train"])
        self.assertFalse(evidence["shared_linear_head_selected_for_repair"])
        self.assertTrue(evidence["shared_linear_head_lane_closed_by_safety_policy"])
        self.assertFalse(evidence["shared_linear_infeasibility_claimed"])
        self.assertFalse(evidence["generalization_proven"])

    def test_topology_is_fixed_small_separate_and_nonlinear(self):
        topology = v.topology_contract()
        self.assertEqual(topology["specialists"], ("2", "3"))
        self.assertEqual(topology["feature_dim"], 64)
        self.assertEqual(topology["hidden_width"], 8)
        self.assertEqual(topology["activation"], "tanh")
        self.assertEqual(topology["parameters_per_rescue"], 529)
        self.assertEqual(topology["total_trainable_parameters_if_later_authorized"], 1058)
        self.assertTrue(topology["separate_state_namespace"])
        self.assertFalse(topology["frozen_checkpoint_replacement"])
        self.assertFalse(topology["architecture_sweep"])

    def test_frozen_positive_is_never_demoted_or_rescored(self):
        result = v.shadow_candidate_decision_v1(
            digit="2",
            frozen_probability=0.80,
            rescue_probability=0.0,
            rescue_artifact_verified=True,
        )
        self.assertTrue(result["frozen_decision"])
        self.assertFalse(result["rescue_eligible"])
        self.assertFalse(result["rescue_evaluated"])
        self.assertIsNone(result["rescue_probability"])
        self.assertTrue(result["shadow_candidate_decision"])
        self.assertTrue(result["production_decision"])
        self.assertFalse(result["production_authority_changed"])

    def test_verified_rescue_can_only_add_a_shadow_positive(self):
        rescued = v.shadow_candidate_decision_v1(
            digit="3",
            frozen_probability=0.20,
            rescue_probability=0.75,
            rescue_artifact_verified=True,
        )
        self.assertFalse(rescued["frozen_decision"])
        self.assertTrue(rescued["rescue_eligible"])
        self.assertTrue(rescued["rescue_evaluated"])
        self.assertTrue(rescued["rescue_decision"])
        self.assertTrue(rescued["shadow_candidate_decision"])
        self.assertFalse(rescued["production_decision"])

        rejected = v.shadow_candidate_decision_v1(
            digit="3",
            frozen_probability=0.20,
            rescue_probability=0.25,
            rescue_artifact_verified=True,
        )
        self.assertFalse(rejected["shadow_candidate_decision"])

    def test_missing_or_unverified_rescue_fails_to_frozen_decision(self):
        result = v.shadow_candidate_decision_v1(
            digit="2",
            frozen_probability=0.10,
            rescue_probability=None,
            rescue_artifact_verified=False,
        )
        self.assertFalse(result["rescue_evaluated"])
        self.assertFalse(result["shadow_candidate_decision"])
        self.assertFalse(result["production_decision"])

        with self.assertRaisesRegex(v.MeterV5_3DContractError, "requires"):
            v.shadow_candidate_decision_v1(
                digit="2",
                frozen_probability=0.10,
                rescue_probability=None,
                rescue_artifact_verified=True,
            )

    def test_digit4_has_no_rescue_and_production_remains_frozen(self):
        with self.assertRaisesRegex(v.MeterV5_3DContractError, "only for 2-AI and 3-AI"):
            v.shadow_candidate_decision_v1(
                digit="4",
                frozen_probability=0.10,
                rescue_probability=0.90,
                rescue_artifact_verified=True,
            )
        self.assertTrue(v.frozen_production_decision_v1(digit="4", frozen_probability=0.47))
        self.assertFalse(v.frozen_production_decision_v1(digit="4", frozen_probability=0.46))

    def test_train_surface_is_frozen_negative_four_group_balanced(self):
        surface = v.train_surface_contract()
        self.assertFalse(surface["future_execution_authorized"])
        self.assertEqual(surface["eligible_rows"], "frozen-negative-only")
        self.assertEqual(surface["groups"], v.TRAIN_GROUPS)
        self.assertEqual(surface["group_weights"], {name: 0.25 for name in v.TRAIN_GROUPS})
        self.assertEqual(surface["group_weight_sum"], 1.0)
        self.assertEqual(
            surface["expected_group_counts"]["2"],
            {
                "v5_frozen_false_negative_positive": 90,
                "v5_frozen_true_negative": 450,
                "historical_frozen_false_negative_positive": 14,
                "historical_frozen_true_negative": 25254,
            },
        )
        self.assertEqual(surface["data_surfaces"], ("v5_train", "historical_train"))
        self.assertFalse(surface["historical_validation_used"])
        self.assertFalse(surface["hyperparameter_sweep"])

    def test_observed_group_count_change_fails_closed(self):
        observed = {
            digit: dict(counts) for digit, counts in v.EXPECTED_TRAIN_GROUP_COUNTS.items()
        }
        result = v.validate_architecture_contract_v1(observed_group_counts=observed)
        self.assertEqual(result["gate"], "PASS")
        observed["3"]["historical_frozen_true_negative"] -= 1
        with self.assertRaisesRegex(v.MeterV5_3DContractError, "group counts changed"):
            v.validate_architecture_contract_v1(observed_group_counts=observed)

    def test_stage_has_no_training_or_persistence_path(self):
        safety = v.safety_boundary()
        self.assertTrue(safety["architecture_contract_only"])
        self.assertFalse(safety["training_authorized"])
        self.assertFalse(safety["training_executed"])
        self.assertFalse(safety["autograd_grad_used"])
        self.assertFalse(safety["backward"])
        self.assertEqual(safety["optimizer_steps"], 0)
        self.assertFalse(safety["model_parameter_mutation"])
        self.assertFalse(safety["candidate_checkpoint_write"])
        self.assertTrue(safety["frozen_backbone"])
        self.assertTrue(safety["frozen_head_weight"])
        self.assertTrue(safety["frozen_head_bias"])
        self.assertFalse(safety["historical_retention_executed"])
        self.assertFalse(safety["first30_opened"])
        self.assertFalse(safety["v5_validation_opened"])
        self.assertTrue(safety["final_holdout_locked"])
        self.assertTrue(safety["digit4_frozen"])
        self.assertFalse(v.training_entry_point_available())
        self.assertFalse(v.production_promotion_allowed())

        source = inspect.getsource(v)
        for forbidden in (
            "import torch",
            "torch.optim",
            ".backward(",
            "optimizer.step",
            "torch.save(",
            "run_historical_retention",
            "run_first30",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_future_gate_order_stops_before_closed_surfaces(self):
        self.assertEqual(
            v.future_gate_order(),
            (
                "separate_fixed_training_recipe_and_exact_ci_green_sha",
                "single_candidate_numerical_and_state_isolation",
                "train_v5_f1_and_frozen_correct_retention",
                "historical_validation_retention_at_frozen_thresholds",
                "immutable_v5_first30_diagnostic",
                "separately_authorized_v5_validation",
                "separately_authorized_final_holdout",
            ),
        )
        decision = v.decision_contract()
        self.assertEqual(decision["multiple_digit_hits"], "preserve-existing-ambiguous-result")
        self.assertFalse(decision["digit4_rescue_allowed"])
        self.assertFalse(decision["rescue_runtime_enabled"])

    def test_nonfinite_probability_and_unknown_specialist_fail_closed(self):
        with self.assertRaisesRegex(v.MeterV5_3DContractError, "inside"):
            v.shadow_candidate_decision_v1(
                digit="2",
                frozen_probability=float("nan"),
                rescue_probability=None,
                rescue_artifact_verified=False,
            )
        with self.assertRaisesRegex(v.MeterV5_3DContractError, "unknown"):
            v.frozen_production_decision_v1(digit="5", frozen_probability=0.9)
        with self.assertRaisesRegex(v.MeterV5_3DContractError, "flag must be bool"):
            v.shadow_candidate_decision_v1(
                digit="2",
                frozen_probability=0.1,
                rescue_probability=0.9,
                rescue_artifact_verified="yes",  # type: ignore[arg-type]
            )

    def test_exact_evidence_hash_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report.json"
            envelope = root / "envelope.json"
            report.write_text("{}", encoding="utf-8")
            envelope.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(v.MeterV5_3DContractError, "report SHA256 mismatch"):
                v.verify_exact_v5_3c_evidence(
                    report_path=report,
                    envelope_path=envelope,
                )

    def test_contract_document_states_the_architecture_and_no_authority(self):
        path = Path("METER_V5_3D_GATED_RESCUE_ARCHITECTURE_CONTRACT_V1.md")
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        self.assertIn("Linear(64, 8) -> tanh -> Linear(8, 1)", text)
        self.assertIn("does not train a model", text)
        self.assertIn("frozen positive cannot be demoted", text)
        self.assertIn("Historical retention", text)


if __name__ == "__main__":
    unittest.main()
