import inspect
import unittest
from collections import OrderedDict

from st_omr_training import meter_v5_3e_rescue_training_preregistration_v1 as e
from st_omr_training import meter_v5_3f_rescue_training_execution_harness_v1 as m


class TestMeterV53FRescueTrainingExecutionHarnessV1(unittest.TestCase):
    @staticmethod
    def _torch():
        from st_omr_training import meter_v5_2b_specialist_adaptation as v52b

        return v52b._import_torch()[0]

    def _synthetic_groups(self):
        torch = self._torch()
        base = torch.arange(1, 129, dtype=torch.float32).reshape(2, 64) / 128.0
        return OrderedDict(
            (
                ("v5_frozen_false_negative_positive", base.clone()),
                ("v5_frozen_true_negative", -base.clone()),
                ("historical_frozen_false_negative_positive", 0.5 * base.clone()),
                ("historical_frozen_true_negative", -0.5 * base.clone()),
            )
        )

    def test_binds_exact_v5_3e_preregistration(self):
        contract = m.prerequisite_contract()
        self.assertEqual(
            contract["v5_3e_head_sha"],
            "f27d2334d9dfbdd8c6c70d3e214573765cee15c6",
        )
        self.assertEqual(
            contract["v5_3e_module_blob_sha"],
            "c6cf28e8ea7301b6b03f3a4d7d6b931444af3795",
        )
        self.assertEqual(
            contract["v5_3e_doc_blob_sha"],
            "46b92c2026b7da98f2fc6c84e6b6030cd86c994a",
        )
        self.assertEqual(contract["v5_3e_schema"], e.SCHEMA)
        self.assertEqual(contract["recipe_id"], e.RECIPE_ID)
        self.assertEqual(contract["candidate_configuration_count"], 1)
        self.assertEqual(contract["fixed_optimizer_steps"], 110)

    def test_model_topology_and_initialization_are_exact_and_repeatable(self):
        first = m._build_rescue_model_v1()
        second = m._build_rescue_model_v1()
        self.assertEqual(first.hidden.in_features, 64)
        self.assertEqual(first.hidden.out_features, 8)
        self.assertEqual(first.output.in_features, 8)
        self.assertEqual(first.output.out_features, 1)
        self.assertEqual(first.activation.__class__.__name__, "Tanh")
        self.assertEqual(sum(p.numel() for p in first.parameters()), 529)
        self.assertEqual(m._state_fingerprint(first), m._state_fingerprint(second))

    def test_wrong_approval_token_fails_before_tensor_access(self):
        with self.assertRaisesRegex(m.MeterV5_3FError, "approval token"):
            m.execute_rescue_tensor_harness_v1(
                digit="2",
                features_by_group=None,
                approval_token="WRONG",
                enforce_preregistered_counts=False,
            )

    def test_group_surface_is_exact_and_preregistered_counts_fail_closed(self):
        groups = self._synthetic_groups()
        reordered = OrderedDict(reversed(list(groups.items())))
        with self.assertRaisesRegex(m.MeterV5_3FError, "exact preregistered order"):
            m._validate_group_features(
                digit="2",
                features_by_group=reordered,
                enforce_preregistered_counts=False,
            )
        with self.assertRaisesRegex(m.MeterV5_3FError, "count changed"):
            m._validate_group_features(
                digit="2",
                features_by_group=groups,
                enforce_preregistered_counts=True,
            )

    def test_nonfinite_input_fails_closed(self):
        torch = self._torch()
        groups = self._synthetic_groups()
        groups["historical_frozen_true_negative"][0, 0] = float("nan")
        with self.assertRaisesRegex(m.MeterV5_3FError, "non-finite features"):
            m.execute_rescue_tensor_harness_v1(
                digit="3",
                features_by_group=groups,
                approval_token=m.APPROVAL_TOKEN,
                enforce_preregistered_counts=False,
            )
        self.assertTrue(torch.isnan(groups["historical_frozen_true_negative"][0, 0]))

    def test_four_group_objective_is_equal_weighted_and_duplication_invariant(self):
        groups = self._synthetic_groups()
        model = m._build_rescue_model_v1()
        first, _, first_counts = m._four_group_objective_v1(model, groups)
        repeated = OrderedDict(
            (name, value.repeat((3, 1))) for name, value in groups.items()
        )
        second, _, second_counts = m._four_group_objective_v1(model, repeated)
        self.assertAlmostEqual(float(first.item()), float(second.item()), places=7)
        self.assertEqual(set(first_counts.values()), {2})
        self.assertEqual(set(second_counts.values()), {6})
        self.assertEqual(
            m.execution_contract()["objective"]["group_weights"],
            {name: 0.25 for name in e.TRAIN_GROUPS},
        )

    def test_synthetic_ci_execution_is_exact_110_steps_and_deterministic(self):
        groups1 = self._synthetic_groups()
        groups2 = self._synthetic_groups()
        model1, evidence1 = m.execute_rescue_tensor_harness_v1(
            digit="2",
            features_by_group=groups1,
            approval_token=m.APPROVAL_TOKEN,
            enforce_preregistered_counts=False,
        )
        model2, evidence2 = m.execute_rescue_tensor_harness_v1(
            digit="2",
            features_by_group=groups2,
            approval_token=m.APPROVAL_TOKEN,
            enforce_preregistered_counts=False,
        )
        self.assertEqual(evidence1["optimizer"], "AdamW")
        self.assertEqual(evidence1["optimizer_steps"], 110)
        self.assertEqual(evidence1["parameter_count"], 529)
        self.assertFalse(evidence1["authoritative_dataset_execution"])
        self.assertFalse(evidence1["preregistered_count_enforcement"])
        self.assertEqual(
            evidence1["initial_state_fingerprint"],
            evidence2["initial_state_fingerprint"],
        )
        self.assertEqual(
            evidence1["final_state_fingerprint"],
            evidence2["final_state_fingerprint"],
        )
        self.assertEqual(m._state_fingerprint(model1), m._state_fingerprint(model2))
        self.assertNotEqual(
            evidence1["initial_state_fingerprint"],
            evidence1["final_state_fingerprint"],
        )
        self.assertTrue(evidence1["finite_losses"])
        self.assertTrue(evidence1["finite_gradients"])
        self.assertTrue(evidence1["finite_post_step_parameters"])
        self.assertEqual(evidence1["gradient_clip_global_norm"], 1.0)
        self.assertFalse(evidence1["checkpoint_write"])
        self.assertFalse(evidence1["protected_evaluation_opened"])

    def test_input_feature_tensors_are_not_mutated(self):
        torch = self._torch()
        groups = self._synthetic_groups()
        before = {name: value.clone() for name, value in groups.items()}
        m.execute_rescue_tensor_harness_v1(
            digit="3",
            features_by_group=groups,
            approval_token=m.APPROVAL_TOKEN,
            enforce_preregistered_counts=False,
        )
        for name in e.TRAIN_GROUPS:
            self.assertTrue(torch.equal(groups[name], before[name]))

    def test_harness_has_no_dataset_checkpoint_or_frozen_model_mutation_path(self):
        source = inspect.getsource(m).lower()
        for forbidden in (
            "torch.save(",
            "torch.load(",
            "load_state_dict(",
            ".copy_(",
            "pathlib",
            "read_text(",
            "read_bytes(",
            "write_text(",
            "write_bytes(",
        ):
            self.assertNotIn(forbidden, source)
        safety = m.safety_boundary()
        self.assertFalse(safety["authoritative_dataset_execution_present"])
        self.assertFalse(safety["dataset_path_access"])
        self.assertFalse(safety["checkpoint_load"])
        self.assertFalse(safety["checkpoint_write"])
        self.assertFalse(safety["frozen_model_reference_accepted"])
        self.assertFalse(safety["frozen_model_mutation_surface"])
        self.assertEqual(safety["trainable_surface"], "new-rescue-parameters-only")
        self.assertTrue(safety["digit4_frozen"])

    def test_protected_surfaces_thresholds_and_production_remain_closed(self):
        safety = m.safety_boundary()
        self.assertFalse(safety["threshold_tuning"])
        self.assertFalse(safety["hyperparameter_sweep"])
        self.assertFalse(safety["automatic_second_configuration"])
        self.assertFalse(safety["historical_validation_opened"])
        self.assertFalse(safety["first30_opened"])
        self.assertFalse(safety["v5_reserve_opened"])
        self.assertFalse(safety["v5_validation_opened"])
        self.assertTrue(safety["final_holdout_locked"])
        self.assertFalse(safety["bbox_access_added"])
        self.assertFalse(safety["crop_geometry_change"])
        self.assertFalse(safety["spatial_heuristic_change"])
        self.assertFalse(safety["resolver_wiring"])
        self.assertFalse(safety["production_promotion"])
        self.assertFalse(m.checkpoint_write_allowed())
        self.assertFalse(m.protected_evaluation_access_allowed())
        self.assertFalse(m.production_promotion_allowed())
        self.assertFalse(m.authoritative_colab_execution_available())

    def test_next_gate_is_separate_single_authoritative_execution(self):
        self.assertEqual(
            m.future_gate_order(),
            (
                "v5_3f_exact_ci_green_sha",
                "separately_authorized_exact_train_tensor_materialization_and_single_execution",
                "single_candidate_numerical_and_state_isolation",
                "train_v5_f1_and_frozen_correct_retention",
                "historical_validation_retention_at_frozen_thresholds",
                "immutable_v5_first30_diagnostic",
                "separately_authorized_v5_validation",
                "separately_authorized_final_holdout",
            ),
        )


if __name__ == "__main__":
    unittest.main()
