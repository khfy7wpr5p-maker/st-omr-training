import inspect
import unittest

from st_omr_training import meter_v5_3d_gated_rescue_architecture_contract_v1 as d
from st_omr_training import meter_v5_3e_rescue_training_preregistration_v1 as m


class TestMeterV53ERescueTrainingPreregistrationV1(unittest.TestCase):
    def test_binds_exact_v5_3d_contract(self):
        prerequisite = m.prerequisite_contract()
        self.assertEqual(
            prerequisite["v5_3d_head_sha"],
            "7d50fbec4d730aa46c69f7dfa3a20917a3478ef8",
        )
        self.assertEqual(
            prerequisite["v5_3d_module_blob_sha"],
            "f74ff71e9999f889086c3cf68a9c6ed5a0e69427",
        )
        self.assertEqual(
            prerequisite["v5_3d_doc_blob_sha"],
            "3be6b10fb18cfee049de05a0c5e21a253990f4c8",
        )
        self.assertEqual(prerequisite["v5_3d_schema"], d.SCHEMA)

    def test_architecture_is_inherited_without_change(self):
        prerequisite = m.prerequisite_contract()
        self.assertEqual(m.RESCUE_SPECIALISTS, ("2", "3"))
        self.assertEqual(m.FEATURE_DIM, 64)
        self.assertEqual(m.HIDDEN_WIDTH, 8)
        self.assertEqual(m.ACTIVATION, "tanh")
        self.assertEqual(m.RESCUE_THRESHOLD, 0.50)
        self.assertEqual(m.PARAMETERS_PER_RESCUE, 529)
        self.assertTrue(prerequisite["architecture_contract_inherited_without_change"])
        self.assertTrue(prerequisite["digit4_frozen"])
        self.assertTrue(prerequisite["frozen_specialist_tensors_authoritative"])

    def test_exact_four_group_objective_is_inherited(self):
        objective = m.objective_contract()
        self.assertEqual(
            objective["groups"],
            (
                "v5_frozen_false_negative_positive",
                "v5_frozen_true_negative",
                "historical_frozen_false_negative_positive",
                "historical_frozen_true_negative",
            ),
        )
        self.assertEqual(
            objective["expected_group_counts"]["2"],
            {
                "v5_frozen_false_negative_positive": 90,
                "v5_frozen_true_negative": 450,
                "historical_frozen_false_negative_positive": 14,
                "historical_frozen_true_negative": 25254,
            },
        )
        self.assertEqual(
            objective["expected_group_counts"]["3"],
            {
                "v5_frozen_false_negative_positive": 90,
                "v5_frozen_true_negative": 450,
                "historical_frozen_false_negative_positive": 12,
                "historical_frozen_true_negative": 25364,
            },
        )

    def test_group_loss_is_exactly_equal_weighted(self):
        objective = m.objective_contract()
        self.assertEqual(objective["loss"], "binary_cross_entropy_with_logits")
        self.assertEqual(
            objective["loss_reduction"],
            "mean_per_group_then_equal_weight_sum",
        )
        self.assertEqual(set(objective["group_weights"].values()), {0.25})
        self.assertEqual(objective["group_weight_sum"], 1.0)
        self.assertEqual(objective["positive_weight"], 1.0)
        self.assertEqual(objective["label_smoothing"], 0.0)

    def test_only_train_surfaces_and_frozen_negative_rows_are_eligible(self):
        objective = m.objective_contract()
        self.assertEqual(
            objective["eligible_rows"],
            "same-specialist-frozen-negative-only",
        )
        self.assertEqual(objective["data_surfaces"], ("v5_train", "historical_train"))
        self.assertEqual(
            objective["objective_rows"],
            "all_exact_rows_in_each_of_four_frozen_negative_groups",
        )
        self.assertEqual(
            objective["canonical_row_order"],
            "stable_manifest_identity_ascending",
        )
        self.assertEqual(
            objective["batching"],
            "full_group_objective_each_optimizer_step",
        )
        self.assertFalse(objective["shuffle"])
        self.assertFalse(objective["sampling_with_replacement"])

    def test_exactly_one_fixed_recipe_exists(self):
        recipe = m.fixed_training_recipe()
        self.assertEqual(recipe["candidate_configuration_count"], 1)
        self.assertEqual(recipe["specialists_trained_independently"], ("2", "3"))
        self.assertEqual(recipe["initialization"], "xavier_uniform_weights_zero_bias")
        self.assertEqual(recipe["initialization_gain"], 1.0)
        self.assertEqual(recipe["seed"], 52023)
        self.assertTrue(recipe["same_initial_parameter_realization_per_specialist"])
        self.assertEqual(recipe["device"], "cpu")
        self.assertEqual(recipe["dtype"], "float32")
        self.assertTrue(recipe["deterministic_algorithms"])
        self.assertFalse(recipe["amp_enabled"])

    def test_optimizer_recipe_is_fully_fixed(self):
        recipe = m.fixed_training_recipe()
        self.assertEqual(recipe["optimizer"], "AdamW")
        self.assertEqual(recipe["learning_rate"], 1e-3)
        self.assertEqual(recipe["weight_decay"], 1e-4)
        self.assertEqual(recipe["betas"], (0.9, 0.999))
        self.assertEqual(recipe["epsilon"], 1e-8)
        self.assertEqual(recipe["fixed_optimizer_steps"], 110)
        self.assertEqual(recipe["gradient_clip_global_norm"], 1.0)
        self.assertEqual(recipe["scheduler"], "none")
        self.assertEqual(recipe["warmup_steps"], 0)
        self.assertFalse(recipe["early_stopping"])
        self.assertTrue(recipe["fixed_final_step_only"])

    def test_sweep_threshold_and_fallback_paths_are_forbidden(self):
        recipe = m.fixed_training_recipe()
        self.assertFalse(recipe["hyperparameter_sweep"])
        self.assertFalse(recipe["threshold_search"])
        self.assertFalse(recipe["architecture_search"])
        self.assertFalse(recipe["automatic_second_configuration"])
        self.assertFalse(recipe["fallback_optimizer"])
        self.assertEqual(m.RESCUE_THRESHOLD, d.RESCUE_THRESHOLD)

    def test_original_models_and_digit4_remain_frozen(self):
        recipe = m.fixed_training_recipe()
        self.assertTrue(recipe["frozen_backbone"])
        self.assertTrue(recipe["frozen_head_weight"])
        self.assertTrue(recipe["frozen_head_bias"])
        self.assertTrue(recipe["digit4_frozen"])

    def test_later_execution_must_fail_closed_numerically(self):
        guards = m.numerical_execution_guards()
        self.assertTrue(guards["finite_input_features_required"])
        self.assertTrue(guards["finite_initial_parameters_required"])
        self.assertTrue(guards["finite_loss_each_step_required"])
        self.assertTrue(guards["finite_gradients_each_step_required"])
        self.assertTrue(guards["finite_parameters_after_step_required"])
        self.assertEqual(guards["gradient_global_norm_clip"], 1.0)
        self.assertTrue(guards["abort_on_nonfinite"])
        self.assertFalse(guards["abort_writes_checkpoint"])
        self.assertTrue(guards["frozen_tensor_bit_identity_required"])
        self.assertTrue(guards["only_rescue_namespace_may_change"])

    def test_protected_surfaces_remain_closed(self):
        protected = m.protected_surface_contract()
        self.assertFalse(protected["historical_validation_opened"])
        self.assertFalse(protected["first30_opened"])
        self.assertFalse(protected["v5_reserve_opened"])
        self.assertFalse(protected["v5_validation_opened"])
        self.assertTrue(protected["final_holdout_locked"])
        self.assertFalse(protected["bbox_access_added"])
        self.assertFalse(protected["crop_geometry_change"])
        self.assertFalse(protected["spatial_heuristic_change"])
        self.assertFalse(protected["threshold_tuning"])
        self.assertFalse(protected["resolver_wiring"])
        self.assertFalse(protected["production_promotion"])

    def test_preregistration_contains_no_training_or_persistence_implementation(self):
        source = inspect.getsource(m).lower()
        self.assertNotIn("import torch", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("torch.save(", source)
        self.assertNotIn("pickle.dump(", source)
        safety = m.safety_boundary()
        self.assertTrue(safety["preregistration_only"])
        self.assertFalse(safety["training_implementation_present"])
        self.assertFalse(safety["training_authorized"])
        self.assertFalse(safety["training_executed"])
        self.assertFalse(safety["autograd_used"])
        self.assertFalse(safety["backward"])
        self.assertEqual(safety["optimizer_steps_executed"], 0)
        self.assertFalse(safety["checkpoint_write"])
        self.assertFalse(safety["rescue_artifact_write"])
        self.assertFalse(safety["frozen_model_mutation"])
        self.assertFalse(safety["retention_executed"])

    def test_no_execution_entry_point_is_exposed(self):
        self.assertFalse(m.training_entry_point_available())
        self.assertFalse(m.checkpoint_write_allowed())
        self.assertFalse(m.protected_evaluation_access_allowed())

    def test_future_gate_order_requires_ci_then_single_execution(self):
        self.assertEqual(
            m.future_gate_order(),
            (
                "v5_3e_exact_ci_green_sha",
                "single_fixed_recipe_execution_harness",
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
