import inspect
import unittest

from st_omr_training import meter_v5_2b_specialist_adaptation as v52b
from st_omr_training import meter_v5_2o_frozen_head_axis_audit_v1 as m


class TestMeterV52OFrozenHeadAxisAuditV1(unittest.TestCase):
    def _torch(self):
        torch, _nn = v52b._import_torch()
        return torch

    def _fixture(self):
        torch = self._torch()
        source_neg = torch.zeros((4, 64), dtype=torch.float32)
        source_pos = torch.zeros((4, 64), dtype=torch.float32)
        source_neg[:, 0] = -2.0
        source_pos[:, 0] = 2.0
        source_features = torch.cat((source_neg, source_pos), dim=0)
        source_targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)

        # Same class direction as source, but translated strongly along head axis.
        v5_neg = torch.zeros((2, 64), dtype=torch.float32)
        v5_pos = torch.zeros((2, 64), dtype=torch.float32)
        v5_neg[:, 0] = 4.0
        v5_pos[:, 0] = 6.0
        v5_features = torch.cat((v5_neg, v5_pos), dim=0)
        v5_targets = torch.tensor([0, 0, 1, 1], dtype=torch.float32)

        head_weight = torch.zeros(64, dtype=torch.float32)
        head_weight[0] = 1.0
        return source_features, source_targets, v5_features, v5_targets, head_weight

    def test_translated_v5_can_preserve_head_ordering_without_fitting(self):
        source_features, source_targets, v5_features, v5_targets, head_weight = self._fixture()
        result = m.head_axis_transfer_metrics_v1(
            source_features=source_features,
            source_targets=source_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
            head_weight=head_weight,
            head_bias=0.0,
            frozen_threshold=0.5,
        )
        source = result["historical_train"]
        v5 = result["v5_adaptation_train"]
        cross = result["source_to_v5"]
        self.assertEqual(source["rank_auc"], 1.0)
        self.assertEqual(v5["rank_auc"], 1.0)
        self.assertEqual(v5["strict_separation_gap_logit"], 2.0)
        self.assertTrue(v5["strictly_separable_under_same_head_direction"])
        self.assertTrue(v5["all_positive_at_or_above_frozen_boundary"])
        self.assertTrue(v5["all_negative_at_or_above_frozen_boundary"])
        self.assertAlmostEqual(cross["midpoint_shift_along_normalized_head_axis"], 5.0, places=12)
        self.assertAlmostEqual(cross["v5_over_source_abs_class_gap_along_head"], 0.5, places=12)
        self.assertAlmostEqual(cross["abs_midpoint_shift_over_abs_v5_class_gap"], 2.5, places=12)
        self.assertTrue(cross["class_gap_direction_preserved_along_head"])
        self.assertTrue(result["same_frozen_head_direction_strictly_separates_v5_train"])
        self.assertFalse(result["bias_or_threshold_selected"])
        self.assertFalse(result["classifier_fit_performed"])

    def test_reversed_v5_head_ordering_is_detected(self):
        torch = self._torch()
        source_features, source_targets, v5_features, v5_targets, head_weight = self._fixture()
        reversed_v5 = v5_features.clone()
        reversed_v5[:, 0] *= -1.0
        result = m.head_axis_transfer_metrics_v1(
            source_features=source_features,
            source_targets=source_targets,
            v5_features=reversed_v5,
            v5_targets=v5_targets,
            head_weight=head_weight,
            head_bias=0.0,
            frozen_threshold=0.5,
        )
        self.assertEqual(result["v5_adaptation_train"]["rank_auc"], 0.0)
        self.assertFalse(result["source_to_v5"]["class_gap_direction_preserved_along_head"])
        self.assertFalse(result["same_frozen_head_direction_strictly_separates_v5_train"])
        self.assertTrue(torch.isfinite(torch.tensor(result["v5_adaptation_train"]["class_mean_logit_gap"])))

    def test_rank_auc_handles_ties_without_pairwise_matrix(self):
        torch = self._torch()
        pos = torch.tensor([1.0, 2.0], dtype=torch.float32)
        neg = torch.tensor([1.0, 1.0], dtype=torch.float32)
        # One positive ties both negatives; the other beats both: (1 + 2) / 4 = 0.75.
        self.assertEqual(m._rank_auc(pos, neg), 0.75)

    def test_zero_head_weight_fails_closed(self):
        torch = self._torch()
        source_features, source_targets, v5_features, v5_targets, _head_weight = self._fixture()
        with self.assertRaises(m.MeterV5_2OError):
            m.head_axis_transfer_metrics_v1(
                source_features=source_features,
                source_targets=source_targets,
                v5_features=v5_features,
                v5_targets=v5_targets,
                head_weight=torch.zeros(64),
                head_bias=0.0,
                frozen_threshold=0.5,
            )

    def test_safety_boundary_is_read_only_and_nonselecting(self):
        safety = m.safety_boundary()
        self.assertFalse(safety["training"])
        self.assertFalse(safety["autograd_grad_used"])
        self.assertFalse(safety["backward"])
        self.assertEqual(safety["optimizer_steps"], 0)
        self.assertTrue(safety["checkpoint_read"])
        self.assertFalse(safety["checkpoint_write"])
        self.assertTrue(safety["image_read"])
        self.assertFalse(safety["runtime_threshold_tuning"])
        self.assertFalse(safety["alternative_threshold_evaluated"])
        self.assertFalse(safety["bias_parameter_selected"])
        self.assertFalse(safety["classifier_fit_performed"])
        self.assertFalse(safety["new_bbox"])
        self.assertFalse(safety["new_crop_geometry"])
        self.assertFalse(safety["new_spatial_heuristic"])
        self.assertFalse(safety["reserve_v5_train_opened"])
        self.assertFalse(safety["v5_validation_opened"])
        self.assertTrue(safety["final_holdout_locked"])
        self.assertTrue(safety["digit4_frozen"])
        self.assertFalse(safety["architecture_selected"])
        self.assertFalse(safety["residual_topology_selected"])
        self.assertFalse(safety["bias_only_repair_selected"])
        self.assertFalse(safety["repair_training_authorized"])

    def test_source_reuses_v52n_pixels_and_has_no_repair_path(self):
        source = inspect.getsource(m)
        self.assertIn("v52n._v5_surface", source)
        self.assertIn("v52n._historical_surface", source)
        self.assertIn("v52n._frozen_models", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("autograd.grad", source)
        self.assertNotIn("derive_staff_relative_slots_v1(", source)
        self.assertNotIn("detect_multistaff_geometry", source)
        self.assertNotIn("normalize_raster_page", source)


if __name__ == "__main__":
    unittest.main()
