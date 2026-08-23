import inspect
import unittest

from st_omr_training import meter_v5_2n_frozen_feature_transfer_audit_v1 as m
from st_omr_training import meter_v5_2b_specialist_adaptation as v52b


class TestMeterV52NFrozenFeatureTransferAuditV1(unittest.TestCase):
    def _torch(self):
        torch, _nn = v52b._import_torch()
        return torch

    def _aligned_fixture(self):
        torch = self._torch()
        source_neg = torch.zeros((4, 64), dtype=torch.float32)
        source_pos = torch.zeros((4, 64), dtype=torch.float32)
        source_neg[:, 0] = -2.0
        source_pos[:, 0] = 2.0
        source_features = torch.cat((source_neg, source_pos), dim=0)
        source_targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)

        v5_neg = torch.zeros((2, 64), dtype=torch.float32)
        v5_pos = torch.zeros((2, 64), dtype=torch.float32)
        v5_neg[:, 0] = -1.0
        v5_pos[:, 0] = 1.0
        v5_features = torch.cat((v5_neg, v5_pos), dim=0)
        v5_targets = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
        head_weight = torch.zeros(64, dtype=torch.float32)
        head_weight[0] = 1.0
        return source_features, source_targets, v5_features, v5_targets, head_weight

    def test_aligned_features_have_positive_transfer_geometry(self):
        source_features, source_targets, v5_features, v5_targets, head_weight = self._aligned_fixture()
        result = m.feature_transfer_metrics_v1(
            source_features=source_features,
            source_targets=source_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
            head_weight=head_weight,
        )
        nearest = result["nearest_historical_centroid"]
        separation = result["class_separation"]
        self.assertEqual(nearest["positive_correct_fraction"], 1.0)
        self.assertEqual(nearest["negative_correct_fraction"], 1.0)
        self.assertEqual(nearest["overall_correct_fraction"], 1.0)
        self.assertEqual(nearest["tie_count"], 0)
        self.assertGreater(nearest["overall_margin"]["median"], 0.0)
        self.assertAlmostEqual(separation["source_v5_delta_cosine"], 1.0, places=12)
        self.assertAlmostEqual(separation["frozen_head_source_delta_cosine"], 1.0, places=12)
        self.assertAlmostEqual(separation["frozen_head_v5_delta_cosine"], 1.0, places=12)

    def test_reversed_v5_separation_is_detected_without_fitting(self):
        torch = self._torch()
        source_features, source_targets, v5_features, v5_targets, head_weight = self._aligned_fixture()
        reversed_features = v5_features.clone()
        reversed_features[:, 0] *= -1.0
        result = m.feature_transfer_metrics_v1(
            source_features=source_features,
            source_targets=source_targets,
            v5_features=reversed_features,
            v5_targets=v5_targets,
            head_weight=head_weight,
        )
        self.assertEqual(result["nearest_historical_centroid"]["overall_correct_fraction"], 0.0)
        self.assertAlmostEqual(result["class_separation"]["source_v5_delta_cosine"], -1.0, places=12)
        self.assertAlmostEqual(result["class_separation"]["frozen_head_v5_delta_cosine"], -1.0, places=12)
        self.assertTrue(torch.isfinite(torch.tensor(result["nearest_historical_centroid"]["overall_margin"]["mean"])))

    def test_wrong_feature_dimension_fails_closed(self):
        torch = self._torch()
        with self.assertRaises(m.MeterV5_2NError):
            m.feature_transfer_metrics_v1(
                source_features=torch.zeros((4, 63)),
                source_targets=torch.tensor([0, 0, 1, 1], dtype=torch.float32),
                v5_features=torch.zeros((4, 63)),
                v5_targets=torch.tensor([0, 0, 1, 1], dtype=torch.float32),
                head_weight=torch.ones(63),
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
        self.assertFalse(safety["threshold_tuning"])
        self.assertFalse(safety["new_bbox"])
        self.assertFalse(safety["new_crop_geometry"])
        self.assertFalse(safety["new_spatial_heuristic"])
        self.assertFalse(safety["old_d11_glyph_window_reused"])
        self.assertFalse(safety["reserve_v5_train_opened"])
        self.assertFalse(safety["v5_validation_opened"])
        self.assertTrue(safety["final_holdout_locked"])
        self.assertTrue(safety["digit4_frozen"])
        self.assertFalse(safety["architecture_selected"])
        self.assertFalse(safety["residual_topology_selected"])
        self.assertFalse(safety["repair_training_authorized"])

    def test_source_uses_only_frozen_existing_pixel_contracts(self):
        source = inspect.getsource(m)
        self.assertIn("v52b._tensor_from_crop", source)
        self.assertIn("ret_legacy._historical_canvas_from_bbox", source)
        self.assertIn("model.features", source)
        self.assertIn("torch.no_grad()", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("autograd.grad", source)
        self.assertNotIn("derive_staff_relative_slots_v1(", source)
        self.assertNotIn("detect_multistaff_geometry", source)
        self.assertNotIn("normalize_raster_page", source)


if __name__ == "__main__":
    unittest.main()
