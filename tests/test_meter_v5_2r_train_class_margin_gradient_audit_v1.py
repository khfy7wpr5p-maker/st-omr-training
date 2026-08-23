import math
import unittest

from st_omr_training import meter_v5_2b_specialist_adaptation as v52b
from st_omr_training import meter_v5_2r_train_class_margin_gradient_audit_v1 as r


class TestMeterV5_2RTrainClassMarginGradientAuditV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, _nn = v52b._import_torch()

    def _synthetic_surfaces(self):
        torch = self.torch
        hist_pos = []
        for value in (2.0, 1.0, 0.2, -0.2):
            row = torch.zeros(64, dtype=torch.float64)
            row[0] = value
            row[1] = 1.0
            hist_pos.append(row)
        hist_neg = []
        for value in (-2.0, -1.0, -0.5, -0.1):
            row = torch.zeros(64, dtype=torch.float64)
            row[0] = value
            row[1] = -1.0
            hist_neg.append(row)
        historical_features = torch.stack(hist_pos + hist_neg)
        historical_targets = torch.tensor(
            [1.0] * len(hist_pos) + [0.0] * len(hist_neg),
            dtype=torch.float64,
        )

        v5_pos = []
        for value in (0.1, 0.2):
            row = torch.zeros(64, dtype=torch.float64)
            row[0] = value
            row[1] = 1.0
            v5_pos.append(row)
        v5_neg = []
        for value in (-0.1, -0.2, -0.3, -0.4, -0.5, -0.6):
            row = torch.zeros(64, dtype=torch.float64)
            row[0] = value
            row[1] = -1.0
            v5_neg.append(row)
        v5_features = torch.stack(v5_pos + v5_neg)
        v5_targets = torch.tensor(
            [1.0] * len(v5_pos) + [0.0] * len(v5_neg),
            dtype=torch.float64,
        )

        frozen_weight = torch.zeros(64, dtype=torch.float64)
        frozen_weight[0] = 1.0
        candidate_weight = frozen_weight.clone()
        candidate_weight[0] = 0.5
        candidate_weight[1] = 0.4

        return {
            "historical_features": historical_features,
            "historical_targets": historical_targets,
            "v5_features": v5_features,
            "v5_targets": v5_targets,
            "frozen_weight": frozen_weight,
            "candidate_weight": candidate_weight,
            "frozen_bias": 0.0,
            "threshold": 0.5,
        }

    def test_safety_boundary_is_train_only_read_only(self):
        boundary = r.safety_boundary()
        self.assertFalse(boundary["training"])
        self.assertFalse(boundary["autograd_grad_used"])
        self.assertFalse(boundary["backward"])
        self.assertEqual(boundary["optimizer_steps"], 0)
        self.assertFalse(boundary["checkpoint_write"])
        self.assertFalse(boundary["objective_changed"])
        self.assertFalse(boundary["new_objective_selected"])
        self.assertFalse(boundary["solver_settings_changed"])
        self.assertFalse(boundary["domain_weights_changed"])
        self.assertFalse(boundary["historical_validation_opened"])
        self.assertFalse(boundary["historical_validation_retention_report_read"])
        self.assertFalse(boundary["historical_validation_error_examples_read"])
        self.assertFalse(boundary["first30_opened"])
        self.assertFalse(boundary["v5_validation_opened"])
        self.assertTrue(boundary["final_holdout_locked"])
        self.assertTrue(boundary["digit4_frozen"])
        self.assertFalse(boundary["repair_selected"])
        self.assertFalse(boundary["repair_training_authorized"])

    def test_historical_positive_transition_matrix_uses_real_logits(self):
        surfaces = self._synthetic_surfaces()
        candidate = surfaces["candidate_weight"].clone()
        candidate[0] = 1.0
        candidate[1] = -0.3
        matrix = r._historical_positive_transition_matrix(
            features=surfaces["historical_features"],
            targets=surfaces["historical_targets"],
            frozen_weight=surfaces["frozen_weight"],
            candidate_weight=candidate,
            frozen_bias=0.0,
            threshold=0.5,
        )
        self.assertEqual(matrix["positive_count"], 4)
        self.assertEqual(matrix["correct_to_correct"], 2)
        self.assertEqual(matrix["correct_to_wrong"], 1)
        self.assertEqual(matrix["wrong_to_correct"], 0)
        self.assertEqual(matrix["wrong_to_wrong"], 1)

    def test_negative_margin_quantiles_are_real_tensor_calculations(self):
        surfaces = self._synthetic_surfaces()
        metrics = r.class_margin_gradient_audit_metrics_v1(**surfaces)
        hist = metrics["negative_margin_distribution"]["historical"]["frozen"]
        self.assertEqual(hist["count"], 4)
        self.assertAlmostEqual(hist["min"], 0.1)
        self.assertAlmostEqual(hist["max"], 2.0)
        self.assertGreater(hist["mean"], 0.0)

    def test_head_geometry_reports_norm_ratio_and_angle(self):
        torch = self.torch
        frozen = torch.zeros(64, dtype=torch.float64)
        frozen[0] = 3.0
        frozen[1] = 4.0
        candidate = torch.zeros(64, dtype=torch.float64)
        candidate[0] = 0.0
        candidate[1] = 5.0
        geometry = r._head_geometry(
            frozen_weight=frozen,
            candidate_weight=candidate,
        )
        self.assertAlmostEqual(geometry["frozen_weight_l2"], 5.0)
        self.assertAlmostEqual(geometry["candidate_weight_l2"], 5.0)
        self.assertAlmostEqual(geometry["delta_weight_l2"], math.sqrt(10.0))
        self.assertAlmostEqual(
            geometry["delta_over_frozen_l2"], math.sqrt(10.0) / 5.0
        )
        self.assertAlmostEqual(geometry["frozen_candidate_cosine"], 0.8)
        self.assertAlmostEqual(
            geometry["head_angle_change_degrees"],
            math.degrees(math.acos(0.8)),
        )

    def test_analytic_bce_gradient_matches_finite_difference(self):
        torch = self.torch
        x = torch.zeros((3, 64), dtype=torch.float64)
        x[:, 0] = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
        y = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
        w = torch.zeros(64, dtype=torch.float64)
        w[0] = 0.2

        mean_bce, grad = r._group_mean_bce_and_gradient(
            features=x,
            targets=y,
            weight=w,
            bias=0.1,
            name="finite-difference",
        )
        self.assertTrue(math.isfinite(mean_bce))

        eps = 1e-6
        wp = w.clone()
        wm = w.clone()
        wp[0] += eps
        wm[0] -= eps
        bp, _ = r._group_mean_bce_and_gradient(
            features=x,
            targets=y,
            weight=wp,
            bias=0.1,
            name="finite-difference-plus",
        )
        bm, _ = r._group_mean_bce_and_gradient(
            features=x,
            targets=y,
            weight=wm,
            bias=0.1,
            name="finite-difference-minus",
        )
        numeric = (bp - bm) / (2.0 * eps)
        self.assertAlmostEqual(float(grad[0].item()), numeric, places=7)

    def test_four_group_objective_coefficients_reconstruct_domain_weights(self):
        metrics = r.class_margin_gradient_audit_metrics_v1(
            **self._synthetic_surfaces()
        )
        frozen = metrics["gradient_and_bce_at_frozen_head"]
        groups = frozen["groups"]
        self.assertAlmostEqual(
            groups["v5_positive"]["objective_coefficient"]
            + groups["v5_negative"]["objective_coefficient"],
            0.5,
        )
        self.assertAlmostEqual(
            groups["historical_positive"]["objective_coefficient"]
            + groups["historical_negative"]["objective_coefficient"],
            0.5,
        )
        self.assertAlmostEqual(
            frozen["objective_reconstruction"]["coefficient_sum"], 1.0
        )
        for name in r.GROUPS:
            self.assertGreater(groups[name]["count"], 0)
            self.assertTrue(math.isfinite(groups[name]["mean_bce"]))
            self.assertTrue(
                math.isfinite(
                    groups[name]["analytic_mean_gradient"]["l2_norm"]
                )
            )

    def test_gradient_conflict_matrix_detects_opposite_groups(self):
        torch = self.torch
        g1 = torch.zeros(64, dtype=torch.float64)
        g2 = torch.zeros(64, dtype=torch.float64)
        g3 = torch.zeros(64, dtype=torch.float64)
        g4 = torch.zeros(64, dtype=torch.float64)
        g1[0] = 1.0
        g2[0] = -2.0
        g3[1] = 1.0
        g4[1] = -1.0
        matrix = r._pairwise_gradient_cosines(
            {
                "v5_positive": g1,
                "v5_negative": g2,
                "historical_positive": g3,
                "historical_negative": g4,
            }
        )
        self.assertAlmostEqual(
            matrix["v5_positive"]["v5_negative"], -1.0
        )
        self.assertAlmostEqual(
            matrix["historical_positive"]["historical_negative"], -1.0
        )
        self.assertAlmostEqual(
            matrix["v5_positive"]["historical_positive"], 0.0
        )

    def test_candidate_metrics_do_not_select_repair_or_mechanism(self):
        metrics = r.class_margin_gradient_audit_metrics_v1(
            **self._synthetic_surfaces()
        )
        self.assertTrue(metrics["descriptive_only"])
        self.assertFalse(metrics["mechanism_selected"])
        self.assertFalse(metrics["repair_selected"])
        self.assertFalse(metrics["new_objective_selected"])

    def test_nonfinite_surface_fails_closed(self):
        surfaces = self._synthetic_surfaces()
        surfaces["v5_features"] = surfaces["v5_features"].clone()
        surfaces["v5_features"][0, 0] = float("nan")
        with self.assertRaises(r.MeterV5_2RError):
            r.class_margin_gradient_audit_metrics_v1(**surfaces)

    def test_invalid_threshold_fails_closed(self):
        surfaces = self._synthetic_surfaces()
        surfaces["threshold"] = 1.0
        with self.assertRaises(r.MeterV5_2RError):
            r.class_margin_gradient_audit_metrics_v1(**surfaces)


if __name__ == "__main__":
    unittest.main()
