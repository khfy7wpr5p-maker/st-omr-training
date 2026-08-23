import unittest

from st_omr_training import meter_v5_2b_specialist_adaptation as v52b
from st_omr_training import meter_v5_2q_historical_positive_margin_audit_v1 as q


class TestMeterV5_2QHistoricalPositiveMarginAuditV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, _nn = v52b._import_torch()

    def test_safety_boundary_is_train_only_read_only(self):
        boundary = q.safety_boundary()
        self.assertFalse(boundary["training"])
        self.assertFalse(boundary["autograd_grad_used"])
        self.assertFalse(boundary["backward"])
        self.assertEqual(boundary["optimizer_steps"], 0)
        self.assertFalse(boundary["checkpoint_write"])
        self.assertFalse(boundary["historical_validation_opened"])
        self.assertFalse(boundary["historical_validation_retention_report_read"])
        self.assertFalse(boundary["historical_validation_error_examples_read"])
        self.assertFalse(boundary["first30_opened"])
        self.assertFalse(boundary["v5_validation_opened"])
        self.assertTrue(boundary["final_holdout_locked"])
        self.assertTrue(boundary["digit4_frozen"])
        self.assertFalse(boundary["repair_objective_selected"])
        self.assertFalse(boundary["repair_training_authorized"])

    def _synthetic_surfaces(self):
        torch = self.torch
        hist_pos = []
        for i in range(40):
            frozen_margin = 0.10 + 0.05 * i
            tail_pressure = -1.5 + 1.0 * (i / 39.0)
            row = torch.zeros(64, dtype=torch.float64)
            row[0] = frozen_margin
            row[1] = tail_pressure
            hist_pos.append(row)
        hist_neg = []
        for _ in range(20):
            row = torch.zeros(64, dtype=torch.float64)
            row[0] = -1.0
            hist_neg.append(row)
        historical_features = torch.stack(hist_pos + hist_neg)
        historical_targets = torch.tensor([1.0] * 40 + [0.0] * 20, dtype=torch.float64)

        v5_pos = []
        for i in range(20):
            row = torch.zeros(64, dtype=torch.float64)
            row[0] = 0.15 + 0.01 * i
            row[1] = 1.0
            v5_pos.append(row)
        v5_neg = []
        for i in range(20):
            row = torch.zeros(64, dtype=torch.float64)
            row[0] = -0.15 - 0.01 * i
            row[1] = -1.0
            v5_neg.append(row)
        v5_features = torch.stack(v5_pos + v5_neg)
        v5_targets = torch.tensor([1.0] * 20 + [0.0] * 20, dtype=torch.float64)

        frozen_weight = torch.zeros(64, dtype=torch.float64)
        frozen_weight[0] = 1.0
        candidate_weight = frozen_weight.clone()
        candidate_weight[1] = 1.0
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

    def test_real_tensor_geometry_detects_opposed_v5_gain_and_historical_loss(self):
        metrics = q.positive_margin_audit_metrics_v1(**self._synthetic_surfaces())
        relation = metrics["directional_relation"]
        self.assertLess(relation["historical_positive_mean_logit_shift"], 0.0)
        self.assertGreater(relation["v5_positive_mean_logit_shift"], 0.0)
        self.assertGreater(relation["v5_all_classification_margin_mean_change"], 0.0)
        self.assertTrue(
            relation["v5_positive_gain_and_historical_positive_loss_have_opposite_sign"]
        )
        hist = metrics["historical_positive_margin"]
        self.assertEqual(hist["fraction_margin_decreased"], 1.0)
        self.assertLess(hist["candidate"]["p50"], hist["frozen"]["p50"])
        self.assertGreater(
            metrics["v5_train_margin"]["fraction_classification_margin_improved"],
            0.99,
        )

    def test_low_margin_rank_tail_can_show_stronger_narrowing(self):
        metrics = q.positive_margin_audit_metrics_v1(**self._synthetic_surfaces())
        bins = metrics["historical_positive_margin"]["shift_by_frozen_margin_rank"]
        self.assertLess(
            bins["bottom_10pct"]["margin_delta_mean"],
            bins["top_10pct"]["margin_delta_mean"],
        )
        self.assertEqual(bins["bottom_10pct"]["fraction_margin_decreased"], 1.0)

    def test_quantiles_are_computed_from_tensor_values(self):
        torch = self.torch
        values = torch.arange(1, 101, dtype=torch.float64)
        summary = q._quantile_summary(values, name="synthetic")
        self.assertEqual(summary["count"], 100)
        self.assertAlmostEqual(summary["min"], 1.0)
        self.assertAlmostEqual(summary["max"], 100.0)
        self.assertAlmostEqual(summary["p50"], 50.5)
        self.assertAlmostEqual(summary["mean"], 50.5)

    def test_nonfinite_feature_fails_closed(self):
        surfaces = self._synthetic_surfaces()
        surfaces["historical_features"] = surfaces["historical_features"].clone()
        surfaces["historical_features"][0, 0] = float("nan")
        with self.assertRaises(q.MeterV5_2QError):
            q.positive_margin_audit_metrics_v1(**surfaces)

    def test_candidate_surface_accepts_head_weight_only_change(self):
        torch = self.torch
        frozen = v52b._build_digit_model().cpu()
        candidate = v52b._build_digit_model().cpu()
        candidate.load_state_dict(frozen.state_dict(), strict=True)
        with torch.no_grad():
            candidate.head.weight.reshape(-1)[0] += 0.125
        evidence = q.verify_candidate_frozen_surface_v1(
            frozen_model=frozen,
            candidate_model=candidate,
        )
        self.assertEqual(evidence["changed_state_keys"], ["head.weight"])
        self.assertTrue(evidence["only_head_weight_changed"])
        self.assertTrue(evidence["backbone_bit_identical"])
        self.assertTrue(evidence["head_bias_bit_identical"])
        self.assertEqual(evidence["head_weight_parameter_count"], 64)

    def test_candidate_surface_rejects_bias_mutation(self):
        torch = self.torch
        frozen = v52b._build_digit_model().cpu()
        candidate = v52b._build_digit_model().cpu()
        candidate.load_state_dict(frozen.state_dict(), strict=True)
        with torch.no_grad():
            candidate.head.bias.reshape(-1)[0] += 0.125
        with self.assertRaises(q.MeterV5_2QError):
            q.verify_candidate_frozen_surface_v1(
                frozen_model=frozen,
                candidate_model=candidate,
            )

    def test_invalid_threshold_fails_closed(self):
        surfaces = self._synthetic_surfaces()
        surfaces["threshold"] = 1.0
        with self.assertRaises(q.MeterV5_2QError):
            q.positive_margin_audit_metrics_v1(**surfaces)


if __name__ == "__main__":
    unittest.main()