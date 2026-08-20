from __future__ import annotations

import unittest

import torch

from st_omr_training.stage7d13_r2_rest_balanced_loss import (
    DIAGNOSTIC_MAX_OPTIMIZER_STEPS,
    Stage7D13R2RestBalancedLossError,
    balanced_rest_heatmap_focal_loss,
    r1_heatmap_focal_loss_for_diagnostic_comparison,
    rest_positive_class_weights,
    select_train_only_diagnostic_partition,
)


class TestStage7D13R2RestBalancedLoss(unittest.TestCase):
    def test_rest_positive_weights_preserve_r1_train_only_policy(self) -> None:
        weights = rest_positive_class_weights()
        self.assertEqual(set(weights), {"half", "quarter", "eighth"})
        self.assertAlmostEqual(sum(weights.values()) / 3.0, 1.0, places=7)
        self.assertGreater(weights["half"], weights["quarter"])
        self.assertGreater(weights["quarter"], weights["eighth"])

    def test_balanced_loss_separately_normalizes_negative_cells(self) -> None:
        # One positive quarter-rest cell and otherwise identical background logits.
        logits_small = torch.zeros((1, 3, 1, 2), dtype=torch.float32)
        heatmap_small = torch.zeros_like(logits_small)
        heatmap_small[0, 1, 0, 0] = 1.0

        logits_large = torch.zeros((1, 3, 1, 20), dtype=torch.float32)
        heatmap_large = torch.zeros_like(logits_large)
        heatmap_large[0, 1, 0, 0] = 1.0

        balanced_small = balanced_rest_heatmap_focal_loss(
            logits_small, heatmap_small
        )
        balanced_large = balanced_rest_heatmap_focal_loss(
            logits_large, heatmap_large
        )
        self.assertAlmostEqual(
            float(balanced_small.negative_term),
            float(balanced_large.negative_term),
            places=7,
        )
        self.assertAlmostEqual(
            float(balanced_small.total),
            float(balanced_large.total),
            places=7,
        )

        r1_small = r1_heatmap_focal_loss_for_diagnostic_comparison(
            logits_small, heatmap_small
        )
        r1_large = r1_heatmap_focal_loss_for_diagnostic_comparison(
            logits_large, heatmap_large
        )
        self.assertGreater(float(r1_large), float(r1_small) * 5.0)

    def test_all_negative_batch_remains_trainable(self) -> None:
        logits = torch.zeros((2, 3, 2, 2), dtype=torch.float32, requires_grad=True)
        heatmap = torch.zeros_like(logits)
        result = balanced_rest_heatmap_focal_loss(logits, heatmap)
        self.assertEqual(result.positive_count, 0)
        self.assertEqual(result.negative_count, logits.numel())
        self.assertEqual(float(result.positive_term), 0.0)
        self.assertGreater(float(result.negative_term), 0.0)
        result.total.backward()
        self.assertIsNotNone(logits.grad)
        assert logits.grad is not None
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_invalid_non_binary_heatmap_fails_closed(self) -> None:
        logits = torch.zeros((1, 3, 1, 1), dtype=torch.float32)
        heatmap = torch.full_like(logits, 0.5)
        with self.assertRaisesRegex(Stage7D13R2RestBalancedLossError, "binary"):
            balanced_rest_heatmap_focal_loss(logits, heatmap)

    def test_partition_uses_train_only_and_is_deterministic(self) -> None:
        rows = [
            {"split": "train", "record_id": f"{index:064x}"}
            for index in range(12)
        ] + [
            {"split": "validation", "record_id": "f" * 64}
        ]
        first = select_train_only_diagnostic_partition(
            rows,
            optimization_records=6,
            evaluation_records=3,
        )
        second = select_train_only_diagnostic_partition(
            list(reversed(rows)),
            optimization_records=6,
            evaluation_records=3,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first.optimization_record_ids), 6)
        self.assertEqual(len(first.evaluation_record_ids), 3)
        self.assertEqual(first.validation_seen, 1)
        self.assertFalse(first.test_opened)
        self.assertTrue(
            set(first.optimization_record_ids).isdisjoint(
                first.evaluation_record_ids
            )
        )

    def test_partition_fails_closed_on_test(self) -> None:
        with self.assertRaisesRegex(Stage7D13R2RestBalancedLossError, "sealed TEST"):
            select_train_only_diagnostic_partition(
                [
                    {"split": "train", "record_id": "a" * 64},
                    {"split": "test", "record_id": "b" * 64},
                ],
                optimization_records=1,
                evaluation_records=1,
            )

    def test_frozen_diagnostic_budget_is_384_steps(self) -> None:
        self.assertEqual(DIAGNOSTIC_MAX_OPTIMIZER_STEPS, 384)


if __name__ == "__main__":
    unittest.main()
