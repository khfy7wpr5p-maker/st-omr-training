from __future__ import annotations

import unittest

import torch

from st_omr_training.stage7d13_r2_rest_peak_alignment import (
    PEAK_ALIGNMENT_RADIUS,
    PEAK_ALIGNMENT_WEIGHT,
    Stage7D13R2RestPeakAlignmentError,
    local_gt_cell_peak_alignment_loss,
    peak_alignment_contract_payload,
)


class TestStage7D13R2RestPeakAlignment(unittest.TestCase):
    def test_frozen_contract_is_radius_two_and_weight_one(self) -> None:
        self.assertEqual(PEAK_ALIGNMENT_RADIUS, 2)
        self.assertEqual(PEAK_ALIGNMENT_WEIGHT, 1.0)
        payload = peak_alignment_contract_payload()
        self.assertEqual(payload["window"], 5)
        self.assertTrue(payload["same_class_only"])
        self.assertTrue(payload["exclude_other_positive_targets"])
        self.assertIsNone(payload["scheduler"])
        self.assertFalse(payload["production_checkpoint"])
        self.assertFalse(payload["test_authorized"])

    def test_exact_gt_peak_has_lower_loss_than_shifted_neighbor_peak(self) -> None:
        heatmap = torch.zeros((1, 3, 5, 5), dtype=torch.float32)
        heatmap[0, 1, 2, 2] = 1.0

        aligned = torch.zeros_like(heatmap)
        aligned[0, 1, 2, 2] = 4.0
        aligned[0, 1, 2, 3] = 1.0

        shifted = torch.zeros_like(heatmap)
        shifted[0, 1, 2, 2] = 1.0
        shifted[0, 1, 2, 3] = 4.0

        aligned_loss = local_gt_cell_peak_alignment_loss(aligned, heatmap)
        shifted_loss = local_gt_cell_peak_alignment_loss(shifted, heatmap)

        self.assertLess(float(aligned_loss.total), float(shifted_loss.total))
        self.assertEqual(aligned_loss.target_count, 1)
        self.assertEqual(aligned_loss.compared_target_count, 1)

    def test_other_positive_in_window_is_not_a_competitor(self) -> None:
        heatmap = torch.zeros((1, 3, 5, 5), dtype=torch.float32)
        heatmap[0, 0, 2, 2] = 1.0
        heatmap[0, 0, 2, 3] = 1.0

        baseline = torch.zeros_like(heatmap)
        baseline[0, 0, 2, 2] = 3.0
        baseline[0, 0, 2, 3] = 3.0

        boosted_other_positive = baseline.clone()
        boosted_other_positive[0, 0, 2, 3] = 30.0

        first = local_gt_cell_peak_alignment_loss(baseline, heatmap)
        second = local_gt_cell_peak_alignment_loss(boosted_other_positive, heatmap)

        # Raising another legitimate positive must not worsen the first target's
        # neighborhood competition; averaged loss can only improve or remain equal.
        self.assertLessEqual(float(second.total), float(first.total) + 1e-6)
        self.assertEqual(first.target_count, 2)
        self.assertEqual(first.compared_target_count, 2)

    def test_edge_target_is_supported_and_backward_is_finite(self) -> None:
        logits = torch.zeros(
            (1, 3, 4, 4), dtype=torch.float32, requires_grad=True
        )
        heatmap = torch.zeros_like(logits)
        heatmap[0, 2, 0, 0] = 1.0

        result = local_gt_cell_peak_alignment_loss(logits, heatmap)
        self.assertTrue(torch.isfinite(result.total))
        result.total.backward()
        self.assertIsNotNone(logits.grad)
        assert logits.grad is not None
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_all_negative_batch_is_graph_safe_zero(self) -> None:
        logits = torch.zeros(
            (2, 3, 4, 4), dtype=torch.float32, requires_grad=True
        )
        heatmap = torch.zeros_like(logits)

        result = local_gt_cell_peak_alignment_loss(logits, heatmap)
        self.assertEqual(result.target_count, 0)
        self.assertEqual(result.compared_target_count, 0)
        self.assertEqual(float(result.total), 0.0)
        result.total.backward()
        self.assertIsNotNone(logits.grad)

    def test_invalid_target_fails_closed(self) -> None:
        logits = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
        heatmap = torch.full_like(logits, 0.5)
        with self.assertRaisesRegex(Stage7D13R2RestPeakAlignmentError, "binary"):
            local_gt_cell_peak_alignment_loss(logits, heatmap)

    def test_invalid_radius_fails_closed(self) -> None:
        logits = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
        heatmap = torch.zeros_like(logits)
        with self.assertRaisesRegex(Stage7D13R2RestPeakAlignmentError, "radius"):
            local_gt_cell_peak_alignment_loss(logits, heatmap, radius=0)


if __name__ == "__main__":
    unittest.main()
