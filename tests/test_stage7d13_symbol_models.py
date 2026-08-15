from __future__ import annotations

import unittest

import torch

from st_omr_training.stage7d13_symbol_models import (
    Detection,
    GroundTruth,
    Stage7D13ModelError,
    acceptance_passed,
    build_symbol_model,
    compute_specialist_metrics,
    decode_detections,
    detector_loss,
    encode_detector_targets,
    model_profile_fingerprint,
)
from st_omr_training.stage7d13_symbol_training_contract import (
    MAX_PARAMETERS_PER_SPECIALIST,
    SPECIALIST_CLASSES,
)
from st_omr_training.training_model import count_trainable_parameters


def target(class_name: str, cx: float, cy: float, width: float = 8.0, height: float = 10.0):
    return {
        "class": class_name,
        "center": {"x": cx, "y": cy},
        "bbox": {
            "x_min": cx - width / 2.0,
            "y_min": cy - height / 2.0,
            "x_max": cx + width / 2.0,
            "y_max": cy + height / 2.0,
        },
    }


class Stage7D13SymbolModelTests(unittest.TestCase):
    def test_three_models_are_separate_bounded_stride4_detectors(self) -> None:
        fingerprints = set()
        for offset, specialist in enumerate(SPECIALIST_CLASSES):
            model = build_symbol_model(specialist, seed=713_013 + offset)
            count = count_trainable_parameters(model)
            self.assertGreater(count, 0)
            self.assertLessEqual(count, MAX_PARAMETERS_PER_SPECIALIST)
            images = torch.zeros((2, 1, 128, 512), dtype=torch.float32)
            outputs = model(images)
            self.assertEqual(outputs["heatmap_logits"].shape, (2, len(SPECIALIST_CLASSES[specialist]), 32, 128))
            self.assertEqual(outputs["bbox_size"].shape, (2, 2, 32, 128))
            self.assertEqual(outputs["center_offset"].shape, (2, 2, 32, 128))
            self.assertTrue(bool((outputs["bbox_size"] > 0).all()))
            self.assertTrue(bool(((outputs["center_offset"] >= 0) & (outputs["center_offset"] <= 1)).all()))
            fingerprints.add(model_profile_fingerprint(specialist))
        self.assertEqual(len(fingerprints), 3)

    def test_target_encoding_and_objective_are_finite_and_backward_safe(self) -> None:
        rows = [[target("open", 42.0, 50.0), target("filled", 130.5, 70.25)]]
        encoded = encode_detector_targets("notehead", rows)
        self.assertEqual(int(encoded.positive_mask.sum()), 2)
        model = build_symbol_model("notehead", seed=713_013)
        images = torch.full((1, 1, 128, 512), 0.5, dtype=torch.float32)
        outputs = model(images)
        loss = detector_loss("notehead", outputs, encoded)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))

    def test_class_agnostic_regression_collision_fails_closed(self) -> None:
        rows = [[
            target("open", 40.1, 40.1),
            target("filled", 41.2, 41.2),
        ]]
        with self.assertRaises(Stage7D13ModelError):
            encode_detector_targets("notehead", rows)

    def test_decoder_is_deterministic_and_uses_frozen_threshold(self) -> None:
        logits = torch.full((1, 2, 32, 128), -20.0, dtype=torch.float32)
        sizes = torch.ones((1, 2, 32, 128), dtype=torch.float32)
        offsets = torch.full((1, 2, 32, 128), 0.5, dtype=torch.float32)
        logits[0, 1, 10, 20] = 20.0
        sizes[0, 0, 10, 20] = 8.0
        sizes[0, 1, 10, 20] = 12.0
        outputs = {
            "heatmap_logits": logits,
            "bbox_size": sizes,
            "center_offset": offsets,
        }
        first = decode_detections("notehead", outputs)
        second = decode_detections("notehead", outputs)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 1)
        row = first[0][0]
        self.assertEqual(row.class_name, "filled")
        self.assertAlmostEqual(row.center_x, 82.0)
        self.assertAlmostEqual(row.center_y, 42.0)
        self.assertEqual(row.bbox, (78.0, 36.0, 86.0, 48.0))

    def test_metrics_are_one_for_exact_predictions_and_accept(self) -> None:
        predictions = [
            Detection("sharp", 0.99, 20.0, 30.0, (16.0, 24.0, 24.0, 36.0)),
            Detection("flat", 0.98, 60.0, 30.0, (56.0, 24.0, 64.0, 36.0)),
            Detection("natural", 0.97, 100.0, 30.0, (96.0, 24.0, 104.0, 36.0)),
        ]
        targets = [
            GroundTruth("sharp", 20.0, 30.0, (16.0, 24.0, 24.0, 36.0)),
            GroundTruth("flat", 60.0, 30.0, (56.0, 24.0, 64.0, 36.0)),
            GroundTruth("natural", 100.0, 30.0, (96.0, 24.0, 104.0, 36.0)),
        ]
        metrics = compute_specialist_metrics("accidental", [(predictions, targets)])
        self.assertEqual(metrics.class_aware_center_f1_4px, 1.0)
        self.assertEqual(metrics.class_aware_bbox_f1_iou50, 1.0)
        self.assertEqual(metrics.macro_class_f1, 1.0)
        self.assertTrue(acceptance_passed("accidental", metrics))

    def test_wrong_class_hurts_center_and_macro_metrics(self) -> None:
        predictions = [Detection("quarter", 0.9, 20.0, 20.0, (15.0, 15.0, 25.0, 25.0))]
        targets = [GroundTruth("half", 20.0, 20.0, (15.0, 15.0, 25.0, 25.0))]
        metrics = compute_specialist_metrics("rest", [(predictions, targets)])
        self.assertEqual(metrics.class_aware_center_f1_4px, 0.0)
        self.assertEqual(metrics.class_aware_bbox_f1_iou50, 0.0)
        self.assertEqual(metrics.macro_class_f1, 0.0)
        self.assertFalse(acceptance_passed("rest", metrics))


if __name__ == "__main__":
    unittest.main()
