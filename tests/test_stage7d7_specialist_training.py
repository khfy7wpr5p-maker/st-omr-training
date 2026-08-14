from __future__ import annotations

from collections.abc import Mapping
import math
import unittest

import torch

from st_omr_training.stage7d7_specialist_training import (
    FROZEN_D7_CONFIG,
    STAFF_CHANNELS,
    STRUCTURE_CHANNELS,
    Stage7D7TrainingError,
    _development_records,
    build_specialist_model,
    dense_geometry_loss,
    specialist_model_fingerprint,
    stage7d7_profile_fingerprint,
    target_masks_from_label,
    train_specialist_batch,
)
from st_omr_training.training_model import TrainingRuntimeError, model_state_sha256


class _HostileTestRecord(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        if key == "split":
            return "test"
        raise AssertionError(f"D7 touched forbidden TEST field: {key}")

    def __iter__(self):
        yield "split"
        yield "png_sha256"

    def __len__(self) -> int:
        return 2

    def get(self, key: str, default: object = None) -> object:
        if key == "split":
            return "test"
        raise AssertionError(f"D7 touched forbidden TEST field: {key}")


def _box(x0: float, y0: float, x1: float, y1: float) -> dict[str, float]:
    return {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}


def _line(x0: float, y0: float, x1: float, y1: float) -> dict[str, dict[str, float]]:
    return {"start": {"x": x0, "y": y0}, "end": {"x": x1, "y": y1}}


def _label() -> dict[str, object]:
    return {
        "image": {"width": 100, "height": 50},
        "geometry": {
            "staff_instances": [
                {
                    "staff_instance_bbox": _box(10, 10, 90, 30),
                    "five_staff_lines": [
                        _line(10, 10 + offset * 5, 90, 10 + offset * 5)
                        for offset in range(5)
                    ],
                }
            ],
            "systems": [{"system_bbox": _box(5, 5, 95, 35)}],
            "measures": [
                {
                    "measure_bbox": _box(20, 8, 80, 33),
                    "barline_segment": _line(80, 8, 80, 33),
                    "clef_g2_bbox": _box(22, 12, 28, 28),
                    "meter_bbox": _box(30, 12, 36, 26),
                    "meter_class": "3/4",
                }
            ],
        },
    }


class Stage7D7ProfileTests(unittest.TestCase):
    def test_frozen_profile_fingerprint_is_exact(self) -> None:
        self.assertEqual(
            stage7d7_profile_fingerprint(),
            "7b7fbc79c748da0f1195bc9273fe012e0b1128b3a1e491bb484653d47cb5201a",
        )
        self.assertEqual(FROZEN_D7_CONFIG.input_height, 96)
        self.assertEqual(FROZEN_D7_CONFIG.input_width, 512)
        self.assertEqual(FROZEN_D7_CONFIG.epochs, 8)
        self.assertEqual(FROZEN_D7_CONFIG.batch_size, 6)

    def test_staff_and_structure_models_are_separate_fingerprints(self) -> None:
        staff = specialist_model_fingerprint("staff")
        structure = specialist_model_fingerprint("structure")
        self.assertEqual(len(staff), 64)
        self.assertEqual(len(structure), 64)
        self.assertNotEqual(staff, structure)
        self.assertEqual(STAFF_CHANNELS, ("staff_lines", "staff_region"))
        self.assertEqual(len(STRUCTURE_CHANNELS), 7)

    def test_test_record_fails_before_any_other_field_access(self) -> None:
        with self.assertRaisesRegex(Stage7D7TrainingError, "sealed TEST"):
            _development_records((_HostileTestRecord(),))


class Stage7D7TargetTests(unittest.TestCase):
    def test_staff_targets_are_dense_and_nonempty(self) -> None:
        target = target_masks_from_label(_label(), "staff")
        self.assertEqual(tuple(target.shape), (2, 96, 512))
        self.assertGreater(float(target[0].sum()), 0.0)
        self.assertGreater(float(target[1].sum()), float(target[0].sum()))
        self.assertTrue(bool(torch.isfinite(target).all()))

    def test_structure_targets_keep_meter_class_isolated(self) -> None:
        target = target_masks_from_label(_label(), "structure")
        self.assertEqual(tuple(target.shape), (7, 96, 512))
        by_name = {name: target[index] for index, name in enumerate(STRUCTURE_CHANNELS)}
        for name in ("system_region", "measure_region", "barline", "clef_g2", "meter_3_4"):
            self.assertGreater(float(by_name[name].sum()), 0.0)
        self.assertEqual(float(by_name["meter_2_4"].sum()), 0.0)
        self.assertEqual(float(by_name["meter_4_4"].sum()), 0.0)


class Stage7D7ModelTests(unittest.TestCase):
    def test_models_have_expected_finite_output_shapes(self) -> None:
        staff = build_specialist_model("staff")
        structure = build_specialist_model("structure")
        image = torch.zeros((2, 1, 96, 512), dtype=torch.float32)
        with torch.no_grad():
            staff_logits = staff(image)
            structure_logits = structure(image)
        self.assertEqual(tuple(staff_logits.shape), (2, 2, 96, 512))
        self.assertEqual(tuple(structure_logits.shape), (2, 7, 96, 512))
        self.assertTrue(bool(torch.isfinite(staff_logits).all()))
        self.assertTrue(bool(torch.isfinite(structure_logits).all()))

    def test_loss_rejects_nan_and_out_of_range_targets(self) -> None:
        logits = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
        targets = torch.zeros_like(logits)
        targets[0, 0, 0, 0] = float("nan")
        with self.assertRaises(TrainingRuntimeError):
            dense_geometry_loss(logits, targets)
        targets = torch.zeros_like(logits)
        targets[0, 0, 0, 0] = 2.0
        with self.assertRaises(TrainingRuntimeError):
            dense_geometry_loss(logits, targets)

    def test_validation_cannot_mutate_model(self) -> None:
        model = build_specialist_model("staff")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        images = torch.zeros((1, 1, 96, 512), dtype=torch.float32)
        targets = torch.zeros((1, 2, 96, 512), dtype=torch.float32)
        before = model_state_sha256(model)
        with self.assertRaisesRegex(TrainingRuntimeError, "only for TRAIN"):
            train_specialist_batch(
                model,
                images,
                targets,
                split="validation",
                optimizer=optimizer,
            )
        self.assertEqual(before, model_state_sha256(model))

    def test_train_batch_updates_finite_model(self) -> None:
        model = build_specialist_model("staff")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        images = torch.zeros((1, 1, 96, 512), dtype=torch.float32)
        images[:, :, 20:70, 40:470] = 1.0
        targets = torch.zeros((1, 2, 96, 512), dtype=torch.float32)
        targets[:, 0, 30:32, 50:460] = 1.0
        targets[:, 1, 20:70, 40:470] = 1.0
        before = model_state_sha256(model)
        loss = train_specialist_batch(
            model,
            images,
            targets,
            split="train",
            optimizer=optimizer,
        )
        self.assertTrue(math.isfinite(loss))
        self.assertNotEqual(before, model_state_sha256(model))


if __name__ == "__main__":
    unittest.main()
