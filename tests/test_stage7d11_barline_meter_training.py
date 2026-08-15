from __future__ import annotations

from collections.abc import Mapping
import inspect
import math
import unittest

import torch

import st_omr_training.stage7d11_barline_meter_training as d11
from st_omr_training.stage7d11_barline_meter_training import (
    BARLINE_MAX_PARAMETERS,
    EXPECTED_TASK_SPLIT_COUNTS,
    FROZEN_D11_CONFIG,
    METER_MAX_PARAMETERS,
    BarlineMetrics,
    MeterMetrics,
    Stage7D11TrainingError,
    _development_manifest_rows,
    _train_barline_batch,
    _train_meter_batch,
    acceptance_from_metrics,
    barline_loss,
    barline_target_mask,
    build_barline_refiner,
    build_meter_refiner,
    meter_loss,
    meter_target,
    refiner_model_fingerprint,
    stage7d11_profile_fingerprint,
)
from st_omr_training.stage7d9_structure_refinement_contract import (
    BARLINE_ROI,
    D9_ACCEPTANCE,
    EXPECTED_D7_STRUCTURE_STATE_SHA256,
    METER_ROI,
)
from st_omr_training.training_model import TrainingRuntimeError, count_trainable_parameters, model_state_sha256


class _HostileTestRecord(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        if key == "split": return "test"
        raise AssertionError(f"D11 touched forbidden TEST field: {key}")
    def __iter__(self):
        yield "split"; yield "image_path"
    def __len__(self) -> int: return 2
    def get(self, key: str, default: object = None) -> object:
        if key == "split": return "test"
        raise AssertionError(f"D11 touched forbidden TEST field: {key}")


class Stage7D11ProfileTests(unittest.TestCase):
    def test_profile_and_model_fingerprints_are_deterministic(self) -> None:
        profile = stage7d11_profile_fingerprint(); self.assertEqual(len(profile), 64); self.assertEqual(profile, stage7d11_profile_fingerprint())
        barline, meter = refiner_model_fingerprint("barline"), refiner_model_fingerprint("meter")
        self.assertEqual(len(barline), 64); self.assertEqual(len(meter), 64); self.assertNotEqual(barline, meter)

    def test_frozen_training_surface_matches_d9_d10(self) -> None:
        self.assertEqual(FROZEN_D11_CONFIG.batch_size, 32); self.assertEqual(FROZEN_D11_CONFIG.epochs, 8)
        self.assertEqual(EXPECTED_TASK_SPLIT_COUNTS, {"train": 9840, "validation": 1224})
        self.assertFalse(D9_ACCEPTANCE.core_model_mutation_allowed); self.assertEqual(D9_ACCEPTANCE.test_records, 0)

    def test_test_row_fails_before_other_field_access(self) -> None:
        with self.assertRaisesRegex(Stage7D11TrainingError, "sealed TEST"): _development_manifest_rows((_HostileTestRecord(),))

    def test_d7_structure_core_is_not_loaded_by_d11(self) -> None:
        source = inspect.getsource(d11); self.assertNotIn("DenseGeometrySpecialist", source); self.assertNotIn("structure_state_dict", source); self.assertEqual(len(EXPECTED_D7_STRUCTURE_STATE_SHA256), 64)


class Stage7D11TargetTests(unittest.TestCase):
    def test_barline_target_is_local_dense_mask(self) -> None:
        target = {"barline_segment": {"start": {"x": 70.0, "y": 25.0}, "end": {"x": 71.0, "y": 165.0}}}
        mask = barline_target_mask(target); self.assertEqual(tuple(mask.shape), (1, BARLINE_ROI.output_height, BARLINE_ROI.output_width)); self.assertGreater(float(mask.sum()), 0.0); self.assertTrue(bool((mask >= 0).all() and (mask <= 1).all()))

    def test_meter_none_and_visible_targets_are_distinct(self) -> None:
        class_index, bbox, positive = meter_target({"meter_class": "none", "meter_bbox": None}); self.assertEqual(class_index, 0); self.assertFalse(positive); self.assertEqual(float(bbox.sum()), 0.0)
        visible = {"meter_class": "3/4", "meter_bbox": {"x_min": 40.0, "y_min": 50.0, "x_max": 90.0, "y_max": 150.0}}
        class_index, bbox, positive = meter_target(visible); self.assertEqual(class_index, 2); self.assertTrue(positive); self.assertTrue(bool((bbox >= 0).all() and (bbox <= 1).all())); self.assertLess(float(bbox[0]), float(bbox[2])); self.assertLess(float(bbox[1]), float(bbox[3]))


class Stage7D11ModelTests(unittest.TestCase):
    def test_models_fit_frozen_parameter_budgets_and_shapes(self) -> None:
        barline, meter = build_barline_refiner(), build_meter_refiner(); barline_count, meter_count = count_trainable_parameters(barline), count_trainable_parameters(meter)
        self.assertGreater(barline_count, 0); self.assertGreater(meter_count, 0); self.assertLessEqual(barline_count, BARLINE_MAX_PARAMETERS); self.assertLessEqual(meter_count, METER_MAX_PARAMETERS); self.assertLessEqual(barline_count + meter_count, D9_ACCEPTANCE.max_total_new_trainable_parameters)
        with torch.no_grad():
            barline_logits = barline(torch.zeros((2, 1, BARLINE_ROI.output_height, BARLINE_ROI.output_width))); meter_logits, meter_boxes = meter(torch.zeros((2, 1, METER_ROI.output_height, METER_ROI.output_width)))
        self.assertEqual(tuple(barline_logits.shape), (2, 1, BARLINE_ROI.output_height, BARLINE_ROI.output_width)); self.assertEqual(tuple(meter_logits.shape), (2, 4)); self.assertEqual(tuple(meter_boxes.shape), (2, 4)); self.assertTrue(bool(torch.isfinite(barline_logits).all())); self.assertTrue(bool(torch.isfinite(meter_logits).all())); self.assertTrue(bool((meter_boxes >= 0).all() and (meter_boxes <= 1).all()))

    def test_losses_reject_nonfinite_or_out_of_range_targets(self) -> None:
        logits = torch.zeros((1, 1, 8, 8)); targets = torch.zeros_like(logits); targets[0, 0, 0, 0] = float("nan")
        with self.assertRaises(TrainingRuntimeError): barline_loss(logits, targets)
        with self.assertRaises(TrainingRuntimeError): meter_loss(torch.zeros((1, 4)), torch.zeros((1, 4)), torch.tensor([4]), torch.zeros((1, 4)), torch.tensor([False]), torch.ones(4))

    def test_validation_cannot_mutate_barline_model(self) -> None:
        model = build_barline_refiner(); optimizer = torch.optim.AdamW(model.parameters(), lr=0.001); images = torch.zeros((1, 1, BARLINE_ROI.output_height, BARLINE_ROI.output_width)); targets = torch.zeros_like(images); before = model_state_sha256(model)
        with self.assertRaisesRegex(TrainingRuntimeError, "only for TRAIN"): _train_barline_batch(model, images, targets, split="validation", optimizer=optimizer, config=FROZEN_D11_CONFIG)
        self.assertEqual(before, model_state_sha256(model))

    def test_validation_cannot_mutate_meter_model(self) -> None:
        model = build_meter_refiner(); optimizer = torch.optim.AdamW(model.parameters(), lr=0.001); images = torch.zeros((1, 1, METER_ROI.output_height, METER_ROI.output_width)); before = model_state_sha256(model)
        with self.assertRaisesRegex(TrainingRuntimeError, "only for TRAIN"): _train_meter_batch(model, images, torch.tensor([0]), torch.zeros((1, 4)), torch.tensor([False]), torch.ones(4), split="validation", optimizer=optimizer, config=FROZEN_D11_CONFIG)
        self.assertEqual(before, model_state_sha256(model))

    def test_train_batches_update_only_new_refiner_weights(self) -> None:
        barline = build_barline_refiner(); optimizer = torch.optim.AdamW(barline.parameters(), lr=0.001); images = torch.zeros((1, 1, BARLINE_ROI.output_height, BARLINE_ROI.output_width)); images[:, :, 20:170, 50:90] = 1.0; targets = torch.zeros_like(images); targets[:, :, 20:170, 70:73] = 1.0; before = model_state_sha256(barline); loss = _train_barline_batch(barline, images, targets, split="train", optimizer=optimizer, config=FROZEN_D11_CONFIG); self.assertTrue(math.isfinite(loss)); self.assertNotEqual(before, model_state_sha256(barline))
        meter = build_meter_refiner(); optimizer = torch.optim.AdamW(meter.parameters(), lr=0.001); meter_images = torch.zeros((1, 1, METER_ROI.output_height, METER_ROI.output_width)); meter_images[:, :, 40:150, 30:100] = 1.0; before = model_state_sha256(meter); loss = _train_meter_batch(meter, meter_images, torch.tensor([1]), torch.tensor([[0.1, 0.2, 0.4, 0.8]]), torch.tensor([True]), torch.ones(4), split="train", optimizer=optimizer, config=FROZEN_D11_CONFIG); self.assertTrue(math.isfinite(loss)); self.assertNotEqual(before, model_state_sha256(meter))


class Stage7D11AcceptanceTests(unittest.TestCase):
    def test_acceptance_uses_frozen_d9_gates(self) -> None:
        self.assertTrue(acceptance_from_metrics(BarlineMetrics(0.50, 0.70), MeterMetrics(0.80, 0.60)))
        self.assertFalse(acceptance_from_metrics(BarlineMetrics(0.4999, 0.99), MeterMetrics(0.99, 0.99)))

    def test_expected_optimizer_steps_are_precomputable(self) -> None:
        self.assertEqual(math.ceil(9840 / FROZEN_D11_CONFIG.batch_size) * FROZEN_D11_CONFIG.epochs, 2464)


if __name__ == "__main__": unittest.main()
