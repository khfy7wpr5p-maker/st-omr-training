from __future__ import annotations

from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image, ImageDraw
import torch

from st_omr_training.meter_v4_1_numerator_specialist import (
    EXPECTED_PARAMETER_COUNT_V4_1,
    EXPECTED_V4_0_EXPERIMENT,
    EXPECTED_V4_0_RESULT_SCHEMA,
    FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1,
    MeterV4_1Error,
    NumeratorRecordV4_1,
    PredictionV4_1,
    build_augmented_train_batch_v4_1,
    build_model_v4_1,
    config_fingerprint_v4_1,
    decision_v4_1,
    summarize_predictions_v4_1,
    translate_ink_v4_1,
    verify_parent_artifact_v4_1,
)
from st_omr_training.training_model import count_trainable_parameters, model_state_sha256


class MeterV41NumeratorSpecialistTests(unittest.TestCase):
    def _records(self) -> tuple[NumeratorRecordV4_1, ...]:
        rows = []
        counter = 0
        for class_name in ("2", "3", "4"):
            for class_index in range(9):
                fold = class_index % 3
                record_id = sha256(f"record-{counter}".encode("ascii")).hexdigest()
                crop_sha = sha256(f"crop-{counter}".encode("ascii")).hexdigest()
                rows.append(
                    NumeratorRecordV4_1(
                        record_id=record_id,
                        family_id=f"family-{counter:02d}",
                        numerator_class=class_name,
                        fold=fold,
                        crop_png_sha256=crop_sha,
                    )
                )
                counter += 1
        return tuple(rows)

    def test_model_shape_parameter_ceiling_and_seed_determinism(self) -> None:
        first = build_model_v4_1(0)
        second = build_model_v4_1(0)
        third = build_model_v4_1(1)
        self.assertEqual(count_trainable_parameters(first), EXPECTED_PARAMETER_COUNT_V4_1)
        self.assertEqual(model_state_sha256(first), model_state_sha256(second))
        self.assertNotEqual(model_state_sha256(first), model_state_sha256(third))
        logits = first(torch.zeros((2, 1, 64, 64), dtype=torch.float32))
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_config_fingerprint_is_stable_hex(self) -> None:
        fingerprint = config_fingerprint_v4_1()
        self.assertEqual(len(fingerprint), 64)
        self.assertTrue(all(ch in "0123456789abcdef" for ch in fingerprint))
        self.assertEqual(FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1.epochs, 160)

    def test_translation_does_not_wrap(self) -> None:
        image = torch.zeros((1, 64, 64), dtype=torch.float32)
        image[0, 0, 0] = 1.0
        shifted = translate_ink_v4_1(image, dx=2, dy=2)
        self.assertEqual(float(shifted[0, 2, 2]), 1.0)
        self.assertEqual(float(shifted[0, 0, 0]), 0.0)
        self.assertEqual(float(shifted[0, -1, -1]), 0.0)
        with self.assertRaises(MeterV4_1Error):
            translate_ink_v4_1(image, dx=1, dy=0)

    def test_augmented_batch_is_exact_balanced_162_and_family_disjoint(self) -> None:
        records = self._records()
        crops = {
            row.record_id: torch.zeros((1, 64, 64), dtype=torch.float32)
            for row in records
        }
        for index, row in enumerate(records):
            crops[row.record_id][0, 10 + (index % 10), 10 + (index % 10)] = 1.0
        images, labels, origins = build_augmented_train_batch_v4_1(
            records, crops, heldout_fold=0
        )
        self.assertEqual(tuple(images.shape), (162, 1, 64, 64))
        self.assertEqual(tuple(labels.shape), (162,))
        self.assertEqual(Counter(labels.tolist()), Counter({0: 54, 1: 54, 2: 54}))
        heldout_ids = {row.record_id for row in records if row.fold == 0}
        self.assertTrue(heldout_ids.isdisjoint(origins))
        self.assertEqual(len(set(origins)), 18)

    def test_summary_and_gate_require_improvement_over_v4_0(self) -> None:
        predictions = []
        records = self._records()
        for row in records:
            predictions.append(
                PredictionV4_1(
                    record_id=row.record_id,
                    family_id=row.family_id,
                    fold=row.fold,
                    true_class=row.numerator_class,
                    predicted_class=row.numerator_class,
                    logits=(1.0, 0.0, -1.0),
                    probabilities=(0.7, 0.2, 0.1),
                )
            )
        summary = summarize_predictions_v4_1(predictions)
        self.assertEqual(summary.accuracy, 1.0)
        self.assertTrue(decision_v4_1(summary).strong_signal)

        # Reproduce the 25/27 V4-0 baseline: one 2->3 and one 3->2.
        changed = list(predictions)
        idx2 = next(i for i, row in enumerate(changed) if row.true_class == "2")
        idx3 = next(i for i, row in enumerate(changed) if row.true_class == "3")
        for index, pred in ((idx2, "3"), (idx3, "2")):
            row = changed[index]
            changed[index] = PredictionV4_1(
                record_id=row.record_id,
                family_id=row.family_id,
                fold=row.fold,
                true_class=row.true_class,
                predicted_class=pred,
                logits=row.logits,
                probabilities=row.probabilities,
            )
        baseline = summarize_predictions_v4_1(changed)
        decision = decision_v4_1(baseline)
        self.assertAlmostEqual(baseline.accuracy, 25 / 27)
        self.assertFalse(decision.strong_signal)
        self.assertIn("OOF_ACCURACY_BELOW_26_OF_27", decision.reasons)

    def _write_parent_fixture(self, root: Path) -> tuple[str, str]:
        crop_dir = root / "crops"
        crop_dir.mkdir(parents=True)
        crop_rows = []
        counter = 0
        for class_name in ("2", "3", "4"):
            for class_index in range(9):
                record_id = sha256(f"fixture-record-{counter}".encode("ascii")).hexdigest()
                family = f"fixture-family-{counter:02d}"
                image = Image.new("L", (64, 64), 255)
                draw = ImageDraw.Draw(image)
                draw.rectangle((10 + class_index, 10, 20 + class_index, 30), fill=0)
                stream = BytesIO()
                image.save(stream, format="PNG", optimize=False, compress_level=9)
                raw = stream.getvalue()
                (crop_dir / f"{record_id}.png").write_bytes(raw)
                crop_rows.append(
                    {
                        "record_id": record_id,
                        "family_id": family,
                        "meter_class": f"{class_name}/4",
                        "numerator_class": class_name,
                        "fold": class_index % 3,
                        "source_image_sha256": sha256(f"source-{counter}".encode("ascii")).hexdigest(),
                        "replayed_roi_transform": {},
                        "mapped_meter_bbox": {},
                        "numerator_crop_bounds": {},
                        "crop_png_sha256": sha256(raw).hexdigest(),
                        "ink_fraction": 0.1,
                    }
                )
                counter += 1
        result = {
            "schema": EXPECTED_V4_0_RESULT_SCHEMA,
            "experiment": EXPECTED_V4_0_EXPERIMENT,
            "repository_sha": "a" * 64,
            "audit_surface": {
                "teacher_positive_train_records": 27,
                "teacher_positive_validation_records": 9,
                "teacher_adaptation_validation_evaluated": False,
                "teacher_adaptation_validation_images_decoded": 0,
                "none_tasks_used": 0,
                "d10_opened": False,
                "test_opened": False,
            },
            "classifier": {"trainable_parameters": 0, "optimizer_steps": 0},
            "configuration": {},
            "contact_sheet_sha256": "b" * 64,
            "crop_records": crop_rows,
            "d11_checkpoint_loaded": False,
            "decision": {"name": "REPRESENTATION_SIGNAL_STRONG", "strong_signal": True, "reasons": []},
            "oof_predictions": [],
            "oof_summary": {
                "record_count": 27,
                "accuracy": 25 / 27,
                "macro_f1": 25 / 27,
                "per_class_recall": {"2": 8 / 9, "3": 8 / 9, "4": 1.0},
                "confusion": [[8, 1, 0], [1, 8, 0], [0, 0, 9]],
            },
            "optimizer_steps": 0,
            "production_promotion_authorized": False,
            "repository_sha": "a" * 64,
            "resolver_connected": False,
            "runtime_connected": False,
            "source_provenance": {},
            "v3_checkpoint_loaded": False,
        }
        raw = json.dumps(
            result, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
        (root / "result.json").write_bytes(raw)
        result_sha = sha256(raw).hexdigest()
        (root / "COMPLETE").write_bytes(f"{result_sha}  result.json\n".encode("ascii"))
        return result_sha, "a" * 64

    def test_parent_verifier_binds_receipt_safety_and_crop_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_sha, repository_binding = self._write_parent_fixture(root)
            with mock.patch(
                "st_omr_training.meter_v4_1_numerator_specialist.EXPECTED_V4_0_RESULT_SHA256",
                result_sha,
            ), mock.patch(
                "st_omr_training.meter_v4_1_numerator_specialist.EXPECTED_V4_0_REPOSITORY_BINDING",
                repository_binding,
            ):
                verified = verify_parent_artifact_v4_1(root)
            self.assertEqual(len(verified.records), 27)
            self.assertEqual(Counter(row.numerator_class for row in verified.records), Counter({"2": 9, "3": 9, "4": 9}))

    def test_parent_verifier_rejects_crop_mutation_and_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_sha, repository_binding = self._write_parent_fixture(root)
            result = json.loads((root / "result.json").read_text(encoding="ascii"))
            result["audit_surface"]["test_opened"] = True
            raw = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
            (root / "result.json").write_bytes(raw)
            mutated_sha = sha256(raw).hexdigest()
            (root / "COMPLETE").write_bytes(f"{mutated_sha}  result.json\n".encode("ascii"))
            with mock.patch(
                "st_omr_training.meter_v4_1_numerator_specialist.EXPECTED_V4_0_RESULT_SHA256",
                mutated_sha,
            ), mock.patch(
                "st_omr_training.meter_v4_1_numerator_specialist.EXPECTED_V4_0_REPOSITORY_BINDING",
                repository_binding,
            ):
                with self.assertRaises(MeterV4_1Error):
                    verify_parent_artifact_v4_1(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_sha, repository_binding = self._write_parent_fixture(root)
            first_crop = next((root / "crops").glob("*.png"))
            first_crop.write_bytes(first_crop.read_bytes() + b"mutated")
            with mock.patch(
                "st_omr_training.meter_v4_1_numerator_specialist.EXPECTED_V4_0_RESULT_SHA256",
                result_sha,
            ), mock.patch(
                "st_omr_training.meter_v4_1_numerator_specialist.EXPECTED_V4_0_REPOSITORY_BINDING",
                repository_binding,
            ):
                with self.assertRaises(MeterV4_1Error):
                    verify_parent_artifact_v4_1(root)


if __name__ == "__main__":
    unittest.main()
