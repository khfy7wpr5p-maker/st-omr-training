from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from st_omr_training.meter_teacher_gold_admission_v1 import (
    EXPECTED_CLASS_SPLIT_COUNTS,
    EXPECTED_SPLIT_COUNTS,
    MeterTeacherGoldAdmissionError,
    build_meter_teacher_gold_bundle_v1,
    checkpoint_loading_allowed,
    optimizer_step_allowed,
    production_promotion_allowed,
    sealed_test_access_allowed,
    verify_meter_teacher_gold_bundle_v1,
)


def _source_data_uri(label: str) -> str:
    image = Image.new("L", (420, 120), 255)
    draw = ImageDraw.Draw(image)
    for y in (30, 42, 54, 66, 78):
        draw.line((0, y, 419, y), fill=55, width=1)
    numerator = int(label[0])
    draw.rectangle((55, 28, 75, 50), fill=35 + numerator)
    draw.rectangle((55, 56, 75, 78), fill=39)
    out = BytesIO()
    image.save(out, format="PNG", optimize=False, compress_level=9)
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def _pilot_and_choices() -> tuple[dict[str, object], dict[str, object]]:
    tasks: list[dict[str, object]] = []
    answers: list[dict[str, object]] = []
    reviewed = "2026-08-20T06:45:42Z"
    source_index = 0
    for meter in ("2/4", "3/4", "4/4"):
        for package, count in (("aa", 4), ("ab", 8)):
            for package_index in range(count):
                source_index += 1
                source_id = f"source-{meter[0]}-{package}-{package_index:02d}"
                family = f"family-{meter[0]}-{package}-{package_index:02d}"
                data_uri = _source_data_uri(meter)
                for kind, expected, suffix in (
                    ("positive", meter, "positive"),
                    ("none", "none", "none-m2"),
                ):
                    task_id = f"{source_id}__{suffix}"
                    tasks.append(
                        {
                            "source_id": source_id,
                            "source_meter": meter,
                            "family_key": family,
                            "package": package,
                            "split": "train",
                            "key_count": 0,
                            "measure_count": 4,
                            "image_data_uri": data_uri,
                            "task_id": task_id,
                            "kind": kind,
                            "expected_class": expected,
                        }
                    )
                    answers.append(
                        {
                            "task_id": task_id,
                            "source_id": source_id,
                            "split": "train",
                            "kind": kind,
                            "status": "accepted",
                            "label": expected,
                            "expected_class": expected,
                            "label_confirmed": True,
                            "crop_usable": True,
                            "bbox": {"x": 48, "y": 20, "w": 36, "h": 68} if kind == "positive" else None,
                            "anchor_x": None if kind == "positive" else 130,
                            "roi_crop_box": {"x": 0, "y": 0, "w": 240, "h": 120}
                            if kind == "positive"
                            else {"x": 120, "y": 0, "w": 240, "h": 120},
                            "notes": "",
                            "reviewed_at": reviewed,
                        }
                    )
    pilot = {
        "schema": "st-omr-meter-teacher-gold-pilot-data-v1",
        "generated_date": "2026-08-20",
        "source": "METER_V1/01_REVIEW/train",
        "selection": {
            "positive_total": 36,
            "per_class": {"2/4": 12, "3/4": 12, "4/4": 12},
            "none_total": 36,
            "package_policy": "4 aa + 8 ab per positive class",
            "test_opened": False,
        },
        "tasks": tasks,
    }
    choices = {
        "schema": "st-omr-meter-teacher-gold-pilot-v1-choices",
        "pilot_version": "st-omr-meter-teacher-gold-pilot-v1",
        "generated_at": reviewed,
        "source": "METER_V1/01_REVIEW/train",
        "test_opened": False,
        "task_count": 72,
        "answered_count": 72,
        "answers": answers,
    }
    return pilot, choices


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


class MeterTeacherGoldAdmissionV1Tests(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path, Path, Path]:
        pilot, choices = _pilot_and_choices()
        pilot_path = root / "pilot.json"
        choices_path = root / "choices.json"
        permission_path = root / "permission.json"
        privacy_path = root / "privacy.json"
        pilot_path.write_bytes(_canonical(pilot))
        choices_path.write_bytes(_canonical(choices))
        permission_path.write_bytes(
            _canonical(
                {
                    "schema_version": "st-omr-meter-training-permission-evidence-v1",
                    "decision": "approved",
                    "allowed_use": "offline-meter-real-domain-adaptation-pilot",
                    "dataset_scope": "METER_V1/TRAIN/teacher-gold-pilot-72",
                    "automatic_learning": False,
                    "production_promotion_authorized": False,
                    "test_access_authorized": False,
                    "approved_at": "2026-08-20T06:45:42Z",
                }
            )
        )
        privacy_path.write_bytes(
            _canonical(
                {
                    "schema_version": "st-omr-meter-privacy-review-evidence-v1",
                    "decision": "approved",
                    "review_scope": "METER_V1/TRAIN/teacher-gold-pilot-72",
                    "personal_data_detected": False,
                    "redistribution_allowed": False,
                    "test_opened": False,
                    "reviewed_at": "2026-08-20T06:58:00Z",
                }
            )
        )
        return pilot_path, choices_path, permission_path, privacy_path

    def test_builds_and_independently_verifies_fixed_balanced_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pilot, choices, permission, privacy = self._inputs(root)
            output = root / "teacher-gold"
            receipt = build_meter_teacher_gold_bundle_v1(
                pilot_path=pilot,
                choices_path=choices,
                permission_evidence_path=permission,
                privacy_review_evidence_path=privacy,
                output_root=output,
                repository_root=Path(__file__).parents[1],
            )
            self.assertEqual(receipt.record_count, 72)
            self.assertEqual(receipt.source_count, 36)
            self.assertEqual(receipt.split_counts, EXPECTED_SPLIT_COUNTS)
            self.assertEqual(receipt.class_split_counts, EXPECTED_CLASS_SPLIT_COUNTS)
            self.assertEqual(receipt.test_records, 0)
            self.assertFalse(receipt.test_opened)
            self.assertEqual(len(list((output / "images").glob("*.png"))), 72)
            self.assertEqual(receipt, verify_meter_teacher_gold_bundle_v1(output))

    def test_rejects_test_before_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pilot_path, choices, permission, privacy = self._inputs(root)
            pilot = json.loads(pilot_path.read_text("ascii"))
            pilot["tasks"][0]["split"] = "test"
            pilot["tasks"][0]["image_data_uri"] = "must-not-be-decoded"
            pilot_path.write_bytes(_canonical(pilot))
            with self.assertRaisesRegex(MeterTeacherGoldAdmissionError, "only source TRAIN"):
                build_meter_teacher_gold_bundle_v1(
                    pilot_path=pilot_path,
                    choices_path=choices,
                    permission_evidence_path=permission,
                    privacy_review_evidence_path=privacy,
                    output_root=root / "blocked",
                    repository_root=Path(__file__).parents[1],
                )

    def test_rejects_missing_explicit_training_permission(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pilot, choices, permission, privacy = self._inputs(root)
            payload = json.loads(permission.read_text("ascii"))
            payload["decision"] = "pending"
            permission.write_bytes(_canonical(payload))
            with self.assertRaisesRegex(MeterTeacherGoldAdmissionError, "decision mismatch"):
                build_meter_teacher_gold_bundle_v1(
                    pilot_path=pilot,
                    choices_path=choices,
                    permission_evidence_path=permission,
                    privacy_review_evidence_path=privacy,
                    output_root=root / "blocked",
                    repository_root=Path(__file__).parents[1],
                )

    def test_tampered_roi_fails_independent_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pilot, choices, permission, privacy = self._inputs(root)
            output = root / "teacher-gold"
            build_meter_teacher_gold_bundle_v1(
                pilot_path=pilot,
                choices_path=choices,
                permission_evidence_path=permission,
                privacy_review_evidence_path=privacy,
                output_root=output,
                repository_root=Path(__file__).parents[1],
            )
            first = sorted((output / "images").glob("*.png"))[0]
            first.write_bytes(first.read_bytes() + b"tamper")
            with self.assertRaisesRegex(MeterTeacherGoldAdmissionError, "SHA-256 mismatch"):
                verify_meter_teacher_gold_bundle_v1(output)

    def test_admission_layer_cannot_train_load_test_or_promote(self) -> None:
        self.assertFalse(checkpoint_loading_allowed())
        self.assertFalse(optimizer_step_allowed())
        self.assertFalse(sealed_test_access_allowed())
        self.assertFalse(production_promotion_allowed())


if __name__ == "__main__":
    unittest.main()
