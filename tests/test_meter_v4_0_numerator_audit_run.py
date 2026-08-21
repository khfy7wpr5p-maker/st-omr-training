from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest import mock

from st_omr_training.meter_teacher_gold_admission_v1 import (
    ALLOWED_USE,
    CHOICES_SCHEMA,
    PERMISSION_SCHEMA,
    PILOT_SCHEMA,
    PRIVACY_SCHEMA,
    _canonical_json,
)
from st_omr_training.meter_v4_0_numerator_audit_run import _validate_and_select_source


class MeterV40NumeratorAuditRunTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        tasks = []
        answers = []
        counter = 0
        for meter_class in ("2/4", "3/4", "4/4"):
            for package, family_count in (("aa", 4), ("ab", 8)):
                for family_index in range(family_count):
                    family = f"{package}_{meter_class.replace('/', '_')}_{family_index:02d}"
                    source_id = f"source-{counter:03d}"
                    counter += 1
                    image_data_uri = "data:image/png;base64,AA=="
                    for kind, expected in (("positive", meter_class), ("none", "none")):
                        task_id = f"{source_id}__{kind}"
                        tasks.append(
                            {
                                "source_id": source_id,
                                "source_meter": meter_class,
                                "family_key": family,
                                "package": package,
                                "split": "train",
                                "key_count": 0,
                                "measure_count": 1,
                                "image_data_uri": image_data_uri,
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
                                "bbox": {"x": 10, "y": 10, "w": 20, "h": 40} if kind == "positive" else None,
                                "anchor_x": None,
                                "roi_crop_box": {"x": 0, "y": 0, "w": 100, "h": 100},
                                "notes": "",
                                "reviewed_at": "2026-08-21T00:00:00Z",
                            }
                        )

        pilot = {
            "schema": PILOT_SCHEMA,
            "generated_date": "2026-08-21",
            "source": "METER_V1/01_REVIEW/train",
            "selection": {"test_opened": False},
            "tasks": tasks,
        }
        choices = {
            "schema": CHOICES_SCHEMA,
            "pilot_version": "v1",
            "generated_at": "2026-08-21T00:00:00Z",
            "source": "fixture",
            "test_opened": False,
            "task_count": 72,
            "answered_count": 72,
            "answers": answers,
        }
        permission = {
            "schema_version": PERMISSION_SCHEMA,
            "decision": "approved",
            "allowed_use": ALLOWED_USE,
            "dataset_scope": "METER_V1/TRAIN/teacher-gold-pilot-72",
            "automatic_learning": False,
            "production_promotion_authorized": False,
            "test_access_authorized": False,
            "approved_at": "2026-08-21T00:00:00Z",
        }
        privacy = {
            "schema_version": PRIVACY_SCHEMA,
            "decision": "approved",
            "review_scope": "METER_V1/TRAIN/teacher-gold-pilot-72",
            "personal_data_detected": False,
            "redistribution_allowed": False,
            "test_opened": False,
            "reviewed_at": "2026-08-21T00:00:00Z",
        }

        pilot_path = root / "pilot.json"
        choices_path = root / "choices.json"
        permission_path = root / "permission.json"
        privacy_path = root / "privacy.json"
        pilot_path.write_bytes(_canonical_json(pilot))
        choices_path.write_bytes(_canonical_json(choices))
        permission_path.write_bytes(_canonical_json(permission))
        privacy_path.write_bytes(_canonical_json(privacy))
        return pilot_path, choices_path, permission_path, privacy_path

    def test_selection_is_27_balanced_train_positives_without_image_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_fixture(Path(tmp))
            with mock.patch(
                "st_omr_training.meter_v4_0_numerator_audit_run._decode_source_png",
                side_effect=AssertionError("selection phase must not decode images"),
            ):
                selected, provenance = _validate_and_select_source(
                    pilot_path=paths[0],
                    choices_path=paths[1],
                    permission_path=paths[2],
                    privacy_path=paths[3],
                )
        self.assertEqual(len(selected), 27)
        counts = {meter_class: 0 for meter_class in ("2/4", "3/4", "4/4")}
        for task, answer, split in selected:
            self.assertEqual(split, "train")
            self.assertEqual(task["kind"], "positive")
            self.assertEqual(answer["status"], "accepted")
            counts[str(task["expected_class"])] += 1
        self.assertEqual(counts, {"2/4": 9, "3/4": 9, "4/4": 9})
        self.assertEqual(set(provenance), {"pilot_sha256", "choices_sha256", "permission_sha256", "privacy_sha256"})

    def test_selection_rejects_test_opened(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._write_fixture(root)
            pilot = __import__("json").loads(paths[0].read_text(encoding="ascii"))
            pilot["selection"]["test_opened"] = True
            paths[0].write_bytes(_canonical_json(pilot))
            with self.assertRaises(Exception):
                _validate_and_select_source(
                    pilot_path=paths[0],
                    choices_path=paths[1],
                    permission_path=paths[2],
                    privacy_path=paths[3],
                )


if __name__ == "__main__":
    unittest.main()
