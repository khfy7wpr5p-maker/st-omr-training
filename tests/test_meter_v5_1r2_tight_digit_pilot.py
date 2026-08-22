from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import st_omr_training.meter_v5_1r2_tight_digit_pilot as r2


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class MeterV5_1R2TightDigitPilotTests(unittest.TestCase):
    def _surface(self, root: Path) -> tuple[Path, str, str]:
        dataset = root / "METER_V2_1500_PACKAGE_AB_CLEAN"
        annotations = dataset / "annotations"
        annotations.mkdir(parents=True)
        selection_rows: list[dict[str, object]] = []
        full_rows: list[dict[str, object]] = []
        index = 0
        for meter, class_dir in (("2/4", "2_4"), ("3/4", "3_4"), ("4/4", "4_4")):
            for local in range(10):
                sample = f"{class_dir}-sample-{local:02d}"
                family = f"family-{sample}"
                folder = f"folder-{sample}"
                rel = Path("train") / class_dir / folder / "image.png"
                image_path = dataset / rel
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image = Image.new("L", (200, 120), 255)
                image.save(image_path, format="PNG")
                image_sha = _sha(image_path)
                selection_rows.append({
                    "index": index,
                    "sample_id": sample,
                    "family_id": family,
                    "meter": meter,
                    "split": "train",
                    "folder": folder,
                    "image_relpath": rel.as_posix(),
                    "image_sha256": image_sha,
                    "image_width": 200,
                    "image_height": 120,
                    "selection_rank": hashlib.sha256(sample.encode()).hexdigest(),
                })
                full_rows.append({
                    "sample_id": sample,
                    "meter": meter,
                    "split": "train",
                    "x": 50,
                    "y": 10,
                    "w": 70,
                    "h": 90,
                    "status": "PASS",
                    "image_sha256": image_sha,
                    "image_width": 200,
                    "image_height": 120,
                    "updated_utc": "2026-08-22T00:00:00Z",
                })
                index += 1
        selection_path = annotations / "bbox_pilot_30_selection.csv"
        full_path = annotations / "bbox_pilot_30.csv"
        _write_csv(selection_path, list(selection_rows[0]), selection_rows)
        _write_csv(full_path, list(full_rows[0]), full_rows)
        return dataset, _sha(selection_path), _sha(full_path)

    def test_fixed_selection_uses_first_three_per_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset, selection_sha, full_sha = self._surface(Path(tmp))
            with patch.object(r2, "V5_SELECTION_SHA256", selection_sha), patch.object(r2, "V5_ANNOTATION_SHA256", full_sha):
                rows = r2.load_or_create_tight_selection(dataset)
            self.assertEqual(len(rows), 9)
            self.assertEqual([row["sample_id"] for row in rows[:3]], ["2_4-sample-00", "2_4-sample-01", "2_4-sample-02"])
            self.assertEqual([row["sample_id"] for row in rows[3:6]], ["3_4-sample-00", "3_4-sample-01", "3_4-sample-02"])
            self.assertEqual([row["sample_id"] for row in rows[6:]], ["4_4-sample-00", "4_4-sample-01", "4_4-sample-02"])

    def test_dual_boxes_may_overlap_and_audit_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset, selection_sha, full_sha = self._surface(Path(tmp))
            with patch.object(r2, "V5_SELECTION_SHA256", selection_sha), patch.object(r2, "V5_ANNOTATION_SHA256", full_sha):
                session = r2.TightDigitAnnotationSession(data_root=dataset)
                for index in range(9):
                    payload = session.sample_payload(index)
                    # preview equals source in this fixture. The two role boxes overlap
                    # vertically between y=48..55; overlap is intentionally admissible.
                    session.save_from_preview(
                        token=payload["binding_token"],
                        numerator={"x0": 60, "y0": 20, "x1": 100, "y1": 55},
                        denominator={"x0": 60, "y0": 48, "x1": 100, "y1": 85},
                        preview_width=payload["preview_width"],
                        preview_height=payload["preview_height"],
                    )
                audit = r2.audit_tight_digit_pilot(dataset)
            self.assertTrue(audit["annotation_contract_ready"])
            self.assertEqual(audit["pass_row_count"], 18)
            self.assertEqual(audit["review_row_count"], 0)
            self.assertEqual(audit["overlap_sample_count"], 9)
            self.assertTrue(audit["overlap_is_allowed"])
            self.assertFalse(audit["safety"]["model_inference"])

    def test_role_box_outside_full_meter_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset, selection_sha, full_sha = self._surface(Path(tmp))
            with patch.object(r2, "V5_SELECTION_SHA256", selection_sha), patch.object(r2, "V5_ANNOTATION_SHA256", full_sha):
                session = r2.TightDigitAnnotationSession(data_root=dataset)
                payload = session.sample_payload(0)
                with self.assertRaises(r2.MeterV5_1R2PilotError):
                    session.save_from_preview(
                        token=payload["binding_token"],
                        numerator={"x0": 20, "y0": 20, "x1": 90, "y1": 55},
                        denominator={"x0": 60, "y0": 48, "x1": 100, "y1": 85},
                        preview_width=payload["preview_width"],
                        preview_height=payload["preview_height"],
                    )

    def test_numerator_center_must_be_above_denominator_center(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset, selection_sha, full_sha = self._surface(Path(tmp))
            with patch.object(r2, "V5_SELECTION_SHA256", selection_sha), patch.object(r2, "V5_ANNOTATION_SHA256", full_sha):
                session = r2.TightDigitAnnotationSession(data_root=dataset)
                payload = session.sample_payload(0)
                with self.assertRaises(r2.MeterV5_1R2PilotError):
                    session.save_from_preview(
                        token=payload["binding_token"],
                        numerator={"x0": 60, "y0": 60, "x1": 100, "y1": 90},
                        denominator={"x0": 60, "y0": 20, "x1": 100, "y1": 50},
                        preview_width=payload["preview_width"],
                        preview_height=payload["preview_height"],
                    )

    def test_review_marks_both_roles_and_blocks_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset, selection_sha, full_sha = self._surface(Path(tmp))
            with patch.object(r2, "V5_SELECTION_SHA256", selection_sha), patch.object(r2, "V5_ANNOTATION_SHA256", full_sha):
                session = r2.TightDigitAnnotationSession(data_root=dataset)
                payload = session.sample_payload(0)
                session.mark_review(token=payload["binding_token"])
                audit = r2.audit_tight_digit_pilot(dataset)
            self.assertFalse(audit["annotation_contract_ready"])
            self.assertEqual(audit["review_row_count"], 2)

    def test_source_image_mutation_fails_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset, selection_sha, full_sha = self._surface(Path(tmp))
            with patch.object(r2, "V5_SELECTION_SHA256", selection_sha), patch.object(r2, "V5_ANNOTATION_SHA256", full_sha):
                rows = r2.load_or_create_tight_selection(dataset)
                image_path = dataset / rows[0]["image_relpath"]
                image_path.write_bytes(image_path.read_bytes() + b"x")
                with self.assertRaises(r2.MeterV5_1R2PilotError):
                    r2.TightDigitAnnotationSession(data_root=dataset)

    def test_safety_flags_are_hard_false(self) -> None:
        self.assertFalse(r2.model_inference_allowed_during_annotation())
        self.assertFalse(r2.validation_or_final_holdout_access_allowed())
        self.assertFalse(r2.training_authorized())


if __name__ == "__main__":
    unittest.main()
