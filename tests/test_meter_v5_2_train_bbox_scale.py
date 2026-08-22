import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

from st_omr_training import meter_v5_1_bbox_pilot as v51
from st_omr_training import meter_v5_2_train_bbox_scale as m


def _png_bytes(width=200, height=100):
    import io
    buf = io.BytesIO()
    Image.new("L", (width, height), 255).save(buf, format="PNG")
    return buf.getvalue()


PNG = _png_bytes()


def make_clean_dataset(base: Path) -> Path:
    root = base / v51.DATASET_NAME
    root.mkdir(parents=True)
    for split in v51.EXPECTED_SPLIT_COUNTS:
        for meter in v51.CLASSES:
            (root / split / v51.CLASS_DIR[meter]).mkdir(parents=True)

    meter_code = {"2/4": "2", "3/4": "3", "4/4": "4"}
    for meter in v51.CLASSES:
        rows = []
        code = meter_code[meter]
        for i in range(500):
            split = "train" if i < 400 else ("val" if i < 450 else "final_holdout")
            sample_id = f"{code}00{i:04d}-1_1_1"
            family_id = f"ab_{code}00{i:04d}"
            folder = f"{v51.CLASS_DIR[meter]}_{family_id}_{sample_id}"
            sample_dir = root / split / v51.CLASS_DIR[meter] / folder
            sample_dir.mkdir()
            (sample_dir / "image.png").write_bytes(PNG)
            source_base = (
                r"D:\workspace\primusCalvoRizoAppliedSciences2018"
                rf"\package_ab\{sample_id}"
            )
            rows.append({
                "Split": split,
                "Meter": meter,
                "Package": "package_ab",
                "FamilyId": family_id,
                "SampleId": sample_id,
                "Folder": folder,
                "SourceImage": source_base + "\\" + sample_id + ".png",
                "SourceSemantic": source_base + "\\" + sample_id + ".semantic",
                "SourceAgnostic": source_base + "\\" + sample_id + ".agnostic",
                "SplitRank": f"{i:064x}",
            })
        manifest = root / v51.MANIFEST_NAME[meter]
        with manifest.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return root


def prepare_v51_seed(root: Path):
    session = v51.AnnotationSession(data_root=root)
    for i in range(30):
        payload = session.sample_payload(i)
        session.save_from_preview(
            token=payload["binding_token"],
            x0=10, y0=10, x1=40, y1=60,
            preview_width=payload["preview_width"],
            preview_height=payload["preview_height"],
        )
    v51.write_pilot_audit(root)
    gate = v51.verify_dataset_structure(root)
    ann = root / v51.ANNOTATIONS_DIR
    hashes = {
        "annotation_csv": v51._sha256_file(ann / v51.PILOT_CSV_NAME),
        "selection_csv": v51._sha256_file(ann / v51.PILOT_SELECTION_NAME),
        "audit_json": v51._sha256_file(ann / v51.PILOT_AUDIT_NAME),
        "holdout_lock_json": v51._sha256_file(ann / v51.FINAL_HOLDOUT_LOCK_NAME),
    }
    return gate["dataset_fingerprint_sha256"], hashes


class TestMeterV52Scale(unittest.TestCase):
    def test_v51_seeds_become_first_30_and_are_immutable(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            fingerprint, hashes = prepare_v51_seed(root)
            session = m.ScaleAnnotationSession(
                data_root=root,
                expected_dataset_fingerprint=fingerprint,
                expected_seed_sha256=hashes,
            )
            self.assertEqual(len(session.samples), 1200)
            self.assertEqual(
                Counter(s.meter for s in session.samples),
                Counter({"2/4": 400, "3/4": 400, "4/4": 400}),
            )
            self.assertEqual(session.handled_count, 30)
            self.assertEqual(session.pass_count, 30)
            self.assertEqual(session.review_count, 0)
            self.assertEqual(session.resume_index(), 30)
            self.assertTrue(all(s.seed_annotation for s in session.samples[:30]))
            self.assertTrue(all(not s.seed_annotation for s in session.samples[30:]))

            seed_payload = session.sample_payload(0)
            self.assertTrue(seed_payload["locked_seed"])
            with self.assertRaises(m.MeterV5_2ScaleError):
                session.save_from_preview(
                    token=seed_payload["binding_token"],
                    x0=12, y0=12, x1=45, y1=62,
                    preview_width=seed_payload["preview_width"],
                    preview_height=seed_payload["preview_height"],
                )

            payload = session.sample_payload(30)
            self.assertFalse(payload["locked_seed"])
            result = session.save_from_preview(
                token=payload["binding_token"],
                x0=10, y0=10, x1=40, y1=60,
                preview_width=payload["preview_width"],
                preview_height=payload["preview_height"],
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(session.handled_count, 31)

    def test_seed_evidence_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            fingerprint, hashes = prepare_v51_seed(root)
            pilot = root / v51.ANNOTATIONS_DIR / v51.PILOT_CSV_NAME
            pilot.write_text(pilot.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(m.MeterV5_2ScaleError):
                m.verify_seed_evidence(
                    root,
                    expected_dataset_fingerprint=fingerprint,
                    expected_seed_sha256=hashes,
                )

    def test_full_1200_mechanical_audit_pass_without_model_or_digit_derivation(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            fingerprint, hashes = prepare_v51_seed(root)
            session = m.ScaleAnnotationSession(
                data_root=root,
                expected_dataset_fingerprint=fingerprint,
                expected_seed_sha256=hashes,
            )
            rows = [dict(session.annotations[s.sample_id]) for s in session.samples[:30]]
            for sample in session.samples[30:]:
                rows.append({
                    "sample_id": sample.sample_id,
                    "meter": sample.meter,
                    "split": "train",
                    "x": "10", "y": "10", "w": "30", "h": "50",
                    "status": "PASS",
                    "image_sha256": sample.image_sha256,
                    "image_width": str(sample.image_width),
                    "image_height": str(sample.image_height),
                    "updated_utc": "2026-08-22T00:00:00Z",
                })
            v51._atomic_write_csv(session.annotation_path, v51.ANNOTATION_COLUMNS, rows)
            audit_path = m.write_train_audit(
                root,
                expected_dataset_fingerprint=fingerprint,
                expected_seed_sha256=hashes,
            )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["annotation_count"], 1200)
            self.assertEqual(audit["pass_count"], 1200)
            self.assertEqual(audit["review_count"], 0)
            self.assertEqual(audit["seed_mutation_count"], 0)
            self.assertEqual(audit["mechanical_gate"], "PASS")
            self.assertTrue(audit["human_visual_review_required"])
            self.assertFalse(audit["training_authorized"])
            self.assertEqual(audit["inference_count"], 0)
            self.assertEqual(audit["digit_bbox_derivation_count"], 0)


if __name__ == "__main__":
    unittest.main()
