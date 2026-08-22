import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

from st_omr_training import meter_v5_1_bbox_pilot as v51
from st_omr_training import meter_v5_2_train_bbox_scale as v52
from st_omr_training import meter_v5_2a_specialist_adaptation as m


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
            Image.new("L", (200, 100), 255).save(sample_dir / "image.png")
            source_base = rf"D:\\package_ab\\{sample_id}"
            rows.append({
                "Split": split,
                "Meter": meter,
                "Package": "package_ab",
                "FamilyId": family_id,
                "SampleId": sample_id,
                "Folder": folder,
                "SourceImage": source_base + "\\image.png",
                "SourceSemantic": source_base + "\\score.semantic",
                "SourceAgnostic": source_base + "\\score.agnostic",
                "SplitRank": f"{i:064x}",
            })
        manifest = root / v51.MANIFEST_NAME[meter]
        with manifest.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return root


def prepare_seed(root: Path):
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


class TestMeterV52A(unittest.TestCase):
    def test_selection_is_300_balanced_and_seed_locked(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            fingerprint, hashes = prepare_seed(root)
            session = m.AdaptationAnnotationSession(
                data_root=root,
                expected_dataset_fingerprint=fingerprint,
                expected_seed_sha256=hashes,
            )
            self.assertEqual(len(session.samples), 300)
            self.assertEqual(
                Counter(s.meter for s in session.samples),
                Counter({"2/4": 100, "3/4": 100, "4/4": 100}),
            )
            self.assertEqual(session.handled_count, 30)
            self.assertEqual(session.pass_count, 30)
            self.assertEqual(session.review_count, 0)
            self.assertEqual(session.resume_index(), 30)
            self.assertTrue(all(s.seed_annotation for s in session.samples[:30]))
            self.assertTrue(all(not s.seed_annotation for s in session.samples[30:]))
            self.assertTrue(all("/train/" in s.image_path.as_posix() for s in session.samples))

            locked = session.sample_payload(0)
            with self.assertRaises(m.MeterV5_2AError):
                session.save_from_preview(
                    token=locked["binding_token"],
                    x0=12, y0=12, x1=45, y1=62,
                    preview_width=locked["preview_width"],
                    preview_height=locked["preview_height"],
                )

            mutable = session.sample_payload(30)
            result = session.save_from_preview(
                token=mutable["binding_token"],
                x0=10, y0=10, x1=40, y1=60,
                preview_width=mutable["preview_width"],
                preview_height=mutable["preview_height"],
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(session.handled_count, 31)

    def test_seed_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            fingerprint, hashes = prepare_seed(root)
            pilot = root / v51.ANNOTATIONS_DIR / v51.PILOT_CSV_NAME
            pilot.write_text(pilot.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            # Seed identity is deliberately delegated to the frozen V5-2 verifier;
            # its fail-closed exception surface remains authoritative here.
            with self.assertRaises(v52.MeterV5_2ScaleError):
                m.verify_seed_evidence(
                    root,
                    expected_dataset_fingerprint=fingerprint,
                    expected_seed_sha256=hashes,
                )

    def test_300_mechanical_audit_never_authorizes_training(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            fingerprint, hashes = prepare_seed(root)
            session = m.AdaptationAnnotationSession(
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
            path = m.write_annotation_audit(
                root,
                expected_dataset_fingerprint=fingerprint,
                expected_seed_sha256=hashes,
            )
            audit = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(audit["mechanical_gate"], "PASS")
            self.assertEqual(audit["annotation_count"], 300)
            self.assertEqual(audit["pass_count"], 300)
            self.assertEqual(audit["review_count"], 0)
            self.assertEqual(audit["seed_mutation_count"], 0)
            self.assertTrue(audit["human_visual_review_required"])
            self.assertTrue(audit["slot_derivation_authorized_after_human_qa"])
            self.assertFalse(audit["training_authorized"])
            self.assertEqual(audit["trainable_specialists"], ["2-AI", "3-AI"])
            self.assertEqual(audit["frozen_control_specialist"], "4-AI")
            self.assertFalse(audit["threshold_tuning_allowed"])
            self.assertTrue(audit["final_holdout_locked"])


if __name__ == "__main__":
    unittest.main()
