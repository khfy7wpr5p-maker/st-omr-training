import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from PIL import Image

from st_omr_training import meter_v5_1_bbox_pilot as m


def _png_bytes(width=200, height=100):
    import io
    buf = io.BytesIO()
    Image.new("L", (width, height), 255).save(buf, format="PNG")
    return buf.getvalue()


PNG = _png_bytes()


def make_clean_dataset(base: Path) -> Path:
    root = base / m.DATASET_NAME
    root.mkdir(parents=True)
    for split in m.EXPECTED_SPLIT_COUNTS:
        for meter in m.CLASSES:
            (root / split / m.CLASS_DIR[meter]).mkdir(parents=True)

    meter_code = {"2/4": "2", "3/4": "3", "4/4": "4"}
    for meter in m.CLASSES:
        rows = []
        code = meter_code[meter]
        for i in range(500):
            if i < 400:
                split = "train"
            elif i < 450:
                split = "val"
            else:
                split = "final_holdout"
            sample_id = f"{code}00{i:04d}-1_1_1"
            family_id = f"ab_{code}00{i:04d}"
            folder = f"{m.CLASS_DIR[meter]}_{family_id}_{sample_id}"
            sample_dir = root / split / m.CLASS_DIR[meter] / folder
            sample_dir.mkdir()
            (sample_dir / "image.png").write_bytes(PNG)
            source_base = (
                r"D:\veri eğitim seti\ST_OMR_WORKSPACE\00_ORIGINAL_DATASETS"
                rf"\primusCalvoRizoAppliedSciences2018\package_ab\{sample_id}"
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
        manifest = root / m.MANIFEST_NAME[meter]
        with manifest.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return root


class TestMeterV51Pilot(unittest.TestCase):
    def test_discovery_zero_one_multiple(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            with self.assertRaises(m.MeterV5_1PilotError):
                m.discover_data_root(base)
            a = base / "TEST" / m.DATASET_NAME
            a.mkdir(parents=True)
            self.assertEqual(m.discover_data_root(base), a)
            b = base / "OTHER" / m.DATASET_NAME
            b.mkdir(parents=True)
            with self.assertRaises(m.MeterV5_1PilotError):
                m.discover_data_root(base)

    def test_clean_gate_and_holdout_lock(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            gate = m.verify_dataset_structure(root)
            self.assertEqual(gate["total"], 1500)
            self.assertEqual(gate["unique_family_id"], 1500)
            self.assertEqual(gate["unique_sample_id"], 1500)
            self.assertEqual(gate["unique_source_image"], 1500)
            self.assertTrue(gate["package_ab_only"])
            self.assertEqual(gate["cross_split_family_leakage"], 0)
            self.assertTrue(gate["final_holdout_locked"])
            lock = m.ensure_final_holdout_lock(root, gate)
            payload = json.loads(lock.read_text(encoding="utf-8"))
            self.assertTrue(payload["locked"])
            self.assertFalse(payload["annotation_opened"])
            self.assertFalse(payload["training_opened"])

    def test_gate_rejects_package_aa_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            path = root / m.MANIFEST_NAME["2/4"]
            rows = m._read_csv(path)
            rows[0]["SourceImage"] = rows[0]["SourceImage"].replace("package_ab", "package_aa")
            m._atomic_write_csv(path, tuple(rows[0].keys()), rows)
            with self.assertRaises(m.MeterV5_1PilotError):
                m.verify_dataset_structure(root)

    def test_gate_rejects_global_family_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            p2 = root / m.MANIFEST_NAME["2/4"]
            p3 = root / m.MANIFEST_NAME["3/4"]
            r2 = m._read_csv(p2)
            r3 = m._read_csv(p3)
            r3[0]["FamilyId"] = r2[0]["FamilyId"]
            m._atomic_write_csv(p3, tuple(r3[0].keys()), r3)
            with self.assertRaises(m.MeterV5_1PilotError):
                m.verify_dataset_structure(root)

    def test_pilot_selection_reads_only_train_image_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            gate = m.verify_dataset_structure(root)
            original = m._read_png_binding
            seen = []

            def wrapped(path):
                seen.append(Path(path))
                return original(path)

            with mock.patch.object(m, "_read_png_binding", side_effect=wrapped):
                samples = m.load_or_create_pilot_selection(root, gate)
            self.assertEqual(len(samples), 30)
            self.assertEqual(Counter(s.meter for s in samples), Counter({"2/4": 10, "3/4": 10, "4/4": 10}))
            self.assertTrue(all(s.split == "train" for s in samples))
            self.assertTrue(seen)
            self.assertTrue(all("final_holdout" not in p.parts for p in seen))
            self.assertTrue(all("val" not in p.parts for p in seen))

    def test_session_save_review_resume_and_single_row_per_sample(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            session = m.AnnotationSession(data_root=root)
            p0 = session.sample_payload(0)
            result = session.save_from_preview(
                token=p0["binding_token"],
                x0=10, y0=10, x1=40, y1=60,
                preview_width=p0["preview_width"],
                preview_height=p0["preview_height"],
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(session.handled_count, 1)
            p1 = session.sample_payload(1)
            session.mark_review(token=p1["binding_token"])
            self.assertEqual(session.handled_count, 2)
            self.assertEqual(session.review_count, 1)
            self.assertEqual(session.resume_index(), 2)

            p0b = session.sample_payload(0)
            session.save_from_preview(
                token=p0b["binding_token"],
                x0=12, y0=12, x1=45, y1=62,
                preview_width=p0b["preview_width"],
                preview_height=p0b["preview_height"],
            )
            rows = m._read_csv(session.annotation_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({r["sample_id"] for r in rows}), 2)

    def test_original_image_mutation_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            session = m.AnnotationSession(data_root=root)
            sample = session.samples[0]
            Image.new("L", (200, 100), 0).save(sample.image_path)
            with self.assertRaises(m.MeterV5_1PilotError):
                session.sample_payload(0)

    def test_outside_bbox_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            session = m.AnnotationSession(data_root=root)
            p = session.sample_payload(0)
            with self.assertRaises(m.MeterV5_1PilotError):
                session.save_from_preview(
                    token=p["binding_token"],
                    x0=0, y0=0,
                    x1=p["preview_width"] + 1,
                    y1=20,
                    preview_width=p["preview_width"],
                    preview_height=p["preview_height"],
                )

    def test_audit_after_30_pass_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            session = m.AnnotationSession(data_root=root)
            for i in range(30):
                p = session.sample_payload(i)
                session.save_from_preview(
                    token=p["binding_token"],
                    x0=10, y0=10, x1=40, y1=60,
                    preview_width=p["preview_width"],
                    preview_height=p["preview_height"],
                )
            path = m.write_pilot_audit(root)
            audit = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(audit["annotation_count"], 30)
            self.assertEqual(audit["unique_sample_id"], 30)
            self.assertEqual(audit["pass_count"], 30)
            self.assertEqual(audit["review_count"], 0)
            self.assertEqual(audit["bbox_outside_image"], 0)
            self.assertEqual(audit["mechanical_gate"], "PASS")
            self.assertTrue(audit["final_holdout_locked"])
            self.assertFalse(audit["training_authorized"])


if __name__ == "__main__":
    unittest.main()
