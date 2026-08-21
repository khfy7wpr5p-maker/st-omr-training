from __future__ import annotations

from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

import st_omr_training.meter_v4_4_final_holdout_bbox_annotation as v44
import st_omr_training.meter_v4_4_bbox_contract as v44_contract
import st_omr_training.meter_v4_4_bbox_state as v44_state
import st_omr_training.meter_v4_4_bbox_qa as v44_qa


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


class MeterV44Tests(unittest.TestCase):
    def _png(self, path: Path, size=(40, 30), value=255) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("L", size, value)
        image.save(path, format="PNG")

    def _fixture(self, root: Path, *, mutate=None):
        selected = []
        pool = root / "03_FINAL_HOLDOUT_150"
        for numerator in ("2", "3", "4"):
            container = pool / numerator / "4"
            for index in range(50):
                family = f"ab_{int(numerator)*100000000 + index:09d}"
                folder_name = f"{numerator}_4_{index:012x}_{family}-1_1_1"
                folder = container / folder_name
                folder.mkdir(parents=True)
                image_path = folder / "image.png"
                bbox_path = folder / "bbox_meter.txt"
                self._png(image_path, size=(40 + index % 3, 30 + index % 2))
                bbox_path.write_text(
                    f"id={numerator}-{index} meter={numerator}/4 split=final "
                    "bbox_x= bbox_y= bbox_w= bbox_h= admit= notes=\n",
                    encoding="utf-8",
                )
                selected.append(
                    {
                        "numerator_class": numerator,
                        "meter_class": f"{numerator}/4",
                        "folder_name": folder_name,
                        "family_id": family,
                        "image_path": str(image_path),
                        "bbox_path": str(bbox_path),
                    }
                )
        if mutate:
            mutate(selected)
        selection_sha = sha256(canonical(selected)).hexdigest()
        manifest = {
            "schema": v44.V4_3_MANIFEST_SCHEMA,
            "selection_sha256": selection_sha,
            "selected_count": 150,
            "selected_classes": {"2": 50, "3": 50, "4": 50},
            "selected": selected,
            "bbox_annotation_complete": False,
            "model_evaluated": False,
            "candidate_checkpoint_opened": False,
            "test_opened": False,
            "runtime_connected": False,
            "production_promotion_authorized": False,
        }
        manifest_path = pool / "FINAL_HOLDOUT_150_SELECTION.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return pool, manifest_path, selection_sha, selected

    def _patch_sha(self, selection_sha):
        stack = ExitStack()
        for module in (v44, v44_contract, v44_state, v44_qa):
            stack.enter_context(mock.patch.object(module, "EXPECTED_SELECTION_SHA256", selection_sha))
        return stack

    def test_manifest_binding_progress_and_read_only_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool, manifest, selection_sha, selected = self._fixture(Path(tmp))
            target_bbox = Path(selected[0]["bbox_path"])
            before = target_bbox.read_bytes()
            with self._patch_sha(selection_sha):
                session = v44.AnnotationSession(candidate_root=pool, manifest_path=manifest)
                self.assertEqual(session.annotated_count, 0)
                payload = session.sample_payload(0)
                self.assertEqual(payload["meter_class"], "2/4")
                self.assertEqual(target_bbox.read_bytes(), before)
                self.assertTrue((pool / v44.IMAGE_BINDING_NAME).is_file())
                self.assertTrue((pool / v44.PROGRESS_NAME).is_file())

    def test_selection_sha_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest, _, _ = self._fixture(Path(tmp))
            with self.assertRaises(v44.MeterV4_4AnnotationError):
                v44.load_and_validate_selection_manifest(manifest)

    def test_path_traversal_folder_name_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            def mutate(selected):
                selected[0]["folder_name"] = "../escape"
            _, manifest, selection_sha, _ = self._fixture(Path(tmp), mutate=mutate)
            with self._patch_sha(selection_sha):
                with self.assertRaises(v44.MeterV4_4AnnotationError):
                    v44.load_and_validate_selection_manifest(manifest)

    def test_duplicate_family_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            def mutate(selected):
                selected[1]["family_id"] = selected[0]["family_id"]
                old = selected[1]["folder_name"]
                prefix = old.split("_ab_", 1)[0]
                selected[1]["folder_name"] = prefix + "_" + selected[0]["family_id"] + "-1_1_1"
            _, manifest, selection_sha, _ = self._fixture(Path(tmp), mutate=mutate)
            with self._patch_sha(selection_sha):
                with self.assertRaises(v44.MeterV4_4AnnotationError):
                    v44.load_and_validate_selection_manifest(manifest)

    def test_preview_mapping_is_integer_deterministic_and_bounded(self):
        bbox = v44.preview_rect_to_original(
            x0=10, y0=5, x1=50, y1=25,
            preview_width=100, preview_height=50,
            image_width=400, image_height=200,
        )
        self.assertEqual(bbox, v44.BBox(40, 20, 160, 80))
        with self.assertRaises(v44.MeterV4_4AnnotationError):
            v44.preview_rect_to_original(
                x0=10.0, y0=5, x1=50, y1=25,
                preview_width=100, preview_height=50, image_width=400, image_height=200,
            )
        with self.assertRaises(v44.MeterV4_4AnnotationError):
            v44.preview_rect_to_original(
                x0=10, y0=5, x1=101, y1=25,
                preview_width=100, preview_height=50, image_width=400, image_height=200,
            )

    def test_bbox_contract_rejects_float_partial_and_wrong_meter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bbox_meter.txt"
            path.write_text(
                "id=x meter=2/4 split=final bbox_x=1.2 bbox_y=2 bbox_w=3 bbox_h=4 admit= notes=\n",
                encoding="utf-8",
            )
            with self.assertRaises(v44.MeterV4_4AnnotationError):
                v44.read_bbox_contract(path, expected_meter="2/4", image_width=100, image_height=100)
            path.write_text(
                "id=x meter=2/4 split=final bbox_x=1 bbox_y= bbox_w=3 bbox_h=4 admit= notes=\n",
                encoding="utf-8",
            )
            with self.assertRaises(v44.MeterV4_4AnnotationError):
                v44.read_bbox_contract(path, expected_meter="2/4", image_width=100, image_height=100)
            path.write_text(
                "id=x meter=3/4 split=final bbox_x= bbox_y= bbox_w= bbox_h= admit= notes=\n",
                encoding="utf-8",
            )
            with self.assertRaises(v44.MeterV4_4AnnotationError):
                v44.read_bbox_contract(path, expected_meter="2/4", image_width=100, image_height=100)

    def test_unexpected_image_format_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            path.write_bytes(b"not-a-png")
            with self.assertRaises(v44.MeterV4_4AnnotationError):
                v44.read_png_info(path)

    def test_explicit_save_preserves_protected_fields_and_resume_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool, manifest, selection_sha, selected = self._fixture(Path(tmp))
            with self._patch_sha(selection_sha):
                session = v44.AnnotationSession(candidate_root=pool, manifest_path=manifest)
                payload = session.sample_payload(0)
                result = session.save_from_preview(
                    token=payload["binding_token"],
                    x0=5, y0=4, x1=20, y1=18,
                    preview_width=payload["preview_width"],
                    preview_height=payload["preview_height"],
                )
                self.assertTrue(result["saved"])
                contract = v44.read_bbox_contract(
                    Path(selected[0]["bbox_path"]),
                    expected_meter="2/4",
                    image_width=payload["image_width"],
                    image_height=payload["image_height"],
                )
                self.assertEqual(contract.fields["id"], "2-0")
                self.assertEqual(contract.fields["split"], "final")
                self.assertEqual(contract.fields["admit"], "")
                self.assertEqual(contract.fields["notes"], "")
                self.assertIsNotNone(contract.bbox)

                sample = session.samples[1]
                info = session.infos[1]
                v44.write_bbox_atomic(sample, v44.BBox(1, 2, 8, 9), expected_image=info)
                session2 = v44.AnnotationSession(candidate_root=pool, manifest_path=manifest)
                self.assertEqual(session2.annotated_count, 2)

    def test_stale_binding_token_cannot_write_another_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool, manifest, selection_sha, selected = self._fixture(Path(tmp))
            with self._patch_sha(selection_sha):
                session = v44.AnnotationSession(candidate_root=pool, manifest_path=manifest)
                p0 = session.sample_payload(0)
                p1 = session.sample_payload(1)
                session.save_from_preview(
                    token=p0["binding_token"],
                    x0=1, y0=1, x1=5, y1=5,
                    preview_width=p0["preview_width"],
                    preview_height=p0["preview_height"],
                )
                self.assertIn("bbox_x= ", Path(selected[1]["bbox_path"]).read_text())
                self.assertNotEqual(p0["binding_token"], p1["binding_token"])

    def test_image_binding_detects_post_binding_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool, manifest, selection_sha, selected = self._fixture(Path(tmp))
            with self._patch_sha(selection_sha):
                session = v44.AnnotationSession(candidate_root=pool, manifest_path=manifest)
                self._png(Path(selected[0]["image_path"]), size=(41, 31), value=0)
                with self.assertRaises(v44.MeterV4_4AnnotationError):
                    session.sample_payload(0)

    def test_completion_gate_requires_all_150_and_zero_review_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool, manifest, selection_sha, _ = self._fixture(Path(tmp))
            with self._patch_sha(selection_sha):
                session = v44.AnnotationSession(candidate_root=pool, manifest_path=manifest)
                with self.assertRaises(v44.MeterV4_4AnnotationError):
                    v44.build_completion_receipt(candidate_root=pool, manifest_path=manifest)

                for sample, info in zip(session.samples, session.infos, strict=True):
                    v44.write_bbox_atomic(sample, v44.BBox(1, 1, min(10, info.width-1), min(10, info.height-1)), expected_image=info)
                session = v44.AnnotationSession(candidate_root=pool, manifest_path=manifest)
                self.assertEqual(session.annotated_count, 150)
                first = session.sample_payload(0)
                session.set_review_flag(token=first["binding_token"], flagged=True)
                with self.assertRaises(v44.MeterV4_4AnnotationError):
                    v44.build_completion_receipt(candidate_root=pool, manifest_path=manifest)
                session.set_review_flag(token=first["binding_token"], flagged=False)
                receipt = v44.build_completion_receipt(candidate_root=pool, manifest_path=manifest)
                self.assertEqual(receipt["annotated_count"], 150)
                self.assertEqual(receipt["class_counts"], {"2": 50, "3": 50, "4": 50})
                self.assertEqual(receipt["missing_bbox"], 0)
                self.assertEqual(receipt["invalid_bbox"], 0)
                self.assertFalse(receipt["human_visual_review_passed"])
                self.assertFalse(receipt["model_evaluated"])
                self.assertEqual(receipt["inference_count"], 0)
                self.assertEqual(len(receipt["records"]), 150)

    def test_symlink_selected_folder_is_rejected_when_supported(self):
        if not hasattr(Path, "symlink_to"):
            self.skipTest("symlink unsupported")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool, manifest, selection_sha, selected = self._fixture(root)
            target = Path(selected[0]["bbox_path"]).parent
            moved = target.with_name(target.name + "_real")
            target.rename(moved)
            try:
                target.symlink_to(moved, target_is_directory=True)
            except OSError:
                self.skipTest("symlink creation not permitted")
            with self._patch_sha(selection_sha):
                _, rows = v44.load_and_validate_selection_manifest(manifest)
                with self.assertRaises(v44.MeterV4_4AnnotationError):
                    v44.discover_selected_samples(pool, rows)


if __name__ == "__main__":
    unittest.main()
