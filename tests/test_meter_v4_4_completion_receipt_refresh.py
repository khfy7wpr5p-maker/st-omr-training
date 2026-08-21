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


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


class MeterV44CompletionReceiptRefreshTests(unittest.TestCase):
    def _fixture(self, root: Path):
        selected = []
        pool = root / "03_FINAL_HOLDOUT_150"
        for numerator in ("2", "3", "4"):
            container = pool / numerator / "4"
            for index in range(50):
                family = f"ab_{int(numerator) * 100000000 + index:09d}"
                folder_name = f"{numerator}_4_{index:012x}_{family}-1_1_1"
                folder = container / folder_name
                folder.mkdir(parents=True)
                image_path = folder / "image.png"
                Image.new("L", (40, 30), 255).save(image_path, format="PNG")
                bbox_path = folder / "bbox_meter.txt"
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
        selection_sha = sha256(_canonical(selected)).hexdigest()
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
        return pool, manifest_path, selection_sha

    def _patch_sha(self, selection_sha: str):
        stack = ExitStack()
        for module in (v44, v44_contract, v44_state, v44_qa):
            stack.enter_context(
                mock.patch.object(module, "EXPECTED_SELECTION_SHA256", selection_sha)
            )
        return stack

    def test_receipt_refreshes_after_human_bbox_correction_while_downstream_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool, manifest, selection_sha = self._fixture(Path(tmp))
            with self._patch_sha(selection_sha):
                session = v44.AnnotationSession(
                    candidate_root=pool,
                    manifest_path=manifest,
                )
                for sample, info in zip(session.samples, session.infos, strict=True):
                    v44.write_bbox_atomic(
                        sample,
                        v44.BBox(1, 1, 10, 10),
                        expected_image=info,
                    )
                session = v44.AnnotationSession(
                    candidate_root=pool,
                    manifest_path=manifest,
                )
                first_path = v44.write_completion_receipt(
                    candidate_root=pool,
                    manifest_path=manifest,
                )
                first = json.loads(first_path.read_text(encoding="ascii"))

                payload = session.sample_payload(0)
                session.save_from_preview(
                    token=payload["binding_token"],
                    x0=2,
                    y0=2,
                    x1=14,
                    y1=14,
                    preview_width=payload["preview_width"],
                    preview_height=payload["preview_height"],
                )

                second_path = v44.write_completion_receipt(
                    candidate_root=pool,
                    manifest_path=manifest,
                )
                second = json.loads(second_path.read_text(encoding="ascii"))

                self.assertNotEqual(first["bbox_manifest_sha256"], second["bbox_manifest_sha256"])
                self.assertFalse(second["human_visual_review_passed"])
                self.assertFalse(second["model_evaluated"])
                self.assertFalse(second["candidate_checkpoint_opened"])
                self.assertEqual(second["inference_count"], 0)


if __name__ == "__main__":
    unittest.main()
