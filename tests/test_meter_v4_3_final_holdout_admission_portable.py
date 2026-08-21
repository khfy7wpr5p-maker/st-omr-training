from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from st_omr_training.meter_v4_3_final_holdout_admission_portable import build_manifest_portable


class MeterV43PortableAdmissionTests(unittest.TestCase):
    def _write_results(self, root: Path) -> tuple[Path, Path]:
        v41 = {
            "schema": "st-omr-meter-v4-1-learned-numerator-specialist-result-v1",
            "records": [{"family_id": f"aa_{i:09d}"} for i in range(27)],
        }
        v42 = {
            "schema": "st-omr-meter-v4-2-full-train-dev-screen-result-v1",
            "development_validation": {
                "predictions": [{"family_id": f"aa_{i:09d}"} for i in range(27, 36)]
            },
        }
        v41p = root / "v41.json"
        v42p = root / "v42.json"
        v41p.write_text(json.dumps(v41), encoding="utf-8")
        v42p.write_text(json.dumps(v42), encoding="utf-8")
        return v41p, v42p

    def _write_sample(self, parent: Path, numerator: str, index: int, family: str) -> None:
        folder = parent / f"{numerator}_4_{index:012x}_{family}-1_1_1"
        folder.mkdir(parents=True)
        (folder / "image.png").write_bytes(b"png")
        (folder / "bbox_meter.txt").write_text(
            f"id=x meter={numerator}/4 split=train bbox_x= bbox_y= bbox_w= bbox_h= admit= notes=\n",
            encoding="utf-8",
        )

    def test_direct_mount_safe_class_containers_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v41, v42 = self._write_results(root)
            pool = root / "03_FINAL_HOLDOUT_150"
            for numerator in ("2", "3", "4"):
                container = pool / f"class-{numerator}-4"
                for index in range(65):
                    family = f"ab_{numerator}{index:08d}"
                    self._write_sample(container, numerator, index, family)
            manifest = build_manifest_portable(
                candidate_root=pool,
                v4_1_result_path=v41,
                v4_2_result_path=v42,
            )
            self.assertEqual(manifest["candidate_count"], 195)
            self.assertEqual(manifest["selected_count"], 150)
            self.assertEqual(
                Counter(row["numerator_class"] for row in manifest["selected"]),
                Counter({"2": 50, "3": 50, "4": 50}),
            )
            self.assertTrue(manifest["mount_portable_discovery"])

    def test_nested_legacy_class_containers_remain_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v41, v42 = self._write_results(root)
            pool = root / "03_FINAL_HOLDOUT_150"
            for numerator in ("2", "3", "4"):
                container = pool / numerator / "4"
                for index in range(65):
                    family = f"ab_{numerator}{index:08d}"
                    self._write_sample(container, numerator, index, family)
            manifest = build_manifest_portable(
                candidate_root=pool,
                v4_1_result_path=v41,
                v4_2_result_path=v42,
            )
            self.assertEqual(manifest["selected_count"], 150)


if __name__ == "__main__":
    unittest.main()
