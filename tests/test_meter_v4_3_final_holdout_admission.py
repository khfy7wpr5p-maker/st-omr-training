from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from st_omr_training.meter_v4_3_final_holdout_admission import (
    MeterV4_3AdmissionError,
    build_manifest,
)


class MeterV43FinalHoldoutAdmissionTests(unittest.TestCase):
    def _write_results(self, root: Path) -> tuple[Path, Path, set[str]]:
        observed: set[str] = set()
        v41_records = []
        for index in range(27):
            family = f"aa_{index:09d}"
            observed.add(family)
            v41_records.append({"family_id": family})
        v42_predictions = []
        for index in range(27, 36):
            family = f"aa_{index:09d}"
            observed.add(family)
            v42_predictions.append({"family_id": family})
        v41 = {
            "schema": "st-omr-meter-v4-1-learned-numerator-specialist-result-v1",
            "records": v41_records,
        }
        v42 = {
            "schema": "st-omr-meter-v4-2-full-train-dev-screen-result-v1",
            "development_validation": {"predictions": v42_predictions},
        }
        v41_path = root / "v41.json"
        v42_path = root / "v42.json"
        v41_path.write_text(json.dumps(v41), encoding="utf-8")
        v42_path.write_text(json.dumps(v42), encoding="utf-8")
        return v41_path, v42_path, observed

    def _write_pool(self, root: Path, observed: set[str]) -> Path:
        pool = root / "03_FINAL_HOLDOUT_150"
        used = sorted(observed)
        for numerator in ("2", "3", "4"):
            class_dir = pool / f"{numerator}/4"
            class_dir.mkdir(parents=True)
            for index in range(65):
                # Inject four previously observed families per class; 61 remain clean.
                if index < 4:
                    family = used[(int(numerator) + index) % len(used)]
                else:
                    family = f"ab_{int(numerator):01d}{index:08d}"
                folder = class_dir / f"{numerator}_4_{index:012x}_{family}-1_1_1"
                folder.mkdir()
                (folder / "image.png").write_bytes(b"png")
                (folder / "bbox_meter.txt").write_text(
                    f"id=x meter={numerator}/4 split=train bbox_x= bbox_y= bbox_w= bbox_h= admit= notes=\n",
                    encoding="utf-8",
                )
        return pool

    def test_build_manifest_selects_balanced_150_and_excludes_observed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v41, v42, observed = self._write_results(root)
            pool = self._write_pool(root, observed)
            manifest = build_manifest(
                candidate_root=pool,
                v4_1_result_path=v41,
                v4_2_result_path=v42,
            )
            self.assertEqual(manifest["candidate_count"], 195)
            self.assertEqual(manifest["selected_count"], 150)
            selected = manifest["selected"]
            self.assertEqual(Counter(row["numerator_class"] for row in selected), Counter({"2": 50, "3": 50, "4": 50}))
            selected_families = {row["family_id"] for row in selected}
            self.assertEqual(len(selected_families), 150)
            self.assertTrue(selected_families.isdisjoint(observed))
            self.assertFalse(manifest["model_evaluated"])
            self.assertFalse(manifest["candidate_checkpoint_opened"])
            self.assertFalse(manifest["production_promotion_authorized"])

    def test_nonblank_bbox_fails_closed_before_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v41, v42, observed = self._write_results(root)
            pool = self._write_pool(root, observed)
            first_bbox = next((pool / "2/4").glob("*/bbox_meter.txt"))
            first_bbox.write_text(
                "id=x meter=2/4 split=train bbox_x=1 bbox_y=2 bbox_w=3 bbox_h=4 admit= notes=\n",
                encoding="utf-8",
            )
            with self.assertRaises(MeterV4_3AdmissionError):
                build_manifest(candidate_root=pool, v4_1_result_path=v41, v4_2_result_path=v42)

    def test_candidate_count_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v41, v42, observed = self._write_results(root)
            pool = self._write_pool(root, observed)
            victim = next((pool / "4/4").iterdir())
            for child in victim.iterdir():
                child.unlink()
            victim.rmdir()
            with self.assertRaises(MeterV4_3AdmissionError):
                build_manifest(candidate_root=pool, v4_1_result_path=v41, v4_2_result_path=v42)


if __name__ == "__main__":
    unittest.main()
