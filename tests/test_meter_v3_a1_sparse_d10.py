from __future__ import annotations

import json
from pathlib import Path
import unittest

from st_omr_training.meter_v3_a1_sparse_d10 import (
    MeterV3A1SparseD10Error,
    SPARSE_CACHE_SCHEMA,
    _safe_relative,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "meter_real_domain_background_runner_v3_a1_sparse.py"
NOTEBOOK = ROOT / "notebooks" / "st_omr_meter_real_domain_v3_a1_sparse_colab.ipynb"


class MeterV3A1SparseD10Tests(unittest.TestCase):
    def test_sparse_path_gate_rejects_escape(self) -> None:
        self.assertEqual(_safe_relative("image", "images/train/a.png"), Path("images/train/a.png"))
        for value in ("../escape.png", "/absolute.png", "images/../escape.png", ""):
            with self.assertRaises(MeterV3A1SparseD10Error):
                _safe_relative("image", value)
        self.assertEqual(SPARSE_CACHE_SCHEMA, "st-omr-meter-v3-a1-sparse-d10-v1")

    def test_sparse_runner_cannot_call_full_materializer(self) -> None:
        source = RUNNER.read_text("utf-8")
        self.assertIn("prepare_sparse_meter_d10_v3_a1", source)
        self.assertIn("patched_stage7d11_for_sparse_v3_a1", source)
        self.assertIn("staged_image_count != 1_736", source)
        self.assertIn('"full_cache_copy": False', source)
        self.assertNotIn("materialize_d10_cache", source)
        self.assertNotIn("--d10-cache-root", source)

    def test_sparse_colab_wires_only_sparse_runner(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text("utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        source = "\n".join(
            "".join(cell.get("source", [])) if isinstance(cell.get("source"), list)
            else str(cell.get("source", ""))
            for cell in notebook["cells"]
        )
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])
                cell_source = cell.get("source", "")
                if isinstance(cell_source, list):
                    cell_source = "".join(cell_source)
                compile(str(cell_source), f"sparse-notebook-cell-{index}", "exec")
        self.assertIn("fix/meter-real-domain-adaptation-v3-a1", source)
        self.assertIn("meter_real_domain_background_runner_v3_a1_sparse.py", source)
        self.assertIn("st-omr-meter-v3-a1-sparse-cache", source)
        self.assertIn("full_cache_copy", source)
        self.assertIn("bbox_frozen_exact", source)
        self.assertNotIn("/content/st-omr-d10-local-cache-v1", source)
        self.assertNotIn("materialize_d10_cache", source)


if __name__ == "__main__":
    unittest.main()
