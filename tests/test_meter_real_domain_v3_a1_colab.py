from __future__ import annotations

import json
from pathlib import Path
import unittest


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "st_omr_meter_real_domain_v3_a1_colab.ipynb"
)


class MeterRealDomainV3A1ColabTests(unittest.TestCase):
    def test_notebook_is_clean_and_wired_to_v3_a1(self) -> None:
        payload = json.loads(NOTEBOOK.read_text("utf-8"))
        source_parts: list[str] = []
        for index, cell in enumerate(payload["cells"]):
            if cell.get("cell_type") != "code":
                continue
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs"), [])
            source = "".join(cell["source"])
            compile(source, f"v3-a1-notebook-cell-{index}", "exec")
            source_parts.append(source)
        source = "\n".join(source_parts)
        for required in (
            "fix/meter-real-domain-adaptation-v3-a1",
            "meter_real_domain_background_runner_v3_a1.py",
            "meter-real-domain-v3-a1-run",
            "bbox_frozen_exact",
            "?/44260" if False else "D10 YEREL KOPYA",
            "best_real_per_class_recall",
        ):
            self.assertIn(required, source)
        self.assertNotIn("meter_real_domain_background_runner_v2.py", source)
        self.assertNotIn("/01_REVIEW/test", source)
        self.assertNotIn("shutil.rmtree(D10_LOCAL_CACHE)", source)


if __name__ == "__main__":
    unittest.main()
