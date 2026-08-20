from __future__ import annotations

import json
from pathlib import Path
import unittest


NOTEBOOK = Path("notebooks/st_omr_meter_real_domain_retention_v4_colab.ipynb")


class MeterRealDomainRetentionV4NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(NOTEBOOK.read_text("utf-8"))
        cls.code = "\n\n".join(
            "".join(cell.get("source", []))
            for cell in cls.payload["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_notebook_json_and_code_cells_compile(self) -> None:
        self.assertEqual(self.payload["nbformat"], 4)
        for index, cell in enumerate(self.payload["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"retention-v3-cell-{index}", "exec")

    def test_notebook_is_bound_to_v3_branch_and_runner(self) -> None:
        self.assertIn("fix/meter-real-domain-retention-v3", self.code)
        self.assertIn("meter_real_domain_background_runner_v3.py", self.code)
        self.assertIn("meter-real-domain-retention-v3-run", self.code)
        self.assertNotIn("meter-real-domain-background-v3-run'", self.code)

    def test_notebook_preserves_closed_boundaries(self) -> None:
        self.assertIn("'test_opened': False", self.code)
        self.assertNotIn("test_opened = True", self.code)
        self.assertNotIn("production_promotion_authorized = True", self.code)
        self.assertNotIn("runtime_connected = True", self.code)
        self.assertIn("retention-v3-receipt-*.json", self.code)

    def test_notebook_uses_new_control_and_output_directories(self) -> None:
        self.assertIn("meter-retention-v3-control", self.code)
        self.assertIn("meter-real-domain-retention-v3-run", self.code)
        self.assertIn("manifest.sha256", self.code)


if __name__ == "__main__":
    unittest.main()
