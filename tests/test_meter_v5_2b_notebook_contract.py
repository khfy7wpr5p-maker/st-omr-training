import json
import unittest
from pathlib import Path


class TestMeterV52BNotebookContract(unittest.TestCase):
    def test_notebook_keeps_closed_surfaces_visible(self):
        path = Path("notebooks/st_omr_meter_v5_2b_23_adaptation_colab.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
        )
        self.assertIn("agent/meter-v5-2b-23-adaptation-training", source)
        self.assertIn("VAL=CLOSED", source)
        self.assertIn("FINAL_HOLDOUT=LOCKED", source)
        self.assertIn("4-AI=FROZEN", source)
        self.assertIn("NO_THRESHOLD_TUNING", source)
        self.assertIn("diagnostic-seed", source)
        self.assertNotIn("final_holdout/", source)
        self.assertNotIn("/val/", source)


if __name__ == "__main__":
    unittest.main()
