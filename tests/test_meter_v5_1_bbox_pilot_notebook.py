import json
import unittest
from pathlib import Path


EXPECTED_CODE_SHA = "0146ecae4f3abb06175872bebc6b7f15644b4773"
NOTEBOOK = Path("notebooks/st_omr_meter_v5_1_bbox_pilot_30_colab.ipynb")


class TestMeterV51BBoxPilotNotebook(unittest.TestCase):
    def test_notebook_is_unexecuted_and_code_pinned(self):
        payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(payload["nbformat"], 4)
        code_cells = [c for c in payload["cells"] if c["cell_type"] == "code"]
        self.assertTrue(code_cells)
        for cell in code_cells:
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs", []), [])
        text = "\n".join(
            "".join(cell.get("source", [])) for cell in payload["cells"]
        )
        self.assertIn(EXPECTED_CODE_SHA, text)
        self.assertIn("MYDRIVE / 'TEST' / DATASET_NAME", text)
        self.assertNotIn("discover_data_root('/content/drive/MyDrive')", text)
        self.assertIn("METER V5-1 LIVE PRECHECK", text)
        self.assertIn("threading.Thread", text)
        self.assertIn("elapsed=", text)
        self.assertIn("DATASET_GATE=PASS", text)
        self.assertIn("FINAL_HOLDOUT=LOCKED", text)
        self.assertIn("train_pilot_30_only", text)

    def test_notebook_has_no_training_or_model_execution(self):
        payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell["cell_type"] == "code"
        )
        forbidden = (
            "torch.load(",
            ".backward(",
            "optimizer.step(",
            "model.fit(",
            "trainer.fit(",
            "final_holdout_bbox.csv",
        )
        for token in forbidden:
            self.assertNotIn(token, code)


if __name__ == "__main__":
    unittest.main()
