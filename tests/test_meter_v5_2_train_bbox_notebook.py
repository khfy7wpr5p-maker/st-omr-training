import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "st_omr_meter_v5_2_train_bbox_1200_colab.ipynb"
PIN = "de46b6c163376a0bab6c6ac768bca6af76c7afa5"


class TestMeterV52Notebook(unittest.TestCase):
    def test_notebook_is_json_and_pins_ci_passed_execution_code(self):
        payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(payload["nbformat"], 4)
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
        )
        self.assertIn(f"EXPECTED_CODE_SHA = '{PIN}'", source)
        self.assertIn("checkout','--detach',EXPECTED_CODE_SHA", source)
        self.assertIn("METER_V2_1500_PACKAGE_AB_CLEAN", source)
        self.assertIn("ScaleAnnotationSession", source)
        self.assertIn("launch_colab_scale(data_root=str(DATA_ROOT), session=SESSION)", source)
        self.assertIn("BBox ikiye bölünmez", source)
        self.assertIn("FIRST_30_SEEDS=LOCKED", source)
        self.assertIn("FINAL_HOLDOUT=LOCKED", source)
        self.assertIn("TRAINING=False", source)
        self.assertIn("INFERENCE_COUNT=0", source)

    def test_notebook_has_no_model_or_digit_bbox_execution_path(self):
        payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertNotIn("torch.load", code)
        self.assertNotIn("optimizer", code.lower())
        self.assertNotIn("numerator_bbox", code)
        self.assertNotIn("denominator_bbox", code)
        self.assertNotIn("digit_bbox", code)
        self.assertNotIn("final_holdout /", code)


if __name__ == "__main__":
    unittest.main()
