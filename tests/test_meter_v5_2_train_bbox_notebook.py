import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "st_omr_meter_v5_2_train_bbox_1200_colab.ipynb"
PIN = "de46b6c163376a0bab6c6ac768bca6af76c7afa5"


class TestMeterV52Notebook(unittest.TestCase):
    def _source(self, *, code_only=False):
        payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cells = payload["cells"]
        if code_only:
            cells = [cell for cell in cells if cell.get("cell_type") == "code"]
        return payload, "\n".join("".join(cell.get("source", [])) for cell in cells)

    def test_notebook_is_json_and_pins_ci_passed_execution_code(self):
        payload, source = self._source()
        self.assertEqual(payload["nbformat"], 4)
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

    def test_long_cells_use_background_worker_and_live_monitor(self):
        _payload, source = self._source(code_only=True)
        self.assertIn("run_monitored_background", source)
        self.assertIn("threading.Thread", source)
        self.assertIn("daemon=False", source)
        self.assertIn("heartbeat=5", source)
        self.assertIn("clear_output(wait=True)", source)
        self.assertIn("V5-2 KURULUM İZLEME", source)
        self.assertIn("V5-2 PRECHECK İZLEME", source)
        self.assertIn("V5-2 ANNOTATION İZLEME", source)
        self.assertIn("V5-2 AUDIT İZLEME", source)
        self.assertIn("Geçen süre", source)
        self.assertIn("Faz:", source)
        self.assertIn("FAIL-CLOSED", source)

    def test_notebook_has_no_model_or_digit_bbox_execution_path(self):
        _payload, code = self._source(code_only=True)
        self.assertNotIn("torch.load", code)
        self.assertNotIn("optimizer", code.lower())
        self.assertNotIn("numerator_bbox", code)
        self.assertNotIn("denominator_bbox", code)
        self.assertNotIn("digit_bbox", code)
        self.assertNotIn("final_holdout /", code)


if __name__ == "__main__":
    unittest.main()
