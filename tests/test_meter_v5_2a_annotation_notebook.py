import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "st_omr_meter_v5_2a_300_bbox_colab.ipynb"


class TestMeterV52ANotebook(unittest.TestCase):
    def test_notebook_is_monitorable_and_annotation_only(self):
        payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn("04e86c57a719aa84714ee476d64c19dff72bee8c", code)
        self.assertIn("METER_V2_1500_PACKAGE_AB_CLEAN", code)
        self.assertIn("AdaptationAnnotationSession", code)
        self.assertIn("TRAIN_TOTAL != 300", code)
        self.assertIn("run_monitored_background", code)
        self.assertIn("threading.Thread", code)
        self.assertIn("heartbeat=5", code)
        self.assertIn("SEEDS=30_LOCKED", code)
        self.assertIn("TRAINING=False", code)
        self.assertIn("FINAL_HOLDOUT=LOCKED", code)
        self.assertIn("human visual QA/contact-sheet review is required", code)

        forbidden = (
            "optimizer.step",
            "torch.optim",
            "loss.backward",
            "model.train(",
            "numerator_bbox",
            "denominator_bbox",
            "midpoint",
            "final_holdout/",
        )
        for token in forbidden:
            self.assertNotIn(token, code)


if __name__ == "__main__":
    unittest.main()
