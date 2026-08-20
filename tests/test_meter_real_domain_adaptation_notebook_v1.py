from __future__ import annotations

import json
from pathlib import Path
import unittest


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "st_omr_meter_real_domain_adaptation_v1_colab.ipynb"
)


class MeterRealDomainAdaptationNotebookV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", ())) for cell in cls.document["cells"]
        )

    def test_is_clean_colab_notebook(self) -> None:
        self.assertEqual(self.document["nbformat"], 4)
        self.assertGreaterEqual(len(self.document["cells"]), 10)
        for cell in self.document["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])

    def test_binds_authoritative_inputs_and_shadow_runner(self) -> None:
        required = (
            "khfy7wpr5p-maker/st-omr-training.git",
            "fix/meter-real-domain-adaptation-v1",
            "build_meter_teacher_gold_bundle_v1",
            "run_meter_real_domain_adaptation_v1",
            "6927e1bcc5251257a983a306e2f1875c9515f97c6724a8fe9f24382c6ff30db4",
            "b72e2f5550c727484ea7226561fcd7c8e405d7d83a5bbab199d2780b8bc5db4d",
            "cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3",
        )
        for value in required:
            self.assertIn(value, self.source)

    def test_keeps_test_runtime_resolver_and_promotion_closed(self) -> None:
        self.assertIn("metrics['test_records'] == 0", self.source)
        self.assertIn("metrics['test_opened'] is False", self.source)
        self.assertIn("metrics['runtime_connected'] is False", self.source)
        self.assertIn("metrics['resolver_connected'] is False", self.source)
        self.assertIn("metrics['production_promotion_authorized'] is False", self.source)
        self.assertIn("metrics['checkpoint_reload_verified'] is True", self.source)
        self.assertNotIn("/01_REVIEW/test", self.source)


if __name__ == "__main__":
    unittest.main()
