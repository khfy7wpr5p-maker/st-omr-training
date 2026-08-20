from __future__ import annotations

import json
from pathlib import Path
import unittest

import tools.meter_real_domain_background_runner_v1 as runner_v1
import tools.meter_real_domain_background_runner_v2 as runner_v2


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "st_omr_meter_real_domain_background_v3_colab.ipynb"
)


class MeterRealDomainBackgroundV3Tests(unittest.TestCase):
    def test_v2_runner_reuses_exact_v1_cache_materializer(self) -> None:
        self.assertIs(runner_v2.materialize_d10_cache, runner_v1.materialize_d10_cache)

    def test_notebook_is_clean_compilable_and_reuses_completed_cache(self) -> None:
        document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(document["nbformat"], 4)
        source = "\n".join("".join(cell.get("source", ())) for cell in document["cells"])
        for index, cell in enumerate(document["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell.get("execution_count"))
            self.assertEqual(cell.get("outputs"), [])
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
        for required in (
            "fix/meter-real-domain-adaptation-v2",
            "meter_real_domain_background_runner_v2.py",
            "/content/st-omr-d10-local-cache-v1",
            "meter-real-domain-background-v3-run",
            "status.get('epochs_total', 20)",
            "d11_fully_frozen",
            "?/44260",
            "SON 25 LOG SATIRI",
        ):
            self.assertIn(required, source)
        self.assertNotIn("meter_real_domain_background_runner_v1.py", source)
        self.assertNotIn("/01_REVIEW/test", source)
        self.assertNotIn("shutil.rmtree(D10_LOCAL_CACHE)", source)


if __name__ == "__main__":
    unittest.main()
