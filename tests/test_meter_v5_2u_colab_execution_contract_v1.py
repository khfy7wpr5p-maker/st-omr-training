import json
from pathlib import Path
import unittest


class TestMeterV5_2UColabExecutionContractV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "st_omr_meter_v5_2u_v5_2t_historical_retention_colab.ipynb"
        )
        cls.notebook = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.code_cells = [
            cell for cell in cls.notebook["cells"] if cell.get("cell_type") == "code"
        ]
        cls.source = "".join(cls.code_cells[0]["source"])

    def test_notebook_is_single_run_and_exact_ci_green_head_pinned(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(len(self.code_cells), 1)
        compile(self.source, str(self.path), "exec")
        self.assertIn(
            'EXPECTED_HEAD = "55c56671fef326a96909e169ee440a22986ff71b"',
            self.source,
        )
        self.assertIn("EXPECTED_CI_RUN_ID = 32673186350", self.source)
        self.assertIn(
            '["git", "-C", str(REPO), "fetch", "origin", EXPECTED_HEAD, "--depth", "1"]',
            self.source,
        )

    def test_notebook_runs_only_read_only_retention(self):
        self.assertEqual(self.source.count("run_historical_retention_v1("), 1)
        forbidden = (
            "train_bounded_class_balanced_head_repair_v1(",
            "run_first30_diagnostic_v1(",
            ".backward(",
            "torch.autograd",
            "torch.optim",
            "optimizer.step",
            "tune_threshold(",
            "select_threshold(",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)

    def test_notebook_binds_outputs_and_stops_on_hold(self):
        required = (
            "Refusing overwrite/rerun",
            "FETCH_HEAD mismatch",
            "Post-run HEAD mismatch",
            "Saved report mismatch",
            "v5_2u_execution_envelope_",
            "FIRST-30 AUTHORIZED = False",
            "STOP BOUNDARY = HOLD",
            "HISTORICAL RETENTION",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_notebook_keeps_all_later_surfaces_closed(self):
        required = (
            '"training": False',
            '"autograd_grad_used": False',
            '"backward": False',
            '"optimizer_steps": 0',
            '"checkpoint_write": False',
            '"runtime_threshold_tuning": False',
            '"first30_opened": False',
            '"v5_validation_opened": False',
            '"final_holdout_locked": True',
            '"digit4_frozen": True',
            '"production_promotion": False',
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
