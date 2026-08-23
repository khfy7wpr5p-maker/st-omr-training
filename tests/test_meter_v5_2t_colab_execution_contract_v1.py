import json
from pathlib import Path
import unittest


class TestMeterV5_2TColabExecutionContractV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "st_omr_meter_v5_2t_bounded_class_balanced_head_repair_colab.ipynb"
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
            'EXPECTED_HEAD = "8d98c1f6ad66ee896d28c02fb7ff1afafab23be9"',
            self.source,
        )
        self.assertIn("EXPECTED_CI_RUN_ID = 32669332005", self.source)
        self.assertIn(
            '["git", "-C", str(REPO), "fetch", "origin", EXPECTED_HEAD, "--depth", "1"]',
            self.source,
        )
        self.assertIn(
            '["git", "-C", str(REPO), "checkout", "--detach", EXPECTED_HEAD]',
            self.source,
        )

    def test_notebook_executes_only_the_preregistered_v5_2t_training_entry(self):
        self.assertEqual(
            self.source.count("train_bounded_class_balanced_head_repair_v1("), 1
        )
        self.assertIn("confirmation=repair.APPROVAL_TOKEN", self.source)
        forbidden = (
            "run_historical_retention_gate",
            "evaluate_diagnostic_gate",
            "tune_threshold(",
            "select_threshold(",
            "torch.optim",
            "optimizer.step",
            "max_iter=",
            "line_search_fn=",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)

    def test_notebook_fails_closed_on_existing_or_unbound_outputs(self):
        required = (
            "Refusing overwrite/rerun",
            "FETCH_HEAD mismatch",
            "Post-run HEAD mismatch",
            "Saved report mismatch",
            "candidate SHA mismatch",
            "v5_2t_execution_envelope_",
            "numerical_integrity_gate",
            "finite_non_increasing_objective",
            "final_gradient_finite",
            "geometry_float32_copy_back",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_notebook_keeps_later_gates_closed(self):
        required = (
            '"historical_validation_opened": False',
            '"historical_retention_executed_by_this_module": False',
            '"first30_opened": False',
            '"v5_validation_opened": False',
            '"final_holdout_locked": True',
            '"digit4_frozen": True',
            '"production_promotion": False',
            '"runtime_threshold_tuning": False',
            '"automatic_second_configuration": False',
            '"hyperparameter_sweep": False',
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
