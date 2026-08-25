import json
from pathlib import Path
import unittest


class TestMeterV53HAuthoritativeRescueColabContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "st_omr_meter_v5_3h_authoritative_rescue_training_colab.ipynb"
        )
        cls.notebook = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.code_cells = [
            cell for cell in cls.notebook["cells"] if cell.get("cell_type") == "code"
        ]
        cls.source = "".join(cls.code_cells[0]["source"])

    def test_notebook_is_single_run_exact_head_and_ci_pinned(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(len(self.code_cells), 1)
        compile(self.source, str(self.path), "exec")
        self.assertIn(
            'EXPECTED_HEAD = "b36a9d2f5daade2c3568cac8cbc736ca75ca435f"',
            self.source,
        )
        self.assertIn("EXPECTED_CI_RUN_ID = 32769348282", self.source)
        self.assertIn(
            '["git", "-C", str(REPO), "fetch", "origin", EXPECTED_HEAD, "--depth", "1"]',
            self.source,
        )
        self.assertIn(
            '["git", "-C", str(REPO), "checkout", "--detach", EXPECTED_HEAD]',
            self.source,
        )

    def test_notebook_executes_only_one_authoritative_rescue_entry(self):
        self.assertEqual(self.source.count("run_authoritative_rescue_training_v1("), 1)
        self.assertIn("confirmation=rescue.APPROVAL_TOKEN", self.source)
        for token in (
            "run_historical_retention_gate",
            "evaluate_diagnostic_gate",
            "tune_threshold(",
            "select_threshold(",
            "torch.optim",
            "optimizer.step",
            "run_validation",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)

    def test_notebook_uses_isolated_pinned_runtime(self):
        for token in (
            'VENV = Path("/content/st-omr-v5-3h-venv")',
            'VENV_PYTHON = VENV / "bin" / "python"',
            '[sys.executable, "-m", "venv", str(VENV)]',
            'env["PYTHONNOUSERSITE"] = "1"',
            '[str(VENV_PYTHON), "-m", "pip", "check"]',
            'sys.prefix == sys.base_prefix',
            '"torch": "2.13.0+cpu"',
            '"scipy": "1.18.0"',
            '"isolated_runtime": True',
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.source)
        self.assertNotIn('[sys.executable, "-m", "pip", "install"', self.source)

    def test_notebook_fails_closed_and_verifies_authoritative_receipt(self):
        for token in (
            "Refusing overwrite/rerun",
            "FETCH_HEAD mismatch",
            "Post-run HEAD mismatch",
            "Authoritative report not written",
            "rescue artifact SHA mismatch",
            "v5_3g_execution_envelope_",
            '"single_authoritative_execution_completed": True',
            '"candidate_configuration_count": 1',
            '"numerical_integrity_gate": "PASS"',
            '"frozen_state_isolation_gate": "PASS"',
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_notebook_verifies_exact_counts_and_frozen_state(self):
        for token in (
            "v53e.EXPECTED_TRAIN_GROUP_COUNTS",
            '"v5_frozen_false_negative_positive": 90',
            '"historical_frozen_false_negative_positive": 14',
            '"historical_frozen_false_negative_positive": 12',
            '"historical_frozen_true_negative": 25254',
            '"historical_frozen_true_negative": 25364',
            '"frozen_state_bit_identical"',
            '"frozen_state_before"',
            '"frozen_state_after"',
            '"optimizer_steps"',
            'execution.get("optimizer_steps") != 110',
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_notebook_keeps_later_gates_closed(self):
        for token in (
            '"train_performance_gate_executed": False',
            '"historical_validation_opened": False',
            '"first30_opened": False',
            '"v5_reserve_opened": False',
            '"v5_validation_opened": False',
            '"final_holdout_locked": True',
            '"digit4_frozen": True',
            '"threshold_tuning": False',
            '"hyperparameter_sweep": False',
            '"automatic_second_configuration": False',
            '"runtime_authority_changed": False',
            '"production_promotion": False',
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
