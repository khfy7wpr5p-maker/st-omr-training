import json
from pathlib import Path
import unittest


class TestMeterV5_2VColabExecutionContractV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "st_omr_meter_v5_2v_functional_logit_drift_audit_colab.ipynb"
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
            'EXPECTED_HEAD = "b1db7923e91cec534fcfd95afad7f8b4ef87607b"',
            self.source,
        )
        self.assertIn("EXPECTED_CI_RUN_ID = 32693748316", self.source)
        self.assertIn(
            '["git", "-C", str(REPO), "fetch", "origin", EXPECTED_HEAD, "--depth", "1"]',
            self.source,
        )

    def test_notebook_runs_only_the_read_only_v5_2v_audit(self):
        self.assertEqual(self.source.count("run_functional_logit_drift_audit_v1("), 1)
        forbidden = (
            "train_bounded_class_balanced_head_repair_v1(",
            "run_historical_retention_v1(",
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

    def test_notebook_fails_closed_and_binds_functional_evidence(self):
        required = (
            "Refusing overwrite/rerun",
            "FETCH_HEAD mismatch",
            "Post-run HEAD mismatch",
            "Saved report mismatch",
            "functional_delta_identity_verified",
            "cauchy_bound_verified",
            "functional_retention_diagnosis",
            "v5_2v_execution_envelope_",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_notebook_keeps_training_and_closed_surfaces_locked(self):
        required = (
            '"training": False',
            '"autograd_grad_used": False',
            '"backward": False',
            '"optimizer_steps": 0',
            '"checkpoint_write": False',
            '"classifier_fit": False',
            '"threshold_tuning": False',
            '"bias_tuning": False',
            '"historical_validation_opened": False',
            '"first30_opened": False',
            '"v5_validation_opened": False',
            '"final_holdout_locked": True',
            '"repair_selected": False',
            '"production_promotion": False',
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
