import json
from pathlib import Path
import unittest


class TestMeterV5_2RColabExecutionContractV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "st_omr_meter_v5_2r_train_class_margin_gradient_audit_colab.ipynb"
        )
        cls.notebook = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.code_cells = [
            cell
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        cls.source = "".join(cls.code_cells[0]["source"])

    def test_notebook_is_single_run_and_exact_head_pinned(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(len(self.code_cells), 1)
        compile(self.source, str(self.path), "exec")
        self.assertIn(
            'EXPECTED_HEAD = "85c0b0083792e8b9ec60ee632cfc7015e885d548"',
            self.source,
        )
        self.assertIn("EXPECTED_CI_RUN_ID = 32660114964", self.source)
        self.assertIn(
            '["git", "-C", str(REPO), "fetch", "origin", EXPECTED_HEAD, "--depth", "1"]',
            self.source,
        )
        self.assertIn(
            '["git", "-C", str(REPO), "checkout", "--detach", EXPECTED_HEAD]',
            self.source,
        )

    def test_notebook_runs_only_the_read_only_v5_2r_audit(self):
        self.assertIn("audit.run_train_class_margin_gradient_audit_v1(", self.source)
        forbidden = (
            ".backward(",
            "torch.autograd",
            "torch.optim",
            "optimizer.step",
            "torch.save",
            "train_fixed_bias_head_repair",
            "evaluate_diagnostic_gate",
            "historical_retention",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)

    def test_notebook_fails_closed_on_existing_or_unbound_evidence(self):
        self.assertIn("Refusing overwrite/rerun", self.source)
        self.assertIn("FETCH_HEAD mismatch", self.source)
        self.assertIn("Post-run HEAD mismatch", self.source)
        self.assertIn("Saved report safety mismatch", self.source)
        self.assertIn("audit_report_sha256", self.source)
        self.assertIn("v5_2r_execution_envelope_", self.source)

    def test_notebook_keeps_all_closed_surfaces_explicit(self):
        required = (
            '"training": False',
            '"autograd_grad_used": False',
            '"backward": False',
            '"optimizer_steps": 0',
            '"checkpoint_write": False',
            '"historical_validation_opened": False',
            '"first30_opened": False',
            '"v5_validation_opened": False',
            '"final_holdout_locked": True',
            '"digit4_frozen": True',
            '"repair_training_authorized": False',
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
