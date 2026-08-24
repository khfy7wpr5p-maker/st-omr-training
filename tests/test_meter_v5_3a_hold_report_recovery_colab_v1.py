import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "st_omr_meter_v5_3a_hold_report_recovery_colab_v1.ipynb"
)


class TestMeterV53AHoldReportRecoveryColabV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "".join(
            line
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
            for line in cell["source"]
        )

    def test_recovery_is_bound_to_the_failed_render_execution(self):
        compile(self.source, str(NOTEBOOK), "exec")
        self.assertIn(
            'EXPECTED_EXECUTION_HEAD = "cdc6683a556c16b00e7b154fca8e89ba5dd848b7"',
            self.source,
        )
        self.assertIn(
            'SOURCE_HARNESS_HEAD = "c2d5f1652adac52387e33b9d2f33078f864f980b"',
            self.source,
        )
        self.assertIn("v5_3a_render_recovery_envelope_", self.source)

    def test_recovery_never_refits_or_publishes_a_candidate(self):
        forbidden = (
            "fit_robust_margin_head_candidates_v1(",
            "solve_robust_margin_minimum_total_change_v1(",
            "torch.optim",
            "optimizer.step",
            "candidate_checkpoint_write_authorized\": True",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)
        for token in (
            '"report_recovery_only": True',
            '"candidate_fit_reexecuted": False',
            '"candidate_checkpoint_written": False',
            '"historical_retention_executed": False',
            '"first30_opened": False',
            '"v5_validation_opened": False',
            '"final_holdout_locked": True',
        ):
            self.assertIn(token, self.source)

    def test_recovery_preserves_and_hashes_the_existing_report(self):
        required = (
            "source_report_sha256_before",
            "source_report_sha256_after",
            "Source report changed during recovery",
            'gate != "HOLD"',
            "candidate_checkpoint_written",
            "model_parameter_mutation_executed",
            "candidate directory exists after HOLD",
            "HOLD REASONS =",
            'specialist.get("runtime_float32_gate", "NOT_RUN_BECAUSE_SELECTION_GATE_HOLD")',
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
