import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "st_omr_meter_v5_3c_guarded_secondary_witness_audit_colab.ipynb"
)


class TestMeterV53CColabExecutionContractV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "".join(
            line
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
            for line in cell["source"]
        )

    def test_notebook_is_exact_sha_and_ci_pinned(self):
        compile(self.source, str(NOTEBOOK), "exec")
        self.assertIn(
            'EXPECTED_HEAD = "61361612abfce132994abaca742c855f91305b44"',
            self.source,
        )
        self.assertIn("EXPECTED_CI_RUN_ID = 32751453021", self.source)
        self.assertIn('EXPECTED_SCIPY_VERSION = "1.18.0"', self.source)
        self.assertIn("FETCH_HEAD mismatch", self.source)
        self.assertIn("Post-run HEAD mismatch", self.source)

    def test_notebook_runs_only_one_guarded_secondary_audit(self):
        self.assertEqual(
            self.source.count("run_guarded_secondary_witness_audit_v1("), 1
        )
        self.assertIn("V5_3C_SINGLE_GUARDED_SECONDARY_AUDIT_APPROVED", self.source)
        forbidden = (
            "fit_robust_margin_head_candidates_v1(",
            "solve_robust_margin_minimum_total_change_v1(",
            "run_historical_retention_v1(",
            "run_first30_diagnostic_v1(",
            "torch.optim",
            "optimizer.step",
            "torch.save(",
            "tune_threshold(",
            "select_threshold(",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, self.source)

    def test_notebook_verifies_guarded_and_external_caps(self):
        required = (
            "external_l1_cap_violations",
            "internal_guarded_l1_cap_violations",
            "primary_lower_bound_conflicts",
            "parameter_bound_violations",
            "v5_solver_margin_constraint_violations",
            "historical_solver_margin_constraint_violations",
            "functional_delta_identity_verified",
            "float32_copy_gate",
            "diagnostic_witness_gate",
            "witness_weight_values_emitted",
            "Saved report mismatch",
            "v5_3c_execution_envelope_",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_notebook_keeps_candidate_and_later_gates_closed(self):
        required = (
            '"candidate_checkpoint_write_authorized": False',
            '"candidate_checkpoint_written": False',
            '"model_parameter_mutation_executed": False',
            '"autograd_grad_used": False',
            '"backward": False',
            '"optimizer_steps": 0',
            '"historical_retention_executed": False',
            '"historical_validation_opened": False',
            '"first30_opened": False',
            '"v5_validation_opened": False',
            '"final_holdout_locked": True',
            '"digit4_frozen": True',
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
