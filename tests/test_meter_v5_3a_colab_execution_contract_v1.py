import json
from pathlib import Path
import unittest


class TestMeterV5_3AColabExecutionContractV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "st_omr_meter_v5_3a_robust_margin_head_candidate_colab.ipynb"
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
            'EXPECTED_HEAD = "cdc6683a556c16b00e7b154fca8e89ba5dd848b7"',
            self.source,
        )
        self.assertIn("EXPECTED_CI_RUN_ID = 32735656612", self.source)
        self.assertIn('EXPECTED_SCIPY_VERSION = "1.18.0"', self.source)
        self.assertIn('"scipy==1.18.0"', self.source)
        self.assertIn(
            '["git", "-C", str(REPO), "fetch", "origin", EXPECTED_HEAD, "--depth", "1"]',
            self.source,
        )

    def test_notebook_runs_only_the_preregistered_candidate_fit_once(self):
        self.assertEqual(
            self.source.count("fit_robust_margin_head_candidates_v1("), 1
        )
        self.assertIn("confirmation=repair.APPROVAL_TOKEN", self.source)
        forbidden = (
            "run_minimum_parameter_change_audit_v1(",
            "run_historical_retention_v1(",
            "run_historical_retention_gate",
            "run_first30_diagnostic_v1(",
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

    def test_notebook_fails_closed_and_verifies_candidate_artifacts(self):
        required = (
            "Refusing overwrite/rerun",
            "FETCH_HEAD mismatch",
            "Post-run HEAD mismatch",
            "Saved report mismatch",
            "candidate SHA mismatch",
            "candidate_selection_gate",
            "CANDIDATE_WITNESS_VERIFIED",
            "minimum_delta_weight_l1",
            "minimum_delta_weight_linf",
            "primary_l1_cap_violations",
            "v5_solver_margin_constraint_violations",
            "historical_solver_margin_constraint_violations",
            "functional_delta_identity_verified",
            "float32_copy_gate",
            "runtime_float32_gate",
            "only_head_weight_changed",
            "backbone_bit_identical",
            "head_bias_bit_identical",
            "reload_verified",
            "v5_3a_execution_envelope_",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_notebook_renders_hold_without_optional_runtime_evidence(self):
        self.assertIn(
            'runtime_gate = specialist.get("runtime_float32_gate")',
            self.source,
        )
        self.assertIn(
            '"NOT_RUN_DUE_TO_CANDIDATE_SELECTION_HOLD"',
            self.source,
        )
        self.assertNotIn(
            'print("RUNTIME FLOAT32 GATE =", specialist["runtime_float32_gate"])',
            self.source,
        )

    def test_notebook_keeps_later_gates_closed(self):
        required = (
            '"linear_head_candidate_fit_authorized": True',
            '"candidate_checkpoint_write_authorized": True',
            '"frozen_backbone": True',
            '"frozen_head_bias": True',
            '"autograd_grad_used": False',
            '"backward": False',
            '"optimizer_steps": 0',
            '"runtime_threshold_tuning": False',
            '"automatic_second_configuration": False',
            '"hyperparameter_sweep": False',
            '"historical_validation_opened": False',
            '"historical_retention_executed_by_this_module": False',
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
