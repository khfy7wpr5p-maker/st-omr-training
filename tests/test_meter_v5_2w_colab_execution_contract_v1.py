import json
from pathlib import Path
import unittest


class TestMeterV5_2WColabExecutionContractV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "st_omr_meter_v5_2w_shared_affine_head_feasibility_audit_colab.ipynb"
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
            'EXPECTED_HEAD = "bdd82204182e3d5043a64907de7e0f0394089a20"',
            self.source,
        )
        self.assertIn("EXPECTED_CI_RUN_ID = 32698554873", self.source)
        self.assertIn(
            '["git", "-C", str(REPO), "fetch", "origin", EXPECTED_HEAD, "--depth", "1"]',
            self.source,
        )
        self.assertIn('EXPECTED_SCIPY_VERSION = "1.18.0"', self.source)
        self.assertIn('"scipy==1.18.0"', self.source)

    def test_notebook_runs_only_the_diagnostic_v5_2w_audit(self):
        self.assertEqual(
            self.source.count("run_shared_affine_head_feasibility_audit_v1("),
            1,
        )
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

    def test_notebook_fails_closed_and_binds_only_aggregate_witness_evidence(self):
        required = (
            "Refusing overwrite/rerun",
            "FETCH_HEAD mismatch",
            "Post-run HEAD mismatch",
            "Saved report mismatch",
            "library_version_matches_expected",
            "feasibility_claim",
            "feasible_witness_verified",
            "witness_values_emitted",
            "witness_persisted",
            "representation_failure_proven",
            "v5_2w_execution_envelope_",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_notebook_keeps_training_and_closed_surfaces_locked(self):
        required = (
            '"model_training": False',
            '"autograd_grad_used": False',
            '"backward": False',
            '"optimizer_steps": 0',
            '"diagnostic_linear_program_solve": True',
            '"diagnostic_affine_witness_fit": True',
            '"diagnostic_witness_persisted": False',
            '"classifier_fit_for_deployment": False',
            '"candidate_checkpoint_write": False',
            '"threshold_tuning": False',
            '"bias_selection": False',
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
