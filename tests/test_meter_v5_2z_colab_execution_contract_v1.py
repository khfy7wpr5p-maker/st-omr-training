import json
from pathlib import Path
import unittest


class TestMeterV5_2ZColabExecutionContractV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "st_omr_meter_v5_2z_minimum_parameter_change_audit_colab.ipynb"
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
            'EXPECTED_HEAD = "040e1d80fcbb09f6cac7b43e15fd34567c3f7dad"',
            self.source,
        )
        self.assertIn("EXPECTED_CI_RUN_ID = 32717722495", self.source)
        self.assertIn(
            '["git", "-C", str(REPO), "fetch", "origin", EXPECTED_HEAD, "--depth", "1"]',
            self.source,
        )
        self.assertIn('EXPECTED_SCIPY_VERSION = "1.18.0"', self.source)
        self.assertIn('"scipy==1.18.0"', self.source)

    def test_notebook_runs_only_the_v5_2z_diagnostic_once(self):
        self.assertEqual(
            self.source.count("run_minimum_parameter_change_audit_v1("),
            1,
        )
        forbidden = (
            "run_lexicographic_parameter_stability_audit_v1(",
            "run_minimum_functional_logit_drift_audit_v1(",
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

    def test_notebook_verifies_unconditional_lp_without_emitting_witness(self):
        required = (
            "Refusing overwrite/rerun",
            "FETCH_HEAD mismatch",
            "Post-run HEAD mismatch",
            "Saved report mismatch",
            "library_version_matches_expected",
            "minimum_max_absolute_delta_weight_under_decision_constraints",
            "minimum_parameter_change_witness_verified",
            "independently_recomputed_max_absolute_delta_weight",
            "v5_2y_conditional_minimum_max_absolute_delta_weight",
            "conditional_upper_bound_consistency_verified",
            "functional_delta_identity_verified",
            "historical_retention_constraint_violations",
            "parameter_bound_violations",
            "historical_logit_drift_constrained",
            "historical_logit_drift_descriptive_only",
            "witness_values_emitted",
            "witness_persisted",
            "v5_2z_execution_envelope_",
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
            '"diagnostic_minimum_parameter_witness_fit": True',
            '"diagnostic_witness_persisted": False',
            '"diagnostic_witness_values_emitted": False',
            '"classifier_fit_for_deployment": False',
            '"candidate_checkpoint_write": False',
            '"model_parameter_mutation": False',
            '"threshold_tuning": False',
            '"bias_selection": False',
            '"historical_validation_opened": False',
            '"first30_opened": False',
            '"v5_validation_opened": False',
            '"final_holdout_locked": True',
            '"digit4_frozen": True',
            '"repair_selected": False',
            '"production_promotion": False',
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
