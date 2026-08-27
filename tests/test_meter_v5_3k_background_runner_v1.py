from pathlib import Path
import unittest

from st_omr_training import meter_v5_3k_feature_domain_shift_forensics_v1 as v53k


class TestMeterV53KBackgroundRunnerContract(unittest.TestCase):
    def _source(self) -> str:
        return Path("tools/meter_v5_3k_background_runner_v1.py").read_text(encoding="utf-8")

    def test_runner_pins_exact_ci_green_forensics_source(self):
        source = self._source()
        self.assertIn('FORENSICS_IMPLEMENTATION_HEAD = "0efa6b3ba315d671e5a449789ff6de103c735439"', source)
        self.assertIn('FORENSICS_MODULE_BLOB = "c719990b5f949759e643b5cdcefc5fe2a7a650f6"', source)
        self.assertIn(v53k.EXPECTED_V53J_REPORT_SHA256, source)
        self.assertIn("hash-object", source)
        self.assertIn("--detach", source)
        self.assertIn("PYTHONNOUSERSITE", source)
        self.assertIn("--without-pip", source)
        self.assertIn('"torch": "2.13.0+cpu"', source)

    def test_runner_has_one_forensics_call_and_no_training_entry(self):
        source = self._source()
        self.assertEqual(source.count("forensics.run_feature_domain_shift_forensics_v1("), 1)
        for forbidden in (
            "run_authoritative_rescue_training_v1(",
            "execute_rescue_tensor_harness_v1(",
            "torch.optim.",
            ".backward(",
            "optimizer.step(",
            "run_historical_retention_gate(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_runner_checks_exact_input_receipts_and_artifacts(self):
        source = self._source()
        self.assertIn("EXPECTED_V53G_REPORT_SHA256", source)
        self.assertIn("EXPECTED_V53H_ENVELOPE_SHA256", source)
        self.assertIn("EXPECTED_V53J_REPORT_SHA256", source)
        self.assertIn("EXPECTED_RESCUE_ARTIFACT_SHA256", source)
        self.assertIn('V53J_REPORT = ANN / "v5_3j_rescue_failure_forensics_v1.json"', source)
        self.assertIn('RESULT = ANN / "v5_3k_feature_domain_shift_forensics_v1.json"', source)
        self.assertIn("if RESULT.exists()", source)
        self.assertIn("refusing to overwrite V5-3K result", source)

    def test_runner_keeps_heartbeat_and_progress_separate(self):
        source = self._source()
        self.assertIn('HEARTBEAT = CONTROL_DIR / f"heartbeat_{FORENSICS_IMPLEMENTATION_HEAD}.json"', source)
        self.assertIn('PROGRESS = CONTROL_DIR / f"progress_{FORENSICS_IMPLEMENTATION_HEAD}.json"', source)
        self.assertIn("atomic_json(\n                HEARTBEAT", source)
        self.assertIn("tmp = PROGRESS.with_suffix", source)

    def test_runner_result_boundary_forbids_repair_and_protected_access(self):
        source = self._source()
        for token in (
            '"repair_recipe_selected": False',
            '"retraining_authorized": False',
            '"historical_validation_opened": False',
            '"first30_opened": False',
            '"v5_reserve_opened": False',
            '"v5_validation_opened": False',
            '"final_holdout_locked": True',
            '"digit4_loaded": False',
            '"threshold_tuning": False',
            '"threshold_sweep": False',
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_runner_writes_only_atomic_forensics_report(self):
        source = self._source()
        self.assertIn('tmp = RESULT.with_suffix(RESULT.suffix + ".tmp")', source)
        self.assertIn("os.replace(tmp, RESULT)", source)
        self.assertNotIn("torch.save(", source)
        self.assertNotIn("checkpoint_write = True", source)


if __name__ == "__main__":
    unittest.main()
