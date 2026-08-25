from pathlib import Path
import unittest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "meter_v5_3j_background_runner_v1.py"
)


class TestMeterV53JBackgroundRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")
        compile(cls.source, str(RUNNER), "exec")

    def test_exact_forensics_and_evidence_pins(self):
        self.assertIn(
            'FORENSICS_IMPLEMENTATION_HEAD = "c978b14fba23f91c60f06d2166bb23e87856d8d6"',
            self.source,
        )
        self.assertIn(
            'FORENSICS_MODULE_BLOB = "092a32504ffee9b9aafa74ddefea1c2aeb831e56"',
            self.source,
        )
        self.assertIn(
            '"448b807086bc9ee66d090fdf173ce54e3c5e2a133e60cf6ae0a791aed2717434"',
            self.source,
        )
        self.assertIn(
            '"682c2d405287051fef18b803e2597777cb7fc55c6ba0814ea3b2d4df0fa35b9d"',
            self.source,
        )
        self.assertIn(
            '"f41b0fddb9d139018e0ddd16c9765d9415031e6308efd67e16aef3a05d205bf7"',
            self.source,
        )

    def test_worker_calls_forensics_exactly_once(self):
        self.assertEqual(
            self.source.count("forensics.run_rescue_failure_forensics_v1("),
            1,
        )

    def test_runner_contains_no_training_or_protected_gate_entry(self):
        for forbidden in (
            "run_authoritative_rescue_training_v1(",
            "execute_rescue_tensor_harness_v1(",
            "run_train_acceptance_gate_v1(",
            "run_historical_retention_gate(",
            "torch.optim.",
            ".backward(",
            "optimizer.step(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.source)

    def test_protected_surfaces_and_retraining_stay_closed(self):
        for required in (
            '"training": False',
            '"checkpoint_write": False',
            '"rescue_artifact_write": False',
            '"threshold_tuning": False',
            '"threshold_sweep": False',
            '"retraining_authorized": False',
            '"historical_validation_opened": False',
            '"first30_opened": False',
            '"v5_reserve_opened": False',
            '"v5_validation_opened": False',
            '"final_holdout_locked": True',
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.source)

    def test_background_channels_are_distinct(self):
        self.assertIn('CONTROL_DIR = ANN / "v5_3j_background_control"', self.source)
        self.assertIn('HEARTBEAT = CONTROL_DIR / f"heartbeat_', self.source)
        self.assertIn('PROGRESS = CONTROL_DIR / f"progress_', self.source)
        self.assertNotEqual(
            "heartbeat_{FORENSICS_IMPLEMENTATION_HEAD}.json",
            "progress_{FORENSICS_IMPLEMENTATION_HEAD}.json",
        )

    def test_isolated_colab_runtime_bootstrap_is_pinned(self):
        self.assertIn('"venv", "--without-pip"', self.source)
        self.assertIn('"-m", "pip", "--python"', self.source)
        self.assertIn('env["PYTHONNOUSERSITE"] = "1"', self.source)
        for package_pin in (
            '"lxml": "6.1.1"',
            '"verovio": "6.2.1"',
            '"CairoSVG": "2.8.2"',
            '"Pillow": "12.3.0"',
            '"scipy": "1.18.0"',
            '"torch": "2.13.0+cpu"',
        ):
            with self.subTest(package_pin=package_pin):
                self.assertIn(package_pin, self.source)

    def test_result_is_single_atomic_non_overwriting_report(self):
        self.assertIn(
            'RESULT = ANN / "v5_3j_rescue_failure_forensics_v1.json"',
            self.source,
        )
        self.assertIn('if RESULT.exists():', self.source)
        self.assertIn('os.replace(tmp, RESULT)', self.source)
        self.assertIn('"repair_recipe_selected": False', self.source)
        self.assertIn('"retraining_authorized": False', self.source)


if __name__ == "__main__":
    unittest.main()
