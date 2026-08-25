from pathlib import Path
import unittest


class TestMeterV53IBackgroundRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "meter_v5_3i_background_runner_v1.py"
        )
        cls.source = cls.path.read_text(encoding="utf-8")

    def test_runner_compiles_and_is_exact_gate_bound(self):
        compile(self.source, str(self.path), "exec")
        self.assertIn(
            'GATE_IMPLEMENTATION_HEAD = "844c6673f03635177a39b1ab20ab62e9392d922a"',
            self.source,
        )
        self.assertIn(
            'GATE_MODULE_BLOB = "abb5f1ae4c42b0c5f3ae26b80f2a467f47582197"',
            self.source,
        )
        self.assertIn(
            'EXPECTED_V53G_REPORT_SHA256 = (',
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

    def test_runner_uses_colab_safe_isolated_runtime(self):
        for token in (
            '[sys.executable, "-m", "venv", "--without-pip", str(VENV)]',
            '[sys.executable, "-m", "pip", "--python", str(VENV_PYTHON)]',
            'env["PYTHONNOUSERSITE"] = "1"',
            'pip_target + ["check"]',
            '"torch": "2.13.0+cpu"',
            '"scipy": "1.18.0"',
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_runner_executes_gate_once_and_never_trains(self):
        self.assertEqual(self.source.count("gate.run_train_acceptance_gate_v1("), 1)
        for forbidden in (
            "run_authoritative_rescue_training_v1(",
            "run_historical_retention_gate(",
            "torch.optim.",
            ".backward(",
            "optimizer.step(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.source)

    def test_runner_keeps_protected_surfaces_closed(self):
        for token in (
            '"historical_validation_opened": False',
            '"first30_opened": False',
            '"v5_reserve_opened": False',
            '"v5_validation_opened": False',
            '"final_holdout_locked": True',
            '"retraining_authorized_on_hold": False',
            '"retraining_authorized": False',
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_runner_has_durable_lock_heartbeat_and_fail_closed_state(self):
        for token in (
            'LOCK = CONTROL_DIR / f"launch_{GATE_IMPLEMENTATION_HEAD}.json"',
            'HEARTBEAT = CONTROL_DIR / f"heartbeat_{GATE_IMPLEMENTATION_HEAD}.json"',
            'state.get("status") != "ALLOCATED"',
            '"status": "FAILED"',
            '"status": "COMPLETED"',
            'threading.Thread(target=heartbeat_loop, daemon=True)',
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_result_is_single_non_overwriting_evidence_write(self):
        self.assertIn('if RESULT.exists():', self.source)
        self.assertIn('raise RuntimeError(f"V5-3I result already exists: {RESULT}")', self.source)
        self.assertIn('raise RuntimeError(f"refusing to overwrite V5-3I result: {RESULT}")', self.source)
        self.assertIn('os.replace(tmp, RESULT)', self.source)


if __name__ == "__main__":
    unittest.main()
