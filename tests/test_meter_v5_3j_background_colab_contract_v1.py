import ast
import json
from pathlib import Path
import unittest

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "st_omr_meter_v5_3j_rescue_failure_forensics_background_colab.ipynb"
)


class TestMeterV53JBackgroundColabContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code_cells = [
            c for c in nb["cells"] if c.get("cell_type") == "code"
        ]
        cls.sources = [
            "".join(c.get("source", [])) for c in cls.code_cells
        ]
        for index, source in enumerate(cls.sources, 1):
            compile(source, f"notebook-cell-{index}", "exec")

    def test_exactly_three_code_cells(self):
        self.assertEqual(len(self.code_cells), 3)

    def test_launch_pins_exact_ci_green_runner_and_forensics(self):
        launch = self.sources[0]
        self.assertIn(
            'RUNNER_HEAD = "3653d2d70aa186330542fc18e1fc0c9a9f01ca8f"',
            launch,
        )
        self.assertIn(
            'RUNNER_BLOB = "d33901331c5e9f5524164682ae13cdb4745ed24c"',
            launch,
        )
        self.assertIn(
            'FORENSICS_IMPLEMENTATION_HEAD = "c978b14fba23f91c60f06d2166bb23e87856d8d6"',
            launch,
        )
        self.assertIn(
            'EXPECTED_V53I_REPORT_SHA256 = "448b807086bc9ee66d090fdf173ce54e3c5e2a133e60cf6ae0a791aed2717434"',
            launch,
        )

    def test_launch_is_detached_and_duplicate_safe(self):
        launch = self.sources[0]
        self.assertIn("os.O_EXCL", launch)
        self.assertIn("start_new_session=True", launch)
        self.assertIn("stdin=subprocess.DEVNULL", launch)
        self.assertIn("stderr=subprocess.STDOUT", launch)
        self.assertIn("if RESULT.exists():", launch)
        self.assertIn("if LOCK.exists():", launch)
        self.assertIn('"status": "ALLOCATED"', launch)

    def test_notebook_never_directly_calls_forensics_or_training(self):
        forbidden_call_names = {
            "run_rescue_failure_forensics_v1",
            "run_authoritative_rescue_training_v1",
            "execute_rescue_tensor_harness_v1",
            "run_train_acceptance_gate_v1",
            "run_historical_retention_gate",
            "backward",
            "step",
        }
        for index, source in enumerate(self.sources, 1):
            tree = ast.parse(source)
            observed = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name):
                        observed.add(func.id)
                    elif isinstance(func, ast.Attribute):
                        observed.add(func.attr)
            with self.subTest(cell=index):
                self.assertTrue(forbidden_call_names.isdisjoint(observed))

    def test_status_is_read_only_monitoring_and_separate_channels(self):
        status = self.sources[1]
        self.assertIn("HEARTBEAT", status)
        self.assertIn("PROGRESS", status)
        self.assertIn("HEARTBEAT AGE SECONDS", status)
        self.assertIn("PROGRESS PERCENT", status)
        self.assertIn("LOG TAIL", status)
        for forbidden in (
            "subprocess.Popen",
            "os.O_EXCL",
            ".write_text(",
            "os.replace(",
        ):
            self.assertNotIn(forbidden, status)

    def test_final_receipt_is_read_only_and_exposes_forensic_evidence(self):
        final = self.sources[2]
        for required in (
            "FORENSICS RECEIPT = READY",
            "FAILURE SIGNATURE",
            "V5 POSITIVE RECOVERY FRACTION",
            "HISTORICAL TN REGRESSION COUNT",
            "CROSS-DOMAIN V5_POS > HIST_NEG RANK FRACTION",
            "V5 POSITIVE SCORE DIST",
            "HIST NEGATIVE SCORE DIST",
            "FROZEN STATE BIT IDENTICAL",
            "RESCUE STATE BIT IDENTICAL DURING FORENSICS",
            "REPAIR RECIPE SELECTED",
            "RETRAINING AUTHORIZED",
            "HISTORICAL VALIDATION OPENED",
            "FIRST-30 OPENED",
            "V5 VALIDATION OPENED",
            "FINAL_HOLDOUT LOCKED",
        ):
            with self.subTest(required=required):
                self.assertIn(required, final)
        for forbidden in (
            "subprocess.Popen",
            "os.O_EXCL",
            ".write_text(",
            "os.replace(",
        ):
            self.assertNotIn(forbidden, final)


if __name__ == "__main__":
    unittest.main()
