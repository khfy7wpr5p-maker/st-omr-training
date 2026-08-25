import ast
import json
from pathlib import Path
import unittest


class TestMeterV53IBackgroundColabContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "st_omr_meter_v5_3i_train_acceptance_background_colab.ipynb"
        )
        cls.notebook = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.code_cells = [
            cell for cell in cls.notebook["cells"] if cell.get("cell_type") == "code"
        ]
        cls.sources = ["".join(cell["source"]) for cell in cls.code_cells]
        cls.launch, cls.status, cls.final = cls.sources

    def test_notebook_has_exact_three_code_cells_and_compiles(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(len(self.code_cells), 3)
        for index, source in enumerate(self.sources):
            compile(source, f"{self.path}:cell{index}", "exec")

    def test_launch_is_pinned_to_ci_green_runner_and_gate(self):
        for token in (
            'RUNNER_HEAD = "d4bfe01b850839328e1f7ce7149d41bce6111e81"',
            'RUNNER_BLOB = "7133edd1967793a9bdc5d53a9bc185b24b758d23"',
            'GATE_IMPLEMENTATION_HEAD = "844c6673f03635177a39b1ab20ab62e9392d922a"',
            'RUNNER_REL = "tools/meter_v5_3i_background_runner_v1.py"',
            'actual_blob != RUNNER_BLOB',
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.launch)

    def test_launch_is_detached_and_second_launch_is_atomically_blocked(self):
        for token in (
            "os.O_EXCL",
            "start_new_session=True",
            "stdin=subprocess.DEVNULL",
            "stderr=subprocess.STDOUT",
            'if LOCK.exists():',
            'if RESULT.exists():',
            'runner_source.count("gate.run_train_acceptance_gate_v1(") != 1',
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.launch)

    def test_notebook_does_not_directly_execute_gate_or_training(self):
        for source in self.sources:
            tree = ast.parse(source)
            called_names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = node.func
                    if isinstance(fn, ast.Name):
                        called_names.append(fn.id)
                    elif isinstance(fn, ast.Attribute):
                        called_names.append(fn.attr)
            for forbidden_call in (
                "run_train_acceptance_gate_v1",
                "run_authoritative_rescue_training_v1",
                "run_historical_retention_gate",
                "backward",
                "step",
            ):
                with self.subTest(forbidden_call=forbidden_call):
                    self.assertNotIn(forbidden_call, called_names)

    def test_status_is_read_only_and_separates_heartbeat_from_progress(self):
        self.assertIn('HEARTBEAT = CONTROL_DIR / f"heartbeat_{GATE_IMPLEMENTATION_HEAD}.json"', self.status)
        self.assertIn('PROGRESS = CONTROL_DIR / f"progress_{GATE_IMPLEMENTATION_HEAD}.json"', self.status)
        self.assertIn('print("HEARTBEAT ="', self.status)
        self.assertIn('print("PROGRESS ="', self.status)
        self.assertNotIn("subprocess.Popen", self.status)
        self.assertNotIn("os.remove", self.status)
        self.assertNotIn("unlink(", self.status)

    def test_final_receipt_is_read_only_and_surfaces_all_acceptance_boundaries(self):
        for token in (
            'print("FINAL RECEIPT = READY")',
            'print("DECISION ="',
            "V5_F1=",
            "V5_FROZEN_CORRECT_REGRESSIONS=",
            "HIST_TRAIN_FROZEN_CORRECT_REGRESSIONS=",
            'print("FROZEN STATE BIT IDENTICAL ="',
            'print("ONLY RESCUE PARAMETERS CHANGED ="',
            'print("HISTORICAL VALIDATION EXECUTED ="',
            'print("FIRST-30 OPENED ="',
            'print("V5 VALIDATION OPENED ="',
            'print("FINAL_HOLDOUT LOCKED ="',
            'print("RETRAINING AUTHORIZED ="',
            'print("REPORT SHA256 ="',
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.final)
        self.assertNotIn("subprocess", self.final)
        self.assertNotIn("write_text", self.final)
        self.assertNotIn("os.replace", self.final)


if __name__ == "__main__":
    unittest.main()
