import ast
import json
from pathlib import Path
import unittest


NOTEBOOK = Path("notebooks/st_omr_meter_v5_3k_feature_domain_shift_forensics_background_colab.ipynb")


class TestMeterV53KBackgroundColabContract(unittest.TestCase):
    def _notebook(self):
        return json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def _code_cells(self):
        return [cell for cell in self._notebook()["cells"] if cell.get("cell_type") == "code"]

    def test_exactly_three_operational_code_cells_and_all_compile(self):
        cells = self._code_cells()
        self.assertEqual(len(cells), 3)
        for index, cell in enumerate(cells, start=1):
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            compile(source, f"v5_3k_colab_cell_{index}", "exec")

    def test_launch_pins_exact_ci_green_runner_and_forensics_source(self):
        launch = self._code_cells()[0]["source"]
        self.assertIn('RUNNER_HEAD = "5c132192dd0949377b4b7291a0e12c970b2cbec1"', launch)
        self.assertIn('RUNNER_BLOB = "4ff07ec4238642d6140c80dc9b544692dfb415b7"', launch)
        self.assertIn('FORENSICS_IMPLEMENTATION_HEAD = "0efa6b3ba315d671e5a449789ff6de103c735439"', launch)
        self.assertIn('EXPECTED_V53J_REPORT_SHA256 = "7a49d29e0d7257be7c59d499ab3d9ab575d369a7473b0b5298ea62aa80c7d37f"', launch)
        self.assertIn("hash-object", launch)
        self.assertIn("--detach", launch)
        self.assertIn("runner FETCH_HEAD mismatch", launch)
        self.assertIn("runner blob mismatch", launch)
        self.assertEqual(launch.count("forensics.run_feature_domain_shift_forensics_v1("), 1)

    def test_launch_is_detached_single_allocation_and_non_overwriting(self):
        launch = self._code_cells()[0]["source"]
        self.assertIn("os.O_EXCL", launch)
        self.assertIn("if RESULT.exists()", launch)
        self.assertIn("if LOCK.exists()", launch)
        self.assertIn("subprocess.Popen", launch)
        self.assertIn("start_new_session=True", launch)
        self.assertIn("stdin=subprocess.DEVNULL", launch)
        self.assertIn("close_fds=True", launch)
        self.assertIn('CONTROL_DIR = ANN / "v5_3k_background_control"', launch)
        self.assertIn('HEARTBEAT = CONTROL_DIR / f"heartbeat_{FORENSICS_IMPLEMENTATION_HEAD}.json"', launch)
        self.assertIn('PROGRESS = CONTROL_DIR / f"progress_{FORENSICS_IMPLEMENTATION_HEAD}.json"', launch)

    def test_launch_contains_no_direct_training_or_protected_gate_entry(self):
        launch = self._code_cells()[0]["source"]
        tree = ast.parse(launch)
        called_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    called_names.add(func.id)
                elif isinstance(func, ast.Attribute):
                    called_names.add(func.attr)
        self.assertNotIn("run_feature_domain_shift_forensics_v1", called_names)
        self.assertNotIn("run_authoritative_rescue_training_v1", called_names)
        self.assertNotIn("execute_rescue_tensor_harness_v1", called_names)
        self.assertNotIn("run_historical_retention_gate", called_names)
        self.assertNotIn("backward", called_names)
        self.assertNotIn("step", called_names)
        # The launch cell intentionally carries these strings as fail-closed
        # source guards for the detached runner. Their textual presence is safe;
        # only direct calls from the launch cell are forbidden.
        self.assertIn('"torch.optim."', launch)
        self.assertIn('".backward("', launch)
        self.assertIn('"optimizer.step("', launch)

    def test_status_and_final_cells_are_read_only(self):
        status, final = self._code_cells()[1:]
        for name, source in (("status", status["source"]), ("final", final["source"])):
            with self.subTest(cell=name):
                for forbidden in (
                    "subprocess.Popen",
                    "subprocess.check_call",
                    "os.open(",
                    ".write_text(",
                    ".write_bytes(",
                    "os.replace(",
                    "torch.save(",
                ):
                    self.assertNotIn(forbidden, source)
        self.assertIn("HEARTBEAT AGE SEC", status["source"])
        self.assertIn("PROGRESS PCT", status["source"])
        self.assertIn("FINAL FORENSICS RECEIPT = READY", final["source"])
        self.assertIn("CRITICAL 64D MAX STD SHIFT", final["source"])
        self.assertIn("CRITICAL TOP HIDDEN CONTRIBUTIONS", final["source"])
        self.assertIn("RETRAINING AUTHORIZED", final["source"])
        self.assertIn("FINAL_HOLDOUT LOCKED", final["source"])


if __name__ == "__main__":
    unittest.main()
