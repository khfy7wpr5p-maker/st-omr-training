import json
from pathlib import Path
import unittest


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "st_omr_synthetic_curriculum_v1_colab.ipynb"
EXPECTED_SOURCE_SHA = "adc8139539d3c8cd6a2e3ee4ce4de6db4dcfeb90"


class ColabSyntheticNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", ()))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_notebook_is_valid_unexecuted_nbformat4(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertGreaterEqual(self.notebook["nbformat_minor"], 5)
        self.assertTrue(self.notebook["cells"])
        for cell in self.notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])

    def test_exact_verified_corpus_source_is_pinned(self) -> None:
        self.assertIn(f"SOURCE_SHA = '{EXPECTED_SOURCE_SHA}'", self.code)
        self.assertIn("checkout', '--detach', SOURCE_SHA", self.code)
        self.assertIn("if head != SOURCE_SHA", self.code)
        self.assertNotIn("git pull", self.code)

    def test_rendering_dependency_contract_is_explicit(self) -> None:
        self.assertIn("requirements.txt", self.code)
        self.assertNotIn("requirements-training.txt", self.code)
        for package, version in {
            "lxml": "6.1.1",
            "verovio": "6.2.1",
            "CairoSVG": "2.8.2",
            "Pillow": "12.3.0",
        }.items():
            self.assertIn(repr(package), self.code)
            self.assertIn(repr(version), self.code)

    def test_frozen_plan_and_external_workspace_are_enforced(self) -> None:
        self.assertIn("build_and_persist_synthetic_curriculum", self.code)
        self.assertIn("plan['family_count'] != 512", self.code)
        self.assertIn("{'test': 51, 'train': 410, 'validation': 51}", self.code)
        self.assertIn("Path('/content/st-omr-training')", self.code)
        self.assertIn("Path('/content/st-omr-synthetic-curriculum-v1')", self.code)

    def test_drive_export_is_after_build_and_integrity_checks(self) -> None:
        build_position = self.code.index("build = build_and_persist_synthetic_curriculum")
        archive_position = self.code.index("archive_command")
        copy_position = self.code.index("shutil.copy2")
        self.assertLess(build_position, archive_position)
        self.assertLess(archive_position, copy_position)
        self.assertIn("persisted_manifest_sha != build.manifest_sha256", self.code)
        self.assertIn("transport_sha256", self.code)
        self.assertIn("Drive transport archive hash mismatch after copy", self.code)
        self.assertIn("tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner", self.code)
        self.assertIn("gzip -n -9", self.code)


if __name__ == "__main__":
    unittest.main()
