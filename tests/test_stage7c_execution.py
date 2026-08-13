from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
import unittest

from st_omr_training.stage7c_execution import (
    REQUIRED_STAGE7C_RUNTIME,
    Stage7CExecutionError,
    verify_repository_checkout,
    verify_stage7c_runtime,
)


class RepositoryProvenanceTests(unittest.TestCase):
    def _git(self, root: Path, *arguments: str) -> str:
        completed = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def test_clean_checkout_returns_exact_head_and_dirty_checkout_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            self._git(root, "init", "-b", "main")
            self._git(root, "config", "user.name", "Stage7C Test")
            self._git(root, "config", "user.email", "stage7c@example.invalid")
            tracked = root / "tracked.txt"
            tracked.write_text("clean\n", encoding="utf-8")
            self._git(root, "add", "tracked.txt")
            self._git(root, "commit", "-m", "baseline")
            expected = self._git(root, "rev-parse", "HEAD")
            self.assertEqual(verify_repository_checkout(root), expected)

            tracked.write_text("dirty\n", encoding="utf-8")
            with self.assertRaises(Stage7CExecutionError):
                verify_repository_checkout(root)

    def test_untracked_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            self._git(root, "init", "-b", "main")
            self._git(root, "config", "user.name", "Stage7C Test")
            self._git(root, "config", "user.email", "stage7c@example.invalid")
            tracked = root / "tracked.txt"
            tracked.write_text("clean\n", encoding="utf-8")
            self._git(root, "add", "tracked.txt")
            self._git(root, "commit", "-m", "baseline")
            (root / "untracked.txt").write_text("unexpected\n", encoding="utf-8")
            with self.assertRaises(Stage7CExecutionError):
                verify_repository_checkout(root)

    def test_nested_directory_is_not_accepted_as_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            self._git(root, "init", "-b", "main")
            self._git(root, "config", "user.name", "Stage7C Test")
            self._git(root, "config", "user.email", "stage7c@example.invalid")
            tracked = root / "tracked.txt"
            tracked.write_text("clean\n", encoding="utf-8")
            self._git(root, "add", "tracked.txt")
            self._git(root, "commit", "-m", "baseline")
            nested = root / "nested"
            nested.mkdir()
            with self.assertRaises(Stage7CExecutionError):
                verify_repository_checkout(nested)


class RuntimeProvenanceTests(unittest.TestCase):
    def test_exact_stage7c_runtime_is_verified(self) -> None:
        self.assertEqual(verify_stage7c_runtime(), REQUIRED_STAGE7C_RUNTIME)


if __name__ == "__main__":
    unittest.main()
