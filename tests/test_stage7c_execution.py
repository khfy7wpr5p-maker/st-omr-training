from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from st_omr_training.dataset_builder import (
    SyntheticDatasetConfig,
    build_synthetic_dataset,
    write_synthetic_dataset,
)
from st_omr_training.stage7c_execution import (
    REQUIRED_STAGE7C_RUNTIME,
    Stage7CExecutionError,
    run_verified_baseline_training,
    verify_repository_checkout,
    verify_stage7c_runtime,
)
from st_omr_training.training_model import TrainerConfig
from st_omr_training.training_run import BaselineRunConfig, PredictionMetrics


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


class AuthoritativeExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._dataset_temp = tempfile.TemporaryDirectory()
        cls.dataset_root = Path(cls._dataset_temp.name) / "dataset"
        cls.build = build_synthetic_dataset(
            SyntheticDatasetConfig(
                family_count=3,
                seed_start=51_000,
                split_seed=31,
                measure_count=1,
                raster_width=512,
                family_profiles=("time-4-4", "note-only", "rest-only"),
                degradation_profiles=("clean",),
            )
        )
        write_synthetic_dataset(cls.build, cls.dataset_root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._dataset_temp.cleanup()

    def test_authoritative_gate_writes_verified_marker(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        expected_head = verify_repository_checkout(repository_root)
        accepted_metrics = PredictionMetrics(
            token_error_rate=0.0,
            exact_sequence_accuracy=1.0,
            detokenization_success_rate=1.0,
            semantic_validity_rate=1.0,
            musicxml_regeneration_validity_rate=1.0,
            validation_samples=1,
            valid_semantic_predictions=1,
        )
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch(
                    "st_omr_training.training_run._mean_validation_loss",
                    side_effect=(2.0, 1.0),
                ),
                patch(
                    "st_omr_training.training_run._evaluate_predictions",
                    return_value=accepted_metrics,
                ),
            ):
                verified = run_verified_baseline_training(
                    self.build,
                    self.dataset_root,
                    Path(temp) / "runs",
                    repository_root,
                    run_config=BaselineRunConfig(
                        epochs=1,
                        batch_size=1,
                        max_train_samples=1,
                        max_validation_samples=1,
                        max_decode_tokens=64,
                    ),
                    trainer_config=TrainerConfig(master_seed=91, smoke_steps=1),
                )

            self.assertEqual(verified.result.repository_sha, expected_head)
            self.assertTrue(verified.verification_path.is_file())
            self.assertTrue(verified.verification_path.name.startswith("VERIFIED-"))
            self.assertEqual(len(verified.verification_sha256), 64)


if __name__ == "__main__":
    unittest.main()
