from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from st_omr_training.dataset_builder import (
    DEFAULT_DEGRADATION_PROFILES,
    DEFAULT_FAMILY_PROFILES,
    synthetic_dataset_config_fingerprint,
)
from st_omr_training.stage7c_cli import validate_workspace_path
from st_omr_training.stage7c_dataset import (
    STAGE7C_BASELINE_DATASET_CONFIG,
    STAGE7C_BASELINE_DATASET_CONFIG_FINGERPRINT,
)


class FrozenDatasetProfileTests(unittest.TestCase):
    def test_stage7c_dataset_profile_is_exact_and_fingerprinted(self) -> None:
        config = STAGE7C_BASELINE_DATASET_CONFIG
        self.assertEqual(config.dataset_name, "st-omr-stage7c-baseline-v1")
        self.assertEqual(config.dataset_version, "v1")
        self.assertEqual(config.family_count, 64)
        self.assertEqual(config.seed_start, 70_000)
        self.assertEqual(config.split_seed, 7_001)
        self.assertEqual(config.measure_count, 8)
        self.assertEqual(config.raster_width, 1000)
        self.assertEqual(config.family_profiles, DEFAULT_FAMILY_PROFILES)
        self.assertEqual(config.degradation_profiles, DEFAULT_DEGRADATION_PROFILES)
        self.assertEqual(
            STAGE7C_BASELINE_DATASET_CONFIG_FINGERPRINT,
            synthetic_dataset_config_fingerprint(config),
        )
        self.assertEqual(len(STAGE7C_BASELINE_DATASET_CONFIG_FINGERPRINT), 64)


class WorkspaceSafetyTests(unittest.TestCase):
    def test_workspace_must_be_fresh_and_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repository = Path(temp) / "repo"
            repository.mkdir()
            outside = Path(temp) / "outside"
            self.assertEqual(validate_workspace_path(outside, repository), outside.resolve())

            with self.assertRaises(ValueError):
                validate_workspace_path(repository / "runs", repository)

            outside.mkdir()
            with self.assertRaises(FileExistsError):
                validate_workspace_path(outside, repository)


if __name__ == "__main__":
    unittest.main()
