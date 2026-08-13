from pathlib import Path
import tempfile
import unittest

from st_omr_training.dataset_builder import (
    DEFAULT_DEGRADATION_PROFILES,
    DEFAULT_FAMILY_PROFILES,
    plan_synthetic_families,
    synthetic_dataset_config_fingerprint,
)
from st_omr_training.synthetic_curriculum import (
    SYNTHETIC_CURRICULUM_CONFIG,
    SYNTHETIC_CURRICULUM_CONFIG_FINGERPRINT,
    SYNTHETIC_CURRICULUM_PROFILE_VERSION,
    curriculum_plan_summary,
    validate_synthetic_output_path,
)


class SyntheticCurriculumProfileTests(unittest.TestCase):
    def test_profile_is_exact_and_fingerprinted(self) -> None:
        config = SYNTHETIC_CURRICULUM_CONFIG
        self.assertEqual(SYNTHETIC_CURRICULUM_PROFILE_VERSION, "st-synthetic-curriculum-v1")
        self.assertEqual(config.dataset_name, "st-omr-synthetic-curriculum-v1")
        self.assertEqual(config.dataset_version, "v1")
        self.assertEqual(config.family_count, 512)
        self.assertEqual(config.seed_start, 100_000)
        self.assertEqual(config.split_seed, 8_001)
        self.assertEqual(config.measure_count, 8)
        self.assertEqual(config.raster_width, 1000)
        self.assertEqual(config.family_profiles, DEFAULT_FAMILY_PROFILES)
        self.assertEqual(config.degradation_profiles, DEFAULT_DEGRADATION_PROFILES)
        self.assertEqual(
            SYNTHETIC_CURRICULUM_CONFIG_FINGERPRINT,
            synthetic_dataset_config_fingerprint(config),
        )

    def test_plan_is_balanced_and_family_exclusive(self) -> None:
        plans = plan_synthetic_families(SYNTHETIC_CURRICULUM_CONFIG)
        self.assertEqual(len(plans), 512)
        self.assertEqual(len({plan.seed for plan in plans}), 512)
        self.assertEqual(
            {name: sum(plan.profile == name for plan in plans) for name in DEFAULT_FAMILY_PROFILES},
            {name: 64 for name in DEFAULT_FAMILY_PROFILES},
        )
        self.assertEqual(
            {name: sum(plan.split.value == name for plan in plans) for name in ("train", "validation", "test")},
            {"train": 410, "validation": 51, "test": 51},
        )

    def test_plan_summary_is_deterministic(self) -> None:
        first = curriculum_plan_summary()
        second = curriculum_plan_summary()
        self.assertEqual(first, second)
        self.assertEqual(first["family_count"], 512)
        self.assertEqual(first["family_split_counts"], {"test": 51, "train": 410, "validation": 51})
        self.assertEqual(first["family_profile_counts"], {name: 64 for name in sorted(DEFAULT_FAMILY_PROFILES)})

    def test_output_must_be_fresh_and_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "repo"
            root.mkdir()
            with self.assertRaises(ValueError):
                validate_synthetic_output_path(root / "dataset", root)
            existing = Path(temp_dir) / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                validate_synthetic_output_path(existing, root)
            fresh = Path(temp_dir) / "fresh"
            self.assertEqual(validate_synthetic_output_path(fresh, root), fresh.resolve())


if __name__ == "__main__":
    unittest.main()
