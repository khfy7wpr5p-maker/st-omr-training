import unittest
from dataclasses import FrozenInstanceError
from hashlib import sha256

from st_omr_training.dataset_builder import (
    DATASET_BUILDER_VERSION,
    DATASET_SPLIT_WEIGHTS,
    DEFAULT_FAMILY_PROFILES,
    DatasetBuildInputError,
    DatasetImageArtifact,
    SyntheticDatasetConfig,
    plan_synthetic_families,
    split_family_counts,
    synthetic_dataset_config_fingerprint,
)
from st_omr_training.dataset_manifest import DatasetSplit


class SyntheticDatasetConfigTests(unittest.TestCase):
    def test_defaults_are_bounded_and_immutable(self):
        config = SyntheticDatasetConfig()
        self.assertEqual(config.family_count, 24)
        self.assertEqual(config.family_profiles, DEFAULT_FAMILY_PROFILES)
        self.assertEqual(DATASET_SPLIT_WEIGHTS, (80, 10, 10))
        self.assertEqual(DATASET_BUILDER_VERSION, "st-synthetic-dataset-builder-v1")
        with self.assertRaises(FrozenInstanceError):
            config.family_count = 25

    def test_bool_and_out_of_range_integer_fields_fail_closed(self):
        with self.assertRaises(DatasetBuildInputError):
            SyntheticDatasetConfig(family_count=True)
        with self.assertRaises(DatasetBuildInputError):
            SyntheticDatasetConfig(family_count=2)
        with self.assertRaises(DatasetBuildInputError):
            SyntheticDatasetConfig(measure_count=0)
        with self.assertRaises(DatasetBuildInputError):
            SyntheticDatasetConfig(raster_width=511)
        with self.assertRaises(DatasetBuildInputError):
            SyntheticDatasetConfig(split_seed=True)

    def test_seed_range_cannot_overflow(self):
        with self.assertRaises(DatasetBuildInputError):
            SyntheticDatasetConfig(family_count=3, seed_start=2**63 - 2)

    def test_profile_contract_requires_immutable_unique_supported_values(self):
        with self.assertRaises(DatasetBuildInputError):
            SyntheticDatasetConfig(family_profiles=["mixed"])
        with self.assertRaises(DatasetBuildInputError):
            SyntheticDatasetConfig(family_profiles=("mixed", "mixed"))
        with self.assertRaises(DatasetBuildInputError):
            SyntheticDatasetConfig(family_profiles=("future-profile",))
        with self.assertRaises(DatasetBuildInputError):
            SyntheticDatasetConfig(degradation_profiles=("clean", "clean"))
        with self.assertRaises(DatasetBuildInputError):
            SyntheticDatasetConfig(degradation_profiles=("severe",))

    def test_config_fingerprint_is_deterministic_and_sensitive(self):
        first = SyntheticDatasetConfig()
        second = SyntheticDatasetConfig()
        changed = SyntheticDatasetConfig(split_seed=2)
        self.assertEqual(
            synthetic_dataset_config_fingerprint(first),
            synthetic_dataset_config_fingerprint(second),
        )
        self.assertNotEqual(
            synthetic_dataset_config_fingerprint(first),
            synthetic_dataset_config_fingerprint(changed),
        )


class SplitPlanningTests(unittest.TestCase):
    def test_family_counts_follow_v1_policy_and_keep_all_splits_nonempty(self):
        self.assertEqual(split_family_counts(3), (1, 1, 1))
        self.assertEqual(split_family_counts(10), (8, 1, 1))
        self.assertEqual(split_family_counts(20), (16, 2, 2))
        for total in range(3, 101):
            counts = split_family_counts(total)
            self.assertEqual(sum(counts), total)
            self.assertTrue(all(count >= 1 for count in counts))

    def test_plan_is_deterministic_and_preserves_sequential_family_seeds(self):
        config = SyntheticDatasetConfig(
            family_count=16,
            seed_start=7000,
            split_seed=77,
        )
        first = plan_synthetic_families(config)
        second = plan_synthetic_families(config)
        self.assertEqual(first, second)
        self.assertEqual(tuple(plan.seed for plan in first), tuple(range(7000, 7016)))
        self.assertEqual(
            tuple(plan.profile for plan in first[:8]),
            DEFAULT_FAMILY_PROFILES,
        )

    def test_split_seed_changes_assignment_without_changing_family_plan_identity(self):
        first = plan_synthetic_families(
            SyntheticDatasetConfig(family_count=24, seed_start=8000, split_seed=1)
        )
        second = plan_synthetic_families(
            SyntheticDatasetConfig(family_count=24, seed_start=8000, split_seed=2)
        )
        self.assertEqual(
            tuple((plan.index, plan.seed, plan.profile) for plan in first),
            tuple((plan.index, plan.seed, plan.profile) for plan in second),
        )
        self.assertNotEqual(
            tuple(plan.split for plan in first),
            tuple(plan.split for plan in second),
        )

    def test_observed_split_counts_match_pure_policy(self):
        config = SyntheticDatasetConfig(family_count=37, split_seed=123)
        plans = plan_synthetic_families(config)
        observed = (
            sum(plan.split is DatasetSplit.TRAIN for plan in plans),
            sum(plan.split is DatasetSplit.VALIDATION for plan in plans),
            sum(plan.split is DatasetSplit.TEST for plan in plans),
        )
        self.assertEqual(observed, split_family_counts(37))


class ArtifactBoundaryTests(unittest.TestCase):
    def test_image_artifact_requires_matching_png_bytes(self):
        fake_png = b"\x89PNG\r\n\x1a\nsynthetic-test"
        digest = sha256(fake_png).hexdigest()
        artifact = DatasetImageArtifact(digest, fake_png)
        self.assertEqual(artifact.sha256, digest)
        with self.assertRaises(DatasetBuildInputError):
            DatasetImageArtifact("0" * 64, fake_png)

    def test_non_png_bytes_fail_even_with_matching_hash(self):
        payload = b"not-a-png"
        with self.assertRaises(DatasetBuildInputError):
            DatasetImageArtifact(sha256(payload).hexdigest(), payload)


if __name__ == "__main__":
    unittest.main()
