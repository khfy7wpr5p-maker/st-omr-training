from __future__ import annotations

from dataclasses import asdict
import unittest

from st_omr_training.degradation import sample_degradation_config
from st_omr_training.stage7d3_execution import (
    EXPECTED_D2_BEST_EPOCH,
    EXPECTED_D2_CHECKPOINT_SHA256,
    EXPECTED_D2_CHECKPOINT_STATE_SHA256,
    EXPECTED_D2_RUN_ID,
    EXPECTED_D2_VERIFICATION_SHA256,
    Stage7D3ExecutionError,
    _classify_degradation,
    _skip_non_validation,
)


class _SkipSentinel(dict):
    def get(self, key, default=None):
        if key != "split":
            raise AssertionError(f"D3 touched skipped row field: {key}")
        return super().get(key, default)


class Stage7D3ExecutionTests(unittest.TestCase):
    def test_train_and_test_rows_skip_before_other_field_access(self):
        self.assertTrue(_skip_non_validation(_SkipSentinel(split="train"), 0))
        self.assertTrue(_skip_non_validation(_SkipSentinel(split="test"), 1))

    def test_validation_row_is_not_skipped(self):
        self.assertFalse(_skip_non_validation({"split": "validation"}, 2))

    def test_invalid_split_fails_closed(self):
        with self.assertRaises(Stage7D3ExecutionError):
            _skip_non_validation({"split": "future"}, 3)

    def test_degradation_profiles_are_recovered_exactly(self):
        for profile in ("clean", "light", "medium"):
            config = sample_degradation_config(12345, profile, raster_width=1000)
            sample = {"degradation_config": asdict(config)}
            self.assertEqual(_classify_degradation(sample), profile)

    def test_degradation_drift_fails_closed(self):
        config = asdict(sample_degradation_config(12345, "light", raster_width=1000))
        config["noise_level"] += 1
        with self.assertRaises(Stage7D3ExecutionError):
            _classify_degradation({"degradation_config": config})

    def test_authoritative_d2_identity_is_frozen(self):
        self.assertEqual(
            EXPECTED_D2_RUN_ID,
            "14d63841254c03463ad76bbed83df95045742c23f71ad91d7b0c5dc19495a373",
        )
        self.assertEqual(
            EXPECTED_D2_CHECKPOINT_SHA256,
            "239cf3dbdf80235bfc7e4a68fe5fecc03e8cd6fefc8a9ff6e27a2ca879ed5291",
        )
        self.assertEqual(
            EXPECTED_D2_CHECKPOINT_STATE_SHA256,
            "466cefcd40887cb0578b7bbc87c6a1b5f676dc0272ab5eee1142e45e7da8e17d",
        )
        self.assertEqual(
            EXPECTED_D2_VERIFICATION_SHA256,
            "6743425d42da77dfacef50388e879d45aa01f01b740cfd2deb381a55436500c3",
        )
        self.assertEqual(EXPECTED_D2_BEST_EPOCH, 20)


if __name__ == "__main__":
    unittest.main()
