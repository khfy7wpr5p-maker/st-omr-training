from __future__ import annotations

import unittest

from st_omr_training.stage7c_profile import (
    STAGE7C_FROZEN_MODEL_CONFIG,
    STAGE7C_FROZEN_PREPROCESS_CONFIG,
    STAGE7C_FROZEN_TRAINER_CONFIG,
)
from st_omr_training.stage7d2_profile import (
    EXPECTED_D1_ARTIFACT_BINDING_SHA256,
    STAGE7D2_FROZEN_MODEL_CONFIG,
    STAGE7D2_FROZEN_PREPROCESS_CONFIG,
    STAGE7D2_FROZEN_RUN_CONFIG,
    STAGE7D2_FROZEN_RUN_FINGERPRINT,
    STAGE7D2_FROZEN_TRAINER_CONFIG,
    Stage7D2RunConfig,
    stage7d2_run_fingerprint,
)


class Stage7D2ProfileTests(unittest.TestCase):
    def test_exact_full_train_validation_surface_is_frozen(self) -> None:
        config = STAGE7D2_FROZEN_RUN_CONFIG
        self.assertEqual(config.epochs, 40)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.train_samples, 1230)
        self.assertEqual(config.validation_samples, 153)
        self.assertEqual(config.train_families, 410)
        self.assertEqual(config.validation_families, 51)
        self.assertEqual(config.max_decode_tokens, 1536)
        self.assertEqual(config.decode_measure_count, 8)
        self.assertEqual(config.retained_checkpoints, 1)

    def test_profile_rejects_training_surface_drift(self) -> None:
        with self.assertRaises(ValueError):
            Stage7D2RunConfig(train_samples=1229)
        with self.assertRaises(ValueError):
            Stage7D2RunConfig(validation_samples=154)
        with self.assertRaises(ValueError):
            Stage7D2RunConfig(epochs=39)

    def test_model_trainer_preprocess_are_held_equal_to_stage7c(self) -> None:
        self.assertEqual(STAGE7D2_FROZEN_MODEL_CONFIG, STAGE7C_FROZEN_MODEL_CONFIG)
        self.assertEqual(STAGE7D2_FROZEN_TRAINER_CONFIG, STAGE7C_FROZEN_TRAINER_CONFIG)
        self.assertEqual(STAGE7D2_FROZEN_PREPROCESS_CONFIG, STAGE7C_FROZEN_PREPROCESS_CONFIG)

    def test_run_fingerprint_and_d1_binding_are_frozen(self) -> None:
        self.assertEqual(STAGE7D2_FROZEN_RUN_FINGERPRINT, stage7d2_run_fingerprint())
        self.assertEqual(len(STAGE7D2_FROZEN_RUN_FINGERPRINT), 64)
        self.assertEqual(
            EXPECTED_D1_ARTIFACT_BINDING_SHA256,
            "e603b945c6dc60cf7e618ae28a7734dee97cf0e05a81891479107b18a87af540",
        )


if __name__ == "__main__":
    unittest.main()
