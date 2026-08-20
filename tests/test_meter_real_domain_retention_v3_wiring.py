from __future__ import annotations

import inspect
import unittest

from st_omr_training.meter_real_domain_adaptation_v2 import FROZEN_ADAPTATION_CONFIG_V2
from st_omr_training.meter_real_domain_retention_v3 import (
    EARLY_LEARNING_RATE_MICROS_V3,
    EXPECTED_BATCHES_PER_EPOCH_V3,
    LATE_LEARNING_RATE_MICROS_V3,
    TOTAL_EPOCHS_V3,
    run_meter_real_domain_retention_v3,
)


class MeterRealDomainRetentionV3WiringTests(unittest.TestCase):
    def test_v2_gates_and_non_lr_training_surface_remain_frozen(self) -> None:
        config = FROZEN_ADAPTATION_CONFIG_V2
        self.assertEqual(config.epochs, TOTAL_EPOCHS_V3)
        self.assertEqual(config.learning_rate_micros, EARLY_LEARNING_RATE_MICROS_V3)
        self.assertEqual(EXPECTED_BATCHES_PER_EPOCH_V3, 30)
        self.assertEqual(LATE_LEARNING_RATE_MICROS_V3, 250)
        self.assertEqual(config.synthetic_replay_per_class, 128)
        self.assertEqual(config.real_balanced_repeat_factor, 4)
        self.assertEqual(config.real_min_macro_f1_milli, 900)
        self.assertEqual(config.real_min_accuracy_milli, 900)
        self.assertEqual(config.real_min_none_recall_milli, 888)
        self.assertEqual(config.real_min_positive_class_recall_milli, 999)
        self.assertEqual(config.synthetic_max_macro_f1_drop_milli, 20)
        self.assertEqual(config.synthetic_max_localization_drop_milli, 30)
        self.assertEqual(config.trainable_surface, "glyph-adapter-only-d11-fully-frozen")

    def test_runtime_wrapper_uses_scoped_public_optimizer_hook(self) -> None:
        source = inspect.getsource(run_meter_real_domain_retention_v3)
        self.assertIn("register_optimizer_step_pre_hook", source)
        self.assertIn("config=FROZEN_ADAPTATION_CONFIG_V2", source.replace("config=config", "config=FROZEN_ADAPTATION_CONFIG_V2") if False else source)
        self.assertIn("config=config", source)
        self.assertIn("finally:", source)
        self.assertIn("handle.remove()", source)
        self.assertIn("metrics.get(\"test_opened\") is not False", source)
        self.assertIn("metrics.get(\"production_promotion_authorized\") is not False", source)

    def test_runner_does_not_open_test_or_authorize_promotion(self) -> None:
        import tools.meter_real_domain_background_runner_v3 as runner

        source = inspect.getsource(runner)
        self.assertNotIn("test_opened=True", source)
        self.assertNotIn("production_promotion_authorized=True", source)
        self.assertIn("run_meter_real_domain_retention_v3", source)


if __name__ == "__main__":
    unittest.main()
