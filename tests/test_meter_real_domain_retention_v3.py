from __future__ import annotations

import unittest

from st_omr_training.meter_real_domain_retention_v3 import (
    EARLY_LEARNING_RATE_MICROS_V3,
    EXPECTED_BATCHES_PER_EPOCH_V3,
    LATE_LEARNING_RATE_MICROS_V3,
    MIDPOINT_DECAY_EPOCH_V3,
    TOTAL_EPOCHS_V3,
    TOTAL_OPTIMIZER_STEPS_V3,
    MeterRealDomainRetentionV3Error,
    apply_learning_rate_v3,
    learning_rate_micros_for_epoch_v3,
    learning_rate_micros_for_optimizer_step_v3,
    schedule_fingerprint_payload_v3,
)


class _OptimizerStub:
    def __init__(self) -> None:
        self.param_groups = [{"lr": 99.0}, {"lr": 98.0}]


class MeterRealDomainRetentionV3Tests(unittest.TestCase):
    def test_schedule_is_exact_and_midpoint_only(self) -> None:
        self.assertEqual(TOTAL_EPOCHS_V3, 20)
        self.assertEqual(EXPECTED_BATCHES_PER_EPOCH_V3, 30)
        self.assertEqual(TOTAL_OPTIMIZER_STEPS_V3, 600)
        self.assertEqual(MIDPOINT_DECAY_EPOCH_V3, 11)
        self.assertEqual(EARLY_LEARNING_RATE_MICROS_V3, 1000)
        self.assertEqual(LATE_LEARNING_RATE_MICROS_V3, 250)
        self.assertTrue(
            all(learning_rate_micros_for_epoch_v3(epoch) == 1000 for epoch in range(1, 11))
        )
        self.assertTrue(
            all(learning_rate_micros_for_epoch_v3(epoch) == 250 for epoch in range(11, 21))
        )

    def test_optimizer_step_mapping_matches_epoch_boundary(self) -> None:
        self.assertEqual(learning_rate_micros_for_optimizer_step_v3(1), 1000)
        self.assertEqual(learning_rate_micros_for_optimizer_step_v3(300), 1000)
        self.assertEqual(learning_rate_micros_for_optimizer_step_v3(301), 250)
        self.assertEqual(learning_rate_micros_for_optimizer_step_v3(600), 250)

    def test_schedule_rejects_noncanonical_epochs_and_steps(self) -> None:
        for value in (0, 21, -1, True, 1.5, "11"):
            with self.subTest(epoch=value):
                with self.assertRaises(MeterRealDomainRetentionV3Error):
                    learning_rate_micros_for_epoch_v3(value)  # type: ignore[arg-type]
        for value in (0, 601, -1, True, 1.5, "301"):
            with self.subTest(step=value):
                with self.assertRaises(MeterRealDomainRetentionV3Error):
                    learning_rate_micros_for_optimizer_step_v3(value)  # type: ignore[arg-type]

    def test_apply_updates_all_param_groups_deterministically(self) -> None:
        optimizer = _OptimizerStub()
        self.assertEqual(apply_learning_rate_v3(optimizer, 1), 1000)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.001, 0.001])
        self.assertEqual(apply_learning_rate_v3(optimizer, 11), 250)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.00025, 0.00025])
        self.assertEqual(apply_learning_rate_v3(optimizer, 20), 250)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.00025, 0.00025])

    def test_apply_fails_closed_on_malformed_optimizer(self) -> None:
        class MissingGroups:
            pass

        class EmptyGroups:
            param_groups: list[dict[str, float]] = []

        class BadGroup:
            param_groups = [{}]

        for optimizer in (MissingGroups(), EmptyGroups(), BadGroup()):
            with self.subTest(optimizer=type(optimizer).__name__):
                with self.assertRaises(MeterRealDomainRetentionV3Error):
                    apply_learning_rate_v3(optimizer, 1)

    def test_schedule_fingerprint_payload_is_frozen_primitive_data(self) -> None:
        self.assertEqual(
            schedule_fingerprint_payload_v3(),
            {
                "version": "meter-real-domain-retention-v3",
                "total_epochs": 20,
                "batches_per_epoch": 30,
                "total_optimizer_steps": 600,
                "midpoint_decay_epoch": 11,
                "early_learning_rate_micros": 1000,
                "late_learning_rate_micros": 250,
            },
        )


if __name__ == "__main__":
    unittest.main()
