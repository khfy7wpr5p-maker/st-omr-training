from __future__ import annotations

import unittest

from st_omr_training.stage7c_benchmark import estimate_runtime_seconds


class Stage7CBenchmarkEstimateTests(unittest.TestCase):
    def test_estimate_scales_every_frozen_work_unit_and_adds_safety_margin(self) -> None:
        estimate = estimate_runtime_seconds(
            dataset_seconds=60.0,
            train_step_seconds=0.5,
            validation_batch_seconds=0.2,
            decode_sample_seconds=1.0,
            training_steps=100,
            validation_batches=10,
            validation_passes=3,
            validation_samples=5,
            fixed_overhead_seconds=300.0,
            safety_factor=2.0,
        )
        self.assertEqual(estimate["projected_seconds"], 421.0)
        self.assertEqual(estimate["safety_adjusted_seconds"], 842.0)

    def test_estimate_rejects_nonpositive_or_boolean_inputs(self) -> None:
        valid = {
            "dataset_seconds": 1.0,
            "train_step_seconds": 1.0,
            "validation_batch_seconds": 1.0,
            "decode_sample_seconds": 1.0,
            "training_steps": 1,
            "validation_batches": 1,
            "validation_passes": 1,
            "validation_samples": 1,
            "fixed_overhead_seconds": 1.0,
            "safety_factor": 2.0,
        }
        for name, value in (("training_steps", True), ("safety_factor", 0.0)):
            invalid = dict(valid)
            invalid[name] = value
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                estimate_runtime_seconds(**invalid)


if __name__ == "__main__":
    unittest.main()
