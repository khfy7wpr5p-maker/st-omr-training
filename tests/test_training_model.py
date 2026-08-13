from __future__ import annotations

import math
import unittest

import torch

from st_omr_training.dataset_manifest import DatasetSplit
from st_omr_training.training_data import TrainingBatch
from st_omr_training.training_model import (
    MAX_TRAINABLE_PARAMETERS,
    TORCH_PINNED_VERSION,
    BaselineModelConfig,
    TrainerConfig,
    TrainingConfigError,
    TrainingRuntimeError,
    assert_optimizer_finite,
    build_baseline_model,
    count_trainable_parameters,
    model_state_sha256,
    run_deterministic_cpu_smoke,
    validation_loss,
    verify_torch_runtime,
)
from st_omr_training.training_tokens import encode_tokens


def make_batch(split: DatasetSplit) -> TrainingBatch:
    token_ids = encode_tokens(
        (
            "BOS",
            "MEASURE_START",
            "TS_4_4",
            "REST",
            "DUR_WHOLE",
            "MEASURE_END",
            "EOS",
        )
    )
    images = torch.full((1, 1, 64, 512), 0.5, dtype=torch.float32)
    decoder = torch.tensor([token_ids[:-1]], dtype=torch.long)
    labels = torch.tensor([token_ids[1:]], dtype=torch.long)
    return TrainingBatch(images, decoder, labels, split)


class RuntimeAndModelTests(unittest.TestCase):
    def test_exact_cpu_torch_runtime_is_pinned(self) -> None:
        self.assertEqual(verify_torch_runtime(), TORCH_PINNED_VERSION)

    def test_model_is_bounded_and_initialization_is_seed_deterministic(self) -> None:
        first = build_baseline_model(seed=123)
        second = build_baseline_model(seed=123)
        count = count_trainable_parameters(first)
        self.assertGreater(count, 0)
        self.assertLessEqual(count, MAX_TRAINABLE_PARAMETERS)
        self.assertEqual(model_state_sha256(first), model_state_sha256(second))

    def test_forward_shape_is_finite(self) -> None:
        batch = make_batch(DatasetSplit.TRAIN)
        model = build_baseline_model(seed=1)
        logits = model(batch.images, batch.decoder_input_ids)
        self.assertEqual(tuple(logits.shape[:2]), tuple(batch.labels.shape))
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_incremental_decode_matches_full_sequence_logits(self) -> None:
        batch = make_batch(DatasetSplit.TRAIN)
        model = build_baseline_model(seed=314)
        model.eval()

        with torch.no_grad():
            expected = model(batch.images, batch.decoder_input_ids)
            conditioning, hidden = model.begin_incremental_decode(batch.images)
            observed_steps = []
            for index in range(batch.decoder_input_ids.shape[1]):
                step_logits, hidden = model.decode_incremental_step(
                    batch.decoder_input_ids[:, index : index + 1],
                    conditioning,
                    hidden,
                )
                observed_steps.append(step_logits)
            observed = torch.cat(observed_steps, dim=1)

        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-7)
        self.assertTrue(
            torch.equal(
                torch.argmax(observed, dim=-1),
                torch.argmax(expected, dim=-1),
            )
        )

        with self.assertRaises(TrainingRuntimeError):
            model.decode_incremental_step(
                batch.decoder_input_ids[:, :2],
                conditioning,
                hidden,
            )

    def test_nan_input_fails_closed(self) -> None:
        batch = make_batch(DatasetSplit.TRAIN)
        model = build_baseline_model(seed=1)
        images = batch.images.clone()
        images[0, 0, 0, 0] = float("nan")
        with self.assertRaises(TrainingRuntimeError):
            model(images, batch.decoder_input_ids)

    def test_optimizer_nan_state_fails_closed(self) -> None:
        model = build_baseline_model(seed=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False, fused=False)
        parameter = next(model.parameters())
        optimizer.state[parameter]["corrupt"] = torch.tensor(float("nan"))
        with self.assertRaises(TrainingRuntimeError):
            assert_optimizer_finite(optimizer)


class TrainerTests(unittest.TestCase):
    def test_cpu_smoke_replay_is_exact_for_same_seed(self) -> None:
        batch = make_batch(DatasetSplit.TRAIN)
        config = TrainerConfig(master_seed=991, smoke_steps=2)
        first = run_deterministic_cpu_smoke(batch, trainer_config=config)
        second = run_deterministic_cpu_smoke(batch, trainer_config=config)
        self.assertEqual(first, second)
        self.assertNotEqual(first.initial_state_sha256, first.final_state_sha256)
        self.assertTrue(all(math.isfinite(loss) for loss in first.losses))

    def test_gradient_update_rejects_validation_split(self) -> None:
        with self.assertRaises(TrainingRuntimeError):
            run_deterministic_cpu_smoke(make_batch(DatasetSplit.VALIDATION))

    def test_validation_loss_is_finite_and_does_not_mutate_model(self) -> None:
        batch = make_batch(DatasetSplit.VALIDATION)
        model = build_baseline_model(seed=88)
        before = model_state_sha256(model)
        value = validation_loss(model, batch)
        after = model_state_sha256(model)
        self.assertTrue(math.isfinite(value))
        self.assertEqual(before, after)

    def test_trainer_config_is_fail_closed(self) -> None:
        with self.assertRaises(TrainingConfigError):
            TrainerConfig(master_seed=True)
        with self.assertRaises(TrainingConfigError):
            TrainerConfig(scheduler="cosine")
        with self.assertRaises(TrainingConfigError):
            BaselineModelConfig(input_height=True)


if __name__ == "__main__":
    unittest.main()
