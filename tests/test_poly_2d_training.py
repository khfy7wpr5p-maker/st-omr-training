from __future__ import annotations

from dataclasses import replace
import math
import unittest

import torch

from st_omr_training.dataset_manifest import DatasetSplit
from st_omr_training.poly_2d_training import (
    FROZEN_POLY_2D_TRAINING_CONFIG,
    POLY_2D_PROVENANCE_VERSION,
    POLY_2D_TRAINER_VERSION,
    POLY_2D_TRAINING_BATCH_VERSION,
    Poly2DTrainingBatch,
    Poly2DTrainingError,
    Poly2DTrainingProvenance,
    Poly2DTrainingConfig,
    build_poly_2d_optimizer,
    build_poly_2d_training_provenance,
    evaluate_poly_2d_validation_loss,
    poly_2d_trainer_fingerprint,
    run_bounded_poly_2d_smoke_training,
    train_poly_2d_one_step,
)
from st_omr_training.poly_2d_transformer import (
    POLY_2D_TRANSFORMER_VERSION,
    Poly2DTransformerConfig,
    build_tiny_poly_2d_transformer,
    poly_2d_config_fingerprint,
)
from st_omr_training.polyphonic_representation import POLYPHONIC_REPRESENTATION_VERSION
from st_omr_training.polyphonic_serialization import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    POLYPHONIC_TOKENIZER_VERSION,
    TOKEN_TO_ID,
    tokenizer_fingerprint,
)
from st_omr_training.training_model import TORCH_PINNED_VERSION, model_state_sha256


_SHA = "1" * 64
_SHA2 = "2" * 64
_GIT = "a" * 40
_OBJ_START = TOKEN_TO_ID["OBJ_START"]
_OBJ_END = TOKEN_TO_ID["OBJ_END"]

_SMALL_MODEL = Poly2DTransformerConfig(
    input_height=32,
    input_width=128,
    patch_height=16,
    patch_width=16,
    model_dim=32,
    encoder_layers=1,
    decoder_layers=1,
    attention_heads=4,
    feedforward_dim=64,
    max_target_tokens=16,
)
_SMALL_TRAINER = Poly2DTrainingConfig(smoke_steps=2)


def _batch(split: DatasetSplit, *, image_value: float = 0.5, manifest: str = _SHA) -> Poly2DTrainingBatch:
    images = torch.full((1, 1, 32, 128), image_value, dtype=torch.float32)
    decoder = torch.tensor([[BOS_TOKEN_ID, _OBJ_START, _OBJ_END, PAD_TOKEN_ID]], dtype=torch.long)
    labels = torch.tensor([[_OBJ_START, _OBJ_END, EOS_TOKEN_ID, PAD_TOKEN_ID]], dtype=torch.long)
    return Poly2DTrainingBatch(
        images=images,
        decoder_input_ids=decoder,
        labels=labels,
        split=split,
        sample_ids=(f"sample-{split.value}",),
        dataset_manifest_sha256=manifest,
    )


def _provenance(manifest: str = _SHA):
    return build_poly_2d_training_provenance(
        repository_sha=_GIT,
        dataset_manifest_sha256=manifest,
        preprocess_fingerprint_sha256=_SHA2,
        training_config=_SMALL_TRAINER,
        model_config=_SMALL_MODEL,
    )


class Poly2DTrainingTests(unittest.TestCase):
    def test_versions_and_profiles_bind_frozen_v2_surface(self) -> None:
        self.assertEqual(POLY_2D_TRAINER_VERSION, "st-omr-poly-2d-trainer-v1")
        self.assertEqual(POLY_2D_TRAINING_BATCH_VERSION, "st-omr-poly-2d-training-batch-v1")
        self.assertEqual(POLY_2D_PROVENANCE_VERSION, "st-omr-poly-2d-training-provenance-v1")
        provenance = _provenance()
        self.assertEqual(provenance.model_profile_sha256, poly_2d_config_fingerprint(_SMALL_MODEL))
        self.assertEqual(
            provenance.trainer_profile_sha256,
            poly_2d_trainer_fingerprint(_SMALL_TRAINER, _SMALL_MODEL),
        )
        self.assertEqual(provenance.tokenizer_fingerprint_sha256, tokenizer_fingerprint())
        self.assertEqual(provenance.representation_version, POLYPHONIC_REPRESENTATION_VERSION)
        self.assertEqual(provenance.tokenizer_version, POLYPHONIC_TOKENIZER_VERSION)
        self.assertEqual(provenance.torch_version, TORCH_PINNED_VERSION)
        self.assertEqual(len(provenance.fingerprint()), 64)

    def test_batch_rejects_test_and_bad_teacher_forcing_boundaries(self) -> None:
        with self.assertRaisesRegex(Poly2DTrainingError, "sealed TEST"):
            _batch(DatasetSplit.TEST)

        valid = _batch(DatasetSplit.TRAIN)
        bad_bos = valid.decoder_input_ids.clone()
        bad_bos[0, 0] = _OBJ_START
        with self.assertRaisesRegex(Poly2DTrainingError, "start with V2 BOS"):
            replace(valid, decoder_input_ids=bad_bos)

        bad_labels = valid.labels.clone()
        bad_labels[0, 2] = PAD_TOKEN_ID
        bad_labels[0, 3] = EOS_TOKEN_ID
        with self.assertRaisesRegex(Poly2DTrainingError, "contiguous right padding"):
            replace(valid, labels=bad_labels)

    def test_batch_rejects_nonfinite_out_of_range_and_padding_mask_drift(self) -> None:
        valid = _batch(DatasetSplit.TRAIN)
        nonfinite = valid.images.clone()
        nonfinite[0, 0, 0, 0] = float("nan")
        with self.assertRaises(Poly2DTrainingError):
            replace(valid, images=nonfinite)

        out_of_range = valid.images.clone()
        out_of_range[0, 0, 0, 0] = 1.1
        with self.assertRaisesRegex(Poly2DTrainingError, "normalized"):
            replace(valid, images=out_of_range)

        mask_drift = valid.labels.clone()
        mask_drift[0, 3] = EOS_TOKEN_ID
        with self.assertRaisesRegex(Poly2DTrainingError, "padding mask"):
            replace(valid, labels=mask_drift)

    def test_train_step_mutates_only_train_and_validation_is_read_only(self) -> None:
        model = build_tiny_poly_2d_transformer(_SMALL_MODEL, seed=82_081)
        optimizer = build_poly_2d_optimizer(model, _SMALL_TRAINER)
        train_batch = _batch(DatasetSplit.TRAIN)
        before = model_state_sha256(model)
        loss = train_poly_2d_one_step(model, train_batch, optimizer, _SMALL_TRAINER)
        after = model_state_sha256(model)
        self.assertTrue(math.isfinite(loss))
        self.assertNotEqual(before, after)

        validation_batch = _batch(DatasetSplit.VALIDATION, image_value=0.4)
        validation_before = model_state_sha256(model)
        validation_loss = evaluate_poly_2d_validation_loss(model, validation_batch)
        validation_after = model_state_sha256(model)
        self.assertTrue(math.isfinite(validation_loss))
        self.assertEqual(validation_before, validation_after)

        with self.assertRaisesRegex(Poly2DTrainingError, "TRAIN split"):
            train_poly_2d_one_step(model, validation_batch, optimizer, _SMALL_TRAINER)
        with self.assertRaisesRegex(Poly2DTrainingError, "VALIDATION split"):
            evaluate_poly_2d_validation_loss(model, train_batch)

    def test_smoke_run_is_deterministic_bounded_and_never_claims_checkpoint_or_test(self) -> None:
        train = (_batch(DatasetSplit.TRAIN),)
        validation = _batch(DatasetSplit.VALIDATION, image_value=0.4)
        provenance = _provenance()
        result_a = run_bounded_poly_2d_smoke_training(
            train_batches=train,
            validation_batch=validation,
            provenance=provenance,
            training_config=_SMALL_TRAINER,
            model_config=_SMALL_MODEL,
        )
        result_b = run_bounded_poly_2d_smoke_training(
            train_batches=train,
            validation_batch=validation,
            provenance=provenance,
            training_config=_SMALL_TRAINER,
            model_config=_SMALL_MODEL,
        )
        self.assertEqual(result_a.initial_state_sha256, result_b.initial_state_sha256)
        self.assertEqual(result_a.final_state_sha256, result_b.final_state_sha256)
        self.assertEqual(result_a.train_losses, result_b.train_losses)
        self.assertEqual(result_a.validation_loss, result_b.validation_loss)
        self.assertEqual(result_a.optimizer_steps, 2)
        self.assertFalse(result_a.checkpoint_written)
        self.assertFalse(result_a.authoritative_dataset_execution)
        self.assertFalse(result_a.test_split_accessed)
        self.assertEqual(result_a.dataset_manifest_sha256, _SHA)
        self.assertEqual(result_a.provenance_sha256, provenance.fingerprint())

    def test_provenance_and_batch_dataset_drift_fail_closed(self) -> None:
        provenance = _provenance()
        with self.assertRaisesRegex(Poly2DTrainingError, "tokenizer fingerprint"):
            replace(provenance, tokenizer_fingerprint_sha256=_SHA2)
        with self.assertRaisesRegex(Poly2DTrainingError, "git SHA"):
            replace(provenance, repository_sha="a" * 39)

        train = (_batch(DatasetSplit.TRAIN, manifest=_SHA2),)
        validation = _batch(DatasetSplit.VALIDATION, image_value=0.4, manifest=_SHA2)
        with self.assertRaisesRegex(Poly2DTrainingError, "dataset identity"):
            run_bounded_poly_2d_smoke_training(
                train_batches=train,
                validation_batch=validation,
                provenance=provenance,
                training_config=_SMALL_TRAINER,
                model_config=_SMALL_MODEL,
            )

    def test_model_or_trainer_profile_drift_fails_closed(self) -> None:
        provenance = _provenance()
        train = (_batch(DatasetSplit.TRAIN),)
        validation = _batch(DatasetSplit.VALIDATION, image_value=0.4)
        drifted_model = replace(_SMALL_MODEL, feedforward_dim=96)
        with self.assertRaisesRegex(Poly2DTrainingError, "model profile"):
            run_bounded_poly_2d_smoke_training(
                train_batches=train,
                validation_batch=validation,
                provenance=provenance,
                training_config=_SMALL_TRAINER,
                model_config=drifted_model,
            )
        drifted_trainer = replace(_SMALL_TRAINER, learning_rate_micros=501)
        with self.assertRaisesRegex(Poly2DTrainingError, "trainer profile"):
            run_bounded_poly_2d_smoke_training(
                train_batches=train,
                validation_batch=validation,
                provenance=provenance,
                training_config=drifted_trainer,
                model_config=_SMALL_MODEL,
            )

    def test_harness_is_bound_to_transformer_model_identity(self) -> None:
        self.assertEqual(POLY_2D_TRANSFORMER_VERSION, "st-omr-poly-2d-transformer-v1")
        self.assertEqual(FROZEN_POLY_2D_TRAINING_CONFIG.smoke_steps, 2)


if __name__ == "__main__":
    unittest.main()
