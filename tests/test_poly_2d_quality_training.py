from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from st_omr_training.dataset_manifest import DatasetSplit
from st_omr_training.poly_2d_quality_training import (
    MAX_QUALITY_OPTIMIZER_STEPS,
    POLY_2D_QUALITY_PLAN_VERSION,
    POLY_2D_QUALITY_PROVENANCE_VERSION,
    POLY_2D_QUALITY_TRAINER_VERSION,
    Poly2DQualityTrainingConfig,
    Poly2DQualityTrainingError,
    build_poly_2d_quality_training_provenance,
    poly_2d_quality_trainer_fingerprint,
    quality_step_config,
    quality_training_plan_fingerprint,
    run_poly_2d_quality_training,
)
from st_omr_training.poly_2d_training import Poly2DTrainingBatch
from st_omr_training.poly_2d_transformer import (
    Poly2DTransformerConfig,
    poly_2d_config_fingerprint,
)
from st_omr_training.polyphonic_serialization import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    TOKEN_TO_ID,
    tokenizer_fingerprint,
)
from st_omr_training.training_model import model_state_sha256


_MANIFEST = "1" * 64
_PREPROCESS = "2" * 64
_REPOSITORY = "a" * 40
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
_SMALL_QUALITY = Poly2DQualityTrainingConfig(
    epochs=3,
    max_optimizer_steps=16,
)


def _batch(
    split: DatasetSplit,
    label: str,
    *,
    image_value: float,
    manifest: str = _MANIFEST,
) -> Poly2DTrainingBatch:
    return Poly2DTrainingBatch(
        images=torch.full((1, 1, 32, 128), image_value, dtype=torch.float32),
        decoder_input_ids=torch.tensor(
            [[BOS_TOKEN_ID, _OBJ_START, _OBJ_END, PAD_TOKEN_ID]],
            dtype=torch.long,
        ),
        labels=torch.tensor(
            [[_OBJ_START, _OBJ_END, EOS_TOKEN_ID, PAD_TOKEN_ID]],
            dtype=torch.long,
        ),
        split=split,
        sample_ids=(label,),
        dataset_manifest_sha256=manifest,
    )


def _provenance(config: Poly2DQualityTrainingConfig = _SMALL_QUALITY):
    return build_poly_2d_quality_training_provenance(
        repository_sha=_REPOSITORY,
        dataset_manifest_sha256=_MANIFEST,
        preprocess_fingerprint_sha256=_PREPROCESS,
        quality_config=config,
        model_config=_SMALL_MODEL,
    )


class Poly2DQualityTrainingTests(unittest.TestCase):
    def test_versions_and_recipe_identity_are_separate_from_smoke_harness(self) -> None:
        self.assertEqual(
            POLY_2D_QUALITY_TRAINER_VERSION,
            "st-omr-poly-2d-quality-trainer-v1",
        )
        self.assertEqual(
            POLY_2D_QUALITY_PROVENANCE_VERSION,
            "st-omr-poly-2d-quality-training-provenance-v1",
        )
        self.assertEqual(
            POLY_2D_QUALITY_PLAN_VERSION,
            "st-omr-poly-2d-quality-training-plan-v1",
        )
        provenance = _provenance()
        self.assertEqual(
            provenance.model_profile_sha256,
            poly_2d_config_fingerprint(_SMALL_MODEL),
        )
        self.assertEqual(
            provenance.quality_trainer_profile_sha256,
            poly_2d_quality_trainer_fingerprint(_SMALL_QUALITY, _SMALL_MODEL),
        )
        self.assertEqual(provenance.tokenizer_fingerprint_sha256, tokenizer_fingerprint())
        self.assertEqual(quality_step_config(_SMALL_QUALITY).smoke_steps, 1)

    def test_quality_config_allows_multi_epoch_but_remains_bounded(self) -> None:
        config = Poly2DQualityTrainingConfig(epochs=32, max_optimizer_steps=50_000)
        self.assertEqual(config.epochs, 32)
        self.assertEqual(config.max_optimizer_steps, 50_000)
        with self.assertRaises(ValueError):
            Poly2DQualityTrainingConfig(epochs=0)
        with self.assertRaises(ValueError):
            Poly2DQualityTrainingConfig(
                max_optimizer_steps=MAX_QUALITY_OPTIMIZER_STEPS + 1
            )
        with self.assertRaises(ValueError):
            Poly2DQualityTrainingConfig(validation_interval_epochs=2)

    def test_multi_epoch_run_is_deterministic_and_returns_best_validation_state(self) -> None:
        train = (
            _batch(DatasetSplit.TRAIN, "train-a", image_value=0.2),
            _batch(DatasetSplit.TRAIN, "train-b", image_value=0.8),
        )
        validation = (
            _batch(DatasetSplit.VALIDATION, "validation-a", image_value=0.4),
        )
        config = Poly2DQualityTrainingConfig(epochs=3, max_optimizer_steps=16)
        provenance = _provenance(config)

        first = run_poly_2d_quality_training(
            train_batches=train,
            validation_batches=validation,
            provenance=provenance,
            quality_config=config,
            model_config=_SMALL_MODEL,
        )
        second = run_poly_2d_quality_training(
            train_batches=train,
            validation_batches=validation,
            provenance=provenance,
            quality_config=config,
            model_config=_SMALL_MODEL,
        )

        self.assertEqual(first.result.fingerprint(), second.result.fingerprint())
        self.assertEqual(first.result.optimizer_steps, 6)
        self.assertEqual(len(first.result.epoch_evidence), 3)
        expected_epoch = min(
            first.result.epoch_evidence,
            key=lambda item: (item.validation_mean_loss, item.epoch),
        )
        self.assertEqual(first.result.selected_epoch, expected_epoch.epoch)
        self.assertEqual(
            first.result.selected_state_sha256,
            expected_epoch.model_state_sha256,
        )
        self.assertEqual(
            model_state_sha256(first.model),
            first.result.selected_state_sha256,
        )
        self.assertFalse(first.result.test_split_accessed)
        self.assertFalse(first.result.production_authority)

    def test_training_plan_fingerprint_binds_order_and_split_membership(self) -> None:
        a = _batch(DatasetSplit.TRAIN, "train-a", image_value=0.2)
        b = _batch(DatasetSplit.TRAIN, "train-b", image_value=0.8)
        validation = _batch(
            DatasetSplit.VALIDATION, "validation-a", image_value=0.4
        )
        first = quality_training_plan_fingerprint((a, b), (validation,))
        second = quality_training_plan_fingerprint((a, b), (validation,))
        reversed_plan = quality_training_plan_fingerprint((b, a), (validation,))
        self.assertEqual(first, second)
        self.assertNotEqual(first, reversed_plan)

    def test_step_bound_is_checked_before_training(self) -> None:
        train = (
            _batch(DatasetSplit.TRAIN, "train-a", image_value=0.2),
            _batch(DatasetSplit.TRAIN, "train-b", image_value=0.8),
        )
        validation = (
            _batch(DatasetSplit.VALIDATION, "validation-a", image_value=0.4),
        )
        config = Poly2DQualityTrainingConfig(epochs=3, max_optimizer_steps=5)
        provenance = _provenance(config)
        with self.assertRaisesRegex(
            Poly2DQualityTrainingError, "exceeds max_optimizer_steps"
        ):
            run_poly_2d_quality_training(
                train_batches=train,
                validation_batches=validation,
                provenance=provenance,
                quality_config=config,
                model_config=_SMALL_MODEL,
            )

    def test_dataset_identity_and_duplicate_sample_leakage_fail_closed(self) -> None:
        train = (
            _batch(DatasetSplit.TRAIN, "shared", image_value=0.2),
        )
        validation_duplicate = (
            _batch(DatasetSplit.VALIDATION, "shared", image_value=0.4),
        )
        with self.assertRaisesRegex(Poly2DQualityTrainingError, "duplicate sample IDs"):
            run_poly_2d_quality_training(
                train_batches=train,
                validation_batches=validation_duplicate,
                provenance=_provenance(),
                quality_config=_SMALL_QUALITY,
                model_config=_SMALL_MODEL,
            )

        validation_drift = (
            _batch(
                DatasetSplit.VALIDATION,
                "validation-a",
                image_value=0.4,
                manifest="3" * 64,
            ),
        )
        with self.assertRaisesRegex(Poly2DQualityTrainingError, "dataset identity"):
            run_poly_2d_quality_training(
                train_batches=train,
                validation_batches=validation_drift,
                provenance=_provenance(),
                quality_config=_SMALL_QUALITY,
                model_config=_SMALL_MODEL,
            )

    def test_provenance_drift_and_test_split_remain_closed(self) -> None:
        provenance = _provenance()
        with self.assertRaisesRegex(Poly2DQualityTrainingError, "trainer profile"):
            run_poly_2d_quality_training(
                train_batches=(
                    _batch(DatasetSplit.TRAIN, "train-a", image_value=0.2),
                ),
                validation_batches=(
                    _batch(
                        DatasetSplit.VALIDATION,
                        "validation-a",
                        image_value=0.4,
                    ),
                ),
                provenance=replace(
                    provenance,
                    quality_trainer_profile_sha256="4" * 64,
                ),
                quality_config=_SMALL_QUALITY,
                model_config=_SMALL_MODEL,
            )

        with self.assertRaisesRegex(Exception, "TEST"):
            _batch(DatasetSplit.TEST, "test-a", image_value=0.5)


if __name__ == "__main__":
    unittest.main()
