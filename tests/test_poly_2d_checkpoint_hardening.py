from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from st_omr_training.dataset_manifest import DatasetSplit
from st_omr_training.poly_2d_checkpoint import (
    Poly2DCheckpointError,
    load_and_verify_poly_2d_checkpoint,
    run_and_persist_bounded_poly_2d_checkpoint,
)
from st_omr_training.poly_2d_training import (
    Poly2DTrainingBatch,
    Poly2DTrainingConfig,
    build_poly_2d_training_provenance,
)
from st_omr_training.poly_2d_transformer import Poly2DTransformerConfig
from st_omr_training.polyphonic_serialization import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    TOKEN_TO_ID,
)
from st_omr_training.training_model import TrainingRuntimeError


_DATASET_SHA = "1" * 64
_PREPROCESS_SHA = "2" * 64
_REPOSITORY_SHA = "a" * 40
_OBJ_START = TOKEN_TO_ID["OBJ_START"]
_OBJ_END = TOKEN_TO_ID["OBJ_END"]

_MODEL_CONFIG = Poly2DTransformerConfig(
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
_TRAINING_CONFIG = Poly2DTrainingConfig(smoke_steps=1)


def _batch(split: DatasetSplit, image_value: float) -> Poly2DTrainingBatch:
    return Poly2DTrainingBatch(
        images=torch.full((1, 1, 32, 128), image_value, dtype=torch.float32),
        decoder_input_ids=torch.tensor(
            [[BOS_TOKEN_ID, _OBJ_START, _OBJ_END, PAD_TOKEN_ID]], dtype=torch.long
        ),
        labels=torch.tensor(
            [[_OBJ_START, _OBJ_END, EOS_TOKEN_ID, PAD_TOKEN_ID]], dtype=torch.long
        ),
        split=split,
        sample_ids=(f"hardening-{split.value}",),
        dataset_manifest_sha256=_DATASET_SHA,
    )


def _persist(root: Path) -> Path:
    provenance = build_poly_2d_training_provenance(
        repository_sha=_REPOSITORY_SHA,
        dataset_manifest_sha256=_DATASET_SHA,
        preprocess_fingerprint_sha256=_PREPROCESS_SHA,
        training_config=_TRAINING_CONFIG,
        model_config=_MODEL_CONFIG,
    )
    output = root / "artifact"
    run_and_persist_bounded_poly_2d_checkpoint(
        train_batches=(_batch(DatasetSplit.TRAIN, 0.5),),
        validation_batch=_batch(DatasetSplit.VALIDATION, 0.4),
        provenance=provenance,
        output_directory=output,
        training_config=_TRAINING_CONFIG,
        model_config=_MODEL_CONFIG,
    )
    return output


class Poly2DCheckpointHardeningTests(unittest.TestCase):
    def test_artifact_directory_rejects_any_extra_entry(self) -> None:
        with TemporaryDirectory() as temporary:
            output = _persist(Path(temporary))
            (output / "unexpected.txt").write_text("not part of the checkpoint contract", encoding="utf-8")
            with self.assertRaisesRegex(Poly2DCheckpointError, "must contain exactly"):
                load_and_verify_poly_2d_checkpoint(output)

    def test_finite_state_failure_is_wrapped_in_checkpoint_error(self) -> None:
        with TemporaryDirectory() as temporary:
            output = _persist(Path(temporary))
            with patch(
                "st_omr_training.poly_2d_checkpoint.assert_model_finite",
                side_effect=TrainingRuntimeError("non-finite test witness"),
            ):
                with self.assertRaisesRegex(Poly2DCheckpointError, "contains non-finite model state"):
                    load_and_verify_poly_2d_checkpoint(output)


if __name__ == "__main__":
    unittest.main()
