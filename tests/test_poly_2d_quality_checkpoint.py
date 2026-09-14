from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from st_omr_training.dataset_manifest import DatasetSplit
from st_omr_training.poly_2d_quality_checkpoint import (
    POLY_2D_QUALITY_CHECKPOINT_RECEIPT_VERSION,
    POLY_2D_QUALITY_CHECKPOINT_SCHEMA_VERSION,
    POLY_2D_QUALITY_REGISTRY_RECORD_ID,
    QUALITY_METADATA_FILENAME,
    QUALITY_RECEIPT_FILENAME,
    QUALITY_TRAINING_RESULT_FILENAME,
    Poly2DQualityCheckpointError,
    load_and_verify_poly_2d_quality_checkpoint,
    persist_poly_2d_quality_checkpoint,
    run_verified_poly_2d_quality_checkpoint_inference,
)
from st_omr_training.poly_2d_quality_training import (
    Poly2DQualityTrainingConfig,
    build_poly_2d_quality_training_provenance,
    run_poly_2d_quality_training,
)
from st_omr_training.poly_2d_training import Poly2DTrainingBatch
from st_omr_training.poly_2d_transformer import Poly2DTransformerConfig
from st_omr_training.polyphonic_serialization import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    TOKEN_TO_ID,
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
_SMALL_QUALITY = Poly2DQualityTrainingConfig(epochs=2, max_optimizer_steps=8)


def _batch(split: DatasetSplit, label: str, image_value: float) -> Poly2DTrainingBatch:
    return Poly2DTrainingBatch(
        images=torch.full((1, 1, 32, 128), image_value, dtype=torch.float32),
        decoder_input_ids=torch.tensor(
            [[BOS_TOKEN_ID, _OBJ_START, _OBJ_END, PAD_TOKEN_ID]], dtype=torch.long
        ),
        labels=torch.tensor(
            [[_OBJ_START, _OBJ_END, EOS_TOKEN_ID, PAD_TOKEN_ID]], dtype=torch.long
        ),
        split=split,
        sample_ids=(label,),
        dataset_manifest_sha256=_MANIFEST,
    )


def _make_run():
    provenance = build_poly_2d_quality_training_provenance(
        repository_sha=_REPOSITORY,
        dataset_manifest_sha256=_MANIFEST,
        preprocess_fingerprint_sha256=_PREPROCESS,
        quality_config=_SMALL_QUALITY,
        model_config=_SMALL_MODEL,
    )
    run = run_poly_2d_quality_training(
        train_batches=(
            _batch(DatasetSplit.TRAIN, "train-a", 0.2),
            _batch(DatasetSplit.TRAIN, "train-b", 0.8),
        ),
        validation_batches=(
            _batch(DatasetSplit.VALIDATION, "validation-a", 0.4),
        ),
        provenance=provenance,
        quality_config=_SMALL_QUALITY,
        model_config=_SMALL_MODEL,
    )
    return run, provenance


class Poly2DQualityCheckpointTests(unittest.TestCase):
    def test_versions_are_separate_from_smoke_checkpoint_contract(self) -> None:
        self.assertEqual(
            POLY_2D_QUALITY_CHECKPOINT_SCHEMA_VERSION,
            "st-omr-poly-2d-quality-checkpoint-v1",
        )
        self.assertEqual(
            POLY_2D_QUALITY_CHECKPOINT_RECEIPT_VERSION,
            "st-omr-poly-2d-quality-checkpoint-receipt-v1",
        )
        self.assertEqual(
            POLY_2D_QUALITY_REGISTRY_RECORD_ID,
            "candidate.poly-2d-transformer.quality-v1",
        )

    def test_persist_reload_is_exact_hash_bound_and_non_overwriting(self) -> None:
        run, provenance = _make_run()
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "candidate"
            loaded = persist_poly_2d_quality_checkpoint(
                directory,
                run=run,
                provenance=provenance,
                quality_config=_SMALL_QUALITY,
                model_config=_SMALL_MODEL,
            )
            self.assertEqual(
                model_state_sha256(loaded.model), run.result.selected_state_sha256
            )
            self.assertEqual(
                loaded.metadata.training_result_fingerprint_sha256,
                run.result.fingerprint(),
            )
            self.assertEqual(
                loaded.metadata.training_plan_sha256,
                run.result.training_plan_sha256,
            )
            self.assertEqual(
                loaded.receipt.artifact_binding["record_id"],
                POLY_2D_QUALITY_REGISTRY_RECORD_ID,
            )
            self.assertFalse(loaded.metadata.benchmark_evidence)
            self.assertFalse(loaded.metadata.test_split_accessed)
            self.assertFalse(loaded.metadata.production_authority)
            with self.assertRaisesRegex(
                Poly2DQualityCheckpointError, "must not already exist"
            ):
                persist_poly_2d_quality_checkpoint(
                    directory,
                    run=run,
                    provenance=provenance,
                    quality_config=_SMALL_QUALITY,
                    model_config=_SMALL_MODEL,
                )

    def test_training_result_tamper_fails_before_model_acceptance(self) -> None:
        run, provenance = _make_run()
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "candidate"
            persist_poly_2d_quality_checkpoint(
                directory,
                run=run,
                provenance=provenance,
                quality_config=_SMALL_QUALITY,
                model_config=_SMALL_MODEL,
            )
            result_path = directory / QUALITY_TRAINING_RESULT_FILENAME
            payload = json.loads(result_path.read_text("ascii"))
            payload["selected_epoch"] = int(payload["selected_epoch"]) + 1
            result_path.write_text(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ),
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                Poly2DQualityCheckpointError,
                "training-result file SHA differs from receipt",
            ):
                load_and_verify_poly_2d_quality_checkpoint(directory)

    def test_metadata_tamper_fails_before_checkpoint_load(self) -> None:
        run, provenance = _make_run()
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "candidate"
            persist_poly_2d_quality_checkpoint(
                directory,
                run=run,
                provenance=provenance,
                quality_config=_SMALL_QUALITY,
                model_config=_SMALL_MODEL,
            )
            metadata_path = directory / QUALITY_METADATA_FILENAME
            payload = json.loads(metadata_path.read_text("ascii"))
            payload["production_authority"] = True
            metadata_path.write_text(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ),
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                Poly2DQualityCheckpointError,
                "metadata SHA differs from receipt",
            ):
                load_and_verify_poly_2d_quality_checkpoint(directory)

    def test_verified_inference_binds_exact_quality_artifact_identity(self) -> None:
        run, provenance = _make_run()
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "candidate"
            loaded = persist_poly_2d_quality_checkpoint(
                directory,
                run=run,
                provenance=provenance,
                quality_config=_SMALL_QUALITY,
                model_config=_SMALL_MODEL,
            )
            image = torch.full((1, 1, 32, 128), 0.4, dtype=torch.float32)
            result = run_verified_poly_2d_quality_checkpoint_inference(
                directory,
                image,
                max_decode_steps=4,
            )
            identity = result.identity
            self.assertTrue(identity.checkpoint_bound)
            self.assertEqual(identity.checkpoint_sha256, loaded.checkpoint_sha256)
            self.assertEqual(
                identity.checkpoint_metadata_file_sha256,
                loaded.metadata_sha256,
            )
            self.assertEqual(
                identity.checkpoint_receipt_sha256,
                loaded.receipt_sha256,
            )
            self.assertEqual(
                identity.dataset_manifest_sha256,
                loaded.metadata.dataset_manifest_sha256,
            )
            self.assertEqual(
                identity.trainer_profile_sha256,
                loaded.metadata.quality_trainer_profile_sha256,
            )
            self.assertEqual(
                identity.provenance_sha256,
                loaded.metadata.provenance_sha256,
            )
            self.assertEqual(
                identity.registry_record_fingerprint_sha256,
                loaded.metadata.registry_record_fingerprint_sha256,
            )
            self.assertEqual(identity.repository_sha, _REPOSITORY)

    def test_receipt_file_set_is_exact(self) -> None:
        run, provenance = _make_run()
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "candidate"
            persist_poly_2d_quality_checkpoint(
                directory,
                run=run,
                provenance=provenance,
                quality_config=_SMALL_QUALITY,
                model_config=_SMALL_MODEL,
            )
            (directory / "unexpected.txt").write_text("x", encoding="ascii")
            with self.assertRaisesRegex(
                Poly2DQualityCheckpointError, "file set mismatch"
            ):
                load_and_verify_poly_2d_quality_checkpoint(directory)
            self.assertTrue((directory / QUALITY_RECEIPT_FILENAME).is_file())


if __name__ == "__main__":
    unittest.main()
