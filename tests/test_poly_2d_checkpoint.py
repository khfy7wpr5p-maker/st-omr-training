from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from st_omr_training.dataset_manifest import DatasetSplit
from st_omr_training.model_registry import ModelLifecycle, ResearchAuthority, registry_by_id
from st_omr_training.poly_2d_checkpoint import (
    CHECKPOINT_FILENAME,
    METADATA_FILENAME,
    POLY_2D_CHECKPOINT_RECEIPT_VERSION,
    POLY_2D_CHECKPOINT_SCHEMA_VERSION,
    POLY_2D_REGISTRY_RECORD_ID,
    RECEIPT_FILENAME,
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
from st_omr_training.training_model import model_state_sha256


_SHA = "1" * 64
_SHA2 = "2" * 64
_GIT = "a" * 40
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


def _batch(split: DatasetSplit, *, image_value: float) -> Poly2DTrainingBatch:
    return Poly2DTrainingBatch(
        images=torch.full((1, 1, 32, 128), image_value, dtype=torch.float32),
        decoder_input_ids=torch.tensor(
            [[BOS_TOKEN_ID, _OBJ_START, _OBJ_END, PAD_TOKEN_ID]], dtype=torch.long
        ),
        labels=torch.tensor(
            [[_OBJ_START, _OBJ_END, EOS_TOKEN_ID, PAD_TOKEN_ID]], dtype=torch.long
        ),
        split=split,
        sample_ids=(f"checkpoint-{split.value}",),
        dataset_manifest_sha256=_SHA,
    )


def _provenance():
    return build_poly_2d_training_provenance(
        repository_sha=_GIT,
        dataset_manifest_sha256=_SHA,
        preprocess_fingerprint_sha256=_SHA2,
        training_config=_TRAINING_CONFIG,
        model_config=_MODEL_CONFIG,
    )


def _persist(root: Path):
    output = root / "artifact"
    receipt = run_and_persist_bounded_poly_2d_checkpoint(
        train_batches=(_batch(DatasetSplit.TRAIN, image_value=0.5),),
        validation_batch=_batch(DatasetSplit.VALIDATION, image_value=0.4),
        provenance=_provenance(),
        output_directory=output,
        training_config=_TRAINING_CONFIG,
        model_config=_MODEL_CONFIG,
    )
    return output, receipt


class Poly2DCheckpointTests(unittest.TestCase):
    def test_versions_and_registry_gate_remain_research_only(self) -> None:
        self.assertEqual(POLY_2D_CHECKPOINT_SCHEMA_VERSION, "st-omr-poly-2d-checkpoint-v1")
        self.assertEqual(
            POLY_2D_CHECKPOINT_RECEIPT_VERSION,
            "st-omr-poly-2d-checkpoint-receipt-v1",
        )
        record = registry_by_id()[POLY_2D_REGISTRY_RECORD_ID]
        self.assertIs(record.lifecycle, ModelLifecycle.ARCHITECTURE_ONLY)
        self.assertIs(record.authority, ResearchAuthority.NONE)

    def test_persist_reload_roundtrip_is_exact_and_hash_bound(self) -> None:
        with TemporaryDirectory() as temporary:
            output, receipt = _persist(Path(temporary))
            self.assertTrue((output / CHECKPOINT_FILENAME).is_file())
            self.assertTrue((output / METADATA_FILENAME).is_file())
            self.assertTrue((output / RECEIPT_FILENAME).is_file())

            loaded = load_and_verify_poly_2d_checkpoint(output)
            self.assertEqual(loaded.checkpoint_sha256, receipt.checkpoint_sha256)
            self.assertEqual(loaded.metadata_sha256, receipt.metadata_sha256)
            self.assertEqual(loaded.receipt_sha256, receipt.receipt_sha256)
            self.assertEqual(loaded.metadata.final_state_sha256, receipt.state_sha256)
            self.assertEqual(model_state_sha256(loaded.model), receipt.state_sha256)
            self.assertEqual(
                loaded.metadata.registry_record_fingerprint_sha256,
                registry_by_id()[POLY_2D_REGISTRY_RECORD_ID].fingerprint(),
            )
            self.assertFalse(loaded.metadata.authoritative_dataset_execution)
            self.assertFalse(loaded.metadata.test_split_accessed)
            self.assertFalse(loaded.metadata.benchmark_evidence)
            self.assertFalse(loaded.metadata.production_authority)

    def test_reload_explicitly_uses_weights_only_true(self) -> None:
        with TemporaryDirectory() as temporary:
            output, _receipt = _persist(Path(temporary))
            with patch(
                "st_omr_training.poly_2d_checkpoint.torch.load",
                wraps=torch.load,
            ) as mocked_load:
                load_and_verify_poly_2d_checkpoint(output)
            self.assertGreaterEqual(mocked_load.call_count, 1)
            self.assertTrue(all(call.kwargs.get("weights_only") is True for call in mocked_load.call_args_list))
            self.assertTrue(all(call.kwargs.get("map_location") == "cpu" for call in mocked_load.call_args_list))

    def test_existing_artifact_directory_is_never_overwritten(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, first = _persist(root)
            before = (output / CHECKPOINT_FILENAME).read_bytes()
            with self.assertRaisesRegex(Poly2DCheckpointError, "overwrite is forbidden"):
                run_and_persist_bounded_poly_2d_checkpoint(
                    train_batches=(_batch(DatasetSplit.TRAIN, image_value=0.5),),
                    validation_batch=_batch(DatasetSplit.VALIDATION, image_value=0.4),
                    provenance=_provenance(),
                    output_directory=output,
                    training_config=_TRAINING_CONFIG,
                    model_config=_MODEL_CONFIG,
                )
            self.assertEqual((output / CHECKPOINT_FILENAME).read_bytes(), before)
            self.assertEqual(load_and_verify_poly_2d_checkpoint(output).checkpoint_sha256, first.checkpoint_sha256)

    def test_checkpoint_byte_tamper_fails_before_deserialization(self) -> None:
        with TemporaryDirectory() as temporary:
            output, _receipt = _persist(Path(temporary))
            checkpoint_path = output / CHECKPOINT_FILENAME
            checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tamper")
            with patch(
                "st_omr_training.poly_2d_checkpoint.torch.load",
                wraps=torch.load,
            ) as mocked_load:
                with self.assertRaisesRegex(Poly2DCheckpointError, "checkpoint file SHA-256 mismatch"):
                    load_and_verify_poly_2d_checkpoint(output)
            mocked_load.assert_not_called()

    def test_metadata_tamper_fails_against_receipt_hash(self) -> None:
        with TemporaryDirectory() as temporary:
            output, _receipt = _persist(Path(temporary))
            metadata_path = output / METADATA_FILENAME
            metadata_path.write_bytes(metadata_path.read_bytes() + b" ")
            with self.assertRaisesRegex(Poly2DCheckpointError, "metadata SHA-256 mismatch"):
                load_and_verify_poly_2d_checkpoint(output)

    def test_receipt_unknown_field_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            output, _receipt = _persist(Path(temporary))
            receipt_path = output / RECEIPT_FILENAME
            payload = json.loads(receipt_path.read_text("ascii"))
            payload["unexpected"] = True
            receipt_path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
                encoding="ascii",
            )
            with self.assertRaisesRegex(Poly2DCheckpointError, "key set mismatch"):
                load_and_verify_poly_2d_checkpoint(output)

    def test_metadata_cannot_claim_benchmark_test_or_production_authority(self) -> None:
        with TemporaryDirectory() as temporary:
            output, _receipt = _persist(Path(temporary))
            metadata = load_and_verify_poly_2d_checkpoint(output).metadata
            for field_name in (
                "authoritative_dataset_execution",
                "test_split_accessed",
                "benchmark_evidence",
                "production_authority",
            ):
                with self.assertRaisesRegex(Poly2DCheckpointError, f"{field_name} must remain false"):
                    replace(metadata, **{field_name: True})

    def test_preexisting_temporary_directory_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "artifact"
            (root / ".artifact.tmp").mkdir()
            with self.assertRaisesRegex(Poly2DCheckpointError, "temporary directory already exists"):
                run_and_persist_bounded_poly_2d_checkpoint(
                    train_batches=(_batch(DatasetSplit.TRAIN, image_value=0.5),),
                    validation_batch=_batch(DatasetSplit.VALIDATION, image_value=0.4),
                    provenance=_provenance(),
                    output_directory=output,
                    training_config=_TRAINING_CONFIG,
                    model_config=_MODEL_CONFIG,
                )


if __name__ == "__main__":
    unittest.main()
