from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from st_omr_training.dataset_builder import (
    SyntheticDatasetConfig,
    build_synthetic_dataset,
    write_synthetic_dataset,
)
from st_omr_training.dataset_manifest import DatasetSplit
from st_omr_training.poly_2d_checkpoint import load_and_verify_poly_2d_checkpoint
from st_omr_training.poly_2d_dataset_execution import (
    POLY_2D_DATASET_EXECUTION_VERSION,
    POLY_2D_TARGET_PROFILE,
    POLY_2D_V1_BRIDGE_VERSION,
    Poly2DDatasetExecutionError,
    bridge_supported_v1_musicxml_to_v2,
    execute_exact_stage6_poly_2d_training,
    make_poly_2d_training_batch,
    materialize_poly_2d_samples,
    poly_2d_materialization_fingerprint,
)
from st_omr_training.poly_2d_training import Poly2DTrainingConfig
from st_omr_training.poly_2d_transformer import Poly2DTransformerConfig
from st_omr_training.polyphonic_serialization import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    detokenize_polyphonic_ids,
    validate_roundtrip,
)


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
    max_target_tokens=8192,
)
_TRAINING_CONFIG = Poly2DTrainingConfig(smoke_steps=1)
_REPOSITORY_SHA = "a" * 40


class Poly2DExactDatasetExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.build = build_synthetic_dataset(
            SyntheticDatasetConfig(
                dataset_name="st-poly-08c-live",
                dataset_version="v1",
                family_count=3,
                seed_start=98200,
                split_seed=29,
                measure_count=1,
                raster_width=512,
                family_profiles=("note-only", "rest-only", "chord-only"),
                degradation_profiles=("clean",),
            )
        )

    def _write(self, parent: Path) -> Path:
        root = parent / "dataset"
        write_synthetic_dataset(self.build, root)
        return root

    def test_versions_and_bridge_are_explicitly_single_voice(self) -> None:
        self.assertEqual(POLY_2D_DATASET_EXECUTION_VERSION, "st-omr-poly-2d-dataset-execution-v1")
        self.assertEqual(POLY_2D_V1_BRIDGE_VERSION, "st-omr-v1-to-poly-v2-single-voice-bridge-v1")
        self.assertEqual(POLY_2D_TARGET_PROFILE, "single_voice_v1_bridge")
        score = bridge_supported_v1_musicxml_to_v2(self.build.targets[0].musicxml)
        target = validate_roundtrip(score)
        self.assertEqual(detokenize_polyphonic_ids(target.token_ids), score)
        self.assertTrue(
            all(event.voice == 1 for part in score.parts for measure in part.measures for event in measure.events)
        )
        self.assertTrue(
            all(event.staff == 1 for part in score.parts for measure in part.measures for event in measure.events)
        )

    def test_materialization_is_hash_bound_deterministic_and_family_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._write(Path(temporary))
            train_first = materialize_poly_2d_samples(
                self.build, root, DatasetSplit.TRAIN, model_config=_MODEL_CONFIG, max_samples=1
            )
            train_second = materialize_poly_2d_samples(
                self.build, root, DatasetSplit.TRAIN, model_config=_MODEL_CONFIG, max_samples=1
            )
            validation = materialize_poly_2d_samples(
                self.build, root, DatasetSplit.VALIDATION, model_config=_MODEL_CONFIG, max_samples=1
            )
            self.assertEqual(train_first[0].sample_id, train_second[0].sample_id)
            self.assertEqual(train_first[0].target.token_ids, train_second[0].target.token_ids)
            self.assertTrue(torch.equal(train_first[0].image, train_second[0].image))
            self.assertEqual(tuple(train_first[0].image.shape), (1, 32, 128))
            self.assertTrue(set(item.family_id for item in train_first).isdisjoint(item.family_id for item in validation))
            self.assertEqual(train_first[0].target_profile, POLY_2D_TARGET_PROFILE)
            self.assertEqual(len(poly_2d_materialization_fingerprint(_MODEL_CONFIG)), 64)

    def test_test_split_is_rejected_before_dataset_root_access(self) -> None:
        missing = Path("/definitely/not/a/dataset")
        with self.assertRaisesRegex(Poly2DDatasetExecutionError, "TEST remains sealed"):
            materialize_poly_2d_samples(
                self.build,
                missing,
                DatasetSplit.TEST,
                model_config=_MODEL_CONFIG,
                max_samples=1,
            )

    def test_train_validation_materialization_never_reads_corrupt_test_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._write(Path(temporary))
            test_sample = next(sample for sample in self.build.manifest.samples if sample.split is DatasetSplit.TEST)
            (root / "targets" / f"{test_sample.source_musicxml_sha256}.musicxml").write_bytes(b"test-sealed-tamper")
            (root / "images" / f"{test_sample.png_sha256}.png").write_bytes(b"test-sealed-tamper")
            train = materialize_poly_2d_samples(
                self.build, root, DatasetSplit.TRAIN, model_config=_MODEL_CONFIG, max_samples=1
            )
            validation = materialize_poly_2d_samples(
                self.build, root, DatasetSplit.VALIDATION, model_config=_MODEL_CONFIG, max_samples=1
            )
            self.assertEqual(len(train), 1)
            self.assertEqual(len(validation), 1)

    def test_selected_target_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._write(Path(temporary))
            train_sample = next(sample for sample in self.build.manifest.samples if sample.split is DatasetSplit.TRAIN)
            path = root / "targets" / f"{train_sample.source_musicxml_sha256}.musicxml"
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(Poly2DDatasetExecutionError, "target hash mismatch"):
                materialize_poly_2d_samples(
                    self.build, root, DatasetSplit.TRAIN, model_config=_MODEL_CONFIG, max_samples=1
                )

    def test_materialized_teacher_forcing_batch_uses_v2_bos_eos(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._write(Path(temporary))
            samples = materialize_poly_2d_samples(
                self.build, root, DatasetSplit.TRAIN, model_config=_MODEL_CONFIG, max_samples=1
            )
            batch = make_poly_2d_training_batch(samples, dataset_manifest_sha256=self.build.manifest_sha256)
            self.assertEqual(batch.split, DatasetSplit.TRAIN)
            self.assertEqual(batch.decoder_input_ids[0, 0].item(), BOS_TOKEN_ID)
            unpadded_labels = batch.labels[0][batch.labels[0] != PAD_TOKEN_ID]
            self.assertEqual(unpadded_labels[-1].item(), EOS_TOKEN_ID)
            self.assertEqual(tuple(batch.images.shape[1:]), (1, 32, 128))

    def test_exact_stage6_bytes_execute_bounded_checkpoint_without_authority_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._write(Path(temporary))
            output = Path(temporary) / "checkpoint"
            result = execute_exact_stage6_poly_2d_training(
                build=self.build,
                dataset_root=root,
                repository_sha=_REPOSITORY_SHA,
                output_directory=output,
                training_config=_TRAINING_CONFIG,
                model_config=_MODEL_CONFIG,
                max_train_samples=1,
                max_validation_samples=1,
            )
            loaded = load_and_verify_poly_2d_checkpoint(output)
            self.assertTrue(result.controlled_dataset_execution)
            self.assertFalse(result.authoritative_dataset_execution)
            self.assertFalse(result.test_split_accessed)
            self.assertFalse(result.benchmark_evidence)
            self.assertFalse(result.polyphonic_voice_evidence)
            self.assertFalse(result.production_authority)
            self.assertEqual(result.target_profile, "single_voice_v1_bridge")
            self.assertEqual(loaded.metadata.dataset_manifest_sha256, self.build.manifest_sha256)
            self.assertEqual(
                loaded.metadata.preprocess_fingerprint_sha256,
                result.materialization_fingerprint_sha256,
            )
            self.assertFalse(loaded.metadata.authoritative_dataset_execution)
            self.assertFalse(loaded.metadata.test_split_accessed)


if __name__ == "__main__":
    unittest.main()
