from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from st_omr_training.dataset_manifest import DatasetSplit
from st_omr_training.poly_2d_quality_checkpoint import (
    load_and_verify_poly_2d_quality_checkpoint,
)
from st_omr_training.poly_2d_quality_training import Poly2DQualityTrainingConfig
from st_omr_training.poly_2d_transformer import Poly2DTransformerConfig
from st_omr_training.polyphonic_representation import (
    ClefAssignment,
    EventKind,
    ExactRational,
    KeySignature,
    NoteAtom,
    NoteType,
    PitchSpelling,
    PolyEvent,
    PolyMeasure,
    PolyPart,
    PolyScore,
    TimeSignature,
)
from st_omr_training.polyphonic_serialization import serialize_polyphonic_score
from st_omr_training.poly_v2_dataset_materialization import (
    NativePolyV2ArtifactInput,
    NativePolyV2Sample,
    NativePolyV2TargetProfile,
    build_native_poly_v2_dataset,
    native_poly_v2_sample_id,
    persist_native_poly_v2_dataset,
)
from st_omr_training.poly_v2_quality_execution import (
    POLY_V2_QUALITY_BATCH_POLICY,
    POLY_V2_QUALITY_EXECUTION_VERSION,
    PolyV2QualityExecutionError,
    execute_native_poly_v2_quality_training,
    materialize_native_poly_v2_quality_batches,
    poly_v2_quality_execution_profile_fingerprint,
)


_REPOSITORY = "a" * 40
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
    max_target_tokens=512,
)


def _score() -> PolyScore:
    return PolyScore(
        parts=(
            PolyPart(
                part_id="P1",
                staff_count=1,
                measures=(
                    PolyMeasure(
                        measure_index=1,
                        source_number="1",
                        time_signature=TimeSignature(beats=(4,), beat_type=4),
                        key_signature=KeySignature(fifths=0),
                        clefs=(ClefAssignment(staff=1, sign="G", line=2),),
                        events=(
                            PolyEvent(
                                event_id="e-v1",
                                kind=EventKind.NOTE,
                                onset=ExactRational(0, 1),
                                duration=ExactRational(1, 4),
                                voice=1,
                                staff=1,
                                note_type=NoteType.QUARTER,
                                noteheads=(
                                    NoteAtom(
                                        atom_id="a-v1",
                                        pitch=PitchSpelling(step="C", alter=0, octave=4),
                                    ),
                                ),
                            ),
                            PolyEvent(
                                event_id="e-v2",
                                kind=EventKind.NOTE,
                                onset=ExactRational(0, 1),
                                duration=ExactRational(1, 4),
                                voice=2,
                                staff=1,
                                note_type=NoteType.QUARTER,
                                noteheads=(
                                    NoteAtom(
                                        atom_id="a-v2",
                                        pitch=PitchSpelling(step="E", alter=0, octave=4),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
    )


def _png(seed: int) -> bytes:
    """Create deterministic grayscale score-like bytes without ImageDraw primitives."""

    image = Image.new("L", (128, 48), 255)
    for row in range(5):
        y = 12 + row * 4
        for x in range(5, 124):
            image.putpixel((x, y), 90)
    note_x = 12 + (seed % 80)
    shade = int(seed % 40)
    for y in range(20, 27):
        for x in range(note_x, min(note_x + 7, image.width)):
            image.putpixel((x, y), shade)
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _sealed_test_sample() -> NativePolyV2Sample:
    profile = NativePolyV2TargetProfile(
        voices=(1, 2),
        event_kinds=("note",),
        has_simultaneous_independent_voices=True,
        has_chord_with_independent_voice_same_onset=False,
        tie_count=0,
        beam_count=0,
        tuplet_count=0,
    )
    provisional = NativePolyV2Sample(
        sample_id="0" * 64,
        family_id="sealed-test-family",
        split=DatasetSplit.TEST,
        target_sha256="1" * 64,
        representation_sha256="2" * 64,
        image_sha256="3" * 64,
        width=128,
        height=48,
        target_token_count=10,
        profile=profile,
    )
    return NativePolyV2Sample(
        sample_id=native_poly_v2_sample_id(provisional),
        family_id=provisional.family_id,
        split=provisional.split,
        target_sha256=provisional.target_sha256,
        representation_sha256=provisional.representation_sha256,
        image_sha256=provisional.image_sha256,
        width=provisional.width,
        height=provisional.height,
        target_token_count=provisional.target_token_count,
        profile=provisional.profile,
    )


def _build(train_count: int = 10, validation_count: int = 3):
    target = serialize_polyphonic_score(_score()).encode("ascii")
    artifacts = []
    for index in range(train_count):
        artifacts.append(
            NativePolyV2ArtifactInput(
                family_id=f"train-family-{index:02d}",
                split=DatasetSplit.TRAIN,
                target_json=target,
                image_png=_png(index + 1),
            )
        )
    for index in range(validation_count):
        artifacts.append(
            NativePolyV2ArtifactInput(
                family_id=f"validation-family-{index:02d}",
                split=DatasetSplit.VALIDATION,
                target_json=target,
                image_png=_png(index + 101),
            )
        )
    return build_native_poly_v2_dataset(
        tuple(artifacts),
        sealed_test_samples=(_sealed_test_sample(),),
    )


class NativePolyV2QualityExecutionTests(unittest.TestCase):
    def test_full_train_population_is_partitioned_into_bounded_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = _build(train_count=10, validation_count=3)
            root = persist_native_poly_v2_dataset(build, Path(directory) / "dataset")
            batch_set = materialize_native_poly_v2_quality_batches(
                build=build,
                dataset_root=root,
                split=DatasetSplit.TRAIN,
                model_config=_SMALL_MODEL,
                batch_size=4,
            )
            self.assertEqual(batch_set.batch_sizes, (4, 4, 2))
            self.assertEqual(len(batch_set.sample_ids), 10)
            self.assertEqual(batch_set.sample_ids, tuple(sorted(batch_set.sample_ids)))
            self.assertEqual(batch_set.batch_policy, POLY_V2_QUALITY_BATCH_POLICY)
            self.assertEqual(len(batch_set.fingerprint()), 64)

    def test_selection_and_batch_set_fingerprints_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = _build(train_count=9, validation_count=2)
            root = persist_native_poly_v2_dataset(build, Path(directory) / "dataset")
            first = materialize_native_poly_v2_quality_batches(
                build=build,
                dataset_root=root,
                split=DatasetSplit.TRAIN,
                model_config=_SMALL_MODEL,
                batch_size=8,
                max_samples=9,
            )
            second = materialize_native_poly_v2_quality_batches(
                build=build,
                dataset_root=root,
                split=DatasetSplit.TRAIN,
                model_config=_SMALL_MODEL,
                batch_size=8,
                max_samples=9,
            )
            self.assertEqual(first.batch_sizes, (8, 1))
            self.assertEqual(first.selection_fingerprint_sha256, second.selection_fingerprint_sha256)
            self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_test_split_is_rejected_before_root_access(self) -> None:
        build = _build(train_count=1, validation_count=1)
        with self.assertRaisesRegex(PolyV2QualityExecutionError, "TEST remains sealed"):
            materialize_native_poly_v2_quality_batches(
                build=build,
                dataset_root=Path("/path/that/must/not/be/read"),
                split=DatasetSplit.TEST,
                model_config=_SMALL_MODEL,
            )

    def test_batch_size_above_model_training_boundary_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = _build(train_count=1, validation_count=1)
            root = persist_native_poly_v2_dataset(build, Path(directory) / "dataset")
            with self.assertRaisesRegex(PolyV2QualityExecutionError, "batch_size"):
                materialize_native_poly_v2_quality_batches(
                    build=build,
                    dataset_root=root,
                    split=DatasetSplit.TRAIN,
                    model_config=_SMALL_MODEL,
                    batch_size=9,
                )

    def test_end_to_end_native_quality_execution_writes_verified_b5_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root_path = Path(directory)
            build = _build(train_count=3, validation_count=2)
            dataset_root = persist_native_poly_v2_dataset(build, root_path / "dataset")
            output = root_path / "quality-checkpoint"
            quality = Poly2DQualityTrainingConfig(epochs=1, max_optimizer_steps=4)
            result = execute_native_poly_v2_quality_training(
                build=build,
                dataset_root=dataset_root,
                repository_sha=_REPOSITORY,
                output_directory=output,
                quality_config=quality,
                model_config=_SMALL_MODEL,
                batch_size=2,
                max_train_samples=2,
                max_validation_samples=1,
            )
            loaded = load_and_verify_poly_2d_quality_checkpoint(output)
            self.assertEqual(result.checkpoint_sha256, loaded.checkpoint_sha256)
            self.assertEqual(result.selected_state_sha256, loaded.metadata.selected_state_sha256)
            self.assertEqual(result.dataset_manifest_sha256, build.manifest_sha256)
            self.assertEqual(result.dataset_build_id, build.build_id)
            self.assertEqual(result.train_batch_sizes, (2,))
            self.assertEqual(result.validation_batch_sizes, (1,))
            self.assertEqual(result.optimizer_steps, 1)
            self.assertFalse(result.test_split_accessed)
            self.assertFalse(result.benchmark_evidence)
            self.assertFalse(result.production_authority)
            self.assertEqual(len(result.fingerprint()), 64)

    def test_execution_profile_binds_batch_size(self) -> None:
        quality = Poly2DQualityTrainingConfig(epochs=1, max_optimizer_steps=8)
        first = poly_v2_quality_execution_profile_fingerprint(
            quality_config=quality,
            model_config=_SMALL_MODEL,
            batch_size=4,
        )
        second = poly_v2_quality_execution_profile_fingerprint(
            quality_config=quality,
            model_config=_SMALL_MODEL,
            batch_size=8,
        )
        self.assertNotEqual(first, second)
        self.assertEqual(POLY_V2_QUALITY_EXECUTION_VERSION, "st-omr-poly-v2-quality-execution-v1")


if __name__ == "__main__":
    unittest.main()
