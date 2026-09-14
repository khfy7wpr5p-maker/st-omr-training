from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from st_omr_training.dataset_manifest import DatasetSplit
from st_omr_training.poly_2d_transformer import (
    Poly2DTransformerConfig,
    build_tiny_poly_2d_transformer,
    poly_2d_config_fingerprint,
)
from st_omr_training.poly_evaluation_contract import (
    BenchmarkIdentity,
    BenchmarkSampleDescriptor,
    PolyphonicComplexityProfile,
    RobustnessBucket,
)
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
    native_poly_v2_materialization_fingerprint,
    native_poly_v2_sample_id,
    persist_native_poly_v2_dataset,
)
from st_omr_training.poly_v2_validation_execution import (
    POLY_V2_VALIDATION_DESCRIPTOR_POLICY,
    POLY_V2_VALIDATION_EXECUTION_VERSION,
    PolyV2ValidationExecutionError,
    build_native_poly_v2_validation_benchmark_identity,
    execute_native_poly_v2_validation_benchmark,
    native_poly_v2_validation_split_manifest_sha256,
)
from st_omr_training.training_model import model_state_sha256


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
    max_target_tokens=2048,
)


def _atom(atom_id: str, step: str, octave: int) -> NoteAtom:
    return NoteAtom(
        atom_id=atom_id,
        pitch=PitchSpelling(step=step, alter=0, octave=octave),
    )


def _score(tag: str) -> PolyScore:
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
                                event_id=f"{tag}-chord-v1",
                                kind=EventKind.CHORD,
                                onset=ExactRational(0, 1),
                                duration=ExactRational(1, 4),
                                voice=1,
                                staff=1,
                                note_type=NoteType.QUARTER,
                                noteheads=(
                                    _atom(f"{tag}-c4", "C", 4),
                                    _atom(f"{tag}-e4", "E", 4),
                                ),
                            ),
                            PolyEvent(
                                event_id=f"{tag}-note-v2",
                                kind=EventKind.NOTE,
                                onset=ExactRational(0, 1),
                                duration=ExactRational(1, 4),
                                voice=2,
                                staff=1,
                                note_type=NoteType.QUARTER,
                                noteheads=(_atom(f"{tag}-g4", "G", 4),),
                            ),
                            PolyEvent(
                                event_id=f"{tag}-rest-v3",
                                kind=EventKind.REST,
                                onset=ExactRational(0, 1),
                                duration=ExactRational(1, 4),
                                voice=3,
                                staff=1,
                                note_type=NoteType.QUARTER,
                            ),
                            PolyEvent(
                                event_id=f"{tag}-note-v4",
                                kind=EventKind.NOTE,
                                onset=ExactRational(0, 1),
                                duration=ExactRational(1, 4),
                                voice=4,
                                staff=1,
                                note_type=NoteType.QUARTER,
                                noteheads=(_atom(f"{tag}-b4", "B", 4),),
                            ),
                        ),
                    ),
                ),
            ),
        )
    )


def _png(seed: int) -> bytes:
    image = Image.new("L", (128, 48), 255)
    for row in range(5):
        y = 12 + row * 4
        for x in range(5, 124):
            image.putpixel((x, y), 90)
    note_x = 12 + (seed % 80)
    for y in range(20, 27):
        for x in range(note_x, min(note_x + 7, image.width)):
            image.putpixel((x, y), seed % 40)
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


def _build():
    artifacts = (
        NativePolyV2ArtifactInput(
            family_id="train-family",
            split=DatasetSplit.TRAIN,
            target_json=serialize_polyphonic_score(_score("train")).encode("ascii"),
            image_png=_png(1),
        ),
        NativePolyV2ArtifactInput(
            family_id="validation-family-a",
            split=DatasetSplit.VALIDATION,
            target_json=serialize_polyphonic_score(_score("validation-a")).encode("ascii"),
            image_png=_png(101),
        ),
        NativePolyV2ArtifactInput(
            family_id="validation-family-b",
            split=DatasetSplit.VALIDATION,
            target_json=serialize_polyphonic_score(_score("validation-b")).encode("ascii"),
            image_png=_png(102),
        ),
    )
    return build_native_poly_v2_dataset(
        artifacts,
        sealed_test_samples=(_sealed_test_sample(),),
    )


def _complexity() -> PolyphonicComplexityProfile:
    return PolyphonicComplexityProfile(
        voice_count=4,
        staff_count=1,
        simultaneous_note_density=0.75,
        chord_density=0.25,
        overlap_density=0.75,
        tie_density=0.0,
        beam_complexity=0.0,
        rhythmic_complexity=0.25,
        tuplet_present=False,
        grace_present=False,
        cross_staff_present=False,
    )


def _descriptors(build):
    samples = tuple(
        sorted(
            (
                sample
                for sample in build.manifest.samples
                if sample.split is DatasetSplit.VALIDATION
            ),
            key=lambda item: item.sample_id,
        )
    )
    buckets = (RobustnessBucket.CLEAN, RobustnessBucket.SCAN)
    return tuple(
        BenchmarkSampleDescriptor(
            sample_id=sample.sample_id,
            family_id=sample.family_id,
            split="validation",
            complexity=_complexity(),
            robustness_bucket=buckets[index],
        )
        for index, sample in enumerate(samples)
    )


class _FakeMetadata:
    def __init__(self, model, dataset_manifest_sha256: str) -> None:
        self.selected_state_sha256 = model_state_sha256(model)
        self.model_profile_sha256 = poly_2d_config_fingerprint(model.config)
        self.dataset_manifest_sha256 = dataset_manifest_sha256
        self.preprocess_fingerprint_sha256 = native_poly_v2_materialization_fingerprint(
            model.config
        )
        self.quality_trainer_profile_sha256 = sha256(b"quality-trainer").hexdigest()
        self.provenance_sha256 = sha256(b"provenance").hexdigest()
        self.registry_record_fingerprint_sha256 = sha256(b"registry").hexdigest()
        self.repository_sha = "a" * 40

    def fingerprint(self) -> str:
        payload = (
            self.selected_state_sha256
            + self.model_profile_sha256
            + self.dataset_manifest_sha256
            + self.preprocess_fingerprint_sha256
        ).encode("ascii")
        return sha256(payload).hexdigest()


def _loaded(build, *, dataset_manifest_sha256: str | None = None):
    model = build_tiny_poly_2d_transformer(_MODEL_CONFIG, seed=82_008)
    metadata = _FakeMetadata(
        model,
        dataset_manifest_sha256 or build.manifest_sha256,
    )
    return SimpleNamespace(
        model=model,
        metadata=metadata,
        checkpoint_sha256=sha256(b"checkpoint").hexdigest(),
        metadata_sha256=sha256(b"metadata-file").hexdigest(),
        receipt_sha256=sha256(b"receipt").hexdigest(),
    )


class PolyV2ValidationExecutionTests(unittest.TestCase):
    def test_split_manifest_binds_explicit_descriptor_metadata_and_artifacts(self) -> None:
        build = _build()
        descriptors = _descriptors(build)
        first = native_poly_v2_validation_split_manifest_sha256(
            build=build,
            descriptors=descriptors,
        )
        second = native_poly_v2_validation_split_manifest_sha256(
            build=build,
            descriptors=tuple(reversed(descriptors)),
        )
        changed = list(descriptors)
        changed[0] = BenchmarkSampleDescriptor(
            sample_id=changed[0].sample_id,
            family_id=changed[0].family_id,
            split="validation",
            complexity=changed[0].complexity,
            robustness_bucket=RobustnessBucket.PHONE,
        )
        third = native_poly_v2_validation_split_manifest_sha256(
            build=build,
            descriptors=tuple(changed),
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual(len(first), 64)
        self.assertEqual(
            POLY_V2_VALIDATION_DESCRIPTOR_POLICY,
            "explicit-hash-bound-no-defaults-v1",
        )

    def test_descriptor_set_must_cover_exact_complete_validation_population(self) -> None:
        build = _build()
        descriptors = _descriptors(build)
        with self.assertRaisesRegex(
            PolyV2ValidationExecutionError,
            "exact complete VALIDATION population",
        ):
            native_poly_v2_validation_split_manifest_sha256(
                build=build,
                descriptors=descriptors[:1],
            )
        bad_family = list(descriptors)
        bad_family[0] = BenchmarkSampleDescriptor(
            sample_id=bad_family[0].sample_id,
            family_id="wrong-family",
            split="validation",
            complexity=bad_family[0].complexity,
            robustness_bucket=bad_family[0].robustness_bucket,
        )
        with self.assertRaisesRegex(PolyV2ValidationExecutionError, "family_id"):
            native_poly_v2_validation_split_manifest_sha256(
                build=build,
                descriptors=tuple(bad_family),
            )

    def test_test_descriptor_is_rejected_without_dataset_root_access(self) -> None:
        build = _build()
        descriptors = list(_descriptors(build))
        descriptors[0] = BenchmarkSampleDescriptor(
            sample_id=descriptors[0].sample_id,
            family_id=descriptors[0].family_id,
            split="test",
            complexity=descriptors[0].complexity,
            robustness_bucket=descriptors[0].robustness_bucket,
        )
        with self.assertRaisesRegex(PolyV2ValidationExecutionError, "TEST remains sealed"):
            build_native_poly_v2_validation_benchmark_identity(
                build=build,
                descriptors=tuple(descriptors),
                benchmark_id="native-v2-validation",
                benchmark_version="v1",
            )

    def test_benchmark_identity_is_deterministic_and_dataset_bound(self) -> None:
        build = _build()
        descriptors = _descriptors(build)
        first = build_native_poly_v2_validation_benchmark_identity(
            build=build,
            descriptors=descriptors,
            benchmark_id="native-v2-validation",
            benchmark_version="v1",
        )
        second = build_native_poly_v2_validation_benchmark_identity(
            build=build,
            descriptors=tuple(reversed(descriptors)),
            benchmark_id="native-v2-validation",
            benchmark_version="v1",
        )
        self.assertEqual(first, second)
        self.assertEqual(first.dataset_manifest_sha256, build.manifest_sha256)
        self.assertEqual(len(first.canonical_sha256()), 64)

    def test_wrong_benchmark_binding_fails_before_checkpoint_load(self) -> None:
        build = _build()
        descriptors = _descriptors(build)
        benchmark = BenchmarkIdentity(
            benchmark_id="native-v2-validation",
            benchmark_version="v1",
            dataset_manifest_sha256=build.manifest_sha256,
            split_manifest_sha256="0" * 64,
        )
        with patch(
            "st_omr_training.poly_v2_validation_execution.load_and_verify_poly_2d_quality_checkpoint"
        ) as loader:
            with self.assertRaisesRegex(
                PolyV2ValidationExecutionError,
                "benchmark identity",
            ):
                execute_native_poly_v2_validation_benchmark(
                    build=build,
                    dataset_root=Path("/must/not/be/read"),
                    checkpoint_directory=Path("/must/not/be/read"),
                    benchmark=benchmark,
                    descriptors=descriptors,
                    max_decode_steps=1,
                )
            loader.assert_not_called()

    def test_checkpoint_dataset_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = _build()
            root = persist_native_poly_v2_dataset(build, Path(directory) / "dataset")
            descriptors = _descriptors(build)
            benchmark = build_native_poly_v2_validation_benchmark_identity(
                build=build,
                descriptors=descriptors,
                benchmark_id="native-v2-validation",
                benchmark_version="v1",
            )
            loaded = _loaded(build, dataset_manifest_sha256="f" * 64)
            with patch(
                "st_omr_training.poly_v2_validation_execution.load_and_verify_poly_2d_quality_checkpoint",
                return_value=loaded,
            ):
                with self.assertRaisesRegex(
                    PolyV2ValidationExecutionError,
                    "different dataset manifest",
                ):
                    execute_native_poly_v2_validation_benchmark(
                        build=build,
                        dataset_root=root,
                        checkpoint_directory=Path(directory) / "checkpoint",
                        benchmark=benchmark,
                        descriptors=descriptors,
                        max_decode_steps=1,
                    )

    def test_end_to_end_validation_execution_reaches_b1_b2_b3_without_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = _build()
            root = persist_native_poly_v2_dataset(build, Path(directory) / "dataset")
            descriptors = _descriptors(build)
            benchmark = build_native_poly_v2_validation_benchmark_identity(
                build=build,
                descriptors=descriptors,
                benchmark_id="native-v2-validation",
                benchmark_version="v1",
            )
            loaded = _loaded(build)
            with patch(
                "st_omr_training.poly_v2_validation_execution.load_and_verify_poly_2d_quality_checkpoint",
                return_value=loaded,
            ):
                first = execute_native_poly_v2_validation_benchmark(
                    build=build,
                    dataset_root=root,
                    checkpoint_directory=Path(directory) / "checkpoint",
                    benchmark=benchmark,
                    descriptors=descriptors,
                    max_decode_steps=1,
                )
                second = execute_native_poly_v2_validation_benchmark(
                    build=build,
                    dataset_root=root,
                    checkpoint_directory=Path(directory) / "checkpoint",
                    benchmark=benchmark,
                    descriptors=tuple(reversed(descriptors)),
                    max_decode_steps=1,
                )

            self.assertEqual(first.fingerprint(), second.fingerprint())
            self.assertEqual(first.sample_ids, tuple(sorted(item.sample_id for item in descriptors)))
            self.assertEqual(first.aggregate.sample_count, len(descriptors))
            self.assertTrue(first.aggregate.checkpoint_bound)
            self.assertTrue(first.validation_benchmark_evidence)
            self.assertFalse(first.test_split_accessed)
            self.assertFalse(first.production_authority)
            self.assertFalse(first.full_metric_contract_ready)
            self.assertFalse(first.common_comparison_ready)
            self.assertEqual(
                set(first.aggregate.unsupported_metric_ids),
                {
                    "musicxml_validity",
                    "tedn",
                    "notehead_stem_f1",
                    "beam_relation_f1",
                    "tie_relation_f1",
                },
            )
            self.assertEqual(
                POLY_V2_VALIDATION_EXECUTION_VERSION,
                "st-omr-poly-v2-validation-execution-v1",
            )
            self.assertEqual(len(first.fingerprint()), 64)


if __name__ == "__main__":
    unittest.main()
