from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
import tempfile
import unittest

from st_omr_training.dataset_manifest import DatasetSplit
from st_omr_training.poly_2d_training import Poly2DTrainingConfig
from st_omr_training.poly_2d_transformer import Poly2DTransformerConfig
from st_omr_training.polyphonic_representation import (
    BeamMark,
    BeamState,
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
    StemDirection,
    TieState,
    TimeSignature,
    TupletBoundary,
    TupletMark,
)
from st_omr_training.polyphonic_serialization import (
    detokenize_polyphonic_target,
    parse_canonical_polyphonic_json,
    serialize_polyphonic_score,
    validate_roundtrip,
)
from st_omr_training.poly_v2_dataset_materialization import (
    NativePolyV2ArtifactInput,
    NativePolyV2DatasetError,
    NativePolyV2Sample,
    NativePolyV2TargetProfile,
    build_native_poly_v2_dataset,
    canonical_native_poly_v2_manifest_bytes,
    execute_native_poly_v2_training,
    make_native_poly_2d_training_batch,
    materialize_native_poly_v2_samples,
    native_poly_v2_manifest_sha256,
    native_poly_v2_sample_id,
    persist_native_poly_v2_dataset,
)


def _atom(atom_id: str, step: str, octave: int, *, tie: bool = False) -> NoteAtom:
    return NoteAtom(
        atom_id=atom_id,
        pitch=PitchSpelling(step=step, alter=0, octave=octave),
        ties=(TieState.START,) if tie else (),
    )


def _rich_score() -> PolyScore:
    events = (
        PolyEvent(
            event_id="e-chord-v1",
            kind=EventKind.CHORD,
            onset=ExactRational(0, 1),
            duration=ExactRational(1, 4),
            voice=1,
            staff=1,
            note_type=NoteType.QUARTER,
            noteheads=(_atom("a-c4", "C", 4), _atom("a-e4", "E", 4)),
            stem=StemDirection.UP,
        ),
        PolyEvent(
            event_id="e-note-v2",
            kind=EventKind.NOTE,
            onset=ExactRational(0, 1),
            duration=ExactRational(1, 8),
            voice=2,
            staff=1,
            note_type=NoteType.EIGHTH,
            noteheads=(_atom("a-g4", "G", 4, tie=True),),
            stem=StemDirection.DOWN,
            beams=(BeamMark(level=1, state=BeamState.BEGIN),),
            tuplets=(
                TupletMark(
                    number=1,
                    actual_notes=3,
                    normal_notes=2,
                    boundary=TupletBoundary.START,
                ),
            ),
        ),
        PolyEvent(
            event_id="e-rest-v3",
            kind=EventKind.REST,
            onset=ExactRational(0, 1),
            duration=ExactRational(1, 4),
            voice=3,
            staff=1,
            note_type=NoteType.QUARTER,
        ),
        PolyEvent(
            event_id="e-note-v4",
            kind=EventKind.NOTE,
            onset=ExactRational(0, 1),
            duration=ExactRational(1, 4),
            voice=4,
            staff=2,
            note_type=NoteType.QUARTER,
            noteheads=(_atom("a-c3", "C", 3),),
        ),
    )
    return PolyScore(
        parts=(
            PolyPart(
                part_id="P1",
                staff_count=2,
                measures=(
                    PolyMeasure(
                        measure_index=1,
                        source_number="1",
                        time_signature=TimeSignature(beats=(4,), beat_type=4),
                        key_signature=KeySignature(fifths=0),
                        clefs=(
                            ClefAssignment(staff=1, sign="G", line=2),
                            ClefAssignment(staff=2, sign="F", line=4),
                        ),
                        events=events,
                    ),
                ),
            ),
        )
    )


def _validation_score() -> PolyScore:
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
                                event_id="v1",
                                kind=EventKind.NOTE,
                                onset=ExactRational(0, 1),
                                duration=ExactRational(1, 4),
                                voice=1,
                                staff=1,
                                note_type=NoteType.QUARTER,
                                noteheads=(_atom("v1-a", "D", 4),),
                            ),
                            PolyEvent(
                                event_id="v2",
                                kind=EventKind.NOTE,
                                onset=ExactRational(0, 1),
                                duration=ExactRational(1, 4),
                                voice=2,
                                staff=1,
                                note_type=NoteType.QUARTER,
                                noteheads=(_atom("v2-a", "A", 3),),
                            ),
                        ),
                    ),
                ),
            ),
        )
    )


def _png(seed: int) -> bytes:
    image = Image.new("L", (128, 48), 255)
    draw = ImageDraw.Draw(image)
    for row in range(5):
        y = 12 + row * 4
        draw.line((5, y, 123, y), fill=80 + seed)
    draw.ellipse((30 + seed, 20, 38 + seed, 26), fill=0)
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _sealed_test_sample(family_id: str = "test-family") -> NativePolyV2Sample:
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
        family_id=family_id,
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
    rich = serialize_polyphonic_score(_rich_score()).encode("ascii")
    validation = serialize_polyphonic_score(_validation_score()).encode("ascii")
    return build_native_poly_v2_dataset(
        (
            NativePolyV2ArtifactInput(
                family_id="train-family",
                split=DatasetSplit.TRAIN,
                target_json=rich,
                image_png=_png(1),
            ),
            NativePolyV2ArtifactInput(
                family_id="validation-family",
                split=DatasetSplit.VALIDATION,
                target_json=validation,
                image_png=_png(2),
            ),
        ),
        sealed_test_samples=(_sealed_test_sample(),),
    )


def _persist(tmp_path: Path):
    build = _build()
    root = persist_native_poly_v2_dataset(build, tmp_path / "native-v2")
    return build, root


class NativePolyV2DatasetMaterializationTests(unittest.TestCase):
    def test_real_voice2_voice3_voice4plus_and_chord_vs_independent_voice_materialize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build, root = _persist(Path(directory))
            samples = materialize_native_poly_v2_samples(build, root, DatasetSplit.TRAIN, max_samples=1)
            self.assertEqual(len(samples), 1)
            profile = samples[0].profile
            self.assertEqual(profile.voices, (1, 2, 3, 4))
            self.assertEqual(set(profile.event_kinds), {"note", "rest", "chord"})
            self.assertTrue(profile.has_simultaneous_independent_voices)
            self.assertTrue(profile.has_chord_with_independent_voice_same_onset)
            self.assertEqual(profile.tie_count, 1)
            self.assertEqual(profile.beam_count, 1)
            self.assertEqual(profile.tuplet_count, 1)

            score = detokenize_polyphonic_target(samples[0].target)
            events = score.parts[0].measures[0].events
            self.assertEqual([(event.voice, event.staff) for event in events], [(1, 1), (2, 1), (3, 1), (4, 2)])
            self.assertIs(events[0].kind, EventKind.CHORD)
            self.assertTrue(all(event.onset == ExactRational(0, 1) for event in events))
            self.assertEqual(events[1].duration, ExactRational(1, 8))

    def test_v2_target_json_and_tokenizer_roundtrip_are_lossless(self) -> None:
        score = _rich_score()
        canonical = serialize_polyphonic_score(score).encode("ascii")
        parsed = parse_canonical_polyphonic_json(canonical)
        target = validate_roundtrip(parsed)
        self.assertEqual(parsed, score)
        self.assertEqual(detokenize_polyphonic_target(target), score)
        self.assertEqual(target.representation_sha256, score.canonical_sha256())

    def test_family_leakage_across_train_validation_test_is_rejected(self) -> None:
        with self.assertRaisesRegex(NativePolyV2DatasetError, "family leakage"):
            build_native_poly_v2_dataset(
                (
                    NativePolyV2ArtifactInput(
                        family_id="shared-family",
                        split=DatasetSplit.TRAIN,
                        target_json=serialize_polyphonic_score(_rich_score()).encode("ascii"),
                        image_png=_png(1),
                    ),
                    NativePolyV2ArtifactInput(
                        family_id="validation-family",
                        split=DatasetSplit.VALIDATION,
                        target_json=serialize_polyphonic_score(_validation_score()).encode("ascii"),
                        image_png=_png(2),
                    ),
                ),
                sealed_test_samples=(_sealed_test_sample("shared-family"),),
            )

    def test_test_artifact_input_and_materialization_fail_closed_before_reads(self) -> None:
        with self.assertRaisesRegex(NativePolyV2DatasetError, "TEST artifact bytes"):
            NativePolyV2ArtifactInput(
                family_id="test-family",
                split=DatasetSplit.TEST,
                target_json=b"not-read-as-v2",
                image_png=b"not-read-as-png",
            )
        with self.assertRaisesRegex(NativePolyV2DatasetError, "TEST remains sealed"):
            materialize_native_poly_v2_samples(object(), object(), DatasetSplit.TEST)

    def test_deterministic_rebuild_has_identical_manifest_and_build_identity(self) -> None:
        first = _build()
        second = _build()
        self.assertEqual(
            canonical_native_poly_v2_manifest_bytes(first.manifest),
            canonical_native_poly_v2_manifest_bytes(second.manifest),
        )
        self.assertEqual(native_poly_v2_manifest_sha256(first.manifest), native_poly_v2_manifest_sha256(second.manifest))
        self.assertEqual(first.manifest_sha256, second.manifest_sha256)
        self.assertEqual(first.build_id, second.build_id)

    def test_tampered_target_image_or_manifest_is_rejected(self) -> None:
        for surface in ("target", "image", "manifest"):
            with self.subTest(surface=surface), tempfile.TemporaryDirectory() as directory:
                build, root = _persist(Path(directory))
                train = next(sample for sample in build.manifest.samples if sample.split is DatasetSplit.TRAIN)
                if surface == "target":
                    (root / "targets" / f"{train.target_sha256}.json").write_bytes(b"{}")
                elif surface == "image":
                    (root / "images" / f"{train.image_sha256}.png").write_bytes(b"tampered")
                else:
                    (root / "manifest.json").write_bytes(b"{}")
                with self.assertRaises(NativePolyV2DatasetError):
                    materialize_native_poly_v2_samples(build, root, DatasetSplit.TRAIN, max_samples=1)

    def test_native_v2_enters_existing_2d_batch_and_checkpoint_training_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            build, root = _persist(tmp_path)
            model_config = Poly2DTransformerConfig(
                input_height=48,
                input_width=128,
                patch_height=16,
                patch_width=16,
                model_dim=32,
                encoder_layers=1,
                decoder_layers=1,
                attention_heads=4,
                feedforward_dim=64,
                max_target_tokens=4096,
            )
            training_config = Poly2DTrainingConfig(smoke_steps=1)
            train = materialize_native_poly_v2_samples(
                build, root, DatasetSplit.TRAIN, model_config=model_config, max_samples=1
            )
            batch = make_native_poly_2d_training_batch(train, dataset_manifest_sha256=build.manifest_sha256)
            self.assertIs(batch.split, DatasetSplit.TRAIN)
            self.assertEqual(tuple(batch.images.shape), (1, 1, 48, 128))
            self.assertEqual(batch.decoder_input_ids.shape, batch.labels.shape)

            result = execute_native_poly_v2_training(
                build=build,
                dataset_root=root,
                repository_sha="9" * 40,
                output_directory=tmp_path / "checkpoint",
                training_config=training_config,
                model_config=model_config,
                max_train_samples=1,
                max_validation_samples=1,
            )
            self.assertTrue(result.native_polyphonic_dataset_verified)
            self.assertTrue(result.training_entry_verified)
            self.assertFalse(result.test_split_accessed)
            self.assertFalse(result.benchmark_evidence)
            self.assertFalse(result.production_authority)
            self.assertTrue(result.checkpoint.artifact_directory.is_dir())

    def test_corrupted_sealed_test_artifacts_are_not_read_during_train_validation_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build, root = _persist(Path(directory))
            sealed = next(sample for sample in build.manifest.samples if sample.split is DatasetSplit.TEST)
            (root / "targets" / f"{sealed.target_sha256}.json").write_bytes(b"corrupt sealed target")
            (root / "images" / f"{sealed.image_sha256}.png").write_bytes(b"corrupt sealed image")
            train = materialize_native_poly_v2_samples(build, root, DatasetSplit.TRAIN, max_samples=1)
            validation = materialize_native_poly_v2_samples(build, root, DatasetSplit.VALIDATION, max_samples=1)
            self.assertTrue(train)
            self.assertTrue(validation)

    def test_semantic_target_truncation_is_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build, root = _persist(Path(directory))
            too_short = Poly2DTransformerConfig(max_target_tokens=8)
            with self.assertRaisesRegex(NativePolyV2DatasetError, "semantic truncation is forbidden"):
                materialize_native_poly_v2_samples(
                    build, root, DatasetSplit.TRAIN, model_config=too_short, max_samples=1
                )


if __name__ == "__main__":
    unittest.main()
