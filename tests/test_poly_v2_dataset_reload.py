from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from st_omr_training.dataset_manifest import DatasetSplit
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
from st_omr_training.poly_v2_dataset_reload import (
    NativePolyV2DatasetReloadError,
    load_and_verify_native_poly_v2_dataset,
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
    x0 = 20 + seed
    for y in range(20, 27):
        for x in range(x0, min(x0 + 7, image.width)):
            image.putpixel((x, y), 20)
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
            family_id="validation-family",
            split=DatasetSplit.VALIDATION,
            target_json=serialize_polyphonic_score(_score("validation")).encode("ascii"),
            image_png=_png(2),
        ),
    )
    return build_native_poly_v2_dataset(
        artifacts,
        sealed_test_samples=(_sealed_test_sample(),),
    )


class PolyV2DatasetReloadTests(unittest.TestCase):
    def _persist(self, parent: Path):
        build = _build()
        root = persist_native_poly_v2_dataset(build, parent / "dataset")
        return build, root

    def test_round_trip_reloads_exact_build_and_emits_fail_closed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            expected, root = self._persist(Path(directory))
            loaded = load_and_verify_native_poly_v2_dataset(root)

            self.assertEqual(loaded.build, expected)
            self.assertEqual(loaded.receipt.manifest_sha256, expected.manifest_sha256)
            self.assertEqual(loaded.receipt.build_id, expected.build_id)
            self.assertEqual(len(loaded.receipt.train_sample_ids), 1)
            self.assertEqual(len(loaded.receipt.validation_sample_ids), 1)
            self.assertEqual(len(loaded.receipt.sealed_test_sample_ids), 1)
            self.assertFalse(loaded.receipt.test_artifact_bytes_accessed)
            self.assertFalse(loaded.receipt.production_authority)
            self.assertEqual(len(loaded.receipt.fingerprint()), 64)

    def test_tampered_train_target_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build, root = self._persist(Path(directory))
            train = next(sample for sample in build.manifest.samples if sample.split is DatasetSplit.TRAIN)
            target = root / "targets" / f"{train.target_sha256}.json"
            target.write_bytes(target.read_bytes() + b" ")

            with self.assertRaises(NativePolyV2DatasetReloadError):
                load_and_verify_native_poly_v2_dataset(root)

    def test_sealed_test_artifact_presence_is_rejected_without_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build, root = self._persist(Path(directory))
            sealed = next(sample for sample in build.manifest.samples if sample.split is DatasetSplit.TEST)
            (root / "targets" / f"{sealed.target_sha256}.json").write_bytes(b"not-test-data")

            with self.assertRaisesRegex(
                NativePolyV2DatasetReloadError,
                "artifact set differs from admitted TRAIN/VALIDATION hashes",
            ):
                load_and_verify_native_poly_v2_dataset(root)

    def test_noncanonical_manifest_encoding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, root = self._persist(Path(directory))
            manifest = root / "manifest.json"
            payload = manifest.read_text(encoding="ascii")
            manifest.write_text(payload + "\n", encoding="ascii")

            with self.assertRaises(NativePolyV2DatasetReloadError):
                load_and_verify_native_poly_v2_dataset(root)

    def test_unexpected_image_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, root = self._persist(Path(directory))
            (root / "images" / ("f" * 64 + ".png")).write_bytes(b"unexpected")

            with self.assertRaisesRegex(
                NativePolyV2DatasetReloadError,
                "artifact set differs from admitted TRAIN/VALIDATION hashes",
            ):
                load_and_verify_native_poly_v2_dataset(root)


if __name__ == "__main__":
    unittest.main()
