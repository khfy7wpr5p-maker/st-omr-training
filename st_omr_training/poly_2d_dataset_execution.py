"""Hash-bound Stage 6 artifact materialization for bounded Polyphonic 2D training.

TR-POLY-08C consumes exact persisted synthetic Stage 6 TRAIN/VALIDATION bytes,
bridges the supported single-voice V1 MusicXML surface into Polyphonic
Representation V2, materializes deterministic image tensors and teacher-forced
V2 batches, and can execute the already-bounded TR-POLY-08A/08B training and
checkpoint path. TEST remains sealed. This is not polyphonic-quality evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from typing import Final
import xml.etree.ElementTree as ET

import torch

from .dataset_builder import SyntheticDatasetBuild, build_metadata_bytes
from .dataset_manifest import (
    DATASET_SOURCE_CLASS,
    DatasetSample,
    DatasetSplit,
    canonical_manifest_bytes,
    dataset_manifest_sha256,
    validate_dataset_manifest,
)
from .musicxml_validator import validate_musicxml
from .poly_2d_checkpoint import (
    Poly2DCheckpointReceipt,
    load_and_verify_poly_2d_checkpoint,
    run_and_persist_bounded_poly_2d_checkpoint,
)
from .poly_2d_training import (
    FROZEN_POLY_2D_TRAINING_CONFIG,
    MAX_POLY_2D_TRAINING_BATCH,
    Poly2DTrainingBatch,
    Poly2DTrainingConfig,
    build_poly_2d_training_provenance,
)
from .poly_2d_transformer import (
    FROZEN_POLY_2D_CONFIG,
    Poly2DTransformerConfig,
    poly_2d_config_fingerprint,
)
from .polyphonic_representation import (
    ClefAssignment,
    DisplayAccidentalV2,
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
from .polyphonic_serialization import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    TokenizedPolyphonicTarget,
    tokenizer_fingerprint,
    validate_roundtrip,
)
from .training_data import (
    InputPreprocessConfig,
    TrainingDataError,
    preprocess_config_fingerprint,
    preprocess_grayscale_png,
)


POLY_2D_DATASET_EXECUTION_VERSION: Final[str] = "st-omr-poly-2d-dataset-execution-v1"
POLY_2D_V1_BRIDGE_VERSION: Final[str] = "st-omr-v1-to-poly-v2-single-voice-bridge-v1"
POLY_2D_TARGET_PROFILE: Final[str] = "single_voice_v1_bridge"
MAX_POLY_2D_MATERIALIZED_SAMPLES: Final[int] = MAX_POLY_2D_TRAINING_BATCH
_HEX = frozenset("0123456789abcdef")
_NOTE_TYPE_DURATION: Final[dict[NoteType, Fraction]] = {
    NoteType.WHOLE: Fraction(1, 1),
    NoteType.HALF: Fraction(1, 2),
    NoteType.QUARTER: Fraction(1, 4),
    NoteType.EIGHTH: Fraction(1, 8),
}
_ACCIDENTAL_MAP: Final[dict[str | None, DisplayAccidentalV2]] = {
    None: DisplayAccidentalV2.NONE,
    "sharp": DisplayAccidentalV2.SHARP,
    "flat": DisplayAccidentalV2.FLAT,
    "natural": DisplayAccidentalV2.NATURAL,
}


class Poly2DDatasetExecutionError(TrainingDataError):
    """Raised when exact Stage 6 bytes cannot enter the TR-POLY-08C boundary."""


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _require_git_sha(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in _HEX for char in value)
    ):
        raise Poly2DDatasetExecutionError("repository_sha must be lowercase git SHA-40 hex")
    return value


def _required_int(element: ET.Element, child_name: str) -> int:
    text = element.findtext(child_name)
    if text is None:
        raise Poly2DDatasetExecutionError(f"supported V1 target is missing {child_name}")
    try:
        return int(text)
    except ValueError as exc:
        raise Poly2DDatasetExecutionError(f"supported V1 target has invalid {child_name}") from exc


def _note_type(note: ET.Element) -> NoteType:
    text = note.findtext("type")
    if text is None:
        raise Poly2DDatasetExecutionError("supported V1 note is missing visible type")
    try:
        result = NoteType(text)
    except ValueError as exc:
        raise Poly2DDatasetExecutionError("supported V1 note type is outside the V2 bridge") from exc
    if result not in _NOTE_TYPE_DURATION:
        raise Poly2DDatasetExecutionError("V1 bridge accepts only whole/half/quarter/eighth note types")
    return result


def _pitch(note: ET.Element) -> PitchSpelling:
    pitch = note.find("pitch")
    if pitch is None:
        raise Poly2DDatasetExecutionError("pitched V1 note is missing pitch")
    step = pitch.findtext("step")
    octave_text = pitch.findtext("octave")
    if step is None or octave_text is None:
        raise Poly2DDatasetExecutionError("pitched V1 note has incomplete pitch")
    alter_text = pitch.findtext("alter")
    accidental_text = note.findtext("accidental")
    if accidental_text not in _ACCIDENTAL_MAP:
        raise Poly2DDatasetExecutionError("V1 display accidental is outside the V2 bridge")
    try:
        return PitchSpelling(
            step=step,
            alter=0 if alter_text is None else int(alter_text),
            octave=int(octave_text),
            display_accidental=_ACCIDENTAL_MAP[accidental_text],
        )
    except (TypeError, ValueError) as exc:
        raise Poly2DDatasetExecutionError("V1 pitch cannot enter the V2 bridge") from exc


def _reject_unsupported_note_surface(note: ET.Element) -> None:
    if note.findall("dot"):
        raise Poly2DDatasetExecutionError("dotted V1 targets are outside the exact TR-POLY-08C bridge")
    for name in ("tie", "beam", "time-modification", "grace", "notations"):
        if note.find(name) is not None:
            raise Poly2DDatasetExecutionError(f"V1 {name} surface is outside the exact TR-POLY-08C bridge")


def bridge_supported_v1_musicxml_to_v2(data: object) -> PolyScore:
    """Bridge only Stage-2-C-valid ST-OMR V1 MusicXML to an explicit V2 score.

    Every admitted event must be voice 1 on staff 1. The bridge is therefore
    training-plumbing evidence, not evidence for independent polyphonic voices.
    """

    if not isinstance(data, bytes) or not data:
        raise Poly2DDatasetExecutionError("MusicXML target must be non-empty bytes")
    validation = validate_musicxml(data)
    if not validation.is_valid:
        codes = ", ".join(issue.code for issue in validation.issues)
        raise Poly2DDatasetExecutionError(f"MusicXML target failed Stage 2-C validation: {codes}")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise Poly2DDatasetExecutionError("validated V1 MusicXML could not be parsed") from exc

    parts = root.findall("part")
    if len(parts) != 1 or parts[0].attrib.get("id") != "P1":
        raise Poly2DDatasetExecutionError("TR-POLY-08C V1 bridge requires exactly part P1")
    part = parts[0]
    measures = part.findall("measure")
    if not measures:
        raise Poly2DDatasetExecutionError("V1 bridge target contains no measures")

    divisions: int | None = None
    active_time: tuple[int, int] | None = None
    active_fifths = 0
    active_clef: tuple[str, int | None] | None = None
    v2_measures: list[PolyMeasure] = []

    for measure_index, measure in enumerate(measures, start=1):
        attributes = measure.find("attributes")
        if attributes is not None:
            divisions_text = attributes.findtext("divisions")
            if divisions_text is not None:
                try:
                    divisions = int(divisions_text)
                except ValueError as exc:
                    raise Poly2DDatasetExecutionError("V1 divisions must be an integer") from exc
                if divisions <= 0:
                    raise Poly2DDatasetExecutionError("V1 divisions must be positive")
            time = attributes.find("time")
            if time is not None:
                beats = _required_int(time, "beats")
                beat_type = _required_int(time, "beat-type")
                if beats <= 0 or beat_type <= 0:
                    raise Poly2DDatasetExecutionError("V1 time signature must be positive")
                active_time = (beats, beat_type)
            key = attributes.find("key")
            if key is not None:
                active_fifths = _required_int(key, "fifths")
            clef = attributes.find("clef")
            if clef is not None:
                sign = clef.findtext("sign")
                if sign is None:
                    raise Poly2DDatasetExecutionError("V1 clef is missing sign")
                line_text = clef.findtext("line")
                try:
                    line = None if line_text is None else int(line_text)
                except ValueError as exc:
                    raise Poly2DDatasetExecutionError("V1 clef line must be an integer") from exc
                active_clef = (sign, line)

        if divisions is None or active_time is None or active_clef is None:
            raise Poly2DDatasetExecutionError("V1 target lacks active divisions/time/clef state")
        if measure.findall("barline"):
            raise Poly2DDatasetExecutionError("V1 barlines are outside the exact TR-POLY-08C bridge")

        notes = [child for child in measure if child.tag == "note"]
        if not notes:
            raise Poly2DDatasetExecutionError("V1 measure contains no note/rest events")
        events: list[PolyEvent] = []
        cursor = Fraction(0, 1)
        note_index = 0
        event_index = 0
        while note_index < len(notes):
            base = notes[note_index]
            _reject_unsupported_note_surface(base)
            if base.find("chord") is not None:
                raise Poly2DDatasetExecutionError("V1 chord continuation has no base note")
            duration_units = _required_int(base, "duration")
            if duration_units <= 0:
                raise Poly2DDatasetExecutionError("V1 event duration units must be positive")
            duration = Fraction(duration_units, 4 * divisions)
            voice = _required_int(base, "voice")
            staff = _required_int(base, "staff")
            if voice != 1 or staff != 1:
                raise Poly2DDatasetExecutionError("V1 bridge admits only source voice 1 / staff 1")
            visible_type = _note_type(base)
            if duration != _NOTE_TYPE_DURATION[visible_type]:
                raise Poly2DDatasetExecutionError("V1 duration and visible note type disagree")

            event_index += 1
            event_id = f"m{measure_index}-e{event_index}"
            onset = ExactRational(cursor.numerator, cursor.denominator)
            exact_duration = ExactRational(duration.numerator, duration.denominator)

            if base.find("rest") is not None:
                events.append(
                    PolyEvent(
                        event_id=event_id,
                        kind=EventKind.REST,
                        onset=onset,
                        duration=exact_duration,
                        voice=1,
                        staff=1,
                        note_type=visible_type,
                    )
                )
                cursor += duration
                note_index += 1
                continue

            noteheads = [NoteAtom(atom_id=f"{event_id}-a1", pitch=_pitch(base))]
            next_index = note_index + 1
            member_index = 1
            while next_index < len(notes) and notes[next_index].find("chord") is not None:
                member = notes[next_index]
                _reject_unsupported_note_surface(member)
                if member.find("rest") is not None:
                    raise Poly2DDatasetExecutionError("V1 chord continuation cannot be a rest")
                member_duration = Fraction(_required_int(member, "duration"), 4 * divisions)
                member_voice = _required_int(member, "voice")
                member_staff = _required_int(member, "staff")
                member_type = _note_type(member)
                if (
                    member_duration != duration
                    or member_voice != voice
                    or member_staff != staff
                    or member_type is not visible_type
                ):
                    raise Poly2DDatasetExecutionError("V1 chord members disagree on duration/voice/staff/type")
                member_index += 1
                noteheads.append(NoteAtom(atom_id=f"{event_id}-a{member_index}", pitch=_pitch(member)))
                next_index += 1

            events.append(
                PolyEvent(
                    event_id=event_id,
                    kind=EventKind.NOTE if len(noteheads) == 1 else EventKind.CHORD,
                    onset=onset,
                    duration=exact_duration,
                    voice=1,
                    staff=1,
                    note_type=visible_type,
                    noteheads=tuple(noteheads),
                )
            )
            cursor += duration
            note_index = next_index

        v2_measures.append(
            PolyMeasure(
                measure_index=measure_index,
                source_number=measure.attrib.get("number", str(measure_index)),
                time_signature=TimeSignature(beats=(active_time[0],), beat_type=active_time[1]),
                key_signature=KeySignature(fifths=active_fifths),
                clefs=(ClefAssignment(staff=1, sign=active_clef[0], line=active_clef[1]),),
                events=tuple(events),
            )
        )

    score = PolyScore(parts=(PolyPart(part_id="P1", staff_count=1, measures=tuple(v2_measures)),))
    validate_roundtrip(score)
    return score


def poly_2d_materialization_fingerprint(
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
) -> str:
    if not isinstance(model_config, Poly2DTransformerConfig):
        raise TypeError("model_config must be Poly2DTransformerConfig")
    preprocess = InputPreprocessConfig(
        target_height=model_config.input_height,
        target_width=model_config.input_width,
    )
    payload = {
        "execution_version": POLY_2D_DATASET_EXECUTION_VERSION,
        "bridge_version": POLY_2D_V1_BRIDGE_VERSION,
        "target_profile": POLY_2D_TARGET_PROFILE,
        "source_class": DATASET_SOURCE_CLASS,
        "preprocess_fingerprint": preprocess_config_fingerprint(preprocess),
        "model_profile_sha256": poly_2d_config_fingerprint(model_config),
        "tokenizer_fingerprint_sha256": tokenizer_fingerprint(),
        "test_policy": "sealed-no-artifact-read",
        "semantic_truncation": "forbidden",
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _verify_dataset_root(build: SyntheticDatasetBuild, root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise Poly2DDatasetExecutionError("persisted Stage 6 dataset root must be a non-symlink directory")
    validation = validate_dataset_manifest(build.manifest)
    if not validation.is_valid:
        codes = ", ".join(issue.code for issue in validation.issues)
        raise Poly2DDatasetExecutionError(f"Stage 5 manifest vetoed the build: {codes}")
    if build.manifest.source_class != DATASET_SOURCE_CLASS:
        raise Poly2DDatasetExecutionError("TR-POLY-08C accepts only the frozen synthetic Stage 6 source class")
    expected_manifest = canonical_manifest_bytes(build.manifest)
    if dataset_manifest_sha256(build.manifest) != build.manifest_sha256:
        raise Poly2DDatasetExecutionError("Stage 6 manifest identity is inconsistent")
    checks: tuple[tuple[Path, bytes, str], ...] = (
        (root / "manifest.json", expected_manifest, "manifest"),
        (root / "build.json", build_metadata_bytes(build), "build metadata"),
        (
            root / "manifest.sha256",
            f"{build.manifest_sha256}  manifest.json\n".encode("ascii"),
            "manifest checksum",
        ),
    )
    for path, expected, label in checks:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise Poly2DDatasetExecutionError(f"persisted {label} differs from the validated Stage 6 build")


def _selected_samples(
    build: SyntheticDatasetBuild,
    split: DatasetSplit,
    max_samples: int,
) -> tuple[DatasetSample, ...]:
    if split is DatasetSplit.TEST:
        raise Poly2DDatasetExecutionError("Stage 6 TEST remains sealed in TR-POLY-08C")
    if split not in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}:
        raise Poly2DDatasetExecutionError("split must be TRAIN or VALIDATION")
    if not _plain_int(max_samples) or not 1 <= max_samples <= MAX_POLY_2D_MATERIALIZED_SAMPLES:
        raise Poly2DDatasetExecutionError("max_samples is outside the bounded TR-POLY-08C batch range")
    selected = tuple(
        sorted(
            (sample for sample in build.manifest.samples if sample.split is split),
            key=lambda item: item.sample_id,
        )[:max_samples]
    )
    if not selected:
        raise Poly2DDatasetExecutionError(f"validated Stage 6 build has no {split.value} samples")
    return selected


@dataclass(slots=True)
class Poly2DMaterializedSample:
    sample_id: str
    family_id: str
    split: DatasetSplit
    image_sha256: str
    target_sha256: str
    representation_sha256: str
    target: TokenizedPolyphonicTarget
    image: torch.Tensor
    source_width: int
    source_height: int
    target_profile: str = POLY_2D_TARGET_PROFILE

    def __post_init__(self) -> None:
        if self.split not in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}:
            raise Poly2DDatasetExecutionError("materialized sample may not expose TEST")
        if self.target_profile != POLY_2D_TARGET_PROFILE:
            raise Poly2DDatasetExecutionError("materialized target profile mismatch")
        if not isinstance(self.target, TokenizedPolyphonicTarget):
            raise Poly2DDatasetExecutionError("materialized target must use the frozen V2 tokenizer")
        if self.target.representation_sha256 != self.representation_sha256:
            raise Poly2DDatasetExecutionError("materialized representation identity mismatch")
        if not isinstance(self.image, torch.Tensor) or self.image.dtype != torch.float32 or self.image.ndim != 3:
            raise Poly2DDatasetExecutionError("materialized image must be float32 [1,height,width]")
        if not bool(torch.isfinite(self.image).all()) or bool((self.image < 0).any()) or bool((self.image > 1).any()):
            raise Poly2DDatasetExecutionError("materialized image is outside finite normalized [0,1]")


def materialize_poly_2d_samples(
    build: object,
    dataset_root: str | Path,
    split: DatasetSplit,
    *,
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
    max_samples: int = MAX_POLY_2D_MATERIALIZED_SAMPLES,
) -> tuple[Poly2DMaterializedSample, ...]:
    """Materialize only exact TRAIN/VALIDATION artifacts; TEST is rejected first."""

    if split is DatasetSplit.TEST:
        raise Poly2DDatasetExecutionError("Stage 6 TEST remains sealed in TR-POLY-08C")
    if not isinstance(build, SyntheticDatasetBuild):
        raise TypeError("build must be SyntheticDatasetBuild")
    if not isinstance(dataset_root, (str, Path)):
        raise TypeError("dataset_root must be str or pathlib.Path")
    if not isinstance(model_config, Poly2DTransformerConfig):
        raise TypeError("model_config must be Poly2DTransformerConfig")

    selected = _selected_samples(build, split, max_samples)
    root = Path(dataset_root)
    _verify_dataset_root(build, root)
    preprocess = InputPreprocessConfig(
        target_height=model_config.input_height,
        target_width=model_config.input_width,
    )

    materialized: list[Poly2DMaterializedSample] = []
    for sample in selected:
        target_path = root / "targets" / f"{sample.source_musicxml_sha256}.musicxml"
        image_path = root / "images" / f"{sample.png_sha256}.png"
        for path, label in ((target_path, "target"), (image_path, "image")):
            if path.is_symlink() or not path.is_file():
                raise Poly2DDatasetExecutionError(f"selected Stage 6 {label} artifact is missing or symlinked")

        target_bytes = target_path.read_bytes()
        if _sha256_bytes(target_bytes) != sample.source_musicxml_sha256:
            raise Poly2DDatasetExecutionError("selected Stage 6 MusicXML target hash mismatch")
        score = bridge_supported_v1_musicxml_to_v2(target_bytes)
        tokenized = validate_roundtrip(score)
        decoder_length = len(tokenized.token_ids) - 1
        if decoder_length < 1 or decoder_length > model_config.max_target_tokens:
            raise Poly2DDatasetExecutionError(
                "V2 target exceeds model max_target_tokens; semantic truncation is forbidden"
            )

        image_bytes = image_path.read_bytes()
        if _sha256_bytes(image_bytes) != sample.png_sha256:
            raise Poly2DDatasetExecutionError("selected Stage 6 PNG image hash mismatch")
        try:
            image = preprocess_grayscale_png(
                image_bytes,
                preprocess,
                expected_width=sample.width,
                expected_height=sample.height,
            )
        except TrainingDataError as exc:
            raise Poly2DDatasetExecutionError("selected Stage 6 PNG failed deterministic preprocessing") from exc

        materialized.append(
            Poly2DMaterializedSample(
                sample_id=sample.sample_id,
                family_id=sample.family_id,
                split=sample.split,
                image_sha256=sample.png_sha256,
                target_sha256=sample.source_musicxml_sha256,
                representation_sha256=score.canonical_sha256(),
                target=tokenized,
                image=image,
                source_width=sample.width,
                source_height=sample.height,
            )
        )
    return tuple(materialized)


def make_poly_2d_training_batch(
    samples: object,
    *,
    dataset_manifest_sha256: str,
) -> Poly2DTrainingBatch:
    if not isinstance(samples, tuple) or not samples:
        raise Poly2DDatasetExecutionError("samples must be a non-empty immutable tuple")
    if len(samples) > MAX_POLY_2D_TRAINING_BATCH or any(
        not isinstance(sample, Poly2DMaterializedSample) for sample in samples
    ):
        raise Poly2DDatasetExecutionError("samples violate the bounded materialized batch contract")
    split = samples[0].split
    if any(sample.split is not split for sample in samples):
        raise Poly2DDatasetExecutionError("one V2 batch may not mix dataset splits")
    if split is DatasetSplit.TEST:
        raise Poly2DDatasetExecutionError("TEST cannot enter a V2 training batch")

    sequences = tuple(sample.target.token_ids for sample in samples)
    if any(sequence[0] != BOS_TOKEN_ID or sequence[-1] != EOS_TOKEN_ID for sequence in sequences):
        raise Poly2DDatasetExecutionError("materialized V2 target lacks BOS/EOS")
    lengths = tuple(len(sequence) - 1 for sequence in sequences)
    maximum = max(lengths)
    decoder = torch.full((len(samples), maximum), PAD_TOKEN_ID, dtype=torch.long)
    labels = torch.full((len(samples), maximum), PAD_TOKEN_ID, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        length = len(sequence) - 1
        decoder[row, :length] = torch.tensor(sequence[:-1], dtype=torch.long)
        labels[row, :length] = torch.tensor(sequence[1:], dtype=torch.long)
    images = torch.stack(tuple(sample.image for sample in samples), dim=0)
    return Poly2DTrainingBatch(
        images=images,
        decoder_input_ids=decoder,
        labels=labels,
        split=split,
        sample_ids=tuple(sample.sample_id for sample in samples),
        dataset_manifest_sha256=dataset_manifest_sha256,
    )


@dataclass(frozen=True, slots=True)
class Poly2DExactDatasetExecutionResult:
    checkpoint: Poly2DCheckpointReceipt
    dataset_manifest_sha256: str
    materialization_fingerprint_sha256: str
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    source_class: str = DATASET_SOURCE_CLASS
    target_profile: str = POLY_2D_TARGET_PROFILE
    controlled_dataset_execution: bool = True
    authoritative_dataset_execution: bool = False
    test_split_accessed: bool = False
    benchmark_evidence: bool = False
    polyphonic_voice_evidence: bool = False
    production_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, Poly2DCheckpointReceipt):
            raise Poly2DDatasetExecutionError("execution result checkpoint is invalid")
        if self.source_class != DATASET_SOURCE_CLASS or self.target_profile != POLY_2D_TARGET_PROFILE:
            raise Poly2DDatasetExecutionError("execution result source/target profile mismatch")
        if not self.controlled_dataset_execution:
            raise Poly2DDatasetExecutionError("execution result must record actual controlled dataset execution")
        if any(
            (
                self.authoritative_dataset_execution,
                self.test_split_accessed,
                self.benchmark_evidence,
                self.polyphonic_voice_evidence,
                self.production_authority,
            )
        ):
            raise Poly2DDatasetExecutionError(
                "TR-POLY-08C may not claim authority, TEST, benchmark, or polyphonic evidence"
            )


def execute_exact_stage6_poly_2d_training(
    *,
    build: SyntheticDatasetBuild,
    dataset_root: str | Path,
    repository_sha: str,
    output_directory: Path,
    training_config: Poly2DTrainingConfig = FROZEN_POLY_2D_TRAINING_CONFIG,
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
    max_train_samples: int = MAX_POLY_2D_MATERIALIZED_SAMPLES,
    max_validation_samples: int = MAX_POLY_2D_MATERIALIZED_SAMPLES,
) -> Poly2DExactDatasetExecutionResult:
    """Train the bounded prototype on exact persisted Stage 6 TRAIN/VALIDATION bytes."""

    _require_git_sha(repository_sha)
    train_samples = materialize_poly_2d_samples(
        build,
        dataset_root,
        DatasetSplit.TRAIN,
        model_config=model_config,
        max_samples=max_train_samples,
    )
    validation_samples = materialize_poly_2d_samples(
        build,
        dataset_root,
        DatasetSplit.VALIDATION,
        model_config=model_config,
        max_samples=max_validation_samples,
    )
    train_families = {sample.family_id for sample in train_samples}
    validation_families = {sample.family_id for sample in validation_samples}
    if train_families & validation_families:
        raise Poly2DDatasetExecutionError("TRAIN/VALIDATION family leakage detected at execution boundary")

    train_batch = make_poly_2d_training_batch(
        train_samples,
        dataset_manifest_sha256=build.manifest_sha256,
    )
    validation_batch = make_poly_2d_training_batch(
        validation_samples,
        dataset_manifest_sha256=build.manifest_sha256,
    )
    materialization_sha = poly_2d_materialization_fingerprint(model_config)
    provenance = build_poly_2d_training_provenance(
        repository_sha=repository_sha,
        dataset_manifest_sha256=build.manifest_sha256,
        preprocess_fingerprint_sha256=materialization_sha,
        training_config=training_config,
        model_config=model_config,
    )
    checkpoint = run_and_persist_bounded_poly_2d_checkpoint(
        train_batches=(train_batch,),
        validation_batch=validation_batch,
        provenance=provenance,
        output_directory=output_directory,
        training_config=training_config,
        model_config=model_config,
    )
    loaded = load_and_verify_poly_2d_checkpoint(output_directory)
    if loaded.metadata.dataset_manifest_sha256 != build.manifest_sha256:
        raise Poly2DDatasetExecutionError("checkpoint dataset identity differs from executed Stage 6 build")
    if loaded.metadata.preprocess_fingerprint_sha256 != materialization_sha:
        raise Poly2DDatasetExecutionError("checkpoint materialization identity differs from executed data path")
    if (
        loaded.metadata.authoritative_dataset_execution
        or loaded.metadata.test_split_accessed
        or loaded.metadata.benchmark_evidence
        or loaded.metadata.production_authority
    ):
        raise Poly2DDatasetExecutionError("TR-POLY-08B checkpoint exceeded the TR-POLY-08C claim boundary")

    return Poly2DExactDatasetExecutionResult(
        checkpoint=checkpoint,
        dataset_manifest_sha256=build.manifest_sha256,
        materialization_fingerprint_sha256=materialization_sha,
        train_sample_ids=tuple(sample.sample_id for sample in train_samples),
        validation_sample_ids=tuple(sample.sample_id for sample in validation_samples),
    )
