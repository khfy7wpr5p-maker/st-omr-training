"""TR-POLY-09A native Polyphonic V2 dataset materialization.

This module is additive to the frozen Stage 6/V1 bridge.  It admits only
explicit canonical Polyphonic Representation V2 targets paired with hash-bound
PNG images, preserves family-exclusive TRAIN/VALIDATION/TEST metadata, refuses
to read TEST artifacts, and feeds verified TRAIN/VALIDATION targets to the
existing bounded 2D Transformer training/checkpoint chain.

It does not parse MusicXML, infer missing voice/onset/duration fields, benchmark
a model, or grant ScoreMosaic/production authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
from typing import Final

from PIL import Image
import torch

from .dataset_manifest import DatasetSplit, MAX_IMAGE_PIXELS
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
from .polyphonic_representation import EventKind, PolyScore
from .polyphonic_serialization import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    TokenizedPolyphonicTarget,
    parse_canonical_polyphonic_json,
    tokenizer_fingerprint,
    validate_roundtrip,
)
from .training_data import (
    InputPreprocessConfig,
    TrainingDataError,
    preprocess_config_fingerprint,
    preprocess_grayscale_png,
)


NATIVE_POLY_V2_MANIFEST_SCHEMA_VERSION: Final[str] = "st-omr-native-poly-v2-manifest-v1"
NATIVE_POLY_V2_BUILD_VERSION: Final[str] = "st-omr-native-poly-v2-build-v1"
NATIVE_POLY_V2_MATERIALIZATION_VERSION: Final[str] = "st-omr-native-poly-v2-materialization-v1"
NATIVE_POLY_V2_SOURCE_CLASS: Final[str] = "explicit_polyphonic_v2"
NATIVE_POLY_V2_TARGET_PROFILE: Final[str] = "native_polyphonic_v2"
NATIVE_POLY_V2_SPLIT_POLICY: Final[str] = "family-exclusive-v1"
NATIVE_POLY_V2_TEST_POLICY: Final[str] = "sealed-no-artifact-read"
MAX_NATIVE_POLY_V2_SAMPLES: Final[int] = 1_000_000
MAX_NATIVE_POLY_V2_ARTIFACT_BYTES: Final[int] = 64 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class NativePolyV2DatasetError(TrainingDataError):
    """Raised when a native V2 dataset boundary fails closed."""


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


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise NativePolyV2DatasetError(f"{name} must be lowercase SHA-256 hex")
    return value


def _require_id(name: str, value: object) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise NativePolyV2DatasetError(f"{name} must match the bounded identifier contract")
    return value


def _plain_positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise NativePolyV2DatasetError(f"{name} must be a positive integer")
    return value


def _inspect_png(data: bytes) -> tuple[int, int]:
    if not isinstance(data, bytes) or not data or len(data) > MAX_NATIVE_POLY_V2_ARTIFACT_BYTES:
        raise NativePolyV2DatasetError("native V2 image must be bounded non-empty bytes")
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise NativePolyV2DatasetError("native V2 image must be PNG")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or image.mode != "L":
                raise NativePolyV2DatasetError("native V2 image must be grayscale PNG mode L")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise NativePolyV2DatasetError("native V2 image dimensions exceed the bounded contract")
            image.load()
    except NativePolyV2DatasetError:
        raise
    except Exception as exc:
        raise NativePolyV2DatasetError("native V2 PNG decode failed") from exc
    return width, height


@dataclass(frozen=True, slots=True)
class NativePolyV2TargetProfile:
    voices: tuple[int, ...]
    event_kinds: tuple[str, ...]
    has_simultaneous_independent_voices: bool
    has_chord_with_independent_voice_same_onset: bool
    tie_count: int
    beam_count: int
    tuplet_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.voices, tuple) or not self.voices:
            raise NativePolyV2DatasetError("target profile requires explicit voices")
        if tuple(sorted(set(self.voices))) != self.voices or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in self.voices
        ):
            raise NativePolyV2DatasetError("target profile voices must be sorted unique positive integers")
        expected_kinds = tuple(sorted(set(self.event_kinds)))
        if self.event_kinds != expected_kinds or any(kind not in {item.value for item in EventKind} for kind in self.event_kinds):
            raise NativePolyV2DatasetError("target profile event kinds are invalid")
        for name in ("tie_count", "beam_count", "tuplet_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise NativePolyV2DatasetError(f"{name} must be a non-negative integer")
        if not isinstance(self.has_simultaneous_independent_voices, bool) or not isinstance(
            self.has_chord_with_independent_voice_same_onset, bool
        ):
            raise NativePolyV2DatasetError("target profile simultaneous flags must be bool")
        if self.has_chord_with_independent_voice_same_onset and not self.has_simultaneous_independent_voices:
            raise NativePolyV2DatasetError("chord/independent-voice distinction requires simultaneous voices")

    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)


def profile_polyphonic_score(score: object) -> NativePolyV2TargetProfile:
    if not isinstance(score, PolyScore):
        raise TypeError("score must be PolyScore")
    voices: set[int] = set()
    kinds: set[str] = set()
    simultaneous = False
    chord_with_independent = False
    tie_count = 0
    beam_count = 0
    tuplet_count = 0
    for part in score.parts:
        for measure in part.measures:
            by_onset: dict[object, list[object]] = {}
            for event in measure.events:
                voices.add(event.voice)
                kinds.add(event.kind.value)
                tie_count += sum(len(note.ties) for note in event.noteheads)
                beam_count += len(event.beams)
                tuplet_count += len(event.tuplets)
                by_onset.setdefault(event.onset.fraction, []).append(event)
            for events in by_onset.values():
                event_voices = {event.voice for event in events}
                if len(event_voices) >= 2:
                    simultaneous = True
                    if any(event.kind is EventKind.CHORD for event in events):
                        chord_with_independent = True
    if not voices:
        raise NativePolyV2DatasetError("native V2 target contains no events")
    return NativePolyV2TargetProfile(
        voices=tuple(sorted(voices)),
        event_kinds=tuple(sorted(kinds)),
        has_simultaneous_independent_voices=simultaneous,
        has_chord_with_independent_voice_same_onset=chord_with_independent,
        tie_count=tie_count,
        beam_count=beam_count,
        tuplet_count=tuplet_count,
    )


@dataclass(frozen=True, slots=True)
class NativePolyV2Sample:
    sample_id: str
    family_id: str
    split: DatasetSplit
    target_sha256: str
    representation_sha256: str
    image_sha256: str
    width: int
    height: int
    target_token_count: int
    profile: NativePolyV2TargetProfile

    def __post_init__(self) -> None:
        _require_sha256("sample_id", self.sample_id)
        _require_id("family_id", self.family_id)
        if not isinstance(self.split, DatasetSplit):
            raise NativePolyV2DatasetError("split must be DatasetSplit")
        _require_sha256("target_sha256", self.target_sha256)
        _require_sha256("representation_sha256", self.representation_sha256)
        _require_sha256("image_sha256", self.image_sha256)
        _plain_positive_int("width", self.width)
        _plain_positive_int("height", self.height)
        if self.width * self.height > MAX_IMAGE_PIXELS:
            raise NativePolyV2DatasetError("sample image dimensions exceed the bounded contract")
        _plain_positive_int("target_token_count", self.target_token_count)
        if not isinstance(self.profile, NativePolyV2TargetProfile):
            raise NativePolyV2DatasetError("sample profile must be NativePolyV2TargetProfile")


def _sample_identity_payload(sample: NativePolyV2Sample) -> dict[str, object]:
    return {
        "schema_version": NATIVE_POLY_V2_MANIFEST_SCHEMA_VERSION,
        "family_id": sample.family_id,
        "target_sha256": sample.target_sha256,
        "representation_sha256": sample.representation_sha256,
        "image_sha256": sample.image_sha256,
        "width": sample.width,
        "height": sample.height,
        "target_token_count": sample.target_token_count,
        "profile": sample.profile.canonical_payload(),
    }


def native_poly_v2_sample_id(sample: NativePolyV2Sample) -> str:
    return sha256(_canonical_json_bytes(_sample_identity_payload(sample))).hexdigest()


def _validate_sample_identity(sample: NativePolyV2Sample) -> None:
    if sample.sample_id != native_poly_v2_sample_id(sample):
        raise NativePolyV2DatasetError("native V2 sample identity mismatch")


@dataclass(frozen=True, slots=True)
class NativePolyV2Manifest:
    dataset_name: str
    dataset_version: str
    samples: tuple[NativePolyV2Sample, ...]
    schema_version: str = NATIVE_POLY_V2_MANIFEST_SCHEMA_VERSION
    source_class: str = NATIVE_POLY_V2_SOURCE_CLASS
    target_profile: str = NATIVE_POLY_V2_TARGET_PROFILE
    split_policy: str = NATIVE_POLY_V2_SPLIT_POLICY
    tokenizer_fingerprint_sha256: str = ""
    test_policy: str = NATIVE_POLY_V2_TEST_POLICY

    def __post_init__(self) -> None:
        _require_id("dataset_name", self.dataset_name)
        _require_id("dataset_version", self.dataset_version)
        if not isinstance(self.samples, tuple) or not self.samples or len(self.samples) > MAX_NATIVE_POLY_V2_SAMPLES:
            raise NativePolyV2DatasetError("native V2 manifest samples are empty or exceed the bounded contract")
        if any(not isinstance(sample, NativePolyV2Sample) for sample in self.samples):
            raise NativePolyV2DatasetError("native V2 manifest samples must be immutable sample records")
        if self.schema_version != NATIVE_POLY_V2_MANIFEST_SCHEMA_VERSION:
            raise NativePolyV2DatasetError("unsupported native V2 manifest schema")
        if self.source_class != NATIVE_POLY_V2_SOURCE_CLASS:
            raise NativePolyV2DatasetError("native V2 source class mismatch")
        if self.target_profile != NATIVE_POLY_V2_TARGET_PROFILE:
            raise NativePolyV2DatasetError("native V2 target profile mismatch")
        if self.split_policy != NATIVE_POLY_V2_SPLIT_POLICY:
            raise NativePolyV2DatasetError("native V2 split policy mismatch")
        expected_tokenizer = tokenizer_fingerprint()
        if self.tokenizer_fingerprint_sha256 in {"", expected_tokenizer}:
            object.__setattr__(self, "tokenizer_fingerprint_sha256", expected_tokenizer)
        else:
            raise NativePolyV2DatasetError("native V2 manifest tokenizer fingerprint mismatch")
        if self.test_policy != NATIVE_POLY_V2_TEST_POLICY:
            raise NativePolyV2DatasetError("native V2 TEST policy mismatch")
        validate_native_poly_v2_manifest(self)


def _profile_payload(profile: NativePolyV2TargetProfile) -> dict[str, object]:
    return profile.canonical_payload()


def _sample_payload(sample: NativePolyV2Sample) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "family_id": sample.family_id,
        "split": sample.split.value,
        "target_sha256": sample.target_sha256,
        "representation_sha256": sample.representation_sha256,
        "image_sha256": sample.image_sha256,
        "width": sample.width,
        "height": sample.height,
        "target_token_count": sample.target_token_count,
        "profile": _profile_payload(sample.profile),
    }


def validate_native_poly_v2_manifest(manifest: object) -> None:
    if not isinstance(manifest, NativePolyV2Manifest):
        raise NativePolyV2DatasetError("manifest must be NativePolyV2Manifest")
    family_split: dict[str, DatasetSplit] = {}
    target_split: dict[str, DatasetSplit] = {}
    image_split: dict[str, DatasetSplit] = {}
    sample_ids: set[str] = set()
    split_counts = {split: 0 for split in DatasetSplit}
    train_validation_profiles: list[NativePolyV2TargetProfile] = []
    for sample in manifest.samples:
        _validate_sample_identity(sample)
        if sample.sample_id in sample_ids:
            raise NativePolyV2DatasetError("duplicate native V2 sample_id")
        sample_ids.add(sample.sample_id)
        split_counts[sample.split] += 1
        prior = family_split.get(sample.family_id)
        if prior is not None and prior is not sample.split:
            raise NativePolyV2DatasetError("TRAIN/VALIDATION/TEST family leakage detected")
        family_split.setdefault(sample.family_id, sample.split)
        for value, seen, label in (
            (sample.target_sha256, target_split, "target"),
            (sample.image_sha256, image_split, "image"),
        ):
            prior_split = seen.get(value)
            if prior_split is not None and prior_split is not sample.split:
                raise NativePolyV2DatasetError(f"identical {label} artifact appears in multiple splits")
            seen.setdefault(value, sample.split)
        if sample.split in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}:
            train_validation_profiles.append(sample.profile)
    for split, count in split_counts.items():
        if count == 0:
            raise NativePolyV2DatasetError(f"native V2 manifest requires sealed metadata for {split.value}")
    if not train_validation_profiles:
        raise NativePolyV2DatasetError("native V2 manifest has no TRAIN/VALIDATION targets")
    voices = {voice for profile in train_validation_profiles for voice in profile.voices}
    kinds = {kind for profile in train_validation_profiles for kind in profile.event_kinds}
    if 2 not in voices or 3 not in voices or not any(voice >= 4 for voice in voices):
        raise NativePolyV2DatasetError("native V2 TRAIN/VALIDATION coverage requires voice 2, voice 3 and voice 4+")
    if not any(profile.has_simultaneous_independent_voices for profile in train_validation_profiles):
        raise NativePolyV2DatasetError("native V2 coverage requires simultaneous independent voices")
    if not any(profile.has_chord_with_independent_voice_same_onset for profile in train_validation_profiles):
        raise NativePolyV2DatasetError("native V2 coverage requires chord versus independent-voice distinction")
    if not {item.value for item in EventKind}.issubset(kinds):
        raise NativePolyV2DatasetError("native V2 coverage requires note, rest and chord events")


def canonical_native_poly_v2_manifest_bytes(manifest: NativePolyV2Manifest) -> bytes:
    validate_native_poly_v2_manifest(manifest)
    samples = sorted(manifest.samples, key=lambda item: (item.family_id, item.sample_id, item.split.value))
    payload = {
        "schema_version": manifest.schema_version,
        "source_class": manifest.source_class,
        "target_profile": manifest.target_profile,
        "split_policy": manifest.split_policy,
        "test_policy": manifest.test_policy,
        "tokenizer_fingerprint_sha256": manifest.tokenizer_fingerprint_sha256,
        "dataset_name": manifest.dataset_name,
        "dataset_version": manifest.dataset_version,
        "samples": [_sample_payload(sample) for sample in samples],
    }
    return _canonical_json_bytes(payload)


def native_poly_v2_manifest_sha256(manifest: NativePolyV2Manifest) -> str:
    return sha256(canonical_native_poly_v2_manifest_bytes(manifest)).hexdigest()


@dataclass(frozen=True, slots=True)
class NativePolyV2ArtifactInput:
    family_id: str
    split: DatasetSplit
    target_json: bytes
    image_png: bytes

    def __post_init__(self) -> None:
        _require_id("family_id", self.family_id)
        if not isinstance(self.split, DatasetSplit):
            raise NativePolyV2DatasetError("split must be DatasetSplit")
        # Fail closed before parsing/inspecting any TEST bytes.
        if self.split is DatasetSplit.TEST:
            raise NativePolyV2DatasetError("TEST artifact bytes may not enter TR-POLY-09A materialization")
        if not isinstance(self.target_json, bytes) or not self.target_json:
            raise NativePolyV2DatasetError("native V2 target must be non-empty canonical JSON bytes")
        if len(self.target_json) > MAX_NATIVE_POLY_V2_ARTIFACT_BYTES:
            raise NativePolyV2DatasetError("native V2 target exceeds the artifact byte limit")
        if not isinstance(self.image_png, bytes) or not self.image_png:
            raise NativePolyV2DatasetError("native V2 image must be non-empty bytes")


@dataclass(frozen=True, slots=True)
class NativePolyV2TargetArtifact:
    sha256: str
    canonical_json: bytes

    def __post_init__(self) -> None:
        _require_sha256("target artifact sha256", self.sha256)
        if not isinstance(self.canonical_json, bytes) or _sha256_bytes(self.canonical_json) != self.sha256:
            raise NativePolyV2DatasetError("target artifact bytes/hash mismatch")


@dataclass(frozen=True, slots=True)
class NativePolyV2ImageArtifact:
    sha256: str
    png: bytes

    def __post_init__(self) -> None:
        _require_sha256("image artifact sha256", self.sha256)
        if not isinstance(self.png, bytes) or _sha256_bytes(self.png) != self.sha256:
            raise NativePolyV2DatasetError("image artifact bytes/hash mismatch")


def _record_from_input(item: NativePolyV2ArtifactInput) -> tuple[
    NativePolyV2Sample, NativePolyV2TargetArtifact, NativePolyV2ImageArtifact
]:
    # TEST is already rejected by the input type before artifact inspection.
    try:
        score = parse_canonical_polyphonic_json(item.target_json)
        target = validate_roundtrip(score)
    except Exception as exc:
        raise NativePolyV2DatasetError("native target is not lossless canonical Polyphonic V2") from exc
    profile = profile_polyphonic_score(score)
    if 2 not in profile.voices:
        raise NativePolyV2DatasetError("native V2 sample must contain an explicit voice 2")
    width, height = _inspect_png(item.image_png)
    target_sha = _sha256_bytes(item.target_json)
    image_sha = _sha256_bytes(item.image_png)
    provisional = NativePolyV2Sample(
        sample_id="0" * 64,
        family_id=item.family_id,
        split=item.split,
        target_sha256=target_sha,
        representation_sha256=score.canonical_sha256(),
        image_sha256=image_sha,
        width=width,
        height=height,
        target_token_count=len(target.token_ids),
        profile=profile,
    )
    sample = NativePolyV2Sample(
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
    return (
        sample,
        NativePolyV2TargetArtifact(sha256=target_sha, canonical_json=item.target_json),
        NativePolyV2ImageArtifact(sha256=image_sha, png=item.image_png),
    )


@dataclass(frozen=True, slots=True)
class NativePolyV2DatasetBuild:
    manifest: NativePolyV2Manifest
    manifest_sha256: str
    build_id: str
    targets: tuple[NativePolyV2TargetArtifact, ...]
    images: tuple[NativePolyV2ImageArtifact, ...]
    builder_version: str = NATIVE_POLY_V2_BUILD_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, NativePolyV2Manifest):
            raise NativePolyV2DatasetError("build manifest must be NativePolyV2Manifest")
        if self.manifest_sha256 != native_poly_v2_manifest_sha256(self.manifest):
            raise NativePolyV2DatasetError("native V2 build manifest identity mismatch")
        _require_sha256("build_id", self.build_id)
        if self.builder_version != NATIVE_POLY_V2_BUILD_VERSION:
            raise NativePolyV2DatasetError("native V2 builder version mismatch")
        if any(not isinstance(item, NativePolyV2TargetArtifact) for item in self.targets):
            raise NativePolyV2DatasetError("native V2 target artifacts are invalid")
        if any(not isinstance(item, NativePolyV2ImageArtifact) for item in self.images):
            raise NativePolyV2DatasetError("native V2 image artifacts are invalid")
        expected = native_poly_v2_build_id(self.manifest_sha256, self.targets, self.images)
        if self.build_id != expected:
            raise NativePolyV2DatasetError("native V2 deterministic build identity mismatch")


def native_poly_v2_build_id(
    manifest_sha256: str,
    targets: tuple[NativePolyV2TargetArtifact, ...],
    images: tuple[NativePolyV2ImageArtifact, ...],
) -> str:
    _require_sha256("manifest_sha256", manifest_sha256)
    payload = {
        "builder_version": NATIVE_POLY_V2_BUILD_VERSION,
        "manifest_sha256": manifest_sha256,
        "target_sha256": sorted(item.sha256 for item in targets),
        "image_sha256": sorted(item.sha256 for item in images),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def build_native_poly_v2_dataset(
    artifacts: object,
    *,
    sealed_test_samples: tuple[NativePolyV2Sample, ...],
    dataset_name: str = "st-omr-native-poly-v2",
    dataset_version: str = "v1",
) -> NativePolyV2DatasetBuild:
    if not isinstance(artifacts, tuple) or not artifacts or any(
        not isinstance(item, NativePolyV2ArtifactInput) for item in artifacts
    ):
        raise NativePolyV2DatasetError("artifacts must be a non-empty immutable NativePolyV2ArtifactInput tuple")
    if not isinstance(sealed_test_samples, tuple) or not sealed_test_samples:
        raise NativePolyV2DatasetError("sealed TEST metadata is required without TEST artifact bytes")
    if any(not isinstance(item, NativePolyV2Sample) or item.split is not DatasetSplit.TEST for item in sealed_test_samples):
        raise NativePolyV2DatasetError("sealed_test_samples must contain TEST metadata only")

    samples: list[NativePolyV2Sample] = []
    targets_by_sha: dict[str, NativePolyV2TargetArtifact] = {}
    images_by_sha: dict[str, NativePolyV2ImageArtifact] = {}
    for item in artifacts:
        sample, target, image = _record_from_input(item)
        samples.append(sample)
        targets_by_sha.setdefault(target.sha256, target)
        images_by_sha.setdefault(image.sha256, image)
    samples.extend(sealed_test_samples)
    manifest = NativePolyV2Manifest(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        samples=tuple(samples),
    )
    manifest_sha = native_poly_v2_manifest_sha256(manifest)
    targets = tuple(sorted(targets_by_sha.values(), key=lambda item: item.sha256))
    images = tuple(sorted(images_by_sha.values(), key=lambda item: item.sha256))
    build_id = native_poly_v2_build_id(manifest_sha, targets, images)
    return NativePolyV2DatasetBuild(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        build_id=build_id,
        targets=targets,
        images=images,
    )


def native_poly_v2_build_metadata_bytes(build: NativePolyV2DatasetBuild) -> bytes:
    if not isinstance(build, NativePolyV2DatasetBuild):
        raise TypeError("build must be NativePolyV2DatasetBuild")
    return _canonical_json_bytes(
        {
            "builder_version": build.builder_version,
            "build_id": build.build_id,
            "manifest_sha256": build.manifest_sha256,
            "source_class": NATIVE_POLY_V2_SOURCE_CLASS,
            "target_profile": NATIVE_POLY_V2_TARGET_PROFILE,
            "test_policy": NATIVE_POLY_V2_TEST_POLICY,
        }
    )


def persist_native_poly_v2_dataset(build: object, output_directory: str | Path) -> Path:
    if not isinstance(build, NativePolyV2DatasetBuild):
        raise TypeError("build must be NativePolyV2DatasetBuild")
    if not isinstance(output_directory, (str, Path)):
        raise TypeError("output_directory must be str or pathlib.Path")
    root = Path(output_directory)
    temporary = root.with_name(f"{root.name}.tmp")
    if root.exists() or root.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise NativePolyV2DatasetError("native V2 dataset persistence is one-shot and non-overwriting")
    parent = root.parent
    if parent.is_symlink() or not parent.is_dir():
        raise NativePolyV2DatasetError("native V2 dataset parent must be an existing non-symlink directory")
    try:
        temporary.mkdir()
        (temporary / "targets").mkdir()
        (temporary / "images").mkdir()
        (temporary / "manifest.json").write_bytes(canonical_native_poly_v2_manifest_bytes(build.manifest))
        (temporary / "manifest.sha256").write_bytes(
            f"{build.manifest_sha256}  manifest.json\n".encode("ascii")
        )
        (temporary / "build.json").write_bytes(native_poly_v2_build_metadata_bytes(build))
        for target in build.targets:
            (temporary / "targets" / f"{target.sha256}.json").write_bytes(target.canonical_json)
        for image in build.images:
            (temporary / "images" / f"{image.sha256}.png").write_bytes(image.png)
        _verify_dataset_root(build, temporary)
        temporary.rename(root)
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    _verify_dataset_root(build, root)
    return root


def _verify_dataset_root(build: NativePolyV2DatasetBuild, root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise NativePolyV2DatasetError("persisted native V2 dataset root must be a non-symlink directory")
    checks = (
        (root / "manifest.json", canonical_native_poly_v2_manifest_bytes(build.manifest), "manifest"),
        (root / "manifest.sha256", f"{build.manifest_sha256}  manifest.json\n".encode("ascii"), "manifest checksum"),
        (root / "build.json", native_poly_v2_build_metadata_bytes(build), "build metadata"),
    )
    for path, expected, label in checks:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise NativePolyV2DatasetError(f"persisted native V2 {label} differs from the validated build")


def native_poly_v2_materialization_fingerprint(
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
) -> str:
    if not isinstance(model_config, Poly2DTransformerConfig):
        raise TypeError("model_config must be Poly2DTransformerConfig")
    preprocess = InputPreprocessConfig(
        target_height=model_config.input_height,
        target_width=model_config.input_width,
    )
    payload = {
        "materialization_version": NATIVE_POLY_V2_MATERIALIZATION_VERSION,
        "manifest_schema_version": NATIVE_POLY_V2_MANIFEST_SCHEMA_VERSION,
        "source_class": NATIVE_POLY_V2_SOURCE_CLASS,
        "target_profile": NATIVE_POLY_V2_TARGET_PROFILE,
        "split_policy": NATIVE_POLY_V2_SPLIT_POLICY,
        "test_policy": NATIVE_POLY_V2_TEST_POLICY,
        "semantic_truncation": "forbidden",
        "preprocess_fingerprint": preprocess_config_fingerprint(preprocess),
        "model_profile_sha256": poly_2d_config_fingerprint(model_config),
        "tokenizer_fingerprint_sha256": tokenizer_fingerprint(),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(slots=True)
class NativePolyV2MaterializedSample:
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
    profile: NativePolyV2TargetProfile
    target_profile: str = NATIVE_POLY_V2_TARGET_PROFILE

    def __post_init__(self) -> None:
        if self.split not in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}:
            raise NativePolyV2DatasetError("materialized native V2 sample may not expose TEST")
        if self.target_profile != NATIVE_POLY_V2_TARGET_PROFILE:
            raise NativePolyV2DatasetError("materialized native V2 target profile mismatch")
        if not isinstance(self.target, TokenizedPolyphonicTarget):
            raise NativePolyV2DatasetError("materialized target must use the frozen V2 tokenizer")
        if self.target.representation_sha256 != self.representation_sha256:
            raise NativePolyV2DatasetError("materialized representation identity mismatch")
        if not isinstance(self.profile, NativePolyV2TargetProfile):
            raise NativePolyV2DatasetError("materialized target profile metadata mismatch")
        if not isinstance(self.image, torch.Tensor) or self.image.dtype != torch.float32 or self.image.ndim != 3:
            raise NativePolyV2DatasetError("materialized image must be float32 [1,height,width]")


def _selected_samples(
    build: NativePolyV2DatasetBuild,
    split: DatasetSplit,
    max_samples: int,
) -> tuple[NativePolyV2Sample, ...]:
    if split is DatasetSplit.TEST:
        raise NativePolyV2DatasetError("TEST remains sealed in TR-POLY-09A")
    if split not in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}:
        raise NativePolyV2DatasetError("split must be TRAIN or VALIDATION")
    if not isinstance(max_samples, int) or isinstance(max_samples, bool) or not 1 <= max_samples <= MAX_POLY_2D_TRAINING_BATCH:
        raise NativePolyV2DatasetError("max_samples is outside the bounded 2D batch range")
    selected = tuple(sorted(
        (sample for sample in build.manifest.samples if sample.split is split),
        key=lambda item: item.sample_id,
    )[:max_samples])
    if not selected:
        raise NativePolyV2DatasetError(f"native V2 build has no {split.value} samples")
    return selected


def materialize_native_poly_v2_samples(
    build: object,
    dataset_root: str | Path,
    split: DatasetSplit,
    *,
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
    max_samples: int = MAX_POLY_2D_TRAINING_BATCH,
) -> tuple[NativePolyV2MaterializedSample, ...]:
    # Reject TEST before touching build/root/artifact surfaces.
    if split is DatasetSplit.TEST:
        raise NativePolyV2DatasetError("TEST remains sealed in TR-POLY-09A")
    if not isinstance(build, NativePolyV2DatasetBuild):
        raise TypeError("build must be NativePolyV2DatasetBuild")
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
    materialized: list[NativePolyV2MaterializedSample] = []
    for sample in selected:
        target_path = root / "targets" / f"{sample.target_sha256}.json"
        image_path = root / "images" / f"{sample.image_sha256}.png"
        for path, label in ((target_path, "target"), (image_path, "image")):
            if path.is_symlink() or not path.is_file():
                raise NativePolyV2DatasetError(f"selected native V2 {label} artifact is missing or symlinked")
        target_bytes = target_path.read_bytes()
        if _sha256_bytes(target_bytes) != sample.target_sha256:
            raise NativePolyV2DatasetError("selected native V2 target hash mismatch")
        try:
            score = parse_canonical_polyphonic_json(target_bytes)
            target = validate_roundtrip(score)
        except Exception as exc:
            raise NativePolyV2DatasetError("selected target failed canonical V2 lossless roundtrip") from exc
        if score.canonical_sha256() != sample.representation_sha256:
            raise NativePolyV2DatasetError("selected target representation SHA-256 mismatch")
        if len(target.token_ids) != sample.target_token_count:
            raise NativePolyV2DatasetError("selected target token count differs from manifest")
        if profile_polyphonic_score(score) != sample.profile:
            raise NativePolyV2DatasetError("selected target polyphony profile differs from manifest")
        decoder_length = len(target.token_ids) - 1
        if decoder_length < 1 or decoder_length > model_config.max_target_tokens:
            raise NativePolyV2DatasetError(
                "V2 target exceeds model max_target_tokens; semantic truncation is forbidden"
            )
        image_bytes = image_path.read_bytes()
        if _sha256_bytes(image_bytes) != sample.image_sha256:
            raise NativePolyV2DatasetError("selected native V2 image hash mismatch")
        try:
            image = preprocess_grayscale_png(
                image_bytes,
                preprocess,
                expected_width=sample.width,
                expected_height=sample.height,
            )
        except TrainingDataError as exc:
            raise NativePolyV2DatasetError("selected native V2 PNG failed deterministic preprocessing") from exc
        materialized.append(
            NativePolyV2MaterializedSample(
                sample_id=sample.sample_id,
                family_id=sample.family_id,
                split=sample.split,
                image_sha256=sample.image_sha256,
                target_sha256=sample.target_sha256,
                representation_sha256=sample.representation_sha256,
                target=target,
                image=image,
                source_width=sample.width,
                source_height=sample.height,
                profile=sample.profile,
            )
        )
    return tuple(materialized)


def make_native_poly_2d_training_batch(
    samples: object,
    *,
    dataset_manifest_sha256: str,
) -> Poly2DTrainingBatch:
    _require_sha256("dataset_manifest_sha256", dataset_manifest_sha256)
    if not isinstance(samples, tuple) or not samples or len(samples) > MAX_POLY_2D_TRAINING_BATCH:
        raise NativePolyV2DatasetError("samples must be a bounded non-empty immutable tuple")
    if any(not isinstance(sample, NativePolyV2MaterializedSample) for sample in samples):
        raise NativePolyV2DatasetError("samples must be NativePolyV2MaterializedSample values")
    split = samples[0].split
    if split is DatasetSplit.TEST or any(sample.split is not split for sample in samples):
        raise NativePolyV2DatasetError("one native V2 batch may not mix splits or contain TEST")
    sequences = tuple(sample.target.token_ids for sample in samples)
    if any(sequence[0] != BOS_TOKEN_ID or sequence[-1] != EOS_TOKEN_ID for sequence in sequences):
        raise NativePolyV2DatasetError("native V2 target lacks BOS/EOS")
    lengths = tuple(len(sequence) - 1 for sequence in sequences)
    maximum = max(lengths)
    decoder = torch.full((len(samples), maximum), PAD_TOKEN_ID, dtype=torch.long)
    labels = torch.full((len(samples), maximum), PAD_TOKEN_ID, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        length = len(sequence) - 1
        decoder[row, :length] = torch.tensor(sequence[:-1], dtype=torch.long)
        labels[row, :length] = torch.tensor(sequence[1:], dtype=torch.long)
    return Poly2DTrainingBatch(
        images=torch.stack(tuple(sample.image for sample in samples), dim=0),
        decoder_input_ids=decoder,
        labels=labels,
        split=split,
        sample_ids=tuple(sample.sample_id for sample in samples),
        dataset_manifest_sha256=dataset_manifest_sha256,
    )


@dataclass(frozen=True, slots=True)
class NativePolyV2TrainingExecutionResult:
    checkpoint: Poly2DCheckpointReceipt
    dataset_manifest_sha256: str
    materialization_fingerprint_sha256: str
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    source_class: str = NATIVE_POLY_V2_SOURCE_CLASS
    target_profile: str = NATIVE_POLY_V2_TARGET_PROFILE
    native_polyphonic_dataset_verified: bool = True
    training_entry_verified: bool = True
    test_split_accessed: bool = False
    benchmark_evidence: bool = False
    production_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, Poly2DCheckpointReceipt):
            raise NativePolyV2DatasetError("native V2 execution checkpoint is invalid")
        if self.source_class != NATIVE_POLY_V2_SOURCE_CLASS or self.target_profile != NATIVE_POLY_V2_TARGET_PROFILE:
            raise NativePolyV2DatasetError("native V2 execution source/target profile mismatch")
        if not self.native_polyphonic_dataset_verified or not self.training_entry_verified:
            raise NativePolyV2DatasetError("native V2 execution must record verified dataset/training entry")
        if self.test_split_accessed or self.benchmark_evidence or self.production_authority:
            raise NativePolyV2DatasetError("TR-POLY-09A may not claim TEST, benchmark, or production authority")


def execute_native_poly_v2_training(
    *,
    build: NativePolyV2DatasetBuild,
    dataset_root: str | Path,
    repository_sha: str,
    output_directory: Path,
    training_config: Poly2DTrainingConfig = FROZEN_POLY_2D_TRAINING_CONFIG,
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
    max_train_samples: int = MAX_POLY_2D_TRAINING_BATCH,
    max_validation_samples: int = MAX_POLY_2D_TRAINING_BATCH,
) -> NativePolyV2TrainingExecutionResult:
    train_samples = materialize_native_poly_v2_samples(
        build, dataset_root, DatasetSplit.TRAIN, model_config=model_config, max_samples=max_train_samples
    )
    validation_samples = materialize_native_poly_v2_samples(
        build, dataset_root, DatasetSplit.VALIDATION, model_config=model_config, max_samples=max_validation_samples
    )
    train_families = {sample.family_id for sample in train_samples}
    validation_families = {sample.family_id for sample in validation_samples}
    if train_families & validation_families:
        raise NativePolyV2DatasetError("TRAIN/VALIDATION family leakage detected at execution boundary")
    train_batch = make_native_poly_2d_training_batch(
        train_samples, dataset_manifest_sha256=build.manifest_sha256
    )
    validation_batch = make_native_poly_2d_training_batch(
        validation_samples, dataset_manifest_sha256=build.manifest_sha256
    )
    materialization_sha = native_poly_v2_materialization_fingerprint(model_config)
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
        raise NativePolyV2DatasetError("checkpoint dataset identity differs from native V2 build")
    if loaded.metadata.preprocess_fingerprint_sha256 != materialization_sha:
        raise NativePolyV2DatasetError("checkpoint materialization identity differs from native V2 path")
    if (
        loaded.metadata.authoritative_dataset_execution
        or loaded.metadata.test_split_accessed
        or loaded.metadata.benchmark_evidence
        or loaded.metadata.production_authority
    ):
        raise NativePolyV2DatasetError("native V2 checkpoint exceeded the TR-POLY-09A claim boundary")
    return NativePolyV2TrainingExecutionResult(
        checkpoint=checkpoint,
        dataset_manifest_sha256=build.manifest_sha256,
        materialization_fingerprint_sha256=materialization_sha,
        train_sample_ids=tuple(sample.sample_id for sample in train_samples),
        validation_sample_ids=tuple(sample.sample_id for sample in validation_samples),
    )
