"""Fail-closed reload/preflight for persisted native Polyphonic V2 datasets.

TR-POLY-09B8A needs to consume an admitted dataset that may live outside the
repository (for example on a mounted artifact store). TR-POLY-09A already
persists an exact manifest/build/target/image layout, but the original API only
kept the validated ``NativePolyV2DatasetBuild`` object in memory.

This module reconstructs that object from persisted TRAIN/VALIDATION artifacts,
independently revalidates their hashes and semantic metadata, and proves that
sealed TEST artifact bytes are absent. It never reads TEST target/image bytes,
never updates model parameters, and grants no benchmark or production authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Final

from .dataset_manifest import DatasetSplit
from .polyphonic_serialization import parse_canonical_polyphonic_json, validate_roundtrip
from .poly_v2_dataset_materialization import (
    NATIVE_POLY_V2_BUILD_VERSION,
    NATIVE_POLY_V2_SOURCE_CLASS,
    NATIVE_POLY_V2_TARGET_PROFILE,
    NATIVE_POLY_V2_TEST_POLICY,
    NativePolyV2DatasetBuild,
    NativePolyV2DatasetError,
    NativePolyV2ImageArtifact,
    NativePolyV2Manifest,
    NativePolyV2Sample,
    NativePolyV2TargetArtifact,
    NativePolyV2TargetProfile,
    _inspect_png,
    _verify_dataset_root,
    canonical_native_poly_v2_manifest_bytes,
    native_poly_v2_build_metadata_bytes,
    native_poly_v2_manifest_sha256,
    profile_polyphonic_score,
)


POLY_V2_DATASET_RELOAD_VERSION: Final[str] = "st-omr-native-poly-v2-reload-v1"

_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "source_class",
        "target_profile",
        "split_policy",
        "test_policy",
        "tokenizer_fingerprint_sha256",
        "dataset_name",
        "dataset_version",
        "samples",
    }
)
_SAMPLE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "sample_id",
        "family_id",
        "split",
        "target_sha256",
        "representation_sha256",
        "image_sha256",
        "width",
        "height",
        "target_token_count",
        "profile",
    }
)
_PROFILE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "voices",
        "event_kinds",
        "has_simultaneous_independent_voices",
        "has_chord_with_independent_voice_same_onset",
        "tie_count",
        "beam_count",
        "tuplet_count",
    }
)
_BUILD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "builder_version",
        "build_id",
        "manifest_sha256",
        "source_class",
        "target_profile",
        "test_policy",
    }
)


class NativePolyV2DatasetReloadError(NativePolyV2DatasetError):
    """Raised when a persisted native V2 artifact set fails B8A preflight."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _read_regular_file(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise NativePolyV2DatasetReloadError(f"{label} is missing or symlinked")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise NativePolyV2DatasetReloadError(f"{label} could not be read") from exc


def _parse_json_object(data: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativePolyV2DatasetReloadError(f"{label} is not canonical ASCII JSON") from exc
    if not isinstance(value, dict):
        raise NativePolyV2DatasetReloadError(f"{label} must contain a JSON object")
    return value


def _require_exact_keys(value: dict[str, object], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise NativePolyV2DatasetReloadError(
            f"{label} fields differ from the frozen contract: missing={missing}, unknown={unknown}"
        )


def _profile_from_payload(payload: object) -> NativePolyV2TargetProfile:
    if not isinstance(payload, dict):
        raise NativePolyV2DatasetReloadError("native V2 sample profile must be an object")
    _require_exact_keys(payload, _PROFILE_KEYS, "native V2 sample profile")
    voices = payload["voices"]
    event_kinds = payload["event_kinds"]
    if not isinstance(voices, list) or not isinstance(event_kinds, list):
        raise NativePolyV2DatasetReloadError("native V2 profile voices/event_kinds must be arrays")
    try:
        return NativePolyV2TargetProfile(
            voices=tuple(voices),
            event_kinds=tuple(event_kinds),
            has_simultaneous_independent_voices=payload["has_simultaneous_independent_voices"],
            has_chord_with_independent_voice_same_onset=payload[
                "has_chord_with_independent_voice_same_onset"
            ],
            tie_count=payload["tie_count"],
            beam_count=payload["beam_count"],
            tuplet_count=payload["tuplet_count"],
        )
    except (TypeError, ValueError, NativePolyV2DatasetError) as exc:
        raise NativePolyV2DatasetReloadError("native V2 sample profile is invalid") from exc


def _sample_from_payload(payload: object) -> NativePolyV2Sample:
    if not isinstance(payload, dict):
        raise NativePolyV2DatasetReloadError("native V2 manifest sample must be an object")
    _require_exact_keys(payload, _SAMPLE_KEYS, "native V2 manifest sample")
    try:
        split = DatasetSplit(payload["split"])
        return NativePolyV2Sample(
            sample_id=payload["sample_id"],
            family_id=payload["family_id"],
            split=split,
            target_sha256=payload["target_sha256"],
            representation_sha256=payload["representation_sha256"],
            image_sha256=payload["image_sha256"],
            width=payload["width"],
            height=payload["height"],
            target_token_count=payload["target_token_count"],
            profile=_profile_from_payload(payload["profile"]),
        )
    except (TypeError, ValueError, NativePolyV2DatasetError) as exc:
        raise NativePolyV2DatasetReloadError("native V2 manifest sample is invalid") from exc


def _manifest_from_bytes(data: bytes) -> NativePolyV2Manifest:
    payload = _parse_json_object(data, "manifest.json")
    _require_exact_keys(payload, _MANIFEST_KEYS, "manifest.json")
    samples_payload = payload["samples"]
    if not isinstance(samples_payload, list):
        raise NativePolyV2DatasetReloadError("manifest.json samples must be an array")
    try:
        manifest = NativePolyV2Manifest(
            dataset_name=payload["dataset_name"],
            dataset_version=payload["dataset_version"],
            samples=tuple(_sample_from_payload(item) for item in samples_payload),
            schema_version=payload["schema_version"],
            source_class=payload["source_class"],
            target_profile=payload["target_profile"],
            split_policy=payload["split_policy"],
            tokenizer_fingerprint_sha256=payload["tokenizer_fingerprint_sha256"],
            test_policy=payload["test_policy"],
        )
    except (TypeError, ValueError, NativePolyV2DatasetError) as exc:
        raise NativePolyV2DatasetReloadError("manifest.json violates the native V2 contract") from exc
    if canonical_native_poly_v2_manifest_bytes(manifest) != data:
        raise NativePolyV2DatasetReloadError("manifest.json is not the exact canonical native V2 encoding")
    return manifest


def _require_exact_directory_entries(directory: Path, expected: set[str], label: str) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise NativePolyV2DatasetReloadError(f"{label} directory is missing or symlinked")
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        raise NativePolyV2DatasetReloadError(f"{label} directory could not be listed") from exc
    if any(entry.is_symlink() for entry in entries):
        raise NativePolyV2DatasetReloadError(f"{label} directory contains a symlink")
    actual = {entry.name for entry in entries}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise NativePolyV2DatasetReloadError(
            f"{label} artifact set differs from admitted TRAIN/VALIDATION hashes: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if any(not entry.is_file() for entry in entries):
        raise NativePolyV2DatasetReloadError(f"{label} directory contains a non-file entry")


def _verify_semantic_target(data: bytes, sample: NativePolyV2Sample) -> None:
    if sha256(data).hexdigest() != sample.target_sha256:
        raise NativePolyV2DatasetReloadError("native V2 target SHA-256 mismatch")
    try:
        score = parse_canonical_polyphonic_json(data)
        tokenized = validate_roundtrip(score)
    except Exception as exc:
        raise NativePolyV2DatasetReloadError(
            "native V2 target failed canonical parse/roundtrip during reload"
        ) from exc
    if score.canonical_sha256() != sample.representation_sha256:
        raise NativePolyV2DatasetReloadError("native V2 representation SHA-256 mismatch")
    if len(tokenized.token_ids) != sample.target_token_count:
        raise NativePolyV2DatasetReloadError("native V2 target token count differs from manifest")
    if profile_polyphonic_score(score) != sample.profile:
        raise NativePolyV2DatasetReloadError("native V2 target profile differs from manifest")


def _verify_image(data: bytes, sample: NativePolyV2Sample) -> None:
    if sha256(data).hexdigest() != sample.image_sha256:
        raise NativePolyV2DatasetReloadError("native V2 image SHA-256 mismatch")
    try:
        width, height = _inspect_png(data)
    except NativePolyV2DatasetError as exc:
        raise NativePolyV2DatasetReloadError("native V2 image failed PNG validation") from exc
    if (width, height) != (sample.width, sample.height):
        raise NativePolyV2DatasetReloadError("native V2 image dimensions differ from manifest")


@dataclass(frozen=True, slots=True)
class NativePolyV2DatasetPreflightReceipt:
    manifest_sha256: str
    build_id: str
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    sealed_test_sample_ids: tuple[str, ...]
    target_artifact_count: int
    image_artifact_count: int
    reload_version: str = POLY_V2_DATASET_RELOAD_VERSION
    test_artifact_bytes_accessed: bool = False
    production_authority: bool = False

    def __post_init__(self) -> None:
        for name in ("manifest_sha256", "build_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise NativePolyV2DatasetReloadError(f"{name} must be SHA-256 text")
        if not self.train_sample_ids or not self.validation_sample_ids or not self.sealed_test_sample_ids:
            raise NativePolyV2DatasetReloadError("preflight requires TRAIN, VALIDATION and sealed TEST metadata")
        all_ids = self.train_sample_ids + self.validation_sample_ids + self.sealed_test_sample_ids
        if len(set(all_ids)) != len(all_ids):
            raise NativePolyV2DatasetReloadError("preflight sample IDs must remain globally unique")
        if self.target_artifact_count < 1 or self.image_artifact_count < 1:
            raise NativePolyV2DatasetReloadError("preflight requires persisted TRAIN/VALIDATION artifacts")
        if self.reload_version != POLY_V2_DATASET_RELOAD_VERSION:
            raise NativePolyV2DatasetReloadError("preflight reload version mismatch")
        if self.test_artifact_bytes_accessed or self.production_authority:
            raise NativePolyV2DatasetReloadError("preflight may not access TEST bytes or grant production authority")

    def fingerprint(self) -> str:
        payload = {
            "reload_version": self.reload_version,
            "manifest_sha256": self.manifest_sha256,
            "build_id": self.build_id,
            "train_sample_ids": list(self.train_sample_ids),
            "validation_sample_ids": list(self.validation_sample_ids),
            "sealed_test_sample_ids": list(self.sealed_test_sample_ids),
            "target_artifact_count": self.target_artifact_count,
            "image_artifact_count": self.image_artifact_count,
            "test_artifact_bytes_accessed": self.test_artifact_bytes_accessed,
            "production_authority": self.production_authority,
        }
        return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class LoadedNativePolyV2Dataset:
    build: NativePolyV2DatasetBuild
    receipt: NativePolyV2DatasetPreflightReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.build, NativePolyV2DatasetBuild):
            raise NativePolyV2DatasetReloadError("loaded dataset build has the wrong type")
        if not isinstance(self.receipt, NativePolyV2DatasetPreflightReceipt):
            raise NativePolyV2DatasetReloadError("loaded dataset receipt has the wrong type")
        if self.build.manifest_sha256 != self.receipt.manifest_sha256:
            raise NativePolyV2DatasetReloadError("loaded dataset/receipt manifest identity mismatch")
        if self.build.build_id != self.receipt.build_id:
            raise NativePolyV2DatasetReloadError("loaded dataset/receipt build identity mismatch")


def load_and_verify_native_poly_v2_dataset(dataset_root: str | Path) -> LoadedNativePolyV2Dataset:
    """Reload one persisted native V2 build without opening sealed TEST artifacts."""

    if not isinstance(dataset_root, (str, Path)):
        raise TypeError("dataset_root must be str or pathlib.Path")
    root = Path(dataset_root)
    if root.is_symlink() or not root.is_dir():
        raise NativePolyV2DatasetReloadError("native V2 dataset root is missing or symlinked")

    manifest_bytes = _read_regular_file(root / "manifest.json", "manifest.json")
    manifest = _manifest_from_bytes(manifest_bytes)
    manifest_sha = native_poly_v2_manifest_sha256(manifest)
    if sha256(manifest_bytes).hexdigest() != manifest_sha:
        raise NativePolyV2DatasetReloadError("manifest.json SHA-256 differs from canonical identity")

    checksum_bytes = _read_regular_file(root / "manifest.sha256", "manifest.sha256")
    expected_checksum = f"{manifest_sha}  manifest.json\n".encode("ascii")
    if checksum_bytes != expected_checksum:
        raise NativePolyV2DatasetReloadError("manifest.sha256 does not bind the exact manifest.json bytes")

    build_bytes = _read_regular_file(root / "build.json", "build.json")
    build_payload = _parse_json_object(build_bytes, "build.json")
    _require_exact_keys(build_payload, _BUILD_KEYS, "build.json")
    if build_payload["builder_version"] != NATIVE_POLY_V2_BUILD_VERSION:
        raise NativePolyV2DatasetReloadError("build.json builder version mismatch")
    if build_payload["manifest_sha256"] != manifest_sha:
        raise NativePolyV2DatasetReloadError("build.json manifest identity mismatch")
    if build_payload["source_class"] != NATIVE_POLY_V2_SOURCE_CLASS:
        raise NativePolyV2DatasetReloadError("build.json source class mismatch")
    if build_payload["target_profile"] != NATIVE_POLY_V2_TARGET_PROFILE:
        raise NativePolyV2DatasetReloadError("build.json target profile mismatch")
    if build_payload["test_policy"] != NATIVE_POLY_V2_TEST_POLICY:
        raise NativePolyV2DatasetReloadError("build.json TEST policy mismatch")
    build_id = build_payload["build_id"]
    if not isinstance(build_id, str):
        raise NativePolyV2DatasetReloadError("build.json build_id must be text")

    admitted_samples = tuple(
        sample for sample in manifest.samples if sample.split in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}
    )
    sealed_test_samples = tuple(sample for sample in manifest.samples if sample.split is DatasetSplit.TEST)
    expected_target_names = {f"{sample.target_sha256}.json" for sample in admitted_samples}
    expected_image_names = {f"{sample.image_sha256}.png" for sample in admitted_samples}
    _require_exact_directory_entries(root / "targets", expected_target_names, "targets")
    _require_exact_directory_entries(root / "images", expected_image_names, "images")

    # Exact directory-set verification proves sealed TEST target/image hashes are
    # absent. No TEST target/image path is read below.
    target_bytes_by_sha: dict[str, bytes] = {}
    image_bytes_by_sha: dict[str, bytes] = {}
    for sample in admitted_samples:
        target_data = target_bytes_by_sha.get(sample.target_sha256)
        if target_data is None:
            target_data = _read_regular_file(
                root / "targets" / f"{sample.target_sha256}.json",
                "native V2 TRAIN/VALIDATION target",
            )
            target_bytes_by_sha[sample.target_sha256] = target_data
        _verify_semantic_target(target_data, sample)

        image_data = image_bytes_by_sha.get(sample.image_sha256)
        if image_data is None:
            image_data = _read_regular_file(
                root / "images" / f"{sample.image_sha256}.png",
                "native V2 TRAIN/VALIDATION image",
            )
            image_bytes_by_sha[sample.image_sha256] = image_data
        _verify_image(image_data, sample)

    targets = tuple(
        NativePolyV2TargetArtifact(sha256=value, canonical_json=target_bytes_by_sha[value])
        for value in sorted(target_bytes_by_sha)
    )
    images = tuple(
        NativePolyV2ImageArtifact(sha256=value, png=image_bytes_by_sha[value])
        for value in sorted(image_bytes_by_sha)
    )
    try:
        build = NativePolyV2DatasetBuild(
            manifest=manifest,
            manifest_sha256=manifest_sha,
            build_id=build_id,
            targets=targets,
            images=images,
            builder_version=NATIVE_POLY_V2_BUILD_VERSION,
        )
    except (TypeError, ValueError, NativePolyV2DatasetError) as exc:
        raise NativePolyV2DatasetReloadError("persisted native V2 deterministic build identity mismatch") from exc
    if native_poly_v2_build_metadata_bytes(build) != build_bytes:
        raise NativePolyV2DatasetReloadError("build.json is not the exact canonical build metadata encoding")
    try:
        _verify_dataset_root(build, root)
    except NativePolyV2DatasetError as exc:
        raise NativePolyV2DatasetReloadError("persisted native V2 root failed independent verification") from exc

    train_ids = tuple(sorted(sample.sample_id for sample in manifest.samples if sample.split is DatasetSplit.TRAIN))
    validation_ids = tuple(
        sorted(sample.sample_id for sample in manifest.samples if sample.split is DatasetSplit.VALIDATION)
    )
    test_ids = tuple(sorted(sample.sample_id for sample in sealed_test_samples))
    receipt = NativePolyV2DatasetPreflightReceipt(
        manifest_sha256=build.manifest_sha256,
        build_id=build.build_id,
        train_sample_ids=train_ids,
        validation_sample_ids=validation_ids,
        sealed_test_sample_ids=test_ids,
        target_artifact_count=len(targets),
        image_artifact_count=len(images),
    )
    return LoadedNativePolyV2Dataset(build=build, receipt=receipt)
