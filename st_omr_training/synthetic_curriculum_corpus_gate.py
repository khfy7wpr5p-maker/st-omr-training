"""Stage 7-D1 byte/manifest acceptance for Synthetic Curriculum v1.

This module has no model/trainer dependency. It verifies the frozen transport
archive and persisted Stage 6 corpus bytes, then emits hash-only evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Final

from .synthetic_curriculum_acceptance import (
    EXPECTED_BUILD_ID,
    EXPECTED_CONFIG_FINGERPRINT,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SOURCE_COMMIT,
    EXPECTED_TRANSPORT_SHA256,
)

EXPECTED_ARCHIVE_NAME: Final = "st-omr-synthetic-curriculum-v1-d9320e362f162cd2a.tar.gz"
EXPECTED_ARCHIVE_SIZE_BYTES: Final = 494_006_801
EXPECTED_DATASET_NAME: Final = "st-omr-synthetic-curriculum-v1"
EXPECTED_DATASET_VERSION: Final = "v1"
EXPECTED_BUILDER_VERSION: Final = "st-synthetic-dataset-builder-v1"
EXPECTED_SAMPLE_COUNTS: Final = {"test": 153, "train": 1230, "validation": 153}
EXPECTED_FAMILY_COUNTS: Final = {"test": 51, "train": 410, "validation": 51}
EXPECTED_SAMPLE_COUNT: Final = 1536
EXPECTED_TARGET_COUNT: Final = 512
EXPECTED_IMAGE_COUNT: Final = 1536
MAX_MANIFEST_BYTES: Final = 32 * 1024 * 1024
MAX_BUILD_BYTES: Final = 64 * 1024
HASH_CHUNK_BYTES: Final = 1024 * 1024

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_TOP = {"manifest.json", "manifest.sha256", "build.json", "images", "targets"}
_MANIFEST_KEYS = {"schema_version", "source_class", "split_policy", "dataset_name", "dataset_version", "samples"}
_SAMPLE_KEYS = {
    "sample_id", "family_id", "split", "page_number", "source_musicxml_sha256",
    "renderer_config_fingerprint", "source_svg_sha256", "clean_raster_sha256",
    "degradation_config_fingerprint", "degradation_config", "derivative_id", "png_sha256",
    "degradation_version", "cairosvg_version", "pillow_version", "cairo_runtime_version",
    "python_version", "platform_system", "platform_machine", "clean_width", "clean_height",
    "width", "height", "mode", "image_format",
}
_BUILD_KEYS = {
    "builder_version", "build_id", "config_fingerprint", "manifest_sha256", "sample_count",
    "target_count", "image_count", "sample_split_counts", "family_split_counts", "layout",
}
_LAYOUT = {
    "manifest": "manifest.json",
    "metadata": "build.json",
    "images": "images/<png_sha256>.png",
    "targets": "targets/<source_musicxml_sha256>.musicxml",
}


class SyntheticCurriculumCorpusAcceptanceError(RuntimeError):
    """D1 corpus acceptance failed closed."""


@dataclass(frozen=True, slots=True)
class SyntheticCurriculumCorpusReceipt:
    source_commit: str
    build_id: str
    config_fingerprint: str
    manifest_sha256: str
    transport_sha256: str
    transport_archive: str
    archive_size_bytes: int
    sample_count: int
    target_count: int
    image_count: int
    family_split_counts: dict[str, int]
    sample_split_counts: dict[str, int]
    target_bytes_total: int
    image_bytes_total: int
    artifact_binding_sha256: str


@dataclass(frozen=True, slots=True)
class _CorpusExpectations:
    source_commit: str
    build_id: str
    config_fingerprint: str
    manifest_sha256: str
    transport_sha256: str
    archive_name: str
    archive_size_bytes: int | None
    dataset_name: str
    dataset_version: str
    builder_version: str
    sample_counts: dict[str, int]
    family_counts: dict[str, int]
    sample_count: int
    target_count: int
    image_count: int


_FROZEN = _CorpusExpectations(
    EXPECTED_SOURCE_COMMIT, EXPECTED_BUILD_ID, EXPECTED_CONFIG_FINGERPRINT,
    EXPECTED_MANIFEST_SHA256, EXPECTED_TRANSPORT_SHA256, EXPECTED_ARCHIVE_NAME,
    EXPECTED_ARCHIVE_SIZE_BYTES, EXPECTED_DATASET_NAME, EXPECTED_DATASET_VERSION,
    EXPECTED_BUILDER_VERSION, EXPECTED_SAMPLE_COUNTS, EXPECTED_FAMILY_COUNTS,
    EXPECTED_SAMPLE_COUNT, EXPECTED_TARGET_COUNT, EXPECTED_IMAGE_COUNT,
)


def _fail(message: str) -> None:
    raise SyntheticCurriculumCorpusAcceptanceError(message)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be a regular non-symlink file")


def _directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        _fail(f"{label} must be a regular non-symlink directory")


def _read(path: Path, maximum: int, label: str) -> bytes:
    _regular_file(path, label)
    size = path.stat().st_size
    if not 1 <= size <= maximum:
        _fail(f"{label} byte length is outside the D1 bound")
    return path.read_bytes()


def _json(path: Path, maximum: int, label: str) -> tuple[dict[str, object], bytes]:
    raw = _read(path, maximum, label)
    try:
        payload = json.loads(raw.decode("ascii"), parse_constant=lambda value: _fail(f"non-finite JSON constant: {value}"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SyntheticCurriculumCorpusAcceptanceError(f"{label} is not valid ASCII JSON") from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        _fail(f"{label} is not canonical JSON object bytes")
    return payload, raw


def _hash_file(path: Path) -> tuple[str, int]:
    _regular_file(path, str(path))
    digest, total = sha256(), 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _hex(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        _fail(f"{label} must be lowercase SHA-256 hex")
    return value


def _verify_transport_archive(archive: Path, expected: _CorpusExpectations) -> tuple[str, int]:
    if archive.name != expected.archive_name:
        _fail("transport archive filename mismatch")
    _regular_file(archive, "transport archive")
    if expected.archive_size_bytes is not None and archive.stat().st_size != expected.archive_size_bytes:
        _fail("transport archive byte length mismatch")
    digest, size = _hash_file(archive)
    if digest != expected.transport_sha256:
        _fail("transport archive SHA-256 mismatch")
    return digest, size


def _verify_build(payload: dict[str, object], expected: _CorpusExpectations) -> None:
    if set(payload) != _BUILD_KEYS:
        _fail("build.json keys mismatch")
    values = {
        "builder_version": expected.builder_version,
        "build_id": expected.build_id,
        "config_fingerprint": expected.config_fingerprint,
        "manifest_sha256": expected.manifest_sha256,
        "sample_count": expected.sample_count,
        "target_count": expected.target_count,
        "image_count": expected.image_count,
        "sample_split_counts": expected.sample_counts,
        "family_split_counts": expected.family_counts,
        "layout": _LAYOUT,
    }
    for key, value in values.items():
        if payload.get(key) != value:
            _fail(f"build.json {key} mismatch")


def _verify_manifest(payload: dict[str, object], expected: _CorpusExpectations) -> tuple[set[str], set[str], dict[str, int], dict[str, int]]:
    if set(payload) != _MANIFEST_KEYS:
        _fail("manifest top-level keys mismatch")
    header = {
        "schema_version": "st-dataset-manifest-v1",
        "source_class": "synthetic",
        "split_policy": "family-exclusive-v1",
        "dataset_name": expected.dataset_name,
        "dataset_version": expected.dataset_version,
    }
    for key, value in header.items():
        if payload.get(key) != value:
            _fail(f"manifest {key} mismatch")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != expected.sample_count:
        _fail("manifest sample count mismatch")

    sample_ids: set[str] = set()
    image_hashes: set[str] = set()
    target_hashes: set[str] = set()
    family_split: dict[str, str] = {}
    family_samples: Counter[str] = Counter()
    sample_splits: Counter[str] = Counter()
    target_family: dict[str, str] = {}
    target_split: dict[str, str] = {}

    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or set(sample) != _SAMPLE_KEYS:
            _fail(f"manifest sample[{index}] structure mismatch")
        sample_id = _hex(sample.get("sample_id"), f"sample[{index}].sample_id")
        if sample_id in sample_ids:
            _fail("duplicate sample_id")
        sample_ids.add(sample_id)
        family = sample.get("family_id")
        if not isinstance(family, str) or _ID.fullmatch(family) is None:
            _fail(f"sample[{index}] family_id mismatch")
        split = sample.get("split")
        if not isinstance(split, str) or split not in expected.sample_counts:
            _fail(f"sample[{index}] split mismatch")
        if family in family_split and family_split[family] != split:
            _fail("family appears in multiple splits")
        family_split.setdefault(family, split)
        page = sample.get("page_number")
        if not isinstance(page, int) or isinstance(page, bool) or not 1 <= page <= 64:
            _fail(f"sample[{index}] page number mismatch")
        if sample.get("mode") != "L" or sample.get("image_format") != "png":
            _fail(f"sample[{index}] image surface mismatch")
        png_hash = _hex(sample.get("png_sha256"), f"sample[{index}].png_sha256")
        target_hash = _hex(sample.get("source_musicxml_sha256"), f"sample[{index}].source_musicxml_sha256")
        if png_hash in image_hashes:
            _fail("duplicate png_sha256")
        image_hashes.add(png_hash)
        target_hashes.add(target_hash)
        if target_hash in target_family and target_family[target_hash] != family:
            _fail("MusicXML target crosses family boundary")
        if target_hash in target_split and target_split[target_hash] != split:
            _fail("MusicXML target crosses split boundary")
        target_family.setdefault(target_hash, family)
        target_split.setdefault(target_hash, split)
        family_samples[family] += 1
        sample_splits[split] += 1

    sample_counts = dict(sorted(sample_splits.items()))
    family_counts = dict(sorted(Counter(family_split.values()).items()))
    if sample_counts != expected.sample_counts:
        _fail("manifest sample split counts mismatch")
    if family_counts != expected.family_counts or len(family_split) != sum(expected.family_counts.values()):
        _fail("manifest family split counts mismatch")
    if set(family_samples.values()) != {3}:
        _fail("each family must have exactly three derivatives")
    if len(target_hashes) != expected.target_count or len(image_hashes) != expected.image_count:
        _fail("manifest artifact cardinality mismatch")
    return target_hashes, image_hashes, family_counts, sample_counts


def _verify_artifacts(directory: Path, hashes: set[str], suffix: str, png: bool) -> tuple[int, list[str]]:
    _directory(directory, directory.name)
    entries = list(directory.iterdir())
    if {entry.name for entry in entries} != {f"{digest}{suffix}" for digest in hashes}:
        _fail(f"{directory.name} filenames do not match manifest hashes")
    if len(entries) != len(hashes):
        _fail(f"{directory.name} artifact count mismatch")
    total, rows = 0, []
    for digest in sorted(hashes):
        path = directory / f"{digest}{suffix}"
        actual, size = _hash_file(path)
        if actual != digest or size < 1:
            _fail(f"artifact hash mismatch: {path.name}")
        if png:
            with path.open("rb") as handle:
                if handle.read(len(_PNG_SIGNATURE)) != _PNG_SIGNATURE:
                    _fail(f"PNG signature mismatch: {path.name}")
        total += size
        rows.append(f"{directory.name}:{digest}:{size}")
    return total, rows


def _verify_corpus_directory(root: Path, expected: _CorpusExpectations) -> tuple[dict[str, int], dict[str, int], int, int, str]:
    _directory(root, "corpus root")
    if {entry.name for entry in root.iterdir()} != _TOP:
        _fail("corpus top-level layout mismatch")
    manifest, raw = _json(root / "manifest.json", MAX_MANIFEST_BYTES, "manifest.json")
    if sha256(raw).hexdigest() != expected.manifest_sha256:
        _fail("manifest SHA-256 mismatch")
    checksum = _read(root / "manifest.sha256", 256, "manifest.sha256")
    if checksum != f"{expected.manifest_sha256}  manifest.json\n".encode("ascii"):
        _fail("manifest.sha256 content mismatch")
    build, _raw_build = _json(root / "build.json", MAX_BUILD_BYTES, "build.json")
    _verify_build(build, expected)
    targets, images, family_counts, sample_counts = _verify_manifest(manifest, expected)
    target_bytes, target_rows = _verify_artifacts(root / "targets", targets, ".musicxml", False)
    image_bytes, image_rows = _verify_artifacts(root / "images", images, ".png", True)
    binding = sha256(("\n".join(target_rows + image_rows) + "\n").encode("ascii")).hexdigest()
    return family_counts, sample_counts, target_bytes, image_bytes, binding


def verify_stage7d_corpus(corpus_root: str | Path, transport_archive: str | Path) -> SyntheticCurriculumCorpusReceipt:
    """Verify frozen archive and corpus bytes without returning sample data."""
    if not isinstance(corpus_root, (str, Path)) or not isinstance(transport_archive, (str, Path)):
        raise TypeError("corpus_root and transport_archive must be str or pathlib.Path")
    archive = Path(transport_archive)
    _digest, archive_size = _verify_transport_archive(archive, _FROZEN)
    family, samples, target_bytes, image_bytes, binding = _verify_corpus_directory(Path(corpus_root), _FROZEN)
    return SyntheticCurriculumCorpusReceipt(
        _FROZEN.source_commit, _FROZEN.build_id, _FROZEN.config_fingerprint,
        _FROZEN.manifest_sha256, _FROZEN.transport_sha256, _FROZEN.archive_name,
        archive_size, _FROZEN.sample_count, _FROZEN.target_count, _FROZEN.image_count,
        family, samples, target_bytes, image_bytes, binding,
    )


def canonical_stage7d_corpus_evidence(receipt: SyntheticCurriculumCorpusReceipt) -> bytes:
    if not isinstance(receipt, SyntheticCurriculumCorpusReceipt):
        raise TypeError("receipt must be SyntheticCurriculumCorpusReceipt")
    return _canonical_json({"schema_version": "st-omr-synthetic-corpus-acceptance-v1", **asdict(receipt)}) + b"\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Stage 7-D1 frozen synthetic corpus bytes.")
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--evidence-output", required=True)
    args = parser.parse_args(argv)
    root, archive, output = map(lambda value: Path(value).resolve(), (args.corpus_root, args.archive, args.evidence_output))
    if output == root or root in output.parents:
        _fail("evidence output must stay outside the frozen corpus root")
    if output == archive:
        _fail("evidence output must not replace the transport archive")
    if not output.parent.is_dir():
        _fail("evidence output parent must already exist")
    evidence = canonical_stage7d_corpus_evidence(verify_stage7d_corpus(root, archive))
    try:
        with output.open("xb") as handle:
            handle.write(evidence)
    except FileExistsError as exc:
        raise SyntheticCurriculumCorpusAcceptanceError("evidence output already exists") from exc
    print(evidence.decode("ascii").rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
