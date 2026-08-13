"""Independent Stage 5-A dataset manifest model and validator.

The Stage 5-A boundary does not build datasets or write artifact files. It
defines immutable metadata for training-eligible synthetic samples, independently
recomputes Stage 4 lineage identities, enforces family-exclusive splits, and
provides deterministic canonical manifest serialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
import re
import struct
import zlib
from typing import Final


DATASET_MANIFEST_SCHEMA_VERSION: Final[str] = "st-dataset-manifest-v1"
DATASET_SPLIT_POLICY: Final[str] = "family-exclusive-v1"
DATASET_SOURCE_CLASS: Final[str] = "synthetic"
SUPPORTED_DEGRADATION_VERSION: Final[str] = "st-controlled-degradation-v1"
SUPPORTED_CAIROSVG_VERSION: Final[str] = "2.8.2"
SUPPORTED_PILLOW_VERSION: Final[str] = "12.3.0"
MAX_DATASET_SAMPLES: Final[int] = 1_000_000
MAX_PAGE_NUMBER: Final[int] = 64
MAX_IMAGE_PIXELS: Final[int] = 16_000_000
MAX_PNG_BYTES: Final[int] = 64 * 1024 * 1024

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class DatasetManifestInputError(ValueError):
    """Raised when a Stage 5-A manifest boundary input fails closed."""


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class DatasetValidationIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class DatasetValidationResult:
    issues: tuple[DatasetValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class DatasetDegradationConfig:
    """Independent Stage 5-A mirror of the frozen Stage 4 V1 replay fields."""

    seed: int
    raster_width: int
    rotation_mdeg: int
    blur_milli: int
    noise_level: int
    brightness_milli: int
    contrast_milli: int
    jpeg_quality: int

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise DatasetManifestInputError(f"{name} must be an integer")
        if not 0 <= self.seed <= (2**63 - 1):
            raise DatasetManifestInputError("seed is outside the Stage 4 V1 range")
        if not 512 <= self.raster_width <= 2400:
            raise DatasetManifestInputError("raster_width is outside the Stage 4 V1 range")
        if not -3000 <= self.rotation_mdeg <= 3000:
            raise DatasetManifestInputError("rotation_mdeg is outside the Stage 4 V1 range")
        if not 0 <= self.blur_milli <= 2000:
            raise DatasetManifestInputError("blur_milli is outside the Stage 4 V1 range")
        if not 0 <= self.noise_level <= 20:
            raise DatasetManifestInputError("noise_level is outside the Stage 4 V1 range")
        if not 800 <= self.brightness_milli <= 1200:
            raise DatasetManifestInputError("brightness_milli is outside the Stage 4 V1 range")
        if not 750 <= self.contrast_milli <= 1250:
            raise DatasetManifestInputError("contrast_milli is outside the Stage 4 V1 range")
        if self.jpeg_quality != 0 and not 65 <= self.jpeg_quality <= 95:
            raise DatasetManifestInputError("jpeg_quality must be 0 or inside the Stage 4 V1 range")


@dataclass(frozen=True, slots=True)
class DatasetSample:
    """Immutable metadata for one synthetic PNG derivative and its symbolic target."""

    sample_id: str
    family_id: str
    split: DatasetSplit
    page_number: int
    source_musicxml_sha256: str
    renderer_config_fingerprint: str
    source_svg_sha256: str
    clean_raster_sha256: str
    degradation_config_fingerprint: str
    degradation_config: DatasetDegradationConfig
    derivative_id: str
    png_sha256: str
    degradation_version: str
    cairosvg_version: str
    pillow_version: str
    cairo_runtime_version: str
    python_version: str
    platform_system: str
    platform_machine: str
    clean_width: int
    clean_height: int
    width: int
    height: int
    mode: str = "L"
    image_format: str = "png"

    def __post_init__(self) -> None:
        _require_identifier("family_id", self.family_id)
        for name in (
            "sample_id",
            "source_musicxml_sha256",
            "renderer_config_fingerprint",
            "source_svg_sha256",
            "clean_raster_sha256",
            "degradation_config_fingerprint",
            "derivative_id",
            "png_sha256",
        ):
            _require_hex64(name, getattr(self, name))
        if not isinstance(self.split, DatasetSplit):
            raise DatasetManifestInputError("split must be a DatasetSplit")
        if not isinstance(self.page_number, int) or isinstance(self.page_number, bool):
            raise DatasetManifestInputError("page_number must be an integer")
        if not 1 <= self.page_number <= MAX_PAGE_NUMBER:
            raise DatasetManifestInputError("page_number is outside the Stage 5-A range")
        if not isinstance(self.degradation_config, DatasetDegradationConfig):
            raise DatasetManifestInputError("degradation_config must be DatasetDegradationConfig")
        _require_text("degradation_version", self.degradation_version)
        _require_text("cairosvg_version", self.cairosvg_version)
        _require_text("pillow_version", self.pillow_version)
        _require_text("cairo_runtime_version", self.cairo_runtime_version)
        _require_text("python_version", self.python_version)
        _require_text("platform_system", self.platform_system)
        _require_text("platform_machine", self.platform_machine)
        for name in ("clean_width", "clean_height", "width", "height"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise DatasetManifestInputError(f"{name} must be a positive integer")
        if self.clean_width * self.clean_height > MAX_IMAGE_PIXELS:
            raise DatasetManifestInputError("clean raster exceeds the Stage 5-A pixel limit")
        if self.width * self.height > MAX_IMAGE_PIXELS:
            raise DatasetManifestInputError("final raster exceeds the Stage 5-A pixel limit")
        if self.mode != "L":
            raise DatasetManifestInputError("Stage 5-A V1 requires grayscale mode L")
        if self.image_format != "png":
            raise DatasetManifestInputError("Stage 5-A V1 requires PNG artifacts")


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Immutable synthetic dataset manifest metadata.

    The tuple order is intentionally not semantic. Canonical serialization sorts
    samples deterministically, so logically identical manifests hash identically.
    """

    dataset_name: str
    dataset_version: str
    samples: tuple[DatasetSample, ...]
    schema_version: str = DATASET_MANIFEST_SCHEMA_VERSION
    source_class: str = DATASET_SOURCE_CLASS
    split_policy: str = DATASET_SPLIT_POLICY

    def __post_init__(self) -> None:
        _require_identifier("dataset_name", self.dataset_name)
        if not isinstance(self.dataset_version, str) or _VERSION_RE.fullmatch(self.dataset_version) is None:
            raise DatasetManifestInputError("dataset_version must match the bounded version contract")
        if not isinstance(self.samples, tuple):
            raise DatasetManifestInputError("samples must be an immutable tuple")
        if len(self.samples) > MAX_DATASET_SAMPLES:
            raise DatasetManifestInputError("manifest exceeds the Stage 5-A sample-count limit")
        if self.schema_version != DATASET_MANIFEST_SCHEMA_VERSION:
            raise DatasetManifestInputError("unsupported dataset manifest schema version")
        if self.source_class != DATASET_SOURCE_CLASS:
            raise DatasetManifestInputError("Stage 5-A V1 accepts synthetic source_class only")
        if self.split_policy != DATASET_SPLIT_POLICY:
            raise DatasetManifestInputError("unsupported dataset split policy")


def _require_hex64(name: str, value: object) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise DatasetManifestInputError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise DatasetManifestInputError(f"{name} must match the bounded identifier contract")
    return value


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise DatasetManifestInputError(f"{name} must be non-empty bounded text")
    return value


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def dataset_degradation_config_fingerprint(config: DatasetDegradationConfig) -> str:
    """Independently recompute the Stage 4 V1 degradation configuration fingerprint."""

    if not isinstance(config, DatasetDegradationConfig):
        raise TypeError("config must be DatasetDegradationConfig")
    payload = {
        "degradation_version": SUPPORTED_DEGRADATION_VERSION,
        "cairosvg_version": SUPPORTED_CAIROSVG_VERSION,
        "pillow_version": SUPPORTED_PILLOW_VERSION,
        "config": asdict(config),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def dataset_derivative_id(
    *,
    family_id: str,
    page_number: int,
    source_musicxml_sha256: str,
    renderer_config_fingerprint: str,
    source_svg_sha256: str,
    clean_raster_sha256: str,
    degradation_config_fingerprint: str,
    png_sha256: str,
    degradation_version: str = SUPPORTED_DEGRADATION_VERSION,
) -> str:
    """Independently recompute the frozen Stage 4 V1 derivative identity."""

    payload = {
        "degradation_version": degradation_version,
        "family_id": family_id,
        "page_number": page_number,
        "source_musicxml_sha256": source_musicxml_sha256,
        "renderer_config_fingerprint": renderer_config_fingerprint,
        "source_svg_sha256": source_svg_sha256,
        "clean_raster_sha256": clean_raster_sha256,
        "degradation_config_fingerprint": degradation_config_fingerprint,
        "png_sha256": png_sha256,
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def dataset_sample_id(
    *,
    family_id: str,
    page_number: int,
    source_musicxml_sha256: str,
    renderer_config_fingerprint: str,
    source_svg_sha256: str,
    clean_raster_sha256: str,
    degradation_config_fingerprint: str,
    derivative_id: str,
    png_sha256: str,
) -> str:
    """Compute sample identity independent of train/validation/test assignment."""

    payload = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "family_id": family_id,
        "page_number": page_number,
        "source_musicxml_sha256": source_musicxml_sha256,
        "renderer_config_fingerprint": renderer_config_fingerprint,
        "source_svg_sha256": source_svg_sha256,
        "clean_raster_sha256": clean_raster_sha256,
        "degradation_config_fingerprint": degradation_config_fingerprint,
        "derivative_id": derivative_id,
        "png_sha256": png_sha256,
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _issue(code: str, path: str, message: str) -> DatasetValidationIssue:
    return DatasetValidationIssue(code=code, path=path, message=message)


def _sorted_result(issues: list[DatasetValidationIssue]) -> DatasetValidationResult:
    return DatasetValidationResult(tuple(sorted(issues, key=lambda item: (item.code, item.path, item.message))))


def validate_dataset_sample(sample: object, *, path: str = "sample") -> DatasetValidationResult:
    """Independently validate one Stage 5-A sample, including Stage 4 lineage hashes."""

    issues: list[DatasetValidationIssue] = []
    if not isinstance(sample, DatasetSample):
        return _sorted_result([_issue("sample.type", path, "sample must be DatasetSample")])

    hex_fields = (
        "sample_id",
        "source_musicxml_sha256",
        "renderer_config_fingerprint",
        "source_svg_sha256",
        "clean_raster_sha256",
        "degradation_config_fingerprint",
        "derivative_id",
        "png_sha256",
    )
    for name in hex_fields:
        value = getattr(sample, name, None)
        if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
            issues.append(_issue("sample.hash", f"{path}.{name}", "must be lowercase SHA-256 hex"))

    if not isinstance(sample.family_id, str) or _ID_RE.fullmatch(sample.family_id) is None:
        issues.append(_issue("sample.family_id", f"{path}.family_id", "invalid family identifier"))
    if not isinstance(sample.split, DatasetSplit):
        issues.append(_issue("sample.split", f"{path}.split", "split must be DatasetSplit"))
    if not isinstance(sample.page_number, int) or isinstance(sample.page_number, bool) or not 1 <= sample.page_number <= MAX_PAGE_NUMBER:
        issues.append(_issue("sample.page_number", f"{path}.page_number", "page number is outside the supported range"))
    if not isinstance(sample.degradation_config, DatasetDegradationConfig):
        issues.append(_issue("sample.config", f"{path}.degradation_config", "invalid degradation config type"))

    if sample.degradation_version != SUPPORTED_DEGRADATION_VERSION:
        issues.append(_issue("sample.degradation_version", f"{path}.degradation_version", "unsupported Stage 4 producer version"))
    if sample.cairosvg_version != SUPPORTED_CAIROSVG_VERSION:
        issues.append(_issue("sample.cairosvg_version", f"{path}.cairosvg_version", "unsupported CairoSVG version"))
    if sample.pillow_version != SUPPORTED_PILLOW_VERSION:
        issues.append(_issue("sample.pillow_version", f"{path}.pillow_version", "unsupported Pillow version"))
    for name in ("cairo_runtime_version", "python_version", "platform_system", "platform_machine"):
        value = getattr(sample, name, None)
        if not isinstance(value, str) or not value or len(value) > 128:
            issues.append(_issue("sample.runtime_text", f"{path}.{name}", "runtime provenance must be bounded non-empty text"))

    for name in ("clean_width", "clean_height", "width", "height"):
        value = getattr(sample, name, None)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            issues.append(_issue("sample.dimension", f"{path}.{name}", "dimension must be a positive integer"))
    if all(isinstance(getattr(sample, name, None), int) and not isinstance(getattr(sample, name, None), bool) for name in ("clean_width", "clean_height")):
        if sample.clean_width > 0 and sample.clean_height > 0 and sample.clean_width * sample.clean_height > MAX_IMAGE_PIXELS:
            issues.append(_issue("sample.pixel_limit", f"{path}.clean", "clean raster exceeds pixel limit"))
    if all(isinstance(getattr(sample, name, None), int) and not isinstance(getattr(sample, name, None), bool) for name in ("width", "height")):
        if sample.width > 0 and sample.height > 0 and sample.width * sample.height > MAX_IMAGE_PIXELS:
            issues.append(_issue("sample.pixel_limit", f"{path}.final", "final raster exceeds pixel limit"))
    if sample.mode != "L":
        issues.append(_issue("sample.mode", f"{path}.mode", "Stage 5-A V1 requires grayscale mode L"))
    if sample.image_format != "png":
        issues.append(_issue("sample.image_format", f"{path}.image_format", "Stage 5-A V1 requires PNG"))

    if isinstance(sample.degradation_config, DatasetDegradationConfig):
        try:
            expected_config_hash = dataset_degradation_config_fingerprint(sample.degradation_config)
        except Exception:
            expected_config_hash = None
        if expected_config_hash is None or sample.degradation_config_fingerprint != expected_config_hash:
            issues.append(_issue("lineage.config_fingerprint", f"{path}.degradation_config_fingerprint", "does not match exact replay parameters"))

    identity_fields_valid = all(
        isinstance(getattr(sample, name, None), str) and _HEX64_RE.fullmatch(getattr(sample, name)) is not None
        for name in (
            "source_musicxml_sha256",
            "renderer_config_fingerprint",
            "source_svg_sha256",
            "clean_raster_sha256",
            "degradation_config_fingerprint",
            "derivative_id",
            "png_sha256",
        )
    ) and isinstance(sample.family_id, str) and _ID_RE.fullmatch(sample.family_id) is not None and isinstance(sample.page_number, int) and not isinstance(sample.page_number, bool)

    if identity_fields_valid:
        expected_derivative = dataset_derivative_id(
            family_id=sample.family_id,
            page_number=sample.page_number,
            source_musicxml_sha256=sample.source_musicxml_sha256,
            renderer_config_fingerprint=sample.renderer_config_fingerprint,
            source_svg_sha256=sample.source_svg_sha256,
            clean_raster_sha256=sample.clean_raster_sha256,
            degradation_config_fingerprint=sample.degradation_config_fingerprint,
            png_sha256=sample.png_sha256,
            degradation_version=sample.degradation_version,
        )
        if sample.derivative_id != expected_derivative:
            issues.append(_issue("lineage.derivative_id", f"{path}.derivative_id", "does not match independently recomputed Stage 4 lineage"))
        expected_sample = dataset_sample_id(
            family_id=sample.family_id,
            page_number=sample.page_number,
            source_musicxml_sha256=sample.source_musicxml_sha256,
            renderer_config_fingerprint=sample.renderer_config_fingerprint,
            source_svg_sha256=sample.source_svg_sha256,
            clean_raster_sha256=sample.clean_raster_sha256,
            degradation_config_fingerprint=sample.degradation_config_fingerprint,
            derivative_id=sample.derivative_id,
            png_sha256=sample.png_sha256,
        )
        if sample.sample_id != expected_sample:
            issues.append(_issue("lineage.sample_id", f"{path}.sample_id", "does not match independently recomputed sample identity"))

    return _sorted_result(issues)


def validate_dataset_manifest(manifest: object) -> DatasetValidationResult:
    """Validate the V1 synthetic manifest and veto family/data leakage."""

    issues: list[DatasetValidationIssue] = []
    if not isinstance(manifest, DatasetManifest):
        return _sorted_result([_issue("manifest.type", "manifest", "manifest must be DatasetManifest")])

    if not isinstance(manifest.dataset_name, str) or _ID_RE.fullmatch(manifest.dataset_name) is None:
        issues.append(_issue("manifest.dataset_name", "manifest.dataset_name", "invalid dataset name"))
    if not isinstance(manifest.dataset_version, str) or _VERSION_RE.fullmatch(manifest.dataset_version) is None:
        issues.append(_issue("manifest.dataset_version", "manifest.dataset_version", "invalid dataset version"))
    if manifest.schema_version != DATASET_MANIFEST_SCHEMA_VERSION:
        issues.append(_issue("manifest.schema_version", "manifest.schema_version", "unsupported schema version"))
    if manifest.source_class != DATASET_SOURCE_CLASS:
        issues.append(_issue("manifest.source_class", "manifest.source_class", "Stage 5-A V1 is synthetic-only"))
    if manifest.split_policy != DATASET_SPLIT_POLICY:
        issues.append(_issue("manifest.split_policy", "manifest.split_policy", "unsupported split policy"))
    if not isinstance(manifest.samples, tuple):
        issues.append(_issue("manifest.samples_type", "manifest.samples", "samples must be an immutable tuple"))
        return _sorted_result(issues)
    if not manifest.samples:
        issues.append(_issue("manifest.empty", "manifest.samples", "training-eligible manifest must contain samples"))
    if len(manifest.samples) > MAX_DATASET_SAMPLES:
        issues.append(_issue("manifest.sample_limit", "manifest.samples", "manifest exceeds sample-count limit"))

    seen_sample_ids: dict[str, int] = {}
    seen_derivative_ids: dict[str, int] = {}
    seen_png_hashes: dict[str, int] = {}
    family_split: dict[str, DatasetSplit] = {}
    target_family: dict[str, str] = {}
    target_split: dict[str, DatasetSplit] = {}
    svg_family: dict[str, str] = {}
    svg_split: dict[str, DatasetSplit] = {}
    split_counts = {split: 0 for split in DatasetSplit}

    for index, sample in enumerate(manifest.samples):
        path = f"manifest.samples[{index}]"
        sample_result = validate_dataset_sample(sample, path=path)
        issues.extend(sample_result.issues)
        if not isinstance(sample, DatasetSample):
            continue
        if isinstance(sample.split, DatasetSplit):
            split_counts[sample.split] += 1

        for code, value, seen in (
            ("duplicate.sample_id", sample.sample_id, seen_sample_ids),
            ("duplicate.derivative_id", sample.derivative_id, seen_derivative_ids),
            ("duplicate.png_sha256", sample.png_sha256, seen_png_hashes),
        ):
            if isinstance(value, str) and value in seen:
                issues.append(_issue(code, path, f"duplicates sample at index {seen[value]}"))
            elif isinstance(value, str):
                seen[value] = index

        if isinstance(sample.family_id, str) and isinstance(sample.split, DatasetSplit):
            prior_split = family_split.get(sample.family_id)
            if prior_split is not None and prior_split != sample.split:
                issues.append(_issue("leakage.family_split", path, "one family_id appears in multiple splits"))
            else:
                family_split.setdefault(sample.family_id, sample.split)

        if isinstance(sample.source_musicxml_sha256, str) and isinstance(sample.family_id, str) and isinstance(sample.split, DatasetSplit):
            prior_family = target_family.get(sample.source_musicxml_sha256)
            prior_split = target_split.get(sample.source_musicxml_sha256)
            if prior_family is not None and prior_family != sample.family_id:
                issues.append(_issue("leakage.target_family", path, "identical MusicXML target appears under multiple family_id values"))
            if prior_split is not None and prior_split != sample.split:
                issues.append(_issue("leakage.target_split", path, "identical MusicXML target appears in multiple splits"))
            target_family.setdefault(sample.source_musicxml_sha256, sample.family_id)
            target_split.setdefault(sample.source_musicxml_sha256, sample.split)

        if isinstance(sample.source_svg_sha256, str) and isinstance(sample.family_id, str) and isinstance(sample.split, DatasetSplit):
            prior_family = svg_family.get(sample.source_svg_sha256)
            prior_split = svg_split.get(sample.source_svg_sha256)
            if prior_family is not None and prior_family != sample.family_id:
                issues.append(_issue("leakage.svg_family", path, "identical clean SVG appears under multiple family_id values"))
            if prior_split is not None and prior_split != sample.split:
                issues.append(_issue("leakage.svg_split", path, "identical clean SVG appears in multiple splits"))
            svg_family.setdefault(sample.source_svg_sha256, sample.family_id)
            svg_split.setdefault(sample.source_svg_sha256, sample.split)

    for split, count in split_counts.items():
        if count == 0:
            issues.append(_issue("manifest.missing_split", f"manifest.split.{split.value}", "training-eligible V1 manifest requires every split"))

    return _sorted_result(issues)


def _sample_payload(sample: DatasetSample) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "family_id": sample.family_id,
        "split": sample.split.value,
        "page_number": sample.page_number,
        "source_musicxml_sha256": sample.source_musicxml_sha256,
        "renderer_config_fingerprint": sample.renderer_config_fingerprint,
        "source_svg_sha256": sample.source_svg_sha256,
        "clean_raster_sha256": sample.clean_raster_sha256,
        "degradation_config_fingerprint": sample.degradation_config_fingerprint,
        "degradation_config": asdict(sample.degradation_config),
        "derivative_id": sample.derivative_id,
        "png_sha256": sample.png_sha256,
        "degradation_version": sample.degradation_version,
        "cairosvg_version": sample.cairosvg_version,
        "pillow_version": sample.pillow_version,
        "cairo_runtime_version": sample.cairo_runtime_version,
        "python_version": sample.python_version,
        "platform_system": sample.platform_system,
        "platform_machine": sample.platform_machine,
        "clean_width": sample.clean_width,
        "clean_height": sample.clean_height,
        "width": sample.width,
        "height": sample.height,
        "mode": sample.mode,
        "image_format": sample.image_format,
    }


def canonical_manifest_bytes(manifest: DatasetManifest) -> bytes:
    """Serialize a valid manifest to deterministic canonical JSON bytes."""

    result = validate_dataset_manifest(manifest)
    if not result.is_valid:
        first = result.issues[0]
        raise DatasetManifestInputError(f"manifest is invalid: {first.code} at {first.path}: {first.message}")
    samples = sorted(
        manifest.samples,
        key=lambda sample: (sample.family_id, sample.page_number, sample.sample_id, sample.split.value),
    )
    payload = {
        "schema_version": manifest.schema_version,
        "source_class": manifest.source_class,
        "split_policy": manifest.split_policy,
        "dataset_name": manifest.dataset_name,
        "dataset_version": manifest.dataset_version,
        "samples": [_sample_payload(sample) for sample in samples],
    }
    return _canonical_json_bytes(payload)


def dataset_manifest_sha256(manifest: DatasetManifest) -> str:
    return sha256(canonical_manifest_bytes(manifest)).hexdigest()


def _inspect_grayscale_png(png: object) -> tuple[int, int]:
    """Independently verify enough PNG structure for the Stage 5-A bridge.

    This intentionally uses only the standard library. It validates the PNG
    signature, the first IHDR chunk and CRC, and the V1 grayscale/8-bit/non-
    interlaced shape. Full artifact storage verification remains a Stage 6 duty.
    """

    if not isinstance(png, bytes) or not png:
        raise DatasetManifestInputError("degraded page PNG must be non-empty bytes")
    if len(png) > MAX_PNG_BYTES:
        raise DatasetManifestInputError("degraded page PNG exceeds the Stage 5-A byte limit")
    if not png.startswith(_PNG_SIGNATURE) or len(png) < 33:
        raise DatasetManifestInputError("artifact is not a valid PNG header")
    length = struct.unpack(">I", png[8:12])[0]
    chunk_type = png[12:16]
    if length != 13 or chunk_type != b"IHDR":
        raise DatasetManifestInputError("PNG must begin with the canonical IHDR chunk")
    ihdr = png[16:29]
    expected_crc = struct.unpack(">I", png[29:33])[0]
    actual_crc = zlib.crc32(chunk_type + ihdr) & 0xFFFFFFFF
    if expected_crc != actual_crc:
        raise DatasetManifestInputError("PNG IHDR CRC mismatch")
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", ihdr)
    if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
        raise DatasetManifestInputError("PNG dimensions are outside Stage 5-A bounds")
    if (bit_depth, color_type, compression, filtering, interlace) != (8, 0, 0, 0, 0):
        raise DatasetManifestInputError("Stage 5-A V1 requires 8-bit non-interlaced grayscale PNG")
    return width, height


def sample_from_degraded_page(page: object, *, split: DatasetSplit) -> DatasetSample:
    """Bridge one Stage 4 DegradedPage-like object into Stage 5-A metadata.

    The bridge independently verifies artifact hash/header, replay parameters,
    producer versions, config fingerprint and derivative identity. It does not
    write the PNG or assign storage paths.
    """

    if not isinstance(split, DatasetSplit):
        raise DatasetManifestInputError("split must be DatasetSplit")
    required = (
        "family_id",
        "page_number",
        "source_musicxml_sha256",
        "renderer_config_fingerprint",
        "source_svg_sha256",
        "clean_raster_sha256",
        "degradation_config_fingerprint",
        "config",
        "derivative_id",
        "png_sha256",
        "degradation_version",
        "cairosvg_version",
        "pillow_version",
        "cairo_runtime_version",
        "python_version",
        "platform_system",
        "platform_machine",
        "clean_width",
        "clean_height",
        "width",
        "height",
        "mode",
        "png",
    )
    try:
        values = {name: getattr(page, name) for name in required}
    except AttributeError as exc:
        raise DatasetManifestInputError("degraded page is missing required Stage 4 provenance") from exc

    png_width, png_height = _inspect_grayscale_png(values["png"])
    if sha256(values["png"]).hexdigest() != values["png_sha256"]:
        raise DatasetManifestInputError("png_sha256 does not match degraded page bytes")
    if png_width != values["width"] or png_height != values["height"]:
        raise DatasetManifestInputError("PNG dimensions do not match Stage 4 metadata")
    if values["mode"] != "L":
        raise DatasetManifestInputError("Stage 4 page mode must be L")

    try:
        source_config = values["config"]
        config = DatasetDegradationConfig(
            seed=source_config.seed,
            raster_width=source_config.raster_width,
            rotation_mdeg=source_config.rotation_mdeg,
            blur_milli=source_config.blur_milli,
            noise_level=source_config.noise_level,
            brightness_milli=source_config.brightness_milli,
            contrast_milli=source_config.contrast_milli,
            jpeg_quality=source_config.jpeg_quality,
        )
    except (AttributeError, TypeError, ValueError, DatasetManifestInputError) as exc:
        raise DatasetManifestInputError("degraded page config does not match Stage 4 V1 replay fields") from exc

    computed_sample_id = dataset_sample_id(
        family_id=values["family_id"],
        page_number=values["page_number"],
        source_musicxml_sha256=values["source_musicxml_sha256"],
        renderer_config_fingerprint=values["renderer_config_fingerprint"],
        source_svg_sha256=values["source_svg_sha256"],
        clean_raster_sha256=values["clean_raster_sha256"],
        degradation_config_fingerprint=values["degradation_config_fingerprint"],
        derivative_id=values["derivative_id"],
        png_sha256=values["png_sha256"],
    )
    sample = DatasetSample(
        sample_id=computed_sample_id,
        family_id=values["family_id"],
        split=split,
        page_number=values["page_number"],
        source_musicxml_sha256=values["source_musicxml_sha256"],
        renderer_config_fingerprint=values["renderer_config_fingerprint"],
        source_svg_sha256=values["source_svg_sha256"],
        clean_raster_sha256=values["clean_raster_sha256"],
        degradation_config_fingerprint=values["degradation_config_fingerprint"],
        degradation_config=config,
        derivative_id=values["derivative_id"],
        png_sha256=values["png_sha256"],
        degradation_version=values["degradation_version"],
        cairosvg_version=values["cairosvg_version"],
        pillow_version=values["pillow_version"],
        cairo_runtime_version=values["cairo_runtime_version"],
        python_version=values["python_version"],
        platform_system=values["platform_system"],
        platform_machine=values["platform_machine"],
        clean_width=values["clean_width"],
        clean_height=values["clean_height"],
        width=values["width"],
        height=values["height"],
        mode=values["mode"],
        image_format="png",
    )
    result = validate_dataset_sample(sample)
    if not result.is_valid:
        first = result.issues[0]
        raise DatasetManifestInputError(f"degraded page lineage rejected: {first.code}: {first.message}")
    return sample
