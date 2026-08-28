"""TR-POLY-04 deterministic external benchmark harness.

This module provides a common, hash-bound adapter surface for external OMR
benchmarks without vendoring dataset bytes or third-party evaluator code. It
supports a strict registry mode and an explicit research-override mode. The
research override is deliberately non-commercial evidence: it can unblock
scientific comparison while preserving the TR-POLY-03 rights record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from typing import Final, Iterable

from .external_dataset_registry import (
    DataUseClass,
    ExternalDatasetRecord,
    RegistryState,
)
from .poly_evaluation_contract import BenchmarkIdentity


EXTERNAL_BENCHMARK_HARNESS_VERSION: Final[str] = "st-omr-external-benchmark-harness-v1"
_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")


class ExternalBenchmarkHarnessError(ValueError):
    """Raised when external benchmark evidence violates the harness contract."""


class BenchmarkDatasetKind(str, Enum):
    OLIMPIC_SYNTHETIC = "olimpic-synthetic"
    OLIMPIC_SCANNED = "olimpic-scanned"
    GRANDSTAFF_LMX = "grandstaff-lmx"
    MUSE_OMR_BENCHMARK = "muse-omr-benchmark"


class AdmissionMode(str, Enum):
    STRICT_REGISTRY = "STRICT_REGISTRY"
    RESEARCH_OVERRIDE = "RESEARCH_OVERRIDE"


class TargetFormat(str, Enum):
    LMX = "LMX"
    MUSICXML = "MUSICXML"
    MUSESCORE = "MUSESCORE"


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExternalBenchmarkHarnessError(f"{name} must be non-empty text")
    return value


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ExternalBenchmarkHarnessError(f"{name} must be lowercase SHA-256 text")
    return value


def _require_safe_relative_path(name: str, value: object) -> str:
    text = _require_text(name, value).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ExternalBenchmarkHarnessError(f"{name} must be a safe relative path")
    if text.startswith("/") or text.endswith("/"):
        raise ExternalBenchmarkHarnessError(f"{name} must point to a file")
    return text


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class ExternalBenchmarkSpec:
    kind: BenchmarkDatasetKind
    benchmark_id: str
    benchmark_version: str
    dataset_name: str
    dataset_component: str
    target_format: TargetFormat
    declared_splits: tuple[str, ...]
    system_level: bool

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BenchmarkDatasetKind):
            raise ExternalBenchmarkHarnessError("kind must be BenchmarkDatasetKind")
        _require_text("benchmark_id", self.benchmark_id)
        _require_text("benchmark_version", self.benchmark_version)
        _require_text("dataset_name", self.dataset_name)
        _require_text("dataset_component", self.dataset_component)
        if not isinstance(self.target_format, TargetFormat):
            raise ExternalBenchmarkHarnessError("target_format must be TargetFormat")
        if (
            not isinstance(self.declared_splits, tuple)
            or not self.declared_splits
            or any(split not in {"train", "validation", "test"} for split in self.declared_splits)
            or len(set(self.declared_splits)) != len(self.declared_splits)
        ):
            raise ExternalBenchmarkHarnessError("declared_splits must be unique train/validation/test values")
        if not isinstance(self.system_level, bool):
            raise ExternalBenchmarkHarnessError("system_level must be boolean")


@dataclass(frozen=True, slots=True)
class BenchmarkManifestRow:
    sample_id: str
    family_id: str
    split: str
    image_relpath: str
    target_relpath: str
    system_id: str

    def __post_init__(self) -> None:
        _require_sha256("sample_id", self.sample_id)
        _require_text("family_id", self.family_id)
        if self.split not in {"train", "validation", "test"}:
            raise ExternalBenchmarkHarnessError("split must be train, validation, or test")
        _require_safe_relative_path("image_relpath", self.image_relpath)
        _require_safe_relative_path("target_relpath", self.target_relpath)
        _require_text("system_id", self.system_id)


@dataclass(frozen=True, slots=True)
class ExternalBenchmarkAdmission:
    spec: ExternalBenchmarkSpec
    admission_mode: AdmissionMode
    registry_record_sha256: str
    data_artifact_sha256: str
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    research_override_reference: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ExternalBenchmarkSpec):
            raise ExternalBenchmarkHarnessError("spec must be ExternalBenchmarkSpec")
        if not isinstance(self.admission_mode, AdmissionMode):
            raise ExternalBenchmarkHarnessError("admission_mode must be AdmissionMode")
        _require_sha256("registry_record_sha256", self.registry_record_sha256)
        _require_sha256("data_artifact_sha256", self.data_artifact_sha256)
        _require_sha256("dataset_manifest_sha256", self.dataset_manifest_sha256)
        _require_sha256("split_manifest_sha256", self.split_manifest_sha256)
        if self.admission_mode is AdmissionMode.RESEARCH_OVERRIDE:
            _require_text("research_override_reference", self.research_override_reference)
        elif self.research_override_reference is not None:
            raise ExternalBenchmarkHarnessError(
                "research_override_reference is allowed only in RESEARCH_OVERRIDE mode"
            )

    @property
    def commercial_evidence_eligible(self) -> bool:
        return self.admission_mode is AdmissionMode.STRICT_REGISTRY

    def benchmark_identity(self) -> BenchmarkIdentity:
        return BenchmarkIdentity(
            benchmark_id=self.spec.benchmark_id,
            benchmark_version=self.spec.benchmark_version,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            split_manifest_sha256=self.split_manifest_sha256,
        )

    def canonical_sha256(self) -> str:
        payload = asdict(self)
        payload["spec"] = asdict(self.spec)
        return sha256(_canonical_json_bytes(payload)).hexdigest()


BENCHMARK_SPECS: Final[tuple[ExternalBenchmarkSpec, ...]] = (
    ExternalBenchmarkSpec(
        kind=BenchmarkDatasetKind.OLIMPIC_SYNTHETIC,
        benchmark_id="olimpic-synthetic-system-omr-v1",
        benchmark_version="OLiMPiC-1.0",
        dataset_name="OLiMPiC",
        dataset_component="synthetic 1.0",
        target_format=TargetFormat.LMX,
        declared_splits=("train", "validation", "test"),
        system_level=True,
    ),
    ExternalBenchmarkSpec(
        kind=BenchmarkDatasetKind.OLIMPIC_SCANNED,
        benchmark_id="olimpic-scanned-system-omr-v1",
        benchmark_version="OLiMPiC-1.0",
        dataset_name="OLiMPiC",
        dataset_component="scanned 1.0",
        target_format=TargetFormat.LMX,
        declared_splits=("validation", "test"),
        system_level=True,
    ),
    ExternalBenchmarkSpec(
        kind=BenchmarkDatasetKind.GRANDSTAFF_LMX,
        benchmark_id="grandstaff-lmx-system-omr-v1",
        benchmark_version="GrandStaff-LMX-2024-extension",
        dataset_name="GrandStaff-LMX",
        dataset_component="added .lmx and .musicxml annotations only",
        target_format=TargetFormat.LMX,
        declared_splits=("train", "validation", "test"),
        system_level=True,
    ),
    ExternalBenchmarkSpec(
        kind=BenchmarkDatasetKind.MUSE_OMR_BENCHMARK,
        benchmark_id="muse-omr-benchmark-score-v1",
        benchmark_version="1077-pair-public-benchmark",
        dataset_name="Muse OMR Benchmark",
        dataset_component="1077 symbolic-score + augmented-PDF pairs",
        target_format=TargetFormat.MUSESCORE,
        declared_splits=("test",),
        system_level=False,
    ),
)


def benchmark_spec(kind: BenchmarkDatasetKind) -> ExternalBenchmarkSpec:
    if not isinstance(kind, BenchmarkDatasetKind):
        raise ExternalBenchmarkHarnessError("kind must be BenchmarkDatasetKind")
    for spec in BENCHMARK_SPECS:
        if spec.kind is kind:
            return spec
    raise ExternalBenchmarkHarnessError(f"no benchmark spec registered for {kind.value}")


def matching_registry_record(
    spec: ExternalBenchmarkSpec,
    records: Iterable[ExternalDatasetRecord],
) -> ExternalDatasetRecord:
    if not isinstance(spec, ExternalBenchmarkSpec):
        raise ExternalBenchmarkHarnessError("spec must be ExternalBenchmarkSpec")
    matches = tuple(
        record
        for record in records
        if isinstance(record, ExternalDatasetRecord)
        and record.dataset_name == spec.dataset_name
        and record.dataset_component == spec.dataset_component
    )
    if len(matches) != 1:
        raise ExternalBenchmarkHarnessError(
            "benchmark spec must match exactly one external dataset registry record"
        )
    return matches[0]


def validate_manifest_rows(
    rows: Iterable[BenchmarkManifestRow],
    spec: ExternalBenchmarkSpec,
) -> tuple[BenchmarkManifestRow, ...]:
    values = tuple(rows)
    if not values:
        raise ExternalBenchmarkHarnessError("benchmark manifest must contain at least one sample")
    if not isinstance(spec, ExternalBenchmarkSpec):
        raise ExternalBenchmarkHarnessError("spec must be ExternalBenchmarkSpec")

    sample_ids: set[str] = set()
    image_paths: set[str] = set()
    target_paths: set[str] = set()
    family_split: dict[str, str] = {}
    for row in values:
        if not isinstance(row, BenchmarkManifestRow):
            raise ExternalBenchmarkHarnessError("manifest rows must be BenchmarkManifestRow values")
        if row.split not in spec.declared_splits:
            raise ExternalBenchmarkHarnessError(
                f"split {row.split!r} is not declared for {spec.kind.value}"
            )
        if row.sample_id in sample_ids:
            raise ExternalBenchmarkHarnessError("duplicate sample_id in benchmark manifest")
        if row.image_relpath in image_paths:
            raise ExternalBenchmarkHarnessError("duplicate image path in benchmark manifest")
        if row.target_relpath in target_paths:
            raise ExternalBenchmarkHarnessError("duplicate target path in benchmark manifest")
        previous_split = family_split.setdefault(row.family_id, row.split)
        if previous_split != row.split:
            raise ExternalBenchmarkHarnessError("family leakage across benchmark splits")
        sample_ids.add(row.sample_id)
        image_paths.add(row.image_relpath)
        target_paths.add(row.target_relpath)

    return tuple(
        sorted(
            values,
            key=lambda row: (row.split, row.family_id, row.system_id, row.sample_id),
        )
    )


def manifest_sha256(
    rows: Iterable[BenchmarkManifestRow],
    spec: ExternalBenchmarkSpec,
) -> str:
    values = validate_manifest_rows(rows, spec)
    payload = {
        "harness_version": EXTERNAL_BENCHMARK_HARNESS_VERSION,
        "spec": asdict(spec),
        "rows": [asdict(row) for row in values],
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def split_manifest_sha256(
    rows: Iterable[BenchmarkManifestRow],
    spec: ExternalBenchmarkSpec,
) -> str:
    values = validate_manifest_rows(rows, spec)
    payload = {
        "harness_version": EXTERNAL_BENCHMARK_HARNESS_VERSION,
        "benchmark_id": spec.benchmark_id,
        "assignments": [
            {
                "sample_id": row.sample_id,
                "family_id": row.family_id,
                "split": row.split,
            }
            for row in values
        ],
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def read_manifest_jsonl(path: str | Path, spec: ExternalBenchmarkSpec) -> tuple[BenchmarkManifestRow, ...]:
    manifest_path = Path(path)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ExternalBenchmarkHarnessError("manifest path must be a regular non-symlink file")
    rows: list[BenchmarkManifestRow] = []
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ExternalBenchmarkHarnessError("unable to read UTF-8 benchmark manifest") from exc
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
            rows.append(BenchmarkManifestRow(**payload))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ExternalBenchmarkHarnessError(
                f"invalid benchmark manifest row at line {line_number}"
            ) from exc
    return validate_manifest_rows(rows, spec)


def directory_tree_sha256(root: str | Path) -> str:
    root_path = Path(root)
    if not root_path.is_dir() or root_path.is_symlink():
        raise ExternalBenchmarkHarnessError("benchmark root must be a regular directory")
    entries: list[dict[str, str | int]] = []
    for path in sorted(root_path.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ExternalBenchmarkHarnessError("benchmark trees must not contain symlinks")
        if not path.is_file():
            continue
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ExternalBenchmarkHarnessError("unable to hash benchmark file") from exc
        entries.append(
            {
                "path": path.relative_to(root_path).as_posix(),
                "size": len(content),
                "sha256": sha256(content).hexdigest(),
            }
        )
    if not entries:
        raise ExternalBenchmarkHarnessError("benchmark root must contain at least one file")
    return sha256(_canonical_json_bytes(entries)).hexdigest()


def create_admission(
    *,
    spec: ExternalBenchmarkSpec,
    registry_record: ExternalDatasetRecord,
    rows: Iterable[BenchmarkManifestRow],
    data_artifact_sha256: str,
    admission_mode: AdmissionMode,
    research_override_reference: str | None = None,
) -> ExternalBenchmarkAdmission:
    if not isinstance(registry_record, ExternalDatasetRecord):
        raise ExternalBenchmarkHarnessError("registry_record must be ExternalDatasetRecord")
    if (
        registry_record.dataset_name != spec.dataset_name
        or registry_record.dataset_component != spec.dataset_component
    ):
        raise ExternalBenchmarkHarnessError("registry record does not match benchmark spec")
    _require_sha256("data_artifact_sha256", data_artifact_sha256)
    values = validate_manifest_rows(rows, spec)

    if admission_mode is AdmissionMode.STRICT_REGISTRY:
        if (
            registry_record.registry_state is not RegistryState.INSTALL_PINNED
            or registry_record.evaluation_allowed is not True
            or registry_record.artifact_sha256 != data_artifact_sha256
        ):
            raise ExternalBenchmarkHarnessError(
                "strict benchmark admission requires matching INSTALL_PINNED evaluation permission"
            )
    elif admission_mode is AdmissionMode.RESEARCH_OVERRIDE:
        _require_text("research_override_reference", research_override_reference)
    else:
        raise ExternalBenchmarkHarnessError("unsupported admission_mode")

    return ExternalBenchmarkAdmission(
        spec=spec,
        admission_mode=admission_mode,
        registry_record_sha256=registry_record.canonical_sha256(),
        data_artifact_sha256=data_artifact_sha256,
        dataset_manifest_sha256=manifest_sha256(values, spec),
        split_manifest_sha256=split_manifest_sha256(values, spec),
        research_override_reference=research_override_reference,
    )


def validate_commercial_evidence(admission: ExternalBenchmarkAdmission) -> None:
    if not isinstance(admission, ExternalBenchmarkAdmission):
        raise ExternalBenchmarkHarnessError("admission must be ExternalBenchmarkAdmission")
    if not admission.commercial_evidence_eligible:
        raise ExternalBenchmarkHarnessError(
            "research-override benchmark evidence is not commercial/production evidence"
        )


REQUIRED_BENCHMARK_KINDS: Final[tuple[BenchmarkDatasetKind, ...]] = tuple(BenchmarkDatasetKind)
