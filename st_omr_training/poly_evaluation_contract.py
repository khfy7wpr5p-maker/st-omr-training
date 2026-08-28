"""TR-POLY-02 versioned polyphonic OMR evaluation contract.

This module is additive to the frozen Stage 7 validation diagnostics.  It defines
the common taxonomy and benchmark identity surface that future Model A/B/C
experiments must share.  It does not load datasets, open sealed TEST material,
run inference, or implement a model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Final, Iterable


POLY_EVALUATION_CONTRACT_VERSION: Final[str] = "st-omr-poly-evaluation-contract-v1"
POLY_ERROR_TAXONOMY_VERSION: Final[str] = "st-omr-poly-error-taxonomy-v1"


class PolyEvaluationContractError(ValueError):
    """Raised when polyphonic evaluation metadata violates the frozen contract."""


class ErrorClass(str, Enum):
    PITCH = "PITCH"
    DURATION = "DURATION"
    ONSET = "ONSET"
    VOICE = "VOICE"
    STAFF = "STAFF"
    REST = "REST"
    ACCIDENTAL = "ACCIDENTAL"
    TIE = "TIE"
    SLUR = "SLUR"
    TUPLET = "TUPLET"
    BEAM = "BEAM"
    STEM = "STEM"
    CHORD_GROUPING = "CHORD_GROUPING"
    CROSS_STAFF = "CROSS_STAFF"
    METER = "METER"
    MEASURE_BOUNDARY = "MEASURE_BOUNDARY"
    GRACE = "GRACE"
    ORNAMENT = "ORNAMENT"
    OTHER = "OTHER"
    AMBIGUOUS = "AMBIGUOUS"


class MetricGroup(str, Enum):
    SERIALIZATION = "serialization"
    SEQUENCE = "sequence"
    STRUCTURAL = "structural"
    MUSICAL_SEMANTIC = "musical_semantic"
    RELATION = "relation"
    ROBUSTNESS = "robustness"


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class RobustnessBucket(str, Enum):
    CLEAN = "clean"
    SCAN = "scan"
    PHONE = "phone"
    BLUR = "blur"
    PERSPECTIVE = "perspective"
    LOW_CONTRAST = "low_contrast"


class VoiceStratum(str, Enum):
    VOICE_1 = "1_voice"
    VOICE_2 = "2_voice"
    VOICE_3 = "3_voice"
    VOICE_4_PLUS = "4_plus_voice"


@dataclass(frozen=True, slots=True)
class MetricSpec:
    metric_id: str
    group: MetricGroup
    direction: MetricDirection

    def __post_init__(self) -> None:
        if not isinstance(self.metric_id, str) or not self.metric_id:
            raise PolyEvaluationContractError("metric_id must be non-empty text")
        if not isinstance(self.group, MetricGroup):
            raise PolyEvaluationContractError("group must be MetricGroup")
        if not isinstance(self.direction, MetricDirection):
            raise PolyEvaluationContractError("direction must be MetricDirection")


REQUIRED_METRICS: Final[tuple[MetricSpec, ...]] = (
    MetricSpec("parse_success", MetricGroup.SERIALIZATION, MetricDirection.HIGHER_IS_BETTER),
    MetricSpec("musicxml_validity", MetricGroup.SERIALIZATION, MetricDirection.HIGHER_IS_BETTER),
    MetricSpec("ter", MetricGroup.SEQUENCE, MetricDirection.LOWER_IS_BETTER),
    MetricSpec("normalized_edit_distance", MetricGroup.SEQUENCE, MetricDirection.LOWER_IS_BETTER),
    MetricSpec("exact_sequence_accuracy", MetricGroup.SEQUENCE, MetricDirection.HIGHER_IS_BETTER),
    MetricSpec("tedn", MetricGroup.STRUCTURAL, MetricDirection.LOWER_IS_BETTER),
    MetricSpec("pitch_accuracy", MetricGroup.MUSICAL_SEMANTIC, MetricDirection.HIGHER_IS_BETTER),
    MetricSpec("duration_accuracy", MetricGroup.MUSICAL_SEMANTIC, MetricDirection.HIGHER_IS_BETTER),
    MetricSpec("onset_accuracy", MetricGroup.MUSICAL_SEMANTIC, MetricDirection.HIGHER_IS_BETTER),
    MetricSpec("voice_accuracy", MetricGroup.MUSICAL_SEMANTIC, MetricDirection.HIGHER_IS_BETTER),
    MetricSpec("staff_accuracy", MetricGroup.MUSICAL_SEMANTIC, MetricDirection.HIGHER_IS_BETTER),
    MetricSpec("notehead_stem_f1", MetricGroup.RELATION, MetricDirection.HIGHER_IS_BETTER),
    MetricSpec("beam_relation_f1", MetricGroup.RELATION, MetricDirection.HIGHER_IS_BETTER),
    MetricSpec("tie_relation_f1", MetricGroup.RELATION, MetricDirection.HIGHER_IS_BETTER),
    MetricSpec("accidental_note_f1", MetricGroup.RELATION, MetricDirection.HIGHER_IS_BETTER),
    MetricSpec("note_staff_f1", MetricGroup.RELATION, MetricDirection.HIGHER_IS_BETTER),
)


def _finite_unit_interval(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PolyEvaluationContractError(f"{name} must be numeric")
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise PolyEvaluationContractError(f"{name} must be in [0, 1]")
    return numeric


def _require_positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PolyEvaluationContractError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PolyEvaluationContractError(f"{name} must be a non-negative integer")
    return value


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PolyEvaluationContractError(f"{name} must be lowercase SHA-256 text")
    return value


@dataclass(frozen=True, slots=True)
class PolyphonicComplexityProfile:
    voice_count: int
    staff_count: int
    simultaneous_note_density: float
    chord_density: float
    overlap_density: float
    tie_density: float
    beam_complexity: float
    rhythmic_complexity: float
    tuplet_present: bool
    grace_present: bool
    cross_staff_present: bool

    def __post_init__(self) -> None:
        _require_positive_int("voice_count", self.voice_count)
        _require_positive_int("staff_count", self.staff_count)
        _finite_unit_interval("simultaneous_note_density", self.simultaneous_note_density)
        _finite_unit_interval("chord_density", self.chord_density)
        _finite_unit_interval("overlap_density", self.overlap_density)
        _finite_unit_interval("tie_density", self.tie_density)
        _finite_unit_interval("beam_complexity", self.beam_complexity)
        _finite_unit_interval("rhythmic_complexity", self.rhythmic_complexity)
        for name in ("tuplet_present", "grace_present", "cross_staff_present"):
            if not isinstance(getattr(self, name), bool):
                raise PolyEvaluationContractError(f"{name} must be boolean")

    @property
    def voice_stratum(self) -> VoiceStratum:
        if self.voice_count == 1:
            return VoiceStratum.VOICE_1
        if self.voice_count == 2:
            return VoiceStratum.VOICE_2
        if self.voice_count == 3:
            return VoiceStratum.VOICE_3
        return VoiceStratum.VOICE_4_PLUS


@dataclass(frozen=True, slots=True)
class BenchmarkIdentity:
    benchmark_id: str
    benchmark_version: str
    dataset_manifest_sha256: str
    split_manifest_sha256: str
    taxonomy_version: str = POLY_ERROR_TAXONOMY_VERSION
    contract_version: str = POLY_EVALUATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark_id, str) or not self.benchmark_id:
            raise PolyEvaluationContractError("benchmark_id must be non-empty text")
        if not isinstance(self.benchmark_version, str) or not self.benchmark_version:
            raise PolyEvaluationContractError("benchmark_version must be non-empty text")
        _require_sha256("dataset_manifest_sha256", self.dataset_manifest_sha256)
        _require_sha256("split_manifest_sha256", self.split_manifest_sha256)
        if self.taxonomy_version != POLY_ERROR_TAXONOMY_VERSION:
            raise PolyEvaluationContractError("unsupported taxonomy_version")
        if self.contract_version != POLY_EVALUATION_CONTRACT_VERSION:
            raise PolyEvaluationContractError("unsupported contract_version")

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class BenchmarkSampleDescriptor:
    sample_id: str
    family_id: str
    split: str
    complexity: PolyphonicComplexityProfile
    robustness_bucket: RobustnessBucket

    def __post_init__(self) -> None:
        _require_sha256("sample_id", self.sample_id)
        if not isinstance(self.family_id, str) or not self.family_id:
            raise PolyEvaluationContractError("family_id must be non-empty text")
        if self.split not in {"train", "validation", "test"}:
            raise PolyEvaluationContractError("split must be train, validation, or test")
        if not isinstance(self.complexity, PolyphonicComplexityProfile):
            raise PolyEvaluationContractError("complexity must be PolyphonicComplexityProfile")
        if not isinstance(self.robustness_bucket, RobustnessBucket):
            raise PolyEvaluationContractError("robustness_bucket must be RobustnessBucket")


@dataclass(frozen=True, slots=True)
class ErrorCount:
    error_class: ErrorClass
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.error_class, ErrorClass):
            raise PolyEvaluationContractError("error_class must be ErrorClass")
        _require_nonnegative_int("count", self.count)


def required_metric_ids() -> tuple[str, ...]:
    return tuple(spec.metric_id for spec in REQUIRED_METRICS)


def validate_error_counts(error_counts: Iterable[ErrorCount]) -> tuple[ErrorCount, ...]:
    values = tuple(error_counts)
    seen: set[ErrorClass] = set()
    for item in values:
        if not isinstance(item, ErrorCount):
            raise PolyEvaluationContractError("error_counts must contain ErrorCount values")
        if item.error_class in seen:
            raise PolyEvaluationContractError("duplicate error class")
        seen.add(item.error_class)
    return tuple(sorted(values, key=lambda item: item.error_class.value))


def validate_required_metric_result(metric_values: dict[str, float]) -> None:
    if not isinstance(metric_values, dict):
        raise PolyEvaluationContractError("metric_values must be a dict")
    required = set(required_metric_ids())
    supplied = set(metric_values)
    missing = required - supplied
    unknown = supplied - required
    if missing:
        raise PolyEvaluationContractError(f"missing required metrics: {sorted(missing)}")
    if unknown:
        raise PolyEvaluationContractError(f"unknown metrics: {sorted(unknown)}")
    for key, value in metric_values.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PolyEvaluationContractError(f"{key} must be numeric")
        numeric = float(value)
        if numeric != numeric or numeric in {float("inf"), float("-inf")}:
            raise PolyEvaluationContractError(f"{key} must be finite")
        if key not in {"ter", "normalized_edit_distance", "tedn"} and not 0.0 <= numeric <= 1.0:
            raise PolyEvaluationContractError(f"{key} must be in [0, 1]")
        if key in {"ter", "normalized_edit_distance", "tedn"} and numeric < 0.0:
            raise PolyEvaluationContractError(f"{key} must be non-negative")


def validate_comparison_benchmark(
    benchmark_identities: Iterable[BenchmarkIdentity],
) -> BenchmarkIdentity:
    values = tuple(benchmark_identities)
    if not values:
        raise PolyEvaluationContractError("at least one benchmark identity is required")
    first = values[0]
    for value in values[1:]:
        if value != first:
            raise PolyEvaluationContractError(
                "model-family comparison requires identical benchmark identity"
            )
    return first


REQUIRED_ERROR_CLASSES: Final[tuple[ErrorClass, ...]] = tuple(ErrorClass)
REQUIRED_ROBUSTNESS_BUCKETS: Final[tuple[RobustnessBucket, ...]] = tuple(RobustnessBucket)
REQUIRED_VOICE_STRATA: Final[tuple[VoiceStratum, ...]] = tuple(VoiceStratum)
