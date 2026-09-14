"""TR-POLY-09B3 deterministic VALIDATION aggregation for Polyphonic V2.

This module aggregates already-produced TR-POLY-09B2 sample metric reports.
It does not run inference, load datasets, open TEST, mutate a model, rank
candidates, or grant production authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Final, Iterable

from .poly_evaluation_contract import (
    REQUIRED_ROBUSTNESS_BUCKETS,
    REQUIRED_VOICE_STRATA,
    required_metric_ids,
)
from .poly_v2_metrics import (
    MetricAvailability,
    POLY_V2_EVENT_ALIGNMENT_VERSION,
    POLY_V2_METRIC_ADAPTER_VERSION,
    POLY_V2_RELATION_METRIC_VERSION,
    PolyV2SampleMetricReport,
)


POLY_V2_VALIDATION_AGGREGATION_VERSION: Final[str] = (
    "st-omr-poly-v2-validation-aggregation-v1"
)
POLY_V2_AGGREGATION_POLICY: Final[str] = "sample-macro-mean-v1"


class PolyV2ValidationAggregationError(ValueError):
    """Raised when B3 aggregation inputs violate the common benchmark contract."""


class ValidationSliceKind(str, Enum):
    OVERALL = "overall"
    VOICE_STRATUM = "voice_stratum"
    ROBUSTNESS_BUCKET = "robustness_bucket"


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PolyV2ValidationAggregationError(
            "B3 evidence is not canonical-JSON serializable"
        ) from exc


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PolyV2ValidationAggregationError(
            f"{name} must be lowercase SHA-256 text"
        )
    return value


@dataclass(frozen=True, slots=True)
class AggregatedMetric:
    metric_id: str
    availability: MetricAvailability
    sample_count: int
    available_count: int
    unsupported_count: int
    mean_value: float | None
    min_value: float | None
    max_value: float | None
    unsupported_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.metric_id not in set(required_metric_ids()):
            raise PolyV2ValidationAggregationError(
                "aggregate metric_id is outside TR-POLY-02"
            )
        if not isinstance(self.availability, MetricAvailability):
            raise PolyV2ValidationAggregationError(
                "aggregate availability must be MetricAvailability"
            )
        for name in ("sample_count", "available_count", "unsupported_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise PolyV2ValidationAggregationError(
                    f"{name} must be a non-negative integer"
                )
        if self.sample_count < 1:
            raise PolyV2ValidationAggregationError(
                "aggregate metric requires at least one sample"
            )
        if self.available_count + self.unsupported_count != self.sample_count:
            raise PolyV2ValidationAggregationError(
                "aggregate metric availability counts do not cover all samples"
            )

        if self.availability is MetricAvailability.AVAILABLE:
            if self.available_count != self.sample_count or self.unsupported_count != 0:
                raise PolyV2ValidationAggregationError(
                    "available aggregate metric must be available for every sample"
                )
            if (
                self.mean_value is None
                or self.min_value is None
                or self.max_value is None
                or self.unsupported_reasons
            ):
                raise PolyV2ValidationAggregationError(
                    "available aggregate metric requires numeric summary and no unsupported reasons"
                )
            for name in ("mean_value", "min_value", "max_value"):
                value = float(getattr(self, name))
                if value != value or value in {float("inf"), float("-inf")}:
                    raise PolyV2ValidationAggregationError(f"{name} must be finite")
            if float(self.min_value) > float(self.mean_value) or float(
                self.mean_value
            ) > float(self.max_value):
                raise PolyV2ValidationAggregationError(
                    "aggregate min/mean/max ordering is invalid"
                )
        else:
            if self.available_count != 0 or self.unsupported_count != self.sample_count:
                raise PolyV2ValidationAggregationError(
                    "unsupported aggregate metric must be unsupported for every sample"
                )
            if any(
                value is not None
                for value in (self.mean_value, self.min_value, self.max_value)
            ):
                raise PolyV2ValidationAggregationError(
                    "unsupported aggregate metric must not contain numeric summary"
                )
            if not self.unsupported_reasons:
                raise PolyV2ValidationAggregationError(
                    "unsupported aggregate metric requires an explicit reason"
                )


@dataclass(frozen=True, slots=True)
class ValidationSliceReport:
    slice_kind: ValidationSliceKind
    slice_value: str
    sample_count: int
    family_count: int
    parse_success_count: int
    metrics: tuple[AggregatedMetric, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.slice_kind, ValidationSliceKind):
            raise PolyV2ValidationAggregationError(
                "slice_kind must be ValidationSliceKind"
            )
        if not isinstance(self.slice_value, str) or not self.slice_value:
            raise PolyV2ValidationAggregationError(
                "slice_value must be non-empty text"
            )
        if not isinstance(self.sample_count, int) or isinstance(
            self.sample_count, bool
        ) or self.sample_count < 1:
            raise PolyV2ValidationAggregationError(
                "slice sample_count must be positive"
            )
        if not isinstance(self.family_count, int) or isinstance(
            self.family_count, bool
        ) or not 1 <= self.family_count <= self.sample_count:
            raise PolyV2ValidationAggregationError(
                "slice family_count is outside the sample boundary"
            )
        if not isinstance(self.parse_success_count, int) or isinstance(
            self.parse_success_count, bool
        ) or not 0 <= self.parse_success_count <= self.sample_count:
            raise PolyV2ValidationAggregationError(
                "slice parse_success_count is outside the sample boundary"
            )
        if (
            not isinstance(self.metrics, tuple)
            or tuple(item.metric_id for item in self.metrics) != required_metric_ids()
        ):
            raise PolyV2ValidationAggregationError(
                "slice metrics must contain every frozen metric in contract order"
            )

    @property
    def parse_success_rate(self) -> float:
        return self.parse_success_count / self.sample_count

    @property
    def unsupported_metric_ids(self) -> tuple[str, ...]:
        return tuple(
            item.metric_id
            for item in self.metrics
            if item.availability is MetricAvailability.UNSUPPORTED
        )

    def metric(self, metric_id: str) -> AggregatedMetric:
        for item in self.metrics:
            if item.metric_id == metric_id:
                return item
        raise KeyError(metric_id)


@dataclass(frozen=True, slots=True)
class PolyV2ValidationBenchmarkReport:
    benchmark_identity_sha256: str
    candidate_identity_sha256: str
    checkpoint_bound: bool
    sample_count: int
    family_count: int
    sample_ids: tuple[str, ...]
    sample_report_fingerprints: tuple[str, ...]
    overall: ValidationSliceReport
    voice_strata: tuple[ValidationSliceReport, ...]
    robustness_buckets: tuple[ValidationSliceReport, ...]
    missing_voice_strata: tuple[str, ...]
    unsupported_metric_ids: tuple[str, ...]
    aggregation_version: str = POLY_V2_VALIDATION_AGGREGATION_VERSION
    aggregation_policy: str = POLY_V2_AGGREGATION_POLICY
    metric_adapter_version: str = POLY_V2_METRIC_ADAPTER_VERSION
    alignment_version: str = POLY_V2_EVENT_ALIGNMENT_VERSION
    relation_metric_version: str = POLY_V2_RELATION_METRIC_VERSION

    def __post_init__(self) -> None:
        _require_sha256("benchmark_identity_sha256", self.benchmark_identity_sha256)
        _require_sha256("candidate_identity_sha256", self.candidate_identity_sha256)
        if not isinstance(self.checkpoint_bound, bool):
            raise PolyV2ValidationAggregationError("checkpoint_bound must be bool")
        if not isinstance(self.sample_count, int) or isinstance(
            self.sample_count, bool
        ) or self.sample_count < 1:
            raise PolyV2ValidationAggregationError(
                "benchmark sample_count must be positive"
            )
        if not isinstance(self.family_count, int) or isinstance(
            self.family_count, bool
        ) or not 1 <= self.family_count <= self.sample_count:
            raise PolyV2ValidationAggregationError(
                "benchmark family_count is outside the sample boundary"
            )
        if (
            not isinstance(self.sample_ids, tuple)
            or len(self.sample_ids) != self.sample_count
            or tuple(sorted(set(self.sample_ids))) != self.sample_ids
        ):
            raise PolyV2ValidationAggregationError(
                "sample_ids must be sorted, unique, and complete"
            )
        for sample_id in self.sample_ids:
            _require_sha256("sample_id", sample_id)
        if (
            not isinstance(self.sample_report_fingerprints, tuple)
            or len(self.sample_report_fingerprints) != self.sample_count
        ):
            raise PolyV2ValidationAggregationError(
                "sample report fingerprints must cover every sample"
            )
        for value in self.sample_report_fingerprints:
            _require_sha256("sample_report_fingerprint", value)
        if not isinstance(self.overall, ValidationSliceReport):
            raise PolyV2ValidationAggregationError("overall must be ValidationSliceReport")
        if self.overall.slice_kind is not ValidationSliceKind.OVERALL:
            raise PolyV2ValidationAggregationError("overall slice kind mismatch")
        if self.overall.sample_count != self.sample_count:
            raise PolyV2ValidationAggregationError(
                "overall sample count differs from benchmark"
            )
        if any(
            not isinstance(item, ValidationSliceReport)
            or item.slice_kind is not ValidationSliceKind.VOICE_STRATUM
            for item in self.voice_strata
        ):
            raise PolyV2ValidationAggregationError(
                "voice_strata contains an invalid slice"
            )
        if any(
            not isinstance(item, ValidationSliceReport)
            or item.slice_kind is not ValidationSliceKind.ROBUSTNESS_BUCKET
            for item in self.robustness_buckets
        ):
            raise PolyV2ValidationAggregationError(
                "robustness_buckets contains an invalid slice"
            )
        if tuple(sorted(item.slice_value for item in self.voice_strata)) != tuple(
            item.slice_value for item in self.voice_strata
        ):
            raise PolyV2ValidationAggregationError(
                "voice strata must be sorted by slice value"
            )
        if tuple(
            sorted(item.slice_value for item in self.robustness_buckets)
        ) != tuple(item.slice_value for item in self.robustness_buckets):
            raise PolyV2ValidationAggregationError(
                "robustness buckets must be sorted by slice value"
            )
        if tuple(sorted(set(self.missing_voice_strata))) != self.missing_voice_strata:
            raise PolyV2ValidationAggregationError(
                "missing_voice_strata must be sorted and unique"
            )
        if tuple(sorted(set(self.unsupported_metric_ids))) != self.unsupported_metric_ids:
            raise PolyV2ValidationAggregationError(
                "unsupported_metric_ids must be sorted and unique"
            )
        if set(self.unsupported_metric_ids) != set(self.overall.unsupported_metric_ids):
            raise PolyV2ValidationAggregationError(
                "benchmark unsupported metric set differs from overall slice"
            )
        if self.aggregation_version != POLY_V2_VALIDATION_AGGREGATION_VERSION:
            raise PolyV2ValidationAggregationError("aggregation version mismatch")
        if self.aggregation_policy != POLY_V2_AGGREGATION_POLICY:
            raise PolyV2ValidationAggregationError("aggregation policy mismatch")
        if self.metric_adapter_version != POLY_V2_METRIC_ADAPTER_VERSION:
            raise PolyV2ValidationAggregationError("metric adapter version mismatch")
        if self.alignment_version != POLY_V2_EVENT_ALIGNMENT_VERSION:
            raise PolyV2ValidationAggregationError("event alignment version mismatch")
        if self.relation_metric_version != POLY_V2_RELATION_METRIC_VERSION:
            raise PolyV2ValidationAggregationError("relation metric version mismatch")

    @property
    def full_metric_contract_ready(self) -> bool:
        return not self.unsupported_metric_ids

    @property
    def voice_coverage_ready(self) -> bool:
        return not self.missing_voice_strata

    @property
    def common_comparison_ready(self) -> bool:
        return self.checkpoint_bound and self.full_metric_contract_ready and self.voice_coverage_ready

    def require_common_comparison_ready(self) -> None:
        problems: list[str] = []
        if not self.checkpoint_bound:
            problems.append("candidate is not checkpoint-bound")
        if self.unsupported_metric_ids:
            problems.append(f"unsupported metrics: {list(self.unsupported_metric_ids)}")
        if self.missing_voice_strata:
            problems.append(f"missing voice strata: {list(self.missing_voice_strata)}")
        if problems:
            raise PolyV2ValidationAggregationError(
                "common comparison gate remains closed: " + "; ".join(problems)
            )

    def fingerprint(self) -> str:
        return sha256(_canonical_json_bytes(_jsonable(asdict(self)))).hexdigest()


def _validate_sample_reports(
    reports: Iterable[PolyV2SampleMetricReport],
) -> tuple[PolyV2SampleMetricReport, ...]:
    values = tuple(reports)
    if not values:
        raise PolyV2ValidationAggregationError(
            "B3 requires at least one B2 VALIDATION report"
        )
    for item in values:
        if not isinstance(item, PolyV2SampleMetricReport):
            raise PolyV2ValidationAggregationError(
                "B3 inputs must be PolyV2SampleMetricReport values"
            )
        if item.split != "validation":
            raise PolyV2ValidationAggregationError(
                "B3 accepts VALIDATION reports only; TEST remains sealed"
            )
        if not item.checkpoint_bound:
            raise PolyV2ValidationAggregationError(
                "B3 common benchmark requires checkpoint-bound inference evidence"
            )
        if item.adapter_version != POLY_V2_METRIC_ADAPTER_VERSION:
            raise PolyV2ValidationAggregationError("B3 metric adapter version mismatch")
        if item.alignment_version != POLY_V2_EVENT_ALIGNMENT_VERSION:
            raise PolyV2ValidationAggregationError("B3 alignment version mismatch")
        if item.relation_metric_version != POLY_V2_RELATION_METRIC_VERSION:
            raise PolyV2ValidationAggregationError("B3 relation metric version mismatch")

    if len({item.benchmark_identity_sha256 for item in values}) != 1:
        raise PolyV2ValidationAggregationError("B3 cannot mix benchmark identities")
    if len({item.candidate_identity_sha256 for item in values}) != 1:
        raise PolyV2ValidationAggregationError(
            "B3 cannot mix candidate/checkpoint identities"
        )
    sample_ids = [item.sample_id for item in values]
    if len(set(sample_ids)) != len(sample_ids):
        raise PolyV2ValidationAggregationError(
            "B3 cannot aggregate duplicate sample_id values"
        )
    return tuple(sorted(values, key=lambda item: item.sample_id))


def _aggregate_metric(
    metric_id: str,
    reports: tuple[PolyV2SampleMetricReport, ...],
) -> AggregatedMetric:
    observations = tuple(item.metric(metric_id) for item in reports)
    availability = {item.availability for item in observations}
    if len(availability) != 1:
        raise PolyV2ValidationAggregationError(
            f"metric {metric_id} availability differs across samples"
        )
    only = observations[0].availability
    if only is MetricAvailability.AVAILABLE:
        values = tuple(float(item.value) for item in observations if item.value is not None)
        if len(values) != len(observations):
            raise PolyV2ValidationAggregationError(
                f"metric {metric_id} lost a numeric sample"
            )
        return AggregatedMetric(
            metric_id=metric_id,
            availability=MetricAvailability.AVAILABLE,
            sample_count=len(observations),
            available_count=len(observations),
            unsupported_count=0,
            mean_value=sum(values) / len(values),
            min_value=min(values),
            max_value=max(values),
            unsupported_reasons=(),
        )

    reasons = tuple(
        sorted(
            {
                str(item.unsupported_reason)
                for item in observations
                if item.unsupported_reason is not None
            }
        )
    )
    if not reasons:
        raise PolyV2ValidationAggregationError(
            f"metric {metric_id} is unsupported without a reason"
        )
    return AggregatedMetric(
        metric_id=metric_id,
        availability=MetricAvailability.UNSUPPORTED,
        sample_count=len(observations),
        available_count=0,
        unsupported_count=len(observations),
        mean_value=None,
        min_value=None,
        max_value=None,
        unsupported_reasons=reasons,
    )


def _build_slice(
    *,
    kind: ValidationSliceKind,
    value: str,
    reports: tuple[PolyV2SampleMetricReport, ...],
) -> ValidationSliceReport:
    if not reports:
        raise PolyV2ValidationAggregationError("cannot build an empty validation slice")
    parse_success_count = sum(
        item.metric("parse_success").availability is MetricAvailability.AVAILABLE
        and float(item.metric("parse_success").value) == 1.0
        for item in reports
    )
    metrics = tuple(
        _aggregate_metric(metric_id, reports) for metric_id in required_metric_ids()
    )
    return ValidationSliceReport(
        slice_kind=kind,
        slice_value=value,
        sample_count=len(reports),
        family_count=len({item.family_id for item in reports}),
        parse_success_count=int(parse_success_count),
        metrics=metrics,
    )


def aggregate_poly_v2_validation_reports(
    reports: Iterable[PolyV2SampleMetricReport],
) -> PolyV2ValidationBenchmarkReport:
    """Aggregate one exact candidate on one exact VALIDATION benchmark identity."""

    ordered = _validate_sample_reports(reports)
    overall = _build_slice(
        kind=ValidationSliceKind.OVERALL,
        value="all_validation",
        reports=ordered,
    )

    voice_values = tuple(item.value for item in REQUIRED_VOICE_STRATA)
    allowed_voice_values = set(voice_values)
    unexpected_voice_values = {item.voice_stratum for item in ordered} - allowed_voice_values
    if unexpected_voice_values:
        raise PolyV2ValidationAggregationError(
            f"unexpected voice strata: {sorted(unexpected_voice_values)}"
        )
    by_voice: list[ValidationSliceReport] = []
    missing_voice: list[str] = []
    for voice_value in voice_values:
        selected = tuple(item for item in ordered if item.voice_stratum == voice_value)
        if selected:
            by_voice.append(
                _build_slice(
                    kind=ValidationSliceKind.VOICE_STRATUM,
                    value=voice_value,
                    reports=selected,
                )
            )
        else:
            missing_voice.append(voice_value)

    robustness_values = tuple(item.value for item in REQUIRED_ROBUSTNESS_BUCKETS)
    allowed_robustness_values = set(robustness_values)
    unexpected_robustness_values = {
        item.robustness_bucket for item in ordered
    } - allowed_robustness_values
    if unexpected_robustness_values:
        raise PolyV2ValidationAggregationError(
            "unexpected robustness buckets: "
            f"{sorted(unexpected_robustness_values)}"
        )
    by_robustness = tuple(
        _build_slice(
            kind=ValidationSliceKind.ROBUSTNESS_BUCKET,
            value=bucket_value,
            reports=tuple(
                item for item in ordered if item.robustness_bucket == bucket_value
            ),
        )
        for bucket_value in robustness_values
        if any(item.robustness_bucket == bucket_value for item in ordered)
    )

    return PolyV2ValidationBenchmarkReport(
        benchmark_identity_sha256=ordered[0].benchmark_identity_sha256,
        candidate_identity_sha256=ordered[0].candidate_identity_sha256,
        checkpoint_bound=True,
        sample_count=len(ordered),
        family_count=len({item.family_id for item in ordered}),
        sample_ids=tuple(item.sample_id for item in ordered),
        sample_report_fingerprints=tuple(item.fingerprint() for item in ordered),
        overall=overall,
        voice_strata=tuple(sorted(by_voice, key=lambda item: item.slice_value)),
        robustness_buckets=tuple(
            sorted(by_robustness, key=lambda item: item.slice_value)
        ),
        missing_voice_strata=tuple(sorted(missing_voice)),
        unsupported_metric_ids=tuple(sorted(overall.unsupported_metric_ids)),
    )


def validate_common_candidate_reports(
    reports: Iterable[PolyV2ValidationBenchmarkReport],
) -> tuple[PolyV2ValidationBenchmarkReport, ...]:
    """Validate comparable candidate reports without ranking or winner selection."""

    values = tuple(reports)
    if len(values) < 2:
        raise PolyV2ValidationAggregationError(
            "candidate comparison requires at least two B3 reports"
        )
    for item in values:
        if not isinstance(item, PolyV2ValidationBenchmarkReport):
            raise PolyV2ValidationAggregationError(
                "candidate comparison inputs must be B3 reports"
            )
        item.require_common_comparison_ready()

    if len({item.benchmark_identity_sha256 for item in values}) != 1:
        raise PolyV2ValidationAggregationError(
            "candidate comparison requires one exact benchmark identity"
        )
    if len({item.sample_ids for item in values}) != 1:
        raise PolyV2ValidationAggregationError(
            "candidate comparison requires the same exact VALIDATION samples"
        )
    candidate_identities = [item.candidate_identity_sha256 for item in values]
    if len(set(candidate_identities)) != len(candidate_identities):
        raise PolyV2ValidationAggregationError(
            "candidate comparison contains duplicate candidate identities"
        )
    return tuple(sorted(values, key=lambda item: item.candidate_identity_sha256))
