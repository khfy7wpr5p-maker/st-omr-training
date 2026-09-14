"""Deterministic metric/adaptor layer for free-running Polyphonic V2 evidence.

TR-POLY-09B2 maps one VALIDATION reference plus one TR-POLY-09B1 prediction to
the frozen TR-POLY-02 metric vocabulary. Metrics that do not yet have an
admitted implementation remain explicitly UNSUPPORTED; this module never
fabricates TEDn or MusicXML validity values just to satisfy the required-metric
shape.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Final, Iterable

from .poly_2d_inference import Poly2DInferenceResult
from .poly_evaluation_contract import (
    BenchmarkIdentity,
    BenchmarkSampleDescriptor,
    PolyEvaluationContractError,
    required_metric_ids,
    validate_required_metric_result,
)
from .polyphonic_representation import (
    DisplayAccidentalV2,
    EventKind,
    NoteAtom,
    PolyEvent,
    PolyMeasure,
    PolyScore,
)
from .polyphonic_serialization import BOS_TOKEN_ID, tokenize_polyphonic_score


POLY_V2_METRIC_ADAPTER_VERSION: Final[str] = "st-omr-poly-v2-metric-adapter-v1"
POLY_V2_EVENT_ALIGNMENT_VERSION: Final[str] = "st-omr-poly-v2-event-alignment-v1"
POLY_V2_RELATION_METRIC_VERSION: Final[str] = "st-omr-poly-v2-relation-metrics-v1"
UNSUPPORTED_MUSICXML_REASON: Final[str] = "v2_musicxml_export_adapter_not_admitted"
UNSUPPORTED_TEDN_REASON: Final[str] = "tedn_implementation_not_admitted"


class PolyV2MetricError(ValueError):
    """Raised when B2 metric evidence violates the deterministic adapter contract."""


class MetricAvailability(str, Enum):
    AVAILABLE = "available"
    UNSUPPORTED = "unsupported"


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
        raise PolyV2MetricError("metric evidence is not canonical-JSON serializable") from exc


def _finite_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PolyV2MetricError(f"{name} must be numeric")
    numeric = float(value)
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        raise PolyV2MetricError(f"{name} must be finite")
    return numeric


def _levenshtein(left: tuple[object, ...], right: tuple[object, ...]) -> int:
    """Deterministic unit-cost Levenshtein distance with O(min(m,n)) memory."""

    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_item != right_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


@dataclass(frozen=True, slots=True)
class MetricObservation:
    metric_id: str
    availability: MetricAvailability
    value: float | None
    unsupported_reason: str | None = None

    def __post_init__(self) -> None:
        if self.metric_id not in set(required_metric_ids()):
            raise PolyV2MetricError("metric_id is outside the frozen TR-POLY-02 metric set")
        if not isinstance(self.availability, MetricAvailability):
            raise PolyV2MetricError("availability must be MetricAvailability")
        if self.availability is MetricAvailability.AVAILABLE:
            if self.value is None or self.unsupported_reason is not None:
                raise PolyV2MetricError("available metric requires a value and no unsupported reason")
            _finite_number(self.value, self.metric_id)
        else:
            if self.value is not None:
                raise PolyV2MetricError("unsupported metric must not contain a numeric value")
            if not isinstance(self.unsupported_reason, str) or not self.unsupported_reason:
                raise PolyV2MetricError("unsupported metric requires an explicit reason")


@dataclass(frozen=True, slots=True)
class PolyV2SampleMetricReport:
    sample_id: str
    benchmark_identity_sha256: str
    inference_evidence_sha256: str
    reference_representation_sha256: str
    prediction_representation_sha256: str | None
    voice_stratum: str
    robustness_bucket: str
    observations: tuple[MetricObservation, ...]
    adapter_version: str = POLY_V2_METRIC_ADAPTER_VERSION
    alignment_version: str = POLY_V2_EVENT_ALIGNMENT_VERSION
    relation_metric_version: str = POLY_V2_RELATION_METRIC_VERSION

    def __post_init__(self) -> None:
        for name in (
            "sample_id",
            "benchmark_identity_sha256",
            "inference_evidence_sha256",
            "reference_representation_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise PolyV2MetricError(f"{name} must be lowercase SHA-256 text")
        if self.prediction_representation_sha256 is not None and (
            len(self.prediction_representation_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.prediction_representation_sha256)
        ):
            raise PolyV2MetricError("prediction_representation_sha256 must be SHA-256 when present")
        if not isinstance(self.voice_stratum, str) or not self.voice_stratum:
            raise PolyV2MetricError("voice_stratum must be non-empty text")
        if not isinstance(self.robustness_bucket, str) or not self.robustness_bucket:
            raise PolyV2MetricError("robustness_bucket must be non-empty text")
        if not isinstance(self.observations, tuple) or any(
            not isinstance(item, MetricObservation) for item in self.observations
        ):
            raise PolyV2MetricError("observations must be an immutable MetricObservation tuple")
        expected = required_metric_ids()
        actual = tuple(item.metric_id for item in self.observations)
        if actual != expected:
            raise PolyV2MetricError("observations must contain every frozen metric exactly once in contract order")
        if self.adapter_version != POLY_V2_METRIC_ADAPTER_VERSION:
            raise PolyV2MetricError("metric adapter version mismatch")
        if self.alignment_version != POLY_V2_EVENT_ALIGNMENT_VERSION:
            raise PolyV2MetricError("event alignment version mismatch")
        if self.relation_metric_version != POLY_V2_RELATION_METRIC_VERSION:
            raise PolyV2MetricError("relation metric version mismatch")

    @property
    def full_contract_ready(self) -> bool:
        return all(item.availability is MetricAvailability.AVAILABLE for item in self.observations)

    @property
    def unsupported_metric_ids(self) -> tuple[str, ...]:
        return tuple(
            item.metric_id
            for item in self.observations
            if item.availability is MetricAvailability.UNSUPPORTED
        )

    def metric(self, metric_id: str) -> MetricObservation:
        for item in self.observations:
            if item.metric_id == metric_id:
                return item
        raise KeyError(metric_id)

    def available_metric_values(self) -> dict[str, float]:
        return {
            item.metric_id: float(item.value)
            for item in self.observations
            if item.availability is MetricAvailability.AVAILABLE and item.value is not None
        }

    def require_complete_required_metrics(self) -> dict[str, float]:
        if not self.full_contract_ready:
            raise PolyV2MetricError(
                "full TR-POLY-02 metric result is unavailable; unsupported metrics: "
                f"{list(self.unsupported_metric_ids)}"
            )
        values = self.available_metric_values()
        try:
            validate_required_metric_result(values)
        except PolyEvaluationContractError as exc:
            raise PolyV2MetricError("metric result violates the frozen TR-POLY-02 contract") from exc
        return values

    def fingerprint(self) -> str:
        payload = asdict(self)
        for item in payload["observations"]:
            item["availability"] = item["availability"].value if isinstance(item["availability"], MetricAvailability) else item["availability"]
        return sha256(_canonical_json_bytes(payload)).hexdigest()


def _pitch_identity(note: NoteAtom) -> tuple[str, int, int]:
    return (note.pitch.step, note.pitch.alter, note.pitch.octave)


def _event_pitch_surface(event: PolyEvent) -> tuple[tuple[str, int, int], ...]:
    return tuple(sorted(_pitch_identity(note) for note in event.noteheads))


def _event_semantic_signature(event: PolyEvent) -> tuple[object, ...]:
    return (
        event.kind.value,
        event.onset.numerator,
        event.onset.denominator,
        event.duration.numerator,
        event.duration.denominator,
        event.voice,
        event.staff,
        event.note_type.value if event.note_type is not None else None,
        _event_pitch_surface(event),
        event.dots,
        event.stem.value if event.stem is not None else None,
        tuple((mark.level, mark.state.value) for mark in event.beams),
        tuple(
            (mark.number, mark.actual_notes, mark.normal_notes, mark.boundary.value)
            for mark in event.tuplets
        ),
        None if event.grace is None else event.grace.slash,
        tuple(
            (
                _pitch_identity(note),
                tuple(tie.value for tie in note.ties),
                note.staff_override,
                note.pitch.display_accidental.value,
            )
            for note in event.noteheads
        ),
    )


def _align_events(
    reference: tuple[PolyEvent, ...],
    predicted: tuple[PolyEvent, ...],
) -> tuple[tuple[PolyEvent | None, PolyEvent | None], ...]:
    """Unit-cost sequence alignment; substitution wins deterministic ties."""

    rows = len(reference) + 1
    cols = len(predicted) + 1
    cost = [[0] * cols for _ in range(rows)]
    for row in range(rows):
        cost[row][0] = row
    for col in range(cols):
        cost[0][col] = col
    for row in range(1, rows):
        for col in range(1, cols):
            substitution = cost[row - 1][col - 1] + (
                _event_semantic_signature(reference[row - 1])
                != _event_semantic_signature(predicted[col - 1])
            )
            deletion = cost[row - 1][col] + 1
            insertion = cost[row][col - 1] + 1
            cost[row][col] = min(substitution, deletion, insertion)

    aligned: list[tuple[PolyEvent | None, PolyEvent | None]] = []
    row = len(reference)
    col = len(predicted)
    while row > 0 or col > 0:
        if row > 0 and col > 0:
            substitution = cost[row - 1][col - 1] + (
                _event_semantic_signature(reference[row - 1])
                != _event_semantic_signature(predicted[col - 1])
            )
            if cost[row][col] == substitution:
                aligned.append((reference[row - 1], predicted[col - 1]))
                row -= 1
                col -= 1
                continue
        if row > 0 and cost[row][col] == cost[row - 1][col] + 1:
            aligned.append((reference[row - 1], None))
            row -= 1
            continue
        if col > 0 and cost[row][col] == cost[row][col - 1] + 1:
            aligned.append((None, predicted[col - 1]))
            col -= 1
            continue
        raise PolyV2MetricError("internal event alignment failure")
    aligned.reverse()
    return tuple(aligned)


def _score_event_alignment(
    reference: PolyScore,
    predicted: PolyScore,
) -> tuple[tuple[PolyEvent | None, PolyEvent | None], ...]:
    aligned: list[tuple[PolyEvent | None, PolyEvent | None]] = []
    part_count = max(len(reference.parts), len(predicted.parts))
    for part_index in range(part_count):
        ref_part = reference.parts[part_index] if part_index < len(reference.parts) else None
        pred_part = predicted.parts[part_index] if part_index < len(predicted.parts) else None
        if ref_part is None:
            for measure in pred_part.measures:  # type: ignore[union-attr]
                aligned.extend((None, event) for event in measure.events)
            continue
        if pred_part is None:
            for measure in ref_part.measures:
                aligned.extend((event, None) for event in measure.events)
            continue
        measure_count = max(len(ref_part.measures), len(pred_part.measures))
        for measure_index in range(measure_count):
            ref_measure: PolyMeasure | None = (
                ref_part.measures[measure_index] if measure_index < len(ref_part.measures) else None
            )
            pred_measure: PolyMeasure | None = (
                pred_part.measures[measure_index] if measure_index < len(pred_part.measures) else None
            )
            if ref_measure is None:
                aligned.extend((None, event) for event in pred_measure.events)  # type: ignore[union-attr]
            elif pred_measure is None:
                aligned.extend((event, None) for event in ref_measure.events)
            else:
                aligned.extend(_align_events(ref_measure.events, pred_measure.events))
    return tuple(aligned)


def _reference_event_count(score: PolyScore) -> int:
    return sum(len(measure.events) for part in score.parts for measure in part.measures)


def _predicted_event_count(score: PolyScore) -> int:
    return _reference_event_count(score)


def _ratio_or_vacuous(correct: int, reference_count: int, predicted_count: int) -> float:
    if reference_count > 0:
        return correct / reference_count
    return 1.0 if predicted_count == 0 else 0.0


def _semantic_metrics(reference: PolyScore, predicted: PolyScore) -> dict[str, float]:
    aligned = _score_event_alignment(reference, predicted)
    ref_events = _reference_event_count(reference)
    pred_events = _predicted_event_count(predicted)
    onset_correct = 0
    duration_correct = 0
    voice_correct = 0
    staff_correct = 0
    pitch_correct = 0
    ref_pitched = 0
    pred_pitched = sum(
        event.kind is not EventKind.REST
        for part in predicted.parts
        for measure in part.measures
        for event in measure.events
    )

    for ref_event, pred_event in aligned:
        if ref_event is None:
            continue
        if ref_event.kind is not EventKind.REST:
            ref_pitched += 1
        if pred_event is None:
            continue
        onset_correct += ref_event.onset == pred_event.onset
        duration_correct += ref_event.duration == pred_event.duration
        voice_correct += ref_event.voice == pred_event.voice
        staff_correct += ref_event.staff == pred_event.staff
        if ref_event.kind is not EventKind.REST and pred_event.kind is not EventKind.REST:
            pitch_correct += _event_pitch_surface(ref_event) == _event_pitch_surface(pred_event)

    return {
        "pitch_accuracy": _ratio_or_vacuous(pitch_correct, ref_pitched, pred_pitched),
        "duration_accuracy": _ratio_or_vacuous(duration_correct, ref_events, pred_events),
        "onset_accuracy": _ratio_or_vacuous(onset_correct, ref_events, pred_events),
        "voice_accuracy": _ratio_or_vacuous(voice_correct, ref_events, pred_events),
        "staff_accuracy": _ratio_or_vacuous(staff_correct, ref_events, pred_events),
    }


def _multiset_match_counts(
    reference_items: Iterable[tuple[object, ...]],
    predicted_items: Iterable[tuple[object, ...]],
) -> tuple[int, int, int]:
    reference_counter = Counter(reference_items)
    predicted_counter = Counter(predicted_items)
    true_positive = sum((reference_counter & predicted_counter).values())
    false_negative = sum(reference_counter.values()) - true_positive
    false_positive = sum(predicted_counter.values()) - true_positive
    return true_positive, false_positive, false_negative


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else (2.0 * true_positive) / denominator


def _notehead_stem_items(event: PolyEvent) -> tuple[tuple[object, ...], ...]:
    if event.kind is EventKind.REST or event.stem is None:
        return ()
    return tuple((_pitch_identity(note), event.stem.value) for note in event.noteheads)


def _beam_items(event: PolyEvent) -> tuple[tuple[object, ...], ...]:
    return tuple((mark.level, mark.state.value) for mark in event.beams)


def _tie_items(event: PolyEvent) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (_pitch_identity(note), tie.value)
        for note in event.noteheads
        for tie in note.ties
    )


def _accidental_items(event: PolyEvent) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (_pitch_identity(note), note.pitch.display_accidental.value)
        for note in event.noteheads
        if note.pitch.display_accidental is not DisplayAccidentalV2.NONE
    )


def _note_staff_items(event: PolyEvent) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (_pitch_identity(note), note.staff_override if note.staff_override is not None else event.staff)
        for note in event.noteheads
    )


def _relation_metrics(reference: PolyScore, predicted: PolyScore) -> dict[str, float]:
    aligned = _score_event_alignment(reference, predicted)
    extractors = {
        "notehead_stem_f1": _notehead_stem_items,
        "beam_relation_f1": _beam_items,
        "tie_relation_f1": _tie_items,
        "accidental_note_f1": _accidental_items,
        "note_staff_f1": _note_staff_items,
    }
    totals = {metric_id: [0, 0, 0] for metric_id in extractors}
    for ref_event, pred_event in aligned:
        for metric_id, extractor in extractors.items():
            ref_items = extractor(ref_event) if ref_event is not None else ()
            pred_items = extractor(pred_event) if pred_event is not None else ()
            tp, fp, fn = _multiset_match_counts(ref_items, pred_items)
            totals[metric_id][0] += tp
            totals[metric_id][1] += fp
            totals[metric_id][2] += fn
    return {
        metric_id: _f1(true_positive, false_positive, false_negative)
        for metric_id, (true_positive, false_positive, false_negative) in totals.items()
    }


def _available(metric_id: str, value: float) -> MetricObservation:
    return MetricObservation(
        metric_id=metric_id,
        availability=MetricAvailability.AVAILABLE,
        value=float(value),
    )


def _unsupported(metric_id: str, reason: str) -> MetricObservation:
    return MetricObservation(
        metric_id=metric_id,
        availability=MetricAvailability.UNSUPPORTED,
        value=None,
        unsupported_reason=reason,
    )


def evaluate_poly_v2_validation_sample(
    *,
    reference: PolyScore,
    prediction: Poly2DInferenceResult,
    benchmark: BenchmarkIdentity,
    descriptor: BenchmarkSampleDescriptor,
) -> PolyV2SampleMetricReport:
    """Score one frozen VALIDATION reference against one free-running B1 result."""

    if not isinstance(reference, PolyScore):
        raise TypeError("reference must be PolyScore")
    if not isinstance(prediction, Poly2DInferenceResult):
        raise TypeError("prediction must be Poly2DInferenceResult")
    if not isinstance(benchmark, BenchmarkIdentity):
        raise TypeError("benchmark must be BenchmarkIdentity")
    if not isinstance(descriptor, BenchmarkSampleDescriptor):
        raise TypeError("descriptor must be BenchmarkSampleDescriptor")
    if descriptor.split != "validation":
        raise PolyV2MetricError("TR-POLY-09B2 accepts VALIDATION descriptors only; TEST remains sealed")

    reference_target = tokenize_polyphonic_score(reference)
    reference_surface = reference_target.token_ids[1:]
    if not prediction.token_ids or prediction.token_ids[0] != BOS_TOKEN_ID:
        raise PolyV2MetricError("B1 prediction must start with BOS")
    predicted_surface = prediction.token_ids[1:]
    edits = _levenshtein(reference_surface, predicted_surface)
    sequence_values = {
        "ter": edits / max(1, len(reference_surface)),
        "normalized_edit_distance": edits / max(1, len(reference_surface), len(predicted_surface)),
        "exact_sequence_accuracy": 1.0 if reference_target.token_ids == prediction.token_ids else 0.0,
    }

    if prediction.semantic_valid:
        if prediction.prediction is None:
            raise PolyV2MetricError("semantic-valid B1 result is missing its PolyScore")
        semantic_values = _semantic_metrics(reference, prediction.prediction)
        relation_values = _relation_metrics(reference, prediction.prediction)
    else:
        semantic_values = {
            "pitch_accuracy": 0.0,
            "duration_accuracy": 0.0,
            "onset_accuracy": 0.0,
            "voice_accuracy": 0.0,
            "staff_accuracy": 0.0,
        }
        relation_values = {
            "notehead_stem_f1": 0.0,
            "beam_relation_f1": 0.0,
            "tie_relation_f1": 0.0,
            "accidental_note_f1": 0.0,
            "note_staff_f1": 0.0,
        }

    available_values: dict[str, float] = {
        "parse_success": 1.0 if prediction.semantic_valid else 0.0,
        **sequence_values,
        **semantic_values,
        **relation_values,
    }
    observations: list[MetricObservation] = []
    for metric_id in required_metric_ids():
        if metric_id == "musicxml_validity":
            observations.append(_unsupported(metric_id, UNSUPPORTED_MUSICXML_REASON))
        elif metric_id == "tedn":
            observations.append(_unsupported(metric_id, UNSUPPORTED_TEDN_REASON))
        else:
            try:
                observations.append(_available(metric_id, available_values[metric_id]))
            except KeyError as exc:
                raise PolyV2MetricError(f"missing B2 implementation for required metric {metric_id}") from exc

    report = PolyV2SampleMetricReport(
        sample_id=descriptor.sample_id,
        benchmark_identity_sha256=benchmark.canonical_sha256(),
        inference_evidence_sha256=prediction.evidence_fingerprint(),
        reference_representation_sha256=reference.canonical_sha256(),
        prediction_representation_sha256=prediction.prediction_sha256,
        voice_stratum=descriptor.complexity.voice_stratum.value,
        robustness_bucket=descriptor.robustness_bucket.value,
        observations=tuple(observations),
    )

    # Available metrics must already satisfy the per-metric numeric domains of
    # TR-POLY-02 even though a full required-result cannot be admitted yet.
    for item in report.observations:
        if item.availability is not MetricAvailability.AVAILABLE:
            continue
        value = float(item.value)
        if item.metric_id in {"ter", "normalized_edit_distance"}:
            if value < 0.0:
                raise PolyV2MetricError(f"{item.metric_id} must be non-negative")
        elif not 0.0 <= value <= 1.0:
            raise PolyV2MetricError(f"{item.metric_id} must be in [0, 1]")
    return report
