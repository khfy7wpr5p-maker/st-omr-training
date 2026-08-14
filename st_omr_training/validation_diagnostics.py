"""Stage 7-D3 validation-only semantic error diagnostics.

This module never trains a model and never accepts TEST samples. It compares
already-decoded supported-V1 predictions with frozen validation targets and
emits deterministic quality diagnostics suitable for deciding what to improve
next.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
import json
from typing import Final

from .core import DisplayAccidental
from .musicxml_roundtrip import (
    SemanticEventProjection,
    SemanticPitchProjection,
    SemanticScoreProjection,
)
from .training_tokens import (
    TokenizationError,
    decode_token_ids,
    detokenize_tokens,
)


STAGE7D3_DIAGNOSTIC_SCHEMA: Final[str] = "st-omr-stage7d3-validation-diagnostics-v1"
_HEX = frozenset("0123456789abcdef")


class ValidationDiagnosticError(ValueError):
    """Raised when a D3 diagnostic input violates the validation-only contract."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_sha(name: str, value: object, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in _HEX for character in value)
    ):
        raise ValidationDiagnosticError(
            f"{name} must be lowercase {length}-character hexadecimal text"
        )
    return value


def _require_token_ids(name: str, value: object) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ValidationDiagnosticError(f"{name} must be a non-empty immutable token-id tuple")
    try:
        decode_token_ids(value)
    except TokenizationError as exc:
        raise ValidationDiagnosticError(f"{name} is outside the frozen tokenizer vocabulary") from exc
    return value


def _levenshtein(left: tuple[object, ...], right: tuple[object, ...]) -> int:
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


def _align_events(
    target: tuple[SemanticEventProjection, ...],
    predicted: tuple[SemanticEventProjection, ...],
) -> tuple[tuple[tuple[SemanticEventProjection | None, SemanticEventProjection | None], ...], int]:
    rows = len(target) + 1
    cols = len(predicted) + 1
    cost = [[0] * cols for _ in range(rows)]
    for index in range(rows):
        cost[index][0] = index
    for index in range(cols):
        cost[0][index] = index

    for row in range(1, rows):
        for col in range(1, cols):
            substitution = cost[row - 1][col - 1] + (target[row - 1] != predicted[col - 1])
            deletion = cost[row - 1][col] + 1
            insertion = cost[row][col - 1] + 1
            cost[row][col] = min(substitution, deletion, insertion)

    aligned: list[tuple[SemanticEventProjection | None, SemanticEventProjection | None]] = []
    row = len(target)
    col = len(predicted)
    while row > 0 or col > 0:
        if row > 0 and col > 0:
            substitution = cost[row - 1][col - 1] + (target[row - 1] != predicted[col - 1])
            if cost[row][col] == substitution:
                aligned.append((target[row - 1], predicted[col - 1]))
                row -= 1
                col -= 1
                continue
        if row > 0 and cost[row][col] == cost[row - 1][col] + 1:
            aligned.append((target[row - 1], None))
            row -= 1
            continue
        if col > 0 and cost[row][col] == cost[row][col - 1] + 1:
            aligned.append((None, predicted[col - 1]))
            col -= 1
            continue
        raise ValidationDiagnosticError("internal event alignment failure")
    aligned.reverse()
    return tuple(aligned), cost[-1][-1]


def _pitch_identity(pitch: SemanticPitchProjection) -> tuple[str, int, int]:
    return (pitch.step, pitch.alter, pitch.octave)


def _pitch_identities(event: SemanticEventProjection) -> tuple[tuple[str, int, int], ...]:
    return tuple(_pitch_identity(pitch) for pitch in event.pitches)


def _accidentals(event: SemanticEventProjection) -> tuple[DisplayAccidental, ...]:
    return tuple(pitch.display_accidental for pitch in event.pitches)


def _duration_tag(duration: Fraction) -> str:
    mapping = {
        Fraction(1, 1): "duration:whole",
        Fraction(1, 2): "duration:half",
        Fraction(1, 4): "duration:quarter",
        Fraction(1, 8): "duration:eighth",
    }
    try:
        return mapping[duration]
    except KeyError as exc:
        raise ValidationDiagnosticError("target duration is outside supported V1") from exc


def _target_feature_tags(projection: SemanticScoreProjection) -> tuple[str, ...]:
    if len(projection.parts) != 1:
        raise ValidationDiagnosticError("D3 expects exactly one supported-V1 part")
    tags: set[str] = set()
    event_types: set[str] = set()
    saw_accidental = False

    for measure in projection.parts[0].measures:
        numerator, denominator = measure.time_signature
        tags.add(f"meter:{numerator}/{denominator}")
        if len(measure.voices) != 1:
            raise ValidationDiagnosticError("D3 expects exactly one supported-V1 voice")
        for event in measure.voices[0].events:
            event_types.add(event.event_type)
            tags.add(f"event:{event.event_type}")
            tags.add(_duration_tag(event.duration))
            if event.event_type == "chord":
                tags.add(f"chord-size:{len(event.pitches)}")
            for pitch in event.pitches:
                if pitch.display_accidental is not DisplayAccidental.NONE:
                    saw_accidental = True

    if len(event_types) > 1:
        tags.add("event:mixed")
    tags.add("accidental:present" if saw_accidental else "accidental:none")
    return tuple(sorted(tags))


@dataclass(frozen=True, slots=True)
class SampleDiagnostic:
    sample_id: str
    family_id: str
    token_edits: int
    reference_tokens: int
    exact_sequence: bool
    target_measures: int
    predicted_measures: int
    exact_measures: int
    meter_correct: int
    reference_events: int
    predicted_events: int
    event_edits: int
    event_type_correct: int
    onset_correct: int
    duration_correct: int
    pitched_events: int
    pitch_identity_correct: int
    accidental_correct: int
    rest_events: int
    rest_recognition_correct: int
    chord_events: int
    chord_size_correct: int
    missing_events: int
    extra_events: int
    feature_tags: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha("sample_id", self.sample_id)
        if not isinstance(self.family_id, str) or not self.family_id:
            raise ValidationDiagnosticError("family_id must be non-empty text")
        if not isinstance(self.exact_sequence, bool):
            raise ValidationDiagnosticError("exact_sequence must be boolean")
        if not isinstance(self.feature_tags, tuple) or any(
            not isinstance(tag, str) or not tag for tag in self.feature_tags
        ):
            raise ValidationDiagnosticError("feature_tags must be an immutable text tuple")

        values = {
            name: getattr(self, name)
            for name in (
                "token_edits",
                "reference_tokens",
                "target_measures",
                "predicted_measures",
                "exact_measures",
                "meter_correct",
                "reference_events",
                "predicted_events",
                "event_edits",
                "event_type_correct",
                "onset_correct",
                "duration_correct",
                "pitched_events",
                "pitch_identity_correct",
                "accidental_correct",
                "rest_events",
                "rest_recognition_correct",
                "chord_events",
                "chord_size_correct",
                "missing_events",
                "extra_events",
            )
        }
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values.values()):
            raise ValidationDiagnosticError("diagnostic counts must be non-negative integers")
        if self.reference_tokens <= 0 or self.target_measures <= 0:
            raise ValidationDiagnosticError("diagnostic reference denominators must be positive")
        bounded = (
            ("exact_measures", self.exact_measures, self.target_measures),
            ("meter_correct", self.meter_correct, self.target_measures),
            ("event_type_correct", self.event_type_correct, self.reference_events),
            ("onset_correct", self.onset_correct, self.reference_events),
            ("duration_correct", self.duration_correct, self.reference_events),
            ("pitch_identity_correct", self.pitch_identity_correct, self.pitched_events),
            ("accidental_correct", self.accidental_correct, self.pitched_events),
            ("rest_recognition_correct", self.rest_recognition_correct, self.rest_events),
            ("chord_size_correct", self.chord_size_correct, self.chord_events),
        )
        for name, value, maximum in bounded:
            if value > maximum:
                raise ValidationDiagnosticError(f"{name} exceeds its denominator")


def analyze_validation_sample(
    *,
    sample_id: str,
    family_id: str,
    target_token_ids: object,
    predicted_token_ids: object,
    extra_feature_tags: object = (),
) -> SampleDiagnostic:
    """Compare one validation target and prediction without accepting a split argument."""

    _require_sha("sample_id", sample_id)
    if (
        not isinstance(extra_feature_tags, tuple)
        or any(not isinstance(tag, str) or not tag for tag in extra_feature_tags)
    ):
        raise ValidationDiagnosticError("extra_feature_tags must be an immutable text tuple")
    target_ids = _require_token_ids("target_token_ids", target_token_ids)
    predicted_ids = _require_token_ids("predicted_token_ids", predicted_token_ids)

    try:
        target_projection = detokenize_tokens(decode_token_ids(target_ids))
        predicted_projection = detokenize_tokens(decode_token_ids(predicted_ids))
    except TokenizationError as exc:
        raise ValidationDiagnosticError("D3 requires supported-V1 semantic sequences") from exc

    target_surface = target_ids[1:]
    predicted_surface = predicted_ids[1:]
    token_edits = _levenshtein(target_surface, predicted_surface)

    target_measures = target_projection.parts[0].measures
    predicted_measures = predicted_projection.parts[0].measures
    exact_measures = 0
    meter_correct = 0
    reference_events = 0
    predicted_events = 0
    event_edits = 0
    event_type_correct = 0
    onset_correct = 0
    duration_correct = 0
    pitched_events = 0
    pitch_identity_correct = 0
    accidental_correct = 0
    rest_events = 0
    rest_recognition_correct = 0
    chord_events = 0
    chord_size_correct = 0
    missing_events = 0
    extra_events = 0

    measure_count = max(len(target_measures), len(predicted_measures))
    for index in range(measure_count):
        target_measure = target_measures[index] if index < len(target_measures) else None
        predicted_measure = predicted_measures[index] if index < len(predicted_measures) else None
        if target_measure is None:
            assert predicted_measure is not None
            extra_measure_events = len(predicted_measure.voices[0].events)
            extra_events += extra_measure_events
            predicted_events += extra_measure_events
            event_edits += extra_measure_events
            continue
        if predicted_measure is None:
            target_events = target_measure.voices[0].events
            reference_events += len(target_events)
            missing_events += len(target_events)
            event_edits += len(target_events)
            for target_event in target_events:
                if target_event.event_type != "rest":
                    pitched_events += 1
                if target_event.event_type == "rest":
                    rest_events += 1
                if target_event.event_type == "chord":
                    chord_events += 1
            continue

        if target_measure == predicted_measure:
            exact_measures += 1
        if target_measure.time_signature == predicted_measure.time_signature:
            meter_correct += 1

        target_events = target_measure.voices[0].events
        predicted_events_for_measure = predicted_measure.voices[0].events
        reference_events += len(target_events)
        predicted_events += len(predicted_events_for_measure)
        aligned, edits = _align_events(target_events, predicted_events_for_measure)
        event_edits += edits

        for target_event, predicted_event in aligned:
            if target_event is None:
                extra_events += 1
                continue
            if predicted_event is None:
                missing_events += 1
            if target_event.event_type == "rest":
                rest_events += 1
            else:
                pitched_events += 1
            if target_event.event_type == "chord":
                chord_events += 1

            if predicted_event is None:
                continue
            if target_event.event_type == predicted_event.event_type:
                event_type_correct += 1
            if target_event.onset == predicted_event.onset:
                onset_correct += 1
            if target_event.duration == predicted_event.duration:
                duration_correct += 1
            if target_event.event_type == "rest" and predicted_event.event_type == "rest":
                rest_recognition_correct += 1

            if target_event.event_type != "rest":
                if _pitch_identities(target_event) == _pitch_identities(predicted_event):
                    pitch_identity_correct += 1
                if _accidentals(target_event) == _accidentals(predicted_event):
                    accidental_correct += 1
            if target_event.event_type == "chord":
                if predicted_event.event_type == "chord" and len(target_event.pitches) == len(predicted_event.pitches):
                    chord_size_correct += 1

    return SampleDiagnostic(
        sample_id=sample_id,
        family_id=family_id,
        token_edits=token_edits,
        reference_tokens=len(target_surface),
        exact_sequence=target_ids == predicted_ids,
        target_measures=len(target_measures),
        predicted_measures=len(predicted_measures),
        exact_measures=exact_measures,
        meter_correct=meter_correct,
        reference_events=reference_events,
        predicted_events=predicted_events,
        event_edits=event_edits,
        event_type_correct=event_type_correct,
        onset_correct=onset_correct,
        duration_correct=duration_correct,
        pitched_events=pitched_events,
        pitch_identity_correct=pitch_identity_correct,
        accidental_correct=accidental_correct,
        rest_events=rest_events,
        rest_recognition_correct=rest_recognition_correct,
        chord_events=chord_events,
        chord_size_correct=chord_size_correct,
        missing_events=missing_events,
        extra_events=extra_events,
        feature_tags=tuple(sorted(set(_target_feature_tags(target_projection)) | set(extra_feature_tags))),
    )


def _rate(correct: int, total: int) -> float | None:
    if total == 0:
        return None
    return float(correct / total)


def _aggregate_bucket(samples: tuple[SampleDiagnostic, ...]) -> dict[str, object]:
    if not samples:
        raise ValidationDiagnosticError("cannot aggregate an empty diagnostic bucket")
    sums = {
        name: sum(getattr(sample, name) for sample in samples)
        for name in (
            "token_edits",
            "reference_tokens",
            "target_measures",
            "predicted_measures",
            "exact_measures",
            "meter_correct",
            "reference_events",
            "predicted_events",
            "event_edits",
            "event_type_correct",
            "onset_correct",
            "duration_correct",
            "pitched_events",
            "pitch_identity_correct",
            "accidental_correct",
            "rest_events",
            "rest_recognition_correct",
            "chord_events",
            "chord_size_correct",
            "missing_events",
            "extra_events",
        )
    }
    exact_sequences = sum(sample.exact_sequence for sample in samples)
    return {
        "samples": len(samples),
        "exact_sequence_accuracy": _rate(exact_sequences, len(samples)),
        "token_error_rate": _rate(sums["token_edits"], sums["reference_tokens"]),
        "measure_exact_accuracy": _rate(sums["exact_measures"], sums["target_measures"]),
        "meter_accuracy": _rate(sums["meter_correct"], sums["target_measures"]),
        "event_error_rate": _rate(sums["event_edits"], sums["reference_events"]),
        "event_type_accuracy": _rate(sums["event_type_correct"], sums["reference_events"]),
        "onset_accuracy": _rate(sums["onset_correct"], sums["reference_events"]),
        "duration_accuracy": _rate(sums["duration_correct"], sums["reference_events"]),
        "pitch_identity_accuracy": _rate(sums["pitch_identity_correct"], sums["pitched_events"]),
        "display_accidental_accuracy": _rate(sums["accidental_correct"], sums["pitched_events"]),
        "rest_recognition_accuracy": _rate(sums["rest_recognition_correct"], sums["rest_events"]),
        "chord_size_accuracy": _rate(sums["chord_size_correct"], sums["chord_events"]),
        "reference_events": sums["reference_events"],
        "predicted_events": sums["predicted_events"],
        "missing_events": sums["missing_events"],
        "extra_events": sums["extra_events"],
    }


def build_validation_diagnostic_report(
    samples: object,
    *,
    repository_sha: str,
    checkpoint_sha256: str,
    checkpoint_state_sha256: str,
    source_run_id: str,
) -> tuple[dict[str, object], bytes, str]:
    """Aggregate validation-only diagnostics into canonical hash-addressed evidence."""

    if not isinstance(samples, tuple) or not samples:
        raise ValidationDiagnosticError("samples must be a non-empty immutable diagnostic tuple")
    if any(not isinstance(sample, SampleDiagnostic) for sample in samples):
        raise ValidationDiagnosticError("samples contains a non-SampleDiagnostic value")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValidationDiagnosticError("duplicate validation sample_id")
    _require_sha("repository_sha", repository_sha, 40)
    _require_sha("checkpoint_sha256", checkpoint_sha256)
    _require_sha("checkpoint_state_sha256", checkpoint_state_sha256)
    _require_sha("source_run_id", source_run_id)

    ordered = tuple(sorted(samples, key=lambda sample: sample.sample_id))
    buckets: dict[str, list[SampleDiagnostic]] = {}
    for sample in ordered:
        for tag in sample.feature_tags:
            buckets.setdefault(tag, []).append(sample)

    report = {
        "schema_version": STAGE7D3_DIAGNOSTIC_SCHEMA,
        "repository_sha": repository_sha,
        "source_run_id": source_run_id,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_state_sha256": checkpoint_state_sha256,
        "validation_samples": len(ordered),
        "test_samples_exposed": 0,
        "aggregate": _aggregate_bucket(ordered),
        "feature_buckets": {
            key: _aggregate_bucket(tuple(value))
            for key, value in sorted(buckets.items())
        },
        "sample_records": [asdict(sample) for sample in ordered],
    }
    raw = _canonical_json_bytes(report)
    digest = sha256(raw).hexdigest()
    return report, raw, digest
