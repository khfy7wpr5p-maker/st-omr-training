"""Independent Stage 1-C validation for score/part/measure/voice structure."""

from __future__ import annotations

from fractions import Fraction
from typing import Final

from .core import ChordEvent, NoteEvent, RationalDuration, RestEvent
from .structure import (
    Clef,
    Measure,
    Part,
    Score,
    TimeSignature,
    V1_INSTRUMENT_CLASS,
    V1_TIME_SIGNATURES,
    Voice,
)
from .validator import ValidationIssue, ValidationResult, validate_v1_event


V1_NOTE_DURATIONS: Final[frozenset[Fraction]] = frozenset(
    {Fraction(1, 1), Fraction(1, 2), Fraction(1, 4), Fraction(1, 8)}
)
V1_REST_DURATIONS: Final[frozenset[Fraction]] = frozenset(
    {Fraction(1, 2), Fraction(1, 4), Fraction(1, 8)}
)


def _issue(code: str, path: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, path=path, message=message)


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _rebase(result: ValidationResult, base: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for issue in result.issues:
        if issue.path == "$":
            path = base
        elif issue.path.startswith("$."):
            path = f"{base}{issue.path[1:]}"
        else:
            path = base
        issues.append(ValidationIssue(issue.code, path, issue.message))
    return issues


def validate_time_signature(value: object) -> ValidationResult:
    if not isinstance(value, TimeSignature):
        return ValidationResult(
            (_issue("time_signature.type", "$", "expected TimeSignature"),)
        )

    issues: list[ValidationIssue] = []
    if not _is_plain_int(value.numerator) or not _is_plain_int(value.denominator):
        issues.append(
            _issue(
                "time_signature.value_type",
                "$",
                "time signature values must be integers",
            )
        )
    elif (value.numerator, value.denominator) not in V1_TIME_SIGNATURES:
        issues.append(
            _issue(
                "time_signature.unsupported",
                "$",
                "V1 time signature must be 2/4, 3/4, or 4/4",
            )
        )
    return ValidationResult(tuple(issues))


def validate_voice(value: object) -> ValidationResult:
    if not isinstance(value, Voice):
        return ValidationResult((_issue("voice.type", "$", "expected Voice"),))

    issues: list[ValidationIssue] = []
    if not _is_plain_int(value.voice_id) or value.voice_id != 1:
        issues.append(
            _issue("v1.voice_id", "$.voice_id", "V1 requires voice_id == 1")
        )

    if not isinstance(value.events, tuple):
        issues.append(
            _issue("voice.events_type", "$.events", "events must be an immutable tuple")
        )
        return ValidationResult(tuple(issues))

    previous_onset: Fraction | None = None
    for index, event in enumerate(value.events):
        path = f"$.events[{index}]"
        if not isinstance(event, (NoteEvent, RestEvent, ChordEvent)):
            issues.append(
                _issue(
                    "voice.event_type",
                    path,
                    "voice event must be NoteEvent, RestEvent, or ChordEvent",
                )
            )
            continue

        issues.extend(_rebase(validate_v1_event(event), path))

        if event.voice != value.voice_id:
            issues.append(
                _issue(
                    "voice.event_voice",
                    f"{path}.voice",
                    "event voice must equal containing voice_id",
                )
            )

        if isinstance(event.onset, Fraction):
            if previous_onset is not None and event.onset < previous_onset:
                issues.append(
                    _issue(
                        "voice.event_order",
                        f"{path}.onset",
                        "events must be ordered by non-decreasing onset",
                    )
                )
            previous_onset = event.onset

    return ValidationResult(tuple(issues))


def _event_duration_policy(event: object, path: str) -> list[ValidationIssue]:
    if not isinstance(event, (NoteEvent, RestEvent, ChordEvent)):
        return []
    if not isinstance(event.duration, RationalDuration):
        return []
    duration = event.duration.fraction
    allowed = V1_REST_DURATIONS if isinstance(event, RestEvent) else V1_NOTE_DURATIONS
    if duration not in allowed:
        return [
            _issue(
                "measure.event_duration_unsupported",
                f"{path}.duration",
                "event duration is outside the Stage 1-C V1 duration set",
            )
        ]
    return []


def _validate_measure_timeline(
    voice: Voice, capacity: Fraction, path: str
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    cursor = Fraction(0, 1)
    total_duration = Fraction(0, 1)

    for index, event in enumerate(voice.events):
        event_path = f"{path}.events[{index}]"
        issues.extend(_event_duration_policy(event, event_path))
        if not isinstance(event, (NoteEvent, RestEvent, ChordEvent)):
            continue
        if not isinstance(event.onset, Fraction):
            continue
        if not isinstance(event.duration, RationalDuration):
            continue
        duration = event.duration.fraction
        if duration <= 0:
            continue
        total_duration += duration

        start = event.onset
        end = start + duration

        if start < cursor:
            issues.append(
                _issue(
                    "measure.overlap",
                    f"{event_path}.onset",
                    "events must not overlap in a V1 measure",
                )
            )
        elif start > cursor:
            issues.append(
                _issue(
                    "measure.gap",
                    f"{event_path}.onset",
                    "silence must be represented by an explicit RestEvent",
                )
            )

        if end > capacity:
            issues.append(
                _issue(
                    "measure.overflow",
                    event_path,
                    "event extends beyond the measure capacity",
                )
            )

        if end > cursor:
            cursor = end

    if total_duration < capacity:
        issues.append(
            _issue(
                "measure.underflow",
                path,
                "measure event durations do not exactly fill the measure capacity",
            )
        )
    elif total_duration > capacity:
        issues.append(
            _issue(
                "measure.duration_overflow",
                path,
                "total event duration exceeds the measure capacity",
            )
        )

    return issues


def validate_measure(value: object) -> ValidationResult:
    if not isinstance(value, Measure):
        return ValidationResult((_issue("measure.type", "$", "expected Measure"),))

    issues: list[ValidationIssue] = []

    if not _is_plain_int(value.number) or value.number <= 0:
        issues.append(
            _issue("measure.number", "$.number", "measure number must be positive")
        )

    time_result = validate_time_signature(value.time_signature)
    issues.extend(_rebase(time_result, "$.time_signature"))

    capacity: Fraction | None = None
    if (
        isinstance(value.time_signature, TimeSignature)
        and _is_plain_int(value.time_signature.numerator)
        and _is_plain_int(value.time_signature.denominator)
        and (value.time_signature.numerator, value.time_signature.denominator)
        in V1_TIME_SIGNATURES
    ):
        capacity = Fraction(
            value.time_signature.numerator, value.time_signature.denominator
        )

    if not _is_plain_int(value.key_signature) or value.key_signature != 0:
        issues.append(
            _issue(
                "v1.key_signature",
                "$.key_signature",
                "V1 requires key_signature == 0",
            )
        )

    if value.clef is not Clef.TREBLE:
        issues.append(_issue("v1.clef", "$.clef", "V1 requires treble clef"))

    if not isinstance(value.expected_duration, Fraction):
        issues.append(
            _issue(
                "measure.expected_duration_type",
                "$.expected_duration",
                "expected_duration must be fractions.Fraction",
            )
        )
    elif value.expected_duration <= 0:
        issues.append(
            _issue(
                "measure.expected_duration",
                "$.expected_duration",
                "expected_duration must be positive",
            )
        )
    elif capacity is not None and value.expected_duration != capacity:
        issues.append(
            _issue(
                "measure.expected_duration_mismatch",
                "$.expected_duration",
                "expected_duration must equal the time-signature capacity",
            )
        )

    if not isinstance(value.voices, tuple):
        issues.append(
            _issue("measure.voices_type", "$.voices", "voices must be an immutable tuple")
        )
        return ValidationResult(tuple(issues))

    if len(value.voices) != 1:
        issues.append(
            _issue("v1.voice_count", "$.voices", "V1 requires exactly one voice")
        )

    for index, voice in enumerate(value.voices):
        path = f"$.voices[{index}]"
        if not isinstance(voice, Voice):
            issues.append(_issue("measure.voice_type", path, "every voice must be Voice"))
            continue
        issues.extend(_rebase(validate_voice(voice), path))

    if len(value.voices) == 1 and isinstance(value.voices[0], Voice) and capacity is not None:
        issues.extend(_validate_measure_timeline(value.voices[0], capacity, "$.voices[0]"))

    return ValidationResult(tuple(issues))


def validate_part(value: object) -> ValidationResult:
    if not isinstance(value, Part):
        return ValidationResult((_issue("part.type", "$", "expected Part"),))

    issues: list[ValidationIssue] = []

    if not isinstance(value.part_id, str) or not value.part_id.strip():
        issues.append(_issue("part.id", "$.part_id", "part_id must be non-empty"))

    if value.instrument_class != V1_INSTRUMENT_CLASS:
        issues.append(
            _issue(
                "v1.instrument_class",
                "$.instrument_class",
                f"V1 requires instrument_class == {V1_INSTRUMENT_CLASS!r}",
            )
        )

    if not _is_plain_int(value.staff_count) or value.staff_count != 1:
        issues.append(
            _issue("v1.staff_count", "$.staff_count", "V1 requires staff_count == 1")
        )

    if not isinstance(value.measures, tuple):
        issues.append(
            _issue("part.measures_type", "$.measures", "measures must be an immutable tuple")
        )
        return ValidationResult(tuple(issues))

    if not value.measures:
        issues.append(_issue("part.empty", "$.measures", "part must contain measures"))

    for index, measure in enumerate(value.measures):
        path = f"$.measures[{index}]"
        if not isinstance(measure, Measure):
            issues.append(_issue("part.measure_type", path, "every measure must be Measure"))
            continue
        issues.extend(_rebase(validate_measure(measure), path))
        expected_number = index + 1
        if measure.number != expected_number:
            issues.append(
                _issue(
                    "part.measure_number_sequence",
                    f"{path}.number",
                    f"expected measure number {expected_number}",
                )
            )

    return ValidationResult(tuple(issues))


def validate_score(value: object) -> ValidationResult:
    if not isinstance(value, Score):
        return ValidationResult((_issue("score.type", "$", "expected Score"),))

    issues: list[ValidationIssue] = []

    for field_name in ("score_id", "schema_version", "generator_version"):
        field_value = getattr(value, field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            issues.append(
                _issue(
                    f"score.{field_name}",
                    f"$.{field_name}",
                    f"{field_name} must be non-empty",
                )
            )

    if not _is_plain_int(value.seed):
        issues.append(_issue("score.seed", "$.seed", "seed must be an integer"))

    if not isinstance(value.provenance, tuple):
        issues.append(
            _issue(
                "score.provenance_type",
                "$.provenance",
                "provenance must be an immutable tuple",
            )
        )
    else:
        for index, item in enumerate(value.provenance):
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(isinstance(entry, str) for entry in item)
            ):
                issues.append(
                    _issue(
                        "score.provenance_entry",
                        f"$.provenance[{index}]",
                        "provenance entries must be two-string tuples",
                    )
                )

    if not isinstance(value.parts, tuple):
        issues.append(_issue("score.parts_type", "$.parts", "parts must be an immutable tuple"))
        return ValidationResult(tuple(issues))

    if len(value.parts) != 1:
        issues.append(_issue("v1.part_count", "$.parts", "V1 requires exactly one part"))

    for index, part in enumerate(value.parts):
        path = f"$.parts[{index}]"
        if not isinstance(part, Part):
            issues.append(_issue("score.part_type", path, "every part must be Part"))
            continue
        issues.extend(_rebase(validate_part(part), path))

    return ValidationResult(tuple(issues))
