"""Independent Stage 1-B validation for canonical ST-OMR core objects.

The validator deliberately re-checks invariants instead of trusting successful
construction of Stage 1-A dataclasses. It defines the active V1 policy boundary:
one voice, one staff, canonical rational timing, V1 pitch limits, and coherent
visible accidental intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Final

from .core import (
    MAX_OCTAVE,
    MIN_OCTAVE,
    V1_ALTERS,
    ChordEvent,
    DisplayAccidental,
    NoteEvent,
    NotationIntent,
    Pitch,
    RationalDuration,
    RestEvent,
)


V1_VOICE: Final[int] = 1
V1_STAFF: Final[int] = 1


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues


def _issue(code: str, path: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, path=path, message=message)


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_onset(value: object, path: str) -> list[ValidationIssue]:
    if not isinstance(value, Fraction):
        return [
            _issue(
                "onset.type",
                path,
                "canonical onset must be fractions.Fraction",
            )
        ]
    if value < 0:
        return [_issue("onset.negative", path, "onset must be non-negative")]
    return []


def _validate_duration(value: object, path: str) -> list[ValidationIssue]:
    if not isinstance(value, RationalDuration):
        return [
            _issue(
                "duration.type",
                path,
                "duration must be RationalDuration",
            )
        ]

    issues: list[ValidationIssue] = []
    numerator = value.numerator
    denominator = value.denominator

    if not _is_plain_int(numerator):
        issues.append(
            _issue(
                "duration.numerator_type",
                f"{path}.numerator",
                "duration numerator must be an integer",
            )
        )
    if not _is_plain_int(denominator):
        issues.append(
            _issue(
                "duration.denominator_type",
                f"{path}.denominator",
                "duration denominator must be an integer",
            )
        )

    if issues:
        return issues

    if denominator <= 0:
        issues.append(
            _issue(
                "duration.denominator",
                f"{path}.denominator",
                "duration denominator must be positive",
            )
        )
    if numerator <= 0:
        issues.append(
            _issue(
                "duration.non_positive",
                f"{path}.numerator",
                "duration must be strictly positive",
            )
        )

    if numerator > 0 and denominator > 0 and gcd(numerator, denominator) != 1:
        issues.append(
            _issue(
                "duration.not_canonical",
                path,
                "duration must be reduced to canonical rational form",
            )
        )

    return issues


def _validate_pitch(value: object, path: str) -> list[ValidationIssue]:
    if not isinstance(value, Pitch):
        return [_issue("pitch.type", path, "pitch must be Pitch")]

    issues: list[ValidationIssue] = []

    if (
        not isinstance(value.step, str)
        or len(value.step) != 1
        or value.step not in "ABCDEFG"
    ):
        issues.append(
            _issue(
                "pitch.step",
                f"{path}.step",
                "canonical V1 step must be uppercase A through G",
            )
        )

    if not _is_plain_int(value.alter) or value.alter not in V1_ALTERS:
        issues.append(
            _issue(
                "pitch.alter",
                f"{path}.alter",
                "V1 alter must be one of -1, 0, +1",
            )
        )

    if (
        not _is_plain_int(value.octave)
        or not (MIN_OCTAVE <= value.octave <= MAX_OCTAVE)
    ):
        issues.append(
            _issue(
                "pitch.octave",
                f"{path}.octave",
                f"V1 octave must be an integer from {MIN_OCTAVE} through {MAX_OCTAVE}",
            )
        )

    return issues


def _validate_notation_intent(value: object, path: str) -> list[ValidationIssue]:
    if not isinstance(value, NotationIntent):
        return [
            _issue(
                "notation_intent.type",
                path,
                "notation_intent must be NotationIntent",
            )
        ]
    if not isinstance(value.display_accidental, DisplayAccidental):
        return [
            _issue(
                "notation_intent.display_accidental",
                f"{path}.display_accidental",
                "display_accidental must be DisplayAccidental",
            )
        ]
    return []


def _validate_v1_placement(
    voice: object, staff: object, path: str
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if not _is_plain_int(voice) or voice != V1_VOICE:
        issues.append(
            _issue(
                "v1.voice",
                f"{path}.voice",
                "Stage 1-B V1 policy requires voice == 1",
            )
        )

    if not _is_plain_int(staff) or staff != V1_STAFF:
        issues.append(
            _issue(
                "v1.staff",
                f"{path}.staff",
                "Stage 1-B V1 policy requires staff == 1",
            )
        )

    return issues


def _validate_accidental_coherence(
    pitch: object, notation_intent: object, path: str
) -> list[ValidationIssue]:
    if not isinstance(pitch, Pitch) or not isinstance(
        notation_intent, NotationIntent
    ):
        return []
    accidental = notation_intent.display_accidental
    if not isinstance(accidental, DisplayAccidental):
        return []

    required_alter = {
        DisplayAccidental.SHARP: 1,
        DisplayAccidental.FLAT: -1,
        DisplayAccidental.NATURAL: 0,
    }.get(accidental)

    if required_alter is not None and pitch.alter != required_alter:
        return [
            _issue(
                "notation_intent.accidental_mismatch",
                f"{path}.notation_intent.display_accidental",
                f"{accidental.value} display intent requires pitch alter {required_alter}",
            )
        ]
    return []


def _validate_note_event(event: NoteEvent, path: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    issues.extend(_validate_onset(event.onset, f"{path}.onset"))
    issues.extend(_validate_duration(event.duration, f"{path}.duration"))
    issues.extend(_validate_pitch(event.pitch, f"{path}.pitch"))
    issues.extend(
        _validate_notation_intent(
            event.notation_intent, f"{path}.notation_intent"
        )
    )
    issues.extend(_validate_v1_placement(event.voice, event.staff, path))
    issues.extend(
        _validate_accidental_coherence(event.pitch, event.notation_intent, path)
    )
    return issues


def validate_note_event(event: object) -> ValidationResult:
    if not isinstance(event, NoteEvent):
        return ValidationResult((_issue("event.type", "$", "expected NoteEvent"),))
    return ValidationResult(tuple(_validate_note_event(event, "$")))


def validate_rest_event(event: object) -> ValidationResult:
    if not isinstance(event, RestEvent):
        return ValidationResult((_issue("event.type", "$", "expected RestEvent"),))

    issues: list[ValidationIssue] = []
    issues.extend(_validate_onset(event.onset, "$.onset"))
    issues.extend(_validate_duration(event.duration, "$.duration"))
    issues.extend(_validate_v1_placement(event.voice, event.staff, "$"))
    return ValidationResult(tuple(issues))


def validate_chord_event(event: object) -> ValidationResult:
    if not isinstance(event, ChordEvent):
        return ValidationResult((_issue("event.type", "$", "expected ChordEvent"),))

    issues: list[ValidationIssue] = []
    issues.extend(_validate_onset(event.onset, "$.onset"))
    issues.extend(_validate_duration(event.duration, "$.duration"))
    issues.extend(_validate_v1_placement(event.voice, event.staff, "$"))

    notes = event.notes
    if not isinstance(notes, tuple):
        issues.append(
            _issue(
                "chord.notes_type",
                "$.notes",
                "chord notes must be an immutable tuple",
            )
        )
        return ValidationResult(tuple(issues))

    if not 2 <= len(notes) <= 4:
        issues.append(
            _issue(
                "chord.size",
                "$.notes",
                "V1 chord must contain 2 through 4 notes",
            )
        )

    valid_members: list[tuple[int, NoteEvent]] = []
    for index, note in enumerate(notes):
        member_path = f"$.notes[{index}]"
        if not isinstance(note, NoteEvent):
            issues.append(
                _issue(
                    "chord.member_type",
                    member_path,
                    "every chord member must be NoteEvent",
                )
            )
            continue

        valid_members.append((index, note))
        issues.extend(_validate_note_event(note, member_path))

        if note.onset != event.onset:
            issues.append(
                _issue(
                    "chord.member_onset",
                    f"{member_path}.onset",
                    "chord member onset must equal chord onset",
                )
            )
        if note.duration != event.duration:
            issues.append(
                _issue(
                    "chord.member_duration",
                    f"{member_path}.duration",
                    "chord member duration must equal chord duration",
                )
            )
        if note.voice != event.voice:
            issues.append(
                _issue(
                    "chord.member_voice",
                    f"{member_path}.voice",
                    "chord member voice must equal chord voice",
                )
            )
        if note.staff != event.staff:
            issues.append(
                _issue(
                    "chord.member_staff",
                    f"{member_path}.staff",
                    "chord member staff must equal chord staff",
                )
            )

    seen_pitches: list[tuple[object, object, object]] = []
    for index, note in valid_members:
        pitch = note.pitch
        if not isinstance(pitch, Pitch):
            continue
        identity = (pitch.step, pitch.alter, pitch.octave)
        if identity in seen_pitches:
            issues.append(
                _issue(
                    "chord.duplicate_pitch",
                    f"$.notes[{index}].pitch",
                    "duplicate pitches are not allowed in a chord",
                )
            )
        else:
            seen_pitches.append(identity)

    return ValidationResult(tuple(issues))


def validate_v1_event(event: object) -> ValidationResult:
    if isinstance(event, NoteEvent):
        return validate_note_event(event)
    if isinstance(event, RestEvent):
        return validate_rest_event(event)
    if isinstance(event, ChordEvent):
        return validate_chord_event(event)
    return ValidationResult(
        (
            _issue(
                "event.unsupported_type",
                "$",
                "V1 event must be NoteEvent, RestEvent, or ChordEvent",
            ),
        )
    )
