"""Deterministic Stage 1-D ST Music Generator v1.

The generator creates only canonical in-memory score structures. It does not
serialize MusicXML, render notation, create datasets, or train models.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from typing import Final, Sequence, TypeVar

from .core import (
    ChordEvent,
    DisplayAccidental,
    NoteEvent,
    NotationIntent,
    Pitch,
    RationalDuration,
    RestEvent,
)
from .structure import Clef, Measure, Part, Score, TimeSignature, Voice, V1_TIME_SIGNATURES
from .structure_validator import validate_score
from .validator import ValidationResult


DEFAULT_SCHEMA_VERSION: Final[str] = "st-canonical-1"
DEFAULT_GENERATOR_VERSION: Final[str] = "st-generator-v1"
MAX_MEASURE_COUNT: Final[int] = 256
MIN_GENERATOR_OCTAVE: Final[int] = 3
MAX_GENERATOR_OCTAVE: Final[int] = 6
DEFAULT_STEPS: Final[tuple[str, ...]] = tuple("CDEFGAB")
DEFAULT_OCTAVES: Final[tuple[int, ...]] = (4, 5)
DEFAULT_EVENT_KINDS: Final[tuple[str, ...]] = ("note", "rest", "chord")
_ALLOWED_EVENT_KINDS: Final[frozenset[str]] = frozenset(DEFAULT_EVENT_KINDS)
_NOTE_DURATIONS: Final[tuple[Fraction, ...]] = (
    Fraction(1, 8),
    Fraction(1, 4),
    Fraction(1, 2),
    Fraction(1, 1),
)
_REST_DURATIONS: Final[tuple[Fraction, ...]] = (
    Fraction(1, 8),
    Fraction(1, 4),
    Fraction(1, 2),
)
_ALTERS: Final[tuple[int, ...]] = (-1, 0, 1)

T = TypeVar("T")


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Immutable deterministic V1 generator policy."""

    measure_count: int = 8
    time_signatures: tuple[tuple[int, int], ...] = ((2, 4), (3, 4), (4, 4))
    steps: tuple[str, ...] = DEFAULT_STEPS
    octaves: tuple[int, ...] = DEFAULT_OCTAVES
    event_kinds: tuple[str, ...] = DEFAULT_EVENT_KINDS
    allow_accidentals: bool = True

    def __post_init__(self) -> None:
        if not _is_plain_int(self.measure_count) or not (1 <= self.measure_count <= MAX_MEASURE_COUNT):
            raise ValueError(f"measure_count must be an integer from 1 through {MAX_MEASURE_COUNT}")

        if not isinstance(self.time_signatures, tuple) or not self.time_signatures:
            raise ValueError("time_signatures must be a non-empty immutable tuple")
        normalized_time_signatures: list[tuple[int, int]] = []
        for item in self.time_signatures:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(_is_plain_int(value) for value in item)
                or item not in V1_TIME_SIGNATURES
            ):
                raise ValueError("time_signatures may contain only V1 2/4, 3/4, and 4/4")
            normalized_time_signatures.append(item)
        if len(set(normalized_time_signatures)) != len(normalized_time_signatures):
            raise ValueError("time_signatures must not contain duplicates")

        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("steps must be a non-empty immutable tuple")
        normalized_steps: list[str] = []
        for step in self.steps:
            if not isinstance(step, str) or len(step) != 1 or step.upper() not in "ABCDEFG":
                raise ValueError("steps may contain only A through G")
            normalized_steps.append(step.upper())
        if len(set(normalized_steps)) != len(normalized_steps):
            raise ValueError("steps must not contain duplicates")
        object.__setattr__(self, "steps", tuple(normalized_steps))

        if not isinstance(self.octaves, tuple) or not self.octaves:
            raise ValueError("octaves must be a non-empty immutable tuple")
        for octave in self.octaves:
            if (
                not _is_plain_int(octave)
                or not (MIN_GENERATOR_OCTAVE <= octave <= MAX_GENERATOR_OCTAVE)
            ):
                raise ValueError(
                    f"generator octaves must be integers from {MIN_GENERATOR_OCTAVE} through {MAX_GENERATOR_OCTAVE}"
                )
        if len(set(self.octaves)) != len(self.octaves):
            raise ValueError("octaves must not contain duplicates")

        if not isinstance(self.event_kinds, tuple) or not self.event_kinds:
            raise ValueError("event_kinds must be a non-empty immutable tuple")
        if any(not isinstance(kind, str) or kind not in _ALLOWED_EVENT_KINDS for kind in self.event_kinds):
            raise ValueError("event_kinds may contain only note, rest, and chord")
        if len(set(self.event_kinds)) != len(self.event_kinds):
            raise ValueError("event_kinds must not contain duplicates")

        if not isinstance(self.allow_accidentals, bool):
            raise TypeError("allow_accidentals must be bool")

        pitch_positions = len(self.steps) * len(self.octaves)
        if "chord" in self.event_kinds and pitch_positions < 2:
            raise ValueError("chord generation requires at least two distinct pitch positions")


class GenerationValidationError(RuntimeError):
    """Raised when independently validating generated output fails."""

    def __init__(self, result: ValidationResult):
        self.result = result
        codes = ", ".join(issue.code for issue in result.issues)
        super().__init__(f"generated score failed independent validation: {codes}")


class _StableRng:
    """Small SHA-256 counter RNG whose sequence is independent of random module changes."""

    def __init__(self, key: bytes):
        self._key = key
        self._counter = 0

    def _next_int(self) -> int:
        counter = self._counter.to_bytes(16, "big", signed=False)
        self._counter += 1
        return int.from_bytes(sha256(self._key + counter).digest(), "big")

    def choose(self, values: Sequence[T]) -> T:
        if not values:
            raise ValueError("cannot choose from an empty sequence")
        return values[self._next_int() % len(values)]

    def integer(self, minimum: int, maximum: int) -> int:
        if minimum > maximum:
            raise ValueError("minimum must not exceed maximum")
        return minimum + (self._next_int() % (maximum - minimum + 1))


def _config_payload(config: GeneratorConfig) -> dict[str, object]:
    return {
        "allow_accidentals": config.allow_accidentals,
        "event_kinds": list(config.event_kinds),
        "measure_count": config.measure_count,
        "octaves": list(config.octaves),
        "steps": list(config.steps),
        "time_signatures": [list(item) for item in config.time_signatures],
    }


def config_fingerprint(config: GeneratorConfig) -> str:
    if not isinstance(config, GeneratorConfig):
        raise TypeError("config must be GeneratorConfig")
    payload = json.dumps(
        _config_payload(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256(payload).hexdigest()


def _rng_for(config: GeneratorConfig, seed: int, generator_version: str) -> _StableRng:
    material = json.dumps(
        {
            "config_fingerprint": config_fingerprint(config),
            "generator_version": generator_version,
            "seed": seed,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return _StableRng(sha256(material).digest())


def _score_id(config: GeneratorConfig, seed: int, generator_version: str) -> str:
    material = json.dumps(
        {
            "config_fingerprint": config_fingerprint(config),
            "generator_version": generator_version,
            "schema_version": DEFAULT_SCHEMA_VERSION,
            "seed": seed,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return f"stomr-{sha256(material).hexdigest()}"


def _duration_object(value: Fraction) -> RationalDuration:
    return RationalDuration(value.numerator, value.denominator)


def _display_for_transition(previous_alter: int, target_alter: int) -> DisplayAccidental:
    if target_alter == previous_alter:
        return DisplayAccidental.NONE
    if target_alter == 1:
        return DisplayAccidental.SHARP
    if target_alter == -1:
        return DisplayAccidental.FLAT
    return DisplayAccidental.NATURAL


def _make_note(
    *,
    rng: _StableRng,
    config: GeneratorConfig,
    onset: Fraction,
    duration: RationalDuration,
    accidental_state: dict[tuple[str, int], int],
    excluded_positions: set[tuple[str, int]] | None = None,
) -> NoteEvent:
    excluded = excluded_positions or set()
    positions = tuple(
        (step, octave)
        for step in config.steps
        for octave in config.octaves
        if (step, octave) not in excluded
    )
    if not positions:
        raise ValueError("no pitch position remains available")

    step, octave = rng.choose(positions)
    position = (step, octave)
    previous_alter = accidental_state.get(position, 0)
    target_alter = rng.choose(_ALTERS) if config.allow_accidentals else 0
    display = _display_for_transition(previous_alter, target_alter)
    accidental_state[position] = target_alter

    return NoteEvent(
        onset,
        duration,
        Pitch(step, target_alter, octave),
        NotationIntent(display),
        voice=1,
        staff=1,
    )


def _make_chord(
    *,
    rng: _StableRng,
    config: GeneratorConfig,
    onset: Fraction,
    duration: RationalDuration,
    accidental_state: dict[tuple[str, int], int],
) -> ChordEvent:
    maximum_size = min(4, len(config.steps) * len(config.octaves))
    chord_size = rng.integer(2, maximum_size)
    used_positions: set[tuple[str, int]] = set()
    notes: list[NoteEvent] = []

    for _ in range(chord_size):
        note = _make_note(
            rng=rng,
            config=config,
            onset=onset,
            duration=duration,
            accidental_state=accidental_state,
            excluded_positions=used_positions,
        )
        used_positions.add((note.pitch.step, note.pitch.octave))
        notes.append(note)

    return ChordEvent(onset, duration, tuple(notes), voice=1, staff=1)


def _generate_measure(
    *,
    number: int,
    rng: _StableRng,
    config: GeneratorConfig,
) -> Measure:
    numerator, denominator = rng.choose(config.time_signatures)
    time_signature = TimeSignature(numerator, denominator)
    capacity = time_signature.capacity
    cursor = Fraction(0, 1)
    accidental_state: dict[tuple[str, int], int] = {}
    events: list[NoteEvent | RestEvent | ChordEvent] = []

    while cursor < capacity:
        remaining = capacity - cursor
        kind = rng.choose(config.event_kinds)
        duration_pool = _REST_DURATIONS if kind == "rest" else _NOTE_DURATIONS
        fitting = tuple(duration for duration in duration_pool if duration <= remaining)
        if not fitting:
            raise RuntimeError("generator could not find a duration that fits the remaining measure")
        duration_fraction = rng.choose(fitting)
        duration = _duration_object(duration_fraction)

        if kind == "rest":
            event: NoteEvent | RestEvent | ChordEvent = RestEvent(
                cursor, duration, voice=1, staff=1
            )
        elif kind == "chord":
            event = _make_chord(
                rng=rng,
                config=config,
                onset=cursor,
                duration=duration,
                accidental_state=accidental_state,
            )
        else:
            event = _make_note(
                rng=rng,
                config=config,
                onset=cursor,
                duration=duration,
                accidental_state=accidental_state,
            )

        events.append(event)
        cursor += duration_fraction

    return Measure(
        number=number,
        time_signature=time_signature,
        voices=(Voice(voice_id=1, events=tuple(events)),),
        key_signature=0,
        clef=Clef.TREBLE,
        expected_duration=capacity,
    )


def generate_score(
    config: GeneratorConfig,
    seed: int,
    *,
    generator_version: str = DEFAULT_GENERATOR_VERSION,
) -> Score:
    """Generate a deterministic V1 score and require independent validation."""

    if not isinstance(config, GeneratorConfig):
        raise TypeError("config must be GeneratorConfig")
    if not _is_plain_int(seed):
        raise TypeError("seed must be an integer")
    generator_version = _require_nonempty_string("generator_version", generator_version)

    fingerprint = config_fingerprint(config)
    score_id = _score_id(config, seed, generator_version)
    rng = _rng_for(config, seed, generator_version)

    measures = tuple(
        _generate_measure(number=index, rng=rng, config=config)
        for index in range(1, config.measure_count + 1)
    )
    part = Part(part_id="P1", measures=measures)
    provenance = (
        ("config_fingerprint", fingerprint),
        ("created_by_pipeline", "st-music-generator"),
        ("generator_version", generator_version),
        ("seed", str(seed)),
        ("source_id", score_id),
        ("source_type", "procedural"),
    )
    score = Score(
        score_id=score_id,
        schema_version=DEFAULT_SCHEMA_VERSION,
        generator_version=generator_version,
        seed=seed,
        provenance=provenance,
        parts=(part,),
    )

    result = validate_score(score)
    if not result.is_valid:
        raise GenerationValidationError(result)
    return score
