"""ST-OMR training laboratory package."""

from .core import (
    ChordEvent,
    DisplayAccidental,
    NoteEvent,
    NotationIntent,
    Pitch,
    RationalDuration,
    RestEvent,
)
from .validator import (
    ValidationIssue,
    ValidationResult,
    validate_chord_event,
    validate_note_event,
    validate_rest_event,
    validate_v1_event,
)
from .structure import Clef, Measure, Part, Score, TimeSignature, Voice
from .structure_validator import (
    validate_measure,
    validate_part,
    validate_score,
    validate_time_signature,
    validate_voice,
)
from .generator import (
    DEFAULT_GENERATOR_VERSION,
    DEFAULT_SCHEMA_VERSION,
    GenerationValidationError,
    GeneratorConfig,
    config_fingerprint,
    generate_score,
)

__all__ = [
    "ChordEvent",
    "DisplayAccidental",
    "NoteEvent",
    "NotationIntent",
    "Pitch",
    "RationalDuration",
    "RestEvent",
    "ValidationIssue",
    "ValidationResult",
    "validate_chord_event",
    "validate_note_event",
    "validate_rest_event",
    "validate_v1_event",
    "Clef",
    "Measure",
    "Part",
    "Score",
    "TimeSignature",
    "Voice",
    "validate_measure",
    "validate_part",
    "validate_score",
    "validate_time_signature",
    "validate_voice",
    "DEFAULT_GENERATOR_VERSION",
    "DEFAULT_SCHEMA_VERSION",
    "GenerationValidationError",
    "GeneratorConfig",
    "config_fingerprint",
    "generate_score",
]
