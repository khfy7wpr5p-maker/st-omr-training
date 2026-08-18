"""Model-agnostic runtime specialist evidence boundary.

This module defines the observations that future Meter/NoteHead/Rest/Accidental
runtime adapters may emit.  It does not load or import any D10/D13 model,
checkpoint, optimizer, dataset, or TEST split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .runtime_geometry_engine_contract import BoxContract


SPECIALIST_TASK_CLASSES: Final[dict[str, tuple[str, ...]]] = {
    "meter": ("none", "2/4", "3/4", "4/4"),
    "notehead": ("open", "filled"),
    "rest": ("half", "quarter", "eighth"),
    "accidental": ("sharp", "flat", "natural"),
}
SPECIALIST_STATUSES: Final[tuple[str, ...]] = ("accepted", "ambiguous", "rejected")
SOURCE_KINDS: Final[tuple[str, ...]] = ("shadow-fixture", "specialist-adapter")


@dataclass(frozen=True, slots=True)
class SpecialistObservation:
    observation_id: str
    task: str
    measure_id: str
    staff_id: str
    status: str
    confidence_milli: int
    class_label: str | None = None
    bbox: BoxContract | None = None
    reasons: tuple[str, ...] = ()
    source_kind: str = "specialist-adapter"

    def __post_init__(self) -> None:
        if not self.observation_id or not self.measure_id or not self.staff_id:
            raise ValueError("specialist observation identities must be non-empty")
        if self.task not in SPECIALIST_TASK_CLASSES:
            raise ValueError("unsupported specialist task")
        if self.status not in SPECIALIST_STATUSES:
            raise ValueError("unsupported specialist status")
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError("unsupported specialist evidence source kind")
        if (
            not isinstance(self.confidence_milli, int)
            or isinstance(self.confidence_milli, bool)
            or not 0 <= self.confidence_milli <= 1000
        ):
            raise ValueError("confidence_milli must be an integer in 0..1000")

        if self.status == "accepted":
            if self.reasons:
                raise ValueError("accepted specialist observation cannot carry ambiguity reasons")
            if self.class_label not in SPECIALIST_TASK_CLASSES[self.task]:
                raise ValueError("accepted observation requires an allowed class label")
            if self.task in {"notehead", "rest", "accidental"} and self.bbox is None:
                raise ValueError("accepted localized symbol observation requires bbox")
            if self.task == "meter" and self.class_label != "none" and self.bbox is None:
                raise ValueError("visible accepted meter requires bbox")
        else:
            if not self.reasons:
                raise ValueError("ambiguous/rejected specialist observation must explain why")
            if self.class_label is not None and self.class_label not in SPECIALIST_TASK_CLASSES[self.task]:
                raise ValueError("non-accepted class label is outside task vocabulary")


@dataclass(frozen=True, slots=True)
class SpecialistEvidenceBatch:
    observations: tuple[SpecialistObservation, ...]

    def __post_init__(self) -> None:
        ids = tuple(item.observation_id for item in self.observations)
        if len(ids) != len(set(ids)):
            raise ValueError("specialist observation ids must be unique")

    def for_measure(self, measure_id: str) -> tuple[SpecialistObservation, ...]:
        return tuple(item for item in self.observations if item.measure_id == measure_id)
