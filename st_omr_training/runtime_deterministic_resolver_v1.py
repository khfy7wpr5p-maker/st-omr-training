"""Deterministic resolver for runtime specialist observations.

The resolver combines model-agnostic Meter/NoteHead/Rest/Accidental evidence
with accepted runtime measure/staff geometry. It performs only deterministic
validation, ordering and accidental-to-notehead association. Pitch, duration,
voice and MusicXML composition remain out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Final

from .runtime_geometry_engine_contract import BoxContract, PageGeometryContract
from .runtime_specialist_evidence_v1 import SpecialistEvidenceBatch, SpecialistObservation


DETERMINISTIC_RESOLVER_V1_VERSION: Final[str] = "runtime-deterministic-resolver-v1"
MAX_ACCIDENTAL_NOTE_DISTANCE_SPACINGS_MILLI: Final[int] = 3000

R01_METER_CONFLICT: Final[str] = "R01_METER_CONFLICT"
R02_ACCIDENTAL_TIE: Final[str] = "R02_ACCIDENTAL_TIE"
R03_ACCIDENTAL_CONFLICT: Final[str] = "R03_ACCIDENTAL_CONFLICT"
R04_ACCIDENTAL_UNASSOCIATED: Final[str] = "R04_ACCIDENTAL_UNASSOCIATED"
R05_SPECIALIST_AMBIGUOUS: Final[str] = "R05_SPECIALIST_AMBIGUOUS"

_REASON_PREFIX_ORDER: Final[tuple[str, ...]] = (
    R01_METER_CONFLICT,
    R02_ACCIDENTAL_TIE,
    R03_ACCIDENTAL_CONFLICT,
    R04_ACCIDENTAL_UNASSOCIATED,
    R05_SPECIALIST_AMBIGUOUS,
)


@dataclass(frozen=True, slots=True)
class ResolvedNoteEvidence:
    notehead_observation_id: str
    notehead_class: str
    bbox: BoxContract
    accidental_class: str | None = None
    accidental_observation_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedRestEvidence:
    rest_observation_id: str
    rest_class: str
    bbox: BoxContract


@dataclass(frozen=True, slots=True)
class ResolvedMeasureEvidence:
    measure_id: str
    staff_id: str
    status: str
    meter_class: str | None
    notes: tuple[ResolvedNoteEvidence, ...]
    rests: tuple[ResolvedRestEvidence, ...]
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"accepted", "ambiguous"}:
            raise ValueError("resolved measure status must be accepted or ambiguous")
        if self.status == "accepted" and self.reasons:
            raise ValueError("accepted resolved measure cannot carry ambiguity reasons")
        if self.status == "ambiguous" and not self.reasons:
            raise ValueError("ambiguous resolved measure must explain why")


@dataclass(frozen=True, slots=True)
class DeterministicResolverResult:
    status: str
    config_fingerprint: str
    measures: tuple[ResolvedMeasureEvidence, ...]

    def __post_init__(self) -> None:
        if self.status not in {"accepted", "ambiguous"}:
            raise ValueError("resolver status must be accepted or ambiguous")
        if len(self.config_fingerprint) != 64:
            raise ValueError("resolver config fingerprint must be SHA-256 length")
        ids = tuple(item.measure_id for item in self.measures)
        if len(set(ids)) != len(ids):
            raise ValueError("resolved measure ids must be unique")
        expected = "ambiguous" if any(item.status == "ambiguous" for item in self.measures) else "accepted"
        if self.status != expected:
            raise ValueError("resolver status must summarize measure statuses")


def deterministic_resolver_v1_config_fingerprint(geometry_fingerprint: str) -> str:
    payload = {
        "version": DETERMINISTIC_RESOLVER_V1_VERSION,
        "geometry_fingerprint": geometry_fingerprint,
        "max_accidental_note_distance_spacings_milli": MAX_ACCIDENTAL_NOTE_DISTANCE_SPACINGS_MILLI,
        "association": "same-measure-same-staff-nearest-following-notehead-fail-on-tie-v1",
        "measure_order": "geometry-system-order-x-then-system-staff-order-v1",
        "pitch_composition": False,
        "duration_composition": False,
        "musicxml_generation": False,
        "stage7d10_access": False,
        "stage7d13_access": False,
        "checkpoint_access": False,
        "test_split_access": False,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return sha256(raw).hexdigest()


def _center_x(box: BoxContract) -> float:
    return (box.x_min + box.x_max) / 2.0


def _box_inside(inner: BoxContract, outer: BoxContract) -> bool:
    tolerance = 1e-9
    return (
        inner.x_min >= outer.x_min - tolerance
        and inner.y_min >= outer.y_min - tolerance
        and inner.x_max <= outer.x_max + tolerance
        and inner.y_max <= outer.y_max + tolerance
    )


def _reason_sort_key(reason: str) -> tuple[int, str]:
    for index, prefix in enumerate(_REASON_PREFIX_ORDER):
        if reason.startswith(prefix):
            return index, reason
    return len(_REASON_PREFIX_ORDER), reason


def _canonical_measure_order(geometry: PageGeometryContract):
    system_order = {system.system_id: index for index, system in enumerate(geometry.systems)}
    if len(system_order) != len(geometry.systems):
        raise ValueError("resolver geometry has duplicate system ids")
    staff_order: dict[str, tuple[int, int]] = {}
    for system_index, system in enumerate(geometry.systems):
        for member_index, staff_id in enumerate(system.staff_ids):
            if staff_id in staff_order:
                raise ValueError("resolver geometry assigns a staff to multiple systems")
            staff_order[staff_id] = (system_index, member_index)
    if set(staff_order) != {staff.staff_id for staff in geometry.staffs}:
        raise ValueError("resolver geometry system/staff membership is incomplete")
    fallback = len(system_order) + 1
    return tuple(
        sorted(
            geometry.measure_proposals,
            key=lambda item: (
                system_order.get(item.system_id, fallback),
                item.bbox.x_min,
                staff_order.get(item.staff_id, (fallback, fallback))[1],
                item.measure_id,
            ),
        )
    )


def _resolve_measure(
    measure,
    staff,
    observations: tuple[SpecialistObservation, ...],
) -> ResolvedMeasureEvidence:
    accepted = tuple(item for item in observations if item.status == "accepted")
    ambiguous = tuple(item for item in observations if item.status == "ambiguous")
    reasons: list[str] = []

    for item in accepted:
        if item.staff_id != measure.staff_id:
            raise ValueError("specialist observation staff does not match measure")
        if item.bbox is not None and not _box_inside(item.bbox, measure.bbox):
            raise ValueError("accepted specialist bbox lies outside its measure")

    for task in sorted({item.task for item in ambiguous}):
        reasons.append(f"{R05_SPECIALIST_AMBIGUOUS}_{task.upper()}")

    meters = tuple(item for item in accepted if item.task == "meter")
    if len(meters) == 0:
        meter_class: str | None = None
    elif len(meters) == 1:
        meter_class = meters[0].class_label
    else:
        meter_class = None
        reasons.append(R01_METER_CONFLICT)

    note_observations = tuple(
        sorted(
            (item for item in accepted if item.task == "notehead"),
            key=lambda item: (_center_x(item.bbox), item.observation_id),
        )
    )
    rest_observations = tuple(
        sorted(
            (item for item in accepted if item.task == "rest"),
            key=lambda item: (_center_x(item.bbox), item.observation_id),
        )
    )
    accidental_observations = tuple(
        sorted(
            (item for item in accepted if item.task == "accidental"),
            key=lambda item: (_center_x(item.bbox), item.observation_id),
        )
    )

    association_candidates: dict[str, list[SpecialistObservation]] = {
        note.observation_id: [] for note in note_observations
    }
    max_distance = staff.staff_spacing * MAX_ACCIDENTAL_NOTE_DISTANCE_SPACINGS_MILLI / 1000.0
    for accidental in accidental_observations:
        possible: list[tuple[float, SpecialistObservation]] = []
        accidental_center = _center_x(accidental.bbox)
        for note in note_observations:
            note_center = _center_x(note.bbox)
            if note_center <= accidental_center:
                continue
            distance = max(0.0, note.bbox.x_min - accidental.bbox.x_max)
            if distance <= max_distance:
                possible.append((distance, note))
        if not possible:
            reasons.append(R04_ACCIDENTAL_UNASSOCIATED)
            continue
        possible.sort(key=lambda item: (item[0], item[1].observation_id))
        best_distance = possible[0][0]
        tied = tuple(item for item in possible if abs(item[0] - best_distance) <= 1e-9)
        if len(tied) != 1:
            reasons.append(R02_ACCIDENTAL_TIE)
            continue
        association_candidates[tied[0][1].observation_id].append(accidental)

    notes: list[ResolvedNoteEvidence] = []
    for note in note_observations:
        assigned = association_candidates[note.observation_id]
        accidental_class: str | None = None
        accidental_id: str | None = None
        if len(assigned) == 1:
            accidental_class = assigned[0].class_label
            accidental_id = assigned[0].observation_id
        elif len(assigned) > 1:
            reasons.append(R03_ACCIDENTAL_CONFLICT)
        notes.append(
            ResolvedNoteEvidence(
                notehead_observation_id=note.observation_id,
                notehead_class=str(note.class_label),
                bbox=note.bbox,
                accidental_class=accidental_class,
                accidental_observation_id=accidental_id,
            )
        )

    rests = tuple(
        ResolvedRestEvidence(
            rest_observation_id=item.observation_id,
            rest_class=str(item.class_label),
            bbox=item.bbox,
        )
        for item in rest_observations
    )
    ordered_reasons = tuple(sorted(set(reasons), key=_reason_sort_key))
    status = "ambiguous" if ordered_reasons else "accepted"
    return ResolvedMeasureEvidence(
        measure_id=measure.measure_id,
        staff_id=measure.staff_id,
        status=status,
        meter_class=meter_class,
        notes=tuple(notes),
        rests=rests,
        reasons=ordered_reasons,
    )


def resolve_specialist_evidence_v1(
    geometry: PageGeometryContract,
    evidence: SpecialistEvidenceBatch,
) -> DeterministicResolverResult:
    """Resolve specialist evidence against accepted runtime measure geometry."""
    if not isinstance(geometry, PageGeometryContract):
        raise TypeError("geometry must be PageGeometryContract")
    if not isinstance(evidence, SpecialistEvidenceBatch):
        raise TypeError("evidence must be SpecialistEvidenceBatch")
    if geometry.status != "accepted" or not geometry.measure_proposals:
        raise ValueError("deterministic resolver requires accepted measure geometry")

    measure_by_id = {item.measure_id: item for item in geometry.measure_proposals}
    if len(measure_by_id) != len(geometry.measure_proposals):
        raise ValueError("resolver geometry has duplicate measure ids")
    staff_by_id = {item.staff_id: item for item in geometry.staffs}
    if len(staff_by_id) != len(geometry.staffs):
        raise ValueError("resolver geometry has duplicate staff ids")
    for observation in evidence.observations:
        measure = measure_by_id.get(observation.measure_id)
        if measure is None:
            raise ValueError("specialist observation references unknown measure")
        if observation.staff_id != measure.staff_id:
            raise ValueError("specialist observation references wrong staff for measure")

    measures: list[ResolvedMeasureEvidence] = []
    for measure in _canonical_measure_order(geometry):
        if measure.system_id not in {system.system_id for system in geometry.systems}:
            raise ValueError("resolver measure references unknown system")
        if measure.staff_id not in staff_by_id:
            raise ValueError("resolver measure references unknown staff")
        observations = evidence.for_measure(measure.measure_id)
        measures.append(_resolve_measure(measure, staff_by_id[measure.staff_id], observations))

    status = "ambiguous" if any(item.status == "ambiguous" for item in measures) else "accepted"
    return DeterministicResolverResult(
        status=status,
        config_fingerprint=deterministic_resolver_v1_config_fingerprint(geometry.geometry_config_fingerprint),
        measures=tuple(measures),
    )
