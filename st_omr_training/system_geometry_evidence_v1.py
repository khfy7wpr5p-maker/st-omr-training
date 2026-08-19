"""Fixture-only evidence contract for future deterministic System Geometry.

This module is intentionally isolated from the historical V1/D5/D6 runtime and
training surfaces.  It represents renderer-observed system membership without
making a runtime grouping decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import re
from typing import Final


SYSTEM_GEOMETRY_EVIDENCE_VERSION: Final[str] = "system-geometry-evidence-v1"
SYSTEM_GEOMETRY_EVIDENCE_COORDINATE_SPACE: Final[str] = "pinned_verovio_svg"
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


class SystemGeometryEvidenceError(ValueError):
    """Raised when fixture evidence violates the frozen pilot contract."""


class StaffSystemRelation(str, Enum):
    SAME_SYSTEM = "SAME_SYSTEM"
    DIFFERENT_SYSTEM = "DIFFERENT_SYSTEM"


def _require_nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemGeometryEvidenceError(f"{name} must be non-empty text")
    return value


def _require_unique_text_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise SystemGeometryEvidenceError(f"{name} must be a non-empty tuple")
    for item in value:
        _require_nonempty_text(item, name)
    if len(set(value)) != len(value):
        raise SystemGeometryEvidenceError(f"{name} must contain unique values")
    return value


@dataclass(frozen=True, slots=True)
class SystemTopologyEvidenceV1:
    """Ground-truth renderer membership for one visible system fixture."""

    system_id: str
    staff_instance_ids: tuple[str, ...]
    measure_ids: tuple[str, ...]
    grouping_tokens: tuple[str, ...] = ()
    barline_group_count: int = 0

    def __post_init__(self) -> None:
        _require_nonempty_text(self.system_id, "system_id")
        _require_unique_text_tuple(self.staff_instance_ids, "staff_instance_ids")
        _require_unique_text_tuple(self.measure_ids, "measure_ids")
        if not isinstance(self.grouping_tokens, tuple):
            raise SystemGeometryEvidenceError("grouping_tokens must be a tuple")
        for token in self.grouping_tokens:
            _require_nonempty_text(token, "grouping_tokens")
        if len(set(self.grouping_tokens)) != len(self.grouping_tokens):
            raise SystemGeometryEvidenceError("grouping_tokens must be unique")
        if (
            not isinstance(self.barline_group_count, int)
            or isinstance(self.barline_group_count, bool)
            or self.barline_group_count < 0
        ):
            raise SystemGeometryEvidenceError(
                "barline_group_count must be a non-negative integer"
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "barline_group_count": self.barline_group_count,
            "grouping_tokens": sorted(self.grouping_tokens),
            "measure_ids": sorted(self.measure_ids),
            "staff_instance_ids": sorted(self.staff_instance_ids),
            "system_id": self.system_id,
        }


@dataclass(frozen=True, slots=True)
class StaffPairRelationEvidenceV1:
    staff_a_id: str
    staff_b_id: str
    system_a_id: str
    system_b_id: str
    relation: StaffSystemRelation

    def __post_init__(self) -> None:
        for name in ("staff_a_id", "staff_b_id", "system_a_id", "system_b_id"):
            _require_nonempty_text(getattr(self, name), name)
        if self.staff_a_id >= self.staff_b_id:
            raise SystemGeometryEvidenceError(
                "staff pair ids must be canonical lexical order and distinct"
            )
        if not isinstance(self.relation, StaffSystemRelation):
            raise SystemGeometryEvidenceError("relation must be StaffSystemRelation")
        if self.relation is StaffSystemRelation.SAME_SYSTEM:
            if self.system_a_id != self.system_b_id:
                raise SystemGeometryEvidenceError(
                    "SAME_SYSTEM requires identical source system ids"
                )
        elif self.system_a_id == self.system_b_id:
            raise SystemGeometryEvidenceError(
                "DIFFERENT_SYSTEM requires distinct source system ids"
            )

    def canonical_payload(self) -> dict[str, str]:
        return {
            "relation": self.relation.value,
            "staff_a_id": self.staff_a_id,
            "staff_b_id": self.staff_b_id,
            "system_a_id": self.system_a_id,
            "system_b_id": self.system_b_id,
        }


@dataclass(frozen=True, slots=True)
class SystemGeometryEvidencePageV1:
    """One fixture page of renderer-authoritative topology evidence."""

    page_id: str
    source_svg_sha256: str
    systems: tuple[SystemTopologyEvidenceV1, ...]
    coordinate_space: str = SYSTEM_GEOMETRY_EVIDENCE_COORDINATE_SPACE

    def __post_init__(self) -> None:
        _require_nonempty_text(self.page_id, "page_id")
        if not isinstance(self.source_svg_sha256, str) or not _SHA256_RE.fullmatch(
            self.source_svg_sha256
        ):
            raise SystemGeometryEvidenceError(
                "source_svg_sha256 must be exact lowercase SHA-256"
            )
        if self.coordinate_space != SYSTEM_GEOMETRY_EVIDENCE_COORDINATE_SPACE:
            raise SystemGeometryEvidenceError(
                "fixture evidence must stay in pinned_verovio_svg coordinate space"
            )
        if not isinstance(self.systems, tuple) or not self.systems:
            raise SystemGeometryEvidenceError("systems must be a non-empty tuple")
        if not all(isinstance(item, SystemTopologyEvidenceV1) for item in self.systems):
            raise SystemGeometryEvidenceError(
                "systems must contain SystemTopologyEvidenceV1 only"
            )

        system_ids = [item.system_id for item in self.systems]
        if len(set(system_ids)) != len(system_ids):
            raise SystemGeometryEvidenceError("system ids must be page-unique")

        seen_staff: set[str] = set()
        for system in self.systems:
            overlap = seen_staff.intersection(system.staff_instance_ids)
            if overlap:
                raise SystemGeometryEvidenceError(
                    "a staff instance cannot belong to more than one system"
                )
            seen_staff.update(system.staff_instance_ids)

    def staff_pair_relations(self) -> tuple[StaffPairRelationEvidenceV1, ...]:
        owner: dict[str, str] = {}
        for system in self.systems:
            for staff_id in system.staff_instance_ids:
                owner[staff_id] = system.system_id

        staff_ids = sorted(owner)
        relations: list[StaffPairRelationEvidenceV1] = []
        for left_index, staff_a in enumerate(staff_ids):
            for staff_b in staff_ids[left_index + 1 :]:
                system_a = owner[staff_a]
                system_b = owner[staff_b]
                relation = (
                    StaffSystemRelation.SAME_SYSTEM
                    if system_a == system_b
                    else StaffSystemRelation.DIFFERENT_SYSTEM
                )
                relations.append(
                    StaffPairRelationEvidenceV1(
                        staff_a_id=staff_a,
                        staff_b_id=staff_b,
                        system_a_id=system_a,
                        system_b_id=system_b,
                        relation=relation,
                    )
                )
        return tuple(relations)

    def canonical_payload(self) -> dict[str, object]:
        systems = sorted(
            (system.canonical_payload() for system in self.systems),
            key=lambda item: str(item["system_id"]),
        )
        relations = [
            relation.canonical_payload() for relation in self.staff_pair_relations()
        ]
        return {
            "coordinate_space": self.coordinate_space,
            "page_id": self.page_id,
            "pair_relations": relations,
            "source_svg_sha256": self.source_svg_sha256,
            "systems": systems,
            "version": SYSTEM_GEOMETRY_EVIDENCE_VERSION,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return sha256(encoded).hexdigest()


def system_geometry_evidence_runtime_connection_allowed() -> bool:
    """The fixture-only pilot is never a runtime/production authorization."""

    return False
