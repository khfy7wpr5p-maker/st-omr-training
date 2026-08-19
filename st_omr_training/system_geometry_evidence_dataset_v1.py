"""Fixture/synthetic-only dataset surface for SystemGeometry Evidence v1.

The pilot freezes renderer-authoritative SAME_SYSTEM and DIFFERENT_SYSTEM pair
records from System Geometry Evidence Extractor reports.  It does not define a
runtime grouping rule and has no TRAIN/VALIDATION/TEST dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Final, Iterable

from .system_geometry_evidence_extractor_v1 import (
    ExtractedSystemTopologyV1,
    SystemGeometryExtractorReportV1,
)
from .system_geometry_evidence_v1 import StaffSystemRelation


SYSTEM_GEOMETRY_EVIDENCE_DATASET_VERSION: Final[str] = (
    "system-geometry-evidence-dataset-v1"
)
SYSTEM_GEOMETRY_EVIDENCE_DATASET_SOURCE: Final[str] = (
    "fixture_synthetic_pinned_verovio_svg_only"
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


class SystemGeometryEvidenceDatasetError(ValueError):
    """Raised when the fixture-only evidence dataset violates its pilot contract."""


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemGeometryEvidenceDatasetError(f"{name} must be non-empty text")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SystemGeometryEvidenceDatasetError(
            f"{name} must be exact lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class SystemGeometryPairDatasetRecordV1:
    """One renderer-authoritative staff-pair topology record."""

    source_page_id: str
    source_svg_sha256: str
    source_extractor_fingerprint: str
    staff_a_id: str
    staff_b_id: str
    system_a_id: str
    system_b_id: str
    relation: StaffSystemRelation
    system_a_index: int
    system_b_index: int
    system_a_staff_count: int
    system_b_staff_count: int
    system_a_measure_count: int
    system_b_measure_count: int
    system_a_grouping_tokens: tuple[str, ...]
    system_b_grouping_tokens: tuple[str, ...]
    system_a_barline_group_count: int
    system_b_barline_group_count: int

    def __post_init__(self) -> None:
        _text(self.source_page_id, "source_page_id")
        _sha(self.source_svg_sha256, "source_svg_sha256")
        _sha(self.source_extractor_fingerprint, "source_extractor_fingerprint")
        for name in ("staff_a_id", "staff_b_id", "system_a_id", "system_b_id"):
            _text(getattr(self, name), name)
        if self.staff_a_id >= self.staff_b_id:
            raise SystemGeometryEvidenceDatasetError(
                "staff pair ids must be canonical lexical order and distinct"
            )
        if not isinstance(self.relation, StaffSystemRelation):
            raise SystemGeometryEvidenceDatasetError(
                "relation must be StaffSystemRelation"
            )
        if self.relation is StaffSystemRelation.SAME_SYSTEM:
            if self.system_a_id != self.system_b_id or self.system_a_index != self.system_b_index:
                raise SystemGeometryEvidenceDatasetError(
                    "SAME_SYSTEM requires identical source system identity"
                )
        else:
            if self.system_a_id == self.system_b_id or self.system_a_index == self.system_b_index:
                raise SystemGeometryEvidenceDatasetError(
                    "DIFFERENT_SYSTEM requires distinct source system identity"
                )
        for name in (
            "system_a_index",
            "system_b_index",
            "system_a_staff_count",
            "system_b_staff_count",
            "system_a_measure_count",
            "system_b_measure_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise SystemGeometryEvidenceDatasetError(
                    f"{name} must be a positive integer"
                )
        for name in ("system_a_barline_group_count", "system_b_barline_group_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SystemGeometryEvidenceDatasetError(
                    f"{name} must be a non-negative integer"
                )
        for name in ("system_a_grouping_tokens", "system_b_grouping_tokens"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                raise SystemGeometryEvidenceDatasetError(f"{name} must be a tuple")
            if len(set(value)) != len(value):
                raise SystemGeometryEvidenceDatasetError(f"{name} must be unique")
            for token in value:
                _text(token, name)

    @property
    def system_order_gap(self) -> int:
        return abs(self.system_a_index - self.system_b_index)

    @property
    def adjacent_different_systems(self) -> bool:
        return (
            self.relation is StaffSystemRelation.DIFFERENT_SYSTEM
            and self.system_order_gap == 1
        )

    @property
    def record_id(self) -> str:
        raw = json.dumps(
            self.canonical_payload(include_record_id=False),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return sha256(raw).hexdigest()

    def canonical_payload(self, *, include_record_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "adjacent_different_systems": self.adjacent_different_systems,
            "relation": self.relation.value,
            "source_extractor_fingerprint": self.source_extractor_fingerprint,
            "source_page_id": self.source_page_id,
            "source_svg_sha256": self.source_svg_sha256,
            "staff_a_id": self.staff_a_id,
            "staff_b_id": self.staff_b_id,
            "system_a_barline_group_count": self.system_a_barline_group_count,
            "system_a_grouping_tokens": list(self.system_a_grouping_tokens),
            "system_a_id": self.system_a_id,
            "system_a_index": self.system_a_index,
            "system_a_measure_count": self.system_a_measure_count,
            "system_a_staff_count": self.system_a_staff_count,
            "system_b_barline_group_count": self.system_b_barline_group_count,
            "system_b_grouping_tokens": list(self.system_b_grouping_tokens),
            "system_b_id": self.system_b_id,
            "system_b_index": self.system_b_index,
            "system_b_measure_count": self.system_b_measure_count,
            "system_b_staff_count": self.system_b_staff_count,
            "system_order_gap": self.system_order_gap,
        }
        if include_record_id:
            payload["record_id"] = self.record_id
        return payload


@dataclass(frozen=True, slots=True)
class SystemGeometryEvidenceDatasetV1:
    dataset_id: str
    records: tuple[SystemGeometryPairDatasetRecordV1, ...]
    source: str = SYSTEM_GEOMETRY_EVIDENCE_DATASET_SOURCE

    def __post_init__(self) -> None:
        _text(self.dataset_id, "dataset_id")
        if self.source != SYSTEM_GEOMETRY_EVIDENCE_DATASET_SOURCE:
            raise SystemGeometryEvidenceDatasetError(
                "pilot dataset source must remain fixture/synthetic pinned-Verovio only"
            )
        if not isinstance(self.records, tuple) or not self.records:
            raise SystemGeometryEvidenceDatasetError("records must be a non-empty tuple")
        if not all(isinstance(x, SystemGeometryPairDatasetRecordV1) for x in self.records):
            raise SystemGeometryEvidenceDatasetError(
                "records must contain SystemGeometryPairDatasetRecordV1 only"
            )
        ids = [x.record_id for x in self.records]
        if len(set(ids)) != len(ids):
            raise SystemGeometryEvidenceDatasetError("dataset record ids must be unique")
        relations = {x.relation for x in self.records}
        if StaffSystemRelation.SAME_SYSTEM not in relations:
            raise SystemGeometryEvidenceDatasetError(
                "pilot surface requires SAME_SYSTEM multi-staff positive evidence"
            )
        if StaffSystemRelation.DIFFERENT_SYSTEM not in relations:
            raise SystemGeometryEvidenceDatasetError(
                "pilot surface requires DIFFERENT_SYSTEM negative evidence"
            )
        if not any(
            x.relation is StaffSystemRelation.SAME_SYSTEM
            and x.system_a_staff_count >= 2
            for x in self.records
        ):
            raise SystemGeometryEvidenceDatasetError(
                "pilot surface requires a same-system multi-staff positive"
            )
        if not any(x.adjacent_different_systems for x in self.records):
            raise SystemGeometryEvidenceDatasetError(
                "pilot surface requires an adjacent different-system negative"
            )

    def relation_counts(self) -> dict[str, int]:
        result = {relation.value: 0 for relation in StaffSystemRelation}
        for record in self.records:
            result[record.relation.value] += 1
        return result

    def canonical_payload(self) -> dict[str, object]:
        records = sorted(
            (x.canonical_payload() for x in self.records),
            key=lambda x: str(x["record_id"]),
        )
        return {
            "dataset_id": self.dataset_id,
            "records": records,
            "relation_counts": self.relation_counts(),
            "source": self.source,
            "version": SYSTEM_GEOMETRY_EVIDENCE_DATASET_VERSION,
        }

    def fingerprint(self) -> str:
        raw = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return sha256(raw).hexdigest()


def _system_owner_map(
    report: SystemGeometryExtractorReportV1,
) -> dict[str, tuple[int, ExtractedSystemTopologyV1]]:
    owner: dict[str, tuple[int, ExtractedSystemTopologyV1]] = {}
    for index, system in enumerate(report.systems, start=1):
        for staff_id in system.staff_instance_ids:
            if staff_id in owner:
                raise SystemGeometryEvidenceDatasetError(
                    "extractor report contains duplicate staff ownership"
                )
            owner[staff_id] = (index, system)
    return owner


def build_system_geometry_evidence_dataset_v1(
    *, dataset_id: str, reports: Iterable[SystemGeometryExtractorReportV1]
) -> SystemGeometryEvidenceDatasetV1:
    """Freeze extractor reports into a small renderer-authoritative pair surface."""

    _text(dataset_id, "dataset_id")
    reports_tuple = tuple(reports)
    if not reports_tuple:
        raise SystemGeometryEvidenceDatasetError("reports must be non-empty")
    if not all(isinstance(x, SystemGeometryExtractorReportV1) for x in reports_tuple):
        raise SystemGeometryEvidenceDatasetError(
            "reports must contain SystemGeometryExtractorReportV1 only"
        )
    page_ids = [x.page_id for x in reports_tuple]
    if len(set(page_ids)) != len(page_ids):
        raise SystemGeometryEvidenceDatasetError("source page ids must be unique")

    records: list[SystemGeometryPairDatasetRecordV1] = []
    for report in reports_tuple:
        owner = _system_owner_map(report)
        extractor_fingerprint = report.fingerprint()
        for relation in report.evidence_page.staff_pair_relations():
            try:
                index_a, system_a = owner[relation.staff_a_id]
                index_b, system_b = owner[relation.staff_b_id]
            except KeyError as exc:
                raise SystemGeometryEvidenceDatasetError(
                    "evidence relation references a staff missing from extractor systems"
                ) from exc
            records.append(
                SystemGeometryPairDatasetRecordV1(
                    source_page_id=report.page_id,
                    source_svg_sha256=report.source_svg_sha256,
                    source_extractor_fingerprint=extractor_fingerprint,
                    staff_a_id=relation.staff_a_id,
                    staff_b_id=relation.staff_b_id,
                    system_a_id=relation.system_a_id,
                    system_b_id=relation.system_b_id,
                    relation=relation.relation,
                    system_a_index=index_a,
                    system_b_index=index_b,
                    system_a_staff_count=len(system_a.staff_instance_ids),
                    system_b_staff_count=len(system_b.staff_instance_ids),
                    system_a_measure_count=len(system_a.measures),
                    system_b_measure_count=len(system_b.measures),
                    system_a_grouping_tokens=system_a.grouping_tokens,
                    system_b_grouping_tokens=system_b.grouping_tokens,
                    system_a_barline_group_count=system_a.barline_group_count,
                    system_b_barline_group_count=system_b.barline_group_count,
                )
            )

    return SystemGeometryEvidenceDatasetV1(
        dataset_id=dataset_id,
        records=tuple(records),
    )


def system_geometry_evidence_dataset_runtime_connection_allowed() -> bool:
    """Dataset pilot carries evidence only and never authorizes runtime grouping."""

    return False
