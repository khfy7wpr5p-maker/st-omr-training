"""Fixture-only SVG extractor for SystemGeometry Evidence v1.

The extractor reads pinned Verovio SVG topology and emits a versioned,
fail-closed evidence record.  It is deliberately isolated from historical
V1/D5/D6, runtime geometry, Meter, Resolver, training, and TEST.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import xml.etree.ElementTree as ET
from typing import Final

from .system_geometry_evidence_v1 import (
    SystemGeometryEvidencePageV1,
    SystemTopologyEvidenceV1,
)


SYSTEM_GEOMETRY_EVIDENCE_EXTRACTOR_VERSION: Final[str] = (
    "system-geometry-evidence-extractor-v1"
)
SYSTEM_GEOMETRY_EVIDENCE_EXTRACTOR_SOURCE: Final[str] = "pinned_verovio_svg"


class SystemGeometryEvidenceExtractorError(ValueError):
    """Raised when renderer topology cannot be extracted uniquely and safely."""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _tokens(element: ET.Element) -> tuple[str, ...]:
    return tuple(sorted(set(element.attrib.get("class", "").split())))


def _visible_groups(root: ET.Element, class_name: str) -> tuple[ET.Element, ...]:
    found: list[ET.Element] = []
    for element in root.iter():
        if _local(element.tag) != "g":
            continue
        tokens = set(_tokens(element))
        if class_name not in tokens:
            continue
        if "bounding-box" in tokens or "content-bounding-box" in tokens:
            continue
        found.append(element)
    return tuple(found)


def _element_id(element: ET.Element, fallback: str) -> str:
    value = element.attrib.get("id")
    if isinstance(value, str) and value.strip():
        return value
    return fallback


@dataclass(frozen=True, slots=True)
class MeasureTopologyObservationV1:
    measure_id: str
    staff_svg_ids: tuple[str, ...]
    barline_svg_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.measure_id, str) or not self.measure_id:
            raise SystemGeometryEvidenceExtractorError("measure_id must be non-empty")
        if not isinstance(self.staff_svg_ids, tuple) or not self.staff_svg_ids:
            raise SystemGeometryEvidenceExtractorError(
                "each measure must expose at least one visible staff"
            )
        if len(set(self.staff_svg_ids)) != len(self.staff_svg_ids):
            raise SystemGeometryEvidenceExtractorError(
                "measure staff SVG ids must be unique"
            )
        if not isinstance(self.barline_svg_ids, tuple):
            raise SystemGeometryEvidenceExtractorError("barline ids must be a tuple")
        if len(set(self.barline_svg_ids)) != len(self.barline_svg_ids):
            raise SystemGeometryEvidenceExtractorError(
                "barline SVG ids must be unique"
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "barline_svg_ids": list(self.barline_svg_ids),
            "measure_id": self.measure_id,
            "staff_svg_ids": list(self.staff_svg_ids),
        }


@dataclass(frozen=True, slots=True)
class ExtractedSystemTopologyV1:
    system_id: str
    staff_instance_ids: tuple[str, ...]
    grouping_tokens: tuple[str, ...]
    measures: tuple[MeasureTopologyObservationV1, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.system_id, str) or not self.system_id:
            raise SystemGeometryEvidenceExtractorError("system_id must be non-empty")
        if not isinstance(self.staff_instance_ids, tuple) or not self.staff_instance_ids:
            raise SystemGeometryEvidenceExtractorError(
                "system must expose at least one canonical staff instance"
            )
        if len(set(self.staff_instance_ids)) != len(self.staff_instance_ids):
            raise SystemGeometryEvidenceExtractorError(
                "canonical staff ids must be unique"
            )
        if not isinstance(self.measures, tuple) or not self.measures:
            raise SystemGeometryEvidenceExtractorError(
                "system must expose at least one measure"
            )
        expected = len(self.staff_instance_ids)
        if any(len(measure.staff_svg_ids) != expected for measure in self.measures):
            raise SystemGeometryEvidenceExtractorError(
                "visible staff count must stay constant across measures in one system"
            )
        if len(set(self.grouping_tokens)) != len(self.grouping_tokens):
            raise SystemGeometryEvidenceExtractorError(
                "grouping tokens must be unique"
            )

    @property
    def barline_group_count(self) -> int:
        return sum(len(measure.barline_svg_ids) for measure in self.measures)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "grouping_tokens": list(self.grouping_tokens),
            "measures": [measure.canonical_payload() for measure in self.measures],
            "staff_instance_ids": list(self.staff_instance_ids),
            "system_id": self.system_id,
        }


@dataclass(frozen=True, slots=True)
class SystemGeometryExtractorReportV1:
    page_id: str
    source_svg_sha256: str
    systems: tuple[ExtractedSystemTopologyV1, ...]
    evidence_page: SystemGeometryEvidencePageV1

    def canonical_payload(self) -> dict[str, object]:
        return {
            "evidence": self.evidence_page.canonical_payload(),
            "extractor_version": SYSTEM_GEOMETRY_EVIDENCE_EXTRACTOR_VERSION,
            "page_id": self.page_id,
            "source": SYSTEM_GEOMETRY_EVIDENCE_EXTRACTOR_SOURCE,
            "source_svg_sha256": self.source_svg_sha256,
            "systems": [system.canonical_payload() for system in self.systems],
        }

    def fingerprint(self) -> str:
        raw = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return sha256(raw).hexdigest()


def extract_system_geometry_evidence_v1(
    *, page_id: str, svg: str | bytes
) -> SystemGeometryExtractorReportV1:
    """Extract renderer-authoritative fixture topology without grouping heuristics."""

    if not isinstance(page_id, str) or not page_id.strip():
        raise SystemGeometryEvidenceExtractorError("page_id must be non-empty text")
    if isinstance(svg, str):
        svg_bytes = svg.encode("utf-8", errors="strict")
    elif isinstance(svg, bytes):
        svg_bytes = svg
    else:
        raise SystemGeometryEvidenceExtractorError("svg must be str or bytes")
    if not svg_bytes:
        raise SystemGeometryEvidenceExtractorError("svg must be non-empty")

    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError as exc:
        raise SystemGeometryEvidenceExtractorError("malformed SVG") from exc
    if _local(root.tag) != "svg":
        raise SystemGeometryEvidenceExtractorError("root must be SVG")

    system_elements = _visible_groups(root, "system")
    if not system_elements:
        raise SystemGeometryEvidenceExtractorError("no visible renderer systems")

    extracted_systems: list[ExtractedSystemTopologyV1] = []
    evidence_systems: list[SystemTopologyEvidenceV1] = []
    seen_canonical_staff: set[str] = set()

    for system_index, system in enumerate(system_elements, start=1):
        system_id = _element_id(system, f"system-{system_index}")
        measures = _visible_groups(system, "measure")
        if not measures:
            raise SystemGeometryEvidenceExtractorError(
                "renderer system has no visible measures"
            )

        measure_observations: list[MeasureTopologyObservationV1] = []
        staff_count: int | None = None
        for measure_index, measure in enumerate(measures, start=1):
            measure_id = _element_id(
                measure, f"{system_id}:measure:{measure_index}"
            )
            staffs = _visible_groups(measure, "staff")
            if not staffs:
                raise SystemGeometryEvidenceExtractorError(
                    "renderer measure has no visible staff"
                )
            if staff_count is None:
                staff_count = len(staffs)
            elif len(staffs) != staff_count:
                raise SystemGeometryEvidenceExtractorError(
                    "staff cardinality changes inside one renderer system"
                )

            staff_svg_ids = tuple(
                _element_id(staff, f"{measure_id}:staff-svg:{staff_index}")
                for staff_index, staff in enumerate(staffs, start=1)
            )
            barlines = tuple(
                element
                for element in measure.iter()
                if _local(element.tag) == "g"
                and ({"barLine", "barLineAttr"} & set(_tokens(element)))
                and "bounding-box" not in set(_tokens(element))
                and "content-bounding-box" not in set(_tokens(element))
            )
            barline_svg_ids = tuple(
                _element_id(barline, f"{measure_id}:barline:{index}")
                for index, barline in enumerate(barlines, start=1)
            )
            measure_observations.append(
                MeasureTopologyObservationV1(
                    measure_id=measure_id,
                    staff_svg_ids=staff_svg_ids,
                    barline_svg_ids=barline_svg_ids,
                )
            )

        assert staff_count is not None
        canonical_staff_ids = tuple(
            f"{system_id}:staff:{ordinal}" for ordinal in range(1, staff_count + 1)
        )
        overlap = seen_canonical_staff.intersection(canonical_staff_ids)
        if overlap:
            raise SystemGeometryEvidenceExtractorError(
                "canonical staff identity collision across systems"
            )
        seen_canonical_staff.update(canonical_staff_ids)

        class_tokens = sorted(
            {token for item in system.iter() for token in _tokens(item)}
        )
        grouping_tokens = tuple(
            token
            for token in class_tokens
            if any(
                piece in token.lower()
                for piece in ("brace", "bracket", "group", "grpsym")
            )
        )

        extracted = ExtractedSystemTopologyV1(
            system_id=system_id,
            staff_instance_ids=canonical_staff_ids,
            grouping_tokens=grouping_tokens,
            measures=tuple(measure_observations),
        )
        extracted_systems.append(extracted)
        evidence_systems.append(
            SystemTopologyEvidenceV1(
                system_id=system_id,
                staff_instance_ids=canonical_staff_ids,
                measure_ids=tuple(m.measure_id for m in measure_observations),
                grouping_tokens=grouping_tokens,
                barline_group_count=extracted.barline_group_count,
            )
        )

    source_sha = sha256(svg_bytes).hexdigest()
    evidence_page = SystemGeometryEvidencePageV1(
        page_id=page_id,
        source_svg_sha256=source_sha,
        systems=tuple(extracted_systems and evidence_systems),
    )
    return SystemGeometryExtractorReportV1(
        page_id=page_id,
        source_svg_sha256=source_sha,
        systems=tuple(extracted_systems),
        evidence_page=evidence_page,
    )


def system_geometry_evidence_extractor_runtime_connection_allowed() -> bool:
    """Fixture extractor pilot is never runtime/production authorization."""

    return False
