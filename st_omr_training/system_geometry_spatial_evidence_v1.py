"""Fixture-only raw spatial evidence for future deterministic System Geometry.

This module reuses the already-frozen D5 SVG coordinate/transform helpers but
never mutates historical D5/D6.  It records geometry only; it does not define a
system-grouping threshold or runtime rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import xml.etree.ElementTree as ET
from typing import Final, Iterable

from ._stage7d5_geometry_v1 import (
    Stage7D5GeometryError,
    _bbox_for_group,
    _class_tokens,
    _coordinate_root,
    _direct_staff_lines,
    _is_visible_object_group,
    _parse_line_path,
)
from .system_geometry_evidence_extractor_v1 import (
    SystemGeometryExtractorReportV1,
    extract_system_geometry_evidence_v1,
)
from .system_geometry_evidence_v1 import StaffSystemRelation


SYSTEM_GEOMETRY_SPATIAL_EVIDENCE_VERSION: Final[str] = (
    "system-geometry-spatial-evidence-v1"
)
SYSTEM_GEOMETRY_SPATIAL_EVIDENCE_CLAIM_BOUNDARY: Final[str] = (
    "FIXTURE_ONLY_RAW_SPATIAL_OBSERVATIONS_NO_GROUPING_RULE"
)


class SystemGeometrySpatialEvidenceError(ValueError):
    """Raised when spatial evidence cannot be extracted uniquely/fail-closed."""


def _finite(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SystemGeometrySpatialEvidenceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SystemGeometrySpatialEvidenceError(f"{name} must be finite")
    return result


def _round(value: float) -> float:
    return round(float(value), 9)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _element_id(element: ET.Element, fallback: str) -> str:
    value = element.attrib.get("id")
    if isinstance(value, str) and value.strip():
        return value
    return fallback


@dataclass(frozen=True, slots=True)
class SpatialBoxV1:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for name in ("x_min", "y_min", "x_max", "y_max"):
            _finite(getattr(self, name), name)
        if not self.x_min < self.x_max or not self.y_min < self.y_max:
            raise SystemGeometrySpatialEvidenceError("spatial box must be positive")

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def canonical_payload(self) -> dict[str, float]:
        return {
            "x_min": _round(self.x_min),
            "y_min": _round(self.y_min),
            "x_max": _round(self.x_max),
            "y_max": _round(self.y_max),
        }


@dataclass(frozen=True, slots=True)
class SpatialLineV1:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        for name in ("x0", "y0", "x1", "y1"):
            _finite(getattr(self, name), name)

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def y_min(self) -> float:
        return min(self.y0, self.y1)

    @property
    def y_max(self) -> float:
        return max(self.y0, self.y1)

    @property
    def vertical(self) -> bool:
        return math.isclose(self.x0, self.x1, abs_tol=1e-7)

    def canonical_payload(self) -> dict[str, float]:
        return {
            "x0": _round(self.x0),
            "y0": _round(self.y0),
            "x1": _round(self.x1),
            "y1": _round(self.y1),
        }


@dataclass(frozen=True, slots=True)
class StaffSpatialObservationV1:
    staff_instance_id: str
    system_id: str
    ordinal: int
    bbox: SpatialBoxV1
    staff_spacing: float
    center_y: float
    measure_x_spans: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.staff_instance_id or not self.system_id:
            raise SystemGeometrySpatialEvidenceError("staff/system ids must be non-empty")
        if not isinstance(self.ordinal, int) or isinstance(self.ordinal, bool) or self.ordinal < 1:
            raise SystemGeometrySpatialEvidenceError("staff ordinal must be positive")
        spacing = _finite(self.staff_spacing, "staff_spacing")
        if spacing <= 0:
            raise SystemGeometrySpatialEvidenceError("staff_spacing must be positive")
        _finite(self.center_y, "center_y")
        if not isinstance(self.measure_x_spans, tuple) or not self.measure_x_spans:
            raise SystemGeometrySpatialEvidenceError("measure_x_spans must be non-empty")
        for left, right in self.measure_x_spans:
            if not math.isfinite(left) or not math.isfinite(right) or not left < right:
                raise SystemGeometrySpatialEvidenceError("measure x span must be finite positive")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "bbox": self.bbox.canonical_payload(),
            "center_y": _round(self.center_y),
            "measure_x_spans": [[_round(a), _round(b)] for a, b in self.measure_x_spans],
            "ordinal": self.ordinal,
            "staff_instance_id": self.staff_instance_id,
            "staff_spacing": _round(self.staff_spacing),
            "system_id": self.system_id,
        }


@dataclass(frozen=True, slots=True)
class GroupingSpanObservationV1:
    svg_id: str
    tokens: tuple[str, ...]
    bbox: SpatialBoxV1 | None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "bbox": None if self.bbox is None else self.bbox.canonical_payload(),
            "svg_id": self.svg_id,
            "tokens": list(self.tokens),
        }


@dataclass(frozen=True, slots=True)
class MeasureSpatialObservationV1:
    measure_id: str
    vertical_barlines: tuple[SpatialLineV1, ...]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "measure_id": self.measure_id,
            "vertical_barlines": [line.canonical_payload() for line in self.vertical_barlines],
        }


@dataclass(frozen=True, slots=True)
class SystemSpatialObservationV1:
    system_id: str
    bbox: SpatialBoxV1
    staffs: tuple[StaffSpatialObservationV1, ...]
    grouping_spans: tuple[GroupingSpanObservationV1, ...]
    measures: tuple[MeasureSpatialObservationV1, ...]

    def canonical_payload(self) -> dict[str, object]:
        return {
            "bbox": self.bbox.canonical_payload(),
            "grouping_spans": [x.canonical_payload() for x in self.grouping_spans],
            "measures": [x.canonical_payload() for x in self.measures],
            "staffs": [x.canonical_payload() for x in self.staffs],
            "system_id": self.system_id,
        }


@dataclass(frozen=True, slots=True)
class StaffPairSpatialObservationV1:
    staff_a_id: str
    staff_b_id: str
    relation: StaffSystemRelation
    normalized_center_distance: float
    normalized_edge_gap: float
    x_overlap_ratio: float
    grouping_span_cover_count: int
    barline_span_cover_count: int
    measure_boundary_exact_match_fraction: float

    def canonical_payload(self) -> dict[str, object]:
        return {
            "barline_span_cover_count": self.barline_span_cover_count,
            "grouping_span_cover_count": self.grouping_span_cover_count,
            "measure_boundary_exact_match_fraction": _round(
                self.measure_boundary_exact_match_fraction
            ),
            "normalized_center_distance": _round(self.normalized_center_distance),
            "normalized_edge_gap": _round(self.normalized_edge_gap),
            "relation": self.relation.value,
            "staff_a_id": self.staff_a_id,
            "staff_b_id": self.staff_b_id,
            "x_overlap_ratio": _round(self.x_overlap_ratio),
        }


@dataclass(frozen=True, slots=True)
class SystemGeometrySpatialEvidenceReportV1:
    page_id: str
    source_svg_sha256: str
    source_topology_fingerprint: str
    systems: tuple[SystemSpatialObservationV1, ...]
    pair_observations: tuple[StaffPairSpatialObservationV1, ...]
    version: str = SYSTEM_GEOMETRY_SPATIAL_EVIDENCE_VERSION
    claim_boundary: str = SYSTEM_GEOMETRY_SPATIAL_EVIDENCE_CLAIM_BOUNDARY

    def canonical_payload(self) -> dict[str, object]:
        return {
            "claim_boundary": self.claim_boundary,
            "page_id": self.page_id,
            "pair_observations": [x.canonical_payload() for x in self.pair_observations],
            "source_svg_sha256": self.source_svg_sha256,
            "source_topology_fingerprint": self.source_topology_fingerprint,
            "systems": [x.canonical_payload() for x in self.systems],
            "version": self.version,
        }

    def fingerprint(self) -> str:
        raw = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return sha256(raw).hexdigest()


def _to_box(box: object) -> SpatialBoxV1:
    return SpatialBoxV1(
        float(getattr(box, "x_min")),
        float(getattr(box, "y_min")),
        float(getattr(box, "x_max")),
        float(getattr(box, "y_max")),
    )


def _safe_group_bbox(
    element: ET.Element,
    coordinate_root: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> SpatialBoxV1 | None:
    for content in (True, False):
        try:
            return _to_box(
                _bbox_for_group(
                    element,
                    content=content,
                    coordinate_root=coordinate_root,
                    parent_map=parent_map,
                )
            )
        except Stage7D5GeometryError:
            pass
    return None


def _vertical_barlines(
    measure: ET.Element,
    coordinate_root: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> tuple[SpatialLineV1, ...]:
    lines: list[SpatialLineV1] = []
    for group in measure.iter():
        if _local(group.tag) != "g":
            continue
        tokens = _class_tokens(group)
        if not ({"barLine", "barLineAttr"} & set(tokens)):
            continue
        if "bounding-box" in tokens or "content-bounding-box" in tokens:
            continue
        for element in group.iter():
            try:
                segment = _parse_line_path(element, coordinate_root, parent_map)
            except Stage7D5GeometryError:
                segment = None
            if segment is None:
                continue
            line = SpatialLineV1(
                segment.start.x,
                segment.start.y,
                segment.end.x,
                segment.end.y,
            )
            if line.vertical:
                lines.append(line)
    lines.sort(key=lambda x: (x.x_center, x.y_min, x.y_max))
    return tuple(lines)


def _aggregate_staff_slot(
    *,
    staff_instance_id: str,
    system_id: str,
    ordinal: int,
    per_measure_lines: list[tuple[object, ...]],
) -> StaffSpatialObservationV1:
    if not per_measure_lines:
        raise SystemGeometrySpatialEvidenceError("staff slot requires measure geometry")
    y_tops: list[float] = []
    y_bottoms: list[float] = []
    spacings: list[float] = []
    x_spans: list[tuple[float, float]] = []
    all_x: list[float] = []
    for lines in per_measure_lines:
        ys = [float(line.start.y) for line in lines]
        gaps = [ys[i + 1] - ys[i] for i in range(4)]
        if min(gaps) <= 0:
            raise SystemGeometrySpatialEvidenceError("staff lines must be ordered")
        spacing = sum(gaps) / len(gaps)
        if max(abs(gap - spacing) for gap in gaps) > 1e-6:
            raise SystemGeometrySpatialEvidenceError("staff spacing must be deterministic")
        x_values = [
            float(value)
            for line in lines
            for value in (line.start.x, line.end.x)
        ]
        left, right = min(x_values), max(x_values)
        y_tops.append(ys[0])
        y_bottoms.append(ys[-1])
        spacings.append(spacing)
        x_spans.append((left, right))
        all_x.extend(x_values)
    y_top = sum(y_tops) / len(y_tops)
    y_bottom = sum(y_bottoms) / len(y_bottoms)
    spacing = sum(spacings) / len(spacings)
    return StaffSpatialObservationV1(
        staff_instance_id=staff_instance_id,
        system_id=system_id,
        ordinal=ordinal,
        bbox=SpatialBoxV1(min(all_x), y_top, max(all_x), y_bottom),
        staff_spacing=spacing,
        center_y=(y_top + y_bottom) / 2.0,
        measure_x_spans=tuple(x_spans),
    )


def _extract_system_spatial(
    *,
    system: ET.Element,
    topology_system: object,
    coordinate_root: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> SystemSpatialObservationV1:
    system_id = str(getattr(topology_system, "system_id"))
    system_bbox = _safe_group_bbox(system, coordinate_root, parent_map)
    if system_bbox is None:
        raise SystemGeometrySpatialEvidenceError("system bbox unavailable")

    measure_groups = [
        element
        for element in system.iter()
        if _is_visible_object_group(element, "measure")
    ]
    measured: list[tuple[ET.Element, SpatialBoxV1]] = []
    for measure in measure_groups:
        bbox = _safe_group_bbox(measure, coordinate_root, parent_map)
        if bbox is None:
            raise SystemGeometrySpatialEvidenceError("measure bbox unavailable")
        measured.append((measure, bbox))
    measured.sort(key=lambda item: (item[1].x_min, item[1].y_min, item[0].attrib.get("id", "")))
    if not measured:
        raise SystemGeometrySpatialEvidenceError("system requires measures")

    expected_staff_count = len(getattr(topology_system, "staff_instance_ids"))
    slot_lines: list[list[tuple[object, ...]]] = [
        [] for _ in range(expected_staff_count)
    ]
    measure_observations: list[MeasureSpatialObservationV1] = []
    for measure_index, (measure, _) in enumerate(measured, start=1):
        staffs = [
            element
            for element in measure.iter()
            if _is_visible_object_group(element, "staff")
        ]
        staff_with_lines: list[tuple[float, tuple[object, ...]]] = []
        for staff in staffs:
            try:
                lines = _direct_staff_lines(staff, coordinate_root, parent_map)
            except Stage7D5GeometryError as exc:
                raise SystemGeometrySpatialEvidenceError(str(exc)) from exc
            center_y = (lines[0].start.y + lines[-1].start.y) / 2.0
            staff_with_lines.append((center_y, lines))
        staff_with_lines.sort(key=lambda item: item[0])
        if len(staff_with_lines) != expected_staff_count:
            raise SystemGeometrySpatialEvidenceError(
                "spatial/topology staff cardinality mismatch"
            )
        for ordinal, (_, lines) in enumerate(staff_with_lines):
            slot_lines[ordinal].append(lines)

        measure_id = _element_id(measure, f"{system_id}:measure:{measure_index}")
        measure_observations.append(
            MeasureSpatialObservationV1(
                measure_id=measure_id,
                vertical_barlines=_vertical_barlines(
                    measure, coordinate_root, parent_map
                ),
            )
        )

    staffs = tuple(
        _aggregate_staff_slot(
            staff_instance_id=getattr(topology_system, "staff_instance_ids")[ordinal - 1],
            system_id=system_id,
            ordinal=ordinal,
            per_measure_lines=slot_lines[ordinal - 1],
        )
        for ordinal in range(1, expected_staff_count + 1)
    )

    grouping: list[GroupingSpanObservationV1] = []
    for element in system.iter():
        if _local(element.tag) != "g":
            continue
        tokens = tuple(sorted(_class_tokens(element)))
        if not any(
            piece in token.lower()
            for token in tokens
            for piece in ("grpsym", "brace", "bracket")
        ):
            continue
        if "bounding-box" in tokens or "content-bounding-box" in tokens:
            continue
        grouping.append(
            GroupingSpanObservationV1(
                svg_id=_element_id(element, f"{system_id}:group:{len(grouping)+1}"),
                tokens=tokens,
                bbox=_safe_group_bbox(element, coordinate_root, parent_map),
            )
        )
    grouping.sort(key=lambda x: (x.svg_id, x.tokens))

    return SystemSpatialObservationV1(
        system_id=system_id,
        bbox=system_bbox,
        staffs=staffs,
        grouping_spans=tuple(grouping),
        measures=tuple(measure_observations),
    )


def _x_overlap_ratio(left: StaffSpatialObservationV1, right: StaffSpatialObservationV1) -> float:
    overlap = max(0.0, min(left.bbox.x_max, right.bbox.x_max) - max(left.bbox.x_min, right.bbox.x_min))
    denominator = min(left.bbox.width, right.bbox.width)
    if denominator <= 0:
        raise SystemGeometrySpatialEvidenceError("staff width must be positive")
    return overlap / denominator


def _measure_boundary_match_fraction(
    left: StaffSpatialObservationV1, right: StaffSpatialObservationV1
) -> float:
    count = max(len(left.measure_x_spans), len(right.measure_x_spans))
    if count == 0:
        return 0.0
    matches = 0
    for index in range(count):
        if index >= len(left.measure_x_spans) or index >= len(right.measure_x_spans):
            continue
        a = left.measure_x_spans[index]
        b = right.measure_x_spans[index]
        if math.isclose(a[0], b[0], abs_tol=1e-7) and math.isclose(a[1], b[1], abs_tol=1e-7):
            matches += 1
    return matches / count


def _pair_observation(
    *,
    left: StaffSpatialObservationV1,
    right: StaffSpatialObservationV1,
    relation: StaffSystemRelation,
    systems_by_id: dict[str, SystemSpatialObservationV1],
) -> StaffPairSpatialObservationV1:
    upper, lower = sorted((left, right), key=lambda item: item.center_y)
    spacing = (upper.staff_spacing + lower.staff_spacing) / 2.0
    if spacing <= 0:
        raise SystemGeometrySpatialEvidenceError("pair spacing must be positive")

    source_systems = {
        systems_by_id[left.system_id].system_id: systems_by_id[left.system_id],
        systems_by_id[right.system_id].system_id: systems_by_id[right.system_id],
    }
    grouping_cover = 0
    barline_cover = 0
    for system in source_systems.values():
        for grouping in system.grouping_spans:
            if grouping.bbox is not None and grouping.bbox.y_min <= upper.center_y <= grouping.bbox.y_max and grouping.bbox.y_min <= lower.center_y <= grouping.bbox.y_max:
                grouping_cover += 1
        for measure in system.measures:
            for line in measure.vertical_barlines:
                if line.y_min <= upper.center_y <= line.y_max and line.y_min <= lower.center_y <= line.y_max:
                    barline_cover += 1

    return StaffPairSpatialObservationV1(
        staff_a_id=min(left.staff_instance_id, right.staff_instance_id),
        staff_b_id=max(left.staff_instance_id, right.staff_instance_id),
        relation=relation,
        normalized_center_distance=abs(lower.center_y - upper.center_y) / spacing,
        normalized_edge_gap=(lower.bbox.y_min - upper.bbox.y_max) / spacing,
        x_overlap_ratio=_x_overlap_ratio(left, right),
        grouping_span_cover_count=grouping_cover,
        barline_span_cover_count=barline_cover,
        measure_boundary_exact_match_fraction=_measure_boundary_match_fraction(left, right),
    )


def extract_system_geometry_spatial_evidence_v1(
    *, page_id: str, svg: str | bytes
) -> SystemGeometrySpatialEvidenceReportV1:
    """Extract raw spatial fixture evidence without designing a grouping rule."""

    if isinstance(svg, str):
        svg_bytes = svg.encode("utf-8", errors="strict")
    elif isinstance(svg, bytes):
        svg_bytes = svg
    else:
        raise SystemGeometrySpatialEvidenceError("svg must be str or bytes")
    if not svg_bytes:
        raise SystemGeometrySpatialEvidenceError("svg must be non-empty")

    topology = extract_system_geometry_evidence_v1(page_id=page_id, svg=svg_bytes)
    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError as exc:
        raise SystemGeometrySpatialEvidenceError("malformed SVG") from exc
    try:
        coordinate_root, _ = _coordinate_root(root)
    except Stage7D5GeometryError as exc:
        raise SystemGeometrySpatialEvidenceError(str(exc)) from exc
    parent_map = {
        child: parent for parent in coordinate_root.iter() for child in parent
    }

    system_elements = [
        element
        for element in coordinate_root.iter()
        if _is_visible_object_group(element, "system")
    ]
    element_by_id = {
        element.attrib.get("id", ""): element
        for element in system_elements
        if element.attrib.get("id", "")
    }
    systems: list[SystemSpatialObservationV1] = []
    for topology_system in topology.systems:
        system_element = element_by_id.get(topology_system.system_id)
        if system_element is None:
            raise SystemGeometrySpatialEvidenceError(
                "topology system missing from spatial coordinate root"
            )
        systems.append(
            _extract_system_spatial(
                system=system_element,
                topology_system=topology_system,
                coordinate_root=coordinate_root,
                parent_map=parent_map,
            )
        )
    systems.sort(key=lambda x: (x.bbox.y_min, x.bbox.x_min, x.system_id))

    staff_by_id = {
        staff.staff_instance_id: staff
        for system in systems
        for staff in system.staffs
    }
    systems_by_id = {system.system_id: system for system in systems}
    pairs: list[StaffPairSpatialObservationV1] = []
    for relation in topology.evidence_page.staff_pair_relations():
        try:
            left = staff_by_id[relation.staff_a_id]
            right = staff_by_id[relation.staff_b_id]
        except KeyError as exc:
            raise SystemGeometrySpatialEvidenceError(
                "topology relation references missing spatial staff"
            ) from exc
        pairs.append(
            _pair_observation(
                left=left,
                right=right,
                relation=relation.relation,
                systems_by_id=systems_by_id,
            )
        )
    pairs.sort(key=lambda x: (x.staff_a_id, x.staff_b_id))

    return SystemGeometrySpatialEvidenceReportV1(
        page_id=page_id,
        source_svg_sha256=sha256(svg_bytes).hexdigest(),
        source_topology_fingerprint=topology.fingerprint(),
        systems=tuple(systems),
        pair_observations=tuple(pairs),
    )


@dataclass(frozen=True, slots=True)
class SystemGeometrySpatialStabilityAuditV1:
    report_fingerprints: tuple[str, ...]
    relation_counts: dict[str, int]
    feature_ranges_by_relation: dict[str, dict[str, tuple[float, float]]]
    overlapping_interval_features: tuple[str, ...]
    disjoint_interval_features_on_fixture_surface: tuple[str, ...]
    version: str = "system-geometry-spatial-stability-audit-v1"
    claim_boundary: str = SYSTEM_GEOMETRY_SPATIAL_EVIDENCE_CLAIM_BOUNDARY

    def canonical_payload(self) -> dict[str, object]:
        return {
            "claim_boundary": self.claim_boundary,
            "disjoint_interval_features_on_fixture_surface": list(
                self.disjoint_interval_features_on_fixture_surface
            ),
            "feature_ranges_by_relation": {
                feature: {
                    relation: [_round(bounds[0]), _round(bounds[1])]
                    for relation, bounds in by_relation.items()
                }
                for feature, by_relation in self.feature_ranges_by_relation.items()
            },
            "overlapping_interval_features": list(self.overlapping_interval_features),
            "relation_counts": self.relation_counts,
            "report_fingerprints": list(self.report_fingerprints),
            "version": self.version,
        }

    def fingerprint(self) -> str:
        raw = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return sha256(raw).hexdigest()


def audit_system_geometry_spatial_stability_v1(
    reports: Iterable[SystemGeometrySpatialEvidenceReportV1],
) -> SystemGeometrySpatialStabilityAuditV1:
    """Compare observed numeric ranges only; never fit a threshold or rule."""

    reports_tuple = tuple(reports)
    if not reports_tuple:
        raise SystemGeometrySpatialEvidenceError("spatial audit requires reports")
    pairs = [pair for report in reports_tuple for pair in report.pair_observations]
    relation_counts = {relation.value: 0 for relation in StaffSystemRelation}
    for pair in pairs:
        relation_counts[pair.relation.value] += 1
    if any(value == 0 for value in relation_counts.values()):
        raise SystemGeometrySpatialEvidenceError(
            "spatial audit requires SAME_SYSTEM and DIFFERENT_SYSTEM evidence"
        )

    feature_names = (
        "normalized_center_distance",
        "normalized_edge_gap",
        "x_overlap_ratio",
        "grouping_span_cover_count",
        "barline_span_cover_count",
        "measure_boundary_exact_match_fraction",
    )
    ranges: dict[str, dict[str, tuple[float, float]]] = {}
    overlap: list[str] = []
    disjoint: list[str] = []
    for feature in feature_names:
        by_relation: dict[str, tuple[float, float]] = {}
        for relation in StaffSystemRelation:
            values = [
                float(getattr(pair, feature))
                for pair in pairs
                if pair.relation is relation
            ]
            by_relation[relation.value] = (min(values), max(values))
        ranges[feature] = by_relation
        same = by_relation[StaffSystemRelation.SAME_SYSTEM.value]
        different = by_relation[StaffSystemRelation.DIFFERENT_SYSTEM.value]
        intervals_overlap = not (same[1] < different[0] or different[1] < same[0])
        (overlap if intervals_overlap else disjoint).append(feature)

    return SystemGeometrySpatialStabilityAuditV1(
        report_fingerprints=tuple(sorted(report.fingerprint() for report in reports_tuple)),
        relation_counts=relation_counts,
        feature_ranges_by_relation=ranges,
        overlapping_interval_features=tuple(sorted(overlap)),
        disjoint_interval_features_on_fixture_surface=tuple(sorted(disjoint)),
    )


def system_geometry_spatial_rule_design_allowed() -> bool:
    return False


def system_geometry_spatial_runtime_connection_allowed() -> bool:
    return False
