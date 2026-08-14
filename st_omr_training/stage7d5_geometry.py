"""Stage 7-D5 deterministic StaffSet + StructureSet geometry pilot.

This module derives synthetic spatial ground truth from the same pinned Verovio
layout used by ST-OMR, with invisible SVG bounding-box instrumentation. It does
not train a model, load a checkpoint, or expose dataset/Test split paths.

The historical Stage-7-D4 declarative contract used the scalar label
``barline_x``. D5 corrects that representation for the operational
``final_png_pixels`` coordinate space: after rotation a barline is a line
segment, not a single x coordinate. The frozen D4 fingerprint is not rewritten;
D5 records the versioned correction explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
import xml.etree.ElementTree as ET
from typing import Final

from .musicxml_roundtrip import parse_supported_v1_musicxml_projection
from .musicxml_validator import validate_musicxml
from .musicxml_writer import musicxml_sha256
from .renderer import (
    MAX_RENDER_PAGES,
    RENDERER_ADAPTER_VERSION,
    RENDERER_NAME,
    RendererConfig,
    RendererUnavailableError,
    RenderExecutionError,
    RenderInputError,
    _load_verovio_runtime,
    _validate_svg,
    renderer_config_fingerprint,
)


STAGE7D5_GEOMETRY_VERSION: Final[str] = "stage7d5-staff-structure-geometry-v1"
STAGE7D5_GEOMETRY_RENDERER_VERSION: Final[str] = "stage7d5-verovio-bbox-instrumentation-v1"
STAGE7D5_TRANSFORM_VERSION: Final[str] = "stage7d5-final-png-transform-v1"

D5_STRUCTURE_LABELS: Final[tuple[str, ...]] = (
    "system_id",
    "system_bbox",
    "measure_id",
    "measure_bbox",
    "barline_segment",
    "clef_g2_bbox",
    "meter_bbox",
    "meter_class",
)
D5_D4_LABEL_CORRECTION: Final[dict[str, str]] = {
    "superseded_label": "barline_x",
    "replacement_label": "barline_segment",
    "reason": "final_png_rotation_makes_a_barline_nonvertical",
}

_SVG_BBOX_OPTIONS: Final[dict[str, bool]] = {
    "svgBoundingBoxes": True,
    "svgContentBoundingBoxes": True,
}
_LINE_RE = re.compile(
    r"^\s*M\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))[\s,]+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*L\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))[\s,]+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)


class Stage7D5GeometryError(ValueError):
    """Raised when geometry cannot be derived uniquely and safely."""


@dataclass(frozen=True, slots=True)
class Point2D:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in (self.x, self.y)
        ):
            raise Stage7D5GeometryError("point coordinates must be finite numbers")


@dataclass(frozen=True, slots=True)
class AxisAlignedBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in values
        ):
            raise Stage7D5GeometryError("box coordinates must be finite numbers")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise Stage7D5GeometryError("box must have positive width and height")

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass(frozen=True, slots=True)
class LineSegment:
    start: Point2D
    end: Point2D

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise Stage7D5GeometryError("line segment must have nonzero length")

    @property
    def length(self) -> float:
        return math.hypot(self.end.x - self.start.x, self.end.y - self.start.y)


@dataclass(frozen=True, slots=True)
class StaffInstanceGeometry:
    staff_instance_id: str
    system_id: str
    five_staff_lines: tuple[LineSegment, ...]
    staff_instance_bbox: AxisAlignedBox
    staff_spacing: float

    def __post_init__(self) -> None:
        if not self.staff_instance_id or not self.system_id:
            raise Stage7D5GeometryError("staff/system ids must be non-empty")
        if len(self.five_staff_lines) != 5:
            raise Stage7D5GeometryError("V1 StaffSet requires exactly five staff lines")
        if not math.isfinite(self.staff_spacing) or self.staff_spacing <= 0:
            raise Stage7D5GeometryError("staff spacing must be positive and finite")


@dataclass(frozen=True, slots=True)
class MeasureGeometry:
    measure_id: str
    measure_number: int
    system_id: str
    measure_bbox: AxisAlignedBox
    barline_segment: LineSegment
    clef_g2_bbox: AxisAlignedBox | None
    meter_bbox: AxisAlignedBox | None
    meter_class: str

    def __post_init__(self) -> None:
        if not self.measure_id or not self.system_id:
            raise Stage7D5GeometryError("measure/system ids must be non-empty")
        if (
            not isinstance(self.measure_number, int)
            or isinstance(self.measure_number, bool)
            or self.measure_number < 1
        ):
            raise Stage7D5GeometryError("measure_number must be a positive integer")
        if self.meter_class not in {"2/4", "3/4", "4/4"}:
            raise Stage7D5GeometryError("meter_class must stay inside the V1 surface")


@dataclass(frozen=True, slots=True)
class SystemGeometry:
    system_id: str
    system_bbox: AxisAlignedBox
    staff_instance_id: str
    measure_numbers: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.system_id or not self.staff_instance_id:
            raise Stage7D5GeometryError("system/staff ids must be non-empty")
        if not self.measure_numbers:
            raise Stage7D5GeometryError("system must contain at least one measure")


@dataclass(frozen=True, slots=True)
class PageGeometry:
    page_number: int
    coordinate_space: str
    view_box: tuple[float, float, float, float]
    source_musicxml_sha256: str
    base_renderer_config_fingerprint: str
    geometry_instrumentation_fingerprint: str
    geometry_svg_sha256: str
    systems: tuple[SystemGeometry, ...]
    staff_instances: tuple[StaffInstanceGeometry, ...]
    measures: tuple[MeasureGeometry, ...]
    geometry_transform_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.page_number, int)
            or isinstance(self.page_number, bool)
            or self.page_number < 1
        ):
            raise Stage7D5GeometryError("page_number must be positive")
        if self.coordinate_space not in {"pinned_verovio_svg", "final_png_pixels"}:
            raise Stage7D5GeometryError("unsupported geometry coordinate space")
        if len(self.view_box) != 4:
            raise Stage7D5GeometryError("view_box must have four numbers")
        if not self.systems or not self.staff_instances or not self.measures:
            raise Stage7D5GeometryError(
                "page geometry must contain systems, staff instances and measures"
            )


@dataclass(frozen=True, slots=True)
class GeometryRenderedPage:
    page_number: int
    svg: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class GeometryRenderResult:
    source_musicxml_sha256: str
    renderer_name: str
    renderer_package_version: str
    renderer_runtime_version: str
    renderer_adapter_version: str
    base_renderer_config_fingerprint: str
    geometry_instrumentation_fingerprint: str
    pages: tuple[GeometryRenderedPage, ...]


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def geometry_instrumentation_fingerprint(config: RendererConfig) -> str:
    if not isinstance(config, RendererConfig):
        raise TypeError("config must be RendererConfig")
    return _canonical_sha256(
        {
            "version": STAGE7D5_GEOMETRY_RENDERER_VERSION,
            "base_renderer_adapter_version": RENDERER_ADAPTER_VERSION,
            "base_renderer_config_fingerprint": renderer_config_fingerprint(config),
            "instrumentation_options": _SVG_BBOX_OPTIONS,
        }
    )


def render_musicxml_geometry_svg(
    data: object, config: RendererConfig | None = None
) -> GeometryRenderResult:
    """Render the same pinned layout with invisible bbox instrumentation enabled."""

    validation = validate_musicxml(data)
    if not validation.is_valid:
        raise RenderInputError("MusicXML failed Stage 2-C validation", validation)
    assert isinstance(data, bytes)

    effective = RendererConfig() if config is None else config
    if not isinstance(effective, RendererConfig):
        raise TypeError("config must be RendererConfig")

    verovio, package_version = _load_verovio_runtime()
    options = dict(effective.verovio_options())
    options.update(_SVG_BBOX_OPTIONS)
    try:
        toolkit = verovio.toolkit()
        runtime_version = str(toolkit.getVersion())
        if not runtime_version.startswith(package_version):
            raise RendererUnavailableError(
                f"Verovio runtime version mismatch: {runtime_version!r}"
            )
        if toolkit.setInputFrom("xml") is False:
            raise RenderExecutionError("Verovio rejected explicit MusicXML input mode")
        if toolkit.setOptions(options) is False:
            raise RenderExecutionError(
                "Verovio rejected D5 geometry instrumentation options"
            )
        xml_text = data.decode("utf-8", errors="strict")
        if toolkit.loadData(xml_text) is False:
            raise RenderExecutionError("Verovio rejected validated MusicXML input")
        page_count = toolkit.getPageCount()
    except (RendererUnavailableError, RenderInputError, RenderExecutionError):
        raise
    except UnicodeDecodeError as exc:
        raise RenderInputError("validated MusicXML is not UTF-8", validation) from exc
    except Exception as exc:
        raise RenderExecutionError(
            f"Verovio geometry render failed: {type(exc).__name__}"
        ) from exc

    if (
        not isinstance(page_count, int)
        or isinstance(page_count, bool)
        or not 1 <= page_count <= MAX_RENDER_PAGES
    ):
        raise RenderExecutionError(
            "Verovio geometry page count is outside the frozen boundary"
        )

    pages: list[GeometryRenderedPage] = []
    for page_number in range(1, page_count + 1):
        try:
            svg_text = toolkit.renderToSVG(page_number, True)
        except Exception as exc:
            raise RenderExecutionError(
                f"Verovio geometry render failed on page {page_number}: {type(exc).__name__}"
            ) from exc
        svg = _validate_svg(svg_text, page_number)
        pages.append(
            GeometryRenderedPage(page_number, svg, sha256(svg).hexdigest())
        )

    return GeometryRenderResult(
        source_musicxml_sha256=musicxml_sha256(data),
        renderer_name=RENDERER_NAME,
        renderer_package_version=package_version,
        renderer_runtime_version=runtime_version,
        renderer_adapter_version=RENDERER_ADAPTER_VERSION,
        base_renderer_config_fingerprint=renderer_config_fingerprint(effective),
        geometry_instrumentation_fingerprint=geometry_instrumentation_fingerprint(
            effective
        ),
        pages=tuple(pages),
    )


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _class_tokens(element: ET.Element) -> frozenset[str]:
    return frozenset(element.attrib.get("class", "").split())


def _is_visible_object_group(element: ET.Element, class_name: str) -> bool:
    tokens = _class_tokens(element)
    return (
        _local(element.tag) == "g"
        and class_name in tokens
        and "bounding-box" not in tokens
        and "content-bounding-box" not in tokens
    )


def _parse_view_box(root: ET.Element) -> tuple[float, float, float, float]:
    text = root.attrib.get("viewBox")
    if not text:
        raise Stage7D5GeometryError("geometry SVG requires an explicit viewBox")
    parts = text.replace(",", " ").split()
    if len(parts) != 4:
        raise Stage7D5GeometryError("viewBox must contain four values")
    try:
        x0, y0, width, height = (float(v) for v in parts)
    except ValueError as exc:
        raise Stage7D5GeometryError("viewBox must be numeric") from exc
    if (
        not all(math.isfinite(v) for v in (x0, y0, width, height))
        or width <= 0
        or height <= 0
    ):
        raise Stage7D5GeometryError("viewBox is not finite and positive")
    return x0, y0, width, height


def _bbox_for_group(group: ET.Element, *, content: bool) -> AxisAlignedBox:
    object_id = group.attrib.get("id")
    if not object_id:
        raise Stage7D5GeometryError("renderer object group is missing id")
    prefix = "cbbox-" if content else "bbox-"
    expected_id = prefix + object_id
    expected_class = "content-bounding-box" if content else "bounding-box"
    matches = [
        element
        for element in group.iter()
        if _local(element.tag) == "g"
        and element.attrib.get("id") == expected_id
        and expected_class in _class_tokens(element)
    ]
    if len(matches) != 1:
        raise Stage7D5GeometryError(f"{expected_id} must be present exactly once")
    rects = [child for child in matches[0] if _local(child.tag) == "rect"]
    if len(rects) != 1:
        raise Stage7D5GeometryError(
            f"{expected_id} must contain exactly one direct rect"
        )
    rect = rects[0]
    try:
        x = float(rect.attrib["x"])
        y = float(rect.attrib["y"])
        width = float(rect.attrib["width"])
        height = float(rect.attrib["height"])
    except (KeyError, ValueError) as exc:
        raise Stage7D5GeometryError(
            f"{expected_id} has invalid rect geometry"
        ) from exc
    return AxisAlignedBox(x, y, x + width, y + height)


def _parse_line_path(path: ET.Element) -> LineSegment | None:
    if _local(path.tag) != "path":
        return None
    match = _LINE_RE.fullmatch(path.attrib.get("d", ""))
    if match is None:
        return None
    values = tuple(float(value) for value in match.groups())
    if not all(math.isfinite(value) for value in values):
        return None
    return LineSegment(
        Point2D(values[0], values[1]), Point2D(values[2], values[3])
    )


def _direct_staff_lines(staff: ET.Element) -> tuple[LineSegment, ...]:
    candidates: list[LineSegment] = []
    for child in staff:
        segment = _parse_line_path(child)
        if segment is None:
            continue
        if not math.isclose(segment.start.y, segment.end.y, abs_tol=1e-9):
            continue
        candidates.append(segment)
    if len(candidates) != 5:
        raise Stage7D5GeometryError(
            "V1 staff group must expose exactly five direct horizontal staff-line paths"
        )
    candidates.sort(key=lambda line: line.start.y)
    return tuple(candidates)


def _trailing_barline(measure: ET.Element) -> LineSegment:
    candidates: list[LineSegment] = []
    for element in measure.iter():
        if _local(element.tag) != "g":
            continue
        tokens = _class_tokens(element)
        if "bounding-box" in tokens or "content-bounding-box" in tokens:
            continue
        if not any(token in {"barLine", "barLineAttr"} for token in tokens):
            continue
        for child in element:
            segment = _parse_line_path(child)
            if segment is None:
                continue
            if math.isclose(segment.start.x, segment.end.x, abs_tol=1e-9):
                candidates.append(segment)
    if not candidates:
        raise Stage7D5GeometryError(
            "measure must expose a vertical trailing barline segment"
        )
    candidates.sort(key=lambda line: (max(line.start.x, line.end.x), line.length))
    return candidates[-1]


def _optional_object_bbox(
    measure: ET.Element, class_name: str
) -> AxisAlignedBox | None:
    groups = [
        element
        for element in measure.iter()
        if _is_visible_object_group(element, class_name)
    ]
    if not groups:
        return None
    if len(groups) != 1:
        raise Stage7D5GeometryError(
            f"measure has ambiguous {class_name} renderer groups"
        )
    try:
        return _bbox_for_group(groups[0], content=False)
    except Stage7D5GeometryError:
        return _bbox_for_group(groups[0], content=True)


@dataclass(frozen=True, slots=True)
class _RawMeasure:
    svg_id: str
    system_id: str
    bbox: AxisAlignedBox
    barline: LineSegment
    clef_bbox: AxisAlignedBox | None
    meter_bbox: AxisAlignedBox | None
    staff_lines: tuple[LineSegment, ...]


@dataclass(frozen=True, slots=True)
class _RawSystem:
    svg_id: str
    bbox: AxisAlignedBox
    measures: tuple[_RawMeasure, ...]


def _parse_geometry_page(
    svg: bytes,
) -> tuple[tuple[float, float, float, float], tuple[_RawSystem, ...]]:
    if not isinstance(svg, bytes) or not svg:
        raise Stage7D5GeometryError("geometry SVG must be non-empty bytes")
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        raise Stage7D5GeometryError("geometry SVG is malformed") from exc
    if _local(root.tag) != "svg":
        raise Stage7D5GeometryError("geometry source must be SVG")
    view_box = _parse_view_box(root)

    systems = [
        element for element in root.iter() if _is_visible_object_group(element, "system")
    ]
    if not systems:
        raise Stage7D5GeometryError("geometry SVG contains no visible system group")

    raw_systems: list[_RawSystem] = []
    seen_ids: set[str] = set()
    for system in systems:
        system_id = system.attrib.get("id", "")
        if not system_id or system_id in seen_ids:
            raise Stage7D5GeometryError("system ids must be non-empty and unique")
        seen_ids.add(system_id)
        system_bbox = _bbox_for_group(system, content=True)

        measure_groups = [
            element
            for element in system.iter()
            if _is_visible_object_group(element, "measure")
        ]
        raw_measures: list[_RawMeasure] = []
        for measure in measure_groups:
            measure_id = measure.attrib.get("id", "")
            if not measure_id or measure_id in seen_ids:
                raise Stage7D5GeometryError("measure ids must be non-empty and unique")
            seen_ids.add(measure_id)
            measure_bbox = _bbox_for_group(measure, content=True)
            staffs = [
                element
                for element in measure.iter()
                if _is_visible_object_group(element, "staff")
            ]
            if len(staffs) != 1:
                raise Stage7D5GeometryError(
                    "V1 measure must contain exactly one visible staff group"
                )
            staff_id = staffs[0].attrib.get("id", "")
            if not staff_id or staff_id in seen_ids:
                raise Stage7D5GeometryError(
                    "staff renderer ids must be non-empty and unique"
                )
            seen_ids.add(staff_id)
            raw_measures.append(
                _RawMeasure(
                    svg_id=measure_id,
                    system_id=system_id,
                    bbox=measure_bbox,
                    barline=_trailing_barline(measure),
                    clef_bbox=_optional_object_bbox(measure, "clef"),
                    meter_bbox=_optional_object_bbox(measure, "meterSig"),
                    staff_lines=_direct_staff_lines(staffs[0]),
                )
            )
        if not raw_measures:
            raise Stage7D5GeometryError("system contains no measure groups")
        raw_measures.sort(
            key=lambda item: (item.bbox.x_min, item.bbox.y_min, item.svg_id)
        )
        raw_systems.append(_RawSystem(system_id, system_bbox, tuple(raw_measures)))

    raw_systems.sort(key=lambda item: (item.bbox.y_min, item.bbox.x_min, item.svg_id))
    return view_box, tuple(raw_systems)


def _aggregate_staff(
    system: _RawSystem, *, page_number: int, system_index: int
) -> StaffInstanceGeometry:
    by_line: list[list[LineSegment]] = [[] for _ in range(5)]
    for measure in system.measures:
        for index, line in enumerate(measure.staff_lines):
            by_line[index].append(line)

    merged: list[LineSegment] = []
    y_values: list[float] = []
    for segments in by_line:
        ys = [segment.start.y for segment in segments]
        if max(ys) - min(ys) > 1e-6:
            raise Stage7D5GeometryError(
                "measure staff segments do not align within one graphical staff instance"
            )
        y = sum(ys) / len(ys)
        x_values = [
            point
            for segment in segments
            for point in (segment.start.x, segment.end.x)
        ]
        merged.append(
            LineSegment(Point2D(min(x_values), y), Point2D(max(x_values), y))
        )
        y_values.append(y)

    spacings = [y_values[i + 1] - y_values[i] for i in range(4)]
    if min(spacings) <= 0 or max(spacings) - min(spacings) > 1e-6:
        raise Stage7D5GeometryError(
            "staff lines must have deterministic equal positive spacing"
        )
    spacing = sum(spacings) / len(spacings)
    x_min = min(line.start.x for line in merged)
    x_max = max(line.end.x for line in merged)
    bbox = AxisAlignedBox(x_min, y_values[0], x_max, y_values[-1])
    instance_id = f"staff-p{page_number:03d}-s{system_index:03d}-n1"
    return StaffInstanceGeometry(
        instance_id, system.svg_id, tuple(merged), bbox, spacing
    )


def extract_staff_structure_geometry(
    render_result: GeometryRenderResult,
    musicxml: bytes,
) -> tuple[PageGeometry, ...]:
    """Extract StaffSet/StructureSet SVG-space GT and bind canonical measure semantics."""

    if not isinstance(render_result, GeometryRenderResult):
        raise TypeError("render_result must be GeometryRenderResult")
    if not isinstance(musicxml, bytes):
        raise TypeError("musicxml must be bytes")
    if musicxml_sha256(musicxml) != render_result.source_musicxml_sha256:
        raise Stage7D5GeometryError(
            "MusicXML bytes do not match geometry render provenance"
        )

    projection = parse_supported_v1_musicxml_projection(musicxml)
    canonical_measures = projection.parts[0].measures
    raw_pages: list[
        tuple[
            GeometryRenderedPage,
            tuple[float, float, float, float],
            tuple[_RawSystem, ...],
        ]
    ] = []
    total_svg_measures = 0
    for page in render_result.pages:
        if sha256(page.svg).hexdigest() != page.sha256:
            raise Stage7D5GeometryError("geometry SVG hash provenance mismatch")
        view_box, systems = _parse_geometry_page(page.svg)
        total_svg_measures += sum(len(system.measures) for system in systems)
        raw_pages.append((page, view_box, systems))
    if total_svg_measures != len(canonical_measures):
        raise Stage7D5GeometryError(
            f"canonical/SVG measure count mismatch: {len(canonical_measures)} != {total_svg_measures}"
        )

    canonical_index = 0
    pages: list[PageGeometry] = []
    for rendered_page, view_box, raw_systems in raw_pages:
        staff_instances: list[StaffInstanceGeometry] = []
        systems: list[SystemGeometry] = []
        measures: list[MeasureGeometry] = []
        for system_index, raw_system in enumerate(raw_systems, start=1):
            staff = _aggregate_staff(
                raw_system,
                page_number=rendered_page.page_number,
                system_index=system_index,
            )
            staff_instances.append(staff)
            system_measure_numbers: list[int] = []
            for raw_measure in raw_system.measures:
                canonical = canonical_measures[canonical_index]
                canonical_index += 1
                meter_class = (
                    f"{canonical.time_signature[0]}/{canonical.time_signature[1]}"
                )
                measure = MeasureGeometry(
                    measure_id=f"measure-{canonical.number:04d}",
                    measure_number=canonical.number,
                    system_id=raw_system.svg_id,
                    measure_bbox=raw_measure.bbox,
                    barline_segment=raw_measure.barline,
                    clef_g2_bbox=raw_measure.clef_bbox,
                    meter_bbox=raw_measure.meter_bbox,
                    meter_class=meter_class,
                )
                measures.append(measure)
                system_measure_numbers.append(canonical.number)
            systems.append(
                SystemGeometry(
                    system_id=raw_system.svg_id,
                    system_bbox=raw_system.bbox,
                    staff_instance_id=staff.staff_instance_id,
                    measure_numbers=tuple(system_measure_numbers),
                )
            )
        pages.append(
            PageGeometry(
                page_number=rendered_page.page_number,
                coordinate_space="pinned_verovio_svg",
                view_box=view_box,
                source_musicxml_sha256=render_result.source_musicxml_sha256,
                base_renderer_config_fingerprint=render_result.base_renderer_config_fingerprint,
                geometry_instrumentation_fingerprint=render_result.geometry_instrumentation_fingerprint,
                geometry_svg_sha256=rendered_page.sha256,
                systems=tuple(systems),
                staff_instances=tuple(staff_instances),
                measures=tuple(measures),
            )
        )
    return tuple(pages)


def _pillow_rotation_reverse_affine(
    width: int,
    height: int,
    angle_degrees: float,
) -> tuple[int, int, tuple[float, float, float, float, float, float]]:
    """Replay Pillow 12.3.0 Image.rotate(expand=True) reverse-affine setup."""

    angle = angle_degrees % 360.0
    radians = -math.radians(angle)
    a = round(math.cos(radians), 15)
    b = round(math.sin(radians), 15)
    d = round(-math.sin(radians), 15)
    e = round(math.cos(radians), 15)
    c = 0.0
    f = 0.0

    def transform(
        x: float,
        y: float,
        matrix: tuple[float, float, float, float, float, float],
    ) -> tuple[float, float]:
        ma, mb, mc, md, me, mf = matrix
        return ma * x + mb * y + mc, md * x + me * y + mf

    matrix = (a, b, c, d, e, f)
    center = (width / 2.0, height / 2.0)
    c0, f0 = transform(-center[0], -center[1], matrix)
    matrix = (a, b, c0 + center[0], d, e, f0 + center[1])

    xx: list[float] = []
    yy: list[float] = []
    for x, y in (
        (0.0, 0.0),
        (float(width), 0.0),
        (float(width), float(height)),
        (0.0, float(height)),
    ):
        tx, ty = transform(x, y, matrix)
        xx.append(tx)
        yy.append(ty)
    new_width = math.ceil(max(xx)) - math.floor(min(xx))
    new_height = math.ceil(max(yy)) - math.floor(min(yy))
    c1, f1 = transform(
        -(new_width - width) / 2.0,
        -(new_height - height) / 2.0,
        matrix,
    )
    matrix = (a, b, c1, d, e, f1)
    return new_width, new_height, matrix


def _source_to_rotated(
    point: Point2D,
    reverse_matrix: tuple[float, float, float, float, float, float],
) -> Point2D:
    a, b, c, d, e, f = reverse_matrix
    det = a * e - b * d
    if not math.isfinite(det) or abs(det) < 1e-12:
        raise Stage7D5GeometryError("rotation affine matrix is singular")
    sx = point.x - c
    sy = point.y - f
    x = (e * sx - b * sy) / det
    y = (-d * sx + a * sy) / det
    return Point2D(x, y)


def _map_box(box: AxisAlignedBox, mapper) -> AxisAlignedBox:
    points = (
        mapper(Point2D(box.x_min, box.y_min)),
        mapper(Point2D(box.x_max, box.y_min)),
        mapper(Point2D(box.x_max, box.y_max)),
        mapper(Point2D(box.x_min, box.y_max)),
    )
    return AxisAlignedBox(
        min(point.x for point in points),
        min(point.y for point in points),
        max(point.x for point in points),
        max(point.y for point in points),
    )


def _map_line(line: LineSegment, mapper) -> LineSegment:
    return LineSegment(mapper(line.start), mapper(line.end))


def map_page_geometry_to_final_png(
    page: PageGeometry, degraded_page: object
) -> PageGeometry:
    """Map SVG geometry through exact CairoSVG scaling and Pillow rotation geometry."""

    if not isinstance(page, PageGeometry) or page.coordinate_space != "pinned_verovio_svg":
        raise Stage7D5GeometryError("page must be SVG-space PageGeometry")
    required = (
        "page_number",
        "source_musicxml_sha256",
        "renderer_config_fingerprint",
        "degradation_config_fingerprint",
        "config",
        "clean_width",
        "clean_height",
        "width",
        "height",
    )
    for name in required:
        if not hasattr(degraded_page, name):
            raise Stage7D5GeometryError(f"degraded_page is missing {name}")
    if degraded_page.page_number != page.page_number:
        raise Stage7D5GeometryError("page_number provenance mismatch")
    if degraded_page.source_musicxml_sha256 != page.source_musicxml_sha256:
        raise Stage7D5GeometryError("MusicXML provenance mismatch")
    if (
        degraded_page.renderer_config_fingerprint
        != page.base_renderer_config_fingerprint
    ):
        raise Stage7D5GeometryError("renderer config provenance mismatch")

    x0, y0, vb_width, vb_height = page.view_box
    clean_width = degraded_page.clean_width
    clean_height = degraded_page.clean_height
    if not all(
        isinstance(v, int) and not isinstance(v, bool) and v > 0
        for v in (clean_width, clean_height)
    ):
        raise Stage7D5GeometryError(
            "clean raster dimensions must be positive integers"
        )
    scale = clean_width / vb_width
    expected_height = vb_height * scale
    if abs(expected_height - clean_height) > 1.0:
        raise Stage7D5GeometryError(
            "CairoSVG raster dimensions do not preserve the SVG viewBox aspect ratio"
        )

    def svg_to_clean(point: Point2D) -> Point2D:
        return Point2D((point.x - x0) * scale, (point.y - y0) * scale)

    rotation_mdeg = getattr(degraded_page.config, "rotation_mdeg", None)
    if not isinstance(rotation_mdeg, int) or isinstance(rotation_mdeg, bool):
        raise Stage7D5GeometryError("rotation_mdeg must be an integer")
    if rotation_mdeg:
        out_width, out_height, reverse = _pillow_rotation_reverse_affine(
            clean_width, clean_height, rotation_mdeg / 1000.0
        )
        if (out_width, out_height) != (
            degraded_page.width,
            degraded_page.height,
        ):
            raise Stage7D5GeometryError(
                "Pillow rotation replay dimensions do not match degraded artifact"
            )

        def mapper(point: Point2D) -> Point2D:
            return _source_to_rotated(svg_to_clean(point), reverse)

    else:
        if (clean_width, clean_height) != (
            degraded_page.width,
            degraded_page.height,
        ):
            raise Stage7D5GeometryError(
                "photometric-only derivative unexpectedly changed geometry dimensions"
            )
        mapper = svg_to_clean

    mapped_staff: list[StaffInstanceGeometry] = []
    for staff in page.staff_instances:
        lines = tuple(_map_line(line, mapper) for line in staff.five_staff_lines)
        spacing = sum(
            math.hypot(
                lines[i + 1].start.x - lines[i].start.x,
                lines[i + 1].start.y - lines[i].start.y,
            )
            for i in range(4)
        ) / 4.0
        mapped_staff.append(
            StaffInstanceGeometry(
                staff_instance_id=staff.staff_instance_id,
                system_id=staff.system_id,
                five_staff_lines=lines,
                staff_instance_bbox=_map_box(staff.staff_instance_bbox, mapper),
                staff_spacing=spacing,
            )
        )

    mapped_measures = tuple(
        MeasureGeometry(
            measure_id=measure.measure_id,
            measure_number=measure.measure_number,
            system_id=measure.system_id,
            measure_bbox=_map_box(measure.measure_bbox, mapper),
            barline_segment=_map_line(measure.barline_segment, mapper),
            clef_g2_bbox=(
                None
                if measure.clef_g2_bbox is None
                else _map_box(measure.clef_g2_bbox, mapper)
            ),
            meter_bbox=(
                None
                if measure.meter_bbox is None
                else _map_box(measure.meter_bbox, mapper)
            ),
            meter_class=measure.meter_class,
        )
        for measure in page.measures
    )
    mapped_systems = tuple(
        SystemGeometry(
            system_id=system.system_id,
            system_bbox=_map_box(system.system_bbox, mapper),
            staff_instance_id=system.staff_instance_id,
            measure_numbers=system.measure_numbers,
        )
        for system in page.systems
    )
    payload = {
        "version": STAGE7D5_TRANSFORM_VERSION,
        "geometry_svg_sha256": page.geometry_svg_sha256,
        "view_box": page.view_box,
        "clean_size": [clean_width, clean_height],
        "final_size": [degraded_page.width, degraded_page.height],
        "degradation_config_fingerprint": degraded_page.degradation_config_fingerprint,
        "rotation_mdeg": rotation_mdeg,
    }
    return PageGeometry(
        page_number=page.page_number,
        coordinate_space="final_png_pixels",
        view_box=page.view_box,
        source_musicxml_sha256=page.source_musicxml_sha256,
        base_renderer_config_fingerprint=page.base_renderer_config_fingerprint,
        geometry_instrumentation_fingerprint=page.geometry_instrumentation_fingerprint,
        geometry_svg_sha256=page.geometry_svg_sha256,
        systems=mapped_systems,
        staff_instances=tuple(mapped_staff),
        measures=mapped_measures,
        geometry_transform_fingerprint=_canonical_sha256(payload),
    )


def geometry_report_payload(pages: tuple[PageGeometry, ...]) -> dict[str, object]:
    """Return canonical JSON-safe audit payload for a D5 pilot result."""

    if not isinstance(pages, tuple) or not pages:
        raise TypeError("pages must be a non-empty tuple")
    return {
        "schema": "stage7d5-staff-structure-geometry-report-v1",
        "version": STAGE7D5_GEOMETRY_VERSION,
        "d4_label_correction": D5_D4_LABEL_CORRECTION,
        "pages": [asdict(page) for page in pages],
    }
