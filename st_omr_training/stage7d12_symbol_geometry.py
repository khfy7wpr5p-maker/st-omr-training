"""Pinned-Verovio symbol geometry pilot for Stage 7-D12.

This module extracts development-only SVG-space ground truth for NoteHeadSet,
RestSet and AccidentalSet. It consumes the already pinned D5 geometry render
(surface with bbox instrumentation) and deterministic V1 MusicXML. No learned
prediction participates in linkage or labeling.

Pinned Verovio 6.2.1 has two renderer details that are explicit D12 invariants:
anonymous ``g.notehead`` wrappers contain one glyph ``use`` reference, while
``g.accid`` exists structurally on every note and is visible only when it
contains a glyph ``use``. D12 therefore binds notehead identity to the owning
renderer note id, derives the anonymous notehead box from its referenced glyph
path, and ignores empty accidental placeholders.

Canonical-to-renderer linkage is fail-closed: measure and note/rest atom order,
symbol cardinality, V1 glyph families, bbox geometry and provenance must all
agree exactly before a label is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import re
import xml.etree.ElementTree as ET
from typing import Final

from . import _stage7d5_geometry_v1 as _d5
from .musicxml_validator import validate_musicxml
from .musicxml_writer import musicxml_sha256
from .stage7d12_symbol_gt_contract import (
    Stage7D12ContractError,
    accidental_class,
    canonical_event_id,
    notehead_fill_class,
    require_link_cardinality,
    rest_class,
)
from .stage7d5_geometry import AxisAlignedBox, GeometryRenderResult, Point2D


STAGE7D12_SYMBOL_GEOMETRY_VERSION: Final[str] = (
    "stage7d12-pinned-verovio-symbol-geometry-v2"
)
_CONTAINMENT_EPSILON: Final[float] = 1e-6
_GEOMETRY_EPSILON: Final[float] = 1e-12
_XLINK_HREF: Final[str] = "{http://www.w3.org/1999/xlink}href"

_NOTEHEAD_GLYPH_BY_DURATION: Final[dict[str, str]] = {
    "whole": "E0A2",
    "half": "E0A3",
    "quarter": "E0A4",
    "eighth": "E0A4",
}
_ACCID_GLYPH_BY_CLASS: Final[dict[str, str]] = {
    "flat": "E260",
    "natural": "E261",
    "sharp": "E262",
}
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_PATH_TOKEN_RE = re.compile(rf"\s*(?:,\s*)?([A-Za-z]|{_NUMBER})")
_USE_TRANSFORM_RE = re.compile(
    rf"^\s*translate\(\s*({_NUMBER})(?:[\s,]+({_NUMBER}))?\s*\)"
    rf"\s*scale\(\s*({_NUMBER})(?:[\s,]+({_NUMBER}))?\s*\)\s*$"
)


class Stage7D12SymbolGeometryError(Stage7D12ContractError):
    """Raised when renderer/canonical symbol geometry cannot be linked safely."""


@dataclass(frozen=True, slots=True)
class SymbolGeometry:
    kind: str
    canonical_event_id: str
    renderer_id: str
    class_name: str
    bbox: AxisAlignedBox
    center: Point2D | None

    def __post_init__(self) -> None:
        if self.kind not in {"notehead", "rest", "accidental"}:
            raise Stage7D12SymbolGeometryError("unsupported D12 symbol kind")
        if not self.renderer_id:
            raise Stage7D12SymbolGeometryError("renderer symbol id must be non-empty")
        if not self.class_name:
            raise Stage7D12SymbolGeometryError("symbol class_name must be non-empty")
        if self.kind == "notehead":
            if self.center is None or not _point_inside_box(self.center, self.bbox):
                raise Stage7D12SymbolGeometryError(
                    "notehead center must lie inside its bbox"
                )
        elif self.center is not None:
            raise Stage7D12SymbolGeometryError(
                "only notehead records carry an explicit center"
            )


@dataclass(frozen=True, slots=True)
class MeasureSymbolGeometry:
    measure_number: int
    renderer_measure_id: str
    measure_bbox: AxisAlignedBox
    noteheads: tuple[SymbolGeometry, ...]
    rests: tuple[SymbolGeometry, ...]
    accidentals: tuple[SymbolGeometry, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.measure_number, int)
            or isinstance(self.measure_number, bool)
            or self.measure_number < 1
        ):
            raise Stage7D12SymbolGeometryError(
                "measure_number must be a positive integer"
            )
        if not self.renderer_measure_id:
            raise Stage7D12SymbolGeometryError(
                "renderer_measure_id must be non-empty"
            )
        for records in (self.noteheads, self.rests, self.accidentals):
            for record in records:
                if not _box_inside_box(record.bbox, self.measure_bbox):
                    raise Stage7D12SymbolGeometryError(
                        "symbol bbox must be contained by owning measure"
                    )


@dataclass(frozen=True, slots=True)
class SymbolGeometryPage:
    page_number: int
    coordinate_space: str
    view_box: tuple[float, float, float, float]
    source_musicxml_sha256: str
    base_renderer_config_fingerprint: str
    geometry_instrumentation_fingerprint: str
    geometry_svg_sha256: str
    measures: tuple[MeasureSymbolGeometry, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.page_number, int)
            or isinstance(self.page_number, bool)
            or self.page_number < 1
        ):
            raise Stage7D12SymbolGeometryError("page_number must be positive")
        if self.coordinate_space != "pinned_verovio_svg":
            raise Stage7D12SymbolGeometryError(
                "D12 pilot output must remain in pinned Verovio SVG space"
            )
        if len(self.view_box) != 4 or not self.measures:
            raise Stage7D12SymbolGeometryError(
                "symbol page requires a viewBox and at least one measure"
            )


@dataclass(frozen=True, slots=True)
class _CanonicalAtom:
    kind: str
    canonical_event_id: str
    visual_class: str
    renderer_glyph_code: str | None
    visible_accidental: str | None


@dataclass(frozen=True, slots=True)
class _CanonicalMeasure:
    number: int
    atoms: tuple[_CanonicalAtom, ...]


@dataclass(frozen=True, slots=True)
class _RawRendererMeasure:
    page_number: int
    view_box: tuple[float, float, float, float]
    page_sha256: str
    measure_group: ET.Element
    measure_bbox: AxisAlignedBox
    coordinate_root: ET.Element
    parent_map: dict[ET.Element, ET.Element]
    id_map: dict[str, ET.Element]


@dataclass(frozen=True, slots=True)
class _Line:
    start: Point2D
    end: Point2D


@dataclass(frozen=True, slots=True)
class _Cubic:
    start: Point2D
    control1: Point2D
    control2: Point2D
    end: Point2D


def _local(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local(child) == name]


def _single_child(element: ET.Element, name: str) -> ET.Element | None:
    children = _children(element, name)
    if len(children) > 1:
        raise Stage7D12SymbolGeometryError(
            f"canonical MusicXML has multiple {name} children"
        )
    return children[0] if children else None


def _single_text(element: ET.Element, name: str) -> str:
    child = _single_child(element, name)
    if child is None or child.text is None or not child.text.strip():
        raise Stage7D12SymbolGeometryError(
            f"canonical MusicXML note requires {name}"
        )
    return child.text.strip()


def _canonical_measures(musicxml: bytes) -> tuple[_CanonicalMeasure, ...]:
    validation = validate_musicxml(musicxml)
    if not validation.is_valid:
        raise Stage7D12SymbolGeometryError(
            "MusicXML failed the supported-V1 validator before D12 linkage"
        )
    try:
        root = ET.fromstring(musicxml)
    except ET.ParseError as exc:
        raise Stage7D12SymbolGeometryError("MusicXML parse failed") from exc

    parts = [element for element in root if _local(element) == "part"]
    if len(parts) != 1:
        raise Stage7D12SymbolGeometryError("D12 requires exactly one V1 part")

    canonical: list[_CanonicalMeasure] = []
    for measure in _children(parts[0], "measure"):
        raw_number = measure.attrib.get("number", "")
        try:
            number = int(raw_number)
        except ValueError as exc:
            raise Stage7D12SymbolGeometryError(
                "measure number must be a canonical positive integer"
            ) from exc
        if number < 1 or str(number) != raw_number:
            raise Stage7D12SymbolGeometryError(
                "measure number must be a canonical positive integer"
            )

        grouped: list[list[ET.Element]] = []
        for note in _children(measure, "note"):
            continuation = _single_child(note, "chord") is not None
            if continuation:
                if not grouped:
                    raise Stage7D12SymbolGeometryError(
                        "chord continuation has no preceding base note"
                    )
                if _single_child(grouped[-1][0], "rest") is not None:
                    raise Stage7D12SymbolGeometryError(
                        "chord continuation cannot follow a rest"
                    )
                grouped[-1].append(note)
            else:
                grouped.append([note])

        atoms: list[_CanonicalAtom] = []
        for event_index, group in enumerate(grouped):
            first = group[0]
            if _single_child(first, "rest") is not None:
                if len(group) != 1:
                    raise Stage7D12SymbolGeometryError(
                        "rest event cannot contain chord members"
                    )
                duration = _single_text(first, "type")
                atoms.append(
                    _CanonicalAtom(
                        kind="rest",
                        canonical_event_id=canonical_event_id(
                            measure_number=number, event_index=event_index
                        ),
                        visual_class=rest_class(duration),
                        renderer_glyph_code=None,
                        visible_accidental=None,
                    )
                )
                continue

            chord = len(group) > 1
            for member_index, note in enumerate(group):
                if _single_child(note, "rest") is not None:
                    raise Stage7D12SymbolGeometryError(
                        "pitched event cannot mix a rest member"
                    )
                duration = _single_text(note, "type")
                try:
                    glyph_code = _NOTEHEAD_GLYPH_BY_DURATION[duration]
                except KeyError as exc:
                    raise Stage7D12SymbolGeometryError(
                        "unsupported V1 notehead duration"
                    ) from exc
                accidental = _single_child(note, "accidental")
                accidental_value = None
                if accidental is not None:
                    if accidental.text is None or not accidental.text.strip():
                        raise Stage7D12SymbolGeometryError(
                            "visible accidental text must be non-empty"
                        )
                    accidental_value = accidental_class(accidental.text.strip())
                atoms.append(
                    _CanonicalAtom(
                        kind="note",
                        canonical_event_id=canonical_event_id(
                            measure_number=number,
                            event_index=event_index,
                            chord_member_index=(member_index if chord else None),
                        ),
                        visual_class=notehead_fill_class(duration),
                        renderer_glyph_code=glyph_code,
                        visible_accidental=accidental_value,
                    )
                )
        canonical.append(_CanonicalMeasure(number=number, atoms=tuple(atoms)))
    if not canonical:
        raise Stage7D12SymbolGeometryError("MusicXML contains no V1 measures")
    return tuple(canonical)


def _bbox_for_visible_group(
    group: ET.Element,
    coordinate_root: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> AxisAlignedBox:
    try:
        return _d5._bbox_for_group(
            group,
            content=False,
            coordinate_root=coordinate_root,
            parent_map=parent_map,
        )
    except _d5.Stage7D5GeometryError:
        try:
            return _d5._bbox_for_group(
                group,
                content=True,
                coordinate_root=coordinate_root,
                parent_map=parent_map,
            )
        except _d5.Stage7D5GeometryError as exc:
            raise Stage7D12SymbolGeometryError(
                "renderer symbol is missing an unambiguous bbox"
            ) from exc


def _point_inside_box(point: Point2D, box: AxisAlignedBox) -> bool:
    return (
        box.x_min - _CONTAINMENT_EPSILON
        <= point.x
        <= box.x_max + _CONTAINMENT_EPSILON
        and box.y_min - _CONTAINMENT_EPSILON
        <= point.y
        <= box.y_max + _CONTAINMENT_EPSILON
    )


def _box_inside_box(inner: AxisAlignedBox, outer: AxisAlignedBox) -> bool:
    return (
        inner.x_min >= outer.x_min - _CONTAINMENT_EPSILON
        and inner.y_min >= outer.y_min - _CONTAINMENT_EPSILON
        and inner.x_max <= outer.x_max + _CONTAINMENT_EPSILON
        and inner.y_max <= outer.y_max + _CONTAINMENT_EPSILON
    )


def _renderer_measures(
    render_result: GeometryRenderResult,
) -> tuple[_RawRendererMeasure, ...]:
    raw: list[_RawRendererMeasure] = []
    seen_measure_ids: set[str] = set()
    for page in render_result.pages:
        if sha256(page.svg).hexdigest() != page.sha256:
            raise Stage7D12SymbolGeometryError(
                "geometry SVG hash provenance mismatch"
            )
        try:
            root = ET.fromstring(page.svg)
        except ET.ParseError as exc:
            raise Stage7D12SymbolGeometryError("geometry SVG is malformed") from exc
        coordinate_root, view_box = _d5._coordinate_root(root)
        parent_map = {
            child: parent for parent in coordinate_root.iter() for child in parent
        }
        id_map: dict[str, ET.Element] = {}
        for element in root.iter():
            element_id = element.attrib.get("id")
            if element_id:
                if element_id in id_map:
                    raise Stage7D12SymbolGeometryError(
                        "geometry SVG ids must be globally unique within a page"
                    )
                id_map[element_id] = element

        systems = [
            element
            for element in coordinate_root.iter()
            if _d5._is_visible_object_group(element, "system")
        ]
        system_rows: list[tuple[AxisAlignedBox, ET.Element]] = []
        for system in systems:
            system_rows.append(
                (
                    _bbox_for_visible_group(system, coordinate_root, parent_map),
                    system,
                )
            )
        system_rows.sort(
            key=lambda item: (
                item[0].y_min,
                item[0].x_min,
                item[1].attrib.get("id", ""),
            )
        )
        for _, system in system_rows:
            measures = [
                element
                for element in system.iter()
                if _d5._is_visible_object_group(element, "measure")
            ]
            measure_rows: list[tuple[AxisAlignedBox, ET.Element]] = []
            for measure in measures:
                renderer_id = measure.attrib.get("id", "")
                if not renderer_id or renderer_id in seen_measure_ids:
                    raise Stage7D12SymbolGeometryError(
                        "renderer measure ids must be non-empty and globally unique"
                    )
                seen_measure_ids.add(renderer_id)
                measure_rows.append(
                    (
                        _bbox_for_visible_group(
                            measure, coordinate_root, parent_map
                        ),
                        measure,
                    )
                )
            measure_rows.sort(
                key=lambda item: (
                    item[0].x_min,
                    item[0].y_min,
                    item[1].attrib.get("id", ""),
                )
            )
            for measure_bbox, measure in measure_rows:
                raw.append(
                    _RawRendererMeasure(
                        page_number=page.page_number,
                        view_box=view_box,
                        page_sha256=page.sha256,
                        measure_group=measure,
                        measure_bbox=measure_bbox,
                        coordinate_root=coordinate_root,
                        parent_map=parent_map,
                        id_map=id_map,
                    )
                )
    if not raw:
        raise Stage7D12SymbolGeometryError(
            "geometry SVG contains no visible measures"
        )
    return tuple(raw)


def _visible_descendants(group: ET.Element, class_name: str) -> list[ET.Element]:
    return [
        element
        for element in group.iter()
        if element is not group
        and _d5._is_visible_object_group(element, class_name)
    ]


def _renderer_atoms(measure: _RawRendererMeasure) -> tuple[ET.Element, ...]:
    atoms = [
        element
        for element in measure.measure_group.iter()
        if element is not measure.measure_group
        and (
            _d5._is_visible_object_group(element, "note")
            or _d5._is_visible_object_group(element, "rest")
        )
    ]
    renderer_ids = [element.attrib.get("id", "") for element in atoms]
    if any(not value for value in renderer_ids) or len(set(renderer_ids)) != len(
        renderer_ids
    ):
        raise Stage7D12SymbolGeometryError(
            "renderer note/rest ids must be non-empty and unique within measure"
        )
    return tuple(atoms)


def _href(element: ET.Element) -> str:
    direct = element.attrib.get("href")
    namespaced = element.attrib.get(_XLINK_HREF)
    if direct and namespaced and direct != namespaced:
        raise Stage7D12SymbolGeometryError(
            "renderer glyph has conflicting href attributes"
        )
    return direct or namespaced or ""


def _glyph_uses(group: ET.Element) -> tuple[ET.Element, ...]:
    return tuple(element for element in group.iter() if _local(element) == "use")


def _visible_accid_groups(note: ET.Element) -> tuple[ET.Element, ...]:
    visible: list[ET.Element] = []
    for group in _visible_descendants(note, "accid"):
        uses = _glyph_uses(group)
        if len(uses) > 1:
            raise Stage7D12SymbolGeometryError(
                "renderer accid group contains multiple glyph uses"
            )
        if uses:
            visible.append(group)
    return tuple(visible)


def _parse_use_transform(text: str) -> tuple[float, float, float, float]:
    match = _USE_TRANSFORM_RE.fullmatch(text)
    if match is None:
        raise Stage7D12SymbolGeometryError(
            "notehead glyph use transform is outside pinned D12 shape"
        )
    tx = float(match.group(1))
    ty = float(match.group(2) or 0.0)
    sx = float(match.group(3))
    sy = float(match.group(4) or match.group(3))
    if not all(math.isfinite(value) for value in (tx, ty, sx, sy)):
        raise Stage7D12SymbolGeometryError(
            "notehead glyph use transform must be finite"
        )
    if sx <= 0.0 or sy <= 0.0:
        raise Stage7D12SymbolGeometryError(
            "notehead glyph use scale must be positive"
        )
    return tx, ty, sx, sy


def _path_tokens(text: str) -> tuple[str, ...]:
    if not text or not text.strip():
        raise Stage7D12SymbolGeometryError("notehead glyph path is empty")
    tokens: list[str] = []
    position = 0
    while position < len(text):
        match = _PATH_TOKEN_RE.match(text, position)
        if match is None:
            if text[position:].strip(" ,\t\r\n") == "":
                break
            raise Stage7D12SymbolGeometryError(
                "notehead glyph path contains unsupported syntax"
            )
        tokens.append(match.group(1))
        position = match.end()
    if not tokens:
        raise Stage7D12SymbolGeometryError("notehead glyph path has no tokens")
    return tuple(tokens)


def _point(x: float, y: float) -> Point2D:
    if not math.isfinite(x) or not math.isfinite(y):
        raise Stage7D12SymbolGeometryError(
            "notehead glyph path coordinate is non-finite"
        )
    return Point2D(x, y)


def _parse_notehead_path(text: str) -> tuple[_Line | _Cubic, ...]:
    tokens = _path_tokens(text)
    index = 0
    current: Point2D | None = None
    subpath_start: Point2D | None = None
    segments: list[_Line | _Cubic] = []

    def number() -> float:
        nonlocal index
        if index >= len(tokens) or tokens[index].isalpha():
            raise Stage7D12SymbolGeometryError(
                "notehead glyph path is missing a numeric argument"
            )
        value = float(tokens[index])
        index += 1
        if not math.isfinite(value):
            raise Stage7D12SymbolGeometryError(
                "notehead glyph path contains non-finite number"
            )
        return value

    while index < len(tokens):
        command = tokens[index]
        index += 1
        if not command.isalpha():
            raise Stage7D12SymbolGeometryError(
                "notehead glyph path requires explicit commands"
            )
        if command == "M":
            x, y = number(), number()
            current = _point(x, y)
            subpath_start = current
        elif command == "l":
            if current is None:
                raise Stage7D12SymbolGeometryError(
                    "relative line appears before move"
                )
            end = _point(current.x + number(), current.y + number())
            segments.append(_Line(current, end))
            current = end
        elif command == "c":
            if current is None:
                raise Stage7D12SymbolGeometryError(
                    "relative cubic appears before move"
                )
            dx1, dy1 = number(), number()
            dx2, dy2 = number(), number()
            dx3, dy3 = number(), number()
            control1 = _point(current.x + dx1, current.y + dy1)
            control2 = _point(current.x + dx2, current.y + dy2)
            end = _point(current.x + dx3, current.y + dy3)
            segments.append(_Cubic(current, control1, control2, end))
            current = end
        elif command == "s":
            if current is None:
                raise Stage7D12SymbolGeometryError(
                    "relative smooth cubic appears before move"
                )
            if segments and isinstance(segments[-1], _Cubic) and segments[-1].end == current:
                prior_control = segments[-1].control2
                control1 = _point(
                    2.0 * current.x - prior_control.x,
                    2.0 * current.y - prior_control.y,
                )
            else:
                control1 = current
            dx2, dy2 = number(), number()
            dx3, dy3 = number(), number()
            control2 = _point(current.x + dx2, current.y + dy2)
            end = _point(current.x + dx3, current.y + dy3)
            segments.append(_Cubic(current, control1, control2, end))
            current = end
        elif command in {"z", "Z"}:
            if current is None or subpath_start is None:
                raise Stage7D12SymbolGeometryError(
                    "closepath appears before move"
                )
            if current != subpath_start:
                segments.append(_Line(current, subpath_start))
            current = subpath_start
        else:
            raise Stage7D12SymbolGeometryError(
                f"unsupported pinned notehead path command: {command}"
            )
    if not segments:
        raise Stage7D12SymbolGeometryError(
            "notehead glyph path contains no drawable segment"
        )
    return tuple(segments)


def _transform_point(point: Point2D, matrix: tuple[float, ...]) -> Point2D:
    try:
        return _d5._apply_affine(point, matrix)
    except _d5.Stage7D5GeometryError as exc:
        raise Stage7D12SymbolGeometryError(
            "notehead glyph affine transform failed"
        ) from exc


def _transform_segment(
    segment: _Line | _Cubic,
    matrix: tuple[float, ...],
) -> _Line | _Cubic:
    if isinstance(segment, _Line):
        return _Line(
            _transform_point(segment.start, matrix),
            _transform_point(segment.end, matrix),
        )
    return _Cubic(
        _transform_point(segment.start, matrix),
        _transform_point(segment.control1, matrix),
        _transform_point(segment.control2, matrix),
        _transform_point(segment.end, matrix),
    )


def _cubic_value(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    u = 1.0 - t
    return (
        u * u * u * p0
        + 3.0 * u * u * t * p1
        + 3.0 * u * t * t * p2
        + t * t * t * p3
    )


def _cubic_extrema(
    p0: float, p1: float, p2: float, p3: float
) -> tuple[float, ...]:
    a = -p0 + 3.0 * p1 - 3.0 * p2 + p3
    b = 2.0 * (p0 - 2.0 * p1 + p2)
    c = p1 - p0
    roots: list[float] = []
    if abs(a) <= _GEOMETRY_EPSILON:
        if abs(b) > _GEOMETRY_EPSILON:
            t = -c / b
            if 0.0 < t < 1.0:
                roots.append(t)
        return tuple(roots)
    discriminant = b * b - 4.0 * a * c
    if discriminant < -_GEOMETRY_EPSILON:
        return ()
    discriminant = max(0.0, discriminant)
    root = math.sqrt(discriminant)
    for t in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
        if 0.0 < t < 1.0 and not any(
            math.isclose(t, prior, abs_tol=1e-12) for prior in roots
        ):
            roots.append(t)
    return tuple(roots)


def _segments_bbox(segments: tuple[_Line | _Cubic, ...]) -> AxisAlignedBox:
    xs: list[float] = []
    ys: list[float] = []
    for segment in segments:
        if isinstance(segment, _Line):
            xs.extend((segment.start.x, segment.end.x))
            ys.extend((segment.start.y, segment.end.y))
            continue
        tx = (0.0, 1.0) + _cubic_extrema(
            segment.start.x,
            segment.control1.x,
            segment.control2.x,
            segment.end.x,
        )
        ty = (0.0, 1.0) + _cubic_extrema(
            segment.start.y,
            segment.control1.y,
            segment.control2.y,
            segment.end.y,
        )
        xs.extend(
            _cubic_value(
                segment.start.x,
                segment.control1.x,
                segment.control2.x,
                segment.end.x,
                t,
            )
            for t in tx
        )
        ys.extend(
            _cubic_value(
                segment.start.y,
                segment.control1.y,
                segment.control2.y,
                segment.end.y,
                t,
            )
            for t in ty
        )
    if not xs or not ys or not all(math.isfinite(value) for value in xs + ys):
        raise Stage7D12SymbolGeometryError(
            "notehead glyph produced invalid bounds"
        )
    try:
        return AxisAlignedBox(min(xs), min(ys), max(xs), max(ys))
    except _d5.Stage7D5GeometryError as exc:
        raise Stage7D12SymbolGeometryError(
            "notehead glyph bbox is degenerate"
        ) from exc


def _notehead_bbox(
    head: ET.Element,
    *,
    expected_glyph_code: str,
    measure: _RawRendererMeasure,
) -> AxisAlignedBox:
    uses = _glyph_uses(head)
    if len(uses) != 1:
        raise Stage7D12SymbolGeometryError(
            "anonymous renderer notehead must contain exactly one glyph use"
        )
    use = uses[0]
    href = _href(use)
    if not href.startswith(f"#{expected_glyph_code}-"):
        raise Stage7D12SymbolGeometryError(
            "renderer notehead glyph disagrees with canonical duration"
        )
    definition = measure.id_map.get(href[1:])
    if definition is None:
        raise Stage7D12SymbolGeometryError(
            "notehead glyph definition is missing from pinned SVG"
        )
    paths = [element for element in definition.iter() if _local(element) == "path"]
    if len(paths) != 1:
        raise Stage7D12SymbolGeometryError(
            "V1 notehead glyph definition must contain exactly one path"
        )
    path = paths[0]
    if path.attrib.get("transform", "").strip() != "scale(1,-1)":
        raise Stage7D12SymbolGeometryError(
            "V1 notehead glyph path transform drifted from pinned shape"
        )
    path_matrix = _d5._parse_transform(path)
    tx, ty, sx, sy = _parse_use_transform(use.attrib.get("transform", ""))
    use_matrix = (sx, 0.0, 0.0, sy, tx, ty)
    parent_matrix = _d5._cumulative_transform(
        head, measure.coordinate_root, measure.parent_map
    )
    combined = _d5._compose(
        parent_matrix,
        _d5._compose(use_matrix, path_matrix),
    )
    segments = tuple(
        _transform_segment(segment, combined)
        for segment in _parse_notehead_path(path.attrib.get("d", ""))
    )
    return _segments_bbox(segments)


def _accid_href(group: ET.Element) -> str:
    uses = _glyph_uses(group)
    if len(uses) != 1:
        raise Stage7D12SymbolGeometryError(
            "visible renderer accidental must contain exactly one glyph use"
        )
    href = _href(uses[0])
    if not href.startswith("#"):
        raise Stage7D12SymbolGeometryError(
            "renderer accidental glyph must be an internal fragment"
        )
    return href


def _glyph_code_matches(href: str, expected_code: str) -> bool:
    target = href[1:] if href.startswith("#") else href
    return target == expected_code or target.startswith(expected_code + "-")


def _extract_measure(
    canonical: _CanonicalMeasure,
    renderer: _RawRendererMeasure,
) -> MeasureSymbolGeometry:
    renderer_atoms = _renderer_atoms(renderer)
    expected_kinds = tuple(atom.kind for atom in canonical.atoms)
    actual_kinds = tuple(
        "note"
        if _d5._is_visible_object_group(group, "note")
        else "rest"
        for group in renderer_atoms
    )
    require_link_cardinality(
        kind="notehead",
        canonical_count=expected_kinds.count("note"),
        renderer_count=actual_kinds.count("note"),
    )
    require_link_cardinality(
        kind="rest",
        canonical_count=expected_kinds.count("rest"),
        renderer_count=actual_kinds.count("rest"),
    )
    if expected_kinds != actual_kinds:
        raise Stage7D12SymbolGeometryError(
            "canonical/renderer note-rest atom ordering mismatch"
        )

    canonical_accidentals = sum(
        atom.visible_accidental is not None for atom in canonical.atoms
    )
    renderer_accidentals = sum(
        len(_visible_accid_groups(group))
        for group, atom in zip(renderer_atoms, canonical.atoms, strict=True)
        if atom.kind == "note"
    )
    require_link_cardinality(
        kind="accidental",
        canonical_count=canonical_accidentals,
        renderer_count=renderer_accidentals,
    )

    noteheads: list[SymbolGeometry] = []
    rests: list[SymbolGeometry] = []
    accidentals: list[SymbolGeometry] = []
    seen_kind_ids: dict[str, set[str]] = {
        "notehead": set(),
        "rest": set(),
        "accidental": set(),
    }
    for canonical_atom, renderer_group in zip(
        canonical.atoms, renderer_atoms, strict=True
    ):
        renderer_id = renderer_group.attrib.get("id", "")
        if canonical_atom.kind == "rest":
            box = _bbox_for_visible_group(
                renderer_group, renderer.coordinate_root, renderer.parent_map
            )
            record = SymbolGeometry(
                kind="rest",
                canonical_event_id=canonical_atom.canonical_event_id,
                renderer_id=renderer_id,
                class_name=canonical_atom.visual_class,
                bbox=box,
                center=None,
            )
            rests.append(record)
            seen_kind_ids["rest"].add(record.canonical_event_id)
            continue

        heads = _visible_descendants(renderer_group, "notehead")
        if len(heads) != 1:
            raise Stage7D12SymbolGeometryError(
                "each renderer note must expose exactly one notehead group"
            )
        if canonical_atom.renderer_glyph_code is None:
            raise Stage7D12SymbolGeometryError(
                "canonical note is missing its expected V1 glyph code"
            )
        box = _notehead_bbox(
            heads[0],
            expected_glyph_code=canonical_atom.renderer_glyph_code,
            measure=renderer,
        )
        center = Point2D(
            (box.x_min + box.x_max) / 2.0,
            (box.y_min + box.y_max) / 2.0,
        )
        note_record = SymbolGeometry(
            kind="notehead",
            canonical_event_id=canonical_atom.canonical_event_id,
            renderer_id=renderer_id,
            class_name=canonical_atom.visual_class,
            bbox=box,
            center=center,
        )
        noteheads.append(note_record)
        seen_kind_ids["notehead"].add(note_record.canonical_event_id)

        accids = _visible_accid_groups(renderer_group)
        expected_accid = canonical_atom.visible_accidental
        if expected_accid is None:
            if accids:
                raise Stage7D12SymbolGeometryError(
                    "renderer note exposes visible accidental without canonical intent"
                )
            continue
        if len(accids) != 1:
            raise Stage7D12SymbolGeometryError(
                "canonical visible accidental requires exactly one visible accid group"
            )
        accid = accids[0]
        accid_id = accid.attrib.get("id", "")
        if not accid_id:
            raise Stage7D12SymbolGeometryError(
                "renderer accidental id must be non-empty"
            )
        expected_accid_code = _ACCID_GLYPH_BY_CLASS[expected_accid]
        if not _glyph_code_matches(_accid_href(accid), expected_accid_code):
            raise Stage7D12SymbolGeometryError(
                "renderer accidental glyph disagrees with canonical accidental class"
            )
        accid_record = SymbolGeometry(
            kind="accidental",
            canonical_event_id=canonical_atom.canonical_event_id,
            renderer_id=accid_id,
            class_name=expected_accid,
            bbox=_bbox_for_visible_group(
                accid, renderer.coordinate_root, renderer.parent_map
            ),
            center=None,
        )
        accidentals.append(accid_record)
        seen_kind_ids["accidental"].add(accid_record.canonical_event_id)

    for kind, records in (
        ("notehead", noteheads),
        ("rest", rests),
        ("accidental", accidentals),
    ):
        if len(seen_kind_ids[kind]) != len(records):
            raise Stage7D12SymbolGeometryError(
                f"duplicate canonical_event_id within {kind} target family"
            )
        renderer_ids = [record.renderer_id for record in records]
        if len(set(renderer_ids)) != len(renderer_ids):
            raise Stage7D12SymbolGeometryError(
                f"duplicate renderer id within {kind} target family"
            )

    return MeasureSymbolGeometry(
        measure_number=canonical.number,
        renderer_measure_id=renderer.measure_group.attrib.get("id", ""),
        measure_bbox=renderer.measure_bbox,
        noteheads=tuple(noteheads),
        rests=tuple(rests),
        accidentals=tuple(accidentals),
    )


def extract_symbol_geometry(
    render_result: GeometryRenderResult,
    musicxml: bytes,
) -> tuple[SymbolGeometryPage, ...]:
    """Extract fail-closed D12 symbol GT from the pinned geometry render."""

    if not isinstance(render_result, GeometryRenderResult):
        raise TypeError("render_result must be GeometryRenderResult")
    if not isinstance(musicxml, bytes):
        raise TypeError("musicxml must be bytes")
    if musicxml_sha256(musicxml) != render_result.source_musicxml_sha256:
        raise Stage7D12SymbolGeometryError(
            "MusicXML bytes do not match geometry render provenance"
        )

    canonical = _canonical_measures(musicxml)
    renderer = _renderer_measures(render_result)
    if len(canonical) != len(renderer):
        raise Stage7D12SymbolGeometryError(
            "canonical/renderer measure cardinality mismatch"
        )

    by_page: dict[int, list[MeasureSymbolGeometry]] = {}
    page_metadata: dict[
        int, tuple[tuple[float, float, float, float], str]
    ] = {}
    for canonical_measure, renderer_measure in zip(
        canonical, renderer, strict=True
    ):
        measure = _extract_measure(canonical_measure, renderer_measure)
        by_page.setdefault(renderer_measure.page_number, []).append(measure)
        current = page_metadata.setdefault(
            renderer_measure.page_number,
            (renderer_measure.view_box, renderer_measure.page_sha256),
        )
        if current != (renderer_measure.view_box, renderer_measure.page_sha256):
            raise Stage7D12SymbolGeometryError(
                "inconsistent renderer page metadata during D12 extraction"
            )

    pages: list[SymbolGeometryPage] = []
    for rendered_page in render_result.pages:
        measures = by_page.get(rendered_page.page_number)
        if not measures:
            raise Stage7D12SymbolGeometryError(
                "every rendered D12 page must contain at least one bound measure"
            )
        view_box, page_sha = page_metadata[rendered_page.page_number]
        if page_sha != rendered_page.sha256:
            raise Stage7D12SymbolGeometryError(
                "renderer page hash changed during D12 extraction"
            )
        pages.append(
            SymbolGeometryPage(
                page_number=rendered_page.page_number,
                coordinate_space="pinned_verovio_svg",
                view_box=view_box,
                source_musicxml_sha256=render_result.source_musicxml_sha256,
                base_renderer_config_fingerprint=(
                    render_result.base_renderer_config_fingerprint
                ),
                geometry_instrumentation_fingerprint=(
                    render_result.geometry_instrumentation_fingerprint
                ),
                geometry_svg_sha256=rendered_page.sha256,
                measures=tuple(measures),
            )
        )
    return tuple(pages)
