"""Pinned-Verovio symbol geometry pilot for Stage 7-D12.

This module extracts development-only SVG-space ground truth for NoteHeadSet,
RestSet and AccidentalSet. It consumes the pinned D5 geometry render (with
Verovio bounding-box instrumentation) and deterministic supported-V1 MusicXML.
No learned prediction participates in linkage or labeling.

Pinned Verovio 6.2.1 exposes an anonymous ``g.notehead`` wrapper inside every
visible ``g.note``. The wrapper carries the notehead glyph ``use`` but no stable
renderer id/bbox of its own. The owning ``g.note`` *bounding-box* is the pinned
renderer notehead box; its ``content-bounding-box`` may additionally cover the
stem. D12 therefore:

- binds notehead identity to the owning renderer note id;
- requires exactly one notehead glyph and verifies its V1 SMuFL family;
- uses the owning note ``bounding-box`` (never its content bbox) as notehead bbox;
- treats ``g.accid`` as visible only when it contains one glyph ``use``;
- binds rests and visible accidentals to their renderer bboxes;
- rejects every canonical/renderer count, order, glyph or provenance mismatch.

This keeps spatial GT entirely renderer-authored and avoids manufacturing a
second geometry authority from glyph path arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
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
    "stage7d12-pinned-verovio-symbol-geometry-v3"
)
_CONTAINMENT_EPSILON: Final[float] = 1e-6
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
                        "symbol bbox must be contained by owning measure: "
                        f"kind={record.kind} event={record.canonical_event_id} "
                        f"symbol={record.bbox!r} measure={self.measure_bbox!r}"
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
    notehead_glyph_code: str | None
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
    except ET.ParseError as exc:  # validator should already reject this
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
                            measure_number=number,
                            event_index=event_index,
                        ),
                        visual_class=rest_class(duration),
                        notehead_glyph_code=None,
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
                accidental_element = _single_child(note, "accidental")
                accidental_value = None
                if accidental_element is not None:
                    if (
                        accidental_element.text is None
                        or not accidental_element.text.strip()
                    ):
                        raise Stage7D12SymbolGeometryError(
                            "visible accidental text must be non-empty"
                        )
                    accidental_value = accidental_class(
                        accidental_element.text.strip()
                    )
                atoms.append(
                    _CanonicalAtom(
                        kind="note",
                        canonical_event_id=canonical_event_id(
                            measure_number=number,
                            event_index=event_index,
                            chord_member_index=(member_index if chord else None),
                        ),
                        visual_class=notehead_fill_class(duration),
                        notehead_glyph_code=glyph_code,
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
    """Return renderer bbox, falling back to content bbox when D5 permits it."""

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


def _notehead_authority_bbox(
    note_group: ET.Element,
    coordinate_root: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> AxisAlignedBox:
    """Use only the pinned renderer note bounding-box, never stem content bbox."""

    try:
        return _d5._bbox_for_group(
            note_group,
            content=False,
            coordinate_root=coordinate_root,
            parent_map=parent_map,
        )
    except _d5.Stage7D5GeometryError as exc:
        raise Stage7D12SymbolGeometryError(
            "renderer note requires an explicit bounding-box for NoteHeadSet"
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
                            measure,
                            coordinate_root,
                            parent_map,
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


def _glyph_code_matches(href: str, expected_code: str) -> bool:
    target = href[1:] if href.startswith("#") else href
    return target == expected_code or target.startswith(expected_code + "-")


def _verify_notehead_glyph(
    note_group: ET.Element,
    expected_code: str,
) -> None:
    heads = _visible_descendants(note_group, "notehead")
    if len(heads) != 1:
        raise Stage7D12SymbolGeometryError(
            "each renderer note must expose exactly one notehead group"
        )
    uses = _glyph_uses(heads[0])
    if len(uses) != 1:
        raise Stage7D12SymbolGeometryError(
            "renderer notehead must expose exactly one glyph use"
        )
    href = _href(uses[0])
    if not href.startswith("#") or not _glyph_code_matches(href, expected_code):
        raise Stage7D12SymbolGeometryError(
            "renderer notehead glyph disagrees with canonical duration"
        )


def _visible_accid_groups(note_group: ET.Element) -> tuple[ET.Element, ...]:
    visible: list[ET.Element] = []
    for group in _visible_descendants(note_group, "accid"):
        uses = _glyph_uses(group)
        if len(uses) > 1:
            raise Stage7D12SymbolGeometryError(
                "renderer accid group contains multiple glyph uses"
            )
        if uses:
            visible.append(group)
    return tuple(visible)


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
        canonical.atoms,
        renderer_atoms,
        strict=True,
    ):
        renderer_id = renderer_group.attrib.get("id", "")

        if canonical_atom.kind == "rest":
            box = _bbox_for_visible_group(
                renderer_group,
                renderer.coordinate_root,
                renderer.parent_map,
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

        expected_head_code = canonical_atom.notehead_glyph_code
        if expected_head_code is None:
            raise Stage7D12SymbolGeometryError(
                "canonical note is missing expected notehead glyph code"
            )
        _verify_notehead_glyph(renderer_group, expected_head_code)
        box = _notehead_authority_bbox(
            renderer_group,
            renderer.coordinate_root,
            renderer.parent_map,
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
                accid,
                renderer.coordinate_root,
                renderer.parent_map,
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
        int,
        tuple[tuple[float, float, float, float], str],
    ] = {}

    for canonical_measure, renderer_measure in zip(
        canonical,
        renderer,
        strict=True,
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
