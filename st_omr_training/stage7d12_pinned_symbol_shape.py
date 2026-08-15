"""Pinned Verovio 6.2.1 symbol-instance shape helpers for Stage 7-D12.

The D12 pilot discovered two renderer details that must be handled explicitly:

* ``g.notehead`` is an instance container without its own renderer id or bbox
  instrumentation.  It contains one ``use`` of a pinned SMuFL notehead glyph.
  The owning ``g.note`` id is therefore the stable renderer-instance identity,
  while the notehead bbox is reconstructed from the exact pinned glyph metrics
  and the renderer's deterministic use transform.
* every rendered pitched note may contain one ``g.accid`` container, including
  notes with no visible accidental.  An accidental is visible only when that
  container has exactly one non-instrumentation ``use`` child.  Empty containers
  containing only bbox instrumentation are structural placeholders, not glyphs.

These helpers are deliberately narrow.  Any renderer-shape drift fails closed
instead of being guessed from proximity or appearance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import xml.etree.ElementTree as ET
from typing import Final

from . import _stage7d5_geometry_v1 as _d5
from .stage7d5_geometry import AxisAlignedBox, Point2D


class Stage7D12PinnedSymbolShapeError(ValueError):
    """Raised when pinned symbol renderer structure drifts or is ambiguous."""


_XLINK_HREF: Final[str] = "{http://www.w3.org/1999/xlink}href"
_USE_TRANSFORM_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*translate\(([^)]*)\)\s+scale\(([^)]*)\)\s*$"
)


@dataclass(frozen=True, slots=True)
class _PinnedNoteheadGlyph:
    code: str
    path_transform: str
    path_data: str
    transformed_bbox: tuple[float, float, float, float]


# Exact Verovio 6.2.1 / Leipzig glyph definitions observed under the already
# pinned D5 renderer.  The random-looking suffix after the SMuFL code is not
# identity-bearing; the code and exact path definition are.
_PINNED_NOTEHEAD_GLYPHS: Final[dict[str, _PinnedNoteheadGlyph]] = {
    "E0A2": _PinnedNoteheadGlyph(
        code="E0A2",
        path_transform="scale(1,-1)",
        path_data=(
            "M198 133c102 0 207 -45 207 -133c0 -92 -118 -133 -227 -133"
            "c-101 0 -178 46 -178 133c0 88 93 133 198 133zM293 -21c0 14"
            " -3 29 -8 44c-7 20 -18 38 -33 54c-20 21 -43 31 -68 31l-20"
            " -2c-15 -5 -27 -14 -36 -28c-4 -9 -6 -17 -8 -24s-3 -16 -3"
            " -27c0 -15 3 -34 9 -57 s18 -41 34 -55c15 -15 36 -23 62"
            " -23c4 0 10 1 18 2c19 5 32 15 40 30s13 34 13 55z"
        ),
        transformed_bbox=(0.0, -133.0, 405.0, 133.0),
    ),
    "E0A3": _PinnedNoteheadGlyph(
        code="E0A3",
        path_transform="scale(1,-1)",
        path_data=(
            "M278 64c0 22 -17 39 -43 39c-12 0 -26 -3 -41 -10c-85 -43"
            " -165 -94 -165 -156c5 -25 15 -32 49 -32c67 11 200 95 200"
            " 159zM0 -36c0 68 73 174 200 174c66 0 114 -39 114 -97c0 -84"
            " -106 -173 -218 -173c-64 0 -96 32 -96 96z"
        ),
        transformed_bbox=(0.0, -138.0, 314.0, 132.0),
    ),
    "E0A4": _PinnedNoteheadGlyph(
        code="E0A4",
        path_transform="scale(1,-1)",
        path_data=(
            "M0 -39c0 68 73 172 200 172c66 0 114 -37 114 -95c0 -84"
            " -106 -171 -218 -171c-64 0 -96 30 -96 94z"
        ),
        transformed_bbox=(0.0, -133.0, 314.0, 133.0),
    ),
}

_NOTEHEAD_CODES_BY_FILL: Final[dict[str, frozenset[str]]] = {
    "open": frozenset({"E0A2", "E0A3"}),
    "filled": frozenset({"E0A4"}),
}
_ACCIDENTAL_CODE_BY_CLASS: Final[dict[str, str]] = {
    "flat": "E260",
    "natural": "E261",
    "sharp": "E262",
}


def _local(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _href(element: ET.Element) -> str:
    return element.attrib.get("href", element.attrib.get(_XLINK_HREF, ""))


def _instrumentation_ancestor(
    element: ET.Element,
    owner: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> bool:
    current = element
    while current is not owner:
        tokens = _d5._class_tokens(current)
        if "bounding-box" in tokens or "content-bounding-box" in tokens:
            return True
        parent = parent_map.get(current)
        if parent is None:
            raise Stage7D12PinnedSymbolShapeError(
                "symbol descendant escaped owning renderer group"
            )
        current = parent
    return False


def _visible_content(
    owner: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> tuple[ET.Element, ...]:
    return tuple(
        element
        for element in owner.iter()
        if element is not owner
        and not _instrumentation_ancestor(element, owner, parent_map)
        and _local(element) != "g"
    )


def _parse_use_matrix(use: ET.Element) -> tuple[float, float, float, float, float, float]:
    match = _USE_TRANSFORM_RE.fullmatch(use.attrib.get("transform", ""))
    if match is None:
        raise Stage7D12PinnedSymbolShapeError(
            "notehead use must have exact translate-then-scale transform shape"
        )
    translate = _d5._parse_numbers(match.group(1))
    scale = _d5._parse_numbers(match.group(2))
    if len(translate) not in {1, 2} or len(scale) not in {1, 2}:
        raise Stage7D12PinnedSymbolShapeError(
            "notehead use transform has unsupported arity"
        )
    tx = translate[0]
    ty = translate[1] if len(translate) == 2 else 0.0
    sx = scale[0]
    sy = scale[1] if len(scale) == 2 else sx
    if sx == 0.0 or sy == 0.0:
        raise Stage7D12PinnedSymbolShapeError("notehead use scale cannot be zero")
    translation = (1.0, 0.0, 0.0, 1.0, tx, ty)
    scaling = (sx, 0.0, 0.0, sy, 0.0, 0.0)
    return _d5._compose(translation, scaling)


def _glyph_definition(
    href: str,
    coordinate_root: ET.Element,
) -> _PinnedNoteheadGlyph:
    if not href.startswith("#") or "-" not in href[1:]:
        raise Stage7D12PinnedSymbolShapeError(
            "notehead use must reference an internal pinned glyph id"
        )
    fragment = href[1:]
    code = fragment.split("-", 1)[0]
    pinned = _PINNED_NOTEHEAD_GLYPHS.get(code)
    if pinned is None:
        raise Stage7D12PinnedSymbolShapeError("unsupported pinned notehead glyph")
    matches = [
        element
        for element in coordinate_root.iter()
        if element.attrib.get("id") == fragment
    ]
    if len(matches) != 1:
        raise Stage7D12PinnedSymbolShapeError(
            "referenced notehead glyph definition must exist exactly once"
        )
    definition = matches[0]
    if _local(definition) != "g" or definition.attrib.get("transform", "").strip():
        raise Stage7D12PinnedSymbolShapeError(
            "notehead glyph definition has unexpected wrapper shape"
        )
    paths = [element for element in definition.iter() if _local(element) == "path"]
    if len(paths) != 1:
        raise Stage7D12PinnedSymbolShapeError(
            "notehead glyph definition must contain exactly one path"
        )
    path = paths[0]
    if (
        path.attrib.get("transform", "") != pinned.path_transform
        or path.attrib.get("d", "") != pinned.path_data
    ):
        raise Stage7D12PinnedSymbolShapeError(
            "pinned notehead glyph definition drifted"
        )
    return pinned


def notehead_instance_bbox(
    note_group: ET.Element,
    fill_class: str,
    coordinate_root: ET.Element,
    parent_map: dict[ET.Element, ET.Element],
) -> tuple[str, AxisAlignedBox]:
    """Return owning note id and exact bbox for one pinned notehead instance."""

    renderer_id = note_group.attrib.get("id", "")
    if not renderer_id:
        raise Stage7D12PinnedSymbolShapeError(
            "owning renderer note id must be non-empty"
        )
    heads = [
        element
        for element in note_group.iter()
        if element is not note_group
        and _d5._is_visible_object_group(element, "notehead")
    ]
    if len(heads) != 1:
        raise Stage7D12PinnedSymbolShapeError(
            "each renderer note must expose exactly one notehead group"
        )
    head = heads[0]
    if head.attrib.get("id", ""):
        raise Stage7D12PinnedSymbolShapeError(
            "pinned Verovio notehead container unexpectedly gained an id"
        )
    content = _visible_content(head, parent_map)
    if len(content) != 1 or _local(content[0]) != "use":
        raise Stage7D12PinnedSymbolShapeError(
            "pinned notehead container must expose exactly one use element"
        )
    use = content[0]
    href = _href(use)
    pinned = _glyph_definition(href, coordinate_root)
    allowed = _NOTEHEAD_CODES_BY_FILL.get(fill_class)
    if allowed is None or pinned.code not in allowed:
        raise Stage7D12PinnedSymbolShapeError(
            "canonical fill class disagrees with pinned notehead glyph"
        )

    parent = parent_map.get(use)
    if parent is None:
        raise Stage7D12PinnedSymbolShapeError(
            "notehead use is outside renderer coordinate root"
        )
    parent_matrix = _d5._cumulative_transform(
        parent, coordinate_root, parent_map
    )
    matrix = _d5._compose(parent_matrix, _parse_use_matrix(use))
    a, b, c, d, _, _ = matrix
    if abs(b) > 1e-12 or abs(c) > 1e-12 or a == 0.0 or d == 0.0:
        raise Stage7D12PinnedSymbolShapeError(
            "notehead instance transform must remain axis-aligned and nonsingular"
        )
    x0, y0, x1, y1 = pinned.transformed_bbox
    corners = (
        _d5._apply_affine(Point2D(x0, y0), matrix),
        _d5._apply_affine(Point2D(x1, y0), matrix),
        _d5._apply_affine(Point2D(x1, y1), matrix),
        _d5._apply_affine(Point2D(x0, y1), matrix),
    )
    box = AxisAlignedBox(
        min(point.x for point in corners),
        min(point.y for point in corners),
        max(point.x for point in corners),
        max(point.y for point in corners),
    )
    return renderer_id, box


def visible_accidental_group(
    note_group: ET.Element,
    expected_class: str | None,
    parent_map: dict[ET.Element, ET.Element],
) -> ET.Element | None:
    """Return the one visible accidental group, ignoring bbox-only placeholders."""

    groups = [
        element
        for element in note_group.iter()
        if element is not note_group
        and _d5._is_visible_object_group(element, "accid")
    ]
    if len(groups) > 1:
        raise Stage7D12PinnedSymbolShapeError(
            "renderer note has multiple accidental containers"
        )
    if not groups:
        if expected_class is not None:
            raise Stage7D12PinnedSymbolShapeError(
                "canonical visible accidental is missing renderer container"
            )
        return None

    group = groups[0]
    content = _visible_content(group, parent_map)
    if not content:
        if expected_class is not None:
            raise Stage7D12PinnedSymbolShapeError(
                "canonical visible accidental mapped to bbox-only placeholder"
            )
        return None
    if len(content) != 1 or _local(content[0]) != "use":
        raise Stage7D12PinnedSymbolShapeError(
            "visible accidental container must expose exactly one use element"
        )
    if expected_class is None:
        raise Stage7D12PinnedSymbolShapeError(
            "renderer exposes visible accidental without canonical intent"
        )
    expected_code = _ACCIDENTAL_CODE_BY_CLASS.get(expected_class)
    if expected_code is None:
        raise Stage7D12PinnedSymbolShapeError(
            "unsupported canonical accidental class"
        )
    href = _href(content[0])
    if not href.startswith(f"#{expected_code}-"):
        raise Stage7D12PinnedSymbolShapeError(
            "canonical accidental class disagrees with pinned renderer glyph"
        )
    return group
