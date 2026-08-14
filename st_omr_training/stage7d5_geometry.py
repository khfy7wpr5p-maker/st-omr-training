"""Stage 7-D5 geometry public compatibility surface used by Stage 7-D6.

The accepted D5-v1 implementation is preserved byte-for-byte in
``_stage7d5_geometry_v1``.  The authoritative D6 corpus run exposed a pinned
Verovio 6.2.1 layout case where a measure group contains both its current time
signature and a courtesy time signature for the following measure.  The
courtesy group is rendered to the right of the current measure's trailing
barline.

V2 keeps every accepted D5 behavior but makes meterSig selection measure-local:
only a visible meterSig whose bbox lies fully on or before the trailing barline
may be bound to the current measure.  Post-barline groups are courtesy material
and are excluded.  Multiple pre-barline candidates or a bbox crossing the
barline still fail closed.
"""

from __future__ import annotations

from . import _stage7d5_geometry_v1 as _legacy
from ._stage7d5_geometry_v1 import *  # noqa: F401,F403


STAGE7D5_GEOMETRY_VERSION = "stage7d5-staff-structure-geometry-v2"

# Existing public functions execute in the preserved module's global namespace.
# Version and selector updates therefore have to be installed there as well.
_legacy.STAGE7D5_GEOMETRY_VERSION = STAGE7D5_GEOMETRY_VERSION

# Preserve the accepted v1 selector for every class except meterSig.
_V1_OPTIONAL_OBJECT_BBOX = _legacy._optional_object_bbox

# Existing regression tests intentionally exercise this private coordinate
# helper, so keep the same import surface while the implementation stays frozen.
_coordinate_root = _legacy._coordinate_root


def _group_bbox(group, coordinate_root, parent_map):
    """Resolve one renderer object's bbox with the accepted D5 fallback rule."""

    try:
        return _legacy._bbox_for_group(
            group,
            content=False,
            coordinate_root=coordinate_root,
            parent_map=parent_map,
        )
    except _legacy.Stage7D5GeometryError:
        return _legacy._bbox_for_group(
            group,
            content=True,
            coordinate_root=coordinate_root,
            parent_map=parent_map,
        )


def _optional_object_bbox(
    measure,
    class_name,
    coordinate_root,
    parent_map,
):
    """Return the current-measure object bbox, excluding courtesy meterSig.

    Pinned Verovio can place the next measure's courtesy time signature inside
    the preceding measure's SVG group, after that measure's trailing barline.
    Geometry to the right of the barline is therefore not authoritative for the
    current measure.
    """

    if class_name != "meterSig":
        return _V1_OPTIONAL_OBJECT_BBOX(
            measure,
            class_name,
            coordinate_root,
            parent_map,
        )

    groups = [
        element
        for element in measure.iter()
        if _legacy._is_visible_object_group(element, "meterSig")
    ]
    if not groups:
        return None

    barline = _legacy._trailing_barline(measure, coordinate_root, parent_map)
    trailing_x = max(barline.start.x, barline.end.x)
    epsilon = 1e-7

    current = []
    for group in groups:
        box = _group_bbox(group, coordinate_root, parent_map)
        if box.x_max <= trailing_x + epsilon:
            current.append(box)
            continue
        if box.x_min >= trailing_x - epsilon:
            # Verovio courtesy/anticipatory meter for the following measure.
            continue
        raise _legacy.Stage7D5GeometryError(
            "meterSig renderer bbox crosses the current measure trailing barline"
        )

    if not current:
        return None
    if len(current) != 1:
        raise _legacy.Stage7D5GeometryError(
            "measure has ambiguous current meterSig renderer groups"
        )
    return current[0]


# Install the v2 selector in the frozen implementation namespace.  Functions
# such as extract_staff_structure_geometry retain their original code object and
# now resolve this single versioned compatibility hook at runtime.
_legacy._optional_object_bbox = _optional_object_bbox
