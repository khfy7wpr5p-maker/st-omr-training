"""Deterministic runtime measure-boundary proposal layer.

This layer consumes accepted staff geometry and searches only for strong
vertical separators spanning the staff.  The separators are geometry proposals,
not semantic barline recognition.  It has no D10/D13/model/TEST dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from typing import Final

from PIL import Image, UnidentifiedImageError

from .runtime_geometry_engine_contract import (
    BoxContract,
    LineSegmentContract,
    MeasureProposalContract,
    PageGeometryContract,
    Point2DContract,
)


MEASURE_GEOMETRY_V1_VERSION: Final[str] = "runtime-measure-geometry-v1"
VERTICAL_DARK_THRESHOLD: Final[int] = 128
MIN_VERTICAL_COVERAGE_MILLI: Final[int] = 800
MIN_MEASURE_WIDTH_SPACINGS_MILLI: Final[int] = 2000
MAX_CROSS_STAFF_BOUNDARY_DELTA_SPACINGS_MILLI: Final[int] = 500

M01_INSUFFICIENT_BOUNDARIES: Final[str] = "M01_INSUFFICIENT_BOUNDARIES"
M02_MEASURE_TOO_NARROW: Final[str] = "M02_MEASURE_TOO_NARROW"
M03_CROSS_STAFF_BOUNDARY_MISMATCH: Final[str] = "M03_CROSS_STAFF_BOUNDARY_MISMATCH"


@dataclass(frozen=True, slots=True)
class MeasureGeometryV1Result:
    page: PageGeometryContract
    boundary_x_by_staff: tuple[tuple[str, tuple[float, ...]], ...]


def measure_geometry_v1_config_fingerprint(parent_geometry_fingerprint: str) -> str:
    payload = {
        "version": MEASURE_GEOMETRY_V1_VERSION,
        "parent_geometry_fingerprint": parent_geometry_fingerprint,
        "vertical_dark_threshold": VERTICAL_DARK_THRESHOLD,
        "min_vertical_coverage_milli": MIN_VERTICAL_COVERAGE_MILLI,
        "min_measure_width_spacings_milli": MIN_MEASURE_WIDTH_SPACINGS_MILLI,
        "max_cross_staff_boundary_delta_spacings_milli": MAX_CROSS_STAFF_BOUNDARY_DELTA_SPACINGS_MILLI,
        "semantic_barline_recognition": False,
        "d10_access": False,
        "d13_access": False,
        "test_split_access": False,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return sha256(raw).hexdigest()


def _decode_page(normalized_png: bytes, geometry: PageGeometryContract) -> Image.Image:
    if not isinstance(normalized_png, bytes) or not normalized_png:
        raise ValueError("normalized_png must be non-empty bytes")
    if sha256(normalized_png).hexdigest() != geometry.normalized_image_sha256:
        raise ValueError("normalized PNG identity does not match geometry")
    try:
        with Image.open(BytesIO(normalized_png)) as opened:
            if opened.format != "PNG" or opened.mode != "L":
                raise ValueError("measure geometry requires normalized gray8 PNG")
            opened.load()
            if opened.size != (geometry.page_width, geometry.page_height):
                raise ValueError("normalized PNG dimensions do not match geometry")
            return opened.copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("normalized PNG cannot be decoded safely") from exc


def _merge_columns(columns: tuple[int, ...]) -> tuple[float, ...]:
    if not columns:
        return ()
    centers: list[float] = []
    start = previous = columns[0]
    for value in columns[1:]:
        if value == previous + 1:
            previous = value
            continue
        centers.append((start + previous) / 2.0)
        start = previous = value
    centers.append((start + previous) / 2.0)
    return tuple(centers)


def _boundary_candidates(image: Image.Image, staff) -> tuple[float, ...]:
    top = int(round(staff.five_staff_lines[0].start.y))
    bottom = int(round(staff.five_staff_lines[-1].start.y))
    top = max(0, min(top, image.height - 1))
    bottom = max(top, min(bottom, image.height - 1))
    span = bottom - top + 1
    minimum_dark = math.ceil(span * MIN_VERTICAL_COVERAGE_MILLI / 1000)
    left = max(0, int(math.floor(staff.staff_bbox.x_min)))
    right = min(image.width - 1, int(math.ceil(staff.staff_bbox.x_max)) - 1)
    pixels = image.load()
    columns: list[int] = []
    for x in range(left, right + 1):
        dark_count = sum(1 for y in range(top, bottom + 1) if pixels[x, y] <= VERTICAL_DARK_THRESHOLD)
        if dark_count >= minimum_dark:
            columns.append(x)
    return _merge_columns(tuple(columns))


def _ambiguous_page(geometry: PageGeometryContract, reason: str) -> PageGeometryContract:
    return PageGeometryContract(
        normalized_image_sha256=geometry.normalized_image_sha256,
        geometry_config_fingerprint=measure_geometry_v1_config_fingerprint(geometry.geometry_config_fingerprint),
        page_width=geometry.page_width,
        page_height=geometry.page_height,
        transform=geometry.transform,
        systems=geometry.systems,
        staffs=geometry.staffs,
        measure_proposals=(),
        status="ambiguous",
        reasons=(reason,),
    )


def propose_measure_geometry_v1(
    normalized_png: bytes,
    geometry: PageGeometryContract,
) -> MeasureGeometryV1Result:
    """Propose deterministic per-staff measure boxes from strong vertical separators."""
    if not isinstance(geometry, PageGeometryContract):
        raise TypeError("geometry must be PageGeometryContract")
    if geometry.status != "accepted" or not geometry.staffs:
        raise ValueError("measure geometry requires accepted non-empty staff geometry")
    image = _decode_page(normalized_png, geometry)

    boundary_map = tuple(
        (staff.staff_id, _boundary_candidates(image, staff)) for staff in geometry.staffs
    )
    if any(len(boundaries) < 2 for _, boundaries in boundary_map):
        page = _ambiguous_page(geometry, M01_INSUFFICIENT_BOUNDARIES)
        return MeasureGeometryV1Result(page=page, boundary_x_by_staff=boundary_map)

    # Staffs in one system must describe the same vertical measure structure.
    system_by_id = {system.system_id: system for system in geometry.systems}
    staff_by_id = {staff.staff_id: staff for staff in geometry.staffs}
    boundary_dict = dict(boundary_map)
    for system in geometry.systems:
        system_staffs = tuple(staff_by_id[staff_id] for staff_id in system.staff_ids)
        if len(system_staffs) <= 1:
            continue
        reference = boundary_dict[system_staffs[0].staff_id]
        for staff in system_staffs[1:]:
            current = boundary_dict[staff.staff_id]
            if len(current) != len(reference):
                page = _ambiguous_page(geometry, M03_CROSS_STAFF_BOUNDARY_MISMATCH)
                return MeasureGeometryV1Result(page=page, boundary_x_by_staff=boundary_map)
            tolerance = max(system_staffs[0].staff_spacing, staff.staff_spacing) * (
                MAX_CROSS_STAFF_BOUNDARY_DELTA_SPACINGS_MILLI / 1000.0
            )
            if any(abs(a - b) > tolerance for a, b in zip(reference, current)):
                page = _ambiguous_page(geometry, M03_CROSS_STAFF_BOUNDARY_MISMATCH)
                return MeasureGeometryV1Result(page=page, boundary_x_by_staff=boundary_map)

    measures: list[MeasureProposalContract] = []
    for staff in geometry.staffs:
        boundaries = boundary_dict[staff.staff_id]
        for index, (left_x, right_x) in enumerate(zip(boundaries, boundaries[1:]), start=1):
            minimum_width = staff.staff_spacing * MIN_MEASURE_WIDTH_SPACINGS_MILLI / 1000.0
            if right_x - left_x < minimum_width:
                page = _ambiguous_page(geometry, M02_MEASURE_TOO_NARROW)
                return MeasureGeometryV1Result(page=page, boundary_x_by_staff=boundary_map)
            bbox = BoxContract(
                left_x,
                staff.staff_bbox.y_min,
                right_x,
                staff.staff_bbox.y_max,
            )
            left_boundary = LineSegmentContract(
                Point2DContract(left_x, staff.staff_bbox.y_min),
                Point2DContract(left_x, staff.staff_bbox.y_max),
            )
            right_boundary = LineSegmentContract(
                Point2DContract(right_x, staff.staff_bbox.y_min),
                Point2DContract(right_x, staff.staff_bbox.y_max),
            )
            measures.append(
                MeasureProposalContract(
                    measure_id=f"{staff.staff_id}-measure-{index}",
                    system_id=staff.system_id,
                    staff_id=staff.staff_id,
                    bbox=bbox,
                    left_boundary=left_boundary,
                    right_boundary=right_boundary,
                    status="accepted",
                )
            )

    page = PageGeometryContract(
        normalized_image_sha256=geometry.normalized_image_sha256,
        geometry_config_fingerprint=measure_geometry_v1_config_fingerprint(geometry.geometry_config_fingerprint),
        page_width=geometry.page_width,
        page_height=geometry.page_height,
        transform=geometry.transform,
        systems=geometry.systems,
        staffs=geometry.staffs,
        measure_proposals=tuple(measures),
        status="accepted",
    )
    return MeasureGeometryV1Result(page=page, boundary_x_by_staff=boundary_map)
