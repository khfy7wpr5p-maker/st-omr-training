"""Bounded deterministic multi-staff runtime geometry slice.

V2 extends the isolated V1 staff detector to return one or more clearly
separated five-line staffs in stable top-to-bottom order.  It still does not
detect measures or musical symbols and remains isolated from D10/D13/TEST.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Final

from .runtime_geometry_ambiguity_v1 import (
    A01_INCOMPLETE_STAFF,
    A02_STAFFS_TOO_CLOSE,
    A03_LOW_VISIBILITY,
    A04_PAGE_CROPPED,
    A05_OVERLAPPING_CANDIDATES,
    A06_IRREGULAR_SPACING,
    A07_EXTRA_LINE_CANDIDATES,
    GeometryAmbiguityReport,
    build_geometry_ambiguity_report,
)
from .runtime_geometry_engine_contract import (
    BoxContract,
    GeometryInputContract,
    LineSegmentContract,
    PageGeometryContract,
    Point2DContract,
    StaffGeometryContract,
    SystemGeometryContract,
)
from .runtime_geometry_engine_v1 import (
    _decode_gray_png,
    _is_staff_window,
    _line_extent,
    _merge_contiguous_rows,
    _qualifying_rows,
)


GEOMETRY_ENGINE_V2_VERSION: Final[str] = "runtime-geometry-engine-v2-multistaff"
MIN_INTER_STAFF_GAP_SPACINGS_MILLI: Final[int] = 1500
EDGE_CROP_MARGIN_SPACINGS_MILLI: Final[int] = 500


@dataclass(frozen=True, slots=True)
class GeometryEngineV2Result:
    page: PageGeometryContract
    candidate_row_centers: tuple[float, ...]
    ambiguity_report: GeometryAmbiguityReport | None = None


def geometry_engine_v2_config_fingerprint() -> str:
    payload = {
        "version": GEOMETRY_ENGINE_V2_VERSION,
        "grouping": "non-overlapping-five-line-windows-top-to-bottom-v1",
        "min_inter_staff_gap_spacings_milli": MIN_INTER_STAFF_GAP_SPACINGS_MILLI,
        "edge_crop_margin_spacings_milli": EDGE_CROP_MARGIN_SPACINGS_MILLI,
        "measure_detection": False,
        "symbol_recognition": False,
        "d10_access": False,
        "d13_access": False,
        "test_split_access": False,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return sha256(raw).hexdigest()


def _indexed_windows(centers: tuple[float, ...]) -> tuple[tuple[int, int, tuple[float, ...]], ...]:
    candidates: list[tuple[int, int, tuple[float, ...]]] = []
    for start in range(max(0, len(centers) - 4)):
        window = centers[start : start + 5]
        if _is_staff_window(window):
            candidates.append((start, start + 4, window))
    return tuple(candidates)


def _ambiguity_result(
    contract: GeometryInputContract,
    centers: tuple[float, ...],
    codes: tuple[str, ...],
) -> GeometryEngineV2Result:
    report = build_geometry_ambiguity_report(codes)
    page = PageGeometryContract(
        normalized_image_sha256=contract.normalized_image_sha256,
        geometry_config_fingerprint=geometry_engine_v2_config_fingerprint(),
        page_width=contract.normalized_width,
        page_height=contract.normalized_height,
        transform=contract.transform,
        systems=(),
        staffs=(),
        measure_proposals=(),
        status="ambiguous",
        reasons=report.active_reasons,
    )
    return GeometryEngineV2Result(page=page, candidate_row_centers=centers, ambiguity_report=report)


def _staff_from_window(image, window: tuple[float, ...], index: int) -> StaffGeometryContract:
    extents = tuple(_line_extent(image, center) for center in window)
    gaps = tuple(next_value - value for value, next_value in zip(window, window[1:]))
    spacing = sum(gaps) / len(gaps)
    x_min = max(0.0, min(start for start, _ in extents))
    x_max = min(float(image.width), max(end + 1.0 for _, end in extents))
    y_min = max(0.0, window[0] - spacing / 2.0)
    y_max = min(float(image.height), window[-1] + spacing / 2.0)
    lines = tuple(
        LineSegmentContract(
            Point2DContract(extent[0], center),
            Point2DContract(extent[1] + 1.0, center),
        )
        for center, extent in zip(window, extents)
    )
    return StaffGeometryContract(
        staff_id=f"staff-{index}",
        system_id="system-1",
        five_staff_lines=lines,
        staff_bbox=BoxContract(x_min, y_min, x_max, y_max),
        staff_spacing=spacing,
    )


def detect_multistaff_geometry_v2(
    normalized_png: bytes,
    contract: GeometryInputContract,
) -> GeometryEngineV2Result:
    """Return all clearly separated five-line staffs, or one canonical AMBIGUOUS report."""
    if not isinstance(contract, GeometryInputContract):
        raise TypeError("contract must be GeometryInputContract")
    image = _decode_gray_png(normalized_png, contract)
    row_groups = _merge_contiguous_rows(_qualifying_rows(image))
    centers = tuple(group[2] for group in row_groups)

    if not centers:
        return _ambiguity_result(contract, centers, (A03_LOW_VISIBILITY,))
    if len(centers) < 5:
        return _ambiguity_result(contract, centers, (A01_INCOMPLETE_STAFF,))

    candidates = _indexed_windows(centers)
    if not candidates:
        return _ambiguity_result(contract, centers, (A06_IRREGULAR_SPACING,))

    # Overlapping valid five-line interpretations are not resolved by guesswork.
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            if max(left[0], right[0]) <= min(left[1], right[1]):
                return _ambiguity_result(contract, centers, (A05_OVERLAPPING_CANDIDATES, A07_EXTRA_LINE_CANDIDATES))

    used_indices: set[int] = set()
    ordered_windows: list[tuple[float, ...]] = []
    for start, end, window in candidates:
        indices = set(range(start, end + 1))
        if used_indices & indices:
            return _ambiguity_result(contract, centers, (A05_OVERLAPPING_CANDIDATES,))
        used_indices.update(indices)
        ordered_windows.append(window)

    if len(used_indices) != len(centers):
        return _ambiguity_result(contract, centers, (A07_EXTRA_LINE_CANDIDATES,))

    staffs = tuple(_staff_from_window(image, window, index + 1) for index, window in enumerate(ordered_windows))

    active_codes: list[str] = []
    for staff in staffs:
        first_center = (staff.five_staff_lines[0].start.y + staff.five_staff_lines[0].end.y) / 2.0
        last_center = (staff.five_staff_lines[-1].start.y + staff.five_staff_lines[-1].end.y) / 2.0
        edge_margin = staff.staff_spacing * EDGE_CROP_MARGIN_SPACINGS_MILLI / 1000.0
        if first_center <= edge_margin or image.height - last_center <= edge_margin:
            active_codes.append(A04_PAGE_CROPPED)

    for upper, lower in zip(staffs, staffs[1:]):
        inter_gap = lower.staff_bbox.y_min - upper.staff_bbox.y_max
        required = max(upper.staff_spacing, lower.staff_spacing) * MIN_INTER_STAFF_GAP_SPACINGS_MILLI / 1000.0
        if inter_gap < required:
            active_codes.append(A02_STAFFS_TOO_CLOSE)

    if active_codes:
        return _ambiguity_result(contract, centers, tuple(active_codes))

    system_bbox = BoxContract(
        min(staff.staff_bbox.x_min for staff in staffs),
        min(staff.staff_bbox.y_min for staff in staffs),
        max(staff.staff_bbox.x_max for staff in staffs),
        max(staff.staff_bbox.y_max for staff in staffs),
    )
    system = SystemGeometryContract(
        system_id="system-1",
        system_bbox=system_bbox,
        staff_ids=tuple(staff.staff_id for staff in staffs),
    )
    page = PageGeometryContract(
        normalized_image_sha256=contract.normalized_image_sha256,
        geometry_config_fingerprint=geometry_engine_v2_config_fingerprint(),
        page_width=contract.normalized_width,
        page_height=contract.normalized_height,
        transform=contract.transform,
        systems=(system,),
        staffs=staffs,
        measure_proposals=(),
        status="accepted",
    )
    return GeometryEngineV2Result(page=page, candidate_row_centers=centers, ambiguity_report=None)
