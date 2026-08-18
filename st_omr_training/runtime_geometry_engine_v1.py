"""First bounded deterministic runtime slice for ST Geometry Engine.

V1 answers only one narrow question: does an accepted normalized gray page
contain exactly one unambiguous five-line staff?  It does not detect measures,
barlines, symbols, pitch, rhythm, or any musical semantics. Multiple plausible
staffs or incomplete/irregular line patterns fail closed as ambiguous.
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
    GeometryInputContract,
    LineSegmentContract,
    PageGeometryContract,
    Point2DContract,
    StaffGeometryContract,
    SystemGeometryContract,
)


GEOMETRY_ENGINE_V1_VERSION: Final[str] = "runtime-geometry-engine-v1-staff-only"
ROW_DARK_THRESHOLD: Final[int] = 128
MIN_HORIZONTAL_DARK_FRACTION_MILLI: Final[int] = 550
MIN_STAFF_SPACING_PX: Final[int] = 4
MAX_STAFF_SPACING_PX: Final[int] = 80
MAX_GAP_DEVIATION_MILLI: Final[int] = 150


class GeometryEngineV1Error(ValueError):
    """Raised when input bytes violate the frozen runtime boundary."""


@dataclass(frozen=True, slots=True)
class GeometryEngineV1Result:
    page: PageGeometryContract
    candidate_row_centers: tuple[float, ...]


def _config_payload() -> dict[str, object]:
    return {
        "version": GEOMETRY_ENGINE_V1_VERSION,
        "row_dark_threshold": ROW_DARK_THRESHOLD,
        "min_horizontal_dark_fraction_milli": MIN_HORIZONTAL_DARK_FRACTION_MILLI,
        "min_staff_spacing_px": MIN_STAFF_SPACING_PX,
        "max_staff_spacing_px": MAX_STAFF_SPACING_PX,
        "max_gap_deviation_milli": MAX_GAP_DEVIATION_MILLI,
        "supported_surface": "exactly-one-horizontal-five-line-staff",
        "measure_detection": False,
        "symbol_recognition": False,
    }


def geometry_engine_v1_config_fingerprint() -> str:
    raw = json.dumps(
        _config_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256(raw).hexdigest()


def _decode_gray_png(data: bytes, contract: GeometryInputContract) -> Image.Image:
    if not isinstance(data, bytes) or not data:
        raise GeometryEngineV1Error("normalized image bytes must be non-empty bytes")
    if sha256(data).hexdigest() != contract.normalized_image_sha256:
        raise GeometryEngineV1Error("normalized image SHA does not match geometry input")
    try:
        with Image.open(BytesIO(data)) as opened:
            if opened.format != "PNG":
                raise GeometryEngineV1Error("geometry v1 accepts normalized PNG only")
            opened.load()
            if opened.mode != "L":
                raise GeometryEngineV1Error("geometry v1 requires gray8 normalized PNG")
            if opened.size != (contract.normalized_width, contract.normalized_height):
                raise GeometryEngineV1Error("normalized PNG dimensions do not match contract")
            return opened.copy()
    except GeometryEngineV1Error:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise GeometryEngineV1Error("normalized PNG cannot be decoded safely") from exc


def _dark_runs_for_row(image: Image.Image, y: int) -> tuple[tuple[int, int], ...]:
    pixels = image.load()
    width = image.width
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for x in range(width):
        is_dark = pixels[x, y] <= ROW_DARK_THRESHOLD
        if is_dark and start is None:
            start = x
        elif not is_dark and start is not None:
            runs.append((start, x - 1))
            start = None
    if start is not None:
        runs.append((start, width - 1))
    return tuple(runs)


def _qualifying_rows(image: Image.Image) -> tuple[int, ...]:
    minimum_run = max(1, math.ceil(image.width * MIN_HORIZONTAL_DARK_FRACTION_MILLI / 1000))
    rows: list[int] = []
    for y in range(image.height):
        runs = _dark_runs_for_row(image, y)
        if any(end - start + 1 >= minimum_run for start, end in runs):
            rows.append(y)
    return tuple(rows)


def _merge_contiguous_rows(rows: tuple[int, ...]) -> tuple[tuple[int, int, float], ...]:
    if not rows:
        return ()
    groups: list[tuple[int, int, float]] = []
    start = rows[0]
    previous = rows[0]
    for value in rows[1:]:
        if value == previous + 1:
            previous = value
            continue
        groups.append((start, previous, (start + previous) / 2.0))
        start = previous = value
    groups.append((start, previous, (start + previous) / 2.0))
    return tuple(groups)


def _is_staff_window(centers: tuple[float, ...]) -> bool:
    if len(centers) != 5:
        return False
    gaps = tuple(next_value - value for value, next_value in zip(centers, centers[1:]))
    mean_gap = sum(gaps) / len(gaps)
    if not MIN_STAFF_SPACING_PX <= mean_gap <= MAX_STAFF_SPACING_PX:
        return False
    maximum_allowed = max(0.75, mean_gap * MAX_GAP_DEVIATION_MILLI / 1000)
    return all(abs(gap - mean_gap) <= maximum_allowed for gap in gaps)


def _staff_windows(centers: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
    if len(centers) < 5:
        return ()
    return tuple(
        window
        for index in range(len(centers) - 4)
        if _is_staff_window(window := centers[index : index + 5])
    )


def _line_extent(image: Image.Image, y: float) -> tuple[float, float]:
    row = min(max(int(round(y)), 0), image.height - 1)
    runs = _dark_runs_for_row(image, row)
    minimum_run = max(1, math.ceil(image.width * MIN_HORIZONTAL_DARK_FRACTION_MILLI / 1000))
    long_runs = tuple((start, end) for start, end in runs if end - start + 1 >= minimum_run)
    if len(long_runs) != 1:
        raise GeometryEngineV1Error("accepted staff line must have one dominant horizontal run")
    start, end = long_runs[0]
    return float(start), float(end)


def _ambiguous_page(
    contract: GeometryInputContract,
    reason: str,
) -> PageGeometryContract:
    return PageGeometryContract(
        normalized_image_sha256=contract.normalized_image_sha256,
        geometry_config_fingerprint=geometry_engine_v1_config_fingerprint(),
        page_width=contract.normalized_width,
        page_height=contract.normalized_height,
        transform=contract.transform,
        systems=(),
        staffs=(),
        measure_proposals=(),
        status="ambiguous",
        reasons=(reason,),
    )


def detect_single_staff_geometry_v1(
    normalized_png: bytes,
    contract: GeometryInputContract,
) -> GeometryEngineV1Result:
    """Detect exactly one clear horizontal five-line staff, otherwise abstain."""

    if not isinstance(contract, GeometryInputContract):
        raise TypeError("contract must be GeometryInputContract")
    image = _decode_gray_png(normalized_png, contract)
    row_groups = _merge_contiguous_rows(_qualifying_rows(image))
    centers = tuple(group[2] for group in row_groups)
    windows = _staff_windows(centers)

    if len(windows) == 0:
        return GeometryEngineV1Result(
            page=_ambiguous_page(contract, "no-unambiguous-five-line-staff"),
            candidate_row_centers=centers,
        )
    if len(windows) != 1 or len(centers) != 5:
        return GeometryEngineV1Result(
            page=_ambiguous_page(contract, "multiple-or-overlapping-staff-candidates"),
            candidate_row_centers=centers,
        )

    staff_centers = windows[0]
    extents = tuple(_line_extent(image, center) for center in staff_centers)
    x_min = max(0.0, min(start for start, _ in extents))
    x_max = min(float(image.width), max(end + 1.0 for _, end in extents))
    gaps = tuple(next_value - value for value, next_value in zip(staff_centers, staff_centers[1:]))
    spacing = sum(gaps) / len(gaps)
    y_min = max(0.0, staff_centers[0] - spacing / 2.0)
    y_max = min(float(image.height), staff_centers[-1] + spacing / 2.0)
    if x_max <= x_min or y_max <= y_min:
        return GeometryEngineV1Result(
            page=_ambiguous_page(contract, "staff-bounds-not-safe"),
            candidate_row_centers=centers,
        )

    lines = tuple(
        LineSegmentContract(
            Point2DContract(extent[0], center),
            Point2DContract(extent[1] + 1.0, center),
        )
        for center, extent in zip(staff_centers, extents)
    )
    staff = StaffGeometryContract(
        staff_id="staff-1",
        system_id="system-1",
        five_staff_lines=lines,
        staff_bbox=BoxContract(x_min, y_min, x_max, y_max),
        staff_spacing=spacing,
    )
    system = SystemGeometryContract(
        system_id="system-1",
        system_bbox=staff.staff_bbox,
        staff_ids=(staff.staff_id,),
    )
    page = PageGeometryContract(
        normalized_image_sha256=contract.normalized_image_sha256,
        geometry_config_fingerprint=geometry_engine_v1_config_fingerprint(),
        page_width=contract.normalized_width,
        page_height=contract.normalized_height,
        transform=contract.transform,
        systems=(system,),
        staffs=(staff,),
        measure_proposals=(),
        status="accepted",
    )
    return GeometryEngineV1Result(page=page, candidate_row_centers=centers)
