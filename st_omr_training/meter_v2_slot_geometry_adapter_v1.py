"""Shadow-only Meter V2 slot geometry adapter.

This adapter does not detect staff lines from Meter ROI pixels.  It consumes the
already-accepted upstream runtime geometry plus a Runtime Local ROI artifact and
maps that geometry into the ROI coordinate frame.  The only Meter-specific
output is two visual digit slots (numerator / denominator); it assigns no meter
class and loads no model/checkpoint.

The v1 scope intentionally supports the translation-only source_to_roi transform
emitted by runtime_local_roi_v1.  Any identity/geometry conflict fails closed.
No D10/D11/D13 import, optimizer, TEST access, Resolver wiring, or production
promotion is permitted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Final

from .meter_v2_deterministic_composer_v1 import MeterBox
from .runtime_geometry_engine_contract import PageGeometryContract
from .runtime_local_roi_v1 import RuntimeRoiArtifact


METER_SLOT_GEOMETRY_ADAPTER_V1_VERSION: Final[str] = "meter-v2-slot-geometry-adapter-v1-shadow"

# Frozen from the TRAIN-only staff-relative candidate used before this adapter
# was introduced.  These values are NOT a production acceptance claim; they are
# deliberately unchanged by the later VALIDATION results.
TRAIN_WIDTH_OVER_STAFF_SPACING: Final[float] = 1.5960569245912566
TRAIN_HEIGHT_OVER_STAFF_SPACING: Final[float] = 2.0
NUMERATOR_STAFF_LINE_INDEX: Final[int] = 1  # second line, zero-based
DENOMINATOR_STAFF_LINE_INDEX: Final[int] = 3  # fourth line, zero-based

ACCEPTED: Final[str] = "accepted"
AMBIGUOUS: Final[str] = "ambiguous"
REJECTED: Final[str] = "rejected"
STATUSES: Final[tuple[str, ...]] = (ACCEPTED, AMBIGUOUS, REJECTED)

R_NONFINITE: Final[str] = "METER_SLOT_NONFINITE_OR_RANGE"
R_UPSTREAM_NOT_ACCEPTED: Final[str] = "METER_SLOT_UPSTREAM_GEOMETRY_NOT_ACCEPTED"
R_ROI_KIND: Final[str] = "METER_SLOT_ROI_NOT_MEASURE_START"
R_IDENTITY: Final[str] = "METER_SLOT_IDENTITY_MISMATCH"
R_STAFF_NOT_FOUND: Final[str] = "METER_SLOT_STAFF_NOT_FOUND"
R_MEASURE_NOT_FOUND: Final[str] = "METER_SLOT_MEASURE_NOT_FOUND"
R_TRANSFORM: Final[str] = "METER_SLOT_UNSUPPORTED_ROI_TRANSFORM"
R_X_OUTSIDE_STAFF: Final[str] = "METER_SLOT_X_OUTSIDE_STAFF_SUPPORT"
R_STAFF_GEOMETRY: Final[str] = "METER_SLOT_STAFF_GEOMETRY_AMBIGUOUS"
R_BBOX_OUTSIDE_ROI: Final[str] = "METER_SLOT_BBOX_OUTSIDE_ROI"

_EPS: Final[float] = 1e-9


@dataclass(frozen=True, slots=True)
class MeterSlotGeometryResult:
    status: str
    measure_id: str
    staff_id: str
    roi_id: str
    profile_fingerprint: str
    local_staff_spacing: float | None
    numerator_bbox: MeterBox | None
    denominator_bbox: MeterBox | None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError("unsupported Meter slot geometry status")
        if not self.measure_id or not self.staff_id or not self.roi_id:
            raise ValueError("Meter slot geometry identities must be non-empty")
        if len(self.profile_fingerprint) != 64 or any(c not in "0123456789abcdef" for c in self.profile_fingerprint):
            raise ValueError("Meter slot geometry profile fingerprint must be SHA-256")
        if self.status == ACCEPTED:
            if self.reasons:
                raise ValueError("accepted Meter slot geometry cannot carry reasons")
            if self.numerator_bbox is None or self.denominator_bbox is None:
                raise ValueError("accepted Meter slot geometry requires two boxes")
            if (
                self.local_staff_spacing is None
                or not math.isfinite(self.local_staff_spacing)
                or self.local_staff_spacing <= 0
            ):
                raise ValueError("accepted Meter slot geometry requires positive spacing")
        else:
            if not self.reasons:
                raise ValueError("non-accepted Meter slot geometry must explain why")
            if self.numerator_bbox is not None or self.denominator_bbox is not None:
                raise ValueError("non-accepted Meter slot geometry cannot expose boxes")


def meter_slot_geometry_adapter_v1_profile_fingerprint() -> str:
    payload = {
        "version": METER_SLOT_GEOMETRY_ADAPTER_V1_VERSION,
        "input_staff_geometry": "upstream-runtime-geometry-only",
        "input_roi": "runtime-local-roi-v1-measure-start",
        "source_to_roi": "translation-only-v1",
        "x_coordinate": "caller-supplied-train-derived-refined-x-in-roi",
        "numerator_staff_line_index": NUMERATOR_STAFF_LINE_INDEX,
        "denominator_staff_line_index": DENOMINATOR_STAFF_LINE_INDEX,
        "width_over_staff_spacing": TRAIN_WIDTH_OVER_STAFF_SPACING,
        "height_over_staff_spacing": TRAIN_HEIGHT_OVER_STAFF_SPACING,
        "measure_index_special_cases": False,
        "roi_staff_redetection": False,
        "validation_tuning": False,
        "model_loading": False,
        "test_split_access": False,
        "resolver_wiring": False,
        "production_accepted": False,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return sha256(raw).hexdigest()


def _result(
    status: str,
    roi: RuntimeRoiArtifact,
    *,
    spacing: float | None = None,
    numerator: MeterBox | None = None,
    denominator: MeterBox | None = None,
    reasons: tuple[str, ...] = (),
) -> MeterSlotGeometryResult:
    return MeterSlotGeometryResult(
        status=status,
        measure_id=roi.measure_id,
        staff_id=roi.staff_id,
        roi_id=roi.roi_id,
        profile_fingerprint=meter_slot_geometry_adapter_v1_profile_fingerprint(),
        local_staff_spacing=spacing,
        numerator_bbox=numerator,
        denominator_bbox=denominator,
        reasons=reasons,
    )


def _translation_matches_roi(roi: RuntimeRoiArtifact) -> bool:
    f = roi.source_to_roi.forward
    i = roi.source_to_roi.inverse
    left = float(roi.crop_bbox.x_min)
    top = float(roi.crop_bbox.y_min)
    expected_f = (1.0, 0.0, -left, 0.0, 1.0, -top, 0.0, 0.0, 1.0)
    expected_i = (1.0, 0.0, left, 0.0, 1.0, top, 0.0, 0.0, 1.0)
    return all(abs(a - b) <= _EPS for a, b in zip(f, expected_f)) and all(
        abs(a - b) <= _EPS for a, b in zip(i, expected_i)
    )


def _line_y_at_source_x(line, source_x: float) -> float | None:
    x0 = float(line.start.x)
    x1 = float(line.end.x)
    y0 = float(line.start.y)
    y1 = float(line.end.y)
    lo = min(x0, x1)
    hi = max(x0, x1)
    if hi - lo <= _EPS or source_x < lo - _EPS or source_x > hi + _EPS:
        return None
    t = (source_x - x0) / (x1 - x0)
    y = y0 + t * (y1 - y0)
    return y if math.isfinite(y) else None


def _box(center_x: float, center_y: float, spacing: float) -> MeterBox:
    width = spacing * TRAIN_WIDTH_OVER_STAFF_SPACING
    height = spacing * TRAIN_HEIGHT_OVER_STAFF_SPACING
    return MeterBox(
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


def _inside_roi(box: MeterBox, roi: RuntimeRoiArtifact) -> bool:
    width = float(roi.crop_bbox.x_max - roi.crop_bbox.x_min)
    height = float(roi.crop_bbox.y_max - roi.crop_bbox.y_min)
    values = (box.x0, box.y0, box.x1, box.y1)
    if not all(math.isfinite(float(v)) for v in values):
        return False
    return 0.0 <= box.x0 < box.x1 <= width and 0.0 <= box.y0 < box.y1 <= height


def adapt_meter_slots_from_upstream_geometry_v1(
    geometry: PageGeometryContract,
    roi: RuntimeRoiArtifact,
    *,
    refined_x_center_roi: float,
) -> MeterSlotGeometryResult:
    """Map accepted upstream staff geometry into two Meter digit slots.

    ``refined_x_center_roi`` remains the TRAIN-derived deterministic horizontal
    anchor/refinement boundary.  This adapter owns only geometry provenance and
    staff-relative vertical placement; it never re-detects staff lines in pixels.
    """
    if not isinstance(geometry, PageGeometryContract) or not isinstance(roi, RuntimeRoiArtifact):
        raise TypeError("geometry/roi must use the runtime contracts")
    if (
        isinstance(refined_x_center_roi, bool)
        or not isinstance(refined_x_center_roi, (int, float))
        or not math.isfinite(float(refined_x_center_roi))
    ):
        return _result(REJECTED, roi, reasons=(R_NONFINITE,))
    x_roi = float(refined_x_center_roi)

    if geometry.status != ACCEPTED:
        return _result(AMBIGUOUS, roi, reasons=(R_UPSTREAM_NOT_ACCEPTED,))
    if roi.kind != "measure-start":
        return _result(REJECTED, roi, reasons=(R_ROI_KIND,))
    if roi.source_image_sha256 != geometry.normalized_image_sha256:
        return _result(REJECTED, roi, reasons=(R_IDENTITY,))
    if not _translation_matches_roi(roi):
        return _result(REJECTED, roi, reasons=(R_TRANSFORM,))

    staffs = tuple(item for item in geometry.staffs if item.staff_id == roi.staff_id)
    if len(staffs) != 1:
        return _result(REJECTED, roi, reasons=(R_STAFF_NOT_FOUND,))
    staff = staffs[0]

    measures = tuple(
        item
        for item in geometry.measure_proposals
        if item.measure_id == roi.measure_id and item.staff_id == roi.staff_id and item.status == ACCEPTED
    )
    if len(measures) != 1:
        return _result(REJECTED, roi, reasons=(R_MEASURE_NOT_FOUND,))
    measure = measures[0]

    # Runtime Local ROI v1 must be a clipped left-side sub-box of this measure.
    c = roi.crop_bbox
    m = measure.bbox
    if (
        c.x_min < m.x_min - _EPS
        or c.x_max > m.x_max + _EPS
        or c.y_min < m.y_min - _EPS
        or c.y_max > m.y_max + _EPS
    ):
        return _result(REJECTED, roi, reasons=(R_IDENTITY,))

    roi_width = float(c.x_max - c.x_min)
    if x_roi < 0.0 or x_roi > roi_width:
        return _result(REJECTED, roi, reasons=(R_NONFINITE,))
    source_x = x_roi + float(c.x_min)

    source_line_y: list[float] = []
    for line in staff.five_staff_lines:
        value = _line_y_at_source_x(line, source_x)
        if value is None:
            return _result(AMBIGUOUS, roi, reasons=(R_X_OUTSIDE_STAFF,))
        source_line_y.append(value)

    roi_line_y = tuple(value - float(c.y_min) for value in source_line_y)
    gaps = tuple(b - a for a, b in zip(roi_line_y, roi_line_y[1:]))
    if len(gaps) != 4 or any(not math.isfinite(g) or g <= 0.0 for g in gaps):
        return _result(AMBIGUOUS, roi, reasons=(R_STAFF_GEOMETRY,))
    local_spacing = sum(gaps) / 4.0
    if not math.isfinite(local_spacing) or local_spacing <= 0.0:
        return _result(AMBIGUOUS, roi, reasons=(R_STAFF_GEOMETRY,))

    numerator = _box(x_roi, roi_line_y[NUMERATOR_STAFF_LINE_INDEX], local_spacing)
    denominator = _box(x_roi, roi_line_y[DENOMINATOR_STAFF_LINE_INDEX], local_spacing)
    if not _inside_roi(numerator, roi) or not _inside_roi(denominator, roi):
        return _result(AMBIGUOUS, roi, reasons=(R_BBOX_OUTSIDE_ROI,))

    return _result(
        ACCEPTED,
        roi,
        spacing=local_spacing,
        numerator=numerator,
        denominator=denominator,
    )


def meter_slot_geometry_adapter_train_derived_only() -> bool:
    return True


def meter_slot_geometry_adapter_validation_tuned() -> bool:
    return False


def meter_slot_geometry_adapter_production_accepted() -> bool:
    return False


def resolver_connection_allowed() -> bool:
    return False
