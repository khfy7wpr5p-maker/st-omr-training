"""Reconstruct the frozen D9/D10 Meter ROI exactly from runtime source pixels.

This runtime adapter consumes the normalized page bytes and accepted runtime
geometry. It deliberately does not import Stage 7-D10 implementation code, but
reproduces the frozen D9/D10 Meter ROI policy and rendering semantics exactly:
measure-start anchor; x margins -0.5/+12 staff spacings; staff-bbox y margins
+/-3 staff spacings; floor/ceil+clip crop; fit-pad to 256x192; BILINEAR resize;
white canvas; deterministic PNG encoding.

No model/checkpoint is loaded and no Resolver path is invoked here.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from typing import Final

from PIL import Image, UnidentifiedImageError

from .runtime_geometry_engine_contract import PageGeometryContract

HISTORICAL_METER_ROI_ADAPTER_V1: Final[str] = "runtime-meter-historical-roi-adapter-v1"
HISTORICAL_POLICY_ID: Final[str] = "measure-start-meter-roi-v1"
X_BEFORE_MILLI: Final[int] = 500
X_AFTER_MILLI: Final[int] = 12_000
Y_BEFORE_MILLI: Final[int] = 3_000
Y_AFTER_MILLI: Final[int] = 3_000
OUTPUT_WIDTH: Final[int] = 256
OUTPUT_HEIGHT: Final[int] = 192


class HistoricalMeterRoiError(RuntimeError):
    pass


def _fp(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def historical_meter_roi_profile_fingerprint_v1() -> str:
    return _fp({
        "version": HISTORICAL_METER_ROI_ADAPTER_V1,
        "policy_id": HISTORICAL_POLICY_ID,
        "x_before_staff_spacings_milli": X_BEFORE_MILLI,
        "x_after_staff_spacings_milli": X_AFTER_MILLI,
        "y_before_staff_spacings_milli": Y_BEFORE_MILLI,
        "y_after_staff_spacings_milli": Y_AFTER_MILLI,
        "output_width": OUTPUT_WIDTH,
        "output_height": OUTPUT_HEIGHT,
        "rounding": "floor-left-top-ceil-right-bottom-clip",
        "resize": "fit-pad-preserve-aspect-bilinear",
        "canvas": "L-white-255",
        "png": "optimize-false-compress-level-9",
    })


@dataclass(frozen=True, slots=True)
class HistoricalMeterRoiArtifactV1:
    measure_id: str
    staff_id: str
    source_image_sha256: str
    image_sha256: str
    crop_box: tuple[int, int, int, int]
    resized_size: tuple[int, int]
    pad_left: int
    pad_top: int
    profile_fingerprint: str
    png_bytes: bytes

    def __post_init__(self) -> None:
        if sha256(self.png_bytes).hexdigest() != self.image_sha256:
            raise ValueError("historical Meter ROI image SHA mismatch")
        if self.profile_fingerprint != historical_meter_roi_profile_fingerprint_v1():
            raise ValueError("historical Meter ROI profile mismatch")


def _decode_source(normalized_png: bytes, geometry: PageGeometryContract) -> Image.Image:
    if not isinstance(normalized_png, bytes) or not normalized_png:
        raise HistoricalMeterRoiError("normalized source PNG bytes are required")
    if sha256(normalized_png).hexdigest() != geometry.normalized_image_sha256:
        raise HistoricalMeterRoiError("normalized source SHA does not match accepted geometry")
    try:
        with Image.open(BytesIO(normalized_png)) as opened:
            if opened.format != "PNG" or opened.mode != "L":
                raise HistoricalMeterRoiError("historical Meter adapter requires gray8 PNG")
            opened.load()
            if opened.size != (geometry.page_width, geometry.page_height):
                raise HistoricalMeterRoiError("normalized source dimensions do not match geometry")
            return opened.copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise HistoricalMeterRoiError("normalized source PNG cannot be decoded") from exc


def reconstruct_historical_meter_roi_v1(
    normalized_png: bytes,
    geometry: PageGeometryContract,
    *,
    measure_id: str,
) -> HistoricalMeterRoiArtifactV1:
    if not isinstance(geometry, PageGeometryContract) or geometry.status != "accepted":
        raise HistoricalMeterRoiError("accepted runtime geometry is required")
    matches = [m for m in geometry.measure_proposals if m.measure_id == measure_id]
    if len(matches) != 1 or matches[0].status != "accepted":
        raise HistoricalMeterRoiError("measure must resolve to one accepted proposal")
    measure = matches[0]
    staffs = [s for s in geometry.staffs if s.staff_id == measure.staff_id and s.system_id == measure.system_id]
    if len(staffs) != 1:
        raise HistoricalMeterRoiError("measure must resolve to exactly one owning staff")
    staff = staffs[0]
    image = _decode_source(normalized_png, geometry)
    spacing = float(staff.staff_spacing)

    left_f = measure.bbox.x_min - spacing * X_BEFORE_MILLI / 1000.0
    right_f = measure.bbox.x_min + spacing * X_AFTER_MILLI / 1000.0
    top_f = staff.staff_bbox.y_min - spacing * Y_BEFORE_MILLI / 1000.0
    bottom_f = staff.staff_bbox.y_max + spacing * Y_AFTER_MILLI / 1000.0

    left = max(0, min(image.width - 1, math.floor(left_f)))
    top = max(0, min(image.height - 1, math.floor(top_f)))
    right = max(left + 1, min(image.width, math.ceil(right_f)))
    bottom = max(top + 1, min(image.height, math.ceil(bottom_f)))
    crop_w, crop_h = right - left, bottom - top
    scale = min(OUTPUT_WIDTH / crop_w, OUTPUT_HEIGHT / crop_h)
    resized_w = max(1, min(OUTPUT_WIDTH, int(round(crop_w * scale))))
    resized_h = max(1, min(OUTPUT_HEIGHT, int(round(crop_h * scale))))
    pad_left = (OUTPUT_WIDTH - resized_w) // 2
    pad_top = (OUTPUT_HEIGHT - resized_h) // 2

    crop = image.crop((left, top, right, bottom))
    resized = crop.resize((resized_w, resized_h), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("L", (OUTPUT_WIDTH, OUTPUT_HEIGHT), 255)
    canvas.paste(resized, (pad_left, pad_top))
    out = BytesIO()
    canvas.save(out, format="PNG", optimize=False, compress_level=9)
    raw = out.getvalue()
    if not raw:
        raise HistoricalMeterRoiError("historical Meter ROI PNG encoding failed")

    return HistoricalMeterRoiArtifactV1(
        measure_id=measure.measure_id,
        staff_id=measure.staff_id,
        source_image_sha256=geometry.normalized_image_sha256,
        image_sha256=sha256(raw).hexdigest(),
        crop_box=(left, top, right, bottom),
        resized_size=(resized_w, resized_h),
        pad_left=pad_left,
        pad_top=pad_top,
        profile_fingerprint=historical_meter_roi_profile_fingerprint_v1(),
        png_bytes=raw,
    )


def checkpoint_loading_allowed() -> bool:
    return False


def resolver_connection_allowed() -> bool:
    return False
