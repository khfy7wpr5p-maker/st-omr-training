"""Shadow-only Meter V2 digit crop adapter.

This freezes the exact pixel transform used by the historical M4 2-AI/3-AI/4-AI
training workers. It does not decide where a digit is located. The caller must
supply a candidate digit bbox from a separately admitted proposal/localization
stage.

Historical transform for valid boxes:
- bbox may be pixel coordinates or normalized coordinates;
- normalized coordinates are scaled by source width/height;
- left/top use floor, right/bottom use ceil, then clip to the image;
- crop is converted to grayscale L;
- PIL thumbnail to 64x64 with LANCZOS preserves aspect ratio and never upscales;
- result is centered on a white 64x64 canvas.

No model/checkpoint, optimizer, sealed TEST split, or runtime Resolver is loaded
or connected here. The adapter is Pillow-only so it does not expand the pinned
runtime dependency surface.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Final, Sequence

from PIL import Image

METER_V2_DIGIT_CROP_SIZE: Final[int] = 64
METER_V2_DIGIT_CROP_PROFILE: Final[dict[str, object]] = {
    "version": "meter-v2-digit-crop-adapter-v1",
    "coordinate_policy": "pixels-or-normalized-if-maxabs-le-1.5",
    "rounding": "floor-left-top-ceil-right-bottom",
    "clip": "source-image-bounds",
    "pixel_mode": "L",
    "resize": "PIL.Image.thumbnail-LANCZOS-no-upscale",
    "canvas": "64x64-white-centered",
    "tensor_semantics": "historical-training-used-uint8-div255",
}


class MeterV2DigitCropError(ValueError):
    """Raised when candidate crop evidence is malformed."""


def meter_v2_digit_crop_profile_fingerprint() -> str:
    raw = json.dumps(
        METER_V2_DIGIT_CROP_PROFILE,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256(raw).hexdigest()


def _finite_box(box: object) -> tuple[float, float, float, float]:
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes, bytearray)):
        raise MeterV2DigitCropError("digit bbox must be a four-value sequence")
    if len(box) != 4:
        raise MeterV2DigitCropError("digit bbox must contain exactly four values")
    values: list[float] = []
    for value in box:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MeterV2DigitCropError("digit bbox values must be finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise MeterV2DigitCropError("digit bbox values must be finite numbers")
        values.append(number)
    x0, y0, x1, y1 = values
    if not x0 < x1 or not y0 < y1:
        raise MeterV2DigitCropError("digit bbox must have positive width and height")
    return x0, y0, x1, y1


def meter_v2_digit_pixel_box_v1(
    image: Image.Image,
    box: Sequence[float],
) -> tuple[int, int, int, int]:
    """Convert one valid candidate bbox to the exact historical pixel crop box."""
    if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        raise MeterV2DigitCropError("source image must be a non-empty PIL image")

    x0, y0, x1, y1 = _finite_box(box)
    width, height = image.size

    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
        x0 *= width
        x1 *= width
        y0 *= height
        y1 *= height

    px0 = max(0, min(width - 1, int(math.floor(x0))))
    py0 = max(0, min(height - 1, int(math.floor(y0))))
    px1 = max(px0 + 1, min(width, int(math.ceil(x1))))
    py1 = max(py0 + 1, min(height, int(math.ceil(y1))))
    return px0, py0, px1, py1


def crop_meter_digit_to_64_v1(
    image: Image.Image,
    box: Sequence[float],
) -> Image.Image:
    """Return the training-equivalent 64x64 grayscale Pillow image.

    The returned image has the same pixel bytes as the historical NumPy uint8
    training crop. This function intentionally performs no thresholding,
    morphology, denoise, staff-line removal, contrast manipulation, or upscaling.
    """
    pixel_box = meter_v2_digit_pixel_box_v1(image, box)
    crop = image.crop(pixel_box).convert("L")
    crop.thumbnail(
        (METER_V2_DIGIT_CROP_SIZE, METER_V2_DIGIT_CROP_SIZE),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "L",
        (METER_V2_DIGIT_CROP_SIZE, METER_V2_DIGIT_CROP_SIZE),
        color=255,
    )
    offset_x = (METER_V2_DIGIT_CROP_SIZE - crop.width) // 2
    offset_y = (METER_V2_DIGIT_CROP_SIZE - crop.height) // 2
    canvas.paste(crop, (offset_x, offset_y))
    return canvas


def runtime_digit_bbox_localization_frozen() -> bool:
    """Pixel preprocessing is frozen; positive digit bbox localization is not."""
    return False


def resolver_connection_allowed() -> bool:
    return False
