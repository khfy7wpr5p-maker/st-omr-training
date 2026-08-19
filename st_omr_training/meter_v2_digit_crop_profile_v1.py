"""Frozen identity-only profile for historical Meter v2 digit crops.

This module contains no crop implementation and no model code. It exposes only
canonical provenance for the already-audited historical 64x64 digit transform.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Final

METER_V2_DIGIT_CROP_PROFILE_V1: Final[dict[str, object]] = {
    "version": "meter-v2-digit-crop-adapter-v1",
    "coordinate_policy": "pixels-or-normalized-if-maxabs-le-1.5",
    "rounding": "floor-left-top-ceil-right-bottom",
    "clip": "source-image-bounds",
    "pixel_mode": "L",
    "resize": "PIL.Image.thumbnail-LANCZOS-no-upscale",
    "canvas": "64x64-white-centered",
    "tensor_semantics": "historical-training-used-uint8-div255",
}


def meter_v2_digit_crop_profile_fingerprint_v1() -> str:
    raw = json.dumps(
        METER_V2_DIGIT_CROP_PROFILE_V1,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256(raw).hexdigest()
