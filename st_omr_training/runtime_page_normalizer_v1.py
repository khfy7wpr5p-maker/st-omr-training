"""First bounded deterministic ST Page Normalizer implementation slice.

This module intentionally does only a small, auditable subset of the future
runtime normalizer:

- decode one PNG/JPEG raster whose bytes are already bound by the input contract;
- honor only non-mirrored EXIF orientation values (1/3/6/8);
- convert the page to staff-preserving gray8;
- apply a monotonic global linear autocontrast operation;
- emit deterministic PNG bytes plus exact coordinate provenance.

It does not deskew, crop, dewarp, detect staff/measure/music symbols, load models,
train, access TEST, or integrate with Stage 7-D10 / Stage 7-D13.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
from io import BytesIO
import json
from typing import Final

from PIL import Image, ImageOps, UnidentifiedImageError

from .runtime_page_normalizer_contract import (
    HomographyContract,
    NormalizationOperationContract,
    NormalizedPageContract,
    RasterPageInputContract,
)


PAGE_NORMALIZER_V1_VERSION: Final[str] = "runtime-page-normalizer-v1-slice1"
EXPECTED_PILLOW_VERSION: Final[str] = "12.3.0"
SUPPORTED_RASTER_FORMATS: Final[tuple[str, ...]] = ("JPEG", "PNG")
SUPPORTED_EXIF_ORIENTATIONS: Final[tuple[int, ...]] = (1, 3, 6, 8)
REJECTED_MIRRORED_EXIF_ORIENTATIONS: Final[tuple[int, ...]] = (2, 4, 5, 7)
EXIF_ORIENTATION_TAG: Final[int] = 274

_MODE_TO_CONTRACT: Final[dict[str, str]] = {
    "L": "gray8",
    "RGB": "rgb8",
    "RGBA": "rgba8",
}


class PageNormalizerV1Error(ValueError):
    """Raised when raster bytes violate the frozen V1 runtime boundary."""


@dataclass(frozen=True, slots=True)
class PageNormalizerV1Result:
    """One accepted or fail-closed V1 outcome."""

    normalized_png: bytes | None
    page: NormalizedPageContract
    source_format: str
    exif_orientation: int

    def __post_init__(self) -> None:
        if self.page.status == "accepted" and self.normalized_png is None:
            raise ValueError("accepted V1 result requires normalized PNG bytes")
        if self.page.status != "accepted" and self.normalized_png is not None:
            raise ValueError("non-accepted V1 result cannot expose normalized PNG bytes")
        if self.source_format not in SUPPORTED_RASTER_FORMATS:
            raise ValueError("V1 result source format is outside the frozen surface")
        if self.exif_orientation not in range(1, 9):
            raise ValueError("V1 result EXIF orientation must be in 1..8")


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256(raw).hexdigest()


def page_normalizer_v1_config_payload() -> dict[str, object]:
    """Return the frozen deterministic implementation profile."""
    return {
        "implementation_version": PAGE_NORMALIZER_V1_VERSION,
        "pillow_version": EXPECTED_PILLOW_VERSION,
        "input_formats": SUPPORTED_RASTER_FORMATS,
        "supported_exif_orientations": SUPPORTED_EXIF_ORIENTATIONS,
        "mirrored_exif_policy": "reject-fail-closed",
        "coordinate_convention": "continuous-pixel-edge-origin-top-left-v1",
        "rgba_background": "opaque-white",
        "grayscale": "pillow-convert-L-v1",
        "contrast": "pillow-autocontrast-cutoff-0-v1",
        "output": {
            "format": "PNG",
            "pixel_mode": "L",
            "optimize": False,
            "compress_level": 9,
            "metadata": "none",
        },
        "forbidden": (
            "deskew",
            "safe_crop",
            "illumination_normalization",
            "perspective_correction",
            "resolution_normalization",
            "staff_line_removal",
            "music_symbol_recognition",
            "stage7d10_access",
            "stage7d13_access",
            "checkpoint_access",
            "optimizer_access",
            "test_split_access",
        ),
    }


def page_normalizer_v1_config_fingerprint() -> str:
    return _canonical_sha256(page_normalizer_v1_config_payload())


def _verify_runtime_version() -> None:
    actual = metadata.version("Pillow")
    if actual != EXPECTED_PILLOW_VERSION:
        raise PageNormalizerV1Error(
            f"Pillow runtime mismatch: expected {EXPECTED_PILLOW_VERSION}, got {actual}"
        )


def _orientation_transform(
    orientation: int, width: int, height: int
) -> HomographyContract:
    if orientation == 1:
        forward = inverse = (
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    elif orientation == 3:
        forward = inverse = (
            -1.0,
            0.0,
            float(width),
            0.0,
            -1.0,
            float(height),
            0.0,
            0.0,
            1.0,
        )
    elif orientation == 6:
        forward = (
            0.0,
            -1.0,
            float(height),
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        inverse = (
            0.0,
            1.0,
            0.0,
            -1.0,
            0.0,
            float(height),
            0.0,
            0.0,
            1.0,
        )
    elif orientation == 8:
        forward = (
            0.0,
            1.0,
            0.0,
            -1.0,
            0.0,
            float(width),
            0.0,
            0.0,
            1.0,
        )
        inverse = (
            0.0,
            -1.0,
            float(width),
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    else:
        raise PageNormalizerV1Error("orientation is not supported by V1")
    return HomographyContract(forward=forward, inverse=inverse)


def _apply_orientation(image: Image.Image, orientation: int) -> Image.Image:
    if orientation == 1:
        return image.copy()
    if orientation == 3:
        return image.transpose(Image.Transpose.ROTATE_180)
    if orientation == 6:
        return image.transpose(Image.Transpose.ROTATE_270)
    if orientation == 8:
        return image.transpose(Image.Transpose.ROTATE_90)
    raise PageNormalizerV1Error("orientation is not supported by V1")


def _to_staff_preserving_gray8(image: Image.Image) -> Image.Image:
    if image.mode == "L":
        return image.copy()
    if image.mode == "RGB":
        return image.convert("L")
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(background, image)
        return composite.convert("L")
    raise PageNormalizerV1Error("decoded image mode is outside the frozen V1 surface")


def _encode_deterministic_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(
        output,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return output.getvalue()


def normalize_raster_page_v1(
    raster_bytes: bytes,
    raster: RasterPageInputContract,
) -> PageNormalizerV1Result:
    """Normalize one already-rasterized page under the bounded V1 slice."""
    _verify_runtime_version()
    if not isinstance(raster_bytes, bytes) or not raster_bytes:
        raise PageNormalizerV1Error("raster_bytes must be non-empty bytes")
    if not isinstance(raster, RasterPageInputContract):
        raise TypeError("raster must be RasterPageInputContract")

    actual_sha = sha256(raster_bytes).hexdigest()
    if actual_sha != raster.raster_sha256:
        raise PageNormalizerV1Error("raster byte SHA-256 does not match the input contract")

    try:
        with Image.open(BytesIO(raster_bytes)) as opened:
            source_format = str(opened.format or "")
            if source_format not in SUPPORTED_RASTER_FORMATS:
                raise PageNormalizerV1Error("raster format is outside the frozen V1 surface")
            if getattr(opened, "n_frames", 1) != 1:
                raise PageNormalizerV1Error("multi-frame raster input is forbidden")
            if opened.size != (raster.width, raster.height):
                raise PageNormalizerV1Error("decoded raster dimensions do not match the input contract")
            decoded_pixel_mode = _MODE_TO_CONTRACT.get(opened.mode)
            if decoded_pixel_mode != raster.pixel_mode:
                raise PageNormalizerV1Error("decoded raster mode does not match the input contract")

            try:
                exif_orientation_raw = opened.getexif().get(EXIF_ORIENTATION_TAG, 1)
                exif_orientation = int(exif_orientation_raw)
            except (TypeError, ValueError) as exc:
                raise PageNormalizerV1Error("EXIF orientation is not a valid integer") from exc
            if exif_orientation not in range(1, 9):
                raise PageNormalizerV1Error("EXIF orientation must be in 1..8")

            opened.load()
            source_image = opened.copy()
    except PageNormalizerV1Error:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise PageNormalizerV1Error("raster bytes are not a valid supported image") from exc

    config_fingerprint = page_normalizer_v1_config_fingerprint()
    if exif_orientation in REJECTED_MIRRORED_EXIF_ORIENTATIONS:
        page = NormalizedPageContract(
            source_raster_sha256=raster.raster_sha256,
            normalizer_config_fingerprint=config_fingerprint,
            normalized_image_sha256=None,
            normalized_width=None,
            normalized_height=None,
            transform=None,
            operations=(),
            status="rejected",
            rejection_reasons=("mirrored-exif-orientation-not-supported-v1",),
        )
        return PageNormalizerV1Result(
            normalized_png=None,
            page=page,
            source_format=source_format,
            exif_orientation=exif_orientation,
        )

    transform = _orientation_transform(exif_orientation, raster.width, raster.height)
    working = _apply_orientation(source_image, exif_orientation)

    operations: list[NormalizationOperationContract] = []
    if exif_orientation != 1:
        operations.append(NormalizationOperationContract("orientation"))
    if source_image.mode != "L":
        operations.append(NormalizationOperationContract("grayscale_conversion"))

    gray = _to_staff_preserving_gray8(working)
    normalized = ImageOps.autocontrast(gray, cutoff=0)
    operations.append(NormalizationOperationContract("contrast_normalization"))

    normalized_png = _encode_deterministic_png(normalized)
    normalized_sha = sha256(normalized_png).hexdigest()
    page = NormalizedPageContract(
        source_raster_sha256=raster.raster_sha256,
        normalizer_config_fingerprint=config_fingerprint,
        normalized_image_sha256=normalized_sha,
        normalized_width=normalized.width,
        normalized_height=normalized.height,
        transform=transform,
        operations=tuple(operations),
        status="accepted",
    )
    return PageNormalizerV1Result(
        normalized_png=normalized_png,
        page=page,
        source_format=source_format,
        exif_orientation=exif_orientation,
    )
