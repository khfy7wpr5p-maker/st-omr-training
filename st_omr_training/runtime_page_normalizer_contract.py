"""Isolated declarative contract for the future ST Page Normalizer runtime lane.

This module defines only metadata, validation boundaries, and deterministic
coordinate-provenance rules. It does not rasterize PDFs, process images, load
models, train, access TEST, or integrate with Stage 7-D10 / Stage 7-D13.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
from typing import Final


PAGE_NORMALIZER_CONTRACT_VERSION: Final[str] = "runtime-page-normalizer-contract-v1"
PAGE_NORMALIZER_SCHEMA: Final[str] = "runtime-page-normalizer-contract-v1"
MAX_PAGE_DIMENSION: Final[int] = 20_000
MAX_PAGE_PIXELS: Final[int] = 200_000_000
SUPPORTED_INPUT_PIXEL_MODES: Final[tuple[str, ...]] = ("gray8", "rgb8", "rgba8")
NORMALIZED_PIXEL_MODE: Final[str] = "gray8"
NORMALIZER_STATUSES: Final[tuple[str, ...]] = ("accepted", "ambiguous", "rejected")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex string")


def _require_plain_positive_int(name: str, value: int, maximum: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{name} is outside the runtime contract bounds")


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class HomographyContract:
    """Forward and inverse 3x3 transforms for coordinate provenance."""

    forward: tuple[float, ...]
    inverse: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.forward) != 9 or len(self.inverse) != 9:
            raise ValueError("homography matrices must each contain exactly 9 values")
        for matrix in (self.forward, self.inverse):
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in matrix
            ):
                raise ValueError("homography values must be finite numbers")
            if abs(self._determinant(matrix)) <= 1e-12:
                raise ValueError("homography matrix must be invertible")
        self._validate_inverse_pair()

    @staticmethod
    def _determinant(matrix: tuple[float, ...]) -> float:
        a, b, c, d, e, f, g, h, i = matrix
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    @staticmethod
    def _multiply(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
        values: list[float] = []
        for row in range(3):
            for column in range(3):
                values.append(
                    sum(left[row * 3 + k] * right[k * 3 + column] for k in range(3))
                )
        return tuple(values)

    def _validate_inverse_pair(self) -> None:
        identity = self._multiply(self.forward, self.inverse)
        expected = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        if any(abs(actual - wanted) > 1e-9 for actual, wanted in zip(identity, expected)):
            raise ValueError("forward/inverse homographies are not a valid inverse pair")

    @staticmethod
    def _map(matrix: tuple[float, ...], x: float, y: float) -> tuple[float, float]:
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in (x, y)
        ):
            raise ValueError("point coordinates must be finite numbers")
        denominator = matrix[6] * x + matrix[7] * y + matrix[8]
        if abs(denominator) <= 1e-12:
            raise ValueError("point maps to an invalid projective coordinate")
        mapped_x = (matrix[0] * x + matrix[1] * y + matrix[2]) / denominator
        mapped_y = (matrix[3] * x + matrix[4] * y + matrix[5]) / denominator
        if not math.isfinite(mapped_x) or not math.isfinite(mapped_y):
            raise ValueError("mapped coordinates must remain finite")
        return mapped_x, mapped_y

    def original_to_normalized(self, x: float, y: float) -> tuple[float, float]:
        return self._map(self.forward, x, y)

    def normalized_to_original(self, x: float, y: float) -> tuple[float, float]:
        return self._map(self.inverse, x, y)


@dataclass(frozen=True, slots=True)
class RasterPageInputContract:
    """Runtime raster boundary before any normalization algorithm exists."""

    source_id: str
    source_sha256: str
    page_number: int
    width: int
    height: int
    pixel_mode: str
    raster_sha256: str
    dpi: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        _require_sha256("source_sha256", self.source_sha256)
        _require_sha256("raster_sha256", self.raster_sha256)
        _require_plain_positive_int("page_number", self.page_number, 10_000)
        _require_plain_positive_int("width", self.width, MAX_PAGE_DIMENSION)
        _require_plain_positive_int("height", self.height, MAX_PAGE_DIMENSION)
        if self.width * self.height > MAX_PAGE_PIXELS:
            raise ValueError("raster page exceeds the frozen pixel ceiling")
        if self.pixel_mode not in SUPPORTED_INPUT_PIXEL_MODES:
            raise ValueError("unsupported input pixel mode")
        if self.dpi is not None:
            _require_plain_positive_int("dpi", self.dpi, 2_400)
            if self.dpi < 36:
                raise ValueError("dpi is below the frozen runtime floor")


@dataclass(frozen=True, slots=True)
class NormalizationOperationContract:
    """One auditable deterministic operation declared by a future implementation."""

    operation_id: str
    deterministic: bool = True
    destructive_symbol_removal_allowed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str) or not self.operation_id.isascii() or not self.operation_id:
            raise ValueError("operation_id must be non-empty ASCII")
        if self.deterministic is not True:
            raise ValueError("normalizer operations must remain deterministic")
        if self.destructive_symbol_removal_allowed is not False:
            raise ValueError("destructive symbol removal is forbidden")


@dataclass(frozen=True, slots=True)
class NormalizedPageContract:
    """Metadata contract emitted by a future deterministic page normalizer."""

    source_raster_sha256: str
    normalizer_config_fingerprint: str
    normalized_image_sha256: str | None
    normalized_width: int | None
    normalized_height: int | None
    transform: HomographyContract | None
    operations: tuple[NormalizationOperationContract, ...]
    status: str
    rejection_reasons: tuple[str, ...] = ()
    pixel_mode: str = NORMALIZED_PIXEL_MODE

    def __post_init__(self) -> None:
        _require_sha256("source_raster_sha256", self.source_raster_sha256)
        _require_sha256("normalizer_config_fingerprint", self.normalizer_config_fingerprint)
        if self.status not in NORMALIZER_STATUSES:
            raise ValueError("unsupported normalizer status")
        if self.pixel_mode != NORMALIZED_PIXEL_MODE:
            raise ValueError("normalized output must remain gray8")
        if len({item.operation_id for item in self.operations}) != len(self.operations):
            raise ValueError("normalizer operation ids must be unique")
        if self.status == "accepted":
            if self.normalized_image_sha256 is None or self.transform is None:
                raise ValueError("accepted normalized page requires image identity and transform")
            _require_sha256("normalized_image_sha256", self.normalized_image_sha256)
            if self.normalized_width is None or self.normalized_height is None:
                raise ValueError("accepted normalized page requires dimensions")
            _require_plain_positive_int("normalized_width", self.normalized_width, MAX_PAGE_DIMENSION)
            _require_plain_positive_int("normalized_height", self.normalized_height, MAX_PAGE_DIMENSION)
            if self.normalized_width * self.normalized_height > MAX_PAGE_PIXELS:
                raise ValueError("normalized page exceeds the frozen pixel ceiling")
            if self.rejection_reasons:
                raise ValueError("accepted page cannot carry rejection reasons")
        else:
            if not self.rejection_reasons:
                raise ValueError("ambiguous/rejected page must explain why")
            if self.normalized_image_sha256 is not None:
                _require_sha256("normalized_image_sha256", self.normalized_image_sha256)


NORMALIZER_ALLOWED_OPERATIONS: Final[tuple[str, ...]] = (
    "orientation",
    "deskew",
    "safe_crop",
    "illumination_normalization",
    "contrast_normalization",
    "perspective_correction",
    "resolution_normalization",
)

NORMALIZER_FORBIDDEN_SEMANTICS: Final[tuple[str, ...]] = (
    "staff_recognition",
    "measure_recognition",
    "meter_recognition",
    "notehead_recognition",
    "rest_recognition",
    "accidental_recognition",
    "pitch_inference",
    "duration_inference",
    "musicxml_generation",
    "staff_line_removal",
)


def runtime_page_normalizer_contract_payload() -> dict[str, object]:
    return {
        "schema_version": PAGE_NORMALIZER_SCHEMA,
        "contract_version": PAGE_NORMALIZER_CONTRACT_VERSION,
        "input": {
            "kind": "raster-page-only",
            "pdf_rasterization_in_scope": False,
            "pixel_modes": SUPPORTED_INPUT_PIXEL_MODES,
            "max_dimension": MAX_PAGE_DIMENSION,
            "max_pixels": MAX_PAGE_PIXELS,
        },
        "output": {
            "pixel_mode": NORMALIZED_PIXEL_MODE,
            "statuses": NORMALIZER_STATUSES,
            "requires_forward_inverse_transform_when_accepted": True,
            "coordinate_roundtrip_required": True,
        },
        "allowed_operations": NORMALIZER_ALLOWED_OPERATIONS,
        "forbidden_semantics": NORMALIZER_FORBIDDEN_SEMANTICS,
        "isolation": {
            "stage7d10_read": False,
            "stage7d10_write": False,
            "stage7d13_read": False,
            "stage7d13_write": False,
            "checkpoint_access": False,
            "optimizer_access": False,
            "test_split_access": False,
        },
    }


def runtime_page_normalizer_contract_fingerprint() -> str:
    return _canonical_sha256(runtime_page_normalizer_contract_payload())
