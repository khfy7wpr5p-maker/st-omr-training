"""Deterministic in-memory Runtime Local ROI extractor.

This is deliberately not Stage 7-D10.  It consumes accepted runtime measure
geometry and emits hash-bound in-memory crops for downstream specialist
adapters.  It imports no D10/D13 code and writes no derivative dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
import re
from typing import Final

from PIL import Image, UnidentifiedImageError

from .runtime_geometry_engine_contract import BoxContract, PageGeometryContract
from .runtime_page_normalizer_contract import HomographyContract


RUNTIME_ROI_V1_VERSION: Final[str] = "runtime-local-roi-v1"
ROI_KINDS: Final[tuple[str, ...]] = ("measure-full", "measure-start")
METER_START_WIDTH_SPACINGS_MILLI: Final[int] = 12_000
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RuntimeRoiArtifact:
    roi_id: str
    kind: str
    measure_id: str
    staff_id: str
    source_image_sha256: str
    roi_image_sha256: str
    crop_bbox: BoxContract
    source_to_roi: HomographyContract
    png_bytes: bytes

    def __post_init__(self) -> None:
        if not self.roi_id or not self.measure_id or not self.staff_id:
            raise ValueError("ROI identities must be non-empty")
        if self.kind not in ROI_KINDS:
            raise ValueError("unsupported runtime ROI kind")
        for value in (self.source_image_sha256, self.roi_image_sha256):
            if _SHA_RE.fullmatch(value) is None:
                raise ValueError("ROI SHA identities must be lowercase SHA-256")
        if not isinstance(self.png_bytes, bytes) or not self.png_bytes:
            raise ValueError("ROI PNG bytes must be non-empty")
        if sha256(self.png_bytes).hexdigest() != self.roi_image_sha256:
            raise ValueError("ROI byte identity mismatch")


@dataclass(frozen=True, slots=True)
class RuntimeRoiBatch:
    source_image_sha256: str
    config_fingerprint: str
    artifacts: tuple[RuntimeRoiArtifact, ...]

    def __post_init__(self) -> None:
        if _SHA_RE.fullmatch(self.source_image_sha256) is None:
            raise ValueError("batch source SHA must be lowercase SHA-256")
        if _SHA_RE.fullmatch(self.config_fingerprint) is None:
            raise ValueError("batch config fingerprint must be lowercase SHA-256")
        ids = tuple(item.roi_id for item in self.artifacts)
        if len(set(ids)) != len(ids):
            raise ValueError("runtime ROI ids must be unique")


def runtime_roi_v1_config_fingerprint(parent_geometry_fingerprint: str) -> str:
    payload = {
        "version": RUNTIME_ROI_V1_VERSION,
        "parent_geometry_fingerprint": parent_geometry_fingerprint,
        "roi_kinds": ROI_KINDS,
        "meter_start_width_spacings_milli": METER_START_WIDTH_SPACINGS_MILLI,
        "crop_rounding": "floor-min-ceil-max-v1",
        "output": "gray8-png-in-memory",
        "stage7d10_import": False,
        "stage7d13_import": False,
        "dataset_write": False,
        "checkpoint_access": False,
        "test_split_access": False,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return sha256(raw).hexdigest()


def _decode(normalized_png: bytes, geometry: PageGeometryContract) -> Image.Image:
    if not isinstance(normalized_png, bytes) or not normalized_png:
        raise ValueError("normalized PNG must be non-empty bytes")
    if sha256(normalized_png).hexdigest() != geometry.normalized_image_sha256:
        raise ValueError("normalized PNG identity does not match geometry")
    try:
        with Image.open(BytesIO(normalized_png)) as opened:
            if opened.format != "PNG" or opened.mode != "L":
                raise ValueError("runtime ROI extractor requires gray8 PNG")
            opened.load()
            if opened.size != (geometry.page_width, geometry.page_height):
                raise ValueError("normalized PNG dimensions do not match geometry")
            return opened.copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("normalized PNG cannot be decoded safely") from exc


def _encode(image: Image.Image) -> bytes:
    out = BytesIO()
    image.save(out, format="PNG", optimize=False, compress_level=9)
    return out.getvalue()


def _integer_crop(box: BoxContract, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, int(math.floor(box.x_min)))
    top = max(0, int(math.floor(box.y_min)))
    right = min(width, int(math.ceil(box.x_max)))
    bottom = min(height, int(math.ceil(box.y_max)))
    if right <= left or bottom <= top:
        raise ValueError("runtime ROI crop is empty")
    return left, top, right, bottom


def _artifact(
    image: Image.Image,
    source_sha: str,
    measure_id: str,
    staff_id: str,
    kind: str,
    box: BoxContract,
) -> RuntimeRoiArtifact:
    left, top, right, bottom = _integer_crop(box, image.width, image.height)
    crop = image.crop((left, top, right, bottom))
    data = _encode(crop)
    crop_bbox = BoxContract(float(left), float(top), float(right), float(bottom))
    transform = HomographyContract(
        forward=(1.0, 0.0, -float(left), 0.0, 1.0, -float(top), 0.0, 0.0, 1.0),
        inverse=(1.0, 0.0, float(left), 0.0, 1.0, float(top), 0.0, 0.0, 1.0),
    )
    return RuntimeRoiArtifact(
        roi_id=f"{measure_id}:{kind}",
        kind=kind,
        measure_id=measure_id,
        staff_id=staff_id,
        source_image_sha256=source_sha,
        roi_image_sha256=sha256(data).hexdigest(),
        crop_bbox=crop_bbox,
        source_to_roi=transform,
        png_bytes=data,
    )


def extract_runtime_rois_v1(
    normalized_png: bytes,
    geometry: PageGeometryContract,
) -> RuntimeRoiBatch:
    """Emit full-measure and measure-start in-memory crops in deterministic order."""
    if not isinstance(geometry, PageGeometryContract):
        raise TypeError("geometry must be PageGeometryContract")
    if geometry.status != "accepted" or not geometry.measure_proposals:
        raise ValueError("runtime ROI extraction requires accepted measure geometry")
    image = _decode(normalized_png, geometry)
    staff_by_id = {staff.staff_id: staff for staff in geometry.staffs}
    measures = tuple(
        sorted(
            geometry.measure_proposals,
            key=lambda item: (item.system_id, item.staff_id, item.bbox.x_min, item.measure_id),
        )
    )
    artifacts: list[RuntimeRoiArtifact] = []
    for measure in measures:
        staff = staff_by_id[measure.staff_id]
        artifacts.append(
            _artifact(
                image,
                geometry.normalized_image_sha256,
                measure.measure_id,
                measure.staff_id,
                "measure-full",
                measure.bbox,
            )
        )
        desired_width = staff.staff_spacing * METER_START_WIDTH_SPACINGS_MILLI / 1000.0
        start_box = BoxContract(
            measure.bbox.x_min,
            measure.bbox.y_min,
            min(measure.bbox.x_max, measure.bbox.x_min + desired_width),
            measure.bbox.y_max,
        )
        artifacts.append(
            _artifact(
                image,
                geometry.normalized_image_sha256,
                measure.measure_id,
                measure.staff_id,
                "measure-start",
                start_box,
            )
        )
    return RuntimeRoiBatch(
        source_image_sha256=geometry.normalized_image_sha256,
        config_fingerprint=runtime_roi_v1_config_fingerprint(geometry.geometry_config_fingerprint),
        artifacts=tuple(artifacts),
    )
