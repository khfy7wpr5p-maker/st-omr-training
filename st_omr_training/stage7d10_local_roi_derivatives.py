"""Stage 7-D10 deterministic local Structure ROI derivatives.

D10 materializes the local image/target surface frozen by Stage 7-D9.  It is a
data-derivative layer only: no model, optimizer, backward pass, checkpoint or
TEST evaluation path is present here.

Ground truth remains the accepted D6 final-PNG geometry.  D10 only crops and
replays that geometry into the D9 barline and meter ROI coordinate spaces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Final

from PIL import Image

from .stage7d9_structure_refinement_contract import (
    BARLINE_ROI,
    METER_CLASSES,
    METER_ROI,
    LocalRoiPolicy,
    stage7d9_contract_fingerprint,
)


STAGE7D10_VERSION: Final[str] = "stage7d10-local-roi-derivatives-v1"
STAGE7D10_LABEL_SCHEMA: Final[str] = "stage7d10-local-roi-label-v1"
STAGE7D10_MANIFEST_SCHEMA: Final[str] = "stage7d10-local-roi-manifest-v1"
_ALLOWED_SPLITS: Final[frozenset[str]] = frozenset({"train", "validation"})
_ALLOWED_KINDS: Final[frozenset[str]] = frozenset({"barline", "meter"})
_MAX_IMAGE_BYTES: Final[int] = 32 * 1024 * 1024


class Stage7D10DerivativeError(RuntimeError):
    """Raised when a D10 derivative cannot be built safely."""


def _fail(message: str) -> None:
    raise Stage7D10DerivativeError(message)


def _canonical_json(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Stage7D10DerivativeError("payload is not canonical JSON") from exc


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _finite_number(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def _positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _fail(f"{name} must be a positive integer")
    return value


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or len(value) > 256:
        _fail(f"{name} must be bounded non-empty ASCII")
    return value


def _hex64(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        _fail(f"{name} must be lowercase SHA-256 hex")
    return value


def _box(name: str, value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {"x_min", "y_min", "x_max", "y_max"}:
        _fail(f"{name} must be a canonical box")
    result = {key: _finite_number(f"{name}.{key}", value.get(key)) for key in value}
    if not result["x_min"] < result["x_max"] or not result["y_min"] < result["y_max"]:
        _fail(f"{name} must have positive area")
    return result


def _point(name: str, value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {"x", "y"}:
        _fail(f"{name} must be a canonical point")
    return {
        "x": _finite_number(f"{name}.x", value.get("x")),
        "y": _finite_number(f"{name}.y", value.get("y")),
    }


def _line(name: str, value: object) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping) or set(value) != {"start", "end"}:
        _fail(f"{name} must be a canonical line")
    start = _point(f"{name}.start", value.get("start"))
    end = _point(f"{name}.end", value.get("end"))
    if start == end:
        _fail(f"{name} must have non-zero length")
    return {"start": start, "end": end}


def development_rows(rows: object) -> tuple[Mapping[str, object], ...]:
    """Return TRAIN/VALIDATION rows while touching only split on TEST rows."""
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        _fail("source rows must be a sequence")
    result: list[Mapping[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"source row[{index}] must be a mapping")
        split = row.get("split")
        if split == "test":
            continue
        if split not in _ALLOWED_SPLITS:
            _fail(f"source row[{index}] has invalid development split")
        result.append(row)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class RoiTransform:
    crop_left: int
    crop_top: int
    crop_right: int
    crop_bottom: int
    resized_width: int
    resized_height: int
    output_width: int
    output_height: int
    pad_left: int
    pad_top: int
    scale_x: float
    scale_y: float

    def __post_init__(self) -> None:
        if not (0 <= self.crop_left < self.crop_right and 0 <= self.crop_top < self.crop_bottom):
            raise ValueError("invalid crop bounds")
        if not (1 <= self.resized_width <= self.output_width):
            raise ValueError("invalid resized width")
        if not (1 <= self.resized_height <= self.output_height):
            raise ValueError("invalid resized height")
        if not (0 <= self.pad_left <= self.output_width - self.resized_width):
            raise ValueError("invalid horizontal pad")
        if not (0 <= self.pad_top <= self.output_height - self.resized_height):
            raise ValueError("invalid vertical pad")
        if not (math.isfinite(self.scale_x) and math.isfinite(self.scale_y)):
            raise ValueError("ROI scales must be finite")
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("ROI scales must be positive")


@dataclass(frozen=True, slots=True)
class D10SourceRecord:
    split: str
    sample_id: str
    family_id: str
    image_sha256: str
    label_sha256: str
    image_bytes: bytes
    label: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.split not in _ALLOWED_SPLITS:
            raise ValueError("D10 source split must be train or validation")
        _identifier("sample_id", self.sample_id)
        _identifier("family_id", self.family_id)
        _hex64("image_sha256", self.image_sha256)
        _hex64("label_sha256", self.label_sha256)
        if not isinstance(self.image_bytes, bytes) or not 1 <= len(self.image_bytes) <= _MAX_IMAGE_BYTES:
            raise ValueError("image_bytes are outside D10 bounds")
        if _sha(self.image_bytes) != self.image_sha256:
            raise ValueError("source image SHA-256 mismatch")
        if not isinstance(self.label, Mapping):
            raise ValueError("source label must be a mapping")
        if _sha(_canonical_json(dict(self.label))) != self.label_sha256:
            raise ValueError("source label SHA-256 mismatch")


@dataclass(frozen=True, slots=True)
class D10RoiArtifact:
    kind: str
    split: str
    sample_id: str
    family_id: str
    measure_number: int
    record_id: str
    image_bytes: bytes
    image_sha256: str
    label: dict[str, object]
    label_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in _ALLOWED_KINDS:
            raise ValueError("invalid D10 ROI kind")
        if self.split not in _ALLOWED_SPLITS:
            raise ValueError("invalid D10 ROI split")
        if self.measure_number <= 0:
            raise ValueError("measure_number must be positive")
        _hex64("record_id", self.record_id)
        _hex64("image_sha256", self.image_sha256)
        _hex64("label_sha256", self.label_sha256)
        if _sha(self.image_bytes) != self.image_sha256:
            raise ValueError("ROI image SHA-256 mismatch")
        if _sha(_canonical_json(self.label)) != self.label_sha256:
            raise ValueError("ROI label SHA-256 mismatch")


@dataclass(frozen=True, slots=True)
class Stage7D10Receipt:
    version: str
    d9_contract_fingerprint: str
    manifest_sha256: str
    artifact_binding_sha256: str
    source_sample_count: int
    source_family_count: int
    roi_record_count: int
    split_counts: dict[str, int]
    kind_counts: dict[str, int]
    meter_class_counts: dict[str, int]
    test_records: int
    optimizer_steps: int


def _open_grayscale_png(raw: bytes) -> Image.Image:
    if not 1 <= len(raw) <= _MAX_IMAGE_BYTES:
        _fail("source image byte length is outside D10 bounds")
    try:
        with Image.open(BytesIO(raw)) as opened:
            opened.load()
            if opened.format != "PNG":
                _fail("D10 source image must be PNG")
            if opened.mode != "L":
                _fail("D10 source image must be grayscale L")
            return opened.copy()
    except Stage7D10DerivativeError:
        raise
    except Exception as exc:
        raise Stage7D10DerivativeError("D10 source PNG decode failed") from exc


def _geometry(label: Mapping[str, object]) -> Mapping[str, object]:
    value = label.get("geometry")
    if not isinstance(value, Mapping):
        _fail("D6 label geometry is missing")
    return value


def _sequence(name: str, value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(f"{name} must be a sequence")
    return value


def _staff_for_measure(
    geometry: Mapping[str, object], measure: Mapping[str, object]
) -> Mapping[str, object]:
    system_id = _identifier("measure.system_id", measure.get("system_id"))
    staffs = _sequence("geometry.staff_instances", geometry.get("staff_instances"))
    matches = [
        staff
        for staff in staffs
        if isinstance(staff, Mapping) and staff.get("system_id") == system_id
    ]
    if len(matches) != 1:
        _fail("measure system must resolve to exactly one D10 staff instance")
    return matches[0]


def _crop_transform(
    *,
    measure_bbox: Mapping[str, object],
    staff_bbox: Mapping[str, object],
    staff_spacing: float,
    image_width: int,
    image_height: int,
    policy: LocalRoiPolicy,
) -> RoiTransform:
    measure = _box("measure_bbox", measure_bbox)
    staff = _box("staff_instance_bbox", staff_bbox)
    if not math.isfinite(staff_spacing) or staff_spacing <= 0:
        _fail("staff_spacing must be finite and positive")

    anchor_x = measure["x_max"] if policy.anchor == "measure_end" else measure["x_min"]
    left_f = anchor_x - staff_spacing * policy.x_before_staff_spacings_milli / 1000.0
    right_f = anchor_x + staff_spacing * policy.x_after_staff_spacings_milli / 1000.0
    top_f = staff["y_min"] - staff_spacing * policy.y_before_staff_spacings_milli / 1000.0
    bottom_f = staff["y_max"] + staff_spacing * policy.y_after_staff_spacings_milli / 1000.0

    left = max(0, min(image_width - 1, math.floor(left_f)))
    top = max(0, min(image_height - 1, math.floor(top_f)))
    right = max(left + 1, min(image_width, math.ceil(right_f)))
    bottom = max(top + 1, min(image_height, math.ceil(bottom_f)))

    crop_w = right - left
    crop_h = bottom - top
    output_w = policy.output_width
    output_h = policy.output_height
    scale = min(output_w / crop_w, output_h / crop_h)
    resized_w = max(1, min(output_w, int(round(crop_w * scale))))
    resized_h = max(1, min(output_h, int(round(crop_h * scale))))
    pad_left = (output_w - resized_w) // 2
    pad_top = (output_h - resized_h) // 2

    return RoiTransform(
        crop_left=left,
        crop_top=top,
        crop_right=right,
        crop_bottom=bottom,
        resized_width=resized_w,
        resized_height=resized_h,
        output_width=output_w,
        output_height=output_h,
        pad_left=pad_left,
        pad_top=pad_top,
        scale_x=resized_w / crop_w,
        scale_y=resized_h / crop_h,
    )


def _map_xy(transform: RoiTransform, x: float, y: float) -> dict[str, float]:
    mapped_x = (x - transform.crop_left) * transform.scale_x + transform.pad_left
    mapped_y = (y - transform.crop_top) * transform.scale_y + transform.pad_top
    epsilon = 1e-6
    if not -epsilon <= mapped_x <= transform.output_width + epsilon:
        _fail("mapped target x lies outside ROI")
    if not -epsilon <= mapped_y <= transform.output_height + epsilon:
        _fail("mapped target y lies outside ROI")
    return {"x": mapped_x, "y": mapped_y}


def _map_line(transform: RoiTransform, value: object) -> dict[str, object]:
    line = _line("barline_segment", value)
    return {
        "start": _map_xy(transform, line["start"]["x"], line["start"]["y"]),
        "end": _map_xy(transform, line["end"]["x"], line["end"]["y"]),
    }


def _map_box(transform: RoiTransform, value: object) -> dict[str, float]:
    box = _box("meter_bbox", value)
    p0 = _map_xy(transform, box["x_min"], box["y_min"])
    p1 = _map_xy(transform, box["x_max"], box["y_max"])
    result = {
        "x_min": min(p0["x"], p1["x"]),
        "y_min": min(p0["y"], p1["y"]),
        "x_max": max(p0["x"], p1["x"]),
        "y_max": max(p0["y"], p1["y"]),
    }
    if not result["x_min"] < result["x_max"] or not result["y_min"] < result["y_max"]:
        _fail("mapped meter bbox has zero area")
    return result


def _render_roi(image: Image.Image, transform: RoiTransform) -> bytes:
    crop = image.crop(
        (
            transform.crop_left,
            transform.crop_top,
            transform.crop_right,
            transform.crop_bottom,
        )
    )
    resized = crop.resize(
        (transform.resized_width, transform.resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    canvas = Image.new("L", (transform.output_width, transform.output_height), 255)
    canvas.paste(resized, (transform.pad_left, transform.pad_top))
    buffer = BytesIO()
    canvas.save(buffer, format="PNG", optimize=False, compress_level=9)
    raw = buffer.getvalue()
    if not raw:
        _fail("ROI PNG encoder returned empty bytes")
    return raw


def _record_id(
    *,
    source: D10SourceRecord,
    kind: str,
    measure_number: int,
    policy_id: str,
) -> str:
    return _sha(
        _canonical_json(
            {
                "version": STAGE7D10_VERSION,
                "d9_contract_fingerprint": stage7d9_contract_fingerprint(),
                "source_sample_id": source.sample_id,
                "source_image_sha256": source.image_sha256,
                "source_label_sha256": source.label_sha256,
                "split": source.split,
                "kind": kind,
                "measure_number": measure_number,
                "roi_policy_id": policy_id,
            }
        )
    )


def _artifact(
    *,
    source: D10SourceRecord,
    image: Image.Image,
    measure: Mapping[str, object],
    geometry: Mapping[str, object],
    kind: str,
) -> D10RoiArtifact:
    measure_number = _positive_int("measure_number", measure.get("measure_number"))
    staff = _staff_for_measure(geometry, measure)
    staff_bbox = staff.get("staff_instance_bbox")
    staff_spacing = _finite_number("staff_spacing", staff.get("staff_spacing"))
    policy = BARLINE_ROI if kind == "barline" else METER_ROI
    transform = _crop_transform(
        measure_bbox=measure.get("measure_bbox"),
        staff_bbox=staff_bbox,
        staff_spacing=staff_spacing,
        image_width=image.width,
        image_height=image.height,
        policy=policy,
    )
    roi_bytes = _render_roi(image, transform)
    roi_sha = _sha(roi_bytes)

    target: dict[str, object]
    if kind == "barline":
        target = {
            "barline_segment": _map_line(transform, measure.get("barline_segment")),
        }
    elif kind == "meter":
        meter_bbox = measure.get("meter_bbox")
        if meter_bbox is None:
            meter_class = "none"
            mapped_bbox = None
        else:
            meter_class = measure.get("meter_class")
            if meter_class not in METER_CLASSES[1:]:
                _fail("visible meter bbox requires supported D9 meter class")
            mapped_bbox = _map_box(transform, meter_bbox)
        target = {"meter_class": meter_class, "meter_bbox": mapped_bbox}
    else:  # pragma: no cover - internal callers are bounded
        _fail("unknown D10 ROI kind")

    record_id = _record_id(
        source=source,
        kind=kind,
        measure_number=measure_number,
        policy_id=policy.policy_id,
    )
    label: dict[str, object] = {
        "schema_version": STAGE7D10_LABEL_SCHEMA,
        "stage7d10_version": STAGE7D10_VERSION,
        "d9_contract_fingerprint": stage7d9_contract_fingerprint(),
        "record_id": record_id,
        "kind": kind,
        "split": source.split,
        "source": {
            "sample_id": source.sample_id,
            "family_id": source.family_id,
            "image_sha256": source.image_sha256,
            "d6_label_sha256": source.label_sha256,
        },
        "measure_number": measure_number,
        "roi_policy_id": policy.policy_id,
        "roi_transform": asdict(transform),
        "roi_image_sha256": roi_sha,
        "target": target,
    }
    label_sha = _sha(_canonical_json(label))
    return D10RoiArtifact(
        kind=kind,
        split=source.split,
        sample_id=source.sample_id,
        family_id=source.family_id,
        measure_number=measure_number,
        record_id=record_id,
        image_bytes=roi_bytes,
        image_sha256=roi_sha,
        label=label,
        label_sha256=label_sha,
    )


def derive_source_record(source: D10SourceRecord) -> tuple[D10RoiArtifact, ...]:
    """Derive deterministic barline+meter ROI artifacts for one D6 sample."""
    image = _open_grayscale_png(source.image_bytes)
    geometry = _geometry(source.label)
    measures = _sequence("geometry.measures", geometry.get("measures"))
    if not measures:
        _fail("D6 geometry must contain at least one measure")
    artifacts: list[D10RoiArtifact] = []
    seen_measure_numbers: set[int] = set()
    for item in measures:
        if not isinstance(item, Mapping):
            _fail("geometry measure must be a mapping")
        number = _positive_int("measure_number", item.get("measure_number"))
        if number in seen_measure_numbers:
            _fail("duplicate measure_number in D6 geometry")
        seen_measure_numbers.add(number)
        artifacts.append(_artifact(source=source, image=image, measure=item, geometry=geometry, kind="barline"))
        artifacts.append(_artifact(source=source, image=image, measure=item, geometry=geometry, kind="meter"))
    return tuple(artifacts)


def _fresh_external_root(output_root: Path, repository_root: Path | None) -> Path:
    output_root = output_root.resolve()
    if output_root.exists():
        _fail("D10 output_root must be fresh")
    if repository_root is not None:
        repo = repository_root.resolve()
        if output_root == repo or repo in output_root.parents:
            _fail("D10 artifacts must stay outside the repository")
    output_root.mkdir(parents=True, exist_ok=False)
    return output_root


def materialize_stage7d10_derivatives(
    sources: Sequence[D10SourceRecord],
    *,
    output_root: Path,
    repository_root: Path | None = None,
) -> Stage7D10Receipt:
    """Persist hash-bound TRAIN/VALIDATION ROI artifacts outside Git."""
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes, bytearray)):
        _fail("D10 sources must be a sequence")
    if not sources:
        _fail("D10 sources must be non-empty")
    root = _fresh_external_root(output_root, repository_root)

    artifacts: list[D10RoiArtifact] = []
    source_ids: set[str] = set()
    families: set[str] = set()
    for source in sources:
        if not isinstance(source, D10SourceRecord):
            _fail("D10 source entry has wrong type")
        if source.sample_id in source_ids:
            _fail("duplicate D10 source sample_id")
        source_ids.add(source.sample_id)
        families.add(source.family_id)
        artifacts.extend(derive_source_record(source))

    artifacts.sort(key=lambda item: (item.split, item.kind, item.family_id, item.sample_id, item.measure_number))
    images_root = root / "images"
    labels_root = root / "labels"
    images_root.mkdir()
    labels_root.mkdir()

    manifest_records: list[dict[str, object]] = []
    split_counts = {"train": 0, "validation": 0}
    kind_counts = {"barline": 0, "meter": 0}
    meter_class_counts = {name: 0 for name in METER_CLASSES}
    binding_rows: list[dict[str, object]] = []

    for artifact in artifacts:
        image_rel = Path("images") / f"{artifact.record_id}.png"
        label_rel = Path("labels") / f"{artifact.record_id}.json"
        image_path = root / image_rel
        label_path = root / label_rel
        image_path.write_bytes(artifact.image_bytes)
        label_raw = _canonical_json(artifact.label)
        label_path.write_bytes(label_raw)
        if _sha(image_path.read_bytes()) != artifact.image_sha256:
            _fail("persisted ROI image hash mismatch")
        if _sha(label_path.read_bytes()) != artifact.label_sha256:
            _fail("persisted ROI label hash mismatch")

        split_counts[artifact.split] += 1
        kind_counts[artifact.kind] += 1
        if artifact.kind == "meter":
            meter_class = artifact.label["target"]["meter_class"]  # type: ignore[index]
            if meter_class not in meter_class_counts:
                _fail("persisted meter class is outside D9 surface")
            meter_class_counts[meter_class] += 1

        record = {
            "record_id": artifact.record_id,
            "kind": artifact.kind,
            "split": artifact.split,
            "family_id": artifact.family_id,
            "source_sample_id": artifact.sample_id,
            "measure_number": artifact.measure_number,
            "image_path": image_rel.as_posix(),
            "image_sha256": artifact.image_sha256,
            "label_path": label_rel.as_posix(),
            "label_sha256": artifact.label_sha256,
        }
        manifest_records.append(record)
        binding_rows.append(
            {
                "record_id": artifact.record_id,
                "image_sha256": artifact.image_sha256,
                "label_sha256": artifact.label_sha256,
            }
        )

    manifest = {
        "schema_version": STAGE7D10_MANIFEST_SCHEMA,
        "stage7d10_version": STAGE7D10_VERSION,
        "d9_contract_fingerprint": stage7d9_contract_fingerprint(),
        "source_sample_count": len(source_ids),
        "source_family_count": len(families),
        "roi_record_count": len(manifest_records),
        "split_counts": split_counts,
        "kind_counts": kind_counts,
        "meter_class_counts": meter_class_counts,
        "test_records": 0,
        "optimizer_steps": 0,
        "records": manifest_records,
    }
    manifest_raw = _canonical_json(manifest)
    manifest_sha = _sha(manifest_raw)
    (root / "manifest.json").write_bytes(manifest_raw)
    (root / "manifest.sha256").write_text(manifest_sha + "\n", encoding="ascii")
    binding_sha = _sha(_canonical_json(binding_rows))

    receipt = Stage7D10Receipt(
        version=STAGE7D10_VERSION,
        d9_contract_fingerprint=stage7d9_contract_fingerprint(),
        manifest_sha256=manifest_sha,
        artifact_binding_sha256=binding_sha,
        source_sample_count=len(source_ids),
        source_family_count=len(families),
        roi_record_count=len(manifest_records),
        split_counts=split_counts,
        kind_counts=kind_counts,
        meter_class_counts=meter_class_counts,
        test_records=0,
        optimizer_steps=0,
    )
    receipt_raw = _canonical_json(asdict(receipt))
    (root / "receipt.json").write_bytes(receipt_raw)
    (root / "COMPLETE").write_text(_sha(receipt_raw) + "\n", encoding="ascii")
    return receipt
