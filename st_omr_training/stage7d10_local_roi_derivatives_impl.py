"""Stage 7-D10 deterministic local Structure ROI derivatives.

D10 materializes the local image/target surface frozen by Stage 7-D9. It is a
data-derivative layer only: no model, optimizer, backward pass, checkpoint or
TEST evaluation path exists here.

Authoritative D10 builds must consume the exact accepted D6 TRAIN/VALIDATION
surface through the already-verified D7 record loader. Persisted output is then
independently reopened and revalidated before COMPLETE is written.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Final

from PIL import Image

from .stage7c_execution import verify_authoritative_repository, verify_stage7c_runtime
from .stage7d6_specialist_derivatives import (
    EXPECTED_DEVELOPMENT_FAMILY_COUNTS,
    EXPECTED_DEVELOPMENT_SAMPLE_COUNTS,
    STAGE7D6_LABEL_SCHEMA,
    STAGE7D6_VERSION,
)
from .stage7d7_specialist_training import (
    Stage7D7Record,
    load_verified_stage7d7_records,
)
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
_MAX_LABEL_BYTES: Final[int] = 2 * 1024 * 1024
_MAX_MANIFEST_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES: Final[int] = 1024 * 1024
_EXPECTED_SAMPLE_COUNT: Final[int] = sum(EXPECTED_DEVELOPMENT_SAMPLE_COUNTS.values())
_EXPECTED_FAMILY_COUNT: Final[int] = sum(EXPECTED_DEVELOPMENT_FAMILY_COUNTS.values())
ProgressCallback = Callable[[str, Mapping[str, object]], None]


class Stage7D10DerivativeError(RuntimeError):
    """Raised when a D10 derivative or verification gate fails closed."""


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


def _regular_file(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        _fail(f"{name} must be a regular non-symlink file")


def _regular_directory(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_dir():
        _fail(f"{name} must be a regular non-symlink directory")


def _read_bounded(path: Path, maximum: int, name: str) -> bytes:
    _regular_file(path, name)
    size = path.stat().st_size
    if not 1 <= size <= maximum:
        _fail(f"{name} byte length is outside D10 bounds")
    return path.read_bytes()


def _read_canonical_json(path: Path, maximum: int, name: str) -> tuple[dict[str, object], bytes]:
    raw = _read_bounded(path, maximum, name)
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda token: _fail(f"non-finite JSON constant in {name}: {token}"),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D10DerivativeError(f"{name} is not valid ASCII JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != raw:
        _fail(f"{name} must be canonical JSON object bytes")
    return value, raw


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
    """Return TRAIN/VALIDATION rows while touching only ``split`` on TEST."""
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
    repository_sha: str | None
    manifest_sha256: str
    artifact_binding_sha256: str
    source_sample_count: int
    source_family_count: int
    source_split_counts: dict[str, int]
    source_family_split_counts: dict[str, int]
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


def _staff_for_measure(geometry: Mapping[str, object], measure: Mapping[str, object]) -> Mapping[str, object]:
    system_id = _identifier("measure.system_id", measure.get("system_id"))
    staffs = _sequence("geometry.staff_instances", geometry.get("staff_instances"))
    matches = [staff for staff in staffs if isinstance(staff, Mapping) and staff.get("system_id") == system_id]
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
    scale = min(policy.output_width / crop_w, policy.output_height / crop_h)
    resized_w = max(1, min(policy.output_width, int(round(crop_w * scale))))
    resized_h = max(1, min(policy.output_height, int(round(crop_h * scale))))
    return RoiTransform(
        crop_left=left,
        crop_top=top,
        crop_right=right,
        crop_bottom=bottom,
        resized_width=resized_w,
        resized_height=resized_h,
        output_width=policy.output_width,
        output_height=policy.output_height,
        pad_left=(policy.output_width - resized_w) // 2,
        pad_top=(policy.output_height - resized_h) // 2,
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
    crop = image.crop((transform.crop_left, transform.crop_top, transform.crop_right, transform.crop_bottom))
    resized = crop.resize((transform.resized_width, transform.resized_height), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("L", (transform.output_width, transform.output_height), 255)
    canvas.paste(resized, (transform.pad_left, transform.pad_top))
    buffer = BytesIO()
    canvas.save(buffer, format="PNG", optimize=False, compress_level=9)
    raw = buffer.getvalue()
    if not raw:
        _fail("ROI PNG encoder returned empty bytes")
    return raw


def _record_id_from_values(
    *,
    sample_id: str,
    image_sha256: str,
    label_sha256: str,
    split: str,
    kind: str,
    measure_number: int,
    policy_id: str,
) -> str:
    return _sha(
        _canonical_json(
            {
                "version": STAGE7D10_VERSION,
                "d9_contract_fingerprint": stage7d9_contract_fingerprint(),
                "source_sample_id": sample_id,
                "source_image_sha256": image_sha256,
                "source_label_sha256": label_sha256,
                "split": split,
                "kind": kind,
                "measure_number": measure_number,
                "roi_policy_id": policy_id,
            }
        )
    )


def _artifact(
    *, source: D10SourceRecord, image: Image.Image, measure: Mapping[str, object],
    geometry: Mapping[str, object], kind: str,
) -> D10RoiArtifact:
    measure_number = _positive_int("measure_number", measure.get("measure_number"))
    staff = _staff_for_measure(geometry, measure)
    policy = BARLINE_ROI if kind == "barline" else METER_ROI
    transform = _crop_transform(
        measure_bbox=measure.get("measure_bbox"),
        staff_bbox=staff.get("staff_instance_bbox"),
        staff_spacing=_finite_number("staff_spacing", staff.get("staff_spacing")),
        image_width=image.width,
        image_height=image.height,
        policy=policy,
    )
    roi_bytes = _render_roi(image, transform)
    roi_sha = _sha(roi_bytes)
    if kind == "barline":
        target: dict[str, object] = {"barline_segment": _map_line(transform, measure.get("barline_segment"))}
    elif kind == "meter":
        meter_bbox = measure.get("meter_bbox")
        if meter_bbox is None:
            meter_class: object = "none"
            mapped_bbox: object = None
        else:
            meter_class = measure.get("meter_class")
            if meter_class not in METER_CLASSES[1:]:
                _fail("visible meter bbox requires supported D9 meter class")
            mapped_bbox = _map_box(transform, meter_bbox)
        target = {"meter_class": meter_class, "meter_bbox": mapped_bbox}
    else:
        _fail("unknown D10 ROI kind")
    record_id = _record_id_from_values(
        sample_id=source.sample_id,
        image_sha256=source.image_sha256,
        label_sha256=source.label_sha256,
        split=source.split,
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
    seen: set[int] = set()
    for item in measures:
        if not isinstance(item, Mapping):
            _fail("geometry measure must be a mapping")
        number = _positive_int("measure_number", item.get("measure_number"))
        if number in seen:
            _fail("duplicate measure_number in D6 geometry")
        seen.add(number)
        artifacts.append(_artifact(source=source, image=image, measure=item, geometry=geometry, kind="barline"))
        artifacts.append(_artifact(source=source, image=image, measure=item, geometry=geometry, kind="meter"))
    return tuple(artifacts)


def _validate_generic_sources(sources: Sequence[D10SourceRecord]) -> tuple[D10SourceRecord, ...]:
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes, bytearray)) or not sources:
        _fail("D10 sources must be a non-empty sequence")
    sample_ids: set[str] = set()
    family_split: dict[str, str] = {}
    checked: list[D10SourceRecord] = []
    for source in sources:
        if not isinstance(source, D10SourceRecord):
            _fail("D10 source entry has wrong type")
        if source.sample_id in sample_ids:
            _fail("duplicate D10 source sample_id")
        sample_ids.add(source.sample_id)
        prior = family_split.setdefault(source.family_id, source.split)
        if prior != source.split:
            _fail("D10 family crosses train/validation split")
        checked.append(source)
    return tuple(sorted(checked, key=lambda item: (item.split, item.family_id, item.sample_id)))


def _assert_authoritative_record_surface(records: Sequence[Stage7D7Record]) -> None:
    if len(records) != _EXPECTED_SAMPLE_COUNT:
        _fail("authoritative D10 source sample count differs from accepted D6")
    sample_ids: set[str] = set()
    sample_counts: Counter[str] = Counter()
    families: dict[str, set[str]] = {"train": set(), "validation": set()}
    family_split: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Stage7D7Record):
            _fail("authoritative D10 record has unexpected type")
        if record.split not in _ALLOWED_SPLITS:
            _fail("authoritative D10 record reached forbidden split")
        if record.sample_id in sample_ids:
            _fail("duplicate authoritative D10 sample_id")
        sample_ids.add(record.sample_id)
        sample_counts[record.split] += 1
        families[record.split].add(record.family_id)
        prior = family_split.setdefault(record.family_id, record.split)
        if prior != record.split:
            _fail("authoritative D10 family crosses split")
    if dict(sorted(sample_counts.items())) != EXPECTED_DEVELOPMENT_SAMPLE_COUNTS:
        _fail("authoritative D10 sample split counts differ from accepted D6")
    if {split: len(values) for split, values in families.items()} != EXPECTED_DEVELOPMENT_FAMILY_COUNTS:
        _fail("authoritative D10 family split counts differ from accepted D6")
    if len(family_split) != _EXPECTED_FAMILY_COUNT:
        _fail("authoritative D10 family count differs from accepted D6")


def load_authoritative_stage7d10_records(
    corpus_root: str | Path, derivative_root: str | Path
) -> tuple[Stage7D7Record, ...]:
    """Load exactly the accepted D6 TRAIN/VALIDATION surface; TEST stays sealed."""
    records = load_verified_stage7d7_records(corpus_root, derivative_root)
    _assert_authoritative_record_surface(records)
    return records


def _source_from_verified_record(record: Stage7D7Record) -> D10SourceRecord:
    label, label_raw = _read_canonical_json(record.label_path, _MAX_LABEL_BYTES, "D10 D6 label")
    if _sha(label_raw) != record.label_sha256:
        _fail("D10 D6 label SHA-256 mismatch")
    if label.get("schema_version") != STAGE7D6_LABEL_SCHEMA or label.get("stage7d6_version") != STAGE7D6_VERSION:
        _fail("D10 source label is not accepted D6 schema/version")
    if label.get("sample_id") != record.sample_id or label.get("family_id") != record.family_id or label.get("split") != record.split:
        _fail("D10 D6 label identity/split mismatch")
    image_meta = label.get("image")
    if not isinstance(image_meta, Mapping):
        _fail("D10 D6 label image metadata missing")
    if image_meta.get("png_sha256") != record.png_sha256 or image_meta.get("mode") != "L" or image_meta.get("image_format") != "png":
        _fail("D10 D6 image metadata mismatch")
    raw = _read_bounded(record.image_path, _MAX_IMAGE_BYTES, "D10 source PNG")
    if _sha(raw) != record.png_sha256:
        _fail("D10 source PNG SHA-256 mismatch")
    image = _open_grayscale_png(raw)
    if image.size != (
        _positive_int("D10 source image width", image_meta.get("width")),
        _positive_int("D10 source image height", image_meta.get("height")),
    ):
        _fail("D10 source PNG dimensions differ from D6 label")
    return D10SourceRecord(
        split=record.split,
        sample_id=record.sample_id,
        family_id=record.family_id,
        image_sha256=record.png_sha256,
        label_sha256=record.label_sha256,
        image_bytes=raw,
        label=label,
    )


def _fresh_external_root(output_root: Path, repository_root: Path | None) -> Path:
    root = output_root.resolve()
    if root.exists():
        _fail("D10 output_root must be fresh")
    if repository_root is not None:
        repo = repository_root.resolve()
        if root == repo or repo in root.parents:
            _fail("D10 artifacts must stay outside the repository")
    root.mkdir(parents=True, exist_ok=False)
    return root


def _write_new(path: Path, raw: bytes, name: str) -> None:
    if path.exists() or path.is_symlink():
        _fail(f"refusing to overwrite D10 {name}")
    path.write_bytes(raw)


def _emit(progress: ProgressCallback | None, event: str, payload: Mapping[str, object]) -> None:
    if progress is not None:
        progress(event, payload)


def _persist_sources(
    sources: Iterable[D10SourceRecord],
    *, output_root: Path, repository_root: Path | None, repository_sha: str | None,
    authoritative: bool, progress: ProgressCallback | None, total_sources: int | None,
) -> Stage7D10Receipt:
    if repository_sha is not None:
        _hex64("repository_sha", repository_sha)
    root = _fresh_external_root(output_root, repository_root)
    images_root = root / "images"
    labels_root = root / "labels"
    images_root.mkdir()
    labels_root.mkdir()

    manifest_records: list[dict[str, object]] = []
    sample_map: dict[str, tuple[str, str]] = {}
    family_split: dict[str, str] = {}
    roi_split_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    meter_counts: Counter[str] = Counter()
    binding_rows: list[dict[str, object]] = []

    for index, source in enumerate(sources, start=1):
        if not isinstance(source, D10SourceRecord):
            _fail("D10 source stream entry has wrong type")
        if source.sample_id in sample_map:
            _fail("duplicate D10 source sample_id")
        sample_map[source.sample_id] = (source.family_id, source.split)
        prior = family_split.setdefault(source.family_id, source.split)
        if prior != source.split:
            _fail("D10 family crosses train/validation split")
        artifacts = derive_source_record(source)
        for artifact in artifacts:
            image_rel = Path("images") / f"{artifact.record_id}.png"
            label_rel = Path("labels") / f"{artifact.record_id}.json"
            _write_new(root / image_rel, artifact.image_bytes, "ROI image")
            label_raw = _canonical_json(artifact.label)
            _write_new(root / label_rel, label_raw, "ROI label")
            if _sha((root / image_rel).read_bytes()) != artifact.image_sha256:
                _fail("persisted ROI image hash mismatch")
            if _sha((root / label_rel).read_bytes()) != artifact.label_sha256:
                _fail("persisted ROI label hash mismatch")
            roi_split_counts[artifact.split] += 1
            kind_counts[artifact.kind] += 1
            if artifact.kind == "meter":
                target = artifact.label.get("target")
                if not isinstance(target, Mapping):
                    _fail("meter target missing during persistence")
                meter_class = target.get("meter_class")
                if meter_class not in METER_CLASSES:
                    _fail("persisted meter class is outside D9 surface")
                meter_counts[str(meter_class)] += 1
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
            binding_rows.append({
                "record_id": artifact.record_id,
                "image_sha256": artifact.image_sha256,
                "label_sha256": artifact.label_sha256,
            })
        _emit(progress, "source_complete", {
            "index": index,
            "total": total_sources,
            "sample_id": source.sample_id,
            "split": source.split,
            "roi_records": len(artifacts),
        })

    if not sample_map:
        _fail("D10 source stream is empty")
    source_split_counts = dict(sorted(Counter(split for _, split in sample_map.values()).items()))
    source_family_split_counts = {
        split: len({family for family, family_value_split in family_split.items() if family_value_split == split})
        for split in ("train", "validation")
    }
    if authoritative:
        if len(sample_map) != _EXPECTED_SAMPLE_COUNT or len(family_split) != _EXPECTED_FAMILY_COUNT:
            _fail("authoritative D10 source cardinality mismatch")
        if source_split_counts != EXPECTED_DEVELOPMENT_SAMPLE_COUNTS:
            _fail("authoritative D10 source split counts mismatch")
        if source_family_split_counts != EXPECTED_DEVELOPMENT_FAMILY_COUNTS:
            _fail("authoritative D10 family split counts mismatch")

    manifest_records.sort(key=lambda item: (
        str(item["split"]), str(item["kind"]), str(item["family_id"]),
        str(item["source_sample_id"]), int(item["measure_number"]),
    ))
    binding_rows.sort(key=lambda item: str(item["record_id"]))
    meter_class_counts = {name: meter_counts.get(name, 0) for name in METER_CLASSES}
    split_counts = {split: roi_split_counts.get(split, 0) for split in ("train", "validation")}
    normalized_kind_counts = {kind: kind_counts.get(kind, 0) for kind in ("barline", "meter")}
    manifest = {
        "schema_version": STAGE7D10_MANIFEST_SCHEMA,
        "stage7d10_version": STAGE7D10_VERSION,
        "d9_contract_fingerprint": stage7d9_contract_fingerprint(),
        "repository_sha": repository_sha,
        "source_sample_count": len(sample_map),
        "source_family_count": len(family_split),
        "source_split_counts": source_split_counts,
        "source_family_split_counts": source_family_split_counts,
        "roi_record_count": len(manifest_records),
        "split_counts": split_counts,
        "kind_counts": normalized_kind_counts,
        "meter_class_counts": meter_class_counts,
        "test_records": 0,
        "optimizer_steps": 0,
        "records": manifest_records,
    }
    manifest_raw = _canonical_json(manifest)
    manifest_sha = _sha(manifest_raw)
    _write_new(root / "manifest.json", manifest_raw, "manifest")
    _write_new(root / "manifest.sha256", f"{manifest_sha}  manifest.json\n".encode("ascii"), "manifest SHA")
    binding_sha = _sha(_canonical_json(binding_rows))
    receipt = Stage7D10Receipt(
        version=STAGE7D10_VERSION,
        d9_contract_fingerprint=stage7d9_contract_fingerprint(),
        repository_sha=repository_sha,
        manifest_sha256=manifest_sha,
        artifact_binding_sha256=binding_sha,
        source_sample_count=len(sample_map),
        source_family_count=len(family_split),
        source_split_counts=source_split_counts,
        source_family_split_counts=source_family_split_counts,
        roi_record_count=len(manifest_records),
        split_counts=split_counts,
        kind_counts=normalized_kind_counts,
        meter_class_counts=meter_class_counts,
        test_records=0,
        optimizer_steps=0,
    )
    receipt_raw = _canonical_json(asdict(receipt))
    _write_new(root / "receipt.json", receipt_raw, "receipt")

    independently_verified = verify_stage7d10_derivatives(
        root,
        expected_authoritative_surface=authoritative,
        expected_repository_sha=repository_sha,
        require_complete=False,
    )
    if independently_verified != receipt:
        _fail("independent D10 verification receipt mismatch")
    _write_new(root / "COMPLETE", (_sha(receipt_raw) + "\n").encode("ascii"), "COMPLETE")
    final_verified = verify_stage7d10_derivatives(
        root,
        expected_authoritative_surface=authoritative,
        expected_repository_sha=repository_sha,
        require_complete=True,
    )
    if final_verified != receipt:
        _fail("final D10 verification receipt mismatch")
    return final_verified


def materialize_stage7d10_derivatives(
    sources: Sequence[D10SourceRecord], *, output_root: Path,
    repository_root: Path | None = None,
) -> Stage7D10Receipt:
    """Materialize a bounded fixture/development sequence; not an authoritative gate."""
    checked = _validate_generic_sources(sources)
    return _persist_sources(
        checked,
        output_root=output_root,
        repository_root=repository_root,
        repository_sha=None,
        authoritative=False,
        progress=None,
        total_sources=len(checked),
    )


def run_verified_stage7d10_derivative_build(
    *, corpus_root: str | Path, derivative_root: str | Path, output_root: str | Path,
    repository_root: str | Path, expected_repository_sha: str,
    progress: ProgressCallback | None = None,
) -> Stage7D10Receipt:
    """Build authoritative D10 derivatives from the exact accepted D6 surface."""
    expected_sha = _hex64("expected_repository_sha", expected_repository_sha)
    actual_sha = verify_authoritative_repository(Path(repository_root))
    if actual_sha != expected_sha:
        _fail("D10 repository head differs from the authorized exact head")
    verify_stage7c_runtime()
    records = load_authoritative_stage7d10_records(corpus_root, derivative_root)
    _emit(progress, "build_started", {
        "repository_sha": actual_sha,
        "source_records": len(records),
        "test_records": 0,
        "optimizer_steps": 0,
    })
    def source_stream() -> Iterable[D10SourceRecord]:
        for record in records:
            yield _source_from_verified_record(record)
    result = _persist_sources(
        source_stream(),
        output_root=Path(output_root),
        repository_root=Path(repository_root),
        repository_sha=actual_sha,
        authoritative=True,
        progress=progress,
        total_sources=len(records),
    )
    if verify_authoritative_repository(Path(repository_root)) != actual_sha:
        _fail("repository changed during D10 authoritative build")
    verify_stage7c_runtime()
    _emit(progress, "build_complete", {
        "manifest_sha256": result.manifest_sha256,
        "artifact_binding_sha256": result.artifact_binding_sha256,
        "roi_records": result.roi_record_count,
        "test_records": result.test_records,
        "optimizer_steps": result.optimizer_steps,
    })
    return result


def _expected_policy(kind: str) -> LocalRoiPolicy:
    if kind == "barline":
        return BARLINE_ROI
    if kind == "meter":
        return METER_ROI
    _fail("unknown persisted D10 kind")


def _validate_output_target(kind: str, target: object, transform: RoiTransform) -> str | None:
    if not isinstance(target, Mapping):
        _fail("persisted D10 target must be an object")
    if kind == "barline":
        if set(target) != {"barline_segment"}:
            _fail("persisted barline target shape mismatch")
        line = _line("persisted barline_segment", target.get("barline_segment"))
        for point in (line["start"], line["end"]):
            if not 0 <= point["x"] <= transform.output_width or not 0 <= point["y"] <= transform.output_height:
                _fail("persisted barline target lies outside ROI")
        return None
    if set(target) != {"meter_class", "meter_bbox"}:
        _fail("persisted meter target shape mismatch")
    meter_class = target.get("meter_class")
    if meter_class not in METER_CLASSES:
        _fail("persisted meter class outside D9 surface")
    bbox = target.get("meter_bbox")
    if meter_class == "none":
        if bbox is not None:
            _fail("meter none class must not carry a bbox")
    else:
        box = _box("persisted meter_bbox", bbox)
        if not (0 <= box["x_min"] < box["x_max"] <= transform.output_width):
            _fail("persisted meter bbox x lies outside ROI")
        if not (0 <= box["y_min"] < box["y_max"] <= transform.output_height):
            _fail("persisted meter bbox y lies outside ROI")
    return str(meter_class)


def verify_stage7d10_derivatives(
    output_root: str | Path, *, expected_authoritative_surface: bool = True,
    expected_repository_sha: str | None = None, require_complete: bool = True,
) -> Stage7D10Receipt:
    """Independently reopen and validate persisted D10 output."""
    root = Path(output_root)
    _regular_directory(root, "D10 output root")
    expected_top = {"images", "labels", "manifest.json", "manifest.sha256", "receipt.json"}
    if require_complete:
        expected_top.add("COMPLETE")
    if {path.name for path in root.iterdir()} != expected_top:
        _fail("D10 output top-level shape mismatch")
    images_root = root / "images"
    labels_root = root / "labels"
    _regular_directory(images_root, "D10 images directory")
    _regular_directory(labels_root, "D10 labels directory")

    manifest, manifest_raw = _read_canonical_json(root / "manifest.json", _MAX_MANIFEST_BYTES, "D10 manifest")
    manifest_sha = _sha(manifest_raw)
    manifest_sha_raw = _read_bounded(root / "manifest.sha256", 256, "D10 manifest SHA")
    if manifest_sha_raw != f"{manifest_sha}  manifest.json\n".encode("ascii"):
        _fail("D10 manifest SHA sidecar mismatch")
    receipt_payload, receipt_raw = _read_canonical_json(root / "receipt.json", _MAX_RECEIPT_BYTES, "D10 receipt")

    if manifest.get("schema_version") != STAGE7D10_MANIFEST_SCHEMA or manifest.get("stage7d10_version") != STAGE7D10_VERSION:
        _fail("D10 manifest schema/version mismatch")
    if manifest.get("d9_contract_fingerprint") != stage7d9_contract_fingerprint():
        _fail("D10 manifest D9 contract fingerprint mismatch")
    repository_sha = manifest.get("repository_sha")
    if repository_sha is not None:
        repository_sha = _hex64("manifest.repository_sha", repository_sha)
    if expected_repository_sha is not None:
        expected = _hex64("expected_repository_sha", expected_repository_sha)
        if repository_sha != expected:
            _fail("D10 manifest repository SHA mismatch")
    if manifest.get("test_records") != 0 or manifest.get("optimizer_steps") != 0:
        _fail("D10 manifest violates TEST/optimizer seal")

    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        _fail("D10 manifest records must be a non-empty list")
    if len(records) > 100_000:
        _fail("D10 manifest record count exceeds safety bound")
    record_ids: set[str] = set()
    expected_images: set[str] = set()
    expected_labels: set[str] = set()
    sample_map: dict[str, tuple[str, str]] = {}
    family_split: dict[str, str] = {}
    pair_kinds: dict[tuple[str, int], set[str]] = {}
    roi_split_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    meter_counts: Counter[str] = Counter()
    binding_rows: list[dict[str, object]] = []

    record_keys = {
        "record_id", "kind", "split", "family_id", "source_sample_id",
        "measure_number", "image_path", "image_sha256", "label_path", "label_sha256",
    }
    label_keys = {
        "schema_version", "stage7d10_version", "d9_contract_fingerprint", "record_id",
        "kind", "split", "source", "measure_number", "roi_policy_id", "roi_transform",
        "roi_image_sha256", "target",
    }
    source_keys = {"sample_id", "family_id", "image_sha256", "d6_label_sha256"}

    for record in records:
        if not isinstance(record, Mapping) or set(record) != record_keys:
            _fail("D10 manifest record shape mismatch")
        record_id = _hex64("record_id", record.get("record_id"))
        if record_id in record_ids:
            _fail("duplicate D10 record_id")
        record_ids.add(record_id)
        kind = record.get("kind")
        if kind not in _ALLOWED_KINDS:
            _fail("persisted D10 kind mismatch")
        kind = str(kind)
        split = record.get("split")
        if split not in _ALLOWED_SPLITS:
            _fail("persisted D10 split mismatch")
        split = str(split)
        family_id = _identifier("family_id", record.get("family_id"))
        sample_id = _identifier("source_sample_id", record.get("source_sample_id"))
        measure_number = _positive_int("measure_number", record.get("measure_number"))
        image_sha = _hex64("image_sha256", record.get("image_sha256"))
        label_sha = _hex64("label_sha256", record.get("label_sha256"))
        expected_image_rel = f"images/{record_id}.png"
        expected_label_rel = f"labels/{record_id}.json"
        if record.get("image_path") != expected_image_rel or record.get("label_path") != expected_label_rel:
            _fail("D10 artifact path is not canonical/hash-addressed")
        image_path = root / expected_image_rel
        label_path = root / expected_label_rel
        image_raw = _read_bounded(image_path, _MAX_IMAGE_BYTES, "D10 ROI image")
        if _sha(image_raw) != image_sha:
            _fail("D10 ROI image hash mismatch")
        policy = _expected_policy(kind)
        image = _open_grayscale_png(image_raw)
        if image.size != (policy.output_width, policy.output_height):
            _fail("D10 ROI image dimensions differ from frozen D9 policy")
        label, label_raw = _read_canonical_json(label_path, _MAX_LABEL_BYTES, "D10 ROI label")
        if _sha(label_raw) != label_sha or set(label) != label_keys:
            _fail("D10 ROI label hash/shape mismatch")
        if label.get("schema_version") != STAGE7D10_LABEL_SCHEMA or label.get("stage7d10_version") != STAGE7D10_VERSION:
            _fail("D10 ROI label schema/version mismatch")
        if label.get("d9_contract_fingerprint") != stage7d9_contract_fingerprint():
            _fail("D10 ROI label D9 fingerprint mismatch")
        if label.get("record_id") != record_id or label.get("kind") != kind or label.get("split") != split or label.get("measure_number") != measure_number:
            _fail("D10 ROI label identity mismatch")
        if label.get("roi_policy_id") != policy.policy_id or label.get("roi_image_sha256") != image_sha:
            _fail("D10 ROI policy/image binding mismatch")
        source = label.get("source")
        if not isinstance(source, Mapping) or set(source) != source_keys:
            _fail("D10 ROI source binding shape mismatch")
        if source.get("sample_id") != sample_id or source.get("family_id") != family_id:
            _fail("D10 ROI source identity mismatch")
        source_image_sha = _hex64("source.image_sha256", source.get("image_sha256"))
        source_label_sha = _hex64("source.d6_label_sha256", source.get("d6_label_sha256"))
        expected_record_id = _record_id_from_values(
            sample_id=sample_id,
            image_sha256=source_image_sha,
            label_sha256=source_label_sha,
            split=split,
            kind=kind,
            measure_number=measure_number,
            policy_id=policy.policy_id,
        )
        if expected_record_id != record_id:
            _fail("D10 record_id provenance binding mismatch")
        transform_payload = label.get("roi_transform")
        if not isinstance(transform_payload, Mapping):
            _fail("D10 ROI transform missing")
        try:
            transform = RoiTransform(**dict(transform_payload))
        except (TypeError, ValueError) as exc:
            raise Stage7D10DerivativeError("D10 ROI transform invalid") from exc
        if transform.output_width != policy.output_width or transform.output_height != policy.output_height:
            _fail("D10 ROI transform output differs from D9 policy")
        meter_class = _validate_output_target(kind, label.get("target"), transform)
        if meter_class is not None:
            meter_counts[meter_class] += 1

        prior_sample = sample_map.setdefault(sample_id, (family_id, split))
        if prior_sample != (family_id, split):
            _fail("D10 source sample changes family/split across ROI records")
        prior_family = family_split.setdefault(family_id, split)
        if prior_family != split:
            _fail("D10 family crosses train/validation split")
        key = (sample_id, measure_number)
        kinds = pair_kinds.setdefault(key, set())
        if kind in kinds:
            _fail("duplicate D10 kind for one source measure")
        kinds.add(kind)
        roi_split_counts[split] += 1
        kind_counts[kind] += 1
        expected_images.add(f"{record_id}.png")
        expected_labels.add(f"{record_id}.json")
        binding_rows.append({"record_id": record_id, "image_sha256": image_sha, "label_sha256": label_sha})

    if any(kinds != _ALLOWED_KINDS for kinds in pair_kinds.values()):
        _fail("every D10 source measure must have exactly barline+meter ROI records")
    actual_images = {path.name for path in images_root.iterdir() if path.is_file() and not path.is_symlink()}
    actual_labels = {path.name for path in labels_root.iterdir() if path.is_file() and not path.is_symlink()}
    if actual_images != expected_images or actual_labels != expected_labels:
        _fail("D10 images/labels directory contents differ from manifest")
    if any(path.is_dir() or path.is_symlink() for path in images_root.iterdir()):
        _fail("D10 images directory contains non-regular artifact")
    if any(path.is_dir() or path.is_symlink() for path in labels_root.iterdir()):
        _fail("D10 labels directory contains non-regular artifact")

    source_split_counts = dict(sorted(Counter(split for _, split in sample_map.values()).items()))
    source_family_split_counts = {
        split: len({family for family, family_value_split in family_split.items() if family_value_split == split})
        for split in ("train", "validation")
    }
    split_counts = {split: roi_split_counts.get(split, 0) for split in ("train", "validation")}
    normalized_kind_counts = {kind: kind_counts.get(kind, 0) for kind in ("barline", "meter")}
    meter_class_counts = {name: meter_counts.get(name, 0) for name in METER_CLASSES}
    if normalized_kind_counts["barline"] != normalized_kind_counts["meter"]:
        _fail("D10 barline/meter ROI counts must match")
    if expected_authoritative_surface:
        if len(sample_map) != _EXPECTED_SAMPLE_COUNT or len(family_split) != _EXPECTED_FAMILY_COUNT:
            _fail("authoritative persisted D10 source cardinality mismatch")
        if source_split_counts != EXPECTED_DEVELOPMENT_SAMPLE_COUNTS:
            _fail("authoritative persisted D10 sample split mismatch")
        if source_family_split_counts != EXPECTED_DEVELOPMENT_FAMILY_COUNTS:
            _fail("authoritative persisted D10 family split mismatch")

    expected_summary = {
        "source_sample_count": len(sample_map),
        "source_family_count": len(family_split),
        "source_split_counts": source_split_counts,
        "source_family_split_counts": source_family_split_counts,
        "roi_record_count": len(records),
        "split_counts": split_counts,
        "kind_counts": normalized_kind_counts,
        "meter_class_counts": meter_class_counts,
        "test_records": 0,
        "optimizer_steps": 0,
    }
    for key, value in expected_summary.items():
        if manifest.get(key) != value:
            _fail(f"D10 manifest summary mismatch: {key}")
    binding_rows.sort(key=lambda item: str(item["record_id"]))
    receipt = Stage7D10Receipt(
        version=STAGE7D10_VERSION,
        d9_contract_fingerprint=stage7d9_contract_fingerprint(),
        repository_sha=repository_sha,
        manifest_sha256=manifest_sha,
        artifact_binding_sha256=_sha(_canonical_json(binding_rows)),
        source_sample_count=len(sample_map),
        source_family_count=len(family_split),
        source_split_counts=source_split_counts,
        source_family_split_counts=source_family_split_counts,
        roi_record_count=len(records),
        split_counts=split_counts,
        kind_counts=normalized_kind_counts,
        meter_class_counts=meter_class_counts,
        test_records=0,
        optimizer_steps=0,
    )
    if receipt_payload != asdict(receipt):
        _fail("D10 receipt does not match independently rederived output")
    if require_complete:
        complete = _read_bounded(root / "COMPLETE", 256, "D10 COMPLETE")
        if complete != (_sha(receipt_raw) + "\n").encode("ascii"):
            _fail("D10 COMPLETE marker mismatch")
    return receipt
