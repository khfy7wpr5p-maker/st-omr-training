"""Independent persisted verifier for Stage 7-D13 measure derivatives.

D13-2 does not call the D13 builder and does not trust its in-memory receipt.
It reopens the accepted D12 bundle, frozen source PNGs, D13 manifest/images/
labels/build evidence, independently recomputes crop/letterbox geometry and
rendered measure PNG bytes, and verifies inventory/binding/cardinality.

The derivative root must still be uncompleted: premature COMPLETE is rejected.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import re
from typing import Final

from PIL import Image

from .stage7d12_symbol_derivative_verifier import verify_stage7d12_symbol_derivatives
from .stage7d12_symbol_derivatives import (
    STAGE7D12_DERIVATIVE_VERSION,
    STAGE7D12_LABEL_SCHEMA,
    STAGE7D12_MANIFEST_SCHEMA,
)
from .stage7d13_measure_derivatives import (
    STAGE7D13_BUILD_SCHEMA,
    STAGE7D13_DERIVATIVE_VERSION,
    STAGE7D13_LABEL_SCHEMA,
    STAGE7D13_MANIFEST_SCHEMA,
    STAGE7D13_TRANSFORM_VERSION,
    stage7d13_derivative_profile_fingerprint,
)
from .stage7d13_symbol_training_contract import (
    D12_CLASS_INVENTORY,
    EXPECTED_D12_ARTIFACT_BINDING_SHA256,
    EXPECTED_D12_DERIVATIVE_BUILD_ID,
    EXPECTED_D12_MANIFEST_SHA256,
    EXPECTED_SOURCE_FAMILY_COUNTS,
    EXPECTED_SOURCE_SAMPLE_COUNTS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    SPECIALIST_CLASSES,
    TEST_SPECIALIST_RECORDS,
    WHITE_BACKGROUND,
    stage7d13_contract_fingerprint,
)


STAGE7D13_VERIFIER_VERSION: Final[str] = "stage7d13-measure-persisted-verifier-v1"
_ALLOWED_SPLITS: Final[frozenset[str]] = frozenset({"train", "validation"})
_EXPECTED_SAMPLE_COUNT: Final[int] = sum(EXPECTED_SOURCE_SAMPLE_COUNTS.values())
_EXPECTED_FAMILY_COUNT: Final[int] = sum(EXPECTED_SOURCE_FAMILY_COUNTS.values())
_TOP_LEVEL: Final[frozenset[str]] = frozenset(
    {"manifest.json", "manifest.sha256", "build.json", "images", "labels"}
)
_MAX_MANIFEST_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_BUILD_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_LABEL_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_IMAGE_BYTES: Final[int] = 32 * 1024 * 1024
_HEX64: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_EPS: Final[float] = 1e-5


class Stage7D13VerificationError(RuntimeError):
    """Raised when persisted D13 derivative evidence fails independent proof."""


def _fail(message: str) -> None:
    raise Stage7D13VerificationError(message)


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
        raise Stage7D13VerificationError("payload is not canonical JSON serializable") from exc


def _hex64(name: str, value: object) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        _fail(f"{name} must be lowercase SHA-256 hex")
    return value


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or len(value) > 256:
        _fail(f"{name} must be bounded non-empty ASCII")
    return value


def _positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _fail(f"{name} must be a positive integer")
    return value


def _finite(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


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
        _fail(f"{name} byte length is outside D13 verifier bound")
    return path.read_bytes()


def _read_canonical_json(path: Path, maximum: int, name: str) -> tuple[dict[str, object], bytes]:
    raw = _read_bounded(path, maximum, name)
    try:
        payload = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda token: _fail(f"non-finite JSON constant in {name}: {token}"),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D13VerificationError(f"{name} is not valid ASCII JSON") from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        _fail(f"{name} must be canonical JSON object bytes")
    return payload, raw


def _bbox(name: str, value: object, *, width: int, height: int) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {"x_min", "y_min", "x_max", "y_max"}:
        _fail(f"{name} must be a canonical bbox")
    box = {key: _finite(f"{name}.{key}", value.get(key)) for key in ("x_min", "y_min", "x_max", "y_max")}
    if not box["x_min"] < box["x_max"] or not box["y_min"] < box["y_max"]:
        _fail(f"{name} must have positive area")
    if box["x_min"] < -_EPS or box["y_min"] < -_EPS or box["x_max"] > width + _EPS or box["y_max"] > height + _EPS:
        _fail(f"{name} leaves image bounds")
    return box


def _point(name: str, value: object, *, width: int, height: int) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {"x", "y"}:
        _fail(f"{name} must be a canonical point")
    point = {"x": _finite(f"{name}.x", value.get("x")), "y": _finite(f"{name}.y", value.get("y"))}
    if point["x"] < -_EPS or point["y"] < -_EPS or point["x"] > width + _EPS or point["y"] > height + _EPS:
        _fail(f"{name} leaves image bounds")
    return point


def _box_inside(inner: Mapping[str, float], outer: Mapping[str, float]) -> bool:
    return (
        inner["x_min"] >= outer["x_min"] - _EPS
        and inner["y_min"] >= outer["y_min"] - _EPS
        and inner["x_max"] <= outer["x_max"] + _EPS
        and inner["y_max"] <= outer["y_max"] + _EPS
    )


def _point_inside(point: Mapping[str, float], box: Mapping[str, float]) -> bool:
    return (
        box["x_min"] - _EPS <= point["x"] <= box["x_max"] + _EPS
        and box["y_min"] - _EPS <= point["y"] <= box["y_max"] + _EPS
    )


@dataclass(frozen=True, slots=True)
class _Plan:
    left: int
    top: int
    right: int
    bottom: int
    scale: float
    pad_x: float
    pad_y: float
    fingerprint: str


@dataclass(frozen=True, slots=True)
class Stage7D13VerificationReceipt:
    verifier_version: str
    derivative_build_id: str
    manifest_sha256: str
    artifact_binding_sha256: str
    record_count: int
    image_count: int
    label_count: int
    source_sample_count: int
    family_count: int
    record_split_counts: dict[str, int]
    source_sample_split_counts: dict[str, int]
    family_split_counts: dict[str, int]
    observed_class_inventory: dict[str, dict[str, dict[str, int]]]
    target_instance_counts: dict[str, dict[str, int]]
    image_bytes_total: int
    label_bytes_total: int
    test_specialist_records: int
    optimizer_steps: int
    complete_marker_present: bool
    verification_passed: bool


def _accepted_d12(corpus_root: Path, d6_root: Path, d12_root: Path) -> None:
    receipt = verify_stage7d12_symbol_derivatives(
        corpus_root=corpus_root,
        d6_root=d6_root,
        derivative_root=d12_root,
    )
    expected = {
        "derivative_build_id": EXPECTED_D12_DERIVATIVE_BUILD_ID,
        "manifest_sha256": EXPECTED_D12_MANIFEST_SHA256,
        "artifact_binding_sha256": EXPECTED_D12_ARTIFACT_BINDING_SHA256,
        "sample_count": _EXPECTED_SAMPLE_COUNT,
        "family_count": _EXPECTED_FAMILY_COUNT,
        "sample_split_counts": EXPECTED_SOURCE_SAMPLE_COUNTS,
        "family_split_counts": EXPECTED_SOURCE_FAMILY_COUNTS,
        "observed_class_inventory": D12_CLASS_INVENTORY,
        "test_specialist_records": TEST_SPECIALIST_RECORDS,
        "optimizer_steps": 0,
        "complete_marker_present": False,
        "verification_passed": True,
    }
    for name, value in expected.items():
        if getattr(receipt, name) != value:
            _fail(f"accepted D12 verifier {name} mismatch")


def _d12_records(d12_root: Path) -> dict[str, Mapping[str, object]]:
    manifest, raw = _read_canonical_json(d12_root / "manifest.json", _MAX_MANIFEST_BYTES, "accepted D12 manifest")
    if sha256(raw).hexdigest() != EXPECTED_D12_MANIFEST_SHA256:
        _fail("accepted D12 manifest SHA-256 mismatch")
    if manifest.get("schema_version") != STAGE7D12_MANIFEST_SCHEMA or manifest.get("stage7d12_derivative_version") != STAGE7D12_DERIVATIVE_VERSION:
        _fail("accepted D12 manifest schema/version mismatch")
    rows = manifest.get("records")
    if not isinstance(rows, list) or len(rows) != _EXPECTED_SAMPLE_COUNT:
        _fail("accepted D12 record cardinality mismatch")
    index: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            _fail("accepted D12 record must be an object")
        split = row.get("split")
        if split == "test":
            _fail("sealed TEST record reached D13 verifier")
        if split not in _ALLOWED_SPLITS:
            _fail("accepted D12 record split invalid")
        sample_id = _hex64("D12 record.sample_id", row.get("sample_id"))
        if sample_id in index:
            _fail("duplicate accepted D12 sample_id")
        index[sample_id] = row
    return index


def _load_d12_label(d12_root: Path, record: Mapping[str, object]) -> tuple[dict[str, object], str]:
    label_sha = _hex64("D12 record.symbol_label_sha256", record.get("symbol_label_sha256"))
    label, raw = _read_canonical_json(d12_root / "labels" / f"{label_sha}.json", _MAX_LABEL_BYTES, "accepted D12 label")
    if sha256(raw).hexdigest() != label_sha:
        _fail("accepted D12 label SHA-256 mismatch")
    if label.get("schema_version") != STAGE7D12_LABEL_SCHEMA or label.get("stage7d12_derivative_version") != STAGE7D12_DERIVATIVE_VERSION:
        _fail("accepted D12 label schema/version mismatch")
    return label, label_sha


def _plan(measure_box: Mapping[str, float], *, image_width: int, image_height: int, source_png_sha: str) -> _Plan:
    box = _bbox("D12 measure_bbox", measure_box, width=image_width, height=image_height)
    left = max(0, int(math.floor(box["x_min"])))
    top = max(0, int(math.floor(box["y_min"])))
    right = min(image_width, int(math.ceil(box["x_max"])))
    bottom = min(image_height, int(math.ceil(box["y_max"])))
    if not 0 <= left < right <= image_width or not 0 <= top < bottom <= image_height:
        _fail("independent D13 integer crop invalid")
    crop_width = right - left
    crop_height = bottom - top
    scale = min(INPUT_WIDTH / crop_width, INPUT_HEIGHT / crop_height)
    pad_x = (INPUT_WIDTH - crop_width * scale) / 2.0
    pad_y = (INPUT_HEIGHT - crop_height * scale) / 2.0
    if not math.isfinite(scale) or scale <= 0 or pad_x < -_EPS or pad_y < -_EPS:
        _fail("independent D13 letterbox plan invalid")
    payload = {
        "version": STAGE7D13_TRANSFORM_VERSION,
        "source_png_sha256": source_png_sha,
        "source_image_size": [image_width, image_height],
        "measure_bbox": dict(box),
        "integer_crop": [left, top, right, bottom],
        "canvas": [INPUT_WIDTH, INPUT_HEIGHT],
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "resample": "pillow-bicubic-affine",
        "background": WHITE_BACKGROUND,
    }
    return _Plan(left, top, right, bottom, scale, pad_x, pad_y, sha256(_canonical_json(payload)).hexdigest())


def _transform_point(point: Mapping[str, float], plan: _Plan) -> dict[str, float]:
    result = {
        "x": (float(point["x"]) - plan.left) * plan.scale + plan.pad_x,
        "y": (float(point["y"]) - plan.top) * plan.scale + plan.pad_y,
    }
    if not all(math.isfinite(v) for v in result.values()):
        _fail("independent transformed point non-finite")
    return result


def _transform_box(box: Mapping[str, float], plan: _Plan) -> dict[str, float]:
    result = {
        "x_min": (float(box["x_min"]) - plan.left) * plan.scale + plan.pad_x,
        "y_min": (float(box["y_min"]) - plan.top) * plan.scale + plan.pad_y,
        "x_max": (float(box["x_max"]) - plan.left) * plan.scale + plan.pad_x,
        "y_max": (float(box["y_max"]) - plan.top) * plan.scale + plan.pad_y,
    }
    if not all(math.isfinite(v) for v in result.values()) or not result["x_min"] < result["x_max"] or not result["y_min"] < result["y_max"]:
        _fail("independent transformed bbox invalid")
    if result["x_min"] < -_EPS or result["y_min"] < -_EPS or result["x_max"] > INPUT_WIDTH + _EPS or result["y_max"] > INPUT_HEIGHT + _EPS:
        _fail("independent transformed bbox leaves canvas")
    return result


def _render(source: Image.Image, plan: _Plan) -> bytes:
    crop = source.crop((plan.left, plan.top, plan.right, plan.bottom))
    inv = 1.0 / plan.scale
    rendered = crop.transform(
        (INPUT_WIDTH, INPUT_HEIGHT),
        Image.Transform.AFFINE,
        (inv, 0.0, -plan.pad_x * inv, 0.0, inv, -plan.pad_y * inv),
        resample=Image.Resampling.BICUBIC,
        fillcolor=WHITE_BACKGROUND,
    )
    if rendered.mode != "L" or rendered.size != (INPUT_WIDTH, INPUT_HEIGHT):
        _fail("independent rendered D13 measure image mismatch")
    out = BytesIO()
    rendered.save(out, format="PNG", optimize=False, compress_level=9)
    return out.getvalue()


def _expected_target(kind: str, source_row: Mapping[str, object], *, measure_box: Mapping[str, float], width: int, height: int, plan: _Plan) -> dict[str, object]:
    if kind == "notehead":
        box = _bbox("notehead_bbox", source_row.get("notehead_bbox"), width=width, height=height)
        center = _point("notehead_center", source_row.get("notehead_center"), width=width, height=height)
        cls = source_row.get("fill_class")
    elif kind == "rest":
        box = _bbox("rest_bbox", source_row.get("rest_bbox"), width=width, height=height)
        center = {"x": (box["x_min"] + box["x_max"]) / 2.0, "y": (box["y_min"] + box["y_max"]) / 2.0}
        cls = source_row.get("rest_class")
    elif kind == "accidental":
        box = _bbox("accidental_bbox", source_row.get("accidental_bbox"), width=width, height=height)
        center = {"x": (box["x_min"] + box["x_max"]) / 2.0, "y": (box["y_min"] + box["y_max"]) / 2.0}
        cls = source_row.get("accidental_class")
    else:
        _fail("unknown D13 target kind")
    if cls not in SPECIALIST_CLASSES[kind]:
        _fail("D12 class outside D13 specialist classes")
    if not _box_inside(box, measure_box) or not _point_inside(center, box):
        _fail("D12 target geometry invalid for independent D13 mapping")
    transformed_box = _transform_box(box, plan)
    transformed_center = _transform_point(center, plan)
    transformed_measure = _transform_box(measure_box, plan)
    if not _box_inside(transformed_box, transformed_measure) or not _point_inside(transformed_center, transformed_box):
        _fail("independently transformed target leaves measure/box")
    return {
        "canonical_event_id": _identifier(f"{kind}.canonical_event_id", source_row.get("canonical_event_id")),
        "renderer_id": _identifier(f"{kind}.renderer_id", source_row.get("renderer_id")),
        "class": cls,
        "bbox": transformed_box,
        "center": transformed_center,
    }


def _find_measure(label: Mapping[str, object], measure_number: int, renderer_measure_id: str) -> Mapping[str, object]:
    geometry = label.get("symbol_geometry")
    if not isinstance(geometry, Mapping) or geometry.get("coordinate_space") != "final_png_pixels":
        _fail("accepted D12 symbol geometry invalid")
    measures = geometry.get("measures")
    if not isinstance(measures, list):
        _fail("accepted D12 measures must be a list")
    matches = [
        row for row in measures
        if isinstance(row, Mapping)
        and row.get("measure_number") == measure_number
        and row.get("renderer_measure_id") == renderer_measure_id
    ]
    if len(matches) != 1:
        _fail("D13 record must resolve exactly one accepted D12 measure")
    return matches[0]


def verify_stage7d13_measure_derivatives(
    *,
    corpus_root: str | Path,
    d6_root: str | Path,
    d12_root: str | Path,
    derivative_root: str | Path,
) -> Stage7D13VerificationReceipt:
    """Independently verify a persisted, still-uncompleted D13 derivative root."""
    if not all(isinstance(value, (str, Path)) for value in (corpus_root, d6_root, d12_root, derivative_root)):
        raise TypeError("corpus_root, d6_root, d12_root and derivative_root must be str or pathlib.Path")
    corpus = Path(corpus_root)
    d6 = Path(d6_root)
    d12 = Path(d12_root)
    root = Path(derivative_root)
    for path, name in ((corpus, "source corpus root"), (d6, "accepted D6 root"), (d12, "accepted D12 root"), (root, "D13 derivative root")):
        _regular_directory(path, name)
    top = {entry.name for entry in root.iterdir()}
    if top != _TOP_LEVEL:
        _fail("D13 derivative top-level layout mismatch or premature COMPLETE")
    _regular_directory(root / "images", "D13 images directory")
    _regular_directory(root / "labels", "D13 labels directory")
    if (root / "COMPLETE").exists() or (root / "COMPLETE").is_symlink():
        _fail("premature D13 COMPLETE marker is forbidden")

    _accepted_d12(corpus, d6, d12)
    d12_index = _d12_records(d12)

    manifest, manifest_raw = _read_canonical_json(root / "manifest.json", _MAX_MANIFEST_BYTES, "D13 manifest")
    manifest_sha = sha256(manifest_raw).hexdigest()
    checksum_raw = _read_bounded(root / "manifest.sha256", 256, "D13 manifest.sha256")
    if checksum_raw != f"{manifest_sha}  manifest.json\n".encode("ascii"):
        _fail("D13 manifest checksum sidecar mismatch")
    if manifest.get("schema_version") != STAGE7D13_MANIFEST_SCHEMA or manifest.get("stage7d13_derivative_version") != STAGE7D13_DERIVATIVE_VERSION:
        _fail("D13 manifest schema/version mismatch")
    derivative_build_id = _hex64("D13 manifest.derivative_build_id", manifest.get("derivative_build_id"))
    if derivative_build_id != stage7d13_derivative_profile_fingerprint():
        _fail("D13 derivative build id/profile mismatch")
    if manifest.get("contract_fingerprint") != stage7d13_contract_fingerprint():
        _fail("D13 contract fingerprint mismatch")
    expected_d12 = {
        "derivative_build_id": EXPECTED_D12_DERIVATIVE_BUILD_ID,
        "manifest_sha256": EXPECTED_D12_MANIFEST_SHA256,
        "artifact_binding_sha256": EXPECTED_D12_ARTIFACT_BINDING_SHA256,
    }
    if manifest.get("accepted_d12") != expected_d12:
        _fail("D13 manifest accepted D12 identity mismatch")
    if manifest.get("split_policy") != "family-exclusive-train-validation-only-test-forbidden" or manifest.get("completion_policy") != "independent-verifier-required-before-COMPLETE":
        _fail("D13 manifest policy mismatch")
    rows = manifest.get("records")
    if not isinstance(rows, list) or not rows:
        _fail("D13 manifest records must be a non-empty list")

    inventory = {
        split: {kind: {name: 0 for name in classes} for kind, classes in SPECIALIST_CLASSES.items()}
        for split in ("train", "validation")
    }
    record_ids: set[str] = set()
    referenced_images: set[str] = set()
    referenced_labels: set[str] = set()
    source_samples_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    families_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    binding_rows: list[str] = []
    record_split_counter: Counter[str] = Counter()
    label_bytes_total = 0

    for record in rows:
        if not isinstance(record, Mapping):
            _fail("D13 manifest record must be an object")
        expected_keys = {
            "record_id", "sample_id", "family_id", "split", "page_number",
            "measure_number", "renderer_measure_id", "source_png_sha256",
            "d12_label_sha256", "transform_fingerprint", "image_sha256", "label_sha256",
        }
        if set(record) != expected_keys:
            _fail("D13 manifest record key set mismatch")
        split = record.get("split")
        if split == "test":
            _fail("sealed TEST record reached D13 persisted verifier")
        if split not in _ALLOWED_SPLITS:
            _fail("D13 manifest record split invalid")
        assert isinstance(split, str)
        record_id = _hex64("D13 record.record_id", record.get("record_id"))
        sample_id = _hex64("D13 record.sample_id", record.get("sample_id"))
        family_id = _identifier("D13 record.family_id", record.get("family_id"))
        page_number = _positive_int("D13 record.page_number", record.get("page_number"))
        measure_number = _positive_int("D13 record.measure_number", record.get("measure_number"))
        renderer_measure_id = _identifier("D13 record.renderer_measure_id", record.get("renderer_measure_id"))
        source_png_sha = _hex64("D13 record.source_png_sha256", record.get("source_png_sha256"))
        d12_label_sha = _hex64("D13 record.d12_label_sha256", record.get("d12_label_sha256"))
        transform_fp = _hex64("D13 record.transform_fingerprint", record.get("transform_fingerprint"))
        image_sha = _hex64("D13 record.image_sha256", record.get("image_sha256"))
        label_sha = _hex64("D13 record.label_sha256", record.get("label_sha256"))
        if record_id in record_ids or label_sha in referenced_labels:
            _fail("duplicate D13 record or label identity")
        record_ids.add(record_id)
        referenced_labels.add(label_sha)
        referenced_images.add(image_sha)
        record_split_counter[split] += 1
        source_samples_by_split[split].add(sample_id)
        families_by_split[split].add(family_id)

        d12_record = d12_index.get(sample_id)
        if d12_record is None:
            _fail("D13 record references unknown accepted D12 sample")
        if d12_record.get("family_id") != family_id or d12_record.get("split") != split or d12_record.get("page_number") != page_number or d12_record.get("png_sha256") != source_png_sha or d12_record.get("symbol_label_sha256") != d12_label_sha:
            _fail("D13 manifest record disagrees with accepted D12 record")
        d12_label, observed_d12_label_sha = _load_d12_label(d12, d12_record)
        if observed_d12_label_sha != d12_label_sha:
            _fail("D13 record D12 label SHA mismatch")
        d12_image = d12_label.get("image")
        if not isinstance(d12_image, Mapping):
            _fail("accepted D12 image metadata invalid")
        width = _positive_int("D12 image.width", d12_image.get("width"))
        height = _positive_int("D12 image.height", d12_image.get("height"))
        if d12_image.get("png_sha256") != source_png_sha or d12_image.get("mode") != "L" or d12_image.get("image_format") != "png":
            _fail("accepted D12 source image identity/format mismatch")
        source_raw = _read_bounded(corpus / "images" / f"{source_png_sha}.png", _MAX_IMAGE_BYTES, "frozen source PNG")
        if sha256(source_raw).hexdigest() != source_png_sha:
            _fail("frozen source PNG SHA-256 mismatch")
        try:
            with Image.open(BytesIO(source_raw)) as source_open:
                source_open.load()
                if source_open.format != "PNG" or source_open.mode != "L" or source_open.size != (width, height):
                    _fail("frozen source PNG metadata mismatch")
                source_image = source_open.copy()
        except (OSError, ValueError) as exc:
            raise Stage7D13VerificationError("frozen source PNG cannot be decoded") from exc

        source_measure = _find_measure(d12_label, measure_number, renderer_measure_id)
        measure_box = _bbox("accepted D12 measure_bbox", source_measure.get("measure_bbox"), width=width, height=height)
        plan = _plan(measure_box, image_width=width, image_height=height, source_png_sha=source_png_sha)
        if plan.fingerprint != transform_fp:
            _fail("D13 manifest transform fingerprint mismatch")

        derivative_image_raw = _read_bounded(root / "images" / f"{image_sha}.png", _MAX_IMAGE_BYTES, "D13 derivative image")
        if sha256(derivative_image_raw).hexdigest() != image_sha:
            _fail("D13 derivative image SHA-256 mismatch")
        expected_image_raw = _render(source_image, plan)
        if derivative_image_raw != expected_image_raw:
            _fail("D13 derivative image bytes differ from independent rendering")
        try:
            with Image.open(BytesIO(derivative_image_raw)) as image_open:
                image_open.load()
                if image_open.format != "PNG" or image_open.mode != "L" or image_open.size != (INPUT_WIDTH, INPUT_HEIGHT):
                    _fail("D13 derivative image metadata mismatch")
        except (OSError, ValueError) as exc:
            raise Stage7D13VerificationError("D13 derivative image cannot be decoded") from exc

        label, label_raw = _read_canonical_json(root / "labels" / f"{label_sha}.json", _MAX_LABEL_BYTES, "D13 measure label")
        if sha256(label_raw).hexdigest() != label_sha:
            _fail("D13 label SHA-256 mismatch")
        label_bytes_total += len(label_raw)
        if label.get("schema_version") != STAGE7D13_LABEL_SCHEMA or label.get("stage7d13_derivative_version") != STAGE7D13_DERIVATIVE_VERSION or label.get("contract_fingerprint") != stage7d13_contract_fingerprint():
            _fail("D13 label schema/version/contract mismatch")
        for key, value in (
            ("record_id", record_id), ("sample_id", sample_id), ("family_id", family_id),
            ("split", split), ("page_number", page_number), ("measure_number", measure_number),
            ("renderer_measure_id", renderer_measure_id),
        ):
            if label.get(key) != value:
                _fail(f"D13 label/manifest {key} mismatch")
        source_meta = label.get("source")
        expected_source_meta = {
            "d12_label_sha256": d12_label_sha,
            "source_png_sha256": source_png_sha,
            "source_image_size": [width, height],
            "accepted_d12_manifest_sha256": EXPECTED_D12_MANIFEST_SHA256,
            "accepted_d12_artifact_binding_sha256": EXPECTED_D12_ARTIFACT_BINDING_SHA256,
        }
        if source_meta != expected_source_meta:
            _fail("D13 label source metadata mismatch")
        image_meta = label.get("image")
        if image_meta != {"png_sha256": image_sha, "width": INPUT_WIDTH, "height": INPUT_HEIGHT, "mode": "L", "image_format": "png"}:
            _fail("D13 label derivative image metadata mismatch")
        expected_transform = {
            "version": STAGE7D13_TRANSFORM_VERSION,
            "source_measure_bbox": dict(measure_box),
            "integer_crop": [plan.left, plan.top, plan.right, plan.bottom],
            "scale": plan.scale,
            "pad_x": plan.pad_x,
            "pad_y": plan.pad_y,
            "canvas": [INPUT_WIDTH, INPUT_HEIGHT],
            "resample": "pillow-bicubic-affine",
            "background": WHITE_BACKGROUND,
            "fingerprint": plan.fingerprint,
            "transformed_measure_bbox": _transform_box(measure_box, plan),
        }
        if label.get("transform") != expected_transform:
            _fail("D13 label transform differs from independent recomputation")

        identity_payload = {
            "version": STAGE7D13_DERIVATIVE_VERSION,
            "sample_id": sample_id,
            "family_id": family_id,
            "split": split,
            "page_number": page_number,
            "measure_number": measure_number,
            "renderer_measure_id": renderer_measure_id,
            "d12_label_sha256": d12_label_sha,
            "source_png_sha256": source_png_sha,
            "transform_fingerprint": plan.fingerprint,
        }
        if sha256(_canonical_json(identity_payload)).hexdigest() != record_id:
            _fail("D13 record_id identity recomputation mismatch")

        targets = label.get("targets")
        if not isinstance(targets, Mapping) or set(targets) != set(SPECIALIST_CLASSES):
            _fail("D13 label targets key set mismatch")
        for kind, d12_key in (("notehead", "noteheads"), ("rest", "rests"), ("accidental", "accidentals")):
            source_targets = source_measure.get(d12_key)
            persisted_targets = targets.get(kind)
            if not isinstance(source_targets, list) or not isinstance(persisted_targets, list) or len(source_targets) != len(persisted_targets):
                _fail("D13 persisted target cardinality differs from accepted D12")
            expected_targets = []
            for source_target in source_targets:
                if not isinstance(source_target, Mapping):
                    _fail("accepted D12 target must be an object")
                expected_targets.append(_expected_target(kind, source_target, measure_box=measure_box, width=width, height=height, plan=plan))
            if persisted_targets != expected_targets:
                _fail("D13 persisted target geometry/class differs from independent mapping")
            for target in persisted_targets:
                assert isinstance(target, Mapping)
                cls = target.get("class")
                if cls not in SPECIALIST_CLASSES[kind]:
                    _fail("D13 persisted target class invalid")
                assert isinstance(cls, str)
                inventory[split][kind][cls] += 1

        binding_rows.append(
            f"{record_id}:{sample_id}:{source_png_sha}:{d12_label_sha}:{plan.fingerprint}:"
            f"{image_sha}:{label_sha}:{len(derivative_image_raw)}:{len(label_raw)}"
        )

    if inventory != D12_CLASS_INVENTORY:
        _fail("independently verified D13 class inventory differs from accepted D12")
    source_sample_split_counts = {split: len(values) for split, values in source_samples_by_split.items()}
    family_split_counts = {split: len(values) for split, values in families_by_split.items()}
    if source_sample_split_counts != EXPECTED_SOURCE_SAMPLE_COUNTS:
        _fail("D13 independently verified source sample counts mismatch")
    if family_split_counts != EXPECTED_SOURCE_FAMILY_COUNTS:
        _fail("D13 independently verified family counts mismatch")
    if families_by_split["train"] & families_by_split["validation"]:
        _fail("D13 family crosses TRAIN/VALIDATION")
    record_split_counts = dict(sorted(record_split_counter.items()))
    target_instance_counts = {
        split: {kind: sum(inventory[split][kind].values()) for kind in SPECIALIST_CLASSES}
        for split in ("train", "validation")
    }

    image_files = {path.stem for path in (root / "images").iterdir() if path.is_file() and not path.is_symlink() and path.suffix == ".png"}
    label_files = {path.stem for path in (root / "labels").iterdir() if path.is_file() and not path.is_symlink() and path.suffix == ".json"}
    if image_files != referenced_images:
        _fail("D13 images directory does not exactly match manifest references")
    if label_files != referenced_labels:
        _fail("D13 labels directory does not exactly match manifest references")
    image_bytes_total = sum((root / "images" / f"{value}.png").stat().st_size for value in referenced_images)
    artifact_binding = sha256(("\n".join(sorted(binding_rows)) + "\n").encode("ascii")).hexdigest()

    build, build_raw = _read_canonical_json(root / "build.json", _MAX_BUILD_BYTES, "D13 build.json")
    expected_build = {
        "schema_version": STAGE7D13_BUILD_SCHEMA,
        "stage7d13_derivative_version": STAGE7D13_DERIVATIVE_VERSION,
        "derivative_build_id": derivative_build_id,
        "contract_fingerprint": stage7d13_contract_fingerprint(),
        "manifest_sha256": manifest_sha,
        "artifact_binding_sha256": artifact_binding,
        "record_count": len(rows),
        "image_count": len(referenced_images),
        "label_count": len(referenced_labels),
        "source_sample_count": sum(source_sample_split_counts.values()),
        "family_count": len(families_by_split["train"] | families_by_split["validation"]),
        "record_split_counts": record_split_counts,
        "source_sample_split_counts": source_sample_split_counts,
        "family_split_counts": family_split_counts,
        "observed_class_inventory": inventory,
        "target_instance_counts": target_instance_counts,
        "image_bytes_total": image_bytes_total,
        "label_bytes_total": label_bytes_total,
        "test_specialist_records": TEST_SPECIALIST_RECORDS,
        "optimizer_steps": 0,
        "complete_marker_written": False,
        "completion_policy": "independent-verifier-required-before-COMPLETE",
        "exact_optimizer_steps": "freeze-after-independent-D13-2-verification",
        "layout": {"manifest": "manifest.json", "images": "images/<image_sha256>.png", "labels": "labels/<label_sha256>.json"},
    }
    if build != expected_build or _canonical_json(expected_build) != build_raw:
        _fail("D13 build.json differs from independently recomputed evidence")

    return Stage7D13VerificationReceipt(
        verifier_version=STAGE7D13_VERIFIER_VERSION,
        derivative_build_id=derivative_build_id,
        manifest_sha256=manifest_sha,
        artifact_binding_sha256=artifact_binding,
        record_count=len(rows),
        image_count=len(referenced_images),
        label_count=len(referenced_labels),
        source_sample_count=sum(source_sample_split_counts.values()),
        family_count=len(families_by_split["train"] | families_by_split["validation"]),
        record_split_counts=record_split_counts,
        source_sample_split_counts=source_sample_split_counts,
        family_split_counts=family_split_counts,
        observed_class_inventory=inventory,
        target_instance_counts=target_instance_counts,
        image_bytes_total=image_bytes_total,
        label_bytes_total=label_bytes_total,
        test_specialist_records=TEST_SPECIALIST_RECORDS,
        optimizer_steps=0,
        complete_marker_present=False,
        verification_passed=True,
    )
