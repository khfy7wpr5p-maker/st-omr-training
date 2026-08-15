"""Stage 7-D13 deterministic measure derivative builder.

D13-1 consumes only the independently verified Stage 7-D12 TRAIN/VALIDATION
symbol bundle and exact frozen source PNGs.  Each accepted D12 measure becomes
one 512x128 grayscale letterboxed measure record with the NoteHeadSet, RestSet,
and AccidentalSet targets transformed by the exact same isotropic scale/pad
mapping.  TEST remains forbidden and no optimizer/model/checkpoint is touched.

The builder deliberately does not write COMPLETE.  D13-2 independently reopens
and verifies the persisted derivative bundle before exact record/optimizer-step
counts may be frozen.
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

from .stage7d12_symbol_derivative_verifier import (
    Stage7D12VerificationReceipt,
    verify_stage7d12_symbol_derivatives,
)
from .stage7d12_symbol_derivatives import (
    STAGE7D12_DERIVATIVE_VERSION,
    STAGE7D12_LABEL_SCHEMA,
    STAGE7D12_MANIFEST_SCHEMA,
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
    class_readiness_violations,
    stage7d13_contract_fingerprint,
)


STAGE7D13_DERIVATIVE_VERSION: Final[str] = "stage7d13-measure-derivatives-v1"
STAGE7D13_LABEL_SCHEMA: Final[str] = "stage7d13-measure-label-v1"
STAGE7D13_MANIFEST_SCHEMA: Final[str] = "stage7d13-measure-manifest-v1"
STAGE7D13_BUILD_SCHEMA: Final[str] = "stage7d13-measure-build-v1"
STAGE7D13_TRANSFORM_VERSION: Final[str] = "stage7d13-measure-letterbox-v1"

_ALLOWED_SPLITS: Final[frozenset[str]] = frozenset({"train", "validation"})
_HEX64: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_SAMPLE_COUNT: Final[int] = sum(EXPECTED_SOURCE_SAMPLE_COUNTS.values())
_EXPECTED_FAMILY_COUNT: Final[int] = sum(EXPECTED_SOURCE_FAMILY_COUNTS.values())
_MAX_MANIFEST_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_BUILD_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_LABEL_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_IMAGE_BYTES: Final[int] = 32 * 1024 * 1024
_EPS: Final[float] = 1e-5


class Stage7D13DerivativeError(RuntimeError):
    """Raised when the D13 measure-derivative boundary cannot be proven."""


def _fail(message: str) -> None:
    raise Stage7D13DerivativeError(message)


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
        raise Stage7D13DerivativeError("payload is not canonical JSON serializable") from exc


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
        _fail(f"{name} byte length is outside the D13 bound")
    return path.read_bytes()


def _read_canonical_json(path: Path, maximum: int, name: str) -> tuple[dict[str, object], bytes]:
    raw = _read_bounded(path, maximum, name)
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda token: _fail(f"non-finite JSON constant in {name}: {token}"),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D13DerivativeError(f"{name} is not valid ASCII JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != raw:
        _fail(f"{name} must be canonical JSON object bytes")
    return value, raw


def development_d12_records(rows: object) -> tuple[Mapping[str, object], ...]:
    """Accept only TRAIN/VALIDATION; TEST is rejected after reading only split."""
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        _fail("D12 records must be a sequence")
    accepted: list[Mapping[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"D12 record[{index}] must be an object")
        split = row.get("split")
        if split == "test":
            _fail("sealed TEST record reached Stage 7-D13")
        if split not in _ALLOWED_SPLITS:
            _fail("D13 record split must be train or validation")
        accepted.append(row)
    return tuple(accepted)


@dataclass(frozen=True, slots=True)
class LetterboxPlan:
    crop_left: int
    crop_top: int
    crop_right: int
    crop_bottom: int
    scale: float
    pad_x: float
    pad_y: float
    transform_fingerprint: str

    @property
    def crop_width(self) -> int:
        return self.crop_right - self.crop_left

    @property
    def crop_height(self) -> int:
        return self.crop_bottom - self.crop_top


@dataclass(frozen=True, slots=True)
class Stage7D13DerivativeReceipt:
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
    complete_marker_written: bool


def _accepted_d12_receipt(
    corpus_root: Path,
    d6_root: Path,
    d12_root: Path,
) -> Stage7D12VerificationReceipt:
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
            _fail(f"accepted D12 receipt {name} mismatch")
    return receipt


def _d12_index(d12_root: Path) -> tuple[dict[str, Mapping[str, object]], tuple[Mapping[str, object], ...]]:
    manifest, raw = _read_canonical_json(
        d12_root / "manifest.json",
        _MAX_MANIFEST_BYTES,
        "accepted D12 manifest.json",
    )
    if sha256(raw).hexdigest() != EXPECTED_D12_MANIFEST_SHA256:
        _fail("accepted D12 manifest SHA-256 mismatch")
    if manifest.get("schema_version") != STAGE7D12_MANIFEST_SCHEMA:
        _fail("accepted D12 manifest schema mismatch")
    if manifest.get("stage7d12_derivative_version") != STAGE7D12_DERIVATIVE_VERSION:
        _fail("accepted D12 derivative version mismatch")
    rows = development_d12_records(manifest.get("records"))
    if len(rows) != _EXPECTED_SAMPLE_COUNT:
        _fail("accepted D12 sample cardinality mismatch")

    index: dict[str, Mapping[str, object]] = {}
    counts: Counter[str] = Counter()
    family_split: dict[str, str] = {}
    for row in rows:
        sample_id = _hex64("D12 record.sample_id", row.get("sample_id"))
        if sample_id in index:
            _fail("duplicate accepted D12 sample_id")
        family_id = _identifier("D12 record.family_id", row.get("family_id"))
        split = row.get("split")
        assert isinstance(split, str)
        prior = family_split.setdefault(family_id, split)
        if prior != split:
            _fail("accepted D12 family crosses TRAIN/VALIDATION")
        counts[split] += 1
        index[sample_id] = row
    if dict(sorted(counts.items())) != EXPECTED_SOURCE_SAMPLE_COUNTS:
        _fail("accepted D12 sample split counts mismatch")
    family_counts = dict(sorted(Counter(family_split.values()).items()))
    if family_counts != EXPECTED_SOURCE_FAMILY_COUNTS:
        _fail("accepted D12 family split counts mismatch")
    return index, rows


def _load_d12_label(d12_root: Path, record: Mapping[str, object]) -> tuple[dict[str, object], str]:
    label_sha = _hex64("D12 record.symbol_label_sha256", record.get("symbol_label_sha256"))
    label, raw = _read_canonical_json(
        d12_root / "labels" / f"{label_sha}.json",
        _MAX_LABEL_BYTES,
        "accepted D12 symbol label",
    )
    if sha256(raw).hexdigest() != label_sha:
        _fail("accepted D12 label SHA-256 mismatch")
    if label.get("schema_version") != STAGE7D12_LABEL_SCHEMA:
        _fail("accepted D12 label schema mismatch")
    if label.get("stage7d12_derivative_version") != STAGE7D12_DERIVATIVE_VERSION:
        _fail("accepted D12 label version mismatch")
    for key in ("sample_id", "family_id", "split", "page_number"):
        if label.get(key) != record.get(key):
            _fail(f"accepted D12 label/manifest {key} mismatch")
    return label, label_sha


def _bbox(name: str, value: object, *, width: int, height: int) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {"x_min", "y_min", "x_max", "y_max"}:
        _fail(f"{name} must be a canonical bbox")
    box = {key: _finite(f"{name}.{key}", value.get(key)) for key in ("x_min", "y_min", "x_max", "y_max")}
    if not box["x_min"] < box["x_max"] or not box["y_min"] < box["y_max"]:
        _fail(f"{name} must have positive area")
    if box["x_min"] < -_EPS or box["y_min"] < -_EPS or box["x_max"] > width + _EPS or box["y_max"] > height + _EPS:
        _fail(f"{name} leaves source image bounds")
    return box


def _point(name: str, value: object, *, width: int, height: int) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {"x", "y"}:
        _fail(f"{name} must be a canonical point")
    point = {"x": _finite(f"{name}.x", value.get("x")), "y": _finite(f"{name}.y", value.get("y"))}
    if point["x"] < -_EPS or point["y"] < -_EPS or point["x"] > width + _EPS or point["y"] > height + _EPS:
        _fail(f"{name} leaves source image bounds")
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


def make_letterbox_plan(
    measure_bbox: Mapping[str, float],
    *,
    image_width: int,
    image_height: int,
    source_png_sha256: str,
) -> LetterboxPlan:
    """Freeze outward integer crop + isotropic scale + centered white padding."""
    box = _bbox("measure_bbox", measure_bbox, width=image_width, height=image_height)
    left = max(0, int(math.floor(box["x_min"])))
    top = max(0, int(math.floor(box["y_min"])))
    right = min(image_width, int(math.ceil(box["x_max"])))
    bottom = min(image_height, int(math.ceil(box["y_max"])))
    if not 0 <= left < right <= image_width or not 0 <= top < bottom <= image_height:
        _fail("D13 measure integer crop is invalid")
    crop_width = right - left
    crop_height = bottom - top
    scale = min(INPUT_WIDTH / crop_width, INPUT_HEIGHT / crop_height)
    if not math.isfinite(scale) or scale <= 0:
        _fail("D13 letterbox scale is invalid")
    pad_x = (INPUT_WIDTH - crop_width * scale) / 2.0
    pad_y = (INPUT_HEIGHT - crop_height * scale) / 2.0
    if pad_x < -_EPS or pad_y < -_EPS:
        _fail("D13 letterbox padding became negative")
    payload = {
        "version": STAGE7D13_TRANSFORM_VERSION,
        "source_png_sha256": _hex64("source_png_sha256", source_png_sha256),
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
    return LetterboxPlan(
        crop_left=left,
        crop_top=top,
        crop_right=right,
        crop_bottom=bottom,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        transform_fingerprint=sha256(_canonical_json(payload)).hexdigest(),
    )


def _transform_point(point: Mapping[str, float], plan: LetterboxPlan) -> dict[str, float]:
    x = (float(point["x"]) - plan.crop_left) * plan.scale + plan.pad_x
    y = (float(point["y"]) - plan.crop_top) * plan.scale + plan.pad_y
    if not all(math.isfinite(value) for value in (x, y)):
        _fail("transformed D13 point is non-finite")
    return {"x": x, "y": y}


def _transform_box(box: Mapping[str, float], plan: LetterboxPlan) -> dict[str, float]:
    transformed = {
        "x_min": (float(box["x_min"]) - plan.crop_left) * plan.scale + plan.pad_x,
        "y_min": (float(box["y_min"]) - plan.crop_top) * plan.scale + plan.pad_y,
        "x_max": (float(box["x_max"]) - plan.crop_left) * plan.scale + plan.pad_x,
        "y_max": (float(box["y_max"]) - plan.crop_top) * plan.scale + plan.pad_y,
    }
    if not all(math.isfinite(value) for value in transformed.values()):
        _fail("transformed D13 bbox is non-finite")
    if not transformed["x_min"] < transformed["x_max"] or not transformed["y_min"] < transformed["y_max"]:
        _fail("transformed D13 bbox has non-positive area")
    if transformed["x_min"] < -_EPS or transformed["y_min"] < -_EPS or transformed["x_max"] > INPUT_WIDTH + _EPS or transformed["y_max"] > INPUT_HEIGHT + _EPS:
        _fail("transformed D13 bbox leaves fixed canvas")
    return transformed


def _render_measure_image(source: Image.Image, plan: LetterboxPlan) -> bytes:
    crop = source.crop((plan.crop_left, plan.crop_top, plan.crop_right, plan.crop_bottom))
    inv = 1.0 / plan.scale
    affine = (
        inv,
        0.0,
        -plan.pad_x * inv,
        0.0,
        inv,
        -plan.pad_y * inv,
    )
    rendered = crop.transform(
        (INPUT_WIDTH, INPUT_HEIGHT),
        Image.Transform.AFFINE,
        affine,
        resample=Image.Resampling.BICUBIC,
        fillcolor=WHITE_BACKGROUND,
    )
    if rendered.mode != "L" or rendered.size != (INPUT_WIDTH, INPUT_HEIGHT):
        _fail("D13 rendered measure image contract mismatch")
    buffer = BytesIO()
    rendered.save(buffer, format="PNG", optimize=False, compress_level=9)
    raw = buffer.getvalue()
    if not 1 <= len(raw) <= _MAX_IMAGE_BYTES:
        _fail("D13 rendered measure PNG byte length is outside bound")
    return raw


def _target(
    kind: str,
    row: Mapping[str, object],
    *,
    measure_box: Mapping[str, float],
    image_width: int,
    image_height: int,
    plan: LetterboxPlan,
) -> dict[str, object]:
    if kind == "notehead":
        source_box = _bbox("notehead_bbox", row.get("notehead_bbox"), width=image_width, height=image_height)
        source_center = _point("notehead_center", row.get("notehead_center"), width=image_width, height=image_height)
        class_name = row.get("fill_class")
    elif kind == "rest":
        source_box = _bbox("rest_bbox", row.get("rest_bbox"), width=image_width, height=image_height)
        source_center = {
            "x": (source_box["x_min"] + source_box["x_max"]) / 2.0,
            "y": (source_box["y_min"] + source_box["y_max"]) / 2.0,
        }
        class_name = row.get("rest_class")
    elif kind == "accidental":
        source_box = _bbox("accidental_bbox", row.get("accidental_bbox"), width=image_width, height=image_height)
        source_center = {
            "x": (source_box["x_min"] + source_box["x_max"]) / 2.0,
            "y": (source_box["y_min"] + source_box["y_max"]) / 2.0,
        }
        class_name = row.get("accidental_class")
    else:
        _fail("unknown D13 target kind")
    if class_name not in SPECIALIST_CLASSES[kind]:
        _fail(f"{kind} class is outside frozen D13 classes")
    if not _box_inside(source_box, measure_box) or not _point_inside(source_center, source_box):
        _fail(f"{kind} source target leaves accepted D12 measure/box")
    target_box = _transform_box(source_box, plan)
    target_center = _transform_point(source_center, plan)
    transformed_measure = _transform_box(measure_box, plan)
    if not _box_inside(target_box, transformed_measure) or not _point_inside(target_center, target_box):
        _fail(f"{kind} transformed target leaves transformed measure/box")
    return {
        "canonical_event_id": _identifier(f"{kind}.canonical_event_id", row.get("canonical_event_id")),
        "renderer_id": _identifier(f"{kind}.renderer_id", row.get("renderer_id")),
        "class": class_name,
        "bbox": target_box,
        "center": target_center,
    }


def _measure_payload(
    *,
    d12_label: Mapping[str, object],
    d12_label_sha: str,
    measure: Mapping[str, object],
    source_png_sha: str,
    derivative_png_sha: str,
    plan: LetterboxPlan,
    image_width: int,
    image_height: int,
) -> dict[str, object]:
    measure_number = _positive_int("measure_number", measure.get("measure_number"))
    renderer_measure_id = _identifier("renderer_measure_id", measure.get("renderer_measure_id"))
    measure_box = _bbox("measure_bbox", measure.get("measure_bbox"), width=image_width, height=image_height)
    transformed_measure = _transform_box(measure_box, plan)
    targets: dict[str, list[dict[str, object]]] = {}
    for kind, source_key in (
        ("notehead", "noteheads"),
        ("rest", "rests"),
        ("accidental", "accidentals"),
    ):
        rows = measure.get(source_key)
        if not isinstance(rows, list):
            _fail(f"D12 measure {source_key} must be a list")
        targets[kind] = [
            _target(
                kind,
                row,
                measure_box=measure_box,
                image_width=image_width,
                image_height=image_height,
                plan=plan,
            )
            for row in rows
            if isinstance(row, Mapping)
        ]
        if len(targets[kind]) != len(rows):
            _fail(f"D12 measure {source_key} contains non-object target")

    sample_id = _hex64("D12 label.sample_id", d12_label.get("sample_id"))
    family_id = _identifier("D12 label.family_id", d12_label.get("family_id"))
    split = d12_label.get("split")
    if split not in _ALLOWED_SPLITS:
        _fail("D13 measure split must be TRAIN or VALIDATION")
    page_number = _positive_int("D12 label.page_number", d12_label.get("page_number"))
    record_identity = {
        "version": STAGE7D13_DERIVATIVE_VERSION,
        "sample_id": sample_id,
        "family_id": family_id,
        "split": split,
        "page_number": page_number,
        "measure_number": measure_number,
        "renderer_measure_id": renderer_measure_id,
        "d12_label_sha256": d12_label_sha,
        "source_png_sha256": source_png_sha,
        "transform_fingerprint": plan.transform_fingerprint,
    }
    record_id = sha256(_canonical_json(record_identity)).hexdigest()
    return {
        "schema_version": STAGE7D13_LABEL_SCHEMA,
        "stage7d13_derivative_version": STAGE7D13_DERIVATIVE_VERSION,
        "contract_fingerprint": stage7d13_contract_fingerprint(),
        "record_id": record_id,
        "sample_id": sample_id,
        "family_id": family_id,
        "split": split,
        "page_number": page_number,
        "measure_number": measure_number,
        "renderer_measure_id": renderer_measure_id,
        "source": {
            "d12_label_sha256": d12_label_sha,
            "source_png_sha256": source_png_sha,
            "source_image_size": [image_width, image_height],
            "accepted_d12_manifest_sha256": EXPECTED_D12_MANIFEST_SHA256,
            "accepted_d12_artifact_binding_sha256": EXPECTED_D12_ARTIFACT_BINDING_SHA256,
        },
        "image": {
            "png_sha256": derivative_png_sha,
            "width": INPUT_WIDTH,
            "height": INPUT_HEIGHT,
            "mode": "L",
            "image_format": "png",
        },
        "transform": {
            "version": STAGE7D13_TRANSFORM_VERSION,
            "source_measure_bbox": dict(measure_box),
            "integer_crop": [plan.crop_left, plan.crop_top, plan.crop_right, plan.crop_bottom],
            "scale": plan.scale,
            "pad_x": plan.pad_x,
            "pad_y": plan.pad_y,
            "canvas": [INPUT_WIDTH, INPUT_HEIGHT],
            "resample": "pillow-bicubic-affine",
            "background": WHITE_BACKGROUND,
            "fingerprint": plan.transform_fingerprint,
            "transformed_measure_bbox": transformed_measure,
        },
        "targets": targets,
    }


def _prepare_output_root(output_root: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    resolved = output_root.resolve()
    if resolved == repo_root or repo_root in resolved.parents:
        _fail("D13 output root must remain outside the repository")
    if output_root.exists() or output_root.is_symlink():
        _fail("D13 output root must be fresh")
    output_root.mkdir(parents=True)
    (output_root / "images").mkdir()
    (output_root / "labels").mkdir()


def _write_new(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        _fail(f"refusing to overwrite D13 artifact: {path.name}")
    path.write_bytes(raw)


def _profile_payload() -> dict[str, object]:
    return {
        "version": STAGE7D13_DERIVATIVE_VERSION,
        "label_schema": STAGE7D13_LABEL_SCHEMA,
        "manifest_schema": STAGE7D13_MANIFEST_SCHEMA,
        "transform_version": STAGE7D13_TRANSFORM_VERSION,
        "contract_fingerprint": stage7d13_contract_fingerprint(),
        "accepted_d12": {
            "derivative_build_id": EXPECTED_D12_DERIVATIVE_BUILD_ID,
            "manifest_sha256": EXPECTED_D12_MANIFEST_SHA256,
            "artifact_binding_sha256": EXPECTED_D12_ARTIFACT_BINDING_SHA256,
        },
        "input": [INPUT_WIDTH, INPUT_HEIGHT],
        "crop_authority": "accepted_d12_measure_bbox",
        "letterbox": "outward_integer_crop_isotropic_affine_white_bicubic_v1",
        "test_specialist_records": TEST_SPECIALIST_RECORDS,
        "optimizer_steps": 0,
        "completion_policy": "independent-verifier-required-before-COMPLETE",
    }


def stage7d13_derivative_profile_fingerprint() -> str:
    return sha256(_canonical_json(_profile_payload())).hexdigest()


def build_stage7d13_measure_derivatives(
    *,
    corpus_root: str | Path,
    d6_root: str | Path,
    d12_root: str | Path,
    output_root: str | Path,
) -> Stage7D13DerivativeReceipt:
    """Build the development-only D13 measure derivative surface."""
    if not all(isinstance(value, (str, Path)) for value in (corpus_root, d6_root, d12_root, output_root)):
        raise TypeError("corpus_root, d6_root, d12_root and output_root must be str or pathlib.Path")
    source_root = Path(corpus_root)
    accepted_d6_root = Path(d6_root)
    accepted_d12_root = Path(d12_root)
    out = Path(output_root)
    _regular_directory(source_root, "source corpus root")
    _regular_directory(accepted_d6_root, "accepted D6 root")
    _regular_directory(accepted_d12_root, "accepted D12 root")
    if class_readiness_violations():
        _fail("frozen D12 class inventory fails D13 readiness")

    _accepted_d12_receipt(source_root, accepted_d6_root, accepted_d12_root)
    _d12_index_map, d12_rows = _d12_index(accepted_d12_root)
    _prepare_output_root(out)

    records: list[dict[str, object]] = []
    seen_record_ids: set[str] = set()
    image_hashes: set[str] = set()
    label_hashes: set[str] = set()
    image_bytes_total = 0
    label_bytes_total = 0
    source_samples_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    families_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    inventory = {
        split: {
            specialist: {name: 0 for name in classes}
            for specialist, classes in SPECIALIST_CLASSES.items()
        }
        for split in ("train", "validation")
    }
    binding_rows: list[str] = []

    for d12_record in d12_rows:
        d12_label, d12_label_sha = _load_d12_label(accepted_d12_root, d12_record)
        split = d12_label.get("split")
        assert isinstance(split, str)
        sample_id = _hex64("D12 label.sample_id", d12_label.get("sample_id"))
        family_id = _identifier("D12 label.family_id", d12_label.get("family_id"))
        image_info = d12_label.get("image")
        if not isinstance(image_info, Mapping):
            _fail("D12 label image must be an object")
        source_png_sha = _hex64("D12 label.image.png_sha256", image_info.get("png_sha256"))
        if source_png_sha != d12_record.get("png_sha256"):
            _fail("D12 label/manifest source PNG identity mismatch")
        width = _positive_int("D12 label.image.width", image_info.get("width"))
        height = _positive_int("D12 label.image.height", image_info.get("height"))
        if image_info.get("mode") != "L" or image_info.get("image_format") != "png":
            _fail("D12 source image format/mode drifted")
        source_raw = _read_bounded(
            source_root / "images" / f"{source_png_sha}.png",
            _MAX_IMAGE_BYTES,
            "frozen source PNG",
        )
        if sha256(source_raw).hexdigest() != source_png_sha:
            _fail("frozen source PNG SHA-256 mismatch")
        try:
            with Image.open(BytesIO(source_raw)) as opened:
                opened.load()
                if opened.format != "PNG" or opened.mode != "L" or opened.size != (width, height):
                    _fail("frozen source PNG metadata mismatch")
                source_image = opened.copy()
        except (OSError, ValueError) as exc:
            raise Stage7D13DerivativeError("frozen source PNG cannot be decoded") from exc

        geometry = d12_label.get("symbol_geometry")
        if not isinstance(geometry, Mapping) or geometry.get("coordinate_space") != "final_png_pixels":
            _fail("D12 symbol geometry coordinate space mismatch")
        measures = geometry.get("measures")
        if not isinstance(measures, list) or not measures:
            _fail("D12 symbol label must contain at least one measure")

        source_samples_by_split[split].add(sample_id)
        families_by_split[split].add(family_id)
        for measure in measures:
            if not isinstance(measure, Mapping):
                _fail("D12 measure must be an object")
            measure_box = _bbox("D12 measure_bbox", measure.get("measure_bbox"), width=width, height=height)
            plan = make_letterbox_plan(
                measure_box,
                image_width=width,
                image_height=height,
                source_png_sha256=source_png_sha,
            )
            image_raw = _render_measure_image(source_image, plan)
            derivative_png_sha = sha256(image_raw).hexdigest()
            image_path = out / "images" / f"{derivative_png_sha}.png"
            if derivative_png_sha not in image_hashes:
                _write_new(image_path, image_raw)
                image_hashes.add(derivative_png_sha)
                image_bytes_total += len(image_raw)
            elif not image_path.is_file() or image_path.read_bytes() != image_raw:
                _fail("D13 duplicate image hash does not resolve to identical bytes")

            label = _measure_payload(
                d12_label=d12_label,
                d12_label_sha=d12_label_sha,
                measure=measure,
                source_png_sha=source_png_sha,
                derivative_png_sha=derivative_png_sha,
                plan=plan,
                image_width=width,
                image_height=height,
            )
            record_id = _hex64("D13 label.record_id", label.get("record_id"))
            if record_id in seen_record_ids:
                _fail("duplicate D13 measure record_id")
            seen_record_ids.add(record_id)
            raw_label = _canonical_json(label)
            if not 1 <= len(raw_label) <= _MAX_LABEL_BYTES:
                _fail("D13 measure label byte length is outside bound")
            label_sha = sha256(raw_label).hexdigest()
            if label_sha in label_hashes:
                _fail("duplicate D13 measure label SHA-256")
            label_hashes.add(label_sha)
            _write_new(out / "labels" / f"{label_sha}.json", raw_label)
            label_bytes_total += len(raw_label)

            targets = label["targets"]
            assert isinstance(targets, dict)
            for specialist, classes in SPECIALIST_CLASSES.items():
                rows = targets[specialist]
                assert isinstance(rows, list)
                for row in rows:
                    assert isinstance(row, dict)
                    class_name = row["class"]
                    assert isinstance(class_name, str)
                    if class_name not in classes:
                        _fail("D13 transformed target class drifted")
                    inventory[split][specialist][class_name] += 1

            record = {
                "record_id": record_id,
                "sample_id": sample_id,
                "family_id": family_id,
                "split": split,
                "page_number": label["page_number"],
                "measure_number": label["measure_number"],
                "renderer_measure_id": label["renderer_measure_id"],
                "source_png_sha256": source_png_sha,
                "d12_label_sha256": d12_label_sha,
                "transform_fingerprint": plan.transform_fingerprint,
                "image_sha256": derivative_png_sha,
                "label_sha256": label_sha,
            }
            records.append(record)
            binding_rows.append(
                f"{record_id}:{sample_id}:{source_png_sha}:{d12_label_sha}:"
                f"{plan.transform_fingerprint}:{derivative_png_sha}:{label_sha}:"
                f"{len(image_raw)}:{len(raw_label)}"
            )

    source_sample_split_counts = {
        split: len(values) for split, values in source_samples_by_split.items()
    }
    family_split_counts = {
        split: len(values) for split, values in families_by_split.items()
    }
    if source_sample_split_counts != EXPECTED_SOURCE_SAMPLE_COUNTS:
        _fail("D13 source sample split counts mismatch")
    if family_split_counts != EXPECTED_SOURCE_FAMILY_COUNTS:
        _fail("D13 family split counts mismatch")
    if families_by_split["train"] & families_by_split["validation"]:
        _fail("D13 family crosses TRAIN/VALIDATION")
    if inventory != D12_CLASS_INVENTORY:
        _fail("D13 transformed target inventory differs from accepted D12")
    if not records or len(records) != len(label_hashes):
        _fail("D13 measure record/label cardinality mismatch")

    records.sort(key=lambda row: str(row["record_id"]))
    record_split_counts = dict(sorted(Counter(str(row["split"]) for row in records).items()))
    target_instance_counts = {
        split: {
            specialist: sum(inventory[split][specialist].values())
            for specialist in SPECIALIST_CLASSES
        }
        for split in ("train", "validation")
    }
    manifest_payload = {
        "schema_version": STAGE7D13_MANIFEST_SCHEMA,
        "stage7d13_derivative_version": STAGE7D13_DERIVATIVE_VERSION,
        "derivative_build_id": stage7d13_derivative_profile_fingerprint(),
        "contract_fingerprint": stage7d13_contract_fingerprint(),
        "accepted_d12": {
            "derivative_build_id": EXPECTED_D12_DERIVATIVE_BUILD_ID,
            "manifest_sha256": EXPECTED_D12_MANIFEST_SHA256,
            "artifact_binding_sha256": EXPECTED_D12_ARTIFACT_BINDING_SHA256,
        },
        "split_policy": "family-exclusive-train-validation-only-test-forbidden",
        "completion_policy": "independent-verifier-required-before-COMPLETE",
        "records": records,
    }
    manifest_raw = _canonical_json(manifest_payload)
    manifest_sha = sha256(manifest_raw).hexdigest()
    _write_new(out / "manifest.json", manifest_raw)
    _write_new(out / "manifest.sha256", f"{manifest_sha}  manifest.json\n".encode("ascii"))

    artifact_binding = sha256(("\n".join(sorted(binding_rows)) + "\n").encode("ascii")).hexdigest()
    build_payload = {
        "schema_version": STAGE7D13_BUILD_SCHEMA,
        "stage7d13_derivative_version": STAGE7D13_DERIVATIVE_VERSION,
        "derivative_build_id": stage7d13_derivative_profile_fingerprint(),
        "contract_fingerprint": stage7d13_contract_fingerprint(),
        "manifest_sha256": manifest_sha,
        "artifact_binding_sha256": artifact_binding,
        "record_count": len(records),
        "image_count": len(image_hashes),
        "label_count": len(label_hashes),
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
        "layout": {
            "manifest": "manifest.json",
            "images": "images/<image_sha256>.png",
            "labels": "labels/<label_sha256>.json",
        },
    }
    build_raw = _canonical_json(build_payload)
    if not 1 <= len(build_raw) <= _MAX_BUILD_BYTES:
        _fail("D13 build.json byte length is outside bound")
    _write_new(out / "build.json", build_raw)

    return Stage7D13DerivativeReceipt(
        derivative_build_id=stage7d13_derivative_profile_fingerprint(),
        manifest_sha256=manifest_sha,
        artifact_binding_sha256=artifact_binding,
        record_count=len(records),
        image_count=len(image_hashes),
        label_count=len(label_hashes),
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
        complete_marker_written=False,
    )
