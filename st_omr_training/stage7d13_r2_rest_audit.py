"""Stage 7-D13-R2 Rest root-cause audit helpers.

This module is intentionally diagnostic-only. It reopens the frozen D13-R1
measure derivative, verifies its persisted identity, and summarizes Rest target
geometry before any R2 optimizer is allowed to run.

It never opens TEST, never constructs an optimizer, never mutates a model, and
never rewrites the R1 derivative. The first goal is to distinguish data/geometry
problems from model/training problems before another full Rest training run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Final, Mapping, Sequence


STAGE7D13_R2_REST_AUDIT_VERSION: Final[str] = "stage7d13-r2-rest-audit-v1"

R1_DERIVATIVE_VERSION: Final[str] = "stage7d13-measure-derivatives-v1"
R1_LABEL_SCHEMA: Final[str] = "stage7d13-measure-label-v1"
R1_DERIVATIVE_BUILD_ID: Final[str] = (
    "44f1932532fb511dfa59a164f94be6b899f3aa0594c0ac0a6f499a38e5fb5697"
)
R1_MANIFEST_SHA256: Final[str] = (
    "8cfb87b5c6135be14b4c9ad488868c0edb0d37bb3bb18ad1b5e79d04fdf24f7b"
)
R1_ARTIFACT_BINDING_SHA256: Final[str] = (
    "c42c1f69e21d61d3eefdacfc40dabf2f0fcd6ac2ceb4d5cf88d8e158246dd33e"
)
R1_RECORD_SPLIT_COUNTS: Final[dict[str, int]] = {"train": 9840, "validation": 1224}
R1_RECORD_COUNT: Final[int] = 11064
R1_IMAGE_COUNT: Final[int] = 11062
R1_LABEL_COUNT: Final[int] = 11064
R1_REST_INSTANCE_COUNTS: Final[dict[str, int]] = {"train": 10602, "validation": 969}
R1_INPUT_WIDTH: Final[int] = 512
R1_INPUT_HEIGHT: Final[int] = 128
R1_OUTPUT_STRIDE: Final[int] = 4
REST_CLASSES: Final[tuple[str, ...]] = ("half", "quarter", "eighth")

_EXPECTED_TOP: Final[frozenset[str]] = frozenset(
    {"manifest.json", "manifest.sha256", "build.json", "images", "labels"}
)
_MAX_JSON_BYTES: Final[int] = 64 * 1024 * 1024
_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")


class Stage7D13R2RestAuditError(RuntimeError):
    """Raised when the frozen R1 Rest diagnostic surface cannot be proven."""


def _fail(message: str) -> None:
    raise Stage7D13R2RestAuditError(message)


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
        raise Stage7D13R2RestAuditError("audit payload is not canonical JSON") from exc


def _sha64(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in _HEX for ch in value)
    ):
        _fail(f"{name} must be lowercase SHA-256")
    return value


def _finite(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def _read_canonical_json(path: Path, name: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{name} must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= _MAX_JSON_BYTES:
        _fail(f"{name} byte length is outside audit bounds")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D13R2RestAuditError(f"{name} is not valid ASCII JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != raw:
        _fail(f"{name} must be canonical JSON object bytes")
    return value, raw


def _bbox(value: object, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, Mapping) or set(value) != {"x_min", "y_min", "x_max", "y_max"}:
        _fail(f"{name} must be a canonical bbox")
    x0 = _finite(value.get("x_min"), f"{name}.x_min")
    y0 = _finite(value.get("y_min"), f"{name}.y_min")
    x1 = _finite(value.get("x_max"), f"{name}.x_max")
    y1 = _finite(value.get("y_max"), f"{name}.y_max")
    if not 0.0 <= x0 < x1 <= R1_INPUT_WIDTH:
        _fail(f"{name} x bounds leave frozen R1 canvas")
    if not 0.0 <= y0 < y1 <= R1_INPUT_HEIGHT:
        _fail(f"{name} y bounds leave frozen R1 canvas")
    return x0, y0, x1, y1


def _center(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, Mapping) or set(value) != {"x", "y"}:
        _fail(f"{name} must be a canonical point")
    x = _finite(value.get("x"), f"{name}.x")
    y = _finite(value.get("y"), f"{name}.y")
    if not 0.0 <= x < R1_INPUT_WIDTH or not 0.0 <= y < R1_INPUT_HEIGHT:
        _fail(f"{name} leaves frozen R1 canvas")
    return x, y


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0,1]")
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "p05": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    ordered = [float(value) for value in values]
    return {
        "count": len(ordered),
        "min": min(ordered),
        "p05": _nearest_rank(ordered, 0.05),
        "p10": _nearest_rank(ordered, 0.10),
        "p25": _nearest_rank(ordered, 0.25),
        "p50": _nearest_rank(ordered, 0.50),
        "p75": _nearest_rank(ordered, 0.75),
        "p90": _nearest_rank(ordered, 0.90),
        "p95": _nearest_rank(ordered, 0.95),
        "max": max(ordered),
        "mean": fmean(ordered),
    }


@dataclass(frozen=True, slots=True)
class RestGeometryObservation:
    split: str
    class_name: str
    record_id: str
    label_sha256: str
    width: float
    height: float
    area: float
    center_x: float
    center_y: float
    transform_scale: float


@dataclass(frozen=True, slots=True)
class Stage7D13R2RestAuditReceipt:
    version: str
    derivative_build_id: str
    manifest_sha256: str
    artifact_binding_sha256: str
    record_count: int
    image_count: int
    label_count: int
    record_split_counts: dict[str, int]
    rest_instance_counts: dict[str, int]
    rest_class_counts: dict[str, dict[str, int]]
    rest_positive_records: dict[str, int]
    rest_zero_records: dict[str, int]
    stride4_collision_records: int
    geometry: dict[str, object]
    smallest_examples: dict[str, list[dict[str, object]]]
    test_opened: bool
    optimizer_steps: int
    model_loaded: bool
    audit_passed: bool


def development_rows(rows: object) -> tuple[Mapping[str, object], ...]:
    """Accept only TRAIN/VALIDATION and fail immediately on sealed TEST."""
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        _fail("manifest records must be a sequence")
    accepted: list[Mapping[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"manifest record[{index}] must be an object")
        split = row.get("split")
        if split == "test":
            _fail("sealed TEST record reached D13-R2 Rest audit")
        if split not in ("train", "validation"):
            _fail("Rest audit split must be train or validation")
        accepted.append(row)
    return tuple(accepted)


def summarize_rest_labels(
    rows: Sequence[tuple[Mapping[str, object], str]],
) -> dict[str, object]:
    """Summarize already-verified D13 labels without touching images/models.

    This helper is deliberately separated from the authoritative root verifier so
    unit tests can exercise geometry logic without manufacturing the 11,064-file
    frozen corpus.
    """
    split_counts: Counter[str] = Counter()
    rest_counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    positive_records: Counter[str] = Counter()
    zero_records: Counter[str] = Counter()
    collisions = 0
    observations: list[RestGeometryObservation] = []
    scales: dict[str, list[float]] = {"train": [], "validation": []}

    for label, label_sha in rows:
        split = label.get("split")
        if split not in ("train", "validation"):
            _fail("verified label split must be train or validation")
        assert isinstance(split, str)
        split_counts[split] += 1
        record_id = _sha64(label.get("record_id"), "label.record_id")
        _sha64(label_sha, "label SHA")
        transform = label.get("transform")
        if not isinstance(transform, Mapping):
            _fail("label.transform must be an object")
        scale = _finite(transform.get("scale"), "label.transform.scale")
        if scale <= 0.0:
            _fail("label.transform.scale must be positive")
        scales[split].append(scale)

        targets = label.get("targets")
        if not isinstance(targets, Mapping):
            _fail("label.targets must be an object")
        rests = targets.get("rest")
        if not isinstance(rests, list):
            _fail("label.targets.rest must be a list")
        if rests:
            positive_records[split] += 1
        else:
            zero_records[split] += 1

        occupied: set[tuple[int, int]] = set()
        collision_here = False
        for index, target in enumerate(rests):
            if not isinstance(target, Mapping):
                _fail(f"rest target[{index}] must be an object")
            class_name = target.get("class")
            if class_name not in REST_CLASSES:
                _fail("rest target class is outside half|quarter|eighth")
            assert isinstance(class_name, str)
            x0, y0, x1, y1 = _bbox(target.get("bbox"), "rest target bbox")
            cx, cy = _center(target.get("center"), "rest target center")
            if not x0 <= cx <= x1 or not y0 <= cy <= y1:
                _fail("rest target center lies outside bbox")
            cell = (int(math.floor(cy / R1_OUTPUT_STRIDE)), int(math.floor(cx / R1_OUTPUT_STRIDE)))
            if cell in occupied:
                collision_here = True
            occupied.add(cell)
            width = x1 - x0
            height = y1 - y0
            rest_counts[split] += 1
            class_counts[split][class_name] += 1
            observations.append(
                RestGeometryObservation(
                    split=split,
                    class_name=class_name,
                    record_id=record_id,
                    label_sha256=label_sha,
                    width=width,
                    height=height,
                    area=width * height,
                    center_x=cx,
                    center_y=cy,
                    transform_scale=scale,
                )
            )
        if collision_here:
            collisions += 1

    geometry: dict[str, object] = {}
    for split in ("train", "validation"):
        split_obs = [row for row in observations if row.split == split]
        geometry[split] = {
            "transform_scale": _distribution(scales[split]),
            "width": _distribution([row.width for row in split_obs]),
            "height": _distribution([row.height for row in split_obs]),
            "area": _distribution([row.area for row in split_obs]),
            "min_dimension": _distribution([min(row.width, row.height) for row in split_obs]),
            "sub_stride_width_count": sum(row.width < R1_OUTPUT_STRIDE for row in split_obs),
            "sub_stride_height_count": sum(row.height < R1_OUTPUT_STRIDE for row in split_obs),
            "sub_stride_min_dimension_count": sum(
                min(row.width, row.height) < R1_OUTPUT_STRIDE for row in split_obs
            ),
            "under_two_stride_min_dimension_count": sum(
                min(row.width, row.height) < 2 * R1_OUTPUT_STRIDE for row in split_obs
            ),
            "by_class": {
                class_name: {
                    "count": sum(row.class_name == class_name for row in split_obs),
                    "width": _distribution(
                        [row.width for row in split_obs if row.class_name == class_name]
                    ),
                    "height": _distribution(
                        [row.height for row in split_obs if row.class_name == class_name]
                    ),
                    "area": _distribution(
                        [row.area for row in split_obs if row.class_name == class_name]
                    ),
                }
                for class_name in REST_CLASSES
            },
        }

    smallest_examples: dict[str, list[dict[str, object]]] = {}
    for class_name in REST_CLASSES:
        candidates = sorted(
            (row for row in observations if row.class_name == class_name),
            key=lambda row: (row.area, row.height, row.width, row.record_id, row.label_sha256),
        )[:12]
        smallest_examples[class_name] = [
            {
                "split": row.split,
                "record_id": row.record_id,
                "label_sha256": row.label_sha256,
                "width": row.width,
                "height": row.height,
                "area": row.area,
                "center_x": row.center_x,
                "center_y": row.center_y,
                "transform_scale": row.transform_scale,
            }
            for row in candidates
        ]

    return {
        "record_split_counts": dict(sorted(split_counts.items())),
        "rest_instance_counts": dict(sorted(rest_counts.items())),
        "rest_class_counts": {
            split: {name: class_counts[split][name] for name in REST_CLASSES}
            for split in ("train", "validation")
        },
        "rest_positive_records": {
            split: positive_records[split] for split in ("train", "validation")
        },
        "rest_zero_records": {
            split: zero_records[split] for split in ("train", "validation")
        },
        "stride4_collision_records": collisions,
        "geometry": geometry,
        "smallest_examples": smallest_examples,
    }


def _verify_authoritative_root(root: Path) -> tuple[dict[str, object], tuple[Mapping[str, object], ...]]:
    if root.is_symlink() or not root.is_dir():
        _fail("R1 derivative root must be a regular non-symlink directory")
    if {path.name for path in root.iterdir()} != _EXPECTED_TOP:
        _fail("R1 derivative top-level surface mismatch")

    build, _build_raw = _read_canonical_json(root / "build.json", "R1 build.json")
    expected_build = {
        "derivative_build_id": R1_DERIVATIVE_BUILD_ID,
        "manifest_sha256": R1_MANIFEST_SHA256,
        "artifact_binding_sha256": R1_ARTIFACT_BINDING_SHA256,
        "record_count": R1_RECORD_COUNT,
        "image_count": R1_IMAGE_COUNT,
        "label_count": R1_LABEL_COUNT,
        "record_split_counts": R1_RECORD_SPLIT_COUNTS,
        "test_specialist_records": 0,
        "optimizer_steps": 0,
        "complete_marker_written": False,
    }
    for name, expected in expected_build.items():
        if build.get(name) != expected:
            _fail(f"R1 build {name} mismatch")

    manifest, manifest_raw = _read_canonical_json(root / "manifest.json", "R1 manifest.json")
    if sha256(manifest_raw).hexdigest() != R1_MANIFEST_SHA256:
        _fail("R1 manifest SHA-256 mismatch")
    if manifest.get("stage7d13_derivative_version") != R1_DERIVATIVE_VERSION:
        _fail("R1 manifest derivative version mismatch")
    if manifest.get("derivative_build_id") != R1_DERIVATIVE_BUILD_ID:
        _fail("R1 manifest build id mismatch")

    sidecar = root / "manifest.sha256"
    if sidecar.is_symlink() or not sidecar.is_file():
        _fail("R1 manifest.sha256 must be regular file")
    expected_sidecar = f"{R1_MANIFEST_SHA256}  manifest.json\n".encode("ascii")
    if sidecar.read_bytes() != expected_sidecar:
        _fail("R1 manifest.sha256 sidecar mismatch")

    rows = development_rows(manifest.get("records"))
    if len(rows) != R1_RECORD_COUNT:
        _fail("R1 manifest record cardinality mismatch")
    counts = dict(sorted(Counter(str(row.get("split")) for row in rows).items()))
    if counts != R1_RECORD_SPLIT_COUNTS:
        _fail("R1 manifest split cardinality mismatch")
    return build, rows


def run_stage7d13_r2_rest_derivative_audit(
    derivative_root: str | Path,
    *,
    report_path: str | Path | None = None,
    heartbeat=print,
) -> Stage7D13R2RestAuditReceipt:
    """Run the authoritative read-only Rest geometry audit on frozen D13-R1."""
    if not isinstance(derivative_root, (str, Path)):
        raise TypeError("derivative_root must be str or pathlib.Path")
    root = Path(derivative_root)
    build, rows = _verify_authoritative_root(root)
    if heartbeat is not None:
        heartbeat(
            f"D13-R2 REST AUDIT START | records={R1_RECORD_COUNT} | "
            f"TRAIN={R1_RECORD_SPLIT_COUNTS['train']} | "
            f"VALIDATION={R1_RECORD_SPLIT_COUNTS['validation']} | TEST=0"
        )

    verified_labels: list[tuple[Mapping[str, object], str]] = []
    seen_labels: set[str] = set()
    seen_images: set[str] = set()
    for index, row in enumerate(rows, start=1):
        record_id = _sha64(row.get("record_id"), "manifest.record_id")
        label_sha = _sha64(row.get("label_sha256"), "manifest.label_sha256")
        image_sha = _sha64(row.get("image_sha256"), "manifest.image_sha256")
        if label_sha in seen_labels:
            _fail("duplicate R1 label SHA reached Rest audit")
        seen_labels.add(label_sha)
        seen_images.add(image_sha)
        label, raw = _read_canonical_json(
            root / "labels" / f"{label_sha}.json",
            "R1 measure label",
        )
        if sha256(raw).hexdigest() != label_sha:
            _fail("R1 measure label SHA-256 mismatch")
        if label.get("schema_version") != R1_LABEL_SCHEMA:
            _fail("R1 measure label schema mismatch")
        if label.get("stage7d13_derivative_version") != R1_DERIVATIVE_VERSION:
            _fail("R1 measure label derivative version mismatch")
        if label.get("record_id") != record_id or label.get("split") != row.get("split"):
            _fail("R1 label/manifest identity mismatch")
        image_info = label.get("image")
        if not isinstance(image_info, Mapping):
            _fail("R1 label.image must be an object")
        if image_info.get("png_sha256") != image_sha:
            _fail("R1 label/manifest image identity mismatch")
        if image_info.get("width") != R1_INPUT_WIDTH or image_info.get("height") != R1_INPUT_HEIGHT:
            _fail("R1 label image dimensions mismatch")
        if image_info.get("mode") != "L" or image_info.get("image_format") != "png":
            _fail("R1 label image mode/format mismatch")
        verified_labels.append((label, label_sha))
        if heartbeat is not None and (index % 1000 == 0 or index == R1_RECORD_COUNT):
            heartbeat(f"D13-R2 REST AUDIT LABELS {index}/{R1_RECORD_COUNT}")

    if len(seen_labels) != R1_LABEL_COUNT or len(seen_images) != R1_IMAGE_COUNT:
        _fail("R1 persisted label/image cardinality mismatch")

    summary = summarize_rest_labels(verified_labels)
    if summary["record_split_counts"] != R1_RECORD_SPLIT_COUNTS:
        _fail("Rest audit record split summary mismatch")
    if summary["rest_instance_counts"] != R1_REST_INSTANCE_COUNTS:
        _fail("Rest audit instance counts differ from frozen R1 evidence")
    if summary["stride4_collision_records"] != 0:
        _fail("Rest audit found class-agnostic stride-4 regression collisions")

    receipt = Stage7D13R2RestAuditReceipt(
        version=STAGE7D13_R2_REST_AUDIT_VERSION,
        derivative_build_id=str(build["derivative_build_id"]),
        manifest_sha256=str(build["manifest_sha256"]),
        artifact_binding_sha256=str(build["artifact_binding_sha256"]),
        record_count=R1_RECORD_COUNT,
        image_count=R1_IMAGE_COUNT,
        label_count=R1_LABEL_COUNT,
        record_split_counts=dict(summary["record_split_counts"]),
        rest_instance_counts=dict(summary["rest_instance_counts"]),
        rest_class_counts=dict(summary["rest_class_counts"]),
        rest_positive_records=dict(summary["rest_positive_records"]),
        rest_zero_records=dict(summary["rest_zero_records"]),
        stride4_collision_records=int(summary["stride4_collision_records"]),
        geometry=dict(summary["geometry"]),
        smallest_examples=dict(summary["smallest_examples"]),
        test_opened=False,
        optimizer_steps=0,
        model_loaded=False,
        audit_passed=True,
    )

    if report_path is not None:
        report = Path(report_path)
        if report.exists() or report.is_symlink():
            _fail("Rest audit report path must be fresh")
        payload = {
            "version": receipt.version,
            "derivative_build_id": receipt.derivative_build_id,
            "manifest_sha256": receipt.manifest_sha256,
            "artifact_binding_sha256": receipt.artifact_binding_sha256,
            "record_count": receipt.record_count,
            "image_count": receipt.image_count,
            "label_count": receipt.label_count,
            "record_split_counts": receipt.record_split_counts,
            "rest_instance_counts": receipt.rest_instance_counts,
            "rest_class_counts": receipt.rest_class_counts,
            "rest_positive_records": receipt.rest_positive_records,
            "rest_zero_records": receipt.rest_zero_records,
            "stride4_collision_records": receipt.stride4_collision_records,
            "geometry": receipt.geometry,
            "smallest_examples": receipt.smallest_examples,
            "test_opened": receipt.test_opened,
            "optimizer_steps": receipt.optimizer_steps,
            "model_loaded": receipt.model_loaded,
            "audit_passed": receipt.audit_passed,
        }
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_bytes(_canonical_json(payload))

    if heartbeat is not None:
        heartbeat(
            "D13-R2 REST AUDIT PASS | TEST=False | optimizer=0 | model_loaded=False"
        )
    return receipt
