"""METER V5-1 clean package_ab bbox pilot.

Safety scope:
- exact clean dataset name only;
- dataset structure/manifests are validated before annotation;
- final_holdout is count-verified but never exposed to the annotation session;
- pilot selection is train-only (10 per class);
- original image.png files are read-only and SHA-bound;
- annotations are stored separately as crash-safe CSV checkpoints.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence

from PIL import Image

DATASET_NAME = "METER_V2_1500_PACKAGE_AB_CLEAN"
CLASSES = ("2/4", "3/4", "4/4")
CLASS_DIR = {"2/4": "2_4", "3/4": "3_4", "4/4": "4_4"}
MANIFEST_NAME = {
    "2/4": "2_4_SELECTION_MANIFEST.csv",
    "3/4": "3_4_SELECTION_MANIFEST.csv",
    "4/4": "4_4_SELECTION_MANIFEST.csv",
}
EXPECTED_SPLIT_COUNTS = {"train": 400, "val": 50, "final_holdout": 50}
EXPECTED_CLASS_COUNT = 500
EXPECTED_TOTAL = 1500
PILOT_PER_CLASS = 10
PILOT_TOTAL = 30
PILOT_SEED = "st-omr-meter-v5-1-clean-bbox-pilot-v1"

ANNOTATIONS_DIR = "annotations"
PILOT_SELECTION_NAME = "bbox_pilot_30_selection.csv"
PILOT_CSV_NAME = "bbox_pilot_30.csv"
PILOT_AUDIT_NAME = "bbox_pilot_30_audit.json"
FINAL_HOLDOUT_LOCK_NAME = "FINAL_HOLDOUT_LOCK.json"

REQUIRED_MANIFEST_COLUMNS = {
    "Split", "Meter", "Package", "FamilyId", "SampleId", "Folder",
    "SourceImage", "SourceSemantic", "SourceAgnostic",
}
ANNOTATION_COLUMNS = (
    "sample_id", "meter", "split", "x", "y", "w", "h", "status",
    "image_sha256", "image_width", "image_height", "updated_utc",
)
PILOT_SELECTION_COLUMNS = (
    "index", "sample_id", "family_id", "meter", "split", "folder",
    "image_relpath", "image_sha256", "image_width", "image_height",
    "selection_rank",
)


class MeterV5_1PilotError(RuntimeError):
    """Fail-closed bbox pilot error."""


def _fail(message: str) -> None:
    raise MeterV5_1PilotError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(tmp), str(path))


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_bytes(path, json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(tmp), str(path))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        _fail(f"missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            _fail(f"CSV header missing: {path}")
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def _package_ab_path(value: str) -> bool:
    normalized = value.replace("/", "\\").lower()
    return "\\package_ab\\" in normalized


def discover_data_root(my_drive_root: str | Path) -> Path:
    """Find exactly one clean dataset directory under MyDrive."""
    root = Path(my_drive_root)
    if not root.is_dir():
        _fail(f"MyDrive root not found: {root}")

    matches: list[Path] = []
    for current, dirs, _files in os.walk(str(root)):
        current_path = Path(current)
        if current_path.name == DATASET_NAME:
            matches.append(current_path)
            dirs[:] = []
            if len(matches) > 1:
                break

    if not matches:
        _fail(f"{DATASET_NAME} not found under {root}")
    if len(matches) > 1:
        joined = "\n".join(str(p) for p in matches)
        _fail(f"multiple {DATASET_NAME} folders found; user selection required:\n{joined}")
    return matches[0]


def _validate_manifest(path: Path, expected_meter: str) -> list[dict[str, str]]:
    rows = _read_csv(path)
    if len(rows) != EXPECTED_CLASS_COUNT:
        _fail(f"{path.name}: expected 500 rows, got {len(rows)}")
    if not rows:
        _fail(f"{path.name}: empty")
    missing = REQUIRED_MANIFEST_COLUMNS - set(rows[0])
    if missing:
        _fail(f"{path.name}: missing columns {sorted(missing)}")

    split_counts = Counter()
    for row in rows:
        if row["Meter"] != expected_meter:
            _fail(f"{path.name}: meter mismatch {row['Meter']!r}")
        if row["Package"] != "package_ab":
            _fail(f"{path.name}: non-package_ab row: {row['SampleId']}")
        if row["Split"] not in EXPECTED_SPLIT_COUNTS:
            _fail(f"{path.name}: invalid split {row['Split']!r}")
        split_counts[row["Split"]] += 1
        if not row["FamilyId"] or not row["SampleId"] or not row["Folder"]:
            _fail(f"{path.name}: empty identity field")
        for key in ("SourceImage", "SourceSemantic", "SourceAgnostic"):
            if not _package_ab_path(row[key]):
                _fail(f"{path.name}: {key} not under package_ab for {row['SampleId']}")

    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        _fail(f"{path.name}: split counts {dict(split_counts)} != {EXPECTED_SPLIT_COUNTS}")
    if len({r["FamilyId"] for r in rows}) != EXPECTED_CLASS_COUNT:
        _fail(f"{path.name}: FamilyId must be unique within class")
    if len({r["SampleId"] for r in rows}) != EXPECTED_CLASS_COUNT:
        _fail(f"{path.name}: SampleId must be unique within class")
    if len({r["SourceImage"] for r in rows}) != EXPECTED_CLASS_COUNT:
        _fail(f"{path.name}: SourceImage must be unique within class")
    return rows


def verify_dataset_structure(data_root: str | Path) -> dict[str, object]:
    """Validate the clean 1500 dataset without reading final_holdout image bytes."""
    root = Path(data_root)
    if root.name != DATASET_NAME:
        _fail(f"refusing non-clean dataset root: {root}")

    manifests: dict[str, list[dict[str, str]]] = {}
    manifest_sha256: dict[str, str] = {}
    all_rows: list[dict[str, str]] = []
    directory_counts: dict[str, int] = {}

    for meter in CLASSES:
        manifest_path = root / MANIFEST_NAME[meter]
        manifests[meter] = _validate_manifest(manifest_path, meter)
        manifest_sha256[meter] = _sha256_file(manifest_path)
        all_rows.extend(manifests[meter])

    if len(all_rows) != EXPECTED_TOTAL:
        _fail(f"expected 1500 manifest rows, got {len(all_rows)}")

    unique_family = {r["FamilyId"] for r in all_rows}
    unique_sample = {r["SampleId"] for r in all_rows}
    unique_source = {r["SourceImage"] for r in all_rows}
    if len(unique_family) != EXPECTED_TOTAL:
        _fail(f"global unique FamilyId gate failed: {len(unique_family)}")
    if len(unique_sample) != EXPECTED_TOTAL:
        _fail(f"global unique SampleId gate failed: {len(unique_sample)}")
    if len(unique_source) != EXPECTED_TOTAL:
        _fail(f"global unique SourceImage gate failed: {len(unique_source)}")

    family_splits: dict[str, set[str]] = defaultdict(set)
    family_meters: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        family_splits[row["FamilyId"]].add(row["Split"])
        family_meters[row["FamilyId"]].add(row["Meter"])
    cross_split = sorted(k for k, v in family_splits.items() if len(v) > 1)
    cross_meter = sorted(k for k, v in family_meters.items() if len(v) > 1)
    if cross_split:
        _fail(f"cross-split family leakage detected: {cross_split[:5]}")
    if cross_meter:
        _fail(f"cross-meter family overlap detected: {cross_meter[:5]}")

    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        for meter in CLASSES:
            class_path = root / split / CLASS_DIR[meter]
            if not class_path.is_dir():
                _fail(f"missing dataset directory: {class_path}")
            sample_dirs = [p for p in class_path.iterdir() if p.is_dir()]
            key = f"{split}/{CLASS_DIR[meter]}"
            directory_counts[key] = len(sample_dirs)
            if len(sample_dirs) != expected:
                _fail(f"{key}: expected {expected} sample folders, got {len(sample_dirs)}")

    for row in all_rows:
        sample_dir = root / row["Split"] / CLASS_DIR[row["Meter"]] / row["Folder"]
        if not sample_dir.is_dir():
            _fail(f"manifest folder missing from copied dataset: {sample_dir}")
        image_path = sample_dir / "image.png"
        if not image_path.is_file():
            _fail(f"image.png missing: {image_path}")

    fingerprint_payload = {
        "dataset": DATASET_NAME,
        "manifest_sha256": manifest_sha256,
        "directory_counts": directory_counts,
        "unique_family_id": len(unique_family),
        "unique_sample_id": len(unique_sample),
        "unique_source_image": len(unique_source),
        "package_ab_only": True,
        "cross_split_family_leakage": 0,
        "cross_meter_family_overlap": 0,
    }
    fingerprint = hashlib.sha256(_canonical_json(fingerprint_payload)).hexdigest()
    return {
        **fingerprint_payload,
        "dataset_fingerprint_sha256": fingerprint,
        "total": EXPECTED_TOTAL,
        "final_holdout_locked": True,
        "annotation_authorized_scope": "train_pilot_30_only",
        "training_authorized": False,
        "model_opened": False,
        "inference_count": 0,
    }


def ensure_final_holdout_lock(data_root: str | Path, gate: Mapping[str, object]) -> Path:
    root = Path(data_root)
    annotations = root / ANNOTATIONS_DIR
    annotations.mkdir(parents=True, exist_ok=True)
    path = annotations / FINAL_HOLDOUT_LOCK_NAME
    payload = {
        "schema": "st-omr-meter-v5-1-final-holdout-lock-v1",
        "dataset": DATASET_NAME,
        "dataset_fingerprint_sha256": gate["dataset_fingerprint_sha256"],
        "final_holdout_count": 150,
        "locked": True,
        "annotation_opened": False,
        "training_opened": False,
        "tuning_opened": False,
        "model_evaluated": False,
        "inference_count": 0,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8-sig"))
        if existing != payload:
            _fail("existing final_holdout lock does not match current clean dataset")
    else:
        _atomic_write_json(path, payload)
    return path


@dataclass(frozen=True)
class PilotSample:
    index: int
    sample_id: str
    family_id: str
    meter: str
    split: str
    folder: str
    image_path: Path
    image_sha256: str
    image_width: int
    image_height: int
    selection_rank: str


def _read_png_binding(path: Path) -> tuple[str, int, int]:
    if not path.is_file():
        _fail(f"pilot image missing: {path}")
    sha = _sha256_file(path)
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
    except Exception as exc:
        raise MeterV5_1PilotError(f"invalid pilot PNG: {path}") from exc
    if width <= 0 or height <= 0:
        _fail(f"invalid pilot image dimensions: {path}")
    return sha, width, height


def _pilot_rank(row: Mapping[str, str]) -> str:
    source_rank = row.get("SplitRank") or row.get("SelectionRank") or ""
    return _sha256_text(
        f"{PILOT_SEED}|{row['Meter']}|{row['FamilyId']}|{row['SampleId']}|{source_rank}"
    )


def load_or_create_pilot_selection(
    data_root: str | Path,
    gate: Mapping[str, object],
) -> tuple[PilotSample, ...]:
    root = Path(data_root)
    annotations = root / ANNOTATIONS_DIR
    annotations.mkdir(parents=True, exist_ok=True)
    path = annotations / PILOT_SELECTION_NAME

    if not path.exists():
        selected_rows: list[dict[str, object]] = []
        index = 0
        for meter in CLASSES:
            manifest = _validate_manifest(root / MANIFEST_NAME[meter], meter)
            train_rows = [r for r in manifest if r["Split"] == "train"]
            ranked = sorted(train_rows, key=lambda r: (_pilot_rank(r), r["SampleId"]))[:PILOT_PER_CLASS]
            if len(ranked) != PILOT_PER_CLASS:
                _fail(f"{meter}: could not select 10 train pilot samples")
            for row in ranked:
                image_path = root / "train" / CLASS_DIR[meter] / row["Folder"] / "image.png"
                sha, width, height = _read_png_binding(image_path)
                selected_rows.append({
                    "index": index,
                    "sample_id": row["SampleId"],
                    "family_id": row["FamilyId"],
                    "meter": meter,
                    "split": "train",
                    "folder": row["Folder"],
                    "image_relpath": image_path.relative_to(root).as_posix(),
                    "image_sha256": sha,
                    "image_width": width,
                    "image_height": height,
                    "selection_rank": _pilot_rank(row),
                })
                index += 1
        _atomic_write_csv(path, PILOT_SELECTION_COLUMNS, selected_rows)

    rows = _read_csv(path)
    if len(rows) != PILOT_TOTAL:
        _fail(f"pilot selection must contain 30 rows, got {len(rows)}")
    samples: list[PilotSample] = []
    meters = Counter()
    ids: set[str] = set()
    for expected_index, row in enumerate(rows):
        try:
            index = int(row["index"])
            width = int(row["image_width"])
            height = int(row["image_height"])
        except (KeyError, ValueError) as exc:
            raise MeterV5_1PilotError("pilot selection contains invalid integer fields") from exc
        if index != expected_index:
            _fail("pilot selection index order mismatch")
        if row["split"] != "train" or row["meter"] not in CLASSES:
            _fail("pilot selection must be train-only and use 2/4,3/4,4/4")
        if row["sample_id"] in ids:
            _fail("duplicate sample_id in pilot selection")
        ids.add(row["sample_id"])
        meters[row["meter"]] += 1
        image_path = root / row["image_relpath"]
        current_sha, current_width, current_height = _read_png_binding(image_path)
        if current_sha != row["image_sha256"] or current_width != width or current_height != height:
            _fail(f"pilot image changed after binding: {row['sample_id']}")
        samples.append(PilotSample(
            index=index,
            sample_id=row["sample_id"],
            family_id=row["family_id"],
            meter=row["meter"],
            split=row["split"],
            folder=row["folder"],
            image_path=image_path,
            image_sha256=current_sha,
            image_width=width,
            image_height=height,
            selection_rank=row["selection_rank"],
        ))
    if meters != Counter({"2/4": 10, "3/4": 10, "4/4": 10}):
        _fail(f"pilot class balance mismatch: {dict(meters)}")
    return tuple(samples)


def _parse_int(value: object, name: str) -> int:
    if type(value) is not int:
        _fail(f"{name} must be integer")
    return value


def _validate_bbox(x: int, y: int, w: int, h: int, image_width: int, image_height: int) -> None:
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        _fail("bbox coordinates must be non-negative and size positive")
    if x + w > image_width or y + h > image_height:
        _fail("bbox extends outside original image")
    if w < 3 or h < 6:
        _fail("bbox is too small to be a full meter pair")
    if w > max(12, math.ceil(image_width * 0.35)):
        _fail("bbox is implausibly wide; redraw tightly around the meter pair")
    if h > max(12, math.ceil(image_height * 0.98)):
        _fail("bbox is implausibly tall; redraw tightly around the meter pair")


def _preview(path: Path, max_width: int = 1100, max_height: int = 760) -> tuple[str, int, int]:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        scale = min(1.0, max_width / width, max_height / height)
        pw = max(1, int(math.floor(width * scale)))
        ph = max(1, int(math.floor(height * scale)))
        if (pw, ph) != (width, height):
            image = image.resize((pw, ph), Image.Resampling.LANCZOS)
        if image.mode not in {"L", "RGB", "RGBA"}:
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"), pw, ph


def _rect_to_original(
    *, x0: int, y0: int, x1: int, y1: int,
    preview_width: int, preview_height: int,
    image_width: int, image_height: int,
) -> tuple[int, int, int, int]:
    if preview_width <= 0 or preview_height <= 0:
        _fail("preview dimensions must be positive")
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if left < 0 or top < 0 or right > preview_width or bottom > preview_height:
        _fail("preview bbox outside canvas")
    ox0 = int(math.floor(left * image_width / preview_width))
    oy0 = int(math.floor(top * image_height / preview_height))
    ox1 = int(math.ceil(right * image_width / preview_width))
    oy1 = int(math.ceil(bottom * image_height / preview_height))
    x = max(0, min(ox0, image_width))
    y = max(0, min(oy0, image_height))
    x2 = max(0, min(ox1, image_width))
    y2 = max(0, min(oy1, image_height))
    w, h = x2 - x, y2 - y
    _validate_bbox(x, y, w, h, image_width, image_height)
    return x, y, w, h


class AnnotationSession:
    """Crash-safe train-only 30-sample bbox pilot session."""

    def __init__(self, *, data_root: str | Path):
        self.data_root = Path(data_root)
        self.gate = verify_dataset_structure(self.data_root)
        self.holdout_lock_path = ensure_final_holdout_lock(self.data_root, self.gate)
        self.samples = load_or_create_pilot_selection(self.data_root, self.gate)
        self.selection_path = self.data_root / ANNOTATIONS_DIR / PILOT_SELECTION_NAME
        self.selection_sha256 = _sha256_file(self.selection_path)
        self.annotation_path = self.data_root / ANNOTATIONS_DIR / PILOT_CSV_NAME
        self.annotations = self._load_annotations()

    def _load_annotations(self) -> dict[str, dict[str, str]]:
        if not self.annotation_path.exists():
            _atomic_write_csv(self.annotation_path, ANNOTATION_COLUMNS, [])
            return {}
        rows = _read_csv(self.annotation_path)
        valid_ids = {s.sample_id: s for s in self.samples}
        result: dict[str, dict[str, str]] = {}
        for row in rows:
            sample_id = row.get("sample_id", "")
            if sample_id not in valid_ids:
                _fail(f"annotation outside frozen pilot selection: {sample_id}")
            if sample_id in result:
                _fail(f"duplicate annotation row: {sample_id}")
            sample = valid_ids[sample_id]
            if row.get("meter") != sample.meter or row.get("split") != "train":
                _fail(f"annotation identity mismatch: {sample_id}")
            if row.get("image_sha256") != sample.image_sha256:
                _fail(f"annotation image binding mismatch: {sample_id}")
            if row.get("status") not in {"PASS", "REVIEW"}:
                _fail(f"invalid annotation status: {sample_id}")
            if row["status"] == "PASS":
                try:
                    x, y, w, h = (int(row[k]) for k in ("x", "y", "w", "h"))
                except ValueError as exc:
                    raise MeterV5_1PilotError(f"invalid bbox integers: {sample_id}") from exc
                _validate_bbox(x, y, w, h, sample.image_width, sample.image_height)
            else:
                if any(row.get(k, "") for k in ("x", "y", "w", "h")):
                    _fail(f"REVIEW row must not contain bbox: {sample_id}")
            result[sample_id] = row
        return result

    @property
    def handled_count(self) -> int:
        return len(self.annotations)

    @property
    def pass_count(self) -> int:
        return sum(row["status"] == "PASS" for row in self.annotations.values())

    @property
    def review_count(self) -> int:
        return sum(row["status"] == "REVIEW" for row in self.annotations.values())

    def resume_index(self) -> int:
        for sample in self.samples:
            if sample.sample_id not in self.annotations:
                return sample.index
        return max(0, len(self.samples) - 1)

    def _token(self, sample: PilotSample) -> str:
        return _sha256_text(
            f"{self.selection_sha256}\n{sample.index}\n{sample.sample_id}\n"
            f"{sample.image_sha256}\n{sample.image_width}x{sample.image_height}\n"
        )

    def _resolve_token(self, token: object) -> PilotSample:
        if not isinstance(token, str) or len(token) != 64:
            _fail("sample binding token malformed")
        for sample in self.samples:
            if self._token(sample) == token:
                sha, width, height = _read_png_binding(sample.image_path)
                if (sha, width, height) != (sample.image_sha256, sample.image_width, sample.image_height):
                    _fail(f"original image changed: {sample.sample_id}")
                return sample
        _fail("sample binding token does not match pilot selection")

    def sample_payload(self, index: int) -> dict[str, object]:
        if type(index) is not int or not 0 <= index < len(self.samples):
            _fail("pilot index outside 0..29")
        sample = self.samples[index]
        sha, width, height = _read_png_binding(sample.image_path)
        if (sha, width, height) != (sample.image_sha256, sample.image_width, sample.image_height):
            _fail(f"original image changed: {sample.sample_id}")
        data_uri, pw, ph = _preview(sample.image_path)
        existing = self.annotations.get(sample.sample_id)
        bbox = None
        status = None
        if existing:
            status = existing["status"]
            if status == "PASS":
                bbox = {k: int(existing[k]) for k in ("x", "y", "w", "h")}
        return {
            "index": sample.index,
            "total": PILOT_TOTAL,
            "sample_id": sample.sample_id,
            "family_id": sample.family_id,
            "meter": sample.meter,
            "split": "train",
            "image_width": sample.image_width,
            "image_height": sample.image_height,
            "preview_width": pw,
            "preview_height": ph,
            "preview_data_uri": data_uri,
            "binding_token": self._token(sample),
            "bbox": bbox,
            "status": status,
            "handled_count": self.handled_count,
            "pass_count": self.pass_count,
            "review_count": self.review_count,
            "final_holdout_locked": True,
        }

    def _persist(self) -> None:
        ordered = []
        for sample in self.samples:
            row = self.annotations.get(sample.sample_id)
            if row is not None:
                ordered.append(row)
        _atomic_write_csv(self.annotation_path, ANNOTATION_COLUMNS, ordered)
        handled = len(ordered)
        if handled in {10, 20, 30}:
            backup = self.annotation_path.with_name(f"bbox_pilot_30.backup_{handled:03d}.csv")
            _atomic_write_csv(backup, ANNOTATION_COLUMNS, ordered)

    def save_from_preview(
        self, *, token: object,
        x0: object, y0: object, x1: object, y1: object,
        preview_width: object, preview_height: object,
    ) -> dict[str, object]:
        sample = self._resolve_token(token)
        x0i = _parse_int(x0, "x0")
        y0i = _parse_int(y0, "y0")
        x1i = _parse_int(x1, "x1")
        y1i = _parse_int(y1, "y1")
        pwi = _parse_int(preview_width, "preview_width")
        phi = _parse_int(preview_height, "preview_height")
        _data, expected_pw, expected_ph = _preview(sample.image_path)
        if (pwi, phi) != (expected_pw, expected_ph):
            _fail("preview dimensions changed; reload sample before save")
        x, y, w, h = _rect_to_original(
            x0=x0i, y0=y0i, x1=x1i, y1=y1i,
            preview_width=pwi, preview_height=phi,
            image_width=sample.image_width, image_height=sample.image_height,
        )
        self.annotations[sample.sample_id] = {
            "sample_id": sample.sample_id,
            "meter": sample.meter,
            "split": "train",
            "x": str(x), "y": str(y), "w": str(w), "h": str(h),
            "status": "PASS",
            "image_sha256": sample.image_sha256,
            "image_width": str(sample.image_width),
            "image_height": str(sample.image_height),
            "updated_utc": _utc_now(),
        }
        self._persist()
        return {
            "sample_id": sample.sample_id,
            "status": "PASS",
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "handled_count": self.handled_count,
            "pass_count": self.pass_count,
            "review_count": self.review_count,
        }

    def mark_review(self, *, token: object) -> dict[str, object]:
        sample = self._resolve_token(token)
        self.annotations[sample.sample_id] = {
            "sample_id": sample.sample_id,
            "meter": sample.meter,
            "split": "train",
            "x": "", "y": "", "w": "", "h": "",
            "status": "REVIEW",
            "image_sha256": sample.image_sha256,
            "image_width": str(sample.image_width),
            "image_height": str(sample.image_height),
            "updated_utc": _utc_now(),
        }
        self._persist()
        return {
            "sample_id": sample.sample_id,
            "status": "REVIEW",
            "handled_count": self.handled_count,
            "pass_count": self.pass_count,
            "review_count": self.review_count,
        }


def write_pilot_audit(data_root: str | Path) -> Path:
    """Write mechanical pilot QA. It never edits human-drawn boxes."""
    session = AnnotationSession(data_root=data_root)
    rows = list(session.annotations.values())
    pass_rows = [r for r in rows if r["status"] == "PASS"]
    review_rows = [r for r in rows if r["status"] == "REVIEW"]
    sample_map = {s.sample_id: s for s in session.samples}

    zero_or_negative = 0
    outside = 0
    suspicious_small: list[str] = []
    suspicious_large: list[str] = []
    widths: list[int] = []
    heights: list[int] = []
    per_class = {meter: {"PASS": 0, "REVIEW": 0} for meter in CLASSES}

    for row in rows:
        per_class[row["meter"]][row["status"]] += 1
        if row["status"] != "PASS":
            continue
        sample = sample_map[row["sample_id"]]
        x, y, w, h = (int(row[k]) for k in ("x", "y", "w", "h"))
        widths.append(w)
        heights.append(h)
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            zero_or_negative += 1
        if x + w > sample.image_width or y + h > sample.image_height:
            outside += 1
        if w < 8 or h < 14:
            suspicious_small.append(sample.sample_id)
        if w > sample.image_width * 0.15 or h > sample.image_height * 0.85:
            suspicious_large.append(sample.sample_id)

    def stats(values: list[int]) -> dict[str, float | int | None]:
        if not values:
            return {"min": None, "median": None, "max": None, "mean": None}
        return {
            "min": min(values),
            "median": float(median(values)),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    mechanical_pass = (
        len(rows) == PILOT_TOTAL
        and len({r["sample_id"] for r in rows}) == PILOT_TOTAL
        and zero_or_negative == 0
        and outside == 0
    )
    freeze_ready = (
        mechanical_pass
        and len(review_rows) == 0
        and not suspicious_small
        and not suspicious_large
    )
    payload = {
        "schema": "st-omr-meter-v5-1-bbox-pilot-audit-v1",
        "dataset": DATASET_NAME,
        "pilot_total_expected": PILOT_TOTAL,
        "annotation_count": len(rows),
        "unique_sample_id": len({r["sample_id"] for r in rows}),
        "pass_count": len(pass_rows),
        "review_count": len(review_rows),
        "per_class": per_class,
        "bbox_width": stats(widths),
        "bbox_height": stats(heights),
        "zero_or_negative_bbox": zero_or_negative,
        "bbox_outside_image": outside,
        "suspicious_too_small_count": len(suspicious_small),
        "suspicious_too_small_sample_ids": suspicious_small,
        "suspicious_too_large_count": len(suspicious_large),
        "suspicious_too_large_sample_ids": suspicious_large,
        "mechanical_gate": "PASS" if mechanical_pass else "HOLD",
        "annotation_contract_freeze_ready": freeze_ready,
        "human_review_resolution_required": bool(review_rows or suspicious_small or suspicious_large),
        "final_holdout_locked": True,
        "training_authorized": False,
        "model_opened": False,
        "inference_count": 0,
        "original_pilot_image_binding_preserved": True,
    }
    path = Path(data_root) / ANNOTATIONS_DIR / PILOT_AUDIT_NAME
    _atomic_write_json(path, payload)
    return path
