"""TRAIN-only human tight-digit annotation pilot for METER V5-1R2.

The module intentionally contains no model/checkpoint code. It selects a fixed
9-sample subset from the already-frozen V5-1 TRAIN pilot, exposes each source
image with its approved full-meter reference box, and persists two independent
human boxes (numerator / denominator). Vertical overlap is explicitly allowed.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Mapping, Sequence

from PIL import Image

from .meter_v5_1_bbox_pilot import (
    ANNOTATIONS_DIR,
    DATASET_NAME,
    PILOT_CSV_NAME,
    PILOT_SELECTION_NAME,
)


V5_SELECTION_SHA256: Final[str] = "4070f46f64efed5b12b26f7dd1d4e3f09b4abf804125140d38a212819bbcbe97"
V5_ANNOTATION_SHA256: Final[str] = "b60a953811aa136752372d7c8cea6fe7a1c1c964a62bd53c0c9d48c56c735665"
CLASSES: Final[tuple[str, ...]] = ("2/4", "3/4", "4/4")
ROLES: Final[tuple[str, ...]] = ("numerator", "denominator")
PER_CLASS: Final[int] = 3
SAMPLE_TOTAL: Final[int] = 9
ROLE_TOTAL: Final[int] = 18
SELECTION_NAME: Final[str] = "digit_bbox_pilot_9_selection.csv"
ANNOTATION_NAME: Final[str] = "digit_bbox_pilot_9.csv"
AUDIT_NAME: Final[str] = "digit_bbox_pilot_9_audit.json"

SELECTION_COLUMNS: Final[tuple[str, ...]] = (
    "index", "source_selection_index", "sample_id", "family_id", "meter", "split",
    "folder", "image_relpath", "image_sha256", "image_width", "image_height",
    "full_x", "full_y", "full_w", "full_h",
)
ANNOTATION_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id", "meter", "role", "x", "y", "w", "h", "status",
    "image_sha256", "image_width", "image_height", "updated_utc",
)


class MeterV5_1R2PilotError(RuntimeError):
    """Fail-closed V5-1R2 annotation error."""


def _fail(message: str) -> None:
    raise MeterV5_1R2PilotError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    raw = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    with tmp.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        _fail(f"required CSV is not a regular file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            _fail(f"CSV header missing: {path}")
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def _as_int(value: str, name: str) -> int:
    try:
        number = int(value, 10)
    except Exception as exc:
        raise MeterV5_1R2PilotError(f"{name} must be integer") from exc
    if str(number) != value.strip():
        _fail(f"{name} must use canonical integer text")
    return number


def _safe_image_path(root: Path, relpath: str) -> Path:
    relative = Path(relpath)
    if relative.is_absolute() or ".." in relative.parts:
        _fail("image path escapes dataset root")
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if not candidate.is_relative_to(resolved_root):
        _fail("image path escapes dataset root")
    if candidate.is_symlink() or not candidate.is_file():
        _fail("pilot image must be a regular non-symlink file")
    return candidate


def _verify_image(path: Path, *, expected_sha: str, width: int, height: int) -> Image.Image:
    if _sha256_file(path) != expected_sha:
        _fail("pilot image SHA-256 changed")
    try:
        with Image.open(path) as opened:
            opened.load()
            if opened.size != (width, height):
                _fail("pilot image dimensions changed")
            return opened.copy()
    except Exception as exc:
        raise MeterV5_1R2PilotError("pilot image cannot be decoded") from exc


def _frozen_parent_rows(root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    if root.name != DATASET_NAME or not root.is_dir():
        _fail("V5-1R2 requires exact clean V5 dataset root")
    annotations = root / ANNOTATIONS_DIR
    selection_path = annotations / PILOT_SELECTION_NAME
    annotation_path = annotations / PILOT_CSV_NAME
    if _sha256_file(selection_path) != V5_SELECTION_SHA256:
        _fail("frozen V5-1 selection CSV SHA mismatch")
    if _sha256_file(annotation_path) != V5_ANNOTATION_SHA256:
        _fail("frozen V5-1 annotation CSV SHA mismatch")
    selection = _read_csv(selection_path)
    annotations_rows = _read_csv(annotation_path)
    if len(selection) != 30 or len(annotations_rows) != 30:
        _fail("frozen V5-1 parent cardinality changed")
    by_id = {row["sample_id"]: row for row in annotations_rows}
    if len(by_id) != 30:
        _fail("frozen V5-1 annotation sample IDs are not unique")
    for row in selection:
        ann = by_id.get(row["sample_id"])
        if ann is None or ann.get("status") != "PASS" or row.get("split") != "train":
            _fail("V5-1R2 parent requires 30 TRAIN PASS samples")
        if row.get("meter") not in CLASSES or ann.get("meter") != row.get("meter"):
            _fail("parent meter binding mismatch")
        for field in ("image_sha256", "image_width", "image_height"):
            if ann.get(field) != row.get(field):
                _fail("parent image binding mismatch")
    return selection, by_id


def load_or_create_tight_selection(data_root: str | Path) -> tuple[dict[str, str], ...]:
    root = Path(data_root)
    parent_selection, parent_annotations = _frozen_parent_rows(root)
    annotations_dir = root / ANNOTATIONS_DIR
    path = annotations_dir / SELECTION_NAME

    expected_rows: list[dict[str, object]] = []
    out_index = 0
    for meter in CLASSES:
        candidates = [row for row in parent_selection if row["meter"] == meter]
        chosen = candidates[:PER_CLASS]
        if len(chosen) != PER_CLASS:
            _fail(f"{meter}: frozen parent does not contain three pilot samples")
        for row in chosen:
            ann = parent_annotations[row["sample_id"]]
            image_width = _as_int(row["image_width"], "image_width")
            image_height = _as_int(row["image_height"], "image_height")
            image_path = _safe_image_path(root, row["image_relpath"])
            _verify_image(image_path, expected_sha=row["image_sha256"], width=image_width, height=image_height)
            expected_rows.append({
                "index": out_index,
                "source_selection_index": row["index"],
                "sample_id": row["sample_id"],
                "family_id": row["family_id"],
                "meter": meter,
                "split": "train",
                "folder": row["folder"],
                "image_relpath": row["image_relpath"],
                "image_sha256": row["image_sha256"],
                "image_width": image_width,
                "image_height": image_height,
                "full_x": _as_int(ann["x"], "full_x"),
                "full_y": _as_int(ann["y"], "full_y"),
                "full_w": _as_int(ann["w"], "full_w"),
                "full_h": _as_int(ann["h"], "full_h"),
            })
            out_index += 1

    if len(expected_rows) != SAMPLE_TOTAL:
        _fail("tight-digit selection must contain exactly 9 rows")

    if path.exists():
        existing = _read_csv(path)
        normalized_expected = [
            {name: str(row[name]) for name in SELECTION_COLUMNS}
            for row in expected_rows
        ]
        if existing != normalized_expected:
            _fail("existing tight-digit selection differs from frozen derivation")
    else:
        _atomic_write_csv(path, SELECTION_COLUMNS, expected_rows)
    return tuple(_read_csv(path))


def _annotation_path(root: Path) -> Path:
    return root / ANNOTATIONS_DIR / ANNOTATION_NAME


def _load_annotations(root: Path) -> list[dict[str, str]]:
    path = _annotation_path(root)
    if not path.exists():
        return []
    rows = _read_csv(path)
    if rows and set(rows[0]) != set(ANNOTATION_COLUMNS):
        _fail("tight-digit annotation schema mismatch")
    keys = [(row["sample_id"], row["role"]) for row in rows]
    if len(keys) != len(set(keys)):
        _fail("duplicate tight-digit annotation role row")
    return rows


def _write_annotations(root: Path, rows: list[dict[str, object]]) -> None:
    ordered = sorted(rows, key=lambda row: (str(row["sample_id"]), ROLES.index(str(row["role"]))))
    _atomic_write_csv(_annotation_path(root), ANNOTATION_COLUMNS, ordered)


def _validate_box(
    *,
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    full_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    if any(type(v) is not int for v in (x, y, w, h)):
        _fail("tight box coordinates must be integers")
    if w <= 1 or h <= 1:
        _fail("tight box must have positive non-trivial area")
    if not (0 <= x < x + w <= image_width and 0 <= y < y + h <= image_height):
        _fail("tight box lies outside source image")
    fx, fy, fw, fh = full_box
    if not (fx <= x and fy <= y and x + w <= fx + fw and y + h <= fy + fh):
        _fail("tight box must remain inside approved full-meter bbox")
    return box


def _preview_box_to_original(
    *,
    x0: object, y0: object, x1: object, y1: object,
    preview_width: object, preview_height: object,
    image_width: int, image_height: int,
) -> tuple[int, int, int, int]:
    values = (x0, y0, x1, y1, preview_width, preview_height)
    if any(type(v) is not int for v in values):
        _fail("preview geometry must be integer")
    assert isinstance(preview_width, int) and isinstance(preview_height, int)
    if preview_width <= 0 or preview_height <= 0:
        _fail("preview dimensions invalid")
    left, right = sorted((int(x0), int(x1)))
    top, bottom = sorted((int(y0), int(y1)))
    if not (0 <= left < right <= preview_width and 0 <= top < bottom <= preview_height):
        _fail("preview box invalid")
    ox0 = (left * image_width) // preview_width
    oy0 = (top * image_height) // preview_height
    ox1 = ((right * image_width) + preview_width - 1) // preview_width
    oy1 = ((bottom * image_height) + preview_height - 1) // preview_height
    ox0 = max(0, min(image_width - 1, ox0))
    oy0 = max(0, min(image_height - 1, oy0))
    ox1 = max(ox0 + 1, min(image_width, ox1))
    oy1 = max(oy0 + 1, min(image_height, oy1))
    return ox0, oy0, ox1 - ox0, oy1 - oy0


def _binding_token(row: Mapping[str, str]) -> str:
    raw = "|".join([
        "meter-v5-1r2-tight-digit-v1",
        row["sample_id"], row["meter"], row["image_sha256"],
        row["image_width"], row["image_height"],
        row["full_x"], row["full_y"], row["full_w"], row["full_h"],
    ]).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class TightDigitAnnotationSession:
    data_root: Path
    selection: tuple[dict[str, str], ...]

    def __init__(self, *, data_root: str | Path):
        self.data_root = Path(data_root)
        self.selection = load_or_create_tight_selection(self.data_root)

    def _row(self, index: int) -> dict[str, str]:
        if type(index) is not int or not 0 <= index < SAMPLE_TOTAL:
            _fail("tight-digit sample index out of range")
        return self.selection[index]

    def _annotation_map(self) -> dict[tuple[str, str], dict[str, str]]:
        return {(row["sample_id"], row["role"]): row for row in _load_annotations(self.data_root)}

    def resume_index(self) -> int:
        amap = self._annotation_map()
        for index, row in enumerate(self.selection):
            role_rows = [amap.get((row["sample_id"], role)) for role in ROLES]
            if not all(role_rows) or any(item is None or item.get("status") not in {"PASS", "REVIEW"} for item in role_rows):
                return index
        return SAMPLE_TOTAL - 1

    def sample_payload(self, index: int) -> dict[str, object]:
        row = self._row(index)
        image_width = _as_int(row["image_width"], "image_width")
        image_height = _as_int(row["image_height"], "image_height")
        path = _safe_image_path(self.data_root, row["image_relpath"])
        image = _verify_image(path, expected_sha=row["image_sha256"], width=image_width, height=image_height)
        max_preview_width = 1100
        scale = min(1.0, max_preview_width / image_width)
        preview_width = max(1, int(round(image_width * scale)))
        preview_height = max(1, int(round(image_height * scale)))
        preview = image.copy()
        if preview.size != (preview_width, preview_height):
            preview = preview.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        preview.save(buffer, format="PNG", optimize=False, compress_level=9)
        data_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        amap = self._annotation_map()
        roles: dict[str, object] = {}
        for role in ROLES:
            existing = amap.get((row["sample_id"], role))
            roles[role] = None if existing is None else {
                "status": existing["status"],
                "x": int(existing["x"]) if existing["x"] else None,
                "y": int(existing["y"]) if existing["y"] else None,
                "w": int(existing["w"]) if existing["w"] else None,
                "h": int(existing["h"]) if existing["h"] else None,
            }
        handled = 0
        pass_samples = 0
        review_samples = 0
        for sample in self.selection:
            pair = [amap.get((sample["sample_id"], role)) for role in ROLES]
            if all(pair):
                handled += 1
                if all(item is not None and item.get("status") == "PASS" for item in pair):
                    pass_samples += 1
                elif all(item is not None and item.get("status") == "REVIEW" for item in pair):
                    review_samples += 1
        return {
            "index": index,
            "sample_id": row["sample_id"],
            "meter": row["meter"],
            "split": "train",
            "image_width": image_width,
            "image_height": image_height,
            "preview_width": preview_width,
            "preview_height": preview_height,
            "preview_data_uri": data_uri,
            "full_bbox": {
                "x": _as_int(row["full_x"], "full_x"),
                "y": _as_int(row["full_y"], "full_y"),
                "w": _as_int(row["full_w"], "full_w"),
                "h": _as_int(row["full_h"], "full_h"),
            },
            "roles": roles,
            "binding_token": _binding_token(row),
            "handled_count": handled,
            "pass_sample_count": pass_samples,
            "review_sample_count": review_samples,
        }

    def save_from_preview(
        self,
        *,
        token: object,
        numerator: Mapping[str, object],
        denominator: Mapping[str, object],
        preview_width: object,
        preview_height: object,
    ) -> dict[str, object]:
        if not isinstance(token, str):
            _fail("binding token must be string")
        row = next((item for item in self.selection if _binding_token(item) == token), None)
        if row is None:
            _fail("binding token does not match frozen selection")
        if not isinstance(numerator, Mapping) or not isinstance(denominator, Mapping):
            _fail("both role boxes are required")
        image_width = _as_int(row["image_width"], "image_width")
        image_height = _as_int(row["image_height"], "image_height")
        path = _safe_image_path(self.data_root, row["image_relpath"])
        _verify_image(path, expected_sha=row["image_sha256"], width=image_width, height=image_height)
        full_box = (
            _as_int(row["full_x"], "full_x"), _as_int(row["full_y"], "full_y"),
            _as_int(row["full_w"], "full_w"), _as_int(row["full_h"], "full_h"),
        )

        converted: dict[str, tuple[int, int, int, int]] = {}
        for role, payload in (("numerator", numerator), ("denominator", denominator)):
            if set(payload) != {"x0", "y0", "x1", "y1"}:
                _fail(f"{role} preview payload malformed")
            box = _preview_box_to_original(
                x0=payload["x0"], y0=payload["y0"], x1=payload["x1"], y1=payload["y1"],
                preview_width=preview_width, preview_height=preview_height,
                image_width=image_width, image_height=image_height,
            )
            converted[role] = _validate_box(
                box=box, image_width=image_width, image_height=image_height, full_box=full_box
            )
        num = converted["numerator"]
        den = converted["denominator"]
        if num[1] + num[3] / 2.0 >= den[1] + den[3] / 2.0:
            _fail("numerator center must be above denominator center")

        existing = _load_annotations(self.data_root)
        kept = [item for item in existing if item["sample_id"] != row["sample_id"]]
        now = _utc_now()
        for role in ROLES:
            x, y, w, h = converted[role]
            kept.append({
                "sample_id": row["sample_id"], "meter": row["meter"], "role": role,
                "x": x, "y": y, "w": w, "h": h, "status": "PASS",
                "image_sha256": row["image_sha256"], "image_width": image_width,
                "image_height": image_height, "updated_utc": now,
            })
        _write_annotations(self.data_root, kept)
        return self.sample_payload(_as_int(row["index"], "index"))

    def mark_review(self, *, token: object) -> dict[str, object]:
        if not isinstance(token, str):
            _fail("binding token must be string")
        row = next((item for item in self.selection if _binding_token(item) == token), None)
        if row is None:
            _fail("binding token does not match frozen selection")
        existing = _load_annotations(self.data_root)
        kept = [item for item in existing if item["sample_id"] != row["sample_id"]]
        now = _utc_now()
        for role in ROLES:
            kept.append({
                "sample_id": row["sample_id"], "meter": row["meter"], "role": role,
                "x": "", "y": "", "w": "", "h": "", "status": "REVIEW",
                "image_sha256": row["image_sha256"], "image_width": row["image_width"],
                "image_height": row["image_height"], "updated_utc": now,
            })
        _write_annotations(self.data_root, kept)
        return self.sample_payload(_as_int(row["index"], "index"))


def audit_tight_digit_pilot(data_root: str | Path) -> dict[str, object]:
    root = Path(data_root)
    selection = load_or_create_tight_selection(root)
    annotations = _load_annotations(root)
    if len(selection) != SAMPLE_TOTAL:
        _fail("tight-digit selection cardinality changed")
    by_key = {(row["sample_id"], row["role"]): row for row in annotations}
    invalid = 0
    pass_rows = 0
    review_rows = 0
    overlap_samples = 0
    overlap_area_total = 0
    per_meter_samples = {meter: 0 for meter in CLASSES}
    complete_samples = 0

    for selected in selection:
        per_meter_samples[selected["meter"]] += 1
        image_width = _as_int(selected["image_width"], "image_width")
        image_height = _as_int(selected["image_height"], "image_height")
        path = _safe_image_path(root, selected["image_relpath"])
        _verify_image(path, expected_sha=selected["image_sha256"], width=image_width, height=image_height)
        full_box = (
            _as_int(selected["full_x"], "full_x"), _as_int(selected["full_y"], "full_y"),
            _as_int(selected["full_w"], "full_w"), _as_int(selected["full_h"], "full_h"),
        )
        pair = [by_key.get((selected["sample_id"], role)) for role in ROLES]
        if not all(pair):
            continue
        complete_samples += 1
        boxes: dict[str, tuple[int, int, int, int]] = {}
        for role, item in zip(ROLES, pair):
            assert item is not None
            if item["status"] == "REVIEW":
                review_rows += 1
                continue
            if item["status"] != "PASS":
                invalid += 1
                continue
            try:
                box = (
                    _as_int(item["x"], "x"), _as_int(item["y"], "y"),
                    _as_int(item["w"], "w"), _as_int(item["h"], "h"),
                )
                _validate_box(box=box, image_width=image_width, image_height=image_height, full_box=full_box)
                boxes[role] = box
                pass_rows += 1
            except MeterV5_1R2PilotError:
                invalid += 1
        if set(boxes) == set(ROLES):
            num = boxes["numerator"]
            den = boxes["denominator"]
            if num[1] + num[3] / 2.0 >= den[1] + den[3] / 2.0:
                invalid += 1
            ix0 = max(num[0], den[0])
            iy0 = max(num[1], den[1])
            ix1 = min(num[0] + num[2], den[0] + den[2])
            iy1 = min(num[1] + num[3], den[1] + den[3])
            if ix0 < ix1 and iy0 < iy1:
                overlap_samples += 1
                overlap_area_total += (ix1 - ix0) * (iy1 - iy0)

    ready = (
        complete_samples == SAMPLE_TOTAL
        and len(annotations) == ROLE_TOTAL
        and pass_rows == ROLE_TOTAL
        and review_rows == 0
        and invalid == 0
        and per_meter_samples == {meter: PER_CLASS for meter in CLASSES}
    )
    result = {
        "schema": "st-omr-meter-v5-1r2-tight-digit-pilot-audit-v1",
        "stage": "METER V5-1R2",
        "selection_count": len(selection),
        "annotation_row_count": len(annotations),
        "complete_sample_count": complete_samples,
        "pass_row_count": pass_rows,
        "review_row_count": review_rows,
        "invalid_row_count": invalid,
        "per_meter_samples": per_meter_samples,
        "overlap_sample_count": overlap_samples,
        "overlap_area_total": overlap_area_total,
        "overlap_is_allowed": True,
        "annotation_contract_ready": ready,
        "safety": {
            "split": "train",
            "validation_opened": False,
            "final_holdout_opened": False,
            "model_inference": False,
            "optimizer_steps": 0,
            "threshold_tuning": False,
            "resolver_connected": False,
            "production_promotion_authorized": False,
        },
    }
    _atomic_write_json(root / ANNOTATIONS_DIR / AUDIT_NAME, result)
    return result


def model_inference_allowed_during_annotation() -> bool:
    return False


def validation_or_final_holdout_access_allowed() -> bool:
    return False


def training_authorized() -> bool:
    return False
