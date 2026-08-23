"""Meter V5-2B: deterministic V5/package_ab adaptation lane for 2-AI and 3-AI.

The lane is deliberately narrow:
- bind the completed V5-2A 300-BBox evidence to an explicit human-QA attestation;
- derive approved staff-relative numerator/denominator slots;
- keep the first 30 immutable seeds diagnostic-only;
- fine-tune only historical 2-AI and 3-AI on the remaining 270 TRAIN samples;
- keep 4-AI frozen and all thresholds unchanged;
- evaluate candidates on the 30 diagnostic seeds before any VAL stage may open.

VAL, FINAL_HOLDOUT, Resolver and production promotion are never accessed here.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final, Mapping

from PIL import Image

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2a_specialist_adaptation as v52a
from .runtime_geometry_engine_contract import GeometryInputContract
from .runtime_geometry_engine_v2 import detect_multistaff_geometry_v2
from .runtime_page_normalizer_contract import RasterPageInputContract
from .runtime_page_normalizer_v1 import normalize_raster_page_v1


V5_2B_SCHEMA: Final[str] = "st-omr-meter-v5-2b-specialist-adaptation-v1"
HUMAN_QA_SCHEMA: Final[str] = "st-omr-meter-v5-2b-human-qa-attestation-v1"
SLOT_MANIFEST_SCHEMA: Final[str] = "st-omr-meter-v5-2b-slot-manifest-v1"
TRAINING_SCHEMA: Final[str] = "st-omr-meter-v5-2b-training-v1"
DIAGNOSTIC_SCHEMA: Final[str] = "st-omr-meter-v5-2b-diagnostic-gate-v1"

HUMAN_QA_CONFIRMATION: Final[str] = "V5_2A_300_CONTACT_SHEETS_15_OF_15_PASS"
HUMAN_QA_NAME: Final[str] = "bbox_adaptation_300_human_qa.json"
SLOT_MANIFEST_NAME: Final[str] = "v5_2b_slot_manifest.csv"
SLOT_AUDIT_NAME: Final[str] = "v5_2b_slot_audit.json"
TRAINING_REPORT_NAME: Final[str] = "v5_2b_training_report.json"
DIAGNOSTIC_REPORT_NAME: Final[str] = "v5_2b_diagnostic_gate.json"

DIGIT2_SHA256: Final[str] = "92b985d989e4338e3ae39b0a984879f4188be32c0d281390839117e1e9a715fa"
DIGIT3_SHA256: Final[str] = "5ee45faf2efe0e2c83dbad716736d7ae16ad7251730431d368c10c4574836485"
DIGIT4_SHA256: Final[str] = "dcd582b60b39e65798aa77aacea3cc797cd7513b7925151f0573be4aec6af43f"

FROZEN_THRESHOLDS: Final[dict[str, float]] = {"2": 0.48, "3": 0.60, "4": 0.47}
WIDTH_OVER_STAFF_SPACING: Final[float] = 1.5960569245912566
HEIGHT_OVER_STAFF_SPACING: Final[float] = 2.0
NUMERATOR_LINE_INDEX: Final[int] = 1
DENOMINATOR_LINE_INDEX: Final[int] = 3
DIAGNOSTIC_SEED_TOTAL: Final[int] = 30
ADAPTATION_TRAIN_TOTAL: Final[int] = 270

SLOT_COLUMNS: Final[tuple[str, ...]] = (
    "schema",
    "sample_index",
    "sample_id",
    "family_id",
    "meter",
    "data_role",
    "slot_role",
    "expected_digit",
    "label_digit2",
    "label_digit3",
    "source_image_sha256",
    "runtime_raster_sha256",
    "pixel_mode_adapter",
    "normalized_image_sha256",
    "staff_spacing",
    "x_anchor_normalized",
    "slot_x",
    "slot_y",
    "slot_w",
    "slot_h",
    "crop_relpath",
    "crop_sha256",
)


class MeterV5_2BError(RuntimeError):
    """Raised whenever the V5-2B evidence or training boundary fails closed."""


def _fail(message: str) -> None:
    raise MeterV5_2BError(message)


def _sha_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        _fail(f"expected regular non-symlink file: {path}")
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        _fail(f"missing JSON evidence: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeterV5_2BError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        _fail(f"JSON evidence must be an object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        _fail(f"missing CSV evidence: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _canonical_json_sha(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256(raw).hexdigest()


def _require_mechanical_audit(root: Path) -> tuple[Path, dict[str, object]]:
    path = root / v51.ANNOTATIONS_DIR / v52a.AUDIT_NAME
    audit = _read_json(path)
    expected_scalars = {
        "schema": "st-omr-meter-v5-2a-annotation-audit-v1",
        "dataset": v51.DATASET_NAME,
        "annotation_count": 300,
        "pass_count": 300,
        "review_count": 0,
        "missing_annotation_count": 0,
        "seed_mutation_count": 0,
        "mechanical_gate": "PASS",
        "training_authorized": False,
        "threshold_tuning_allowed": False,
        "single_three_class_model_allowed": False,
        "validation_opened": False,
        "final_holdout_locked": True,
        "model_opened": False,
        "inference_count": 0,
        "frozen_control_specialist": "4-AI",
    }
    for key, expected in expected_scalars.items():
        if audit.get(key) != expected:
            _fail(f"V5-2A mechanical audit field changed: {key}")
    if audit.get("trainable_specialists") != ["2-AI", "3-AI"]:
        _fail("V5-2A trainable specialist set changed")
    per_class = audit.get("per_class")
    if per_class != {
        "2/4": {"PASS": 100, "REVIEW": 0},
        "3/4": {"PASS": 100, "REVIEW": 0},
        "4/4": {"PASS": 100, "REVIEW": 0},
    }:
        _fail("V5-2A per-class audit is not 100/100/100 PASS")
    return path, audit


def write_human_qa_attestation(
    data_root: str | Path,
    *,
    confirmation: str,
) -> Path:
    """Bind the user's completed 15/15 contact-sheet review to exact evidence."""
    if confirmation != HUMAN_QA_CONFIRMATION:
        _fail("human QA confirmation token is not the frozen 15/15 PASS statement")
    root = Path(data_root)
    session = v52a.AdaptationAnnotationSession(data_root=root)
    if session.handled_count != 300 or session.pass_count != 300 or session.review_count != 0:
        _fail("human QA attestation requires the completed 300/300 annotation set")
    audit_path, _audit = _require_mechanical_audit(root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    selection_path = ann_dir / v52a.SELECTION_NAME
    annotation_path = ann_dir / v52a.ANNOTATION_NAME
    payload = {
        "schema": HUMAN_QA_SCHEMA,
        "dataset": v51.DATASET_NAME,
        "dataset_fingerprint_sha256": v52a.EXPECTED_DATASET_FINGERPRINT,
        "selection_sha256": _sha_file(selection_path),
        "annotation_sha256": _sha_file(annotation_path),
        "mechanical_audit_sha256": _sha_file(audit_path),
        "contact_sheets_reviewed": 15,
        "contact_sheet_visual_errors_reported": 0,
        "human_visual_qa": "PASS",
        "slot_derivation_authorized": True,
        "adaptation_training_boundary_authorized": True,
        "trainable_specialists": ["2-AI", "3-AI"],
        "frozen_control_specialist": "4-AI",
        "threshold_tuning_allowed": False,
        "validation_opened": False,
        "final_holdout_locked": True,
    }
    path = ann_dir / HUMAN_QA_NAME
    v51._atomic_write_json(path, payload)
    return path


def verify_human_qa_attestation(data_root: str | Path) -> dict[str, object]:
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    path = ann_dir / HUMAN_QA_NAME
    payload = _read_json(path)
    audit_path, _audit = _require_mechanical_audit(root)
    expected = {
        "schema": HUMAN_QA_SCHEMA,
        "dataset": v51.DATASET_NAME,
        "dataset_fingerprint_sha256": v52a.EXPECTED_DATASET_FINGERPRINT,
        "selection_sha256": _sha_file(ann_dir / v52a.SELECTION_NAME),
        "annotation_sha256": _sha_file(ann_dir / v52a.ANNOTATION_NAME),
        "mechanical_audit_sha256": _sha_file(audit_path),
        "contact_sheets_reviewed": 15,
        "contact_sheet_visual_errors_reported": 0,
        "human_visual_qa": "PASS",
        "slot_derivation_authorized": True,
        "adaptation_training_boundary_authorized": True,
        "trainable_specialists": ["2-AI", "3-AI"],
        "frozen_control_specialist": "4-AI",
        "threshold_tuning_allowed": False,
        "validation_opened": False,
        "final_holdout_locked": True,
    }
    if payload != expected:
        _fail("human QA attestation no longer matches exact V5-2A evidence")
    return payload


def _prepare_runtime_raster(raw: bytes) -> tuple[bytes, str, int, int, str]:
    """TRAIN-only deterministic palette compatibility with explicit provenance."""
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        mode = image.mode
        width, height = image.size
        orientation = int(image.getexif().get(274, 1))
        if mode in {"L", "RGB", "RGBA"}:
            return raw, mode, width, height, "IDENTITY"
        if mode == "P":
            if orientation != 1:
                _fail("palette source with non-default EXIF orientation is not adapted")
            converted = image.convert("RGB")
            output = io.BytesIO()
            converted.save(output, format="PNG", optimize=False, compress_level=9)
            return output.getvalue(), "RGB", width, height, "P_TO_RGB_TRAINING_V1"
    _fail(f"unsupported source pixel mode: {mode}")


def _mode_to_contract(mode: str) -> str:
    mapping = {"L": "gray8", "RGB": "rgb8", "RGBA": "rgba8"}
    try:
        return mapping[mode]
    except KeyError as exc:
        raise MeterV5_2BError(f"unsupported prepared pixel mode: {mode}") from exc


def _slot_edges(cx: float, cy: float, spacing: float) -> tuple[float, float, float, float]:
    width = WIDTH_OVER_STAFF_SPACING * spacing
    height = HEIGHT_OVER_STAFF_SPACING * spacing
    return cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0


def _integer_crop_box(
    edges: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = edges
    if not (0.0 <= x0 < x1 <= image_width and 0.0 <= y0 < y1 <= image_height):
        _fail("approved staff-relative slot extends outside normalized image")
    ix0, iy0 = math.floor(x0), math.floor(y0)
    ix1, iy1 = math.ceil(x1), math.ceil(y1)
    if not (0 <= ix0 < ix1 <= image_width and 0 <= iy0 < iy1 <= image_height):
        _fail("integerized approved slot extends outside normalized image")
    return ix0, iy0, ix1, iy1


def _historical_canvas(crop: Image.Image) -> Image.Image:
    gray = crop.convert("L")
    thumb = gray.copy()
    thumb.thumbnail((64, 64), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (64, 64), 255)
    x = (64 - thumb.width) // 2
    y = (64 - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    return canvas


def _encode_png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def derive_staff_relative_slots_v1(data_root: str | Path) -> Path:
    """Derive 600 immutable 64x64 specialist inputs from the approved 300 BBoxes."""
    root = Path(data_root)
    verify_human_qa_attestation(root)
    session = v52a.AdaptationAnnotationSession(data_root=root)
    if len(session.samples) != 300 or session.pass_count != 300:
        _fail("slot derivation requires 300/300 PASS annotations")
    if not all(sample.seed_annotation for sample in session.samples[:30]):
        _fail("first 30 V5-2B samples must remain immutable diagnostic seeds")
    if any(sample.seed_annotation for sample in session.samples[30:]):
        _fail("only the first 30 V5-2B samples may be diagnostic seeds")

    ann_dir = root / v51.ANNOTATIONS_DIR
    qa_sha = _sha_file(ann_dir / HUMAN_QA_NAME)
    annotation_sha = _sha_file(ann_dir / v52a.ANNOTATION_NAME)
    selection_sha = _sha_file(ann_dir / v52a.SELECTION_NAME)
    derivation_fp = _canonical_json_sha({
        "schema": SLOT_MANIFEST_SCHEMA,
        "qa_sha256": qa_sha,
        "annotation_sha256": annotation_sha,
        "selection_sha256": selection_sha,
        "width_over_staff_spacing": WIDTH_OVER_STAFF_SPACING,
        "height_over_staff_spacing": HEIGHT_OVER_STAFF_SPACING,
        "numerator_line_index": NUMERATOR_LINE_INDEX,
        "denominator_line_index": DENOMINATOR_LINE_INDEX,
        "preprocessing": "gray-L;thumbnail-64-LANCZOS-no-upscale;center-white-64",
    })
    crop_root = ann_dir / f"v5_2b_slot_crops_{derivation_fp[:16]}"
    crop_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    adapter_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    meter_counts: Counter[str] = Counter()

    for sample in session.samples:
        annotation = session.annotations[sample.sample_id]
        if annotation["status"] != "PASS":
            _fail(f"non-PASS annotation reached slot derivation: {sample.sample_id}")
        raw = sample.image_path.read_bytes()
        if sha256(raw).hexdigest() != sample.image_sha256:
            _fail(f"source image SHA changed: {sample.sample_id}")
        runtime_raw, mode, width, height, adapter = _prepare_runtime_raster(raw)
        runtime_sha = sha256(runtime_raw).hexdigest()
        raster = RasterPageInputContract(
            source_id=sample.sample_id,
            source_sha256=sample.image_sha256,
            page_number=1,
            width=width,
            height=height,
            pixel_mode=_mode_to_contract(mode),
            raster_sha256=runtime_sha,
            dpi=None,
        )
        normalized = normalize_raster_page_v1(runtime_raw, raster)
        if normalized.page.status != "accepted" or normalized.normalized_png is None:
            _fail(f"normalizer HOLD during V5-2B derivation: {sample.sample_id}")
        page = normalized.page
        geometry_input = GeometryInputContract(
            normalized_image_sha256=page.normalized_image_sha256,
            normalizer_config_fingerprint=page.normalizer_config_fingerprint,
            normalized_width=page.normalized_width,
            normalized_height=page.normalized_height,
            transform=page.transform,
        )
        geometry = detect_multistaff_geometry_v2(normalized.normalized_png, geometry_input)
        if geometry.page.status != "accepted" or len(geometry.page.staffs) != 1:
            _fail(f"V5-2B requires exactly one accepted staff: {sample.sample_id}")
        staff = geometry.page.staffs[0]
        line_y = tuple(
            (line.start.y + line.end.y) / 2.0 for line in staff.five_staff_lines
        )
        x = int(annotation["x"])
        y = int(annotation["y"])
        w = int(annotation["w"])
        h = int(annotation["h"])
        center_x_original = x + w / 2.0
        center_y_original = y + h / 2.0
        center_x_normalized, _center_y_normalized = page.transform.original_to_normalized(
            center_x_original, center_y_original
        )
        spacing = float(staff.staff_spacing)
        boxes = {
            "numerator": _integer_crop_box(
                _slot_edges(center_x_normalized, line_y[NUMERATOR_LINE_INDEX], spacing),
                image_width=page.normalized_width,
                image_height=page.normalized_height,
            ),
            "denominator": _integer_crop_box(
                _slot_edges(center_x_normalized, line_y[DENOMINATOR_LINE_INDEX], spacing),
                image_width=page.normalized_width,
                image_height=page.normalized_height,
            ),
        }
        data_role = "diagnostic_seed" if sample.index < DIAGNOSTIC_SEED_TOTAL else "adaptation_train"
        expected_numerator = sample.meter.split("/")[0]
        with Image.open(io.BytesIO(normalized.normalized_png)) as image:
            image.load()
            gray = image.convert("L")
            for slot_role, box in boxes.items():
                ix0, iy0, ix1, iy1 = box
                crop = _historical_canvas(gray.crop(box))
                encoded = _encode_png(crop)
                relative = Path(f"v5_2b_slot_crops_{derivation_fp[:16]}") / data_role / f"{sample.sample_id}_{slot_role}.png"
                target = ann_dir / relative
                _atomic_bytes(target, encoded)
                expected_digit = expected_numerator if slot_role == "numerator" else "4"
                label2 = int(slot_role == "numerator" and expected_numerator == "2")
                label3 = int(slot_role == "numerator" and expected_numerator == "3")
                rows.append({
                    "schema": SLOT_MANIFEST_SCHEMA,
                    "sample_index": str(sample.index),
                    "sample_id": sample.sample_id,
                    "family_id": sample.family_id,
                    "meter": sample.meter,
                    "data_role": data_role,
                    "slot_role": slot_role,
                    "expected_digit": expected_digit,
                    "label_digit2": str(label2),
                    "label_digit3": str(label3),
                    "source_image_sha256": sample.image_sha256,
                    "runtime_raster_sha256": runtime_sha,
                    "pixel_mode_adapter": adapter,
                    "normalized_image_sha256": page.normalized_image_sha256,
                    "staff_spacing": f"{spacing:.12f}",
                    "x_anchor_normalized": f"{center_x_normalized:.12f}",
                    "slot_x": str(ix0),
                    "slot_y": str(iy0),
                    "slot_w": str(ix1 - ix0),
                    "slot_h": str(iy1 - iy0),
                    "crop_relpath": relative.as_posix(),
                    "crop_sha256": sha256(encoded).hexdigest(),
                })
        adapter_counts[adapter] += 1
        role_counts[data_role] += 1
        meter_counts[f"{data_role}:{sample.meter}"] += 1

    if len(rows) != 600:
        _fail(f"V5-2B slot manifest must contain 600 rows, got {len(rows)}")
    if role_counts != Counter({"adaptation_train": 270, "diagnostic_seed": 30}):
        _fail(f"V5-2B sample roles changed: {dict(role_counts)}")
    expected_meter_counts = Counter({
        "diagnostic_seed:2/4": 10,
        "diagnostic_seed:3/4": 10,
        "diagnostic_seed:4/4": 10,
        "adaptation_train:2/4": 90,
        "adaptation_train:3/4": 90,
        "adaptation_train:4/4": 90,
    })
    if meter_counts != expected_meter_counts:
        _fail(f"V5-2B role/class balance changed: {dict(meter_counts)}")
    train_rows = [row for row in rows if row["data_role"] == "adaptation_train"]
    if sum(int(row["label_digit2"]) for row in train_rows) != 90:
        _fail("2-AI adaptation positives must equal 90")
    if sum(int(row["label_digit3"]) for row in train_rows) != 90:
        _fail("3-AI adaptation positives must equal 90")

    manifest_path = ann_dir / SLOT_MANIFEST_NAME
    v51._atomic_write_csv(manifest_path, SLOT_COLUMNS, rows)
    audit = {
        "schema": SLOT_MANIFEST_SCHEMA,
        "derivation_fingerprint": derivation_fp,
        "human_qa_sha256": qa_sha,
        "selection_sha256": selection_sha,
        "annotation_sha256": annotation_sha,
        "manifest_sha256": _sha_file(manifest_path),
        "slot_count": 600,
        "diagnostic_seed_samples": 30,
        "adaptation_train_samples": 270,
        "adaptation_train_slots": 540,
        "per_role_meter": dict(sorted(meter_counts.items())),
        "pixel_mode_adapter_counts_per_source": dict(sorted(adapter_counts.items())),
        "digit2_train_positive": 90,
        "digit2_train_negative": 450,
        "digit3_train_positive": 90,
        "digit3_train_negative": 450,
        "midpoint_derivation": False,
        "tight_digit_gt": False,
        "model_generated_gt": False,
        "automatic_bbox_correction": False,
        "training_authorized": True,
        "trainable_specialists": ["2-AI", "3-AI"],
        "frozen_control_specialist": "4-AI",
        "threshold_tuning_allowed": False,
        "validation_opened": False,
        "final_holdout_locked": True,
    }
    v51._atomic_write_json(ann_dir / SLOT_AUDIT_NAME, audit)
    return manifest_path


def verify_slot_manifest_v1(data_root: str | Path) -> tuple[Path, list[dict[str, str]], dict[str, object]]:
    root = Path(data_root)
    verify_human_qa_attestation(root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    manifest_path = ann_dir / SLOT_MANIFEST_NAME
    audit = _read_json(ann_dir / SLOT_AUDIT_NAME)
    rows = _read_csv(manifest_path)
    if len(rows) != 600:
        _fail("V5-2B slot manifest is not exactly 600 rows")
    if audit.get("manifest_sha256") != _sha_file(manifest_path):
        _fail("V5-2B slot manifest SHA no longer matches its audit")
    if audit.get("training_authorized") is not True:
        _fail("V5-2B slot audit does not authorize the narrow 2/3 training lane")
    if audit.get("trainable_specialists") != ["2-AI", "3-AI"]:
        _fail("V5-2B slot audit trainable surface changed")
    if audit.get("frozen_control_specialist") != "4-AI":
        _fail("V5-2B slot audit no longer freezes 4-AI")
    if audit.get("threshold_tuning_allowed") is not False:
        _fail("threshold tuning must remain disabled")
    if audit.get("validation_opened") is not False or audit.get("final_holdout_locked") is not True:
        _fail("VAL/final-holdout boundary changed")
    for row in rows:
        if row.get("schema") != SLOT_MANIFEST_SCHEMA:
            _fail("slot manifest schema mismatch")
        crop = ann_dir / row["crop_relpath"]
        if _sha_file(crop) != row["crop_sha256"]:
            _fail(f"derived crop SHA changed: {row.get('sample_id')} {row.get('slot_role')}")
    train = [row for row in rows if row["data_role"] == "adaptation_train"]
    diagnostic = [row for row in rows if row["data_role"] == "diagnostic_seed"]
    if len(train) != 540 or len(diagnostic) != 60:
        _fail("slot role counts changed")
    if len({row["sample_id"] for row in train}) != 270:
        _fail("adaptation TRAIN must contain exactly 270 unique samples")
    if len({row["sample_id"] for row in diagnostic}) != 30:
        _fail("diagnostic seed set must contain exactly 30 unique samples")
    return manifest_path, rows, audit


@dataclass(frozen=True, slots=True)
class SpecialistAdaptationConfigV1:
    epochs: int = 12
    batch_size: int = 64
    learning_rate_micros: int = 100
    weight_decay_micros: int = 100
    master_seed: int = 52_023
    device: str = "cpu"
    optimizer: str = "AdamW"
    trainable_surface: str = "full-historical-digit-specialist"
    objective: str = "binary-bce-fixed-positive-weight-from-90-vs-450"
    checkpoint_selection: str = "fixed-final-epoch-no-sweep"

    def __post_init__(self) -> None:
        if self.epochs != 12 or self.batch_size != 64:
            raise ValueError("V5-2B epochs/batch size are frozen")
        if self.learning_rate_micros != 100 or self.weight_decay_micros != 100:
            raise ValueError("V5-2B optimizer rates are frozen")
        if self.master_seed != 52_023 or self.device != "cpu" or self.optimizer != "AdamW":
            raise ValueError("V5-2B deterministic execution profile is frozen")
        if self.trainable_surface != "full-historical-digit-specialist":
            raise ValueError("V5-2B trainable surface is frozen")
        if self.objective != "binary-bce-fixed-positive-weight-from-90-vs-450":
            raise ValueError("V5-2B objective is frozen")
        if self.checkpoint_selection != "fixed-final-epoch-no-sweep":
            raise ValueError("V5-2B checkpoint selection is frozen")


FROZEN_TRAIN_CONFIG: Final[SpecialistAdaptationConfigV1] = SpecialistAdaptationConfigV1()


def training_config_fingerprint_v1(config: SpecialistAdaptationConfigV1 = FROZEN_TRAIN_CONFIG) -> str:
    if config != FROZEN_TRAIN_CONFIG:
        _fail("V5-2B requires the frozen training config")
    return _canonical_json_sha({"schema": TRAINING_SCHEMA, "config": asdict(config)})


def locate_checkpoint_by_sha_v1(
    search_root: str | Path,
    expected_sha256: str,
    *,
    maximum_files: int = 5000,
) -> Path:
    root = Path(search_root)
    if root.is_symlink() or not root.is_dir():
        _fail("checkpoint search root must be a regular directory")
    checked = 0
    matches: list[Path] = []
    for path in sorted(root.rglob("*")):
        if checked >= maximum_files:
            break
        if path.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
            continue
        if path.is_symlink() or not path.is_file():
            continue
        checked += 1
        size = path.stat().st_size
        if not 1 <= size <= 5_000_000:
            continue
        if _sha_file(path) == expected_sha256:
            matches.append(path)
    if len(matches) != 1:
        _fail(f"expected exactly one checkpoint matching {expected_sha256}, found {len(matches)}")
    return matches[0]


def _import_torch():
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise MeterV5_2BError("PyTorch is required only for the V5-2B training/inference step") from exc
    return torch, nn


def _build_digit_model():
    torch, nn = _import_torch()

    class DigitSpecialist(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Linear(64, 1)

        def forward(self, images):
            hidden = self.features(images).flatten(1)
            return self.head(hidden).squeeze(1)

    return DigitSpecialist()


def _tensor_from_crop(path: Path):
    torch, _nn = _import_torch()
    with Image.open(path) as image:
        image.load()
        if image.mode != "L" or image.size != (64, 64):
            _fail(f"derived specialist crop contract changed: {path}")
        values = torch.tensor(list(image.getdata()), dtype=torch.float32)
    return values.reshape(1, 64, 64) / 255.0


def _state_fingerprint(model) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _binary_counts(probabilities, labels, threshold: float) -> dict[str, object]:
    preds = probabilities >= threshold
    truth = labels >= 0.5
    tp = int((preds & truth).sum().item())
    fp = int((preds & ~truth).sum().item())
    fn = int((~preds & truth).sum().item())
    tn = int((~preds & ~truth).sum().item())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def train_adapted_specialists_v1(
    data_root: str | Path,
    *,
    digit2_checkpoint: str | Path,
    digit3_checkpoint: str | Path,
    output_dir: str | Path | None = None,
    config: SpecialistAdaptationConfigV1 = FROZEN_TRAIN_CONFIG,
) -> dict[str, object]:
    """Fine-tune exactly 2-AI and 3-AI from audited historical checkpoints on 270 TRAIN samples."""
    if config != FROZEN_TRAIN_CONFIG:
        _fail("V5-2B training config must remain frozen")
    root = Path(data_root)
    manifest_path, rows, slot_audit = verify_slot_manifest_v1(root)
    torch, nn = _import_torch()
    from .runtime_meter_real_checkpoint_audit_v1 import audit_digit_checkpoint_v1

    checkpoints = {"2": Path(digit2_checkpoint), "3": Path(digit3_checkpoint)}
    expected_sha = {"2": DIGIT2_SHA256, "3": DIGIT3_SHA256}
    train_rows = [row for row in rows if row["data_role"] == "adaptation_train"]
    ann_dir = root / v51.ANNOTATIONS_DIR
    target_dir = Path(output_dir) if output_dir is not None else ann_dir / "v5_2b_candidates"
    target_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "schema": TRAINING_SCHEMA,
        "slot_manifest_sha256": _sha_file(manifest_path),
        "slot_audit_sha256": _sha_file(ann_dir / SLOT_AUDIT_NAME),
        "training_config": asdict(config),
        "training_config_fingerprint": training_config_fingerprint_v1(config),
        "device": "cpu",
        "train_sample_count": 270,
        "train_slot_count": 540,
        "diagnostic_seed_gradient_updates": 0,
        "threshold_tuning_allowed": False,
        "frozen_control_specialist": "4-AI",
        "validation_opened": False,
        "final_holdout_locked": True,
        "candidates": {},
    }

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    for digit in ("2", "3"):
        audited = audit_digit_checkpoint_v1(
            checkpoints[digit], role=f"digit-{digit}", expected_sha256=expected_sha[digit]
        )
        model = _build_digit_model().cpu()
        model.load_state_dict(dict(audited.model_state), strict=True)
        model.train()
        labels_list = [int(row[f"label_digit{digit}"]) for row in train_rows]
        positives = sum(labels_list)
        negatives = len(labels_list) - positives
        if (positives, negatives) != (90, 450):
            _fail(f"{digit}-AI train balance changed: {positives} positives, {negatives} negatives")
        images = torch.stack([_tensor_from_crop(ann_dir / row["crop_relpath"]) for row in train_rows], dim=0)
        labels = torch.tensor(labels_list, dtype=torch.float32)
        pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate_micros / 1_000_000.0,
            weight_decay=config.weight_decay_micros / 1_000_000.0,
        )
        epoch_losses: list[float] = []
        seed = config.master_seed + int(digit)
        torch.manual_seed(seed)
        for epoch in range(config.epochs):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + epoch)
            order = torch.randperm(len(train_rows), generator=generator)
            total_loss = 0.0
            total_seen = 0
            for start in range(0, len(train_rows), config.batch_size):
                batch_index = order[start : start + config.batch_size]
                batch_images = images[batch_index]
                batch_labels = labels[batch_index]
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_images)
                loss = criterion(logits, batch_labels)
                if not bool(torch.isfinite(loss).item()):
                    _fail(f"{digit}-AI training produced non-finite loss")
                loss.backward()
                optimizer.step()
                count = int(batch_index.numel())
                total_loss += float(loss.detach().item()) * count
                total_seen += count
            epoch_losses.append(total_loss / total_seen)

        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(images))
        train_metrics = _binary_counts(probabilities, labels, FROZEN_THRESHOLDS[digit])
        state_fp = _state_fingerprint(model)
        candidate_path = target_dir / f"digit{digit}_v5_2b_candidate.pt"
        payload = {
            "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "metadata": {
                "schema": TRAINING_SCHEMA,
                "role": f"digit-{digit}-v5-2b-candidate",
                "source_checkpoint_sha256": expected_sha[digit],
                "slot_manifest_sha256": _sha_file(manifest_path),
                "training_config_fingerprint": training_config_fingerprint_v1(config),
                "state_fingerprint": state_fp,
                "threshold": FROZEN_THRESHOLDS[digit],
                "threshold_tuned": False,
                "diagnostic_seed_gradient_updates": 0,
                "validation_opened": False,
                "final_holdout_locked": True,
            },
        }
        torch.save(payload, candidate_path)
        candidate_sha = _sha_file(candidate_path)
        report["candidates"][digit] = {
            "role": f"digit-{digit}-v5-2b-candidate",
            "source_checkpoint_sha256": expected_sha[digit],
            "candidate_path": str(candidate_path),
            "candidate_sha256": candidate_sha,
            "state_fingerprint": state_fp,
            "epoch_losses": epoch_losses,
            "final_train_metrics_at_frozen_threshold": train_metrics,
            "positive_weight": negatives / positives,
        }

    v51._atomic_write_json(ann_dir / TRAINING_REPORT_NAME, report)
    return report


def _load_candidate_model(path: Path, *, digit: str, manifest_sha256: str):
    torch, _nn = _import_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise MeterV5_2BError(f"cannot load V5-2B candidate {digit}-AI") from exc
    if not isinstance(payload, Mapping):
        _fail("candidate checkpoint payload must be a mapping")
    metadata = payload.get("metadata")
    state = payload.get("model_state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
        _fail("candidate checkpoint is missing state/metadata")
    expected_source = DIGIT2_SHA256 if digit == "2" else DIGIT3_SHA256
    expected_meta = {
        "schema": TRAINING_SCHEMA,
        "role": f"digit-{digit}-v5-2b-candidate",
        "source_checkpoint_sha256": expected_source,
        "slot_manifest_sha256": manifest_sha256,
        "training_config_fingerprint": training_config_fingerprint_v1(),
        "threshold": FROZEN_THRESHOLDS[digit],
        "threshold_tuned": False,
        "diagnostic_seed_gradient_updates": 0,
        "validation_opened": False,
        "final_holdout_locked": True,
    }
    for key, expected in expected_meta.items():
        if metadata.get(key) != expected:
            _fail(f"candidate {digit}-AI metadata changed: {key}")
    model = _build_digit_model().cpu()
    model.load_state_dict(dict(state), strict=True)
    if metadata.get("state_fingerprint") != _state_fingerprint(model):
        _fail(f"candidate {digit}-AI state fingerprint mismatch")
    model.eval()
    return model


def _probability(model, crop_path: Path) -> float:
    torch, _nn = _import_torch()
    with torch.no_grad():
        value = torch.sigmoid(model(_tensor_from_crop(crop_path).unsqueeze(0))).item()
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        _fail("candidate inference produced invalid probability")
    return float(value)


def evaluate_diagnostic_gate_v1(
    data_root: str | Path,
    *,
    digit2_candidate: str | Path,
    digit3_candidate: str | Path,
    digit4_checkpoint: str | Path,
) -> dict[str, object]:
    """Evaluate the 30 untouched seeds with candidate 2/3 and frozen 4 using unchanged thresholds."""
    root = Path(data_root)
    manifest_path, rows, _slot_audit = verify_slot_manifest_v1(root)
    manifest_sha = _sha_file(manifest_path)
    training_report = _read_json(root / v51.ANNOTATIONS_DIR / TRAINING_REPORT_NAME)
    candidates_report = training_report.get("candidates")
    if not isinstance(candidates_report, Mapping):
        _fail("training report does not contain candidate evidence")
    if candidates_report.get("2", {}).get("candidate_sha256") != _sha_file(Path(digit2_candidate)):
        _fail("2-AI candidate differs from training report")
    if candidates_report.get("3", {}).get("candidate_sha256") != _sha_file(Path(digit3_candidate)):
        _fail("3-AI candidate differs from training report")

    torch, _nn = _import_torch()
    from .runtime_meter_real_checkpoint_audit_v1 import audit_digit_checkpoint_v1
    model2 = _load_candidate_model(Path(digit2_candidate), digit="2", manifest_sha256=manifest_sha)
    model3 = _load_candidate_model(Path(digit3_candidate), digit="3", manifest_sha256=manifest_sha)
    audited4 = audit_digit_checkpoint_v1(Path(digit4_checkpoint), role="digit-4", expected_sha256=DIGIT4_SHA256)
    model4 = _build_digit_model().cpu()
    model4.load_state_dict(dict(audited4.model_state), strict=True)
    model4.eval()

    ann_dir = root / v51.ANNOTATIONS_DIR
    diagnostic = [row for row in rows if row["data_role"] == "diagnostic_seed"]
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in diagnostic:
        grouped.setdefault(row["sample_id"], {})[row["slot_role"]] = row
    if len(grouped) != 30 or any(set(slots) != {"numerator", "denominator"} for slots in grouped.values()):
        _fail("diagnostic manifest must contain exactly two slots for each of 30 seeds")

    samples: list[dict[str, object]] = []
    per_meter_pass: Counter[str] = Counter()
    denominator_exact4 = 0
    for sample_id in sorted(grouped, key=lambda sid: int(grouped[sid]["numerator"]["sample_index"])):
        slots = grouped[sample_id]
        meter = slots["numerator"]["meter"]
        expected_num = meter.split("/")[0]
        slot_results: dict[str, dict[str, object]] = {}
        for slot_role in ("numerator", "denominator"):
            crop_path = ann_dir / slots[slot_role]["crop_relpath"]
            probabilities = {
                "2": _probability(model2, crop_path),
                "3": _probability(model3, crop_path),
                "4": _probability(model4, crop_path),
            }
            hits = [digit for digit in ("2", "3", "4") if probabilities[digit] >= FROZEN_THRESHOLDS[digit]]
            slot_results[slot_role] = {"probabilities": probabilities, "hits": hits}
        numerator_ok = slot_results["numerator"]["hits"] == [expected_num]
        denominator_ok = slot_results["denominator"]["hits"] == ["4"]
        meter_pass = bool(numerator_ok and denominator_ok)
        if denominator_ok:
            denominator_exact4 += 1
        if meter_pass:
            per_meter_pass[meter] += 1
        samples.append({
            "sample_id": sample_id,
            "meter": meter,
            "numerator": slot_results["numerator"],
            "denominator": slot_results["denominator"],
            "numerator_correct": numerator_ok,
            "denominator_correct": denominator_ok,
            "meter_pass": meter_pass,
        })

    required = {"2/4": 8, "3/4": 8, "4/4": 9}
    reasons = [
        f"{meter}_PASS_BELOW_{minimum}_OF_10"
        for meter, minimum in required.items()
        if per_meter_pass[meter] < minimum
    ]
    if denominator_exact4 < 26:
        reasons.append("DENOMINATOR_EXACT4_BELOW_26_OF_30")
    accepted = not reasons
    report = {
        "schema": DIAGNOSTIC_SCHEMA,
        "slot_manifest_sha256": manifest_sha,
        "digit2_candidate_sha256": _sha_file(Path(digit2_candidate)),
        "digit3_candidate_sha256": _sha_file(Path(digit3_candidate)),
        "digit4_frozen_sha256": DIGIT4_SHA256,
        "thresholds": dict(FROZEN_THRESHOLDS),
        "threshold_tuned": False,
        "diagnostic_seed_count": 30,
        "diagnostic_seed_gradient_updates": 0,
        "per_meter_pass": {meter: per_meter_pass[meter] for meter in ("2/4", "3/4", "4/4")},
        "denominator_exact4": denominator_exact4,
        "gate": "PASS" if accepted else "HOLD",
        "reasons": reasons,
        "validation_bbox_stage_authorized": accepted,
        "validation_opened": False,
        "final_holdout_locked": True,
        "resolver_wiring_authorized": False,
        "production_promotion_authorized": False,
        "samples": samples,
    }
    v51._atomic_write_json(ann_dir / DIAGNOSTIC_REPORT_NAME, report)
    return report


def production_promotion_allowed() -> bool:
    return False


def validation_opened_by_this_module() -> bool:
    return False


def final_holdout_locked() -> bool:
    return True
