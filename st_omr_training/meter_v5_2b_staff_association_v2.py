"""Meter V5-2B approved multi-staff association and fail-closed preflight.

This module fixes one integration mistake in the first V5-2B slot derivation:
the page is not required to contain exactly one staff. Instead, the human
full-meter BBox center-y is mapped into normalized coordinates and MUST be
contained by exactly one already-accepted Geometry Engine staff bbox.

Frozen rule approved by the user:
- use human full-meter BBox center-y only;
- choose the unique accepted staff whose staff_bbox vertical interval contains it;
- zero matches -> HOLD;
- multiple matches -> HOLD;
- no nearest-staff fallback;
- no tolerance expansion;
- no midpoint, automatic BBox correction, model-generated GT, or new crop rule.

The preflight evaluates all 300 TRAIN annotations before any slot crop is
written. Slot derivation then consumes the PASS preflight evidence and writes
the existing V5-2B slot-manifest contract expected by the narrow 2-AI/3-AI
training lane. VAL, FINAL_HOLDOUT, Resolver and production remain closed.
"""
from __future__ import annotations

import io
import math
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Callable, Final, Iterable

from PIL import Image

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2a_specialist_adaptation as v52a
from . import meter_v5_2b_specialist_adaptation as v52b
from .runtime_geometry_engine_contract import GeometryInputContract, StaffGeometryContract
from .runtime_geometry_engine_v2 import detect_multistaff_geometry_v2
from .runtime_page_normalizer_contract import RasterPageInputContract
from .runtime_page_normalizer_v1 import normalize_raster_page_v1


STAFF_ASSOCIATION_SCHEMA: Final[str] = "st-omr-meter-v5-2b-staff-association-preflight-v2"
STAFF_ASSOCIATION_RULE: Final[str] = "human-full-meter-bbox-center-y-unique-containing-staff-v1"
PREFLIGHT_CSV_NAME: Final[str] = "v5_2b_staff_association_preflight_v2.csv"
PREFLIGHT_AUDIT_NAME: Final[str] = "v5_2b_staff_association_preflight_v2.json"

PREFLIGHT_COLUMNS: Final[tuple[str, ...]] = (
    "schema",
    "sample_index",
    "sample_id",
    "meter",
    "source_image_sha256",
    "runtime_raster_sha256",
    "pixel_mode_adapter",
    "normalized_image_sha256",
    "center_x_normalized",
    "center_y_normalized",
    "geometry_staff_count",
    "selected_staff_id",
    "selected_staff_bbox_y_min",
    "selected_staff_bbox_y_max",
    "staff_spacing",
    "line_y_0",
    "line_y_1",
    "line_y_2",
    "line_y_3",
    "line_y_4",
)

ProgressCallback = Callable[[int, int, str, str], None]


def _fail(message: str) -> None:
    raise v52b.MeterV5_2BError(message)


def select_unique_containing_staff_v1(
    staffs: Iterable[StaffGeometryContract],
    *,
    center_y_normalized: float,
    sample_id: str,
) -> StaffGeometryContract:
    """Resolve only by exact vertical containment; never by nearest distance."""
    if not isinstance(center_y_normalized, (int, float)) or isinstance(center_y_normalized, bool):
        _fail(f"non-numeric normalized BBox center-y: {sample_id}")
    center_y = float(center_y_normalized)
    if not math.isfinite(center_y):
        _fail(f"non-finite normalized BBox center-y: {sample_id}")
    staff_tuple = tuple(staffs)
    matches = tuple(
        staff
        for staff in staff_tuple
        if staff.staff_bbox.y_min <= center_y <= staff.staff_bbox.y_max
    )
    if len(matches) != 1:
        _fail(
            "staff association requires exactly one center-y containment: "
            f"{sample_id}; center_y={center_y:.12f}; "
            f"matches={len(matches)}; accepted_staffs={len(staff_tuple)}"
        )
    return matches[0]


def _load_bound_rows(root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    """Load only the exact selection/annotation files bound by human-QA evidence."""
    v52b.verify_human_qa_attestation(root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    selection = v52b._read_csv(ann_dir / v52a.SELECTION_NAME)
    annotations = v52b._read_csv(ann_dir / v52a.ANNOTATION_NAME)
    if len(selection) != 300 or len(annotations) != 300:
        _fail("V5-2B staff preflight requires exact 300 selection + 300 annotations")
    if len({row.get("sample_id", "") for row in selection}) != 300:
        _fail("V5-2B staff preflight selection has duplicate sample ids")
    annotation_by_id = {row.get("sample_id", ""): row for row in annotations}
    if len(annotation_by_id) != 300:
        _fail("V5-2B staff preflight annotations have duplicate sample ids")
    if set(annotation_by_id) != {row["sample_id"] for row in selection}:
        _fail("V5-2B staff preflight selection/annotation identity mismatch")
    for row in selection:
        annotation = annotation_by_id[row["sample_id"]]
        if annotation.get("status") != "PASS":
            _fail(f"non-PASS annotation reached staff preflight: {row['sample_id']}")
        if annotation.get("meter") != row.get("meter"):
            _fail(f"meter mismatch between selection and annotation: {row['sample_id']}")
        if annotation.get("image_sha256") != row.get("image_sha256"):
            _fail(f"image binding mismatch between selection and annotation: {row['sample_id']}")
    return selection, annotation_by_id


def _normalize_and_associate(
    root: Path,
    selection: dict[str, str],
    annotation: dict[str, str],
) -> tuple[bytes, object, StaffGeometryContract, dict[str, object]]:
    sample_id = selection["sample_id"]
    image_path = root / selection["image_relpath"]
    raw = image_path.read_bytes()
    if sha256(raw).hexdigest() != selection["image_sha256"]:
        _fail(f"source image SHA changed: {sample_id}")
    runtime_raw, mode, width, height, adapter = v52b._prepare_runtime_raster(raw)
    if width != int(selection["image_width"]) or height != int(selection["image_height"]):
        _fail(f"source image dimensions changed: {sample_id}")
    runtime_sha = sha256(runtime_raw).hexdigest()
    raster = RasterPageInputContract(
        source_id=sample_id,
        source_sha256=selection["image_sha256"],
        page_number=1,
        width=width,
        height=height,
        pixel_mode=v52b._mode_to_contract(mode),
        raster_sha256=runtime_sha,
        dpi=None,
    )
    normalized = normalize_raster_page_v1(runtime_raw, raster)
    if normalized.page.status != "accepted" or normalized.normalized_png is None:
        _fail(f"normalizer HOLD during V5-2B staff preflight: {sample_id}")
    page = normalized.page
    geometry_input = GeometryInputContract(
        normalized_image_sha256=page.normalized_image_sha256,
        normalizer_config_fingerprint=page.normalizer_config_fingerprint,
        normalized_width=page.normalized_width,
        normalized_height=page.normalized_height,
        transform=page.transform,
    )
    geometry = detect_multistaff_geometry_v2(normalized.normalized_png, geometry_input)
    if geometry.page.status != "accepted":
        reasons = ",".join(geometry.page.reasons)
        _fail(f"geometry not accepted during V5-2B staff preflight: {sample_id}; reasons={reasons}")

    x = int(annotation["x"])
    y = int(annotation["y"])
    w = int(annotation["w"])
    h = int(annotation["h"])
    center_x_original = x + w / 2.0
    center_y_original = y + h / 2.0
    center_x_normalized, center_y_normalized = page.transform.original_to_normalized(
        center_x_original, center_y_original
    )
    staff = select_unique_containing_staff_v1(
        geometry.page.staffs,
        center_y_normalized=center_y_normalized,
        sample_id=sample_id,
    )
    line_y = tuple((line.start.y + line.end.y) / 2.0 for line in staff.five_staff_lines)
    metadata = {
        "runtime_raster_sha256": runtime_sha,
        "pixel_mode_adapter": adapter,
        "normalized_image_sha256": page.normalized_image_sha256,
        "center_x_normalized": center_x_normalized,
        "center_y_normalized": center_y_normalized,
        "geometry_staff_count": len(geometry.page.staffs),
        "selected_staff_id": staff.staff_id,
        "selected_staff_bbox_y_min": staff.staff_bbox.y_min,
        "selected_staff_bbox_y_max": staff.staff_bbox.y_max,
        "staff_spacing": float(staff.staff_spacing),
        "line_y": line_y,
    }
    return normalized.normalized_png, page, staff, metadata


def preflight_staff_association_v2(
    data_root: str | Path,
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    """Check all 300 center-y staff associations before any slot crop is written."""
    root = Path(data_root)
    selection, annotation_by_id = _load_bound_rows(root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    pass_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    staff_count_histogram: Counter[str] = Counter()
    adapter_counts: Counter[str] = Counter()

    for position, selected in enumerate(selection, start=1):
        sample_id = selected["sample_id"]
        if progress is not None:
            progress(position - 1, 300, sample_id, "checking")
        try:
            _normalized_png, _page, _staff, meta = _normalize_and_associate(
                root, selected, annotation_by_id[sample_id]
            )
            line_y = meta["line_y"]
            pass_rows.append({
                "schema": STAFF_ASSOCIATION_SCHEMA,
                "sample_index": selected["index"],
                "sample_id": sample_id,
                "meter": selected["meter"],
                "source_image_sha256": selected["image_sha256"],
                "runtime_raster_sha256": str(meta["runtime_raster_sha256"]),
                "pixel_mode_adapter": str(meta["pixel_mode_adapter"]),
                "normalized_image_sha256": str(meta["normalized_image_sha256"]),
                "center_x_normalized": f"{float(meta['center_x_normalized']):.12f}",
                "center_y_normalized": f"{float(meta['center_y_normalized']):.12f}",
                "geometry_staff_count": str(meta["geometry_staff_count"]),
                "selected_staff_id": str(meta["selected_staff_id"]),
                "selected_staff_bbox_y_min": f"{float(meta['selected_staff_bbox_y_min']):.12f}",
                "selected_staff_bbox_y_max": f"{float(meta['selected_staff_bbox_y_max']):.12f}",
                "staff_spacing": f"{float(meta['staff_spacing']):.12f}",
                "line_y_0": f"{float(line_y[0]):.12f}",
                "line_y_1": f"{float(line_y[1]):.12f}",
                "line_y_2": f"{float(line_y[2]):.12f}",
                "line_y_3": f"{float(line_y[3]):.12f}",
                "line_y_4": f"{float(line_y[4]):.12f}",
            })
            staff_count_histogram[str(meta["geometry_staff_count"])] += 1
            adapter_counts[str(meta["pixel_mode_adapter"])] += 1
        except Exception as exc:
            failures.append({"sample_id": sample_id, "meter": selected["meter"], "reason": str(exc)})
        if progress is not None:
            progress(position, 300, sample_id, "pass" if not failures or failures[-1].get("sample_id") != sample_id else "hold")

    csv_path = ann_dir / PREFLIGHT_CSV_NAME
    v51._atomic_write_csv(csv_path, PREFLIGHT_COLUMNS, pass_rows)
    gate = "PASS" if len(pass_rows) == 300 and not failures else "HOLD"
    audit = {
        "schema": STAFF_ASSOCIATION_SCHEMA,
        "association_rule": STAFF_ASSOCIATION_RULE,
        "selection_sha256": v52b._sha_file(ann_dir / v52a.SELECTION_NAME),
        "annotation_sha256": v52b._sha_file(ann_dir / v52a.ANNOTATION_NAME),
        "human_qa_sha256": v52b._sha_file(ann_dir / v52b.HUMAN_QA_NAME),
        "preflight_manifest_sha256": v52b._sha_file(csv_path),
        "sample_count_expected": 300,
        "pass_count": len(pass_rows),
        "hold_count": len(failures),
        "failures": failures,
        "geometry_staff_count_histogram": dict(sorted(staff_count_histogram.items())),
        "pixel_mode_adapter_counts": dict(sorted(adapter_counts.items())),
        "nearest_staff_fallback": False,
        "staff_bbox_tolerance_px": 0,
        "midpoint_derivation": False,
        "automatic_bbox_correction": False,
        "model_generated_gt": False,
        "gate": gate,
        "slot_derivation_authorized": gate == "PASS",
        "training_authorized": False,
        "validation_opened": False,
        "final_holdout_locked": True,
        "frozen_control_specialist": "4-AI",
    }
    audit_path = ann_dir / PREFLIGHT_AUDIT_NAME
    v51._atomic_write_json(audit_path, audit)
    return audit_path


def verify_staff_association_preflight_v2(data_root: str | Path) -> tuple[Path, list[dict[str, str]], dict[str, object]]:
    root = Path(data_root)
    v52b.verify_human_qa_attestation(root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    csv_path = ann_dir / PREFLIGHT_CSV_NAME
    audit_path = ann_dir / PREFLIGHT_AUDIT_NAME
    rows = v52b._read_csv(csv_path)
    audit = v52b._read_json(audit_path)
    expected_hashes = {
        "selection_sha256": v52b._sha_file(ann_dir / v52a.SELECTION_NAME),
        "annotation_sha256": v52b._sha_file(ann_dir / v52a.ANNOTATION_NAME),
        "human_qa_sha256": v52b._sha_file(ann_dir / v52b.HUMAN_QA_NAME),
        "preflight_manifest_sha256": v52b._sha_file(csv_path),
    }
    if audit.get("schema") != STAFF_ASSOCIATION_SCHEMA or audit.get("association_rule") != STAFF_ASSOCIATION_RULE:
        _fail("staff-association preflight schema/rule changed")
    for key, expected in expected_hashes.items():
        if audit.get(key) != expected:
            _fail(f"staff-association preflight evidence hash changed: {key}")
    if audit.get("gate") != "PASS" or audit.get("pass_count") != 300 or audit.get("hold_count") != 0:
        _fail("staff-association preflight is not 300/300 PASS")
    if len(rows) != 300 or len({row["sample_id"] for row in rows}) != 300:
        _fail("staff-association preflight manifest is not 300 unique rows")
    if audit.get("nearest_staff_fallback") is not False or audit.get("staff_bbox_tolerance_px") != 0:
        _fail("staff-association preflight introduced nearest/tolerance behavior")
    if audit.get("validation_opened") is not False or audit.get("final_holdout_locked") is not True:
        _fail("staff-association preflight changed VAL/holdout boundary")
    return csv_path, rows, audit


def derive_staff_relative_slots_v2(
    data_root: str | Path,
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    """Derive the existing 600-slot contract only after a 300/300 PASS preflight."""
    root = Path(data_root)
    preflight_path, preflight_rows, preflight_audit = verify_staff_association_preflight_v2(root)
    selection, annotation_by_id = _load_bound_rows(root)
    preflight_by_id = {row["sample_id"]: row for row in preflight_rows}
    ann_dir = root / v51.ANNOTATIONS_DIR
    qa_sha = v52b._sha_file(ann_dir / v52b.HUMAN_QA_NAME)
    annotation_sha = v52b._sha_file(ann_dir / v52a.ANNOTATION_NAME)
    selection_sha = v52b._sha_file(ann_dir / v52a.SELECTION_NAME)
    derivation_fp = v52b._canonical_json_sha({
        "schema": v52b.SLOT_MANIFEST_SCHEMA,
        "qa_sha256": qa_sha,
        "annotation_sha256": annotation_sha,
        "selection_sha256": selection_sha,
        "staff_association_rule": STAFF_ASSOCIATION_RULE,
        "staff_association_preflight_sha256": v52b._sha_file(preflight_path),
        "width_over_staff_spacing": v52b.WIDTH_OVER_STAFF_SPACING,
        "height_over_staff_spacing": v52b.HEIGHT_OVER_STAFF_SPACING,
        "numerator_line_index": v52b.NUMERATOR_LINE_INDEX,
        "denominator_line_index": v52b.DENOMINATOR_LINE_INDEX,
        "preprocessing": "gray-L;thumbnail-64-LANCZOS-no-upscale;center-white-64",
    })
    crop_root = ann_dir / f"v5_2b_slot_crops_{derivation_fp[:16]}"
    crop_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    adapter_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    meter_counts: Counter[str] = Counter()

    for position, selected in enumerate(selection, start=1):
        sample_id = selected["sample_id"]
        if progress is not None:
            progress(position - 1, 300, sample_id, "deriving")
        annotation = annotation_by_id[sample_id]
        preflight = preflight_by_id[sample_id]
        raw = (root / selected["image_relpath"]).read_bytes()
        if sha256(raw).hexdigest() != selected["image_sha256"]:
            _fail(f"source image SHA changed during slot derivation: {sample_id}")
        runtime_raw, mode, width, height, adapter = v52b._prepare_runtime_raster(raw)
        runtime_sha = sha256(runtime_raw).hexdigest()
        if runtime_sha != preflight["runtime_raster_sha256"] or adapter != preflight["pixel_mode_adapter"]:
            _fail(f"runtime raster no longer matches staff preflight: {sample_id}")
        raster = RasterPageInputContract(
            source_id=sample_id,
            source_sha256=selected["image_sha256"],
            page_number=1,
            width=width,
            height=height,
            pixel_mode=v52b._mode_to_contract(mode),
            raster_sha256=runtime_sha,
            dpi=None,
        )
        normalized = normalize_raster_page_v1(runtime_raw, raster)
        if normalized.page.status != "accepted" or normalized.normalized_png is None:
            _fail(f"normalizer HOLD during V5-2B slot derivation: {sample_id}")
        page = normalized.page
        if page.normalized_image_sha256 != preflight["normalized_image_sha256"]:
            _fail(f"normalized image no longer matches staff preflight: {sample_id}")

        x = int(annotation["x"])
        y = int(annotation["y"])
        w = int(annotation["w"])
        h = int(annotation["h"])
        center_x_original = x + w / 2.0
        center_y_original = y + h / 2.0
        center_x_normalized, center_y_normalized = page.transform.original_to_normalized(
            center_x_original, center_y_original
        )
        if abs(center_x_normalized - float(preflight["center_x_normalized"])) > 1e-9:
            _fail(f"normalized center-x changed after staff preflight: {sample_id}")
        if abs(center_y_normalized - float(preflight["center_y_normalized"])) > 1e-9:
            _fail(f"normalized center-y changed after staff preflight: {sample_id}")
        bbox_y_min = float(preflight["selected_staff_bbox_y_min"])
        bbox_y_max = float(preflight["selected_staff_bbox_y_max"])
        if not bbox_y_min <= center_y_normalized <= bbox_y_max:
            _fail(f"center-y no longer contained by preflight staff bbox: {sample_id}")
        spacing = float(preflight["staff_spacing"])
        line_y = tuple(float(preflight[f"line_y_{index}"]) for index in range(5))
        boxes = {
            "numerator": v52b._integer_crop_box(
                v52b._slot_edges(center_x_normalized, line_y[v52b.NUMERATOR_LINE_INDEX], spacing),
                image_width=page.normalized_width,
                image_height=page.normalized_height,
            ),
            "denominator": v52b._integer_crop_box(
                v52b._slot_edges(center_x_normalized, line_y[v52b.DENOMINATOR_LINE_INDEX], spacing),
                image_width=page.normalized_width,
                image_height=page.normalized_height,
            ),
        }
        sample_index = int(selected["index"])
        data_role = "diagnostic_seed" if sample_index < v52b.DIAGNOSTIC_SEED_TOTAL else "adaptation_train"
        expected_numerator = selected["meter"].split("/")[0]
        with Image.open(io.BytesIO(normalized.normalized_png)) as image:
            image.load()
            gray = image.convert("L")
            for slot_role, box in boxes.items():
                ix0, iy0, ix1, iy1 = box
                crop = v52b._historical_canvas(gray.crop(box))
                encoded = v52b._encode_png(crop)
                relative = Path(f"v5_2b_slot_crops_{derivation_fp[:16]}") / data_role / f"{sample_id}_{slot_role}.png"
                target = ann_dir / relative
                v52b._atomic_bytes(target, encoded)
                expected_digit = expected_numerator if slot_role == "numerator" else "4"
                label2 = int(slot_role == "numerator" and expected_numerator == "2")
                label3 = int(slot_role == "numerator" and expected_numerator == "3")
                rows.append({
                    "schema": v52b.SLOT_MANIFEST_SCHEMA,
                    "sample_index": str(sample_index),
                    "sample_id": sample_id,
                    "family_id": selected["family_id"],
                    "meter": selected["meter"],
                    "data_role": data_role,
                    "slot_role": slot_role,
                    "expected_digit": expected_digit,
                    "label_digit2": str(label2),
                    "label_digit3": str(label3),
                    "source_image_sha256": selected["image_sha256"],
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
        meter_counts[f"{data_role}:{selected['meter']}"] += 1
        if progress is not None:
            progress(position, 300, sample_id, "pass")

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

    manifest_path = ann_dir / v52b.SLOT_MANIFEST_NAME
    v51._atomic_write_csv(manifest_path, v52b.SLOT_COLUMNS, rows)
    audit = {
        "schema": v52b.SLOT_MANIFEST_SCHEMA,
        "derivation_fingerprint": derivation_fp,
        "staff_association_rule": STAFF_ASSOCIATION_RULE,
        "staff_association_preflight_sha256": v52b._sha_file(preflight_path),
        "staff_association_preflight_manifest_sha256": preflight_audit["preflight_manifest_sha256"],
        "nearest_staff_fallback": False,
        "staff_bbox_tolerance_px": 0,
        "human_qa_sha256": qa_sha,
        "selection_sha256": selection_sha,
        "annotation_sha256": annotation_sha,
        "manifest_sha256": v52b._sha_file(manifest_path),
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
    v51._atomic_write_json(ann_dir / v52b.SLOT_AUDIT_NAME, audit)
    return manifest_path
