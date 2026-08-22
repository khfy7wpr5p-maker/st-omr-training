"""V5-2B bounded replacement of two approved adaptation-TRAIN samples.

The only authorized selection change in this module is the explicit replacement
of the two non-seed TRAIN samples that failed the V5-2B staff-association
preflight with A04_PAGE_CROPPED. Each replacement is the deterministic first
unused TRAIN row in the same meter class under the original V5-2A ordering
(selection_rank, sample_id). No geometry result is used to skip ahead to a
later candidate; instead, the selected next-unused candidate is screened and
fails closed if its page geometry is not accepted.

This module never changes the first 30 diagnostic seeds, never opens VAL or
FINAL_HOLDOUT, never trains a model, and never changes 4-AI or thresholds.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Final, Iterable, Mapping

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2a_specialist_adaptation as v52a
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2b_staff_association_v2 as v52b2
from .runtime_geometry_engine_contract import GeometryInputContract
from .runtime_geometry_engine_v2 import detect_multistaff_geometry_v2
from .runtime_page_normalizer_contract import RasterPageInputContract
from .runtime_page_normalizer_v1 import normalize_raster_page_v1


REPLACEMENT_SCHEMA: Final[str] = "st-omr-meter-v5-2b-train-replacement-v1"
REPLACEMENT_RULE: Final[str] = "same-meter-next-unused-train-by-selection-rank-then-sample-id-v1"
REPLACEMENT_PLAN_NAME: Final[str] = "v5_2b_train_replacement_plan_v1.json"
REPLACEMENT_APPLY_NAME: Final[str] = "v5_2b_train_replacement_apply_v1.json"
REPLACEMENT_ARCHIVE_DIR: Final[str] = "v5_2b_pre_replacement_archive"

APPROVED_HOLDS: Final[dict[str, dict[str, object]]] = {
    "150201200-1_1_1": {
        "meter": "2/4",
        "index": 63,
        "required_reason": "A04_PAGE_CROPPED",
    },
    "110003725-1_1_1": {
        "meter": "3/4",
        "index": 125,
        "required_reason": "A04_PAGE_CROPPED",
    },
}


def _fail(message: str) -> None:
    raise v52b.MeterV5_2BError(message)


def _read_rows(path: Path) -> list[dict[str, str]]:
    return v52b._read_csv(path)


def _manifest_train_rows(root: Path, meter: str) -> list[dict[str, str]]:
    rows = v51._validate_manifest(root / v51.MANIFEST_NAME[meter], meter)
    train_rows = [row for row in rows if row["Split"] == "train"]
    if len(train_rows) != 400:
        _fail(f"{meter}: expected exactly 400 TRAIN rows, got {len(train_rows)}")
    train_rows.sort(key=lambda row: (v52a._source_rank(row), row["SampleId"]))
    return train_rows


def select_next_unused_train_row_v1(
    manifest_train_rows: Iterable[Mapping[str, str]],
    *,
    meter: str,
    existing_sample_ids: set[str],
    existing_family_ids: set[str],
) -> Mapping[str, str]:
    """Return the first unused TRAIN row under the frozen V5-2A ordering."""
    rows = list(manifest_train_rows)
    rows.sort(key=lambda row: (v52a._source_rank(row), row["SampleId"]))
    for row in rows:
        if row.get("Split") != "train" or row.get("Meter") != meter:
            continue
        if row["SampleId"] in existing_sample_ids:
            continue
        if row["FamilyId"] in existing_family_ids:
            continue
        return row
    _fail(f"{meter}: no unused TRAIN replacement candidate remains")


def _require_exact_hold_evidence(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    ann_dir = root / v51.ANNOTATIONS_DIR
    v52b.verify_human_qa_attestation(root)
    selection = _read_rows(ann_dir / v52a.SELECTION_NAME)
    annotations = _read_rows(ann_dir / v52a.ANNOTATION_NAME)
    audit = v52b._read_json(ann_dir / v52b2.PREFLIGHT_AUDIT_NAME)

    if len(selection) != 300 or len(annotations) != 300:
        _fail("replacement requires the completed pre-replacement 300/300 selection and annotations")
    if audit.get("gate") != "HOLD" or audit.get("pass_count") != 298 or audit.get("hold_count") != 2:
        _fail("replacement requires the exact 298 PASS / 2 HOLD staff preflight evidence")
    failures = audit.get("failures")
    if not isinstance(failures, list) or len(failures) != 2:
        _fail("replacement preflight failures are not exactly two records")

    failure_by_id = {str(item.get("sample_id")): item for item in failures if isinstance(item, dict)}
    if set(failure_by_id) != set(APPROVED_HOLDS):
        _fail("replacement failure identities differ from the two approved HOLD samples")

    selection_by_id = {row["sample_id"]: row for row in selection}
    annotation_by_id = {row["sample_id"]: row for row in annotations}
    for sample_id, expected in APPROVED_HOLDS.items():
        selected = selection_by_id.get(sample_id)
        annotation = annotation_by_id.get(sample_id)
        failure = failure_by_id[sample_id]
        if selected is None or annotation is None:
            _fail(f"approved HOLD sample missing from selection/annotations: {sample_id}")
        if int(selected["index"]) != int(expected["index"]):
            _fail(f"approved HOLD index changed: {sample_id}")
        if selected["meter"] != expected["meter"] or selected["seed_annotation"] != "0":
            _fail(f"approved HOLD is not the expected non-seed TRAIN sample: {sample_id}")
        if annotation.get("status") != "PASS":
            _fail(f"approved HOLD annotation is no longer PASS: {sample_id}")
        if failure.get("meter") != expected["meter"]:
            _fail(f"approved HOLD meter changed in preflight evidence: {sample_id}")
        reason = str(failure.get("reason", ""))
        if not reason.endswith(f"reasons={expected['required_reason']}"):
            _fail(f"approved HOLD reason changed: {sample_id}: {reason}")

    if audit.get("selection_sha256") != v52b._sha_file(ann_dir / v52a.SELECTION_NAME):
        _fail("preflight selection hash no longer matches current selection")
    if audit.get("annotation_sha256") != v52b._sha_file(ann_dir / v52a.ANNOTATION_NAME):
        _fail("preflight annotation hash no longer matches current annotations")
    return selection, annotations, audit


def _selection_row_from_manifest(root: Path, manifest_row: Mapping[str, str], *, index: int) -> dict[str, str]:
    meter = manifest_row["Meter"]
    image_path = root / "train" / v51.CLASS_DIR[meter] / manifest_row["Folder"] / "image.png"
    image_sha, width, height = v51._read_png_binding(image_path)
    return {
        "index": str(index),
        "sample_id": manifest_row["SampleId"],
        "family_id": manifest_row["FamilyId"],
        "meter": meter,
        "split": "train",
        "folder": manifest_row["Folder"],
        "image_relpath": image_path.relative_to(root).as_posix(),
        "image_sha256": image_sha,
        "image_width": str(width),
        "image_height": str(height),
        "seed_annotation": "0",
        "selection_rank": v52a._source_rank(manifest_row),
    }


def _screen_page_geometry_v1(root: Path, selection_row: Mapping[str, str]) -> dict[str, object]:
    sample_id = selection_row["sample_id"]
    image_path = root / selection_row["image_relpath"]
    raw = image_path.read_bytes()
    if sha256(raw).hexdigest() != selection_row["image_sha256"]:
        _fail(f"replacement candidate image SHA changed: {sample_id}")
    runtime_raw, mode, width, height, adapter = v52b._prepare_runtime_raster(raw)
    runtime_sha = sha256(runtime_raw).hexdigest()
    raster = RasterPageInputContract(
        source_id=sample_id,
        source_sha256=selection_row["image_sha256"],
        page_number=1,
        width=width,
        height=height,
        pixel_mode=v52b._mode_to_contract(mode),
        raster_sha256=runtime_sha,
        dpi=None,
    )
    normalized = normalize_raster_page_v1(runtime_raw, raster)
    if normalized.page.status != "accepted" or normalized.normalized_png is None:
        return {
            "gate": "HOLD",
            "reason": "NORMALIZER_NOT_ACCEPTED",
            "pixel_mode_adapter": adapter,
            "runtime_raster_sha256": runtime_sha,
        }
    page = normalized.page
    geometry_input = GeometryInputContract(
        normalized_image_sha256=page.normalized_image_sha256,
        normalizer_config_fingerprint=page.normalizer_config_fingerprint,
        normalized_width=page.normalized_width,
        normalized_height=page.normalized_height,
        transform=page.transform,
    )
    geometry = detect_multistaff_geometry_v2(normalized.normalized_png, geometry_input)
    reasons = list(geometry.page.reasons)
    return {
        "gate": "PASS" if geometry.page.status == "accepted" else "HOLD",
        "geometry_status": geometry.page.status,
        "geometry_reasons": reasons,
        "accepted_staff_count": len(geometry.page.staffs),
        "pixel_mode_adapter": adapter,
        "runtime_raster_sha256": runtime_sha,
        "normalized_image_sha256": page.normalized_image_sha256,
    }


def plan_approved_train_replacements_v1(data_root: str | Path) -> Path:
    """Plan exactly two next-unused same-meter replacements without mutating annotations."""
    root = Path(data_root)
    selection, annotations, preflight_audit = _require_exact_hold_evidence(root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    existing_ids = {row["sample_id"] for row in selection}
    existing_families = {row["family_id"] for row in selection}
    if len(existing_ids) != 300 or len(existing_families) != 300:
        _fail("pre-replacement selection is not 300 unique samples/families")

    planned: list[dict[str, object]] = []
    reserved_ids: set[str] = set()
    reserved_families: set[str] = set()
    for old_id, expected in sorted(APPROVED_HOLDS.items(), key=lambda item: int(item[1]["index"])):
        meter = str(expected["meter"])
        rows = _manifest_train_rows(root, meter)
        candidate = select_next_unused_train_row_v1(
            rows,
            meter=meter,
            existing_sample_ids=existing_ids | reserved_ids,
            existing_family_ids=existing_families | reserved_families,
        )
        new_row = _selection_row_from_manifest(root, candidate, index=int(expected["index"]))
        screen = _screen_page_geometry_v1(root, new_row)
        planned.append({
            "old_sample_id": old_id,
            "old_index": int(expected["index"]),
            "meter": meter,
            "required_old_reason": expected["required_reason"],
            "new_selection_row": new_row,
            "candidate_page_geometry": screen,
        })
        reserved_ids.add(new_row["sample_id"])
        reserved_families.add(new_row["family_id"])

    page_gate = "PASS" if all(item["candidate_page_geometry"]["gate"] == "PASS" for item in planned) else "HOLD"
    payload = {
        "schema": REPLACEMENT_SCHEMA,
        "replacement_rule": REPLACEMENT_RULE,
        "dataset": v51.DATASET_NAME,
        "selection_sha256_before": v52b._sha_file(ann_dir / v52a.SELECTION_NAME),
        "annotation_sha256_before": v52b._sha_file(ann_dir / v52a.ANNOTATION_NAME),
        "human_qa_sha256_before": v52b._sha_file(ann_dir / v52b.HUMAN_QA_NAME),
        "preflight_audit_sha256_before": v52b._sha_file(ann_dir / v52b2.PREFLIGHT_AUDIT_NAME),
        "preflight_gate_before": preflight_audit["gate"],
        "preflight_pass_before": preflight_audit["pass_count"],
        "preflight_hold_before": preflight_audit["hold_count"],
        "replacement_count": 2,
        "planned_replacements": planned,
        "candidate_page_geometry_gate": page_gate,
        "selection_mutation_authorized": page_gate == "PASS",
        "existing_annotation_rows_preserved_after_apply": 298,
        "new_human_bboxes_required_after_apply": 2,
        "training_authorized": False,
        "validation_opened": False,
        "final_holdout_locked": True,
        "frozen_control_specialist": "4-AI",
        "threshold_tuning_allowed": False,
    }
    path = ann_dir / REPLACEMENT_PLAN_NAME
    v51._atomic_write_json(path, payload)
    return path


def _archive_regular_file(source: Path, destination: Path) -> None:
    if source.exists():
        if source.is_symlink() or not source.is_file():
            _fail(f"refusing to archive non-regular evidence file: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def apply_approved_train_replacements_v1(data_root: str | Path) -> Path:
    """Apply the exact PASS replacement plan and leave the two new samples unannotated."""
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    plan_path = ann_dir / REPLACEMENT_PLAN_NAME
    plan = v52b._read_json(plan_path)
    if plan.get("schema") != REPLACEMENT_SCHEMA or plan.get("replacement_rule") != REPLACEMENT_RULE:
        _fail("replacement plan schema/rule changed")
    if plan.get("candidate_page_geometry_gate") != "PASS" or plan.get("selection_mutation_authorized") is not True:
        _fail("replacement plan candidate page geometry is not PASS")

    selection, annotations, _audit = _require_exact_hold_evidence(root)
    if plan.get("selection_sha256_before") != v52b._sha_file(ann_dir / v52a.SELECTION_NAME):
        _fail("selection changed after replacement plan")
    if plan.get("annotation_sha256_before") != v52b._sha_file(ann_dir / v52a.ANNOTATION_NAME):
        _fail("annotations changed after replacement plan")
    if plan.get("preflight_audit_sha256_before") != v52b._sha_file(ann_dir / v52b2.PREFLIGHT_AUDIT_NAME):
        _fail("preflight evidence changed after replacement plan")

    planned = plan.get("planned_replacements")
    if not isinstance(planned, list) or len(planned) != 2:
        _fail("replacement plan does not contain exactly two replacements")

    archive = ann_dir / REPLACEMENT_ARCHIVE_DIR
    archive.mkdir(parents=True, exist_ok=True)
    for name in (
        v52a.SELECTION_NAME,
        v52a.ANNOTATION_NAME,
        v52a.AUDIT_NAME,
        v52b.HUMAN_QA_NAME,
        v52b2.PREFLIGHT_CSV_NAME,
        v52b2.PREFLIGHT_AUDIT_NAME,
    ):
        _archive_regular_file(ann_dir / name, archive / name)
    _archive_regular_file(plan_path, archive / REPLACEMENT_PLAN_NAME)

    selection_by_index = {int(row["index"]): dict(row) for row in selection}
    old_ids: set[str] = set()
    new_ids: set[str] = set()
    for item in planned:
        old_id = str(item["old_sample_id"])
        index = int(item["old_index"])
        new_row = dict(item["new_selection_row"])
        expected = APPROVED_HOLDS.get(old_id)
        if expected is None or int(expected["index"]) != index or expected["meter"] != new_row.get("meter"):
            _fail(f"replacement plan identity/index changed: {old_id}")
        if selection_by_index[index]["sample_id"] != old_id:
            _fail(f"selection row no longer contains approved HOLD: {old_id}")
        if new_row.get("seed_annotation") != "0" or new_row.get("split") != "train":
            _fail(f"replacement candidate is not non-seed TRAIN: {new_row.get('sample_id')}")
        selection_by_index[index] = new_row
        old_ids.add(old_id)
        new_ids.add(new_row["sample_id"])

    new_selection = [selection_by_index[index] for index in range(300)]
    if Counter(row["meter"] for row in new_selection) != Counter({meter: 100 for meter in v51.CLASSES}):
        _fail("replacement changed 100/100/100 class balance")
    if len({row["sample_id"] for row in new_selection}) != 300:
        _fail("replacement selection has duplicate sample ids")
    if len({row["family_id"] for row in new_selection}) != 300:
        _fail("replacement selection has duplicate family ids")
    if any(row["seed_annotation"] != "1" for row in new_selection[:30]):
        _fail("replacement changed diagnostic seed flags")
    if any(row["sample_id"] in old_ids for row in new_selection):
        _fail("replaced HOLD sample still exists in selection")
    if not new_ids.issubset({row["sample_id"] for row in new_selection}):
        _fail("replacement candidates missing from selection")

    retained_annotations = [dict(row) for row in annotations if row["sample_id"] not in old_ids]
    if len(retained_annotations) != 298:
        _fail("replacement must preserve exactly 298 existing annotation rows")
    if any(row["sample_id"] in new_ids for row in retained_annotations):
        _fail("new replacement candidates must remain unannotated for human BBox input")

    v51._atomic_write_csv(ann_dir / v52a.SELECTION_NAME, v52a.SELECTION_COLUMNS, new_selection)
    v51._atomic_write_csv(ann_dir / v52a.ANNOTATION_NAME, v51.ANNOTATION_COLUMNS, retained_annotations)

    invalidated: list[str] = []
    for name in (
        v52a.AUDIT_NAME,
        v52b.HUMAN_QA_NAME,
        v52b2.PREFLIGHT_CSV_NAME,
        v52b2.PREFLIGHT_AUDIT_NAME,
        v52b.SLOT_MANIFEST_NAME,
        v52b.SLOT_AUDIT_NAME,
        v52b.TRAINING_REPORT_NAME,
        v52b.DIAGNOSTIC_REPORT_NAME,
    ):
        path = ann_dir / name
        if path.exists():
            if path.is_symlink() or not path.is_file():
                _fail(f"refusing to invalidate non-regular evidence path: {path}")
            path.unlink()
            invalidated.append(name)

    apply_payload = {
        "schema": REPLACEMENT_SCHEMA,
        "replacement_rule": REPLACEMENT_RULE,
        "plan_sha256": v52b._sha_file(plan_path),
        "selection_sha256_after": v52b._sha_file(ann_dir / v52a.SELECTION_NAME),
        "annotation_sha256_after": v52b._sha_file(ann_dir / v52a.ANNOTATION_NAME),
        "old_sample_ids_removed": sorted(old_ids),
        "new_sample_ids_unannotated": sorted(new_ids),
        "selection_count": 300,
        "annotation_count": 298,
        "preserved_annotation_count": 298,
        "new_human_bboxes_required": 2,
        "diagnostic_seed_count": 30,
        "class_balance": {meter: 100 for meter in v51.CLASSES},
        "stale_evidence_invalidated": sorted(invalidated),
        "archive_dir": archive.relative_to(root).as_posix(),
        "training_authorized": False,
        "validation_opened": False,
        "final_holdout_locked": True,
        "frozen_control_specialist": "4-AI",
        "threshold_tuning_allowed": False,
    }
    apply_path = ann_dir / REPLACEMENT_APPLY_NAME
    v51._atomic_write_json(apply_path, apply_payload)
    return apply_path


def verify_applied_replacements_v1(data_root: str | Path) -> dict[str, object]:
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    payload = v52b._read_json(ann_dir / REPLACEMENT_APPLY_NAME)
    if payload.get("schema") != REPLACEMENT_SCHEMA or payload.get("replacement_rule") != REPLACEMENT_RULE:
        _fail("replacement apply evidence schema/rule changed")
    if payload.get("selection_count") != 300 or payload.get("annotation_count") != 298:
        _fail("replacement apply evidence counts changed")
    if payload.get("new_human_bboxes_required") != 2:
        _fail("replacement apply evidence no longer requires exactly two human BBoxes")
    if payload.get("training_authorized") is not False:
        _fail("replacement apply evidence must keep training closed")
    if payload.get("validation_opened") is not False or payload.get("final_holdout_locked") is not True:
        _fail("replacement apply evidence changed VAL/holdout boundary")
    if payload.get("selection_sha256_after") != v52b._sha_file(ann_dir / v52a.SELECTION_NAME):
        _fail("replacement selection changed after apply")
    if payload.get("annotation_sha256_after") != v52b._sha_file(ann_dir / v52a.ANNOTATION_NAME):
        _fail("replacement annotations changed after apply")
    return payload
