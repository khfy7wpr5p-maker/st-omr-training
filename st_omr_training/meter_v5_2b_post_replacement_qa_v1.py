"""Post-replacement evidence gate for Meter V5-2B.

This module does not train, infer, derive digit slots, change geometry, or open
VAL/final holdout. It verifies that the approved two-row TRAIN replacement left
298 prior human annotations byte-for-byte unchanged, that the two new human
full-meter BBoxes are the only new annotations, and that the first 30 diagnostic
seeds remain immutable. It then produces a two-sample visual QA sheet showing
exactly the two replacement BBoxes.
"""
from __future__ import annotations

import io
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Final, Mapping, Sequence

from PIL import Image, ImageDraw

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2a_specialist_adaptation as v52a
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2b_train_replacement_v1 as repl


POST_AUDIT_SCHEMA: Final[str] = "st-omr-meter-v5-2b-post-replacement-annotation-audit-v1"
POST_AUDIT_NAME: Final[str] = "v5_2b_post_replacement_annotation_audit_v1.json"
QA_PREVIEW_SCHEMA: Final[str] = "st-omr-meter-v5-2b-replacement-qa-preview-v1"
QA_PREVIEW_NAME: Final[str] = "v5_2b_replacement_qa_2_of_2.png"
QA_PREVIEW_MANIFEST_NAME: Final[str] = "v5_2b_replacement_qa_2_of_2.json"

TARGETS: Final[tuple[tuple[int, str, str, str], ...]] = (
    (63, "150200092-1_1_1", "2/4", "150201200-1_1_1"),
    (125, "150207112-1_1_1", "3/4", "110003725-1_1_1"),
)
NEW_IDS: Final[frozenset[str]] = frozenset(item[1] for item in TARGETS)
OLD_IDS: Final[frozenset[str]] = frozenset(item[3] for item in TARGETS)


def _fail(message: str) -> None:
    raise v52b.MeterV5_2BError(message)


def _row_map(rows: Sequence[Mapping[str, str]], *, label: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in result:
            _fail(f"{label}: duplicate/empty sample_id: {sample_id}")
        result[sample_id] = dict(row)
    return result


def _validate_target_annotation(row: Mapping[str, str], selection: Mapping[str, str]) -> dict[str, int]:
    sample_id = selection["sample_id"]
    if row.get("status") != "PASS":
        _fail(f"replacement annotation is not PASS: {sample_id}")
    if row.get("meter") != selection.get("meter") or row.get("split") != "train":
        _fail(f"replacement annotation identity mismatch: {sample_id}")
    if row.get("image_sha256") != selection.get("image_sha256"):
        _fail(f"replacement annotation image SHA mismatch: {sample_id}")
    try:
        image_width = int(selection["image_width"])
        image_height = int(selection["image_height"])
        if int(row["image_width"]) != image_width or int(row["image_height"]) != image_height:
            _fail(f"replacement annotation dimensions mismatch: {sample_id}")
        x, y, w, h = (int(row[key]) for key in ("x", "y", "w", "h"))
    except (KeyError, ValueError) as exc:
        raise v52b.MeterV5_2BError(f"replacement annotation numeric field invalid: {sample_id}") from exc
    v51._validate_bbox(x, y, w, h, image_width, image_height)
    return {"x": x, "y": y, "w": w, "h": h}


def validate_post_replacement_rows_v1(
    *,
    selection_rows: Sequence[Mapping[str, str]],
    annotation_rows: Sequence[Mapping[str, str]],
    archived_annotation_rows: Sequence[Mapping[str, str]],
    seed_sample_ids: Sequence[str],
    seed_annotation_rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    """Pure fail-closed validator for the 298-preserved + 2-new annotation state."""
    if len(selection_rows) != 300 or len(annotation_rows) != 300 or len(archived_annotation_rows) != 300:
        _fail("post-replacement gate requires 300 selection, 300 current annotations, and 300 archived annotations")

    selection = _row_map(selection_rows, label="selection")
    annotations = _row_map(annotation_rows, label="current annotations")
    archived = _row_map(archived_annotation_rows, label="archived annotations")
    if set(selection) != set(annotations):
        _fail("current selection/annotation identities differ")
    if OLD_IDS & set(selection):
        _fail("approved old HOLD identities remain in current selection")
    if not NEW_IDS.issubset(selection):
        _fail("one or both approved replacement identities are missing")
    if not OLD_IDS.issubset(archived) or NEW_IDS & set(archived):
        _fail("pre-replacement archive identities are inconsistent")

    ordered_selection = sorted(selection_rows, key=lambda row: int(row["index"]))
    if [int(row["index"]) for row in ordered_selection] != list(range(300)):
        _fail("selection index order is not exactly 0..299")
    if any(row.get("split") != "train" for row in ordered_selection):
        _fail("post-replacement selection must remain TRAIN-only")
    if Counter(row.get("meter") for row in ordered_selection) != Counter({"2/4": 100, "3/4": 100, "4/4": 100}):
        _fail("post-replacement selection class balance changed")
    if any(row.get("seed_annotation") != "1" for row in ordered_selection[:30]):
        _fail("first 30 diagnostic seed flags changed")
    if any(row.get("seed_annotation") != "0" for row in ordered_selection[30:]):
        _fail("non-seed row marked as diagnostic seed")

    selection_by_index = {int(row["index"]): dict(row) for row in ordered_selection}
    target_bboxes: dict[str, dict[str, int]] = {}
    for index, new_id, meter, old_id in TARGETS:
        selected = selection_by_index.get(index)
        if selected is None or selected.get("sample_id") != new_id or selected.get("meter") != meter:
            _fail(f"approved replacement binding changed at index {index}")
        if selected.get("seed_annotation") != "0":
            _fail(f"replacement target became a diagnostic seed: {new_id}")
        if old_id not in archived:
            _fail(f"archived old replacement identity missing: {old_id}")
        target_bboxes[new_id] = _validate_target_annotation(annotations[new_id], selected)

    preserved_current_ids = set(selection) - NEW_IDS
    preserved_archived_ids = set(archived) - OLD_IDS
    if preserved_current_ids != preserved_archived_ids or len(preserved_current_ids) != 298:
        _fail("preserved 298-annotation identity set changed")
    for sample_id in sorted(preserved_current_ids):
        if annotations[sample_id] != archived[sample_id]:
            _fail(f"preserved annotation changed after replacement: {sample_id}")

    seed_ids = list(seed_sample_ids)
    if len(seed_ids) != 30 or len(set(seed_ids)) != 30:
        _fail("diagnostic seed identity set is not exactly 30 unique samples")
    if [row["sample_id"] for row in ordered_selection[:30]] != seed_ids:
        _fail("first 30 selection rows no longer match immutable diagnostic seed order")
    seed_annotations = _row_map(seed_annotation_rows, label="seed annotations")
    if set(seed_annotations) != set(seed_ids):
        _fail("seed annotation evidence identity set changed")
    seed_mutations = sum(annotations[sample_id] != seed_annotations[sample_id] for sample_id in seed_ids)
    if seed_mutations != 0:
        _fail("one or more immutable diagnostic seed annotations changed")

    per_class = {
        meter: {
            "PASS": sum(row.get("meter") == meter and row.get("status") == "PASS" for row in annotation_rows),
            "REVIEW": sum(row.get("meter") == meter and row.get("status") == "REVIEW" for row in annotation_rows),
        }
        for meter in ("2/4", "3/4", "4/4")
    }
    expected_per_class = {meter: {"PASS": 100, "REVIEW": 0} for meter in ("2/4", "3/4", "4/4")}
    if per_class != expected_per_class:
        _fail(f"post-replacement annotation class/status balance changed: {per_class}")

    return {
        "selection_count": 300,
        "annotation_count": 300,
        "pass_count": 300,
        "review_count": 0,
        "preserved_annotation_count": 298,
        "replacement_annotation_count": 2,
        "seed_mutation_count": 0,
        "per_class": per_class,
        "target_bboxes": target_bboxes,
    }


def _verify_apply_selection_binding(root: Path) -> dict[str, object]:
    ann_dir = root / v51.ANNOTATIONS_DIR
    apply = v52b._read_json(ann_dir / repl.REPLACEMENT_APPLY_NAME)
    if apply.get("schema") != repl.REPLACEMENT_SCHEMA or apply.get("replacement_rule") != repl.REPLACEMENT_RULE:
        _fail("replacement apply schema/rule changed")
    if apply.get("selection_count") != 300 or apply.get("preserved_annotation_count") != 298:
        _fail("replacement apply evidence counts changed")
    if set(apply.get("old_sample_ids_removed", [])) != OLD_IDS:
        _fail("replacement apply old identities changed")
    if set(apply.get("new_sample_ids_unannotated", [])) != NEW_IDS:
        _fail("replacement apply new identities changed")
    if apply.get("selection_sha256_after") != v52b._sha_file(ann_dir / v52a.SELECTION_NAME):
        _fail("selection changed after approved replacement apply")
    if apply.get("training_authorized") is not False:
        _fail("replacement apply unexpectedly opened training")
    if apply.get("validation_opened") is not False or apply.get("final_holdout_locked") is not True:
        _fail("replacement apply changed VAL/final-holdout boundary")
    return apply


def write_post_replacement_mechanical_audit_v1(data_root: str | Path) -> Path:
    """Write exact 300/300 mechanical evidence without deriving slots or running models."""
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    apply = _verify_apply_selection_binding(root)
    selection_path = ann_dir / v52a.SELECTION_NAME
    annotation_path = ann_dir / v52a.ANNOTATION_NAME
    archive_annotation_path = ann_dir / repl.REPLACEMENT_ARCHIVE_DIR / v52a.ANNOTATION_NAME
    selection_rows = v52b._read_csv(selection_path)
    annotation_rows = v52b._read_csv(annotation_path)
    archived_annotation_rows = v52b._read_csv(archive_annotation_path)
    seed = v52a.verify_seed_evidence(root)
    summary = validate_post_replacement_rows_v1(
        selection_rows=selection_rows,
        annotation_rows=annotation_rows,
        archived_annotation_rows=archived_annotation_rows,
        seed_sample_ids=list(seed["seed_sample_ids"]),
        seed_annotation_rows=list(seed["annotation_rows"]),
    )

    standard_audit = {
        "schema": "st-omr-meter-v5-2a-annotation-audit-v1",
        "dataset": v51.DATASET_NAME,
        "dataset_fingerprint_sha256": v52a.EXPECTED_DATASET_FINGERPRINT,
        "train_total_expected": 300,
        "annotation_count": 300,
        "pass_count": 300,
        "review_count": 0,
        "per_class": summary["per_class"],
        "missing_annotation_count": 0,
        "seed_mutation_count": 0,
        "mechanical_gate": "PASS",
        "human_visual_review_required": True,
        "training_authorized": False,
        "slot_derivation_authorized_after_human_qa": True,
        "trainable_specialists": ["2-AI", "3-AI"],
        "frozen_control_specialist": "4-AI",
        "threshold_tuning_allowed": False,
        "single_three_class_model_allowed": False,
        "validation_opened": False,
        "final_holdout_locked": True,
        "model_opened": False,
        "inference_count": 0,
    }
    standard_path = ann_dir / v52a.AUDIT_NAME
    v51._atomic_write_json(standard_path, standard_audit)

    payload = {
        "schema": POST_AUDIT_SCHEMA,
        "dataset": v51.DATASET_NAME,
        "dataset_fingerprint_sha256": v52a.EXPECTED_DATASET_FINGERPRINT,
        "replacement_apply_sha256": v52b._sha_file(ann_dir / repl.REPLACEMENT_APPLY_NAME),
        "replacement_apply_selection_sha256": apply["selection_sha256_after"],
        "selection_sha256": v52b._sha_file(selection_path),
        "annotation_sha256": v52b._sha_file(annotation_path),
        "archived_annotation_sha256": v52b._sha_file(archive_annotation_path),
        "standard_mechanical_audit_sha256": v52b._sha_file(standard_path),
        "selection_count": summary["selection_count"],
        "annotation_count": summary["annotation_count"],
        "pass_count": summary["pass_count"],
        "review_count": summary["review_count"],
        "preserved_annotation_count": summary["preserved_annotation_count"],
        "replacement_annotation_count": summary["replacement_annotation_count"],
        "replacement_sample_ids": sorted(NEW_IDS),
        "removed_sample_ids": sorted(OLD_IDS),
        "seed_mutation_count": 0,
        "per_class": summary["per_class"],
        "mechanical_gate": "PASS",
        "human_visual_qa_required_for_replacements": True,
        "slot_derivation_authorized": False,
        "training_authorized": False,
        "validation_opened": False,
        "final_holdout_locked": True,
        "frozen_control_specialist": "4-AI",
    }
    path = ann_dir / POST_AUDIT_NAME
    v51._atomic_write_json(path, payload)
    return path


def verify_post_replacement_mechanical_audit_v1(data_root: str | Path) -> dict[str, object]:
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    _verify_apply_selection_binding(root)
    payload = v52b._read_json(ann_dir / POST_AUDIT_NAME)
    if payload.get("schema") != POST_AUDIT_SCHEMA or payload.get("mechanical_gate") != "PASS":
        _fail("post-replacement mechanical audit is not PASS")
    expected_hashes = {
        "selection_sha256": v52b._sha_file(ann_dir / v52a.SELECTION_NAME),
        "annotation_sha256": v52b._sha_file(ann_dir / v52a.ANNOTATION_NAME),
        "archived_annotation_sha256": v52b._sha_file(ann_dir / repl.REPLACEMENT_ARCHIVE_DIR / v52a.ANNOTATION_NAME),
        "standard_mechanical_audit_sha256": v52b._sha_file(ann_dir / v52a.AUDIT_NAME),
        "replacement_apply_sha256": v52b._sha_file(ann_dir / repl.REPLACEMENT_APPLY_NAME),
    }
    for key, expected in expected_hashes.items():
        if payload.get(key) != expected:
            _fail(f"post-replacement mechanical audit hash changed: {key}")
    expected_scalars = {
        "selection_count": 300,
        "annotation_count": 300,
        "pass_count": 300,
        "review_count": 0,
        "preserved_annotation_count": 298,
        "replacement_annotation_count": 2,
        "seed_mutation_count": 0,
        "slot_derivation_authorized": False,
        "training_authorized": False,
        "validation_opened": False,
        "final_holdout_locked": True,
        "frozen_control_specialist": "4-AI",
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            _fail(f"post-replacement mechanical audit field changed: {key}")
    if set(payload.get("replacement_sample_ids", [])) != NEW_IDS or set(payload.get("removed_sample_ids", [])) != OLD_IDS:
        _fail("post-replacement mechanical audit target identities changed")
    return payload


def write_replacement_qa_preview_v1(data_root: str | Path) -> Path:
    """Render only the two existing human BBoxes for visual QA; never alter labels."""
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    post_audit = verify_post_replacement_mechanical_audit_v1(root)
    selection_rows = v52b._read_csv(ann_dir / v52a.SELECTION_NAME)
    annotation_rows = v52b._read_csv(ann_dir / v52a.ANNOTATION_NAME)
    selection_by_id = {row["sample_id"]: row for row in selection_rows}
    annotation_by_id = {row["sample_id"]: row for row in annotation_rows}

    panels: list[Image.Image] = []
    manifest_targets: list[dict[str, object]] = []
    max_width = 0
    for index, sample_id, meter, _old_id in TARGETS:
        selected = selection_by_id[sample_id]
        annotation = annotation_by_id[sample_id]
        bbox = _validate_target_annotation(annotation, selected)
        image_path = root / selected["image_relpath"]
        image_sha, width, height = v51._read_png_binding(image_path)
        if image_sha != selected["image_sha256"] or width != int(selected["image_width"]) or height != int(selected["image_height"]):
            _fail(f"replacement QA source binding changed: {sample_id}")
        with Image.open(image_path) as source:
            source.load()
            panel = source.convert("RGB")
        draw = ImageDraw.Draw(panel)
        x0, y0 = bbox["x"], bbox["y"]
        x1, y1 = x0 + bbox["w"], y0 + bbox["h"]
        draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=4)
        panels.append(panel)
        max_width = max(max_width, panel.width)
        manifest_targets.append({
            "index": index,
            "sample_id": sample_id,
            "meter": meter,
            "source_image_sha256": image_sha,
            "bbox": bbox,
        })

    header_height = 34
    gap = 18
    total_height = sum(panel.height + header_height for panel in panels) + gap * (len(panels) - 1)
    sheet = Image.new("RGB", (max_width, total_height), "white")
    draw = ImageDraw.Draw(sheet)
    y_cursor = 0
    for panel, target in zip(panels, manifest_targets):
        draw.text((8, y_cursor + 8), f"index {target['index']} | {target['meter']} | {target['sample_id']}", fill="black")
        y_cursor += header_height
        sheet.paste(panel, (0, y_cursor))
        y_cursor += panel.height + gap

    output = io.BytesIO()
    sheet.save(output, format="PNG", optimize=False, compress_level=9)
    png_path = ann_dir / QA_PREVIEW_NAME
    v52b._atomic_bytes(png_path, output.getvalue())
    manifest = {
        "schema": QA_PREVIEW_SCHEMA,
        "post_replacement_audit_sha256": v52b._sha_file(ann_dir / POST_AUDIT_NAME),
        "selection_sha256": post_audit["selection_sha256"],
        "annotation_sha256": post_audit["annotation_sha256"],
        "preview_png_sha256": v52b._sha_file(png_path),
        "target_count": 2,
        "targets": manifest_targets,
        "visualization_only": True,
        "annotation_mutation": False,
        "slot_derivation": False,
        "training_authorized": False,
        "validation_opened": False,
        "final_holdout_locked": True,
        "frozen_control_specialist": "4-AI",
    }
    v51._atomic_write_json(ann_dir / QA_PREVIEW_MANIFEST_NAME, manifest)
    return png_path


def verify_replacement_qa_preview_v1(data_root: str | Path) -> dict[str, object]:
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    post = verify_post_replacement_mechanical_audit_v1(root)
    manifest = v52b._read_json(ann_dir / QA_PREVIEW_MANIFEST_NAME)
    if manifest.get("schema") != QA_PREVIEW_SCHEMA or manifest.get("target_count") != 2:
        _fail("replacement QA preview manifest changed")
    expected = {
        "post_replacement_audit_sha256": v52b._sha_file(ann_dir / POST_AUDIT_NAME),
        "selection_sha256": post["selection_sha256"],
        "annotation_sha256": post["annotation_sha256"],
        "preview_png_sha256": v52b._sha_file(ann_dir / QA_PREVIEW_NAME),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            _fail(f"replacement QA preview evidence hash changed: {key}")
    if manifest.get("visualization_only") is not True or manifest.get("annotation_mutation") is not False:
        _fail("replacement QA preview is not visualization-only")
    if manifest.get("slot_derivation") is not False or manifest.get("training_authorized") is not False:
        _fail("replacement QA preview unexpectedly opened downstream work")
    if manifest.get("validation_opened") is not False or manifest.get("final_holdout_locked") is not True:
        _fail("replacement QA preview changed VAL/final-holdout boundary")
    target_ids = {str(item.get("sample_id")) for item in manifest.get("targets", []) if isinstance(item, dict)}
    if target_ids != NEW_IDS:
        _fail("replacement QA preview identities changed")
    return manifest
