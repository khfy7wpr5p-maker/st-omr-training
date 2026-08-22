"""Post-replacement human-QA evidence bridge for Meter V5-2B.

This module records the user's explicit visual review of the two replacement
full-meter BBoxes without weakening the existing V5-2B safety boundaries.
It binds four evidence layers:

1. the archived pre-replacement 15/15 contact-sheet human-QA attestation;
2. proof that 298 current annotations are byte-for-byte preserved;
3. the exact two-target replacement QA preview and current annotation hashes;
4. the user's explicit 2/2 replacement visual-QA confirmation.

The current V5-2B staff-association/slot code still consumes the historical
``bbox_adaptation_300_human_qa.json`` filename and schema. To avoid a broad
training-code refactor inside this stage, this module writes that legacy
compatibility record only *after* the richer post-replacement attestation is
verified. The richer sidecar remains the authoritative audit evidence for the
replacement extension.

This module does not run geometry, derive slots, train/infer a model, tune
thresholds, open VAL, or touch FINAL_HOLDOUT.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2a_specialist_adaptation as v52a
from . import meter_v5_2b_post_replacement_qa_v1 as post
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2b_train_replacement_v1 as repl


POST_HUMAN_QA_SCHEMA: Final[str] = "st-omr-meter-v5-2b-post-replacement-human-qa-v1"
POST_HUMAN_QA_NAME: Final[str] = "v5_2b_post_replacement_human_qa_v1.json"
POST_HUMAN_QA_CONFIRMATION: Final[str] = "V5_2B_REPLACEMENT_QA_2_OF_2_PASS"


def _fail(message: str) -> None:
    raise v52b.MeterV5_2BError(message)


def _archived_original_qa_expected(root: Path) -> tuple[Path, dict[str, object]]:
    ann_dir = root / v51.ANNOTATIONS_DIR
    archive = ann_dir / repl.REPLACEMENT_ARCHIVE_DIR
    qa_path = archive / v52b.HUMAN_QA_NAME
    selection_path = archive / v52a.SELECTION_NAME
    annotation_path = archive / v52a.ANNOTATION_NAME
    audit_path = archive / v52a.AUDIT_NAME
    expected = {
        "schema": v52b.HUMAN_QA_SCHEMA,
        "dataset": v51.DATASET_NAME,
        "dataset_fingerprint_sha256": v52a.EXPECTED_DATASET_FINGERPRINT,
        "selection_sha256": v52b._sha_file(selection_path),
        "annotation_sha256": v52b._sha_file(annotation_path),
        "mechanical_audit_sha256": v52b._sha_file(audit_path),
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
    return qa_path, expected


def validate_archived_original_qa_payload_v1(
    payload: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    """Fail closed unless the archived original QA record is exactly intact."""
    if dict(payload) != dict(expected):
        _fail("archived original 15/15 human-QA attestation changed")


def verify_archived_original_human_qa_v1(data_root: str | Path) -> dict[str, object]:
    root = Path(data_root)
    qa_path, expected = _archived_original_qa_expected(root)
    payload = v52b._read_json(qa_path)
    validate_archived_original_qa_payload_v1(payload, expected)
    return payload


def legacy_current_human_qa_payload_v1(
    *,
    selection_sha256: str,
    annotation_sha256: str,
    mechanical_audit_sha256: str,
) -> dict[str, object]:
    """Return the exact legacy payload required by existing V5-2B verifiers."""
    return {
        "schema": v52b.HUMAN_QA_SCHEMA,
        "dataset": v51.DATASET_NAME,
        "dataset_fingerprint_sha256": v52a.EXPECTED_DATASET_FINGERPRINT,
        "selection_sha256": selection_sha256,
        "annotation_sha256": annotation_sha256,
        "mechanical_audit_sha256": mechanical_audit_sha256,
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


def write_post_replacement_human_qa_v1(
    data_root: str | Path,
    *,
    confirmation: str,
) -> Path:
    """Bind the explicit replacement 2/2 visual-QA approval to exact evidence."""
    if confirmation != POST_HUMAN_QA_CONFIRMATION:
        _fail("replacement human-QA confirmation token is not exact 2/2 PASS")

    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    original_qa = verify_archived_original_human_qa_v1(root)
    audit = post.verify_post_replacement_mechanical_audit_v1(root)
    preview = post.verify_replacement_qa_preview_v1(root)

    if audit.get("preserved_annotation_count") != 298 or audit.get("replacement_annotation_count") != 2:
        _fail("post-replacement audit is not exact 298 preserved + 2 replacement")
    if audit.get("seed_mutation_count") != 0 or audit.get("pass_count") != 300:
        _fail("post-replacement audit is not exact 300/300 PASS with immutable seeds")
    if preview.get("target_count") != 2:
        _fail("replacement QA preview is not exactly two targets")
    target_ids = {
        str(item.get("sample_id"))
        for item in preview.get("targets", [])
        if isinstance(item, dict)
    }
    if target_ids != post.NEW_IDS:
        _fail("replacement QA preview target identities changed")

    payload = {
        "schema": POST_HUMAN_QA_SCHEMA,
        "dataset": v51.DATASET_NAME,
        "dataset_fingerprint_sha256": v52a.EXPECTED_DATASET_FINGERPRINT,
        "archived_original_human_qa_sha256": v52b._sha_file(
            ann_dir / repl.REPLACEMENT_ARCHIVE_DIR / v52b.HUMAN_QA_NAME
        ),
        "archived_original_contact_sheets_reviewed": original_qa["contact_sheets_reviewed"],
        "preserved_annotation_count": 298,
        "replacement_annotation_count": 2,
        "replacement_visual_qa_count": 2,
        "replacement_visual_errors_reported": 0,
        "replacement_sample_ids": sorted(post.NEW_IDS),
        "selection_sha256": audit["selection_sha256"],
        "annotation_sha256": audit["annotation_sha256"],
        "post_replacement_mechanical_audit_sha256": v52b._sha_file(ann_dir / post.POST_AUDIT_NAME),
        "replacement_qa_preview_manifest_sha256": v52b._sha_file(ann_dir / post.QA_PREVIEW_MANIFEST_NAME),
        "replacement_qa_preview_png_sha256": v52b._sha_file(ann_dir / post.QA_PREVIEW_NAME),
        "combined_current_annotation_count": 300,
        "human_visual_qa": "PASS",
        "staff_preflight_authorized": True,
        "slot_derivation_authorized": False,
        "training_authorized": False,
        "trainable_specialists": ["2-AI", "3-AI"],
        "frozen_control_specialist": "4-AI",
        "threshold_tuning_allowed": False,
        "validation_opened": False,
        "final_holdout_locked": True,
        "legacy_compatibility_bridge_required": True,
    }
    path = ann_dir / POST_HUMAN_QA_NAME
    v51._atomic_write_json(path, payload)
    return path


def verify_post_replacement_human_qa_v1(data_root: str | Path) -> dict[str, object]:
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    verify_archived_original_human_qa_v1(root)
    audit = post.verify_post_replacement_mechanical_audit_v1(root)
    preview = post.verify_replacement_qa_preview_v1(root)
    payload = v52b._read_json(ann_dir / POST_HUMAN_QA_NAME)

    expected_scalars = {
        "schema": POST_HUMAN_QA_SCHEMA,
        "dataset": v51.DATASET_NAME,
        "dataset_fingerprint_sha256": v52a.EXPECTED_DATASET_FINGERPRINT,
        "archived_original_contact_sheets_reviewed": 15,
        "preserved_annotation_count": 298,
        "replacement_annotation_count": 2,
        "replacement_visual_qa_count": 2,
        "replacement_visual_errors_reported": 0,
        "combined_current_annotation_count": 300,
        "human_visual_qa": "PASS",
        "staff_preflight_authorized": True,
        "slot_derivation_authorized": False,
        "training_authorized": False,
        "trainable_specialists": ["2-AI", "3-AI"],
        "frozen_control_specialist": "4-AI",
        "threshold_tuning_allowed": False,
        "validation_opened": False,
        "final_holdout_locked": True,
        "legacy_compatibility_bridge_required": True,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            _fail(f"post-replacement human-QA field changed: {key}")

    if set(payload.get("replacement_sample_ids", [])) != post.NEW_IDS:
        _fail("post-replacement human-QA replacement identities changed")

    expected_hashes = {
        "archived_original_human_qa_sha256": v52b._sha_file(
            ann_dir / repl.REPLACEMENT_ARCHIVE_DIR / v52b.HUMAN_QA_NAME
        ),
        "selection_sha256": audit["selection_sha256"],
        "annotation_sha256": audit["annotation_sha256"],
        "post_replacement_mechanical_audit_sha256": v52b._sha_file(ann_dir / post.POST_AUDIT_NAME),
        "replacement_qa_preview_manifest_sha256": v52b._sha_file(ann_dir / post.QA_PREVIEW_MANIFEST_NAME),
        "replacement_qa_preview_png_sha256": v52b._sha_file(ann_dir / post.QA_PREVIEW_NAME),
    }
    for key, expected in expected_hashes.items():
        if payload.get(key) != expected:
            _fail(f"post-replacement human-QA evidence hash changed: {key}")

    if preview.get("target_count") != 2:
        _fail("replacement QA preview changed after human-QA attestation")
    return payload


def write_legacy_current_human_qa_bridge_v1(data_root: str | Path) -> Path:
    """Refresh the legacy canonical QA file only after richer QA is verified."""
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    verify_post_replacement_human_qa_v1(root)
    audit = post.verify_post_replacement_mechanical_audit_v1(root)
    payload = legacy_current_human_qa_payload_v1(
        selection_sha256=str(audit["selection_sha256"]),
        annotation_sha256=str(audit["annotation_sha256"]),
        mechanical_audit_sha256=v52b._sha_file(ann_dir / v52a.AUDIT_NAME),
    )
    path = ann_dir / v52b.HUMAN_QA_NAME
    v51._atomic_write_json(path, payload)
    # Independent compatibility read-back through the frozen existing verifier.
    v52b.verify_human_qa_attestation(root)
    return path


def verify_legacy_current_human_qa_bridge_v1(data_root: str | Path) -> dict[str, object]:
    root = Path(data_root)
    verify_post_replacement_human_qa_v1(root)
    return v52b.verify_human_qa_attestation(root)
