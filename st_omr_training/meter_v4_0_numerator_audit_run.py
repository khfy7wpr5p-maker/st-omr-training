"""Execution helper for the bounded Meter V4-0 numerator representation audit.

The runner replays the approved Teacher Gold admission split from pilot/choices
metadata, decodes only the 27 positive adaptation-TRAIN source images, derives
numerator crops, and runs the zero-training family-disjoint centroid probe.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Final

from PIL import Image, ImageDraw

from .meter_teacher_gold_admission_v1 import (
    ALLOWED_USE,
    CHOICES_SCHEMA,
    EXPECTED_SOURCES,
    EXPECTED_TASKS,
    METER_CLASSES,
    METER_TEACHER_GOLD_ADMISSION_V1,
    PILOT_SCHEMA,
    _adaptation_split_by_family,
    _bounded_ascii,
    _canonical_json as _teacher_canonical_json,
    _decode_source_png,
    _json_file,
    _map_bbox,
    _mapping,
    _render_roi,
    _sequence,
    _sha as _teacher_sha,
    _validate_permission,
    _validate_privacy,
    _xywh,
)
from .meter_v4_0_numerator_audit import (
    AuditRecordIdentityV4_0,
    FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0,
    METER_TO_NUMERATOR,
    METER_V4_0_NUMERATOR_AUDIT,
    audit_decision_v4_0,
    centroid_oof_probe_v4_0,
    fold_plan_v4_0,
    normalized_ink_vector_v4_0,
    numerator_crop_bounds_v4_0,
    render_numerator_crop_v4_0,
)


RESULT_SCHEMA: Final[str] = "st-omr-meter-v4-0-numerator-representation-audit-result-v1"
_MAX_PILOT_BYTES: Final[int] = 32 * 1024 * 1024
_MAX_CHOICES_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_EVIDENCE_BYTES: Final[int] = 64 * 1024


class MeterV4_0AuditRunError(RuntimeError):
    """Raised when V4-0 execution provenance or artifacts fail closed."""


def _fail(message: str) -> None:
    raise MeterV4_0AuditRunError(message)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _hex64(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        _fail(f"{name} must be canonical lowercase SHA-256")
    return value


def _validate_and_select_source(
    *,
    pilot_path: Path,
    choices_path: Path,
    permission_path: Path,
    privacy_path: Path,
) -> tuple[tuple[tuple[Mapping[str, object], Mapping[str, object], str], ...], dict[str, str]]:
    """Return exactly 27 positive adaptation-TRAIN task/answer pairs without decoding validation images."""
    pilot, pilot_raw = _json_file(pilot_path, maximum=_MAX_PILOT_BYTES, name="V4-0 pilot data")
    choices, choices_raw = _json_file(choices_path, maximum=_MAX_CHOICES_BYTES, name="V4-0 review choices")
    permission, permission_raw = _json_file(
        permission_path, maximum=_MAX_EVIDENCE_BYTES, name="V4-0 training permission evidence", canonical=True
    )
    privacy, privacy_raw = _json_file(
        privacy_path, maximum=_MAX_EVIDENCE_BYTES, name="V4-0 privacy review evidence", canonical=True
    )
    _validate_permission(permission)
    _validate_privacy(privacy)
    if permission.get("allowed_use") != ALLOWED_USE:
        _fail("V4-0 permission use differs from the approved offline Meter pilot")
    if pilot.get("schema") != PILOT_SCHEMA or choices.get("schema") != CHOICES_SCHEMA:
        _fail("V4-0 pilot/choices schema mismatch")
    if pilot.get("source") != "METER_V1/01_REVIEW/train":
        _fail("V4-0 pilot source must remain METER_V1 TRAIN review surface")
    selection = _mapping("V4-0 pilot selection", pilot.get("selection"))
    if selection.get("test_opened") is not False or choices.get("test_opened") is not False:
        _fail("sealed TEST evidence reached V4-0")
    if choices.get("task_count") != EXPECTED_TASKS or choices.get("answered_count") != EXPECTED_TASKS:
        _fail("V4-0 requires the exact 72 answered Teacher Gold tasks")

    tasks_raw = _sequence("V4-0 pilot tasks", pilot.get("tasks"))
    answers_raw = _sequence("V4-0 review answers", choices.get("answers"))
    if len(tasks_raw) != EXPECTED_TASKS or len(answers_raw) != EXPECTED_TASKS:
        _fail("V4-0 Teacher Gold task cardinality changed")
    tasks = [_mapping(f"V4-0 task[{index}]", item) for index, item in enumerate(tasks_raw)]
    answers = [_mapping(f"V4-0 answer[{index}]", item) for index, item in enumerate(answers_raw)]
    task_ids = [_bounded_ascii("V4-0 task_id", task.get("task_id")) for task in tasks]
    answer_ids = [_bounded_ascii("V4-0 answer task_id", answer.get("task_id")) for answer in answers]
    if len(set(task_ids)) != EXPECTED_TASKS or len(set(answer_ids)) != EXPECTED_TASKS or set(task_ids) != set(answer_ids):
        _fail("V4-0 pilot/answer task identities must be unique and identical")
    task_by_id = dict(zip(task_ids, tasks))
    answer_by_id = dict(zip(answer_ids, answers))

    if Counter(task.get("kind") for task in tasks) != {"positive": 36, "none": 36}:
        _fail("V4-0 pilot must remain 36 positive + 36 none")
    if Counter(task.get("expected_class") for task in tasks if task.get("kind") == "positive") != {
        "2/4": 12,
        "3/4": 12,
        "4/4": 12,
    }:
        _fail("V4-0 positive pilot classes changed")

    source_tasks: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for task in tasks:
        source_id = _bounded_ascii("V4-0 source_id", task.get("source_id"))
        if task.get("split") != "train":
            _fail("V4-0 may consume only source TRAIN")
        if task.get("expected_class") not in METER_CLASSES:
            _fail("V4-0 task class is outside Meter classes")
        source_tasks[source_id].append(task)
    if len(source_tasks) != EXPECTED_SOURCES:
        _fail("V4-0 pilot must remain 36 source families")
    for source_id, paired in source_tasks.items():
        if Counter(task.get("kind") for task in paired) != {"positive": 1, "none": 1}:
            _fail(f"V4-0 source {source_id} lost its positive/none pair")
        if len({task.get("family_key") for task in paired}) != 1:
            _fail("V4-0 paired tasks disagree on family identity")
        # Compare encoded source bytes without decoding either task here.
        if paired[0].get("image_data_uri") != paired[1].get("image_data_uri"):
            _fail("V4-0 paired tasks disagree on source image data URI")

    family_splits = _adaptation_split_by_family(tasks)
    selected: list[tuple[Mapping[str, object], Mapping[str, object], str]] = []
    validation_positive_count = 0
    for task_id in sorted(task_ids):
        task = task_by_id[task_id]
        answer = answer_by_id[task_id]
        family = _bounded_ascii("V4-0 family_key", task.get("family_key"))
        adaptation_split = family_splits[family]
        if task.get("kind") != "positive":
            continue
        if adaptation_split == "validation":
            validation_positive_count += 1
            continue
        if adaptation_split != "train":
            _fail("V4-0 adaptation split is outside train/validation")
        source_id = _bounded_ascii("V4-0 selected source_id", task.get("source_id"))
        expected = task.get("expected_class")
        if expected not in METER_TO_NUMERATOR:
            _fail("V4-0 selected positive class must be 2/4, 3/4, or 4/4")
        if answer.get("source_id") != source_id or answer.get("split") != "train" or answer.get("kind") != "positive":
            _fail("V4-0 selected answer identity differs from task")
        if answer.get("status") != "accepted" or answer.get("label_confirmed") is not True or answer.get("crop_usable") is not True:
            _fail("V4-0 accepts only explicitly confirmed usable Teacher Gold positives")
        if answer.get("expected_class") != expected or answer.get("label") != expected:
            _fail("V4-0 selected answer label differs from expected class")
        # Validate selected geometry before any image decoding.
        _xywh("V4-0 selected roi_crop_box", answer.get("roi_crop_box"))
        _xywh("V4-0 selected bbox", answer.get("bbox"))
        selected.append((task, answer, adaptation_split))

    if validation_positive_count != 9:
        _fail("V4-0 expected exactly 9 positive adaptation-validation families")
    if len(selected) != 27:
        _fail("V4-0 requires exactly 27 positive adaptation-TRAIN families")
    if Counter(task.get("expected_class") for task, _answer, _split in selected) != {"2/4": 9, "3/4": 9, "4/4": 9}:
        _fail("V4-0 selected positive TRAIN classes must be 9/9/9")
    if len({_bounded_ascii("V4-0 selected family", task.get("family_key")) for task, _answer, _split in selected}) != 27:
        _fail("V4-0 requires one selected positive per family")

    provenance = {
        "pilot_sha256": _teacher_sha(pilot_raw),
        "choices_sha256": _teacher_sha(choices_raw),
        "permission_sha256": _teacher_sha(permission_raw),
        "privacy_sha256": _teacher_sha(privacy_raw),
    }
    return tuple(selected), provenance


def _derive_selected_crop(
    task: Mapping[str, object],
    answer: Mapping[str, object],
    adaptation_split: str,
):
    """Decode and transform exactly one selected positive TRAIN source image."""
    task_id = _bounded_ascii("V4-0 selected task_id", task.get("task_id"))
    source_id = _bounded_ascii("V4-0 selected source_id", task.get("source_id"))
    family = _bounded_ascii("V4-0 selected family", task.get("family_key"))
    expected = str(task.get("expected_class"))
    source_image, source_raw = _decode_source_png(task.get("image_data_uri"))
    roi_box = _xywh("V4-0 selected roi_crop_box", answer.get("roi_crop_box"))
    full_bbox = _xywh("V4-0 selected bbox", answer.get("bbox"))
    roi_raw, transform = _render_roi(source_image, roi_box)
    mapped_bbox = _map_bbox(full_bbox, transform)
    record_id = _teacher_sha(
        _teacher_canonical_json(
            {
                "version": METER_TEACHER_GOLD_ADMISSION_V1,
                "task_id": task_id,
                "source_id": source_id,
                "source_image_sha256": _teacher_sha(source_raw),
                "adaptation_split": adaptation_split,
                "meter_class": expected,
                "roi_transform": transform,
                "meter_bbox": mapped_bbox,
            }
        )
    )
    with Image.open(BytesIO(roi_raw)) as opened:
        opened.load()
        if opened.format != "PNG" or opened.mode != "L" or opened.size != (256, 192):
            _fail("V4-0 replayed Teacher Gold ROI must be gray8 PNG 256x192")
        roi_image = opened.copy()
    numerator = render_numerator_crop_v4_0(roi_image, mapped_bbox)
    vector = normalized_ink_vector_v4_0(numerator)
    return (
        AuditRecordIdentityV4_0(record_id=record_id, family_id=family, meter_class=expected),
        numerator,
        vector,
        dict(mapped_bbox),
        dict(transform),
        _teacher_sha(source_raw),
    )


def _png_bytes(image: Image.Image) -> bytes:
    out = BytesIO()
    image.save(out, format="PNG", optimize=False, compress_level=9)
    return out.getvalue()


def _contact_sheet(crop_rows: Sequence[Mapping[str, object]], crop_bytes: Mapping[str, bytes]) -> bytes:
    columns = 9
    rows = 3
    tile_w = 96
    tile_h = 96
    sheet = Image.new("L", (columns * tile_w, rows * tile_h), 255)
    draw = ImageDraw.Draw(sheet)
    ordered = sorted(crop_rows, key=lambda row: (str(row["meter_class"]), str(row["family_id"])))
    for index, row in enumerate(ordered):
        x = (index % columns) * tile_w
        y = (index // columns) * tile_h
        with Image.open(BytesIO(crop_bytes[str(row["record_id"])])) as crop:
            sheet.paste(crop, (x + 16, y + 20))
        draw.text((x + 2, y + 2), f"{row['meter_class']} F{row['fold']}", fill=0)
        draw.text((x + 2, y + 84), str(row["record_id"])[-8:], fill=0)
    out = BytesIO()
    sheet.save(out, format="PNG", optimize=False, compress_level=9)
    return out.getvalue()


def run_meter_v4_0_numerator_audit(
    *,
    pilot_path: str | Path,
    choices_path: str | Path,
    permission_path: str | Path,
    privacy_path: str | Path,
    output_root: str | Path,
    repository_sha: str,
) -> dict[str, object]:
    """Run the fixed 27-family zero-training numerator representation audit."""
    repository_sha = _hex64("repository_sha", repository_sha)
    selected, provenance = _validate_and_select_source(
        pilot_path=Path(pilot_path),
        choices_path=Path(choices_path),
        permission_path=Path(permission_path),
        privacy_path=Path(privacy_path),
    )

    identities: list[AuditRecordIdentityV4_0] = []
    vectors: dict[str, tuple[float, ...]] = {}
    crops: dict[str, Image.Image] = {}
    crop_png: dict[str, bytes] = {}
    mapped_bboxes: dict[str, dict[str, object]] = {}
    transforms: dict[str, dict[str, object]] = {}
    source_hashes: dict[str, str] = {}
    for task, answer, adaptation_split in selected:
        identity, numerator, vector, mapped_bbox, transform, source_sha = _derive_selected_crop(
            task, answer, adaptation_split
        )
        if identity.record_id in vectors:
            _fail("V4-0 derived duplicate record id")
        identities.append(identity)
        vectors[identity.record_id] = vector
        crops[identity.record_id] = numerator
        crop_png[identity.record_id] = _png_bytes(numerator)
        mapped_bboxes[identity.record_id] = mapped_bbox
        transforms[identity.record_id] = transform
        source_hashes[identity.record_id] = source_sha

    assignments = fold_plan_v4_0(tuple(identities))
    assignment_by_id = {item.record_id: item for item in assignments}
    probe = centroid_oof_probe_v4_0(tuple(identities), vectors)
    decision = audit_decision_v4_0(probe.summary)

    crop_rows: list[dict[str, object]] = []
    for identity in sorted(identities, key=lambda item: item.record_id):
        assignment = assignment_by_id[identity.record_id]
        left, top, right, bottom = numerator_crop_bounds_v4_0(mapped_bboxes[identity.record_id])
        ink_values = [(255 - value) / 255.0 for value in crops[identity.record_id].tobytes()]
        crop_rows.append(
            {
                "record_id": identity.record_id,
                "family_id": identity.family_id,
                "meter_class": identity.meter_class,
                "numerator_class": identity.numerator_class,
                "fold": assignment.fold,
                "source_image_sha256": source_hashes[identity.record_id],
                "replayed_roi_transform": transforms[identity.record_id],
                "mapped_meter_bbox": mapped_bboxes[identity.record_id],
                "numerator_crop_bounds": {"left": left, "top": top, "right": right, "bottom": bottom},
                "crop_png_sha256": _sha(crop_png[identity.record_id]),
                "ink_fraction": sum(ink_values) / len(ink_values),
            }
        )

    output = Path(output_root)
    if output.exists() or output.is_symlink():
        _fail("V4-0 output root must be fresh")
    temporary = output.with_name(f".{output.name}.part")
    if temporary.exists() or temporary.is_symlink():
        _fail("V4-0 temporary output root already exists")
    temporary.mkdir(parents=True)
    (temporary / "crops").mkdir()

    for row in crop_rows:
        record_id = str(row["record_id"])
        (temporary / "crops" / f"{record_id}.png").write_bytes(crop_png[record_id])
    sheet_raw = _contact_sheet(crop_rows, crop_png)
    (temporary / "numerator-crops-contact-sheet.png").write_bytes(sheet_raw)

    result = {
        "schema": RESULT_SCHEMA,
        "experiment": METER_V4_0_NUMERATOR_AUDIT,
        "repository_sha": repository_sha,
        "source_provenance": provenance,
        "audit_surface": {
            "teacher_positive_train_records": 27,
            "teacher_positive_validation_records": 9,
            "teacher_adaptation_validation_evaluated": False,
            "teacher_adaptation_validation_images_decoded": 0,
            "none_tasks_used": 0,
            "d10_opened": False,
            "test_opened": False,
        },
        "configuration": asdict(FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0),
        "classifier": {
            "type": "l2-normalized-class-centroid-cosine",
            "trainable_parameters": 0,
            "optimizer_steps": 0,
            "tie_break_order": ["2", "3", "4"],
        },
        "crop_records": crop_rows,
        "oof_predictions": [
            {
                "record_id": row.record_id,
                "family_id": row.family_id,
                "fold": row.fold,
                "true": row.true_class,
                "pred": row.predicted_class,
                "correct": row.true_class == row.predicted_class,
                "cosine_scores": dict(row.cosine_scores),
            }
            for row in probe.predictions
        ],
        "oof_summary": {
            "record_count": probe.summary.record_count,
            "accuracy": probe.summary.accuracy,
            "macro_f1": probe.summary.macro_f1,
            "per_class_recall": dict(probe.summary.per_class_recall),
            "confusion": [list(row) for row in probe.summary.confusion],
        },
        "decision": {
            "name": decision.decision,
            "strong_signal": decision.strong_signal,
            "reasons": list(decision.reasons),
        },
        "contact_sheet_sha256": _sha(sheet_raw),
        "optimizer_steps": 0,
        "d11_checkpoint_loaded": False,
        "v3_checkpoint_loaded": False,
        "runtime_connected": False,
        "resolver_connected": False,
        "production_promotion_authorized": False,
    }
    result_raw = _canonical_json(result)
    (temporary / "result.json").write_bytes(result_raw)
    result_sha = _sha(result_raw)
    (temporary / "COMPLETE").write_bytes(f"{result_sha}  result.json\n".encode("ascii"))
    temporary.replace(output)
    return result
