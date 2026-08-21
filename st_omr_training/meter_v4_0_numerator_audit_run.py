"""Execution helper for the bounded Meter V4-0 numerator representation audit.

The runner verifies Teacher Gold COMPLETE/manifest/receipt metadata without
reopening adaptation-validation artifacts, then opens only the 27 positive
TRAIN label/image pairs required by the audit.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
from typing import Final

from PIL import Image, ImageDraw, UnidentifiedImageError

from .meter_teacher_gold_admission_v1 import (
    EXPECTED_CLASS_SPLIT_COUNTS,
    EXPECTED_SOURCES,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TASKS,
    LABEL_SCHEMA,
    MANIFEST_SCHEMA,
    METER_TEACHER_GOLD_ADMISSION_V1,
    RECEIPT_SCHEMA,
)
from .meter_v4_0_numerator_audit import (
    AuditRecordIdentityV4_0,
    FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0,
    METER_TO_NUMERATOR,
    METER_V4_0_NUMERATOR_AUDIT,
    NUMERATOR_CLASSES,
    audit_decision_v4_0,
    classification_summary_v4_0,
    fold_plan_v4_0,
    numerator_crop_bounds_v4_0,
    numerator_crop_tensor_v4_0,
    train_fold_v4_0,
)


RESULT_SCHEMA: Final[str] = "st-omr-meter-v4-0-numerator-representation-audit-result-v1"
_MAX_MANIFEST_BYTES: Final[int] = 16 * 1024 * 1024
_MAX_RECEIPT_BYTES: Final[int] = 1024 * 1024
_MAX_LABEL_BYTES: Final[int] = 256 * 1024
_MAX_IMAGE_BYTES: Final[int] = 2 * 1024 * 1024


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


def _read_regular(path: Path, *, maximum: int, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        _fail(f"{name} must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= maximum:
        _fail(f"{name} byte length is outside V4-0 bounds")
    return path.read_bytes()


def _read_canonical_json(path: Path, *, maximum: int, name: str) -> tuple[dict[str, object], bytes]:
    raw = _read_regular(path, maximum=maximum, name=name)
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MeterV4_0AuditRunError(f"{name} is not valid ASCII JSON") from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        _fail(f"{name} must be a canonical JSON object")
    return payload, raw


def _sequence(name: str, value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(f"{name} must be a sequence")
    return value


def _mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be an object")
    return value


def _verify_teacher_metadata_only(bundle_root: Path) -> tuple[tuple[dict[str, object], ...], str, str]:
    """Verify Teacher Gold control-plane evidence without opening record artifacts."""
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        _fail("Teacher Gold bundle root must be a regular directory")
    complete = _read_regular(bundle_root / "COMPLETE", maximum=1024, name="Teacher Gold COMPLETE")
    manifest, manifest_raw = _read_canonical_json(
        bundle_root / "manifest.json", maximum=_MAX_MANIFEST_BYTES, name="Teacher Gold manifest"
    )
    receipt, receipt_raw = _read_canonical_json(
        bundle_root / "receipt.json", maximum=_MAX_RECEIPT_BYTES, name="Teacher Gold receipt"
    )
    expected_complete = f"{_sha(receipt_raw)}  receipt.json\n{_sha(manifest_raw)}  manifest.json\n".encode("ascii")
    if complete != expected_complete:
        _fail("Teacher Gold COMPLETE binding mismatch")
    if manifest.get("schema_version") != MANIFEST_SCHEMA or receipt.get("schema_version") != RECEIPT_SCHEMA:
        _fail("Teacher Gold schema mismatch")
    if manifest.get("admission_version") != METER_TEACHER_GOLD_ADMISSION_V1 or receipt.get("admission_version") != METER_TEACHER_GOLD_ADMISSION_V1:
        _fail("Teacher Gold admission version mismatch")
    if receipt.get("manifest_sha256") != _sha(manifest_raw):
        _fail("Teacher Gold receipt does not bind manifest bytes")
    if manifest.get("record_count") != EXPECTED_TASKS or manifest.get("source_count") != EXPECTED_SOURCES:
        _fail("Teacher Gold manifest cardinality mismatch")
    if manifest.get("split_counts") != EXPECTED_SPLIT_COUNTS or manifest.get("class_split_counts") != EXPECTED_CLASS_SPLIT_COUNTS:
        _fail("Teacher Gold split/class counts changed")
    for payload in (manifest, receipt):
        if payload.get("test_records") != 0 or payload.get("test_opened") is not False:
            _fail("Teacher Gold metadata exposes sealed TEST")
        if payload.get("optimizer_steps") != 0 or payload.get("model_loaded") is not False:
            _fail("Teacher Gold admission metadata unexpectedly trained or loaded a model")

    rows_raw = _sequence("Teacher Gold records", manifest.get("records"))
    if len(rows_raw) != EXPECTED_TASKS:
        _fail("Teacher Gold manifest must contain exactly 72 records")
    rows: list[dict[str, object]] = []
    family_split: dict[str, str] = {}
    bindings: list[dict[str, str]] = []
    for index, raw_row in enumerate(rows_raw):
        row = dict(_mapping(f"Teacher Gold record[{index}]", raw_row))
        record_id = _hex64("record_id", row.get("record_id"))
        split = row.get("split")
        meter_class = row.get("meter_class")
        family = row.get("family_id")
        if split not in {"train", "validation"} or meter_class not in {"none", "2/4", "3/4", "4/4"}:
            _fail("Teacher Gold row split/class is invalid")
        if not isinstance(family, str) or not family:
            _fail("Teacher Gold family_id must be non-empty")
        prior = family_split.setdefault(family, str(split))
        if prior != split:
            _fail("Teacher Gold family crosses adaptation split")
        if row.get("image_path") != f"images/{record_id}.png" or row.get("label_path") != f"labels/{record_id}.json":
            _fail("Teacher Gold artifact paths are not canonical")
        image_sha = _hex64("image_sha256", row.get("image_sha256"))
        label_sha = _hex64("label_sha256", row.get("label_sha256"))
        bindings.append({"record_id": record_id, "image_sha256": image_sha, "label_sha256": label_sha})
        rows.append(row)
    binding = _sha(_canonical_json(bindings))
    if manifest.get("artifact_binding_sha256") != binding or receipt.get("artifact_binding_sha256") != binding:
        _fail("Teacher Gold metadata artifact binding mismatch")

    counts = Counter((str(row["split"]), str(row["meter_class"])) for row in rows)
    if counts[("train", "2/4")] != 9 or counts[("train", "3/4")] != 9 or counts[("train", "4/4")] != 9:
        _fail("V4-0 requires exact 9/9/9 positive Teacher Gold TRAIN metadata")
    return tuple(rows), _sha(manifest_raw), _sha(receipt_raw)


def _load_train_positive_record(bundle_root: Path, row: Mapping[str, object]):
    """Open exactly one selected TRAIN-positive label/image pair."""
    import torch

    record_id = _hex64("record_id", row.get("record_id"))
    if row.get("split") != "train" or row.get("meter_class") not in METER_TO_NUMERATOR:
        _fail("V4-0 artifact loader accepts only positive Teacher Gold TRAIN records")
    family = row.get("family_id")
    if not isinstance(family, str) or not family:
        _fail("Teacher Gold family_id must be non-empty")

    label_path = bundle_root / str(row["label_path"])
    image_path = bundle_root / str(row["image_path"])
    root_resolved = bundle_root.resolve()
    if root_resolved not in label_path.resolve().parents or root_resolved not in image_path.resolve().parents:
        _fail("Teacher Gold selected artifact path escapes bundle root")

    label, label_raw = _read_canonical_json(label_path, maximum=_MAX_LABEL_BYTES, name="selected Teacher Gold label")
    image_raw = _read_regular(image_path, maximum=_MAX_IMAGE_BYTES, name="selected Teacher Gold ROI image")
    if _sha(label_raw) != row["label_sha256"] or _sha(image_raw) != row["image_sha256"]:
        _fail("selected Teacher Gold artifact SHA mismatch")
    if label.get("schema_version") != LABEL_SCHEMA or label.get("record_id") != record_id:
        _fail("selected Teacher Gold label schema/identity mismatch")
    if label.get("adaptation_split") != "train" or label.get("family_id") != family:
        _fail("selected Teacher Gold label split/family mismatch")
    target = _mapping("selected Teacher Gold target", label.get("target"))
    if target.get("meter_class") != row["meter_class"]:
        _fail("selected Teacher Gold target class mismatch")
    bbox = _mapping("selected Teacher Gold meter_bbox", target.get("meter_bbox"))

    try:
        with Image.open(BytesIO(image_raw)) as opened:
            opened.load()
            if opened.format != "PNG" or opened.mode != "L" or opened.size != (256, 192):
                _fail("selected Teacher Gold ROI must be gray8 PNG 256x192")
            pixels = bytearray(opened.tobytes())
    except MeterV4_0AuditRunError:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise MeterV4_0AuditRunError("selected Teacher Gold ROI cannot be decoded") from exc
    tensor = torch.frombuffer(pixels, dtype=torch.uint8).clone().reshape(192, 256)
    roi_tensor = (1.0 - tensor.to(dtype=torch.float32) / 255.0).unsqueeze(0)
    crop = numerator_crop_tensor_v4_0(roi_tensor, bbox)
    return AuditRecordIdentityV4_0(record_id=record_id, family_id=family, meter_class=str(row["meter_class"])), crop, dict(bbox)


def _crop_png_bytes(crop) -> bytes:
    values = ((1.0 - crop.squeeze(0)).clamp(0, 1) * 255.0).round().to(dtype=__import__("torch").uint8)
    image = Image.frombytes("L", (64, 64), bytes(values.flatten().tolist()))
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
    teacher_bundle_root: str | Path,
    output_root: str | Path,
    repository_sha: str,
) -> dict[str, object]:
    """Run the fixed 27-record family-disjoint numerator OOF audit."""
    import torch
    from .training_model import verify_torch_runtime

    verify_torch_runtime()
    repository_sha = _hex64("repository_sha", repository_sha)
    bundle_root = Path(teacher_bundle_root)
    rows, manifest_sha, receipt_sha = _verify_teacher_metadata_only(bundle_root)

    selected_rows = [
        row
        for row in rows
        if row["split"] == "train" and row["meter_class"] in METER_TO_NUMERATOR
    ]
    if len(selected_rows) != 27:
        _fail("V4-0 selected TRAIN-positive cardinality changed")

    identities: list[AuditRecordIdentityV4_0] = []
    crops: dict[str, object] = {}
    bboxes: dict[str, dict[str, object]] = {}
    crop_png: dict[str, bytes] = {}
    for row in sorted(selected_rows, key=lambda item: (str(item["meter_class"]), str(item["family_id"]), str(item["record_id"]))):
        identity, crop, bbox = _load_train_positive_record(bundle_root, row)
        identities.append(identity)
        crops[identity.record_id] = crop
        bboxes[identity.record_id] = bbox
        crop_png[identity.record_id] = _crop_png_bytes(crop)

    assignments = fold_plan_v4_0(tuple(identities))
    assignment_by_id = {item.record_id: item for item in assignments}
    identity_by_id = {item.record_id: item for item in identities}

    crop_rows: list[dict[str, object]] = []
    for identity in sorted(identities, key=lambda item: item.record_id):
        assignment = assignment_by_id[identity.record_id]
        left, top, right, bottom = numerator_crop_bounds_v4_0(bboxes[identity.record_id])
        crop_rows.append(
            {
                "record_id": identity.record_id,
                "family_id": identity.family_id,
                "meter_class": identity.meter_class,
                "numerator_class": identity.numerator_class,
                "fold": assignment.fold,
                "source_meter_bbox": bboxes[identity.record_id],
                "numerator_crop_bounds": {"left": left, "top": top, "right": right, "bottom": bottom},
                "crop_png_sha256": _sha(crop_png[identity.record_id]),
                "ink_fraction": float(crops[identity.record_id].mean().item()),
            }
        )

    predictions: list[dict[str, object]] = []
    fold_runs: list[dict[str, object]] = []
    for fold in range(3):
        train_ids = sorted(
            (item.record_id for item in assignments if item.fold != fold),
            key=lambda record_id: (
                identity_by_id[record_id].meter_class,
                identity_by_id[record_id].family_id,
                record_id,
            ),
        )
        holdout_ids = sorted(
            (item.record_id for item in assignments if item.fold == fold),
            key=lambda record_id: (
                identity_by_id[record_id].meter_class,
                identity_by_id[record_id].family_id,
                record_id,
            ),
        )
        if len(train_ids) != 18 or len(holdout_ids) != 9:
            _fail("V4-0 fold cardinality changed")
        train_images = torch.stack([crops[record_id] for record_id in train_ids], dim=0)
        train_labels = torch.tensor([identity_by_id[record_id].class_index for record_id in train_ids], dtype=torch.int64)
        model, final_loss, state_sha = train_fold_v4_0(train_images, train_labels, fold=fold)
        holdout_images = torch.stack([crops[record_id] for record_id in holdout_ids], dim=0)
        with torch.no_grad():
            logits = model(holdout_images)
            probabilities = torch.softmax(logits, dim=1)
            guesses = logits.argmax(1).tolist()
        for row_index, record_id in enumerate(holdout_ids):
            identity = identity_by_id[record_id]
            guess = int(guesses[row_index])
            predictions.append(
                {
                    "record_id": record_id,
                    "family_id": identity.family_id,
                    "fold": fold,
                    "true": identity.numerator_class,
                    "pred": NUMERATOR_CLASSES[guess],
                    "correct": NUMERATOR_CLASSES[guess] == identity.numerator_class,
                    "probabilities": {
                        label: float(probabilities[row_index, class_index].item())
                        for class_index, label in enumerate(NUMERATOR_CLASSES)
                    },
                }
            )
        fold_runs.append(
            {
                "fold": fold,
                "train_records": 18,
                "holdout_records": 9,
                "final_train_loss": final_loss,
                "model_state_sha256": state_sha,
            }
        )

    predictions.sort(key=lambda row: (int(row["fold"]), str(row["true"]), str(row["family_id"]), str(row["record_id"])))
    if len(predictions) != 27 or len({str(row["record_id"]) for row in predictions}) != 27:
        _fail("V4-0 OOF predictions must cover each selected record exactly once")
    truth = [NUMERATOR_CLASSES.index(str(row["true"])) for row in predictions]
    guessed = [NUMERATOR_CLASSES.index(str(row["pred"])) for row in predictions]
    summary = classification_summary_v4_0(truth, guessed)
    decision = audit_decision_v4_0(summary)

    output = Path(output_root)
    if output.exists() or output.is_symlink():
        _fail("V4-0 output root must be fresh")
    temporary = output.with_name(f".{output.name}.part")
    if temporary.exists() or temporary.is_symlink():
        _fail("V4-0 temporary output root already exists")
    temporary.mkdir(parents=True)
    (temporary / "crops").mkdir()
    try:
        for row in crop_rows:
            record_id = str(row["record_id"])
            (temporary / "crops" / f"{record_id}.png").write_bytes(crop_png[record_id])
        sheet_raw = _contact_sheet(crop_rows, crop_png)
        (temporary / "numerator-crops-contact-sheet.png").write_bytes(sheet_raw)
        result = {
            "schema": RESULT_SCHEMA,
            "experiment": METER_V4_0_NUMERATOR_AUDIT,
            "repository_sha": repository_sha,
            "teacher_manifest_sha256": manifest_sha,
            "teacher_receipt_sha256": receipt_sha,
            "audit_surface": {
                "teacher_train_positive_records": 27,
                "classes": list(NUMERATOR_CLASSES),
                "records_per_class": 9,
                "folds": 3,
                "teacher_adaptation_validation_evaluated": False,
                "teacher_adaptation_validation_artifacts_opened_by_audit": 0,
                "d10_opened": False,
                "test_opened": False,
            },
            "configuration": {
                "output_size": FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0.output_size,
                "epochs_per_fold": FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0.epochs,
                "shift_pixels": FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0.shift_pixels,
                "horizontal_padding_milli": FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0.horizontal_padding_milli,
                "vertical_padding_milli": FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0.vertical_padding_milli,
                "numerator_fraction_milli": FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0.numerator_fraction_milli,
            },
            "crop_records": crop_rows,
            "fold_runs": fold_runs,
            "oof_predictions": predictions,
            "oof_summary": {
                "record_count": summary.record_count,
                "accuracy": summary.accuracy,
                "macro_f1": summary.macro_f1,
                "per_class_recall": dict(summary.per_class_recall),
                "confusion": [list(row) for row in summary.confusion],
            },
            "decision": {
                "name": decision.decision,
                "strong_signal": decision.strong_signal,
                "reasons": list(decision.reasons),
            },
            "contact_sheet_sha256": _sha(sheet_raw),
            "optimizer_steps": 3 * FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0.epochs,
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
    except BaseException:
        raise


def clean_incomplete_v4_0_output(path: str | Path) -> None:
    """Explicit helper for a caller that chooses to discard only a known `.part` audit directory."""
    target = Path(path)
    if target.name.startswith(".") and target.name.endswith(".part") and target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    else:
        raise ValueError("only an explicit V4-0 .part directory may be removed")
