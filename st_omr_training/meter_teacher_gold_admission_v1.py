"""Admit explicitly approved Meter teacher-gold into a sealed offline bundle.

The source pilot and reviewer choices remain outside Git.  This module accepts
only the fixed 72-record pilot, rejects TEST immediately, binds explicit
training-permission and privacy-review evidence, and emits deterministic
256x192 gray8 Meter ROI derivatives for a later shadow adaptation run.

No model, checkpoint, optimizer, network, Drive client, or runtime Resolver is
loaded here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import base64
import binascii
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import re
from typing import Final

from PIL import Image, UnidentifiedImageError


METER_TEACHER_GOLD_ADMISSION_V1: Final[str] = "meter-teacher-gold-admission-v1"
PILOT_SCHEMA: Final[str] = "st-omr-meter-teacher-gold-pilot-data-v1"
CHOICES_SCHEMA: Final[str] = "st-omr-meter-teacher-gold-pilot-v1-choices"
PERMISSION_SCHEMA: Final[str] = "st-omr-meter-training-permission-evidence-v1"
PRIVACY_SCHEMA: Final[str] = "st-omr-meter-privacy-review-evidence-v1"
MANIFEST_SCHEMA: Final[str] = "st-omr-meter-teacher-gold-manifest-v1"
LABEL_SCHEMA: Final[str] = "st-omr-meter-teacher-gold-label-v1"
RECEIPT_SCHEMA: Final[str] = "st-omr-meter-teacher-gold-receipt-v1"
ALLOWED_USE: Final[str] = "offline-meter-real-domain-adaptation-pilot"
METER_CLASSES: Final[tuple[str, ...]] = ("none", "2/4", "3/4", "4/4")
OUTPUT_WIDTH: Final[int] = 256
OUTPUT_HEIGHT: Final[int] = 192
EXPECTED_TASKS: Final[int] = 72
EXPECTED_SOURCES: Final[int] = 36
EXPECTED_SPLIT_COUNTS: Final[dict[str, int]] = {"train": 54, "validation": 18}
EXPECTED_CLASS_SPLIT_COUNTS: Final[dict[str, dict[str, int]]] = {
    "train": {"none": 27, "2/4": 9, "3/4": 9, "4/4": 9},
    "validation": {"none": 9, "2/4": 3, "3/4": 3, "4/4": 3},
}
_DATA_URI = re.compile(r"^data:image/png;base64,([A-Za-z0-9+/=]+)$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MAX_PILOT_BYTES = 32 * 1024 * 1024
_MAX_CHOICES_BYTES = 4 * 1024 * 1024
_MAX_EVIDENCE_BYTES = 64 * 1024
_MAX_IMAGE_BYTES = 16 * 1024 * 1024


class MeterTeacherGoldAdmissionError(RuntimeError):
    """Raised when teacher-gold provenance, labels, or bytes fail closed."""


def _fail(message: str) -> None:
    raise MeterTeacherGoldAdmissionError(message)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise MeterTeacherGoldAdmissionError("payload is not canonical JSON serializable") from exc


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _regular_file(path: Path, *, maximum: int, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        _fail(f"{name} must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= maximum:
        _fail(f"{name} byte length is outside the admission boundary")
    return path.read_bytes()


def _json_file(path: Path, *, maximum: int, name: str, canonical: bool = False) -> tuple[dict[str, object], bytes]:
    raw = _regular_file(path, maximum=maximum, name=name)
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: _fail(f"non-finite JSON constant in {name}: {token}"),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MeterTeacherGoldAdmissionError(f"{name} is not valid JSON") from exc
    if not isinstance(payload, dict):
        _fail(f"{name} must be a JSON object")
    if canonical and raw not in {_canonical_json(payload), _canonical_json(payload) + b"\n"}:
        _fail(f"{name} must use canonical JSON bytes with at most one final newline")
    return payload, raw


def _bounded_ascii(name: str, value: object, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or len(value) > maximum:
        _fail(f"{name} must be bounded non-empty ASCII")
    return value


def _hex64(name: str, value: object) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        _fail(f"{name} must be canonical lowercase SHA-256")
    return value


def _sequence(name: str, value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(f"{name} must be a sequence")
    return value


def _mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be an object")
    return value


def _finite(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def _xywh(name: str, value: object) -> tuple[float, float, float, float]:
    box = _mapping(name, value)
    if set(box) != {"x", "y", "w", "h"}:
        _fail(f"{name} must contain exactly x/y/w/h")
    x, y, width, height = (_finite(f"{name}.{key}", box.get(key)) for key in ("x", "y", "w", "h"))
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        _fail(f"{name} must have non-negative origin and positive area")
    return x, y, width, height


def _decode_source_png(data_uri: object) -> tuple[Image.Image, bytes]:
    if not isinstance(data_uri, str):
        _fail("pilot image_data_uri must be a PNG data URI")
    match = _DATA_URI.fullmatch(data_uri)
    if match is None:
        _fail("pilot image_data_uri must be canonical PNG base64")
    try:
        raw = base64.b64decode(match.group(1), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MeterTeacherGoldAdmissionError("pilot PNG base64 is invalid") from exc
    if not 1 <= len(raw) <= _MAX_IMAGE_BYTES:
        _fail("pilot source PNG byte length is outside bounds")
    try:
        with Image.open(BytesIO(raw)) as opened:
            opened.load()
            if opened.format != "PNG":
                _fail("pilot source image must decode as PNG")
            if opened.width < 64 or opened.height < 32 or opened.width > 20_000 or opened.height > 4_000:
                _fail("pilot source PNG dimensions are outside bounds")
            return opened.convert("L"), raw
    except MeterTeacherGoldAdmissionError:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise MeterTeacherGoldAdmissionError("pilot source PNG cannot be decoded") from exc


def _validate_permission(payload: Mapping[str, object]) -> None:
    checks = {
        "schema_version": PERMISSION_SCHEMA,
        "decision": "approved",
        "allowed_use": ALLOWED_USE,
        "dataset_scope": "METER_V1/TRAIN/teacher-gold-pilot-72",
        "automatic_learning": False,
        "production_promotion_authorized": False,
        "test_access_authorized": False,
    }
    for name, expected in checks.items():
        if payload.get(name) != expected:
            _fail(f"training permission evidence {name} mismatch")
    _bounded_ascii("training permission approved_at", payload.get("approved_at"), maximum=64)


def _validate_privacy(payload: Mapping[str, object]) -> None:
    checks = {
        "schema_version": PRIVACY_SCHEMA,
        "decision": "approved",
        "review_scope": "METER_V1/TRAIN/teacher-gold-pilot-72",
        "personal_data_detected": False,
        "redistribution_allowed": False,
        "test_opened": False,
    }
    for name, expected in checks.items():
        if payload.get(name) != expected:
            _fail(f"privacy review evidence {name} mismatch")
    _bounded_ascii("privacy review reviewed_at", payload.get("reviewed_at"), maximum=64)


def _fresh_output_root(output_root: Path, repository_root: Path) -> None:
    output = output_root.resolve()
    repository = repository_root.resolve()
    if output == repository or repository in output.parents:
        _fail("teacher-gold bytes must remain outside the Git repository")
    if output_root.exists() or output_root.is_symlink():
        _fail("teacher-gold output root must be fresh")
    output_root.mkdir(parents=True)
    (output_root / "images").mkdir()
    (output_root / "labels").mkdir()


def _adaptation_split_by_family(tasks: Sequence[Mapping[str, object]]) -> dict[str, str]:
    positive = [task for task in tasks if task.get("kind") == "positive"]
    strata: defaultdict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for task in positive:
        label = _bounded_ascii("positive expected_class", task.get("expected_class"))
        package = _bounded_ascii("positive package", task.get("package"))
        strata[(label, package)].append(task)
    expected_strata = {
        (meter, package): count
        for meter in METER_CLASSES[1:]
        for package, count in (("aa", 4), ("ab", 8))
    }
    if {key: len(values) for key, values in strata.items()} != expected_strata:
        _fail("pilot positive class/package strata differ from the approved 4-aa/8-ab selection")
    result: dict[str, str] = {}
    for key, values in sorted(strata.items()):
        ranked = sorted(
            values,
            key=lambda task: _sha(
                _canonical_json(
                    {
                        "version": METER_TEACHER_GOLD_ADMISSION_V1,
                        "split_policy": "family-stratified-25-percent-validation-v1",
                        "class": key[0],
                        "package": key[1],
                        "family_key": task.get("family_key"),
                    }
                )
            ),
        )
        validation_count = len(ranked) // 4
        validation_families = {
            _bounded_ascii("validation family_key", task.get("family_key"))
            for task in ranked[:validation_count]
        }
        for task in ranked:
            family = _bounded_ascii("family_key", task.get("family_key"))
            split = "validation" if family in validation_families else "train"
            prior = result.setdefault(family, split)
            if prior != split:
                _fail("family crosses adaptation split")
    if len(result) != EXPECTED_SOURCES:
        _fail("approved pilot must contain 36 unique families")
    return result


@dataclass(frozen=True, slots=True)
class TeacherGoldReceiptV1:
    schema_version: str
    admission_version: str
    pilot_sha256: str
    choices_sha256: str
    permission_evidence_sha256: str
    privacy_review_evidence_sha256: str
    manifest_sha256: str
    artifact_binding_sha256: str
    record_count: int
    source_count: int
    split_counts: dict[str, int]
    class_split_counts: dict[str, dict[str, int]]
    test_records: int
    test_opened: bool
    optimizer_steps: int
    model_loaded: bool


def _render_roi(
    image: Image.Image,
    roi_box: tuple[float, float, float, float],
) -> tuple[bytes, dict[str, int | float]]:
    x, y, width, height = roi_box
    left = max(0, min(image.width - 1, math.floor(x)))
    top = max(0, min(image.height - 1, math.floor(y)))
    right = max(left + 1, min(image.width, math.ceil(x + width)))
    bottom = max(top + 1, min(image.height, math.ceil(y + height)))
    crop_width, crop_height = right - left, bottom - top
    scale = min(OUTPUT_WIDTH / crop_width, OUTPUT_HEIGHT / crop_height)
    resized_width = max(1, min(OUTPUT_WIDTH, int(round(crop_width * scale))))
    resized_height = max(1, min(OUTPUT_HEIGHT, int(round(crop_height * scale))))
    pad_left = (OUTPUT_WIDTH - resized_width) // 2
    pad_top = (OUTPUT_HEIGHT - resized_height) // 2
    scale_x = resized_width / crop_width
    scale_y = resized_height / crop_height
    crop = image.crop((left, top, right, bottom))
    resized = crop.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("L", (OUTPUT_WIDTH, OUTPUT_HEIGHT), 255)
    canvas.paste(resized, (pad_left, pad_top))
    out = BytesIO()
    canvas.save(out, format="PNG", optimize=False, compress_level=9)
    raw = out.getvalue()
    transform: dict[str, int | float] = {
        "crop_left": left,
        "crop_top": top,
        "crop_right": right,
        "crop_bottom": bottom,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "output_width": OUTPUT_WIDTH,
        "output_height": OUTPUT_HEIGHT,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }
    return raw, transform


def _map_bbox(
    bbox: tuple[float, float, float, float],
    transform: Mapping[str, int | float],
) -> dict[str, float]:
    x, y, width, height = bbox
    left = float(transform["crop_left"])
    top = float(transform["crop_top"])
    right = float(transform["crop_right"])
    bottom = float(transform["crop_bottom"])
    if x < left or y < top or x + width > right or y + height > bottom:
        _fail("accepted positive bbox lies outside its accepted ROI crop")
    scale_x = float(transform["scale_x"])
    scale_y = float(transform["scale_y"])
    pad_left = float(transform["pad_left"])
    pad_top = float(transform["pad_top"])
    mapped = {
        "x_min": (x - left) * scale_x + pad_left,
        "y_min": (y - top) * scale_y + pad_top,
        "x_max": (x + width - left) * scale_x + pad_left,
        "y_max": (y + height - top) * scale_y + pad_top,
    }
    if not (
        0 <= mapped["x_min"] < mapped["x_max"] <= OUTPUT_WIDTH
        and 0 <= mapped["y_min"] < mapped["y_max"] <= OUTPUT_HEIGHT
    ):
        _fail("mapped positive bbox lies outside 256x192 ROI")
    return mapped


def build_meter_teacher_gold_bundle_v1(
    *,
    pilot_path: str | Path,
    choices_path: str | Path,
    permission_evidence_path: str | Path,
    privacy_review_evidence_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
) -> TeacherGoldReceiptV1:
    """Build the one approved pilot bundle; TEST and automatic learning fail closed."""
    pilot, pilot_raw = _json_file(Path(pilot_path), maximum=_MAX_PILOT_BYTES, name="pilot data")
    choices, choices_raw = _json_file(Path(choices_path), maximum=_MAX_CHOICES_BYTES, name="review choices")
    permission, permission_raw = _json_file(
        Path(permission_evidence_path), maximum=_MAX_EVIDENCE_BYTES, name="training permission evidence", canonical=True
    )
    privacy, privacy_raw = _json_file(
        Path(privacy_review_evidence_path), maximum=_MAX_EVIDENCE_BYTES, name="privacy review evidence", canonical=True
    )
    _validate_permission(permission)
    _validate_privacy(privacy)
    if pilot.get("schema") != PILOT_SCHEMA or choices.get("schema") != CHOICES_SCHEMA:
        _fail("pilot or choices schema mismatch")
    if pilot.get("source") != "METER_V1/01_REVIEW/train":
        _fail("pilot source must be METER_V1 TRAIN review surface")
    selection = _mapping("pilot selection", pilot.get("selection"))
    if selection.get("test_opened") is not False or choices.get("test_opened") is not False:
        _fail("sealed TEST evidence reached teacher-gold admission")
    if choices.get("task_count") != EXPECTED_TASKS or choices.get("answered_count") != EXPECTED_TASKS:
        _fail("review choices must contain 72 answered tasks")

    tasks_raw = _sequence("pilot tasks", pilot.get("tasks"))
    answers_raw = _sequence("review answers", choices.get("answers"))
    if len(tasks_raw) != EXPECTED_TASKS or len(answers_raw) != EXPECTED_TASKS:
        _fail("approved pilot cardinality is 72 tasks")
    tasks = [_mapping(f"pilot task[{index}]", item) for index, item in enumerate(tasks_raw)]
    answers = [_mapping(f"review answer[{index}]", item) for index, item in enumerate(answers_raw)]
    task_ids = [_bounded_ascii("task_id", task.get("task_id")) for task in tasks]
    answer_ids = [_bounded_ascii("answer task_id", answer.get("task_id")) for answer in answers]
    if len(set(task_ids)) != EXPECTED_TASKS or len(set(answer_ids)) != EXPECTED_TASKS or set(task_ids) != set(answer_ids):
        _fail("pilot and answer task identities must be unique and identical")
    task_by_id = dict(zip(task_ids, tasks))
    answer_by_id = dict(zip(answer_ids, answers))

    positive_counts = Counter(task.get("expected_class") for task in tasks if task.get("kind") == "positive")
    kind_counts = Counter(task.get("kind") for task in tasks)
    if kind_counts != {"positive": 36, "none": 36}:
        _fail("pilot must contain 36 positive and 36 none tasks")
    if positive_counts != {"2/4": 12, "3/4": 12, "4/4": 12}:
        _fail("pilot positive classes must be balanced 12/12/12")

    source_tasks: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    source_images: dict[str, tuple[Image.Image, bytes]] = {}
    source_meta: dict[str, Mapping[str, object]] = {}
    for task in tasks:
        source_id = _bounded_ascii("source_id", task.get("source_id"))
        if task.get("split") != "train":
            _fail("teacher-gold pilot may consume only source TRAIN")
        if task.get("expected_class") not in METER_CLASSES:
            _fail("pilot expected_class is outside Meter classes")
        source_tasks[source_id].append(task)
        image, raw = _decode_source_png(task.get("image_data_uri"))
        if source_id in source_images:
            if source_images[source_id][1] != raw:
                _fail("paired tasks disagree on source image bytes")
        else:
            source_images[source_id] = (image, raw)
            source_meta[source_id] = task
    if len(source_tasks) != EXPECTED_SOURCES:
        _fail("pilot must contain 36 source pairs")
    for source_id, paired in source_tasks.items():
        if Counter(task.get("kind") for task in paired) != {"positive": 1, "none": 1}:
            _fail(f"source {source_id} must have one positive and one none task")
        families = {task.get("family_key") for task in paired}
        if len(families) != 1:
            _fail("paired tasks disagree on family identity")
    family_splits = _adaptation_split_by_family(tasks)

    root = Path(output_root)
    _fresh_output_root(root, Path(repository_root))
    records: list[dict[str, object]] = []
    try:
        for task_id in sorted(task_ids):
            task = task_by_id[task_id]
            answer = answer_by_id[task_id]
            source_id = _bounded_ascii("source_id", task.get("source_id"))
            family = _bounded_ascii("family_key", task.get("family_key"))
            kind = task.get("kind")
            expected = task.get("expected_class")
            if answer.get("source_id") != source_id or answer.get("split") != "train" or answer.get("kind") != kind:
                _fail("answer identity/kind/source split differs from pilot task")
            if answer.get("status") != "accepted" or answer.get("label_confirmed") is not True or answer.get("crop_usable") is not True:
                _fail("only explicitly accepted, confirmed, usable answers may enter teacher-gold")
            if answer.get("expected_class") != expected or answer.get("label") != expected:
                _fail("accepted answer label differs from expected teacher label")
            if kind == "positive" and expected not in METER_CLASSES[1:]:
                _fail("positive task requires 2/4, 3/4, or 4/4")
            if kind == "none" and (expected != "none" or answer.get("bbox") is not None):
                _fail("none task must carry class none and no bbox")
            roi_box = _xywh("answer roi_crop_box", answer.get("roi_crop_box"))
            source_image, source_raw = source_images[source_id]
            roi_raw, transform = _render_roi(source_image, roi_box)
            mapped_bbox = None
            if kind == "positive":
                mapped_bbox = _map_bbox(_xywh("answer bbox", answer.get("bbox")), transform)
            adaptation_split = family_splits[family]
            record_id = _sha(
                _canonical_json(
                    {
                        "version": METER_TEACHER_GOLD_ADMISSION_V1,
                        "task_id": task_id,
                        "source_id": source_id,
                        "source_image_sha256": _sha(source_raw),
                        "adaptation_split": adaptation_split,
                        "meter_class": expected,
                        "roi_transform": transform,
                        "meter_bbox": mapped_bbox,
                    }
                )
            )
            image_rel = f"images/{record_id}.png"
            label_rel = f"labels/{record_id}.json"
            image_sha = _sha(roi_raw)
            label = {
                "schema_version": LABEL_SCHEMA,
                "admission_version": METER_TEACHER_GOLD_ADMISSION_V1,
                "record_id": record_id,
                "task_id": task_id,
                "source_id": source_id,
                "family_id": family,
                "package": _bounded_ascii("package", task.get("package")),
                "source_split": "train",
                "adaptation_split": adaptation_split,
                "source_image_sha256": _sha(source_raw),
                "roi_image_sha256": image_sha,
                "roi_transform": transform,
                "target": {"meter_class": expected, "meter_bbox": mapped_bbox},
                "review": {
                    "status": "accepted",
                    "label_confirmed": True,
                    "crop_usable": True,
                    "reviewed_at": _bounded_ascii("reviewed_at", answer.get("reviewed_at"), maximum=64),
                },
            }
            label_raw = _canonical_json(label)
            label_sha = _sha(label_raw)
            (root / image_rel).write_bytes(roi_raw)
            (root / label_rel).write_bytes(label_raw)
            records.append(
                {
                    "record_id": record_id,
                    "task_id": task_id,
                    "source_id": source_id,
                    "family_id": family,
                    "source_split": "train",
                    "split": adaptation_split,
                    "meter_class": expected,
                    "image_path": image_rel,
                    "image_sha256": image_sha,
                    "label_path": label_rel,
                    "label_sha256": label_sha,
                }
            )

        records.sort(key=lambda row: (str(row["split"]), str(row["family_id"]), str(row["record_id"])))
        split_counts = Counter(str(row["split"]) for row in records)
        class_split_counts = {
            split: Counter(str(row["meter_class"]) for row in records if row["split"] == split)
            for split in ("train", "validation")
        }
        if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
            _fail("adaptation split counts differ from 54 TRAIN / 18 VALIDATION")
        if {split: dict(counts) for split, counts in class_split_counts.items()} != EXPECTED_CLASS_SPLIT_COUNTS:
            _fail("adaptation class/split counts differ from the approved balanced policy")
        binding = _sha(
            _canonical_json(
                [
                    {
                        "record_id": row["record_id"],
                        "image_sha256": row["image_sha256"],
                        "label_sha256": row["label_sha256"],
                    }
                    for row in records
                ]
            )
        )
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "admission_version": METER_TEACHER_GOLD_ADMISSION_V1,
            "pilot_sha256": _sha(pilot_raw),
            "choices_sha256": _sha(choices_raw),
            "permission_evidence_sha256": _sha(permission_raw),
            "privacy_review_evidence_sha256": _sha(privacy_raw),
            "split_policy": "family-and-class-package-stratified-25-percent-validation-v1",
            "source_split": "train",
            "records": records,
            "record_count": len(records),
            "source_count": len(source_tasks),
            "split_counts": dict(split_counts),
            "class_split_counts": {split: dict(counts) for split, counts in class_split_counts.items()},
            "test_records": 0,
            "test_opened": False,
            "optimizer_steps": 0,
            "model_loaded": False,
            "artifact_binding_sha256": binding,
        }
        manifest_raw = _canonical_json(manifest)
        manifest_sha = _sha(manifest_raw)
        (root / "manifest.json").write_bytes(manifest_raw)
        receipt = TeacherGoldReceiptV1(
            schema_version=RECEIPT_SCHEMA,
            admission_version=METER_TEACHER_GOLD_ADMISSION_V1,
            pilot_sha256=_sha(pilot_raw),
            choices_sha256=_sha(choices_raw),
            permission_evidence_sha256=_sha(permission_raw),
            privacy_review_evidence_sha256=_sha(privacy_raw),
            manifest_sha256=manifest_sha,
            artifact_binding_sha256=binding,
            record_count=len(records),
            source_count=len(source_tasks),
            split_counts=dict(split_counts),
            class_split_counts={split: dict(counts) for split, counts in class_split_counts.items()},
            test_records=0,
            test_opened=False,
            optimizer_steps=0,
            model_loaded=False,
        )
        receipt_raw = _canonical_json(asdict(receipt))
        receipt_sha = _sha(receipt_raw)
        (root / "receipt.json").write_bytes(receipt_raw)
        (root / "COMPLETE").write_bytes(
            f"{receipt_sha}  receipt.json\n{manifest_sha}  manifest.json\n".encode("ascii")
        )
    except Exception:
        # The fresh output remains inspectable but never receives COMPLETE on failure.
        raise
    return verify_meter_teacher_gold_bundle_v1(root)


def verify_meter_teacher_gold_bundle_v1(bundle_root: str | Path) -> TeacherGoldReceiptV1:
    """Independently reopen every teacher-gold byte and verify the fixed pilot."""
    root = Path(bundle_root)
    if root.is_symlink() or not root.is_dir():
        _fail("teacher-gold bundle root must be a regular directory")
    complete = _regular_file(root / "COMPLETE", maximum=1024, name="teacher-gold COMPLETE")
    manifest, manifest_raw = _json_file(
        root / "manifest.json", maximum=16 * 1024 * 1024, name="teacher-gold manifest", canonical=True
    )
    receipt_payload, receipt_raw = _json_file(
        root / "receipt.json", maximum=1024 * 1024, name="teacher-gold receipt", canonical=True
    )
    expected_complete = f"{_sha(receipt_raw)}  receipt.json\n{_sha(manifest_raw)}  manifest.json\n".encode("ascii")
    if complete != expected_complete:
        _fail("teacher-gold COMPLETE binding mismatch")
    if manifest.get("schema_version") != MANIFEST_SCHEMA or receipt_payload.get("schema_version") != RECEIPT_SCHEMA:
        _fail("teacher-gold manifest/receipt schema mismatch")
    if receipt_payload.get("manifest_sha256") != _sha(manifest_raw):
        _fail("teacher-gold receipt does not bind manifest bytes")
    if manifest.get("record_count") != EXPECTED_TASKS or manifest.get("source_count") != EXPECTED_SOURCES:
        _fail("teacher-gold manifest cardinality mismatch")
    if manifest.get("split_counts") != EXPECTED_SPLIT_COUNTS or manifest.get("class_split_counts") != EXPECTED_CLASS_SPLIT_COUNTS:
        _fail("teacher-gold manifest class/split counts mismatch")
    for payload in (manifest, receipt_payload):
        if payload.get("test_records") != 0 or payload.get("test_opened") is not False:
            _fail("teacher-gold verification reached sealed TEST evidence")
        if payload.get("optimizer_steps") != 0 or payload.get("model_loaded") is not False:
            _fail("teacher-gold admission cannot train or load a model")
    rows_raw = _sequence("teacher-gold manifest records", manifest.get("records"))
    if len(rows_raw) != EXPECTED_TASKS:
        _fail("teacher-gold manifest must contain 72 records")
    rows = [_mapping(f"teacher-gold record[{index}]", item) for index, item in enumerate(rows_raw)]
    seen: set[str] = set()
    family_split: dict[str, str] = {}
    bindings: list[dict[str, str]] = []
    for row in rows:
        record_id = _hex64("record_id", row.get("record_id"))
        if record_id in seen:
            _fail("duplicate teacher-gold record_id")
        seen.add(record_id)
        split = row.get("split")
        meter_class = row.get("meter_class")
        if split not in {"train", "validation"} or meter_class not in METER_CLASSES:
            _fail("teacher-gold row split/class is invalid")
        family = _bounded_ascii("family_id", row.get("family_id"))
        prior = family_split.setdefault(family, str(split))
        if prior != split:
            _fail("teacher-gold family crosses adaptation split")
        image_rel = row.get("image_path")
        label_rel = row.get("label_path")
        if image_rel != f"images/{record_id}.png" or label_rel != f"labels/{record_id}.json":
            _fail("teacher-gold artifact path is not canonical")
        image_path = root / str(image_rel)
        label_path = root / str(label_rel)
        if root.resolve() not in image_path.resolve().parents or root.resolve() not in label_path.resolve().parents:
            _fail("teacher-gold artifact path escapes bundle root")
        image_raw = _regular_file(image_path, maximum=2 * 1024 * 1024, name="teacher-gold ROI image")
        label, label_raw = _json_file(label_path, maximum=256 * 1024, name="teacher-gold label", canonical=True)
        image_sha = _hex64("image_sha256", row.get("image_sha256"))
        label_sha = _hex64("label_sha256", row.get("label_sha256"))
        if _sha(image_raw) != image_sha or _sha(label_raw) != label_sha:
            _fail("teacher-gold artifact SHA-256 mismatch")
        if label.get("schema_version") != LABEL_SCHEMA or label.get("record_id") != record_id:
            _fail("teacher-gold label schema/identity mismatch")
        if label.get("adaptation_split") != split or label.get("family_id") != family:
            _fail("teacher-gold label split/family binding mismatch")
        target = _mapping("teacher-gold target", label.get("target"))
        if target.get("meter_class") != meter_class:
            _fail("teacher-gold target class differs from manifest")
        if meter_class == "none":
            if target.get("meter_bbox") is not None:
                _fail("teacher-gold none target carries a bbox")
        else:
            box = _mapping("teacher-gold target bbox", target.get("meter_bbox"))
            values = [_finite(f"teacher-gold bbox {key}", box.get(key)) for key in ("x_min", "y_min", "x_max", "y_max")]
            if not (0 <= values[0] < values[2] <= OUTPUT_WIDTH and 0 <= values[1] < values[3] <= OUTPUT_HEIGHT):
                _fail("teacher-gold positive bbox lies outside ROI")
        try:
            with Image.open(BytesIO(image_raw)) as opened:
                opened.load()
                if opened.format != "PNG" or opened.mode != "L" or opened.size != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
                    _fail("teacher-gold ROI must be gray8 PNG 256x192")
        except MeterTeacherGoldAdmissionError:
            raise
        except (UnidentifiedImageError, OSError) as exc:
            raise MeterTeacherGoldAdmissionError("teacher-gold ROI PNG cannot be decoded") from exc
        bindings.append({"record_id": record_id, "image_sha256": image_sha, "label_sha256": label_sha})
    binding = _sha(_canonical_json(bindings))
    if manifest.get("artifact_binding_sha256") != binding or receipt_payload.get("artifact_binding_sha256") != binding:
        _fail("teacher-gold artifact binding SHA-256 mismatch")
    expected_receipt_fields = {
        "admission_version": METER_TEACHER_GOLD_ADMISSION_V1,
        "record_count": EXPECTED_TASKS,
        "source_count": EXPECTED_SOURCES,
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "class_split_counts": EXPECTED_CLASS_SPLIT_COUNTS,
        "test_records": 0,
        "test_opened": False,
        "optimizer_steps": 0,
        "model_loaded": False,
    }
    for name, expected in expected_receipt_fields.items():
        if receipt_payload.get(name) != expected:
            _fail(f"teacher-gold receipt {name} mismatch")
    return TeacherGoldReceiptV1(**receipt_payload)


def checkpoint_loading_allowed() -> bool:
    return False


def optimizer_step_allowed() -> bool:
    return False


def sealed_test_access_allowed() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False
