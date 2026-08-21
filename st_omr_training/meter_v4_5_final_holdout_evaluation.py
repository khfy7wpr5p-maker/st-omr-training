"""Meter V4-5 one-time independent final-holdout evaluation.

This module is intentionally fail-closed.  It verifies the frozen V4-4 holdout
and the exact V4-2 candidate before creating a persistent one-shot lock.  Only
after that lock exists may the trusted-hash checkpoint be deserialized and the
150-record final evaluation run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Final

from PIL import Image, UnidentifiedImageError
import torch

from .meter_v4_0_numerator_audit import FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0
from .meter_v4_1_numerator_specialist import (
    EXPECTED_PARAMETER_COUNT_V4_1,
    FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1,
    NUMERATOR_CLASSES_V4_1,
    NumeratorSpecialistV4_1,
    config_fingerprint_v4_1,
)
from .meter_v4_4_bbox_contract import (
    BBox,
    COMPLETE_SCHEMA,
    EXPECTED_CLASS_COUNTS,
    EXPECTED_SELECTED_COUNT,
    canonical_json,
    discover_selected_samples,
    load_and_validate_selection_manifest,
    read_bbox_contract,
    read_json,
    read_png_info,
    sha256_file,
)
from .training_model import (
    assert_finite_tensor,
    assert_model_finite,
    count_trainable_parameters,
    model_state_sha256,
)


V4_5_SCHEMA: Final[str] = "st-omr-meter-v4-5-final-holdout-result-v1"
V4_5_STAGE: Final[str] = "meter-v4-5-one-time-final-holdout-evaluation-v1"
PREREG_SCHEMA: Final[str] = "st-omr-meter-v4-5-preregistration-v1"
HUMAN_REVIEW_SCHEMA: Final[str] = "st-omr-meter-v4-4-human-visual-review-evidence-v1"
CHECKPOINT_SCHEMA: Final[str] = "st-omr-meter-v4-2-development-candidate-v1"
EXPECTED_SELECTION_SHA256: Final[str] = "4335a48a091912ba422c16d8fcbaaa7bbf5f7a0a43f088146a50a3e02e3ed7dc"
EXPECTED_IMAGE_BINDING_SHA256: Final[str] = "73c932e8cbb55c9b57a482bb05fbf7e033cdd375f4929e34851cca01f9c1cd66"
EXPECTED_BBOX_MANIFEST_SHA256: Final[str] = "0242fe99d39393a78d5b5d69ed95dbfc6d65975a3bef21ec43368386cdfdca70"
EXPECTED_COMPLETION_RECEIPT_SHA256: Final[str] = "24aea7f900128858a1fecdb5ed01427161e1ba02768da4a191be3e70226d2548"
EXPECTED_V4_2_RESULT_SHA256: Final[str] = "bf32ccd4bf9512ad6a39b3e34ac2b7b0ab708f140d8e8bcf48721fab28cab04b"
EXPECTED_CHECKPOINT_SHA256: Final[str] = "2dc820bc0cbadf5db90a7ddee7f5a9daba06e546dcae1da560d1ac9718e3692a"
EXPECTED_MODEL_STATE_SHA256: Final[str] = "0ca4831729ba4723d6aac71c73dc24501569492ebb72cf97e6f4bcc33596ead1"
EXPECTED_CONFIG_FINGERPRINT: Final[str] = "414625aa8d2617cf89b324263205e8bbdbc92536a081e8b245823325bd78a4ba"
ACCURACY_MIN: Final[float] = 0.90
MACRO_F1_MIN: Final[float] = 0.90
PER_CLASS_RECALL_MIN: Final[dict[str, float]] = {"2": 0.90, "3": 0.90, "4": 0.90}
_MAX_JSON_BYTES: Final[int] = 16 * 1024 * 1024
_MAX_CHECKPOINT_BYTES: Final[int] = 16 * 1024 * 1024


class MeterV4_5Error(RuntimeError):
    """Raised when the one-time final-holdout protocol is violated."""


def _fail(message: str) -> None:
    raise MeterV4_5Error(message)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise MeterV4_5Error("value is not canonical JSON") from exc


def _hex64(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        _fail(f"{name} must be canonical lowercase SHA-256")
    return value


def _read_bounded_json(path: Path, *, expected_sha256: str | None = None) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        _fail(f"JSON must be a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > _MAX_JSON_BYTES:
        _fail(f"JSON size outside V4-5 bounds: {path}")
    raw = path.read_bytes()
    if expected_sha256 is not None and sha256(raw).hexdigest() != expected_sha256:
        _fail(f"JSON SHA-256 mismatch: {path.name}")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeterV4_5Error(f"invalid canonical ASCII JSON: {path}") from exc
    if not isinstance(value, dict):
        _fail("JSON root must be an object")
    return value


def _require_false_fields(name: str, value: Mapping[str, object], fields: Sequence[str]) -> None:
    for field in fields:
        if value.get(field) is not False:
            _fail(f"{name}.{field} must remain false")


def validate_preregistration(path: str | Path) -> dict[str, object]:
    value = _read_bounded_json(Path(path))
    if value.get("schema") != PREREG_SCHEMA or value.get("stage") != V4_5_STAGE:
        _fail("V4-5 preregistration identity mismatch")
    if value.get("status") != "PREREGISTERED_NO_FINAL_INFERENCE":
        _fail("V4-5 preregistration status changed")
    parents = value.get("parents")
    if not isinstance(parents, Mapping):
        _fail("V4-5 preregistration parent bindings missing")
    expected_parents = {
        "v4_4_selection_sha256": EXPECTED_SELECTION_SHA256,
        "v4_4_image_binding_sha256": EXPECTED_IMAGE_BINDING_SHA256,
        "v4_4_bbox_manifest_sha256": EXPECTED_BBOX_MANIFEST_SHA256,
        "v4_4_completion_receipt_file_sha256": EXPECTED_COMPLETION_RECEIPT_SHA256,
        "v4_2_result_file_sha256": EXPECTED_V4_2_RESULT_SHA256,
        "v4_2_checkpoint_file_sha256": EXPECTED_CHECKPOINT_SHA256,
        "v4_2_model_state_sha256": EXPECTED_MODEL_STATE_SHA256,
        "v4_2_configuration_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
    }
    if dict(parents) != expected_parents:
        _fail("V4-5 preregistration parent bindings differ from frozen protocol")
    surface = value.get("surface")
    if not isinstance(surface, Mapping) or dict(surface) != {
        "records": 150,
        "families": 150,
        "classes": {"2": 50, "3": 50, "4": 50},
        "holdout_read_only": True,
        "sealed_test_opened": False,
    }:
        _fail("V4-5 preregistration surface changed")
    representation = value.get("representation")
    expected_representation = {
        "source_coordinates": "original_image_integer_pixels",
        "horizontal_padding_milli": 150,
        "vertical_padding_milli": 50,
        "numerator_fraction_milli": 500,
        "output_size": 64,
        "grayscale_mode": "L",
        "resample": "BILINEAR",
        "ink_normalization": "(255-gray)/255",
    }
    if not isinstance(representation, Mapping) or dict(representation) != expected_representation:
        _fail("V4-5 representation policy changed")
    evaluation = value.get("evaluation")
    expected_eval = {
        "checkpoint_deserializations_max": 1,
        "inference_records_exact": 150,
        "optimizer_steps": 0,
        "training": False,
        "tuning": False,
        "calibration": False,
        "threshold_search": False,
        "test_time_augmentation": False,
        "ensemble": False,
        "rerun_after_lock": False,
    }
    if not isinstance(evaluation, Mapping) or dict(evaluation) != expected_eval:
        _fail("V4-5 one-shot evaluation policy changed")
    gate = value.get("gate")
    if not isinstance(gate, Mapping) or dict(gate) != {
        "accuracy_min": ACCURACY_MIN,
        "macro_f1_min": MACRO_F1_MIN,
        "per_class_recall_min": PER_CLASS_RECALL_MIN,
        "pass_name": "FINAL_HOLDOUT_PASS",
        "fail_name": "FINAL_HOLDOUT_FAIL",
    }:
        _fail("V4-5 preregistered metric gate changed")
    downstream = value.get("downstream")
    if not isinstance(downstream, Mapping):
        _fail("V4-5 downstream safety section missing")
    _require_false_fields("preregistration.downstream", downstream, ("runtime_connected", "resolver_connected", "production_promotion_authorized"))
    return value


def validate_human_review_evidence(path: str | Path) -> dict[str, object]:
    value = _read_bounded_json(Path(path))
    if value.get("schema") != HUMAN_REVIEW_SCHEMA or value.get("review_status") != "PASS":
        _fail("V4-4 human visual review PASS evidence missing")
    if value.get("reviewed_count") != 150 or value.get("contact_sheet_count") != 6:
        _fail("V4-4 human visual review cardinality changed")
    files = value.get("contact_sheet_files")
    if files != [f"contact_sheet_{index:02d}.png" for index in range(1, 7)]:
        _fail("V4-4 contact-sheet evidence changed")
    confirmation = value.get("human_confirmation")
    if not isinstance(confirmation, Mapping) or confirmation.get("explicit") is not True:
        _fail("V4-4 explicit human confirmation missing")
    receipt = value.get("mechanical_receipt")
    if not isinstance(receipt, Mapping):
        _fail("V4-4 mechanical receipt binding missing")
    expected_receipt = {
        "schema": COMPLETE_SCHEMA,
        "sha256": EXPECTED_COMPLETION_RECEIPT_SHA256,
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "image_binding_sha256": EXPECTED_IMAGE_BINDING_SHA256,
        "bbox_manifest_sha256": EXPECTED_BBOX_MANIFEST_SHA256,
        "annotated_count": 150,
        "missing_bbox": 0,
        "invalid_bbox": 0,
        "unique_family_count": 150,
        "class_counts": {"2": 50, "3": 50, "4": 50},
    }
    if dict(receipt) != expected_receipt:
        _fail("V4-4 human evidence is not bound to the exact mechanical receipt")
    assertions = value.get("review_assertions")
    if not isinstance(assertions, Mapping) or set(assertions) != {
        "both_meter_digits_contained", "digits_not_cut", "correct_meter_target", "no_symbol_drift", "visible_meter_sign"
    } or any(flag is not True for flag in assertions.values()):
        _fail("V4-4 human-review assertions are incomplete")
    downstream = value.get("downstream_gates")
    if not isinstance(downstream, Mapping):
        _fail("V4-4 downstream gates missing")
    _require_false_fields(
        "human_review.downstream_gates",
        downstream,
        ("model_evaluated", "candidate_checkpoint_opened", "test_opened", "runtime_connected", "production_promotion_authorized"),
    )
    if downstream.get("inference_count") != 0:
        _fail("V4-4 evidence shows prior final-holdout inference")
    decision = value.get("decision")
    if not isinstance(decision, Mapping) or decision.get("v4_4_complete") is not True or decision.get("v4_5_one_time_independent_evaluation_may_be_prepared") is not True:
        _fail("V4-4 evidence does not authorize V4-5 preparation")
    if decision.get("production_promotion_authorized") is not False:
        _fail("V4-4 evidence unexpectedly authorizes production")
    return value


@dataclass(frozen=True, slots=True)
class PreparedRecord:
    index: int
    family_id: str
    folder_name: str
    true_class: str
    image_sha256: str
    bbox_file_sha256: str
    bbox: BBox
    tensor: torch.Tensor


@dataclass(frozen=True, slots=True)
class MetricSummary:
    record_count: int
    accuracy: float
    macro_f1: float
    per_class_recall: dict[str, float]
    confusion: tuple[tuple[int, int, int], ...]


def numerator_bounds_original_v4_5(
    bbox: BBox, *, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    if any(type(v) is not int for v in (image_width, image_height)) or image_width <= 0 or image_height <= 0:
        _fail("original image dimensions must be positive integers")
    if any(type(v) is not int for v in (bbox.x, bbox.y, bbox.w, bbox.h)):
        _fail("full-meter bbox must use integer pixels")
    if bbox.x < 0 or bbox.y < 0 or bbox.w <= 0 or bbox.h <= 0 or bbox.x + bbox.w > image_width or bbox.y + bbox.h > image_height:
        _fail("full-meter bbox is outside original image")
    cfg = FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0
    x_pad = bbox.w * (cfg.horizontal_padding_milli / 1000.0)
    y_pad = bbox.h * (cfg.vertical_padding_milli / 1000.0)
    numerator_bottom = bbox.y + bbox.h * (cfg.numerator_fraction_milli / 1000.0)
    left = max(0, int(math.floor(bbox.x - x_pad)))
    top = max(0, int(math.floor(bbox.y - y_pad)))
    right = min(image_width, int(math.ceil(bbox.x + bbox.w + x_pad)))
    bottom = min(image_height, int(math.ceil(numerator_bottom + y_pad)))
    if not (0 <= left < right <= image_width and 0 <= top < bottom <= image_height):
        _fail("derived numerator crop is empty")
    return left, top, right, bottom


def render_numerator_crop_original_v4_5(image: Image.Image, bbox: BBox) -> Image.Image:
    if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
        _fail("V4-5 source image is invalid")
    bounds = numerator_bounds_original_v4_5(bbox, image_width=image.width, image_height=image.height)
    gray = image.convert("L")
    crop = gray.crop(bounds)
    output_size = FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0.output_size
    scale = min(output_size / crop.width, output_size / crop.height)
    resized_w = max(1, min(output_size, int(round(crop.width * scale))))
    resized_h = max(1, min(output_size, int(round(crop.height * scale))))
    resized = crop.resize((resized_w, resized_h), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("L", (output_size, output_size), 255)
    canvas.paste(resized, ((output_size - resized_w) // 2, (output_size - resized_h) // 2))
    return canvas


def crop_tensor_v4_5(image: Image.Image, bbox: BBox) -> torch.Tensor:
    crop = render_numerator_crop_original_v4_5(image, bbox)
    values = torch.tensor(list(crop.tobytes()), dtype=torch.float32).reshape(1, 64, 64)
    tensor = (255.0 - values) / 255.0
    assert_finite_tensor("V4-5 numerator tensor", tensor)
    if bool((tensor < 0).any()) or bool((tensor > 1).any()):
        _fail("V4-5 numerator tensor left [0,1]")
    return tensor


def _receipt_records(receipt: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    if receipt.get("schema") != COMPLETE_SCHEMA:
        _fail("V4-4 completion receipt schema mismatch")
    required = {
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "image_binding_sha256": EXPECTED_IMAGE_BINDING_SHA256,
        "bbox_manifest_sha256": EXPECTED_BBOX_MANIFEST_SHA256,
        "annotated_count": 150,
        "missing_bbox": 0,
        "invalid_bbox": 0,
        "unique_family_count": 150,
        "class_counts": {"2": 50, "3": 50, "4": 50},
        "human_visual_review_passed": False,
        "model_evaluated": False,
        "inference_count": 0,
        "candidate_checkpoint_opened": False,
        "test_opened": False,
        "runtime_connected": False,
        "production_promotion_authorized": False,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            _fail(f"V4-4 completion receipt field changed: {key}")
    records = receipt.get("records")
    if not isinstance(records, list) or len(records) != 150 or any(not isinstance(row, Mapping) for row in records):
        _fail("V4-4 completion receipt must contain exactly 150 records")
    if sha256(canonical_json(records)).hexdigest() != EXPECTED_BBOX_MANIFEST_SHA256:
        _fail("V4-4 receipt records no longer reproduce bbox manifest SHA")
    return tuple(records)


def prepare_final_holdout_v4_5(
    *, candidate_root: str | Path, manifest_path: str | Path, completion_receipt_path: str | Path
) -> tuple[PreparedRecord, ...]:
    receipt_path = Path(completion_receipt_path)
    if sha256_file(receipt_path) != EXPECTED_COMPLETION_RECEIPT_SHA256:
        _fail("V4-4 completion receipt file SHA mismatch")
    receipt = read_json(receipt_path)
    receipt_rows = _receipt_records(receipt)
    _manifest, rows = load_and_validate_selection_manifest(
        manifest_path, expected_selection_sha256=EXPECTED_SELECTION_SHA256
    )
    samples = discover_selected_samples(candidate_root, rows)
    if len(samples) != EXPECTED_SELECTED_COUNT:
        _fail("V4-5 final holdout cardinality is not 150")
    by_folder = {str(row.get("folder_name")): row for row in receipt_rows}
    if len(by_folder) != 150:
        _fail("V4-4 receipt folder identities are not unique")
    prepared: list[PreparedRecord] = []
    families: set[str] = set()
    classes: Counter[str] = Counter()
    for sample in samples:
        row = by_folder.get(sample.folder_name)
        if row is None:
            _fail("selected folder missing from V4-4 completion receipt")
        info = read_png_info(sample.image_path)
        bbox_contract = read_bbox_contract(
            sample.bbox_path,
            expected_meter=sample.meter_class,
            image_width=info.width,
            image_height=info.height,
        )
        if bbox_contract.bbox is None:
            _fail("V4-5 cannot evaluate a missing bbox")
        expected_bbox = bbox_contract.bbox.as_dict()
        expected_identity = {
            "index": sample.index,
            "family_id": sample.family_id,
            "folder_name": sample.folder_name,
            "numerator_class": sample.numerator_class,
            "meter_class": sample.meter_class,
            "image_sha256": info.sha256,
            "image_width": info.width,
            "image_height": info.height,
            "bbox": expected_bbox,
            "bbox_file_sha256": sha256_file(sample.bbox_path),
        }
        for key, expected in expected_identity.items():
            if row.get(key) != expected:
                _fail(f"V4-5 current holdout differs from V4-4 receipt: {sample.folder_name}:{key}")
        try:
            with Image.open(sample.image_path) as opened:
                opened.load()
                if opened.format != "PNG":
                    _fail("V4-5 source must remain PNG")
                tensor = crop_tensor_v4_5(opened, bbox_contract.bbox)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise MeterV4_5Error(f"V4-5 image decode failed: {sample.folder_name}") from exc
        if sample.family_id in families:
            _fail("V4-5 family duplication detected")
        families.add(sample.family_id)
        classes[sample.numerator_class] += 1
        prepared.append(
            PreparedRecord(
                index=sample.index,
                family_id=sample.family_id,
                folder_name=sample.folder_name,
                true_class=sample.numerator_class,
                image_sha256=info.sha256,
                bbox_file_sha256=str(expected_identity["bbox_file_sha256"]),
                bbox=bbox_contract.bbox,
                tensor=tensor,
            )
        )
    if len(families) != 150 or dict(classes) != EXPECTED_CLASS_COUNTS:
        _fail("V4-5 final holdout family/class balance changed")
    return tuple(prepared)


def validate_v4_2_result(path: str | Path) -> dict[str, object]:
    value = _read_bounded_json(Path(path), expected_sha256=EXPECTED_V4_2_RESULT_SHA256)
    checkpoint = value.get("candidate_checkpoint")
    decision = value.get("decision")
    safety = value.get("safety")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("sha256") != EXPECTED_CHECKPOINT_SHA256 or checkpoint.get("model_state_sha256") != EXPECTED_MODEL_STATE_SHA256:
        _fail("V4-2 result candidate binding mismatch")
    if checkpoint.get("development_candidate_authorized") is not True or checkpoint.get("production_candidate_authorized") is not False:
        _fail("V4-2 result candidate authorization state changed")
    if value.get("configuration_fingerprint") != EXPECTED_CONFIG_FINGERPRINT:
        _fail("V4-2 configuration fingerprint mismatch")
    if not isinstance(decision, Mapping) or decision.get("accepted_for_shadow_planning") is not True or decision.get("fresh_independent_holdout_required") is not True or decision.get("production_promotion_authorized") is not False:
        _fail("V4-2 decision does not admit independent final holdout")
    if not isinstance(safety, Mapping):
        _fail("V4-2 safety evidence missing")
    if safety.get("test_opened") is not False or safety.get("runtime_connected") is not False or safety.get("resolver_connected") is not False or safety.get("production_promotion_authorized") is not False:
        _fail("V4-2 crossed a forbidden safety boundary")
    return value


def validate_checkpoint_file_hash(path: str | Path) -> Path:
    checkpoint = Path(path)
    if not checkpoint.is_file() or checkpoint.is_symlink():
        _fail("V4-2 checkpoint must be a regular non-symlink file")
    size = checkpoint.stat().st_size
    if size <= 0 or size > _MAX_CHECKPOINT_BYTES:
        _fail("V4-2 checkpoint size outside V4-5 bounds")
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        _fail("V4-2 checkpoint SHA-256 mismatch")
    return checkpoint


def _load_exact_candidate_after_lock(checkpoint: Path) -> NumeratorSpecialistV4_1:
    # This function MUST only be called after the persistent one-shot lock exists.
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise MeterV4_5Error("trusted-hash V4-2 checkpoint could not be loaded safely") from exc
    if not isinstance(payload, Mapping):
        _fail("V4-2 checkpoint payload is not a mapping")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        _fail("V4-2 checkpoint schema mismatch")
    if payload.get("model_state_sha256") != EXPECTED_MODEL_STATE_SHA256:
        _fail("V4-2 checkpoint model-state binding mismatch")
    if payload.get("config_fingerprint_v4_1") != EXPECTED_CONFIG_FINGERPRINT:
        _fail("V4-2 checkpoint config fingerprint mismatch")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        _fail("V4-2 checkpoint state_dict missing")
    model = NumeratorSpecialistV4_1(FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1).cpu()
    if count_trainable_parameters(model) != EXPECTED_PARAMETER_COUNT_V4_1:
        _fail("V4-1 frozen architecture parameter count changed")
    try:
        incompatible = model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise MeterV4_5Error("V4-2 checkpoint state_dict is incompatible") from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        _fail("V4-2 state_dict key set changed")
    assert_model_finite(model)
    if model_state_sha256(model) != EXPECTED_MODEL_STATE_SHA256:
        _fail("loaded V4-2 model-state SHA mismatch")
    if config_fingerprint_v4_1() != EXPECTED_CONFIG_FINGERPRINT:
        _fail("runtime frozen V4-1 configuration fingerprint changed")
    model.eval()
    return model


def summarize_predictions_v4_5(true_classes: Sequence[str], predicted_classes: Sequence[str]) -> MetricSummary:
    if len(true_classes) != 150 or len(predicted_classes) != 150:
        _fail("V4-5 metrics require exactly 150 predictions")
    classes = NUMERATOR_CLASSES_V4_1
    if tuple(classes) != ("2", "3", "4"):
        _fail("V4-5 class order changed")
    if Counter(true_classes) != Counter({"2": 50, "3": 50, "4": 50}):
        _fail("V4-5 truth class counts changed")
    if any(value not in classes for value in predicted_classes):
        _fail("V4-5 prediction outside frozen classes")
    confusion = [[0, 0, 0] for _ in classes]
    for truth, pred in zip(true_classes, predicted_classes):
        confusion[classes.index(truth)][classes.index(pred)] += 1
    correct = sum(confusion[i][i] for i in range(3))
    recalls: dict[str, float] = {}
    f1s: list[float] = []
    for i, name in enumerate(classes):
        tp = confusion[i][i]
        fn = sum(confusion[i]) - tp
        fp = sum(confusion[row][i] for row in range(3)) - tp
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls[name] = recall
        f1s.append(f1)
    return MetricSummary(
        record_count=150,
        accuracy=correct / 150.0,
        macro_f1=sum(f1s) / 3.0,
        per_class_recall=recalls,
        confusion=tuple(tuple(row) for row in confusion),
    )


def final_decision_v4_5(summary: MetricSummary) -> dict[str, object]:
    reasons: list[str] = []
    if summary.record_count != 150:
        reasons.append("FINAL_RECORD_COUNT_NOT_150")
    if summary.accuracy < ACCURACY_MIN:
        reasons.append("FINAL_ACCURACY_BELOW_90_PERCENT")
    if summary.macro_f1 < MACRO_F1_MIN:
        reasons.append("FINAL_MACRO_F1_BELOW_90_PERCENT")
    for name in ("2", "3", "4"):
        if summary.per_class_recall.get(name, -1.0) < PER_CLASS_RECALL_MIN[name]:
            reasons.append(f"FINAL_{name}_RECALL_BELOW_90_PERCENT")
    passed = not reasons
    return {
        "name": "FINAL_HOLDOUT_PASS" if passed else "FINAL_HOLDOUT_FAIL",
        "accepted_for_next_bounded_review": passed,
        "production_promotion_authorized": False,
        "reasons": reasons,
        "thresholds": {
            "accuracy_min": ACCURACY_MIN,
            "macro_f1_min": MACRO_F1_MIN,
            "per_class_recall_min": dict(PER_CLASS_RECALL_MIN),
        },
    }


def _exclusive_lock(path: Path, payload: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json(dict(payload)) + b"\n"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise MeterV4_5Error(f"V4-5 one-shot lock already exists: {path}") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise
    return sha256(raw).hexdigest()


def _write_fresh_result(output: Path, result: Mapping[str, object]) -> str:
    temporary = output.with_name("." + output.name + ".part")
    if output.exists() or output.is_symlink() or temporary.exists() or temporary.is_symlink():
        _fail("V4-5 output/partial output must be fresh")
    temporary.mkdir(parents=True)
    try:
        raw = _canonical_json(dict(result))
        (temporary / "result.json").write_bytes(raw)
        result_sha = sha256(raw).hexdigest()
        (temporary / "COMPLETE").write_bytes(f"{result_sha}  result.json\n".encode("ascii"))
        temporary.replace(output)
    except Exception:
        raise
    return result_sha


def run_meter_v4_5_one_time_final_holdout_evaluation(
    *,
    candidate_root: str | Path,
    manifest_path: str | Path,
    completion_receipt_path: str | Path,
    human_review_evidence_path: str | Path,
    preregistration_path: str | Path,
    v4_2_result_path: str | Path,
    checkpoint_path: str | Path,
    output_root: str | Path,
    git_commit_sha: str,
) -> dict[str, object]:
    if not isinstance(git_commit_sha, str) or len(git_commit_sha) != 40 or any(ch not in "0123456789abcdef" for ch in git_commit_sha):
        _fail("git_commit_sha must be canonical lowercase SHA-1")
    output = Path(output_root)
    lock_path = output.with_name(output.name + ".ONE_SHOT_LOCK.json")
    part = output.with_name("." + output.name + ".part")
    if output.exists() or output.is_symlink() or part.exists() or part.is_symlink() or lock_path.exists() or lock_path.is_symlink():
        _fail("V4-5 output/partial/one-shot lock must all be absent before first run")

    prereg = validate_preregistration(preregistration_path)
    human = validate_human_review_evidence(human_review_evidence_path)
    prepared = prepare_final_holdout_v4_5(
        candidate_root=candidate_root,
        manifest_path=manifest_path,
        completion_receipt_path=completion_receipt_path,
    )
    v4_2_result = validate_v4_2_result(v4_2_result_path)
    checkpoint = validate_checkpoint_file_hash(checkpoint_path)

    preflight_binding = sha256(
        _canonical_json(
            {
                "stage": V4_5_STAGE,
                "git_commit_sha": git_commit_sha,
                "selection_sha256": EXPECTED_SELECTION_SHA256,
                "image_binding_sha256": EXPECTED_IMAGE_BINDING_SHA256,
                "bbox_manifest_sha256": EXPECTED_BBOX_MANIFEST_SHA256,
                "completion_receipt_sha256": EXPECTED_COMPLETION_RECEIPT_SHA256,
                "v4_2_result_sha256": EXPECTED_V4_2_RESULT_SHA256,
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "model_state_sha256": EXPECTED_MODEL_STATE_SHA256,
                "record_bindings": [
                    {
                        "index": row.index,
                        "family_id": row.family_id,
                        "folder_name": row.folder_name,
                        "true_class": row.true_class,
                        "image_sha256": row.image_sha256,
                        "bbox_file_sha256": row.bbox_file_sha256,
                        "bbox": row.bbox.as_dict(),
                    }
                    for row in prepared
                ],
            }
        )
    ).hexdigest()
    lock_payload = {
        "schema": "st-omr-meter-v4-5-one-shot-lock-v1",
        "stage": V4_5_STAGE,
        "status": "LOCKED_BEFORE_CHECKPOINT_OPEN",
        "git_commit_sha": git_commit_sha,
        "preflight_binding_sha256": preflight_binding,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "inference_records_exact": 150,
        "rerun_authorized": False,
    }
    lock_sha = _exclusive_lock(lock_path, lock_payload)

    model = _load_exact_candidate_after_lock(checkpoint)
    batch = torch.stack([row.tensor for row in prepared], dim=0)
    if tuple(batch.shape) != (150, 1, 64, 64) or batch.dtype != torch.float32:
        _fail("V4-5 final inference batch shape/dtype changed")
    assert_finite_tensor("V4-5 final batch", batch)
    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)
    assert_finite_tensor("V4-5 final logits", logits)
    assert_finite_tensor("V4-5 final probabilities", probabilities)
    if tuple(logits.shape) != (150, 3) or tuple(probabilities.shape) != (150, 3):
        _fail("V4-5 model output shape changed")
    pred_indexes = torch.argmax(probabilities, dim=1).cpu().tolist()
    predicted_classes = [NUMERATOR_CLASSES_V4_1[int(index)] for index in pred_indexes]
    true_classes = [row.true_class for row in prepared]
    summary = summarize_predictions_v4_5(true_classes, predicted_classes)
    decision = final_decision_v4_5(summary)

    predictions = []
    for index, row in enumerate(prepared):
        logit_values = [float(v) for v in logits[index].cpu().tolist()]
        prob_values = [float(v) for v in probabilities[index].cpu().tolist()]
        if any(not math.isfinite(v) for v in (*logit_values, *prob_values)):
            _fail("V4-5 prediction contains non-finite values")
        predictions.append(
            {
                "index": row.index,
                "family_id": row.family_id,
                "folder_name": row.folder_name,
                "true": row.true_class,
                "pred": predicted_classes[index],
                "logits": {"2": logit_values[0], "3": logit_values[1], "4": logit_values[2]},
                "probabilities": {"2": prob_values[0], "3": prob_values[1], "4": prob_values[2]},
            }
        )

    result = {
        "schema": V4_5_SCHEMA,
        "stage": V4_5_STAGE,
        "git_commit_sha": git_commit_sha,
        "preflight_binding_sha256": preflight_binding,
        "one_shot_lock": {"filename": lock_path.name, "sha256": lock_sha, "rerun_authorized": False},
        "parents": {
            "selection_sha256": EXPECTED_SELECTION_SHA256,
            "image_binding_sha256": EXPECTED_IMAGE_BINDING_SHA256,
            "bbox_manifest_sha256": EXPECTED_BBOX_MANIFEST_SHA256,
            "completion_receipt_sha256": EXPECTED_COMPLETION_RECEIPT_SHA256,
            "human_review_schema": human["schema"],
            "preregistration_schema": prereg["schema"],
            "v4_2_result_sha256": EXPECTED_V4_2_RESULT_SHA256,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "model_state_sha256": EXPECTED_MODEL_STATE_SHA256,
            "configuration_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
        },
        "representation": dict(prereg["representation"]),
        "final_holdout": {
            "record_count": summary.record_count,
            "families": 150,
            "classes": {"2": 50, "3": 50, "4": 50},
            "accuracy": summary.accuracy,
            "macro_f1": summary.macro_f1,
            "per_class_recall": dict(summary.per_class_recall),
            "confusion": [list(row) for row in summary.confusion],
            "predictions": predictions,
        },
        "decision": decision,
        "safety": {
            "model_evaluated": True,
            "inference_count": 150,
            "candidate_checkpoint_opened": True,
            "checkpoint_deserializations": 1,
            "optimizer_steps": 0,
            "training": False,
            "tuning": False,
            "calibration": False,
            "threshold_search": False,
            "test_time_augmentation": False,
            "holdout_written": False,
            "sealed_test_opened": False,
            "runtime_connected": False,
            "resolver_connected": False,
            "production_promotion_authorized": False,
            "rerun_authorized": False,
        },
        "v4_2_candidate_decision": v4_2_result["decision"],
    }
    _write_fresh_result(output, result)
    return result
