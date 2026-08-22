"""METER V5-1R1 specialist-input contract audit.

This stage is intentionally narrow: it consumes only the already-approved
30-sample V5-1 TRAIN bbox pilot, splits each full-meter bbox once at its integer
vertical midpoint, applies the frozen historical 64x64 digit preprocessing, and
asks the already-frozen 2-AI / 3-AI / 4-AI specialists the resulting visual
question.

No training, threshold tuning, validation/final-holdout image access, D11
execution, runtime geometry mutation, Resolver wiring, or production promotion
is permitted here.
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
from typing import Callable, Final, Mapping

from PIL import Image, UnidentifiedImageError
import torch
from torch import nn

from .meter_v2_digit_crop_profile_v1 import meter_v2_digit_crop_profile_fingerprint_v1
from .meter_v5_1_bbox_pilot import (
    DATASET_NAME,
    PILOT_CSV_NAME,
    PILOT_SELECTION_NAME,
    PILOT_TOTAL,
)
from .runtime_meter_integration_v3 import DIGIT_THRESHOLDS_MILLI
from .runtime_meter_real_checkpoint_audit_v1 import (
    AuditedCheckpointStateV1,
    DIGIT2_SHA256,
    DIGIT3_SHA256,
    DIGIT4_SHA256,
    audit_digit_checkpoint_v1,
    conservative_probability_to_milli_v1,
)


METER_V5_1R1_VERSION: Final[str] = "meter-v5-1r1-specialist-input-audit-v1"
PARENT_HEAD_SHA: Final[str] = "4f744d0bf8a2f6180a62f0f08abb96b83cfb5da8"
PILOT_EVIDENCE_SCHEMA: Final[str] = "st-omr-meter-v5-1-bbox-pilot-result-evidence-v1"
PILOT_EVIDENCE_RELATIVE_PATH: Final[str] = "evidence/METER_V5_1_BBOX_PILOT_RESULT.json"
EXPECTED_METERS: Final[tuple[str, ...]] = ("2/4", "3/4", "4/4")
EXPECTED_PER_CLASS: Final[int] = 10
REPLAY_COUNT: Final[int] = 10
CROP_SIZE: Final[int] = 64

EXPECTED_SLOT_DIGITS: Final[dict[str, tuple[int, int]]] = {
    "2/4": (2, 4),
    "3/4": (3, 4),
    "4/4": (4, 4),
}
EXPECTED_CHECKPOINT_SHA: Final[dict[int, str]] = {
    2: DIGIT2_SHA256,
    3: DIGIT3_SHA256,
    4: DIGIT4_SHA256,
}
EXPECTED_CHECKPOINT_ROLE: Final[dict[int, str]] = {
    2: "digit-2",
    3: "digit-3",
    4: "digit-4",
}


class MeterV5_1R1AuditError(RuntimeError):
    """Fail-closed error at the V5-1R1 audit boundary."""


@dataclass(frozen=True, slots=True)
class DigitSlotBoxV1:
    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        for value in (self.x0, self.y0, self.x1, self.y1):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("digit slot coordinates must be plain integers")
        if not self.x0 < self.x1 or not self.y0 < self.y1:
            raise ValueError("digit slot must have positive area")

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass(frozen=True, slots=True)
class CropScoreV1:
    probabilities: Mapping[int, float]
    replay_stable: bool

    def __post_init__(self) -> None:
        if set(self.probabilities) != {2, 3, 4}:
            raise ValueError("crop score must contain exactly 2/3/4 probabilities")
        for digit, value in self.probabilities.items():
            if digit not in (2, 3, 4):
                raise ValueError("unsupported digit specialist")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("specialist probability must be numeric")
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError("specialist probability must be finite in 0..1")
        if not isinstance(self.replay_stable, bool):
            raise ValueError("replay_stable must be bool")


class RuntimeDigitSpecialistV1(nn.Module):
    """Runtime mirror of the already-audited frozen 2/3/4 architecture."""

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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype != torch.float32 or tuple(images.shape[1:]) != (1, CROP_SIZE, CROP_SIZE):
            raise MeterV5_1R1AuditError("digit specialist input must be float32 [B,1,64,64]")
        features = self.features(images).flatten(1)
        logits = self.head(features).squeeze(1)
        if not bool(torch.isfinite(logits).all().item()):
            raise MeterV5_1R1AuditError("digit specialist produced non-finite logits")
        return logits


def _sha256_bytes(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def specialist_input_audit_profile_fingerprint_v1() -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "version": METER_V5_1R1_VERSION,
                "parent_head_sha": PARENT_HEAD_SHA,
                "surface": "v5-1-train-pilot-30-only",
                "full_meter_to_slots": "same-x-integer-midpoint-no-padding-v1",
                "digit_crop_profile": meter_v2_digit_crop_profile_fingerprint_v1(),
                "digit_checkpoint_sha": EXPECTED_CHECKPOINT_SHA,
                "digit_thresholds_milli": DIGIT_THRESHOLDS_MILLI,
                "replay_count": REPLAY_COUNT,
                "threshold_transport": "conservative-floor-milli-v1",
                "d11": False,
                "optimizer": False,
                "threshold_tuning": False,
                "validation": False,
                "final_holdout": False,
                "resolver": False,
            }
        )
    )


def derive_digit_slots_from_full_meter_bbox_v1(
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    image_width: int,
    image_height: int,
) -> tuple[DigitSlotBoxV1, DigitSlotBoxV1]:
    """Split one approved full-meter bbox once at floor(h/2), without search."""
    for name, value in {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "image_width": image_width,
        "image_height": image_height,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise MeterV5_1R1AuditError(f"{name} must be a plain integer")
    if image_width <= 0 or image_height <= 0 or w <= 0 or h <= 1:
        raise MeterV5_1R1AuditError("full-meter bbox/image dimensions are invalid")
    x1 = x + w
    y1 = y + h
    if not (0 <= x < x1 <= image_width and 0 <= y < y1 <= image_height):
        raise MeterV5_1R1AuditError("full-meter bbox lies outside the source image")
    split_y = y + h // 2
    if not y < split_y < y1:
        raise MeterV5_1R1AuditError("full-meter bbox cannot produce two positive slots")
    return (
        DigitSlotBoxV1(x, y, x1, split_y),
        DigitSlotBoxV1(x, split_y, x1, y1),
    )


def crop_digit_to_64_v1(image: Image.Image, box: DigitSlotBoxV1) -> Image.Image:
    """Apply the frozen historical digit crop transform to one admitted slot."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be PIL.Image.Image")
    if not (0 <= box.x0 < box.x1 <= image.width and 0 <= box.y0 < box.y1 <= image.height):
        raise MeterV5_1R1AuditError("digit slot lies outside source image")
    crop = image.crop((box.x0, box.y0, box.x1, box.y1)).convert("L")
    crop.thumbnail((CROP_SIZE, CROP_SIZE), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (CROP_SIZE, CROP_SIZE), 255)
    offset_x = (CROP_SIZE - crop.width) // 2
    offset_y = (CROP_SIZE - crop.height) // 2
    canvas.paste(crop, (offset_x, offset_y))
    return canvas


def digit_crop_png_bytes_v1(crop: Image.Image) -> bytes:
    if not isinstance(crop, Image.Image) or crop.mode != "L" or crop.size != (CROP_SIZE, CROP_SIZE):
        raise MeterV5_1R1AuditError("audit crop must be exact L 64x64")
    out = BytesIO()
    crop.save(out, format="PNG", optimize=False, compress_level=9)
    raw = out.getvalue()
    if not raw:
        raise MeterV5_1R1AuditError("digit crop PNG encoding failed")
    return raw


def _tensor_from_crop(crop: Image.Image) -> torch.Tensor:
    if crop.mode != "L" or crop.size != (CROP_SIZE, CROP_SIZE):
        raise MeterV5_1R1AuditError("digit crop must be exact L 64x64")
    raw = bytearray(crop.tobytes())
    tensor = torch.frombuffer(raw, dtype=torch.uint8).clone().reshape(CROP_SIZE, CROP_SIZE)
    # Historical digit training used uint8 / 255.0, not ink inversion.
    tensor = tensor.to(dtype=torch.float32) / 255.0
    return tensor.unsqueeze(0).unsqueeze(0)


def build_models_from_audited_states_v1(
    audited_states: Mapping[int, AuditedCheckpointStateV1],
) -> dict[int, RuntimeDigitSpecialistV1]:
    if set(audited_states) != {2, 3, 4}:
        raise MeterV5_1R1AuditError("audited digit states must contain exactly 2/3/4")
    models: dict[int, RuntimeDigitSpecialistV1] = {}
    for digit in (2, 3, 4):
        audited = audited_states[digit]
        if audited.role != EXPECTED_CHECKPOINT_ROLE[digit]:
            raise MeterV5_1R1AuditError("audited digit checkpoint role mismatch")
        if audited.checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA[digit]:
            raise MeterV5_1R1AuditError("audited digit checkpoint identity mismatch")
        model = RuntimeDigitSpecialistV1().cpu()
        try:
            model.load_state_dict(dict(audited.model_state), strict=True)
        except Exception as exc:
            raise MeterV5_1R1AuditError("audited digit state cannot populate runtime mirror") from exc
        model.eval()
        models[digit] = model
    return models


def make_frozen_checkpoint_scorer_v1(
    audited_states: Mapping[int, AuditedCheckpointStateV1],
) -> Callable[[Image.Image], CropScoreV1]:
    """Return an inference-only scorer requiring exact 10/10 replay stability."""
    models = build_models_from_audited_states_v1(audited_states)

    def score(crop: Image.Image) -> CropScoreV1:
        tensor = _tensor_from_crop(crop)
        per_digit: dict[int, list[float]] = {2: [], 3: [], 4: []}
        with torch.inference_mode():
            for _ in range(REPLAY_COUNT):
                for digit in (2, 3, 4):
                    logit = models[digit](tensor)[0]
                    probability = float(torch.sigmoid(logit).item())
                    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                        raise MeterV5_1R1AuditError("digit specialist probability is invalid")
                    per_digit[digit].append(probability)
        stable = all(len(set(values)) == 1 for values in per_digit.values())
        return CropScoreV1(
            probabilities={digit: values[0] for digit, values in per_digit.items()},
            replay_stable=stable,
        )

    return score


def arbitrate_digit_probabilities_v1(probabilities: Mapping[int, float]) -> dict[str, object]:
    if set(probabilities) != {2, 3, 4}:
        raise MeterV5_1R1AuditError("arbitration requires exactly 2/3/4 probabilities")
    milli: dict[int, int] = {}
    passing: list[int] = []
    for digit in (2, 3, 4):
        value = float(probabilities[digit])
        score = conservative_probability_to_milli_v1(value)
        milli[digit] = score
        if score >= DIGIT_THRESHOLDS_MILLI[digit]:
            passing.append(digit)
    if not passing:
        state = "NO_HIT"
    elif len(passing) == 1:
        state = "UNIQUE"
    else:
        state = "CONFLICT"
    return {
        "state": state,
        "passing_digits": passing,
        "scores_milli": {str(digit): milli[digit] for digit in (2, 3, 4)},
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise MeterV5_1R1AuditError(f"required CSV is not a regular file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise MeterV5_1R1AuditError(f"CSV header missing: {path}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise MeterV5_1R1AuditError(f"required JSON is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MeterV5_1R1AuditError(f"JSON cannot be parsed: {path}") from exc
    if not isinstance(payload, dict):
        raise MeterV5_1R1AuditError("pilot evidence must be a JSON object")
    return payload


def _plain_int(value: str, field: str) -> int:
    try:
        number = int(value, 10)
    except Exception as exc:
        raise MeterV5_1R1AuditError(f"{field} must be an integer") from exc
    if str(number) != value.strip() and value.strip() not in {f"+{number}", f"-{abs(number)}"}:
        # CSVs generated by the V5-1 tool use canonical decimal integers.
        raise MeterV5_1R1AuditError(f"{field} must use canonical integer text")
    return number


def _safe_selected_image(dataset_root: Path, relpath: str) -> Path:
    relative = Path(relpath)
    if relative.is_absolute() or ".." in relative.parts:
        raise MeterV5_1R1AuditError("pilot image path escapes dataset root")
    root = dataset_root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise MeterV5_1R1AuditError("pilot image path escapes dataset root")
    if candidate.is_symlink() or not candidate.is_file():
        raise MeterV5_1R1AuditError("pilot image must be a regular non-symlink file")
    return candidate


def _validate_parent_evidence(
    evidence: dict[str, object],
    *,
    selection_path: Path,
    annotation_path: Path,
) -> None:
    if evidence.get("schema") != PILOT_EVIDENCE_SCHEMA or evidence.get("stage") != "METER V5-1":
        raise MeterV5_1R1AuditError("wrong V5-1 parent evidence schema/stage")
    if evidence.get("pilot_result") != "PASS":
        raise MeterV5_1R1AuditError("V5-1 pilot result is not PASS")
    dataset = evidence.get("dataset")
    annotation_audit = evidence.get("annotation_audit")
    artifacts = evidence.get("artifacts")
    safety = evidence.get("safety")
    if not all(isinstance(item, dict) for item in (dataset, annotation_audit, artifacts, safety)):
        raise MeterV5_1R1AuditError("V5-1 parent evidence sections are malformed")
    assert isinstance(dataset, dict)
    assert isinstance(annotation_audit, dict)
    assert isinstance(artifacts, dict)
    assert isinstance(safety, dict)
    if dataset.get("name") != DATASET_NAME:
        raise MeterV5_1R1AuditError("V5-1 dataset identity mismatch")
    if not (
        annotation_audit.get("annotation_count") == PILOT_TOTAL
        and annotation_audit.get("pass_count") == PILOT_TOTAL
        and annotation_audit.get("review_count") == 0
        and annotation_audit.get("mechanical_gate") == "PASS"
        and annotation_audit.get("original_pilot_image_binding_preserved") is True
        and annotation_audit.get("annotation_contract_freeze_ready") is True
    ):
        raise MeterV5_1R1AuditError("V5-1 annotation gate is not frozen PASS")
    if not (
        safety.get("annotation_scope") == "train_pilot_30_only"
        and safety.get("final_holdout_locked") is True
        and safety.get("training_authorized") is False
        and safety.get("model_opened") is False
    ):
        raise MeterV5_1R1AuditError("V5-1 safety state does not permit this bounded audit")
    selection_meta = artifacts.get("selection_csv")
    annotation_meta = artifacts.get("annotation_csv")
    if not isinstance(selection_meta, dict) or not isinstance(annotation_meta, dict):
        raise MeterV5_1R1AuditError("V5-1 artifact bindings are missing")
    if _sha256_file(selection_path) != selection_meta.get("sha256"):
        raise MeterV5_1R1AuditError("pilot selection CSV SHA-256 changed")
    if _sha256_file(annotation_path) != annotation_meta.get("sha256"):
        raise MeterV5_1R1AuditError("pilot annotation CSV SHA-256 changed")


def _load_admitted_rows(
    dataset_root: Path,
    evidence_path: Path,
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, object]]:
    if dataset_root.name != DATASET_NAME or not dataset_root.is_dir():
        raise MeterV5_1R1AuditError("audit requires exact clean V5 dataset root")
    annotations_dir = dataset_root / "annotations"
    selection_path = annotations_dir / PILOT_SELECTION_NAME
    annotation_path = annotations_dir / PILOT_CSV_NAME
    evidence = _read_json(evidence_path)
    _validate_parent_evidence(evidence, selection_path=selection_path, annotation_path=annotation_path)

    selection = _read_csv(selection_path)
    annotations = _read_csv(annotation_path)
    if len(selection) != PILOT_TOTAL or len(annotations) != PILOT_TOTAL:
        raise MeterV5_1R1AuditError("V5-1R1 admits exactly 30 selection and annotation rows")

    required_selection = {
        "sample_id", "family_id", "meter", "split", "image_relpath",
        "image_sha256", "image_width", "image_height",
    }
    required_annotation = {
        "sample_id", "meter", "split", "x", "y", "w", "h", "status",
        "image_sha256", "image_width", "image_height",
    }
    if not selection or not required_selection.issubset(selection[0]):
        raise MeterV5_1R1AuditError("pilot selection schema is incomplete")
    if not annotations or not required_annotation.issubset(annotations[0]):
        raise MeterV5_1R1AuditError("pilot annotation schema is incomplete")

    selection_ids = [row["sample_id"] for row in selection]
    annotation_ids = [row["sample_id"] for row in annotations]
    if len(set(selection_ids)) != PILOT_TOTAL or len(set(annotation_ids)) != PILOT_TOTAL:
        raise MeterV5_1R1AuditError("pilot sample ids must be unique")
    if set(selection_ids) != set(annotation_ids):
        raise MeterV5_1R1AuditError("selection/annotation sample ids differ")

    class_counts = Counter(row["meter"] for row in selection)
    if class_counts != Counter({meter: EXPECTED_PER_CLASS for meter in EXPECTED_METERS}):
        raise MeterV5_1R1AuditError("pilot class cardinality changed")
    if any(row["split"] != "train" or row["meter"] not in EXPECTED_METERS for row in selection):
        raise MeterV5_1R1AuditError("V5-1R1 accepts TRAIN pilot meter rows only")

    annotation_by_id = {row["sample_id"]: row for row in annotations}
    for row in selection:
        ann = annotation_by_id[row["sample_id"]]
        if ann["status"] != "PASS" or ann["split"] != "train" or ann["meter"] != row["meter"]:
            raise MeterV5_1R1AuditError("pilot annotation is not matching TRAIN PASS evidence")
        for field in ("image_sha256", "image_width", "image_height"):
            if ann[field] != row[field]:
                raise MeterV5_1R1AuditError("selection/annotation image binding mismatch")
    return selection, annotation_by_id, evidence


def _open_bound_image(path: Path, *, expected_sha: str, width: int, height: int) -> Image.Image:
    if _sha256_file(path) != expected_sha:
        raise MeterV5_1R1AuditError("selected image SHA-256 changed")
    try:
        with Image.open(path) as opened:
            opened.load()
            if opened.size != (width, height):
                raise MeterV5_1R1AuditError("selected image dimensions changed")
            return opened.copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise MeterV5_1R1AuditError("selected image cannot be decoded") from exc


def _slot_record(
    *,
    role: str,
    expected_digit: int,
    box: DigitSlotBoxV1,
    crop: Image.Image,
    score: CropScoreV1,
) -> dict[str, object]:
    arbitration = arbitrate_digit_probabilities_v1(score.probabilities)
    passing = arbitration["passing_digits"]
    assert isinstance(passing, list)
    state = str(arbitration["state"])
    selected = passing[0] if state == "UNIQUE" and len(passing) == 1 else None
    if not score.replay_stable:
        outcome = "NONDETERMINISTIC"
    elif state == "NO_HIT":
        outcome = "NO_HIT"
    elif state == "CONFLICT":
        outcome = "CONFLICT"
    elif selected != expected_digit:
        outcome = "WRONG_UNIQUE"
    else:
        outcome = "CORRECT"
    crop_bytes = digit_crop_png_bytes_v1(crop)
    return {
        "role": role,
        "expected_digit": expected_digit,
        "slot_box": box.as_list(),
        "crop_sha256": _sha256_bytes(crop_bytes),
        "crop_profile_fingerprint": meter_v2_digit_crop_profile_fingerprint_v1(),
        "probabilities": {str(digit): float(score.probabilities[digit]) for digit in (2, 3, 4)},
        "scores_milli": arbitration["scores_milli"],
        "passing_digits": passing,
        "arbitration": state,
        "replay_count": REPLAY_COUNT,
        "replay_stable": score.replay_stable,
        "outcome": outcome,
    }


def audit_pilot_with_scorer_v1(
    *,
    dataset_root: Path,
    pilot_evidence_path: Path,
    score_crop: Callable[[Image.Image], CropScoreV1],
) -> dict[str, object]:
    """Audit the exact V5-1 pilot without writing or accessing other splits."""
    if not isinstance(dataset_root, Path) or not isinstance(pilot_evidence_path, Path):
        raise TypeError("dataset_root and pilot_evidence_path must be pathlib.Path")
    selection, annotation_by_id, evidence = _load_admitted_rows(dataset_root, pilot_evidence_path)

    sample_records: list[dict[str, object]] = []
    slot_outcomes: Counter[str] = Counter()
    per_class: dict[str, Counter[str]] = {meter: Counter() for meter in EXPECTED_METERS}

    for row in selection:
        sample_id = row["sample_id"]
        meter = row["meter"]
        annotation = annotation_by_id[sample_id]
        image_width = _plain_int(row["image_width"], "image_width")
        image_height = _plain_int(row["image_height"], "image_height")
        image_path = _safe_selected_image(dataset_root, row["image_relpath"])
        image = _open_bound_image(
            image_path,
            expected_sha=row["image_sha256"],
            width=image_width,
            height=image_height,
        )

        x = _plain_int(annotation["x"], "x")
        y = _plain_int(annotation["y"], "y")
        w = _plain_int(annotation["w"], "w")
        h = _plain_int(annotation["h"], "h")
        numerator_box, denominator_box = derive_digit_slots_from_full_meter_bbox_v1(
            x=x,
            y=y,
            w=w,
            h=h,
            image_width=image_width,
            image_height=image_height,
        )
        numerator_crop = crop_digit_to_64_v1(image, numerator_box)
        denominator_crop = crop_digit_to_64_v1(image, denominator_box)
        expected_numerator, expected_denominator = EXPECTED_SLOT_DIGITS[meter]
        numerator_record = _slot_record(
            role="numerator",
            expected_digit=expected_numerator,
            box=numerator_box,
            crop=numerator_crop,
            score=score_crop(numerator_crop),
        )
        denominator_record = _slot_record(
            role="denominator",
            expected_digit=expected_denominator,
            box=denominator_box,
            crop=denominator_crop,
            score=score_crop(denominator_crop),
        )
        for slot in (numerator_record, denominator_record):
            outcome = str(slot["outcome"])
            slot_outcomes[outcome] += 1
            per_class[meter][outcome] += 1
        sample_pass = numerator_record["outcome"] == "CORRECT" and denominator_record["outcome"] == "CORRECT"
        sample_records.append(
            {
                "sample_id": sample_id,
                "family_id": row["family_id"],
                "meter": meter,
                "split": "train",
                "image_relpath": row["image_relpath"],
                "image_sha256": row["image_sha256"],
                "image_width": image_width,
                "image_height": image_height,
                "full_meter_bbox": [x, y, w, h],
                "numerator": numerator_record,
                "denominator": denominator_record,
                "sample_outcome": "CORRECT" if sample_pass else "FAIL",
            }
        )

    correct_samples = sum(1 for record in sample_records if record["sample_outcome"] == "CORRECT")
    correct_slots = slot_outcomes["CORRECT"]
    strict_pass = (
        len(sample_records) == PILOT_TOTAL
        and correct_samples == PILOT_TOTAL
        and correct_slots == PILOT_TOTAL * 2
        and slot_outcomes["NO_HIT"] == 0
        and slot_outcomes["CONFLICT"] == 0
        and slot_outcomes["WRONG_UNIQUE"] == 0
        and slot_outcomes["NONDETERMINISTIC"] == 0
    )
    parent_dataset = evidence.get("dataset")
    assert isinstance(parent_dataset, dict)
    return {
        "schema": "st-omr-meter-v5-1r1-specialist-input-audit-result-v1",
        "stage": "METER V5-1R1",
        "decision": "PASS_SCALE_ANNOTATION" if strict_pass else "HOLD_INPUT_CONTRACT",
        "profile_fingerprint": specialist_input_audit_profile_fingerprint_v1(),
        "parent": {
            "v5_1_head_sha": PARENT_HEAD_SHA,
            "pilot_evidence_schema": PILOT_EVIDENCE_SCHEMA,
            "dataset_name": parent_dataset.get("name"),
            "dataset_fingerprint_sha256": parent_dataset.get("fingerprint_sha256"),
        },
        "surface": {
            "split": "train",
            "sample_count": len(sample_records),
            "slot_count": len(sample_records) * 2,
            "per_meter_samples": dict(Counter(record["meter"] for record in sample_records)),
            "validation_opened": False,
            "final_holdout_opened": False,
        },
        "specialists": {
            "checkpoint_sha256": {str(digit): EXPECTED_CHECKPOINT_SHA[digit] for digit in (2, 3, 4)},
            "thresholds_milli": {str(digit): DIGIT_THRESHOLDS_MILLI[digit] for digit in (2, 3, 4)},
            "replay_count": REPLAY_COUNT,
        },
        "aggregate": {
            "correct_samples": correct_samples,
            "correct_slots": correct_slots,
            "slot_outcomes": dict(sorted(slot_outcomes.items())),
            "per_meter_slot_outcomes": {
                meter: dict(sorted(per_class[meter].items())) for meter in EXPECTED_METERS
            },
        },
        "records": sample_records,
        "safety": {
            "training": False,
            "optimizer_steps": 0,
            "threshold_tuning": False,
            "crop_policy_search": False,
            "d11_executed": False,
            "resolver_connected": False,
            "production_promotion_authorized": False,
        },
    }


def _write_immutable_result(output_dir: Path, result: dict[str, object]) -> tuple[str, Path]:
    if output_dir.exists():
        raise MeterV5_1R1AuditError("audit output directory already exists")
    partial = output_dir.with_name(output_dir.name + ".partial")
    if partial.exists():
        raise MeterV5_1R1AuditError("partial audit output already exists")
    payload = _canonical_json_bytes(result)
    digest = _sha256_bytes(payload)
    partial.mkdir(parents=True, exist_ok=False)
    result_path = partial / "result.json"
    complete_path = partial / "COMPLETE"
    result_path.write_bytes(payload)
    complete_path.write_text(digest + "\n", encoding="ascii")
    os.replace(partial, output_dir)
    return digest, output_dir / "result.json"


def run_specialist_input_audit_from_checkpoints_v1(
    *,
    dataset_root: Path,
    pilot_evidence_path: Path,
    digit2_checkpoint: Path,
    digit3_checkpoint: Path,
    digit4_checkpoint: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Execute the bounded external audit against exact frozen private checkpoints."""
    audited_states = {
        2: audit_digit_checkpoint_v1(digit2_checkpoint, role="digit-2", expected_sha256=DIGIT2_SHA256),
        3: audit_digit_checkpoint_v1(digit3_checkpoint, role="digit-3", expected_sha256=DIGIT3_SHA256),
        4: audit_digit_checkpoint_v1(digit4_checkpoint, role="digit-4", expected_sha256=DIGIT4_SHA256),
    }
    scorer = make_frozen_checkpoint_scorer_v1(audited_states)
    result = audit_pilot_with_scorer_v1(
        dataset_root=dataset_root,
        pilot_evidence_path=pilot_evidence_path,
        score_crop=scorer,
    )
    digest, result_path = _write_immutable_result(output_dir, result)
    return {
        "decision": result["decision"],
        "result_sha256": digest,
        "result_path": str(result_path),
        "output_dir": str(output_dir),
    }


def training_authorized() -> bool:
    return False


def validation_or_final_holdout_access_allowed() -> bool:
    return False


def resolver_connection_allowed() -> bool:
    return False
