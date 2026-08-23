"""Meter V5-2C historical retention audit for adapted 2-AI / 3-AI candidates.

Inference-only. Replays the exact historical M4A validation digit surface from
D10 source pixels, first reproduces frozen baseline confusion counts exactly,
then compares the V5-2B candidates under preregistered retention gates.
"""
from __future__ import annotations

import io
import json
import math
from collections import Counter
from pathlib import Path
from typing import Callable, Final, Mapping

from PIL import Image

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b


RETENTION_SCHEMA: Final[str] = "st-omr-meter-v5-2c-historical-retention-v1"
RETENTION_REPORT_NAME: Final[str] = "v5_2c_historical_retention_v1.json"

M4A_MANIFEST_SHA256: Final[str] = "ebda40dae10f0d6490df2c7728dab5cc2cc6f58b5420b198dfbb441a99ecebb9"
D10_MANIFEST_SHA256: Final[str] = "6927e1bcc5251257a983a306e2f1875c9515f97c6724a8fe9f24382c6ff30db4"

DIGIT2_CANDIDATE_SHA256: Final[str] = "61e4ed5c595d66214ab863f53094998e5cc5167094dc8a9b5934470e3188d4f2"
DIGIT3_CANDIDATE_SHA256: Final[str] = "5d8dd8ea3aed5c2aaa383d2a494e762276afa952f5da6d37fc1dc214900f1c62"

EXPECTED_VALIDATION_LABEL_COUNTS: Final[dict[str, int]] = {
    "2": 186,
    "3": 204,
    "4": 792,
    "NONE": 2190,
}

EXPECTED_FROZEN_COUNTS: Final[dict[str, dict[str, int]]] = {
    "2": {"tp": 185, "fp": 4, "fn": 1, "tn": 3182},
    "3": {"tp": 203, "fp": 0, "fn": 1, "tn": 3168},
    "4": {"tp": 788, "fp": 23, "fn": 4, "tn": 2557},
}

MAX_F1_DROP: Final[float] = 0.005
MAX_RECALL_DROP: Final[float] = 0.005
MIN_CANDIDATE_PRECISION: Final[float] = 0.98
MIN_CANDIDATE_RECALL: Final[float] = 0.98

ProgressCallback = Callable[[int, int, str], None]


def _fail(message: str) -> None:
    raise v52b.MeterV5_2BError(message)


def _read_json(path: Path) -> dict[str, object]:
    return v52b._read_json(path)


def _binary_counts(probabilities, truth, threshold: float) -> dict[str, object]:
    pred = probabilities >= threshold
    positive = truth >= 0.5
    tp = int((pred & positive).sum().item())
    fp = int((pred & ~positive).sum().item())
    fn = int((~pred & positive).sum().item())
    tn = int((~pred & ~positive).sum().item())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def evaluate_retention_gate_v1(
    *,
    frozen_metrics: Mapping[str, Mapping[str, object]],
    candidate_metrics: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Pure preregistered gate; useful for tests and report generation."""
    reasons: list[str] = []
    per_digit: dict[str, dict[str, object]] = {}
    for digit in ("2", "3"):
        baseline = frozen_metrics[digit]
        candidate = candidate_metrics[digit]
        f1_drop = float(baseline["f1"]) - float(candidate["f1"])
        recall_drop = float(baseline["recall"]) - float(candidate["recall"])
        precision = float(candidate["precision"])
        recall = float(candidate["recall"])
        digit_reasons: list[str] = []
        if f1_drop > MAX_F1_DROP + 1e-12:
            digit_reasons.append("F1_DROP_GT_0.005")
        if recall_drop > MAX_RECALL_DROP + 1e-12:
            digit_reasons.append("RECALL_DROP_GT_0.005")
        if precision < MIN_CANDIDATE_PRECISION:
            digit_reasons.append("PRECISION_LT_0.98")
        if recall < MIN_CANDIDATE_RECALL:
            digit_reasons.append("RECALL_LT_0.98")
        reasons.extend(f"{digit}-AI_{reason}" for reason in digit_reasons)
        per_digit[digit] = {
            "f1_drop": f1_drop,
            "recall_drop": recall_drop,
            "candidate_precision": precision,
            "candidate_recall": recall,
            "reasons": digit_reasons,
        }
    return {
        "gate": "PASS" if not reasons else "HOLD",
        "reasons": reasons,
        "per_digit": per_digit,
    }


def _historical_canvas_from_bbox(image: Image.Image, bbox: object) -> Image.Image:
    if not isinstance(bbox, list) or len(bbox) != 4:
        _fail("M4A bbox must be a four-value list")
    values = [float(value) for value in bbox]
    if not all(math.isfinite(value) for value in values):
        _fail("M4A bbox contains non-finite values")
    x0, y0, x1, y1 = values
    width, height = image.size
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
        x0 *= width
        x1 *= width
        y0 *= height
        y1 *= height
    x0 = max(0, min(width - 1, math.floor(x0)))
    y0 = max(0, min(height - 1, math.floor(y0)))
    x1 = max(x0 + 1, min(width, math.ceil(x1)))
    y1 = max(y0 + 1, min(height, math.ceil(y1)))
    crop = image.crop((x0, y0, x1, y1)).convert("L")
    crop.thumbnail((64, 64), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (64, 64), 255)
    canvas.paste(crop, ((64 - crop.width) // 2, (64 - crop.height) // 2))
    return canvas


def _load_manifests(
    *,
    m4a_root: Path,
    d10_root: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    m4a_manifest_path = m4a_root / "dataset-manifest.json"
    d10_manifest_path = d10_root / "manifest.json"
    if v52b._sha_file(m4a_manifest_path) != M4A_MANIFEST_SHA256:
        _fail("M4A manifest SHA mismatch")
    if v52b._sha_file(d10_manifest_path) != D10_MANIFEST_SHA256:
        _fail("D10 manifest SHA mismatch")
    m4a = _read_json(m4a_manifest_path)
    d10 = _read_json(d10_manifest_path)
    records = m4a.get("records")
    d10_records = d10.get("records")
    if not isinstance(records, list) or not isinstance(d10_records, list):
        _fail("historical manifests missing records")
    validation = [row for row in records if isinstance(row, dict) and row.get("split") == "validation"]
    counts = Counter(str(row.get("digit_label")) for row in validation)
    if len(validation) != 3372 or dict(counts) != EXPECTED_VALIDATION_LABEL_COUNTS:
        _fail(f"M4A validation identity changed: total={len(validation)} counts={dict(counts)}")
    d10_meter = {
        str(row.get("record_id")): row
        for row in d10_records
        if isinstance(row, dict) and row.get("kind") == "meter"
    }
    if len(d10_meter) != 11064:
        _fail(f"D10 meter record count changed: {len(d10_meter)}")
    missing = [str(row.get("source_record_id")) for row in validation if str(row.get("source_record_id")) not in d10_meter]
    if missing:
        _fail(f"M4A validation references missing D10 record: {missing[0]}")
    return validation, d10_meter


def _prepare_inputs(
    *,
    validation: list[dict[str, object]],
    d10_meter: Mapping[str, Mapping[str, object]],
    d10_root: Path,
    progress: ProgressCallback | None,
):
    torch, _nn = v52b._import_torch()
    tensors = []
    labels: list[str] = []
    current_path: Path | None = None
    current_image: Image.Image | None = None
    try:
        for index, row in enumerate(validation, start=1):
            source_id = str(row["source_record_id"])
            d10_row = d10_meter[source_id]
            image_relpath = d10_row.get("image_path")
            if not isinstance(image_relpath, str) or not image_relpath:
                _fail(f"D10 meter image path missing: {source_id}")
            image_path = d10_root / image_relpath
            if image_path != current_path:
                if current_image is not None:
                    current_image.close()
                if image_path.is_symlink() or not image_path.is_file():
                    _fail(f"D10 source image missing/non-regular: {image_path}")
                current_image = Image.open(image_path).convert("L")
                current_path = image_path
            assert current_image is not None
            canvas = _historical_canvas_from_bbox(current_image, row.get("bbox"))
            values = torch.tensor(list(canvas.getdata()), dtype=torch.uint8).reshape(64, 64)
            tensors.append(values)
            labels.append(str(row["digit_label"]))
            if progress is not None and (index == 1 or index % 50 == 0 or index == len(validation)):
                progress(index, len(validation), "staging-historical-crops")
    finally:
        if current_image is not None:
            current_image.close()
    images = torch.stack(tensors, dim=0).to(dtype=torch.float32).unsqueeze(1) / 255.0
    return images, labels


def _probabilities(model, images, *, progress: ProgressCallback | None, phase: str):
    torch, _nn = v52b._import_torch()
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), 256):
            batch = images[start:start + 256]
            values = torch.sigmoid(model(batch)).cpu()
            if not bool(torch.isfinite(values).all().item()):
                _fail(f"non-finite probabilities during {phase}")
            if bool(((values < 0.0) | (values > 1.0)).any().item()):
                _fail(f"out-of-range probabilities during {phase}")
            outputs.append(values)
            if progress is not None:
                progress(min(start + len(batch), len(images)), len(images), phase)
    return torch.cat(outputs, dim=0)


def _truth_tensor(labels: list[str], digit: str):
    torch, _nn = v52b._import_torch()
    return torch.tensor([1.0 if label == digit else 0.0 for label in labels], dtype=torch.float32)


def _frozen_model(path: Path, *, digit: str):
    from .runtime_meter_real_checkpoint_audit_v1 import audit_digit_checkpoint_v1
    expected = {"2": v52b.DIGIT2_SHA256, "3": v52b.DIGIT3_SHA256, "4": v52b.DIGIT4_SHA256}[digit]
    audited = audit_digit_checkpoint_v1(path, role=f"digit-{digit}", expected_sha256=expected)
    model = v52b._build_digit_model().cpu()
    model.load_state_dict(dict(audited.model_state), strict=True)
    model.eval()
    return model


def run_historical_retention_v1(
    v5_data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    digit4_frozen: str | Path,
    digit2_candidate: str | Path,
    digit3_candidate: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    root = Path(v5_data_root)
    m4a = Path(m4a_root)
    d10 = Path(d10_root)
    frozen_paths = {
        "2": Path(digit2_frozen),
        "3": Path(digit3_frozen),
        "4": Path(digit4_frozen),
    }
    candidate_paths = {"2": Path(digit2_candidate), "3": Path(digit3_candidate)}

    manifest_path, _slot_rows, _slot_audit = v52b.verify_slot_manifest_v1(root)
    training_report = _read_json(root / v51.ANNOTATIONS_DIR / v52b.TRAINING_REPORT_NAME)
    candidates_report = training_report.get("candidates")
    if not isinstance(candidates_report, Mapping):
        _fail("V5-2B training report missing candidates")
    expected_candidate_sha = {"2": DIGIT2_CANDIDATE_SHA256, "3": DIGIT3_CANDIDATE_SHA256}
    for digit in ("2", "3"):
        actual = v52b._sha_file(candidate_paths[digit])
        if actual != expected_candidate_sha[digit]:
            _fail(f"{digit}-AI candidate SHA changed")
        if candidates_report.get(digit, {}).get("candidate_sha256") != actual:
            _fail(f"{digit}-AI candidate differs from V5-2B training report")
    for digit in ("2", "3", "4"):
        expected = {"2": v52b.DIGIT2_SHA256, "3": v52b.DIGIT3_SHA256, "4": v52b.DIGIT4_SHA256}[digit]
        if v52b._sha_file(frozen_paths[digit]) != expected:
            _fail(f"frozen {digit}-AI SHA changed")

    validation, d10_meter = _load_manifests(m4a_root=m4a, d10_root=d10)
    images, labels = _prepare_inputs(
        validation=validation,
        d10_meter=d10_meter,
        d10_root=d10,
        progress=progress,
    )

    frozen_metrics: dict[str, dict[str, object]] = {}
    for digit in ("2", "3", "4"):
        probs = _probabilities(_frozen_model(frozen_paths[digit], digit=digit), images, progress=progress, phase=f"frozen-{digit}-AI")
        metrics = _binary_counts(probs, _truth_tensor(labels, digit), v52b.FROZEN_THRESHOLDS[digit])
        frozen_metrics[digit] = metrics
        expected = EXPECTED_FROZEN_COUNTS[digit]
        if any(metrics[key] != expected[key] for key in ("tp", "fp", "fn", "tn")):
            _fail(f"historical pixel-path reproduction failed for {digit}-AI: {metrics}")

    manifest_sha = v52b._sha_file(manifest_path)
    candidate_metrics: dict[str, dict[str, object]] = {}
    for digit in ("2", "3"):
        model = v52b._load_candidate_model(candidate_paths[digit], digit=digit, manifest_sha256=manifest_sha)
        probs = _probabilities(model, images, progress=progress, phase=f"candidate-{digit}-AI")
        candidate_metrics[digit] = _binary_counts(probs, _truth_tensor(labels, digit), v52b.FROZEN_THRESHOLDS[digit])

    gate = evaluate_retention_gate_v1(frozen_metrics=frozen_metrics, candidate_metrics=candidate_metrics)
    report = {
        "schema": RETENTION_SCHEMA,
        "m4a_manifest_sha256": M4A_MANIFEST_SHA256,
        "d10_manifest_sha256": D10_MANIFEST_SHA256,
        "validation_record_count": 3372,
        "validation_label_counts": dict(EXPECTED_VALIDATION_LABEL_COUNTS),
        "thresholds": {"2": v52b.FROZEN_THRESHOLDS["2"], "3": v52b.FROZEN_THRESHOLDS["3"], "4": v52b.FROZEN_THRESHOLDS["4"]},
        "frozen_checkpoint_sha256": {"2": v52b.DIGIT2_SHA256, "3": v52b.DIGIT3_SHA256, "4": v52b.DIGIT4_SHA256},
        "candidate_sha256": dict(expected_candidate_sha),
        "historical_pixel_path_reproduced": True,
        "frozen_metrics": frozen_metrics,
        "candidate_metrics": candidate_metrics,
        "retention_limits": {
            "max_f1_drop": MAX_F1_DROP,
            "max_recall_drop": MAX_RECALL_DROP,
            "min_candidate_precision": MIN_CANDIDATE_PRECISION,
            "min_candidate_recall": MIN_CANDIDATE_RECALL,
        },
        "gate": gate["gate"],
        "reasons": gate["reasons"],
        "per_digit_retention": gate["per_digit"],
        "validation_bbox_stage_authorized": gate["gate"] == "PASS",
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "optimizer_steps": 0,
        "threshold_tuning": False,
        "resolver_wiring_authorized": False,
        "production_promotion_authorized": False,
    }
    v51._atomic_write_json(root / v51.ANNOTATIONS_DIR / RETENTION_REPORT_NAME, report)
    return report


def validation_opened_by_this_module() -> bool:
    return False


def final_holdout_locked() -> bool:
    return True


def production_promotion_allowed() -> bool:
    return False
