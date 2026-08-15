"""Stage 7-D13 compact symbol specialists, objective, decoder and metrics.

This module implements the frozen D13-4 learned surface without running an
optimizer at import/build time.  NoteHeadSet, RestSet and AccidentalSet share
one compact architecture implementation but always use independent model state.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Final, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .stage7d13_symbol_training_contract import (
    ACCEPTANCE,
    FROZEN_D13_CONFIG,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    MAX_PARAMETERS_PER_SPECIALIST,
    OUTPUT_STRIDE,
    SPECIALIST_CLASSES,
    positive_class_weights,
    stage7d13_contract_fingerprint,
)
from .stage7d13_verified_surface import (
    D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
    D13_DERIVATIVE_BUILD_ID,
    D13_DERIVATIVE_MANIFEST_SHA256,
)
from .training_model import (
    TrainingRuntimeError,
    assert_finite_tensor,
    assert_model_finite,
    count_trainable_parameters,
    set_deterministic_cpu,
)


STAGE7D13_MODEL_VERSION: Final[str] = "stage7d13-compact-center-detector-v1"
STAGE7D13_METRIC_VERSION: Final[str] = "stage7d13-symbol-metrics-v1"
HEATMAP_FOCAL_GAMMA: Final[float] = 2.0
_HEATMAP_EPS: Final[float] = 1e-6
_GRID_HEIGHT: Final[int] = INPUT_HEIGHT // OUTPUT_STRIDE
_GRID_WIDTH: Final[int] = INPUT_WIDTH // OUTPUT_STRIDE


class Stage7D13ModelError(TrainingRuntimeError):
    """Raised when D13 model, target, decoder or metric state fails closed."""


def _fail(message: str) -> None:
    raise Stage7D13ModelError(message)


def _canonical_json(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Stage7D13ModelError("D13 model payload is not canonical JSON") from exc


def _finite(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def _specialist(name: str) -> tuple[str, ...]:
    classes = SPECIALIST_CLASSES.get(name)
    if classes is None:
        _fail("unknown D13 specialist")
    return classes


class _PredictionHead(nn.Module):
    def __init__(self, channels: int, outputs: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, outputs, kernel_size=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class SymbolCenterDetector(nn.Module):
    """Small stride-4 fully convolutional detector for one D13 specialist."""

    def __init__(self, specialist: str) -> None:
        super().__init__()
        classes = _specialist(specialist)
        self.specialist = specialist
        self.classes = classes
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.heatmap_head = _PredictionHead(64, len(classes))
        self.bbox_size_head = _PredictionHead(64, 2)
        self.center_offset_head = _PredictionHead(64, 2)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if not isinstance(images, torch.Tensor):
            _fail("D13 detector images must be a torch tensor")
        if images.dtype != torch.float32 or images.ndim != 4:
            _fail("D13 detector images must be float32 [B,1,H,W]")
        if tuple(images.shape[1:]) != (1, INPUT_HEIGHT, INPUT_WIDTH):
            _fail("D13 detector image shape differs from frozen input")
        assert_finite_tensor("D13 detector input", images)
        features = self.encoder(images)
        expected = (images.shape[0], 64, _GRID_HEIGHT, _GRID_WIDTH)
        if tuple(features.shape) != expected:
            _fail("D13 detector feature geometry differs from stride-4 contract")
        heatmap_logits = self.heatmap_head(features)
        bbox_size = F.softplus(self.bbox_size_head(features)) + 1e-4
        center_offset = torch.sigmoid(self.center_offset_head(features))
        for name, value in (
            ("heatmap_logits", heatmap_logits),
            ("bbox_size", bbox_size),
            ("center_offset", center_offset),
        ):
            assert_finite_tensor(f"D13 {name}", value)
        return {
            "heatmap_logits": heatmap_logits,
            "bbox_size": bbox_size,
            "center_offset": center_offset,
        }


def model_profile_payload(specialist: str) -> dict[str, object]:
    classes = _specialist(specialist)
    return {
        "version": STAGE7D13_MODEL_VERSION,
        "specialist": specialist,
        "classes": classes,
        "input": [1, INPUT_HEIGHT, INPUT_WIDTH],
        "output_stride": OUTPUT_STRIDE,
        "encoder": [
            [1, 24, 3, 2],
            [24, 48, 3, 2],
            [48, 64, 3, 1],
            [64, 64, 3, 1],
        ],
        "heads": {
            "heatmap": len(classes),
            "bbox_size": 2,
            "center_offset": 2,
        },
        "bbox_activation": "softplus_plus_1e-4",
        "offset_activation": "sigmoid",
        "heatmap_focal_gamma": HEATMAP_FOCAL_GAMMA,
        "contract_fingerprint": stage7d13_contract_fingerprint(),
        "accepted_derivative": {
            "build_id": D13_DERIVATIVE_BUILD_ID,
            "manifest_sha256": D13_DERIVATIVE_MANIFEST_SHA256,
            "artifact_binding_sha256": D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
        },
    }


def model_profile_fingerprint(specialist: str) -> str:
    return sha256(_canonical_json(model_profile_payload(specialist))).hexdigest()


def build_symbol_model(specialist: str, *, seed: int | None = None) -> SymbolCenterDetector:
    _specialist(specialist)
    chosen_seed = FROZEN_D13_CONFIG.master_seed if seed is None else seed
    if not isinstance(chosen_seed, int) or isinstance(chosen_seed, bool):
        raise TypeError("seed must be an integer")
    set_deterministic_cpu(chosen_seed)
    model = SymbolCenterDetector(specialist).cpu()
    count = count_trainable_parameters(model)
    if not 0 < count <= MAX_PARAMETERS_PER_SPECIALIST:
        _fail(f"{specialist} parameter count {count} exceeds D13 cap")
    assert_model_finite(model)
    return model


@dataclass(frozen=True, slots=True)
class DetectorTargets:
    heatmap: torch.Tensor
    bbox_size: torch.Tensor
    center_offset: torch.Tensor
    positive_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class Detection:
    class_name: str
    score: float
    center_x: float
    center_y: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class GroundTruth:
    class_name: str
    center_x: float
    center_y: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SpecialistMetrics:
    class_aware_center_f1_4px: float
    class_aware_bbox_f1_iou50: float
    macro_class_f1: float


def _target_record(specialist: str, row: Mapping[str, object]) -> tuple[int, float, float, float, float, float, float]:
    classes = _specialist(specialist)
    class_name = row.get("class")
    if class_name not in classes:
        _fail(f"{specialist} target class is outside frozen classes")
    assert isinstance(class_name, str)
    class_index = classes.index(class_name)
    center = row.get("center")
    bbox = row.get("bbox")
    if not isinstance(center, Mapping) or set(center) != {"x", "y"}:
        _fail(f"{specialist} target center must be canonical")
    if not isinstance(bbox, Mapping) or set(bbox) != {"x_min", "y_min", "x_max", "y_max"}:
        _fail(f"{specialist} target bbox must be canonical")
    cx = _finite("target.center.x", center.get("x"))
    cy = _finite("target.center.y", center.get("y"))
    x0 = _finite("target.bbox.x_min", bbox.get("x_min"))
    y0 = _finite("target.bbox.y_min", bbox.get("y_min"))
    x1 = _finite("target.bbox.x_max", bbox.get("x_max"))
    y1 = _finite("target.bbox.y_max", bbox.get("y_max"))
    if not 0.0 <= cx < INPUT_WIDTH or not 0.0 <= cy < INPUT_HEIGHT:
        _fail("D13 target center leaves fixed input canvas")
    if not 0.0 <= x0 < x1 <= INPUT_WIDTH or not 0.0 <= y0 < y1 <= INPUT_HEIGHT:
        _fail("D13 target bbox leaves fixed input canvas")
    if not x0 <= cx <= x1 or not y0 <= cy <= y1:
        _fail("D13 target center lies outside target bbox")
    return class_index, cx, cy, x0, y0, x1, y1


def encode_detector_targets(
    specialist: str,
    batch_targets: Sequence[Sequence[Mapping[str, object]]],
) -> DetectorTargets:
    """Encode exact center cells; fail closed on class-agnostic regression collisions."""
    classes = _specialist(specialist)
    if not isinstance(batch_targets, Sequence) or isinstance(batch_targets, (str, bytes, bytearray)):
        raise TypeError("batch_targets must be a sequence")
    batch = len(batch_targets)
    if batch < 1:
        _fail("D13 target batch must not be empty")
    heatmap = torch.zeros((batch, len(classes), _GRID_HEIGHT, _GRID_WIDTH), dtype=torch.float32)
    bbox_size = torch.zeros((batch, 2, _GRID_HEIGHT, _GRID_WIDTH), dtype=torch.float32)
    center_offset = torch.zeros((batch, 2, _GRID_HEIGHT, _GRID_WIDTH), dtype=torch.float32)
    positive_mask = torch.zeros((batch, 1, _GRID_HEIGHT, _GRID_WIDTH), dtype=torch.bool)

    for batch_index, rows in enumerate(batch_targets):
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            _fail("D13 per-image targets must be a sequence")
        occupied: set[tuple[int, int]] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                _fail("D13 target row must be a mapping")
            class_index, cx, cy, x0, y0, x1, y1 = _target_record(specialist, row)
            gx = cx / OUTPUT_STRIDE
            gy = cy / OUTPUT_STRIDE
            cell_x = int(math.floor(gx))
            cell_y = int(math.floor(gy))
            if not 0 <= cell_x < _GRID_WIDTH or not 0 <= cell_y < _GRID_HEIGHT:
                _fail("D13 target center maps outside output grid")
            cell = (cell_y, cell_x)
            if cell in occupied:
                _fail(
                    f"{specialist} has two targets in one class-agnostic regression cell"
                )
            occupied.add(cell)
            heatmap[batch_index, class_index, cell_y, cell_x] = 1.0
            bbox_size[batch_index, 0, cell_y, cell_x] = x1 - x0
            bbox_size[batch_index, 1, cell_y, cell_x] = y1 - y0
            center_offset[batch_index, 0, cell_y, cell_x] = gx - cell_x
            center_offset[batch_index, 1, cell_y, cell_x] = gy - cell_y
            positive_mask[batch_index, 0, cell_y, cell_x] = True

    return DetectorTargets(
        heatmap=heatmap,
        bbox_size=bbox_size,
        center_offset=center_offset,
        positive_mask=positive_mask,
    )


def detector_loss(
    specialist: str,
    outputs: Mapping[str, torch.Tensor],
    targets: DetectorTargets,
) -> torch.Tensor:
    classes = _specialist(specialist)
    logits = outputs.get("heatmap_logits")
    sizes = outputs.get("bbox_size")
    offsets = outputs.get("center_offset")
    if not all(isinstance(value, torch.Tensor) for value in (logits, sizes, offsets)):
        _fail("D13 detector outputs are incomplete")
    assert isinstance(logits, torch.Tensor)
    assert isinstance(sizes, torch.Tensor)
    assert isinstance(offsets, torch.Tensor)
    expected_heatmap = (targets.heatmap.shape[0], len(classes), _GRID_HEIGHT, _GRID_WIDTH)
    expected_regression = (targets.heatmap.shape[0], 2, _GRID_HEIGHT, _GRID_WIDTH)
    if tuple(logits.shape) != expected_heatmap or tuple(sizes.shape) != expected_regression or tuple(offsets.shape) != expected_regression:
        _fail("D13 detector output/target shape mismatch")

    heatmap = targets.heatmap.to(dtype=torch.float32, device=logits.device)
    target_sizes = targets.bbox_size.to(dtype=torch.float32, device=logits.device)
    target_offsets = targets.center_offset.to(dtype=torch.float32, device=logits.device)
    mask = targets.positive_mask.to(device=logits.device)
    probs = torch.sigmoid(logits).clamp(_HEATMAP_EPS, 1.0 - _HEATMAP_EPS)
    positive = heatmap.eq(1.0)
    negative = ~positive
    class_weights = positive_class_weights(specialist)
    weight_tensor = torch.tensor(
        [class_weights[name] for name in classes],
        dtype=torch.float32,
        device=logits.device,
    ).view(1, len(classes), 1, 1)
    positive_loss = -weight_tensor * (1.0 - probs).pow(HEATMAP_FOCAL_GAMMA) * torch.log(probs)
    negative_loss = -probs.pow(HEATMAP_FOCAL_GAMMA) * torch.log(1.0 - probs)
    positive_count = positive.sum().clamp_min(1).to(dtype=torch.float32)
    heatmap_loss = (
        positive_loss.masked_select(positive).sum()
        + negative_loss.masked_select(negative).sum()
    ) / positive_count

    reg_mask = mask.expand(-1, 2, -1, -1)
    if bool(mask.any()):
        size_loss = F.smooth_l1_loss(
            sizes.masked_select(reg_mask),
            target_sizes.masked_select(reg_mask),
            reduction="mean",
        )
        offset_loss = F.smooth_l1_loss(
            offsets.masked_select(reg_mask),
            target_offsets.masked_select(reg_mask),
            reduction="mean",
        )
    else:
        size_loss = sizes.sum() * 0.0
        offset_loss = offsets.sum() * 0.0
    total = heatmap_loss + size_loss + offset_loss
    assert_finite_tensor("D13 detector loss", total)
    return total


def _local_max_scores(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    pooled = F.max_pool2d(probabilities, kernel_size=3, stride=1, padding=1)
    keep = probabilities.eq(pooled) & probabilities.ge(0.25)
    return probabilities * keep.to(dtype=probabilities.dtype)


def decode_detections(
    specialist: str,
    outputs: Mapping[str, torch.Tensor],
) -> list[list[Detection]]:
    classes = _specialist(specialist)
    logits = outputs.get("heatmap_logits")
    sizes = outputs.get("bbox_size")
    offsets = outputs.get("center_offset")
    if not all(isinstance(value, torch.Tensor) for value in (logits, sizes, offsets)):
        _fail("D13 decode outputs are incomplete")
    assert isinstance(logits, torch.Tensor)
    assert isinstance(sizes, torch.Tensor)
    assert isinstance(offsets, torch.Tensor)
    if logits.ndim != 4 or logits.shape[1:] != (len(classes), _GRID_HEIGHT, _GRID_WIDTH):
        _fail("D13 decode heatmap shape mismatch")
    if tuple(sizes.shape) != (logits.shape[0], 2, _GRID_HEIGHT, _GRID_WIDTH):
        _fail("D13 decode bbox shape mismatch")
    if tuple(offsets.shape) != tuple(sizes.shape):
        _fail("D13 decode offset shape mismatch")
    for name, tensor in (("logits", logits), ("sizes", sizes), ("offsets", offsets)):
        assert_finite_tensor(f"D13 decode {name}", tensor)

    scores = _local_max_scores(logits).detach().cpu()
    sizes_cpu = sizes.detach().cpu()
    offsets_cpu = offsets.detach().cpu()
    decoded: list[list[Detection]] = []
    for batch_index in range(logits.shape[0]):
        candidates: list[tuple[float, int, int, int]] = []
        for class_index in range(len(classes)):
            ys, xs = torch.nonzero(scores[batch_index, class_index] > 0.0, as_tuple=True)
            for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
                candidates.append((float(scores[batch_index, class_index, y, x]), class_index, y, x))
        candidates.sort(key=lambda row: (-row[0], row[2], row[3], row[1]))
        rows: list[Detection] = []
        for score, class_index, y, x in candidates[:256]:
            offset_x = float(offsets_cpu[batch_index, 0, y, x])
            offset_y = float(offsets_cpu[batch_index, 1, y, x])
            width = float(sizes_cpu[batch_index, 0, y, x])
            height = float(sizes_cpu[batch_index, 1, y, x])
            if not all(math.isfinite(value) for value in (score, offset_x, offset_y, width, height)):
                _fail("D13 decoded prediction is non-finite")
            if width <= 0.0 or height <= 0.0:
                _fail("D13 decoded bbox size must be positive")
            center_x = (x + offset_x) * OUTPUT_STRIDE
            center_y = (y + offset_y) * OUTPUT_STRIDE
            bbox = (
                center_x - width / 2.0,
                center_y - height / 2.0,
                center_x + width / 2.0,
                center_y + height / 2.0,
            )
            rows.append(
                Detection(
                    class_name=classes[class_index],
                    score=score,
                    center_x=center_x,
                    center_y=center_y,
                    bbox=bbox,
                )
            )
        decoded.append(rows)
    return decoded


def ground_truth_rows(
    specialist: str,
    rows: Sequence[Mapping[str, object]],
) -> list[GroundTruth]:
    result: list[GroundTruth] = []
    for row in rows:
        _class_index, cx, cy, x0, y0, x1, y1 = _target_record(specialist, row)
        class_name = row.get("class")
        assert isinstance(class_name, str)
        result.append(GroundTruth(class_name, cx, cy, (x0, y0, x1, y1)))
    return result


def _center_distance(prediction: Detection, target: GroundTruth) -> float:
    return math.hypot(prediction.center_x - target.center_x, prediction.center_y - target.center_y)


def _iou(prediction: Detection, target: GroundTruth) -> float:
    ax0, ay0, ax1, ay1 = prediction.bbox
    bx0, by0, bx1, by1 = target.bbox
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return (2.0 * tp / denominator) if denominator else 0.0


def compute_specialist_metrics(
    specialist: str,
    examples: Sequence[tuple[Sequence[Detection], Sequence[GroundTruth]]],
) -> SpecialistMetrics:
    classes = _specialist(specialist)
    center_tp = center_fp = center_fn = 0
    bbox_tp = bbox_fp = bbox_fn = 0
    class_counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in classes}

    for predictions_raw, targets_raw in examples:
        predictions = list(predictions_raw)
        targets = list(targets_raw)
        predictions.sort(key=lambda row: (-row.score, row.center_y, row.center_x, row.class_name))

        unmatched = set(range(len(targets)))
        matched = 0
        for prediction in predictions:
            options = [
                ( _center_distance(prediction, targets[index]), index )
                for index in unmatched
                if targets[index].class_name == prediction.class_name
                and _center_distance(prediction, targets[index]) <= 4.0
            ]
            if options:
                _distance, chosen = min(options, key=lambda row: (row[0], row[1]))
                unmatched.remove(chosen)
                matched += 1
        center_tp += matched
        center_fp += len(predictions) - matched
        center_fn += len(targets) - matched

        unmatched = set(range(len(targets)))
        matched = 0
        for prediction in predictions:
            options = [
                (_iou(prediction, targets[index]), index)
                for index in unmatched
                if targets[index].class_name == prediction.class_name
                and _iou(prediction, targets[index]) >= 0.50
            ]
            if options:
                _overlap, chosen = max(options, key=lambda row: (row[0], -row[1]))
                unmatched.remove(chosen)
                matched += 1
        bbox_tp += matched
        bbox_fp += len(predictions) - matched
        bbox_fn += len(targets) - matched

        unmatched = set(range(len(targets)))
        matched_prediction_indices: set[int] = set()
        for pred_index, prediction in enumerate(predictions):
            options = [
                (_center_distance(prediction, targets[index]), index)
                for index in unmatched
                if _center_distance(prediction, targets[index]) <= 4.0
            ]
            if not options:
                continue
            _distance, chosen = min(options, key=lambda row: (row[0], row[1]))
            unmatched.remove(chosen)
            matched_prediction_indices.add(pred_index)
            target = targets[chosen]
            if prediction.class_name == target.class_name:
                class_counts[target.class_name]["tp"] += 1
            else:
                class_counts[prediction.class_name]["fp"] += 1
                class_counts[target.class_name]["fn"] += 1
        for pred_index, prediction in enumerate(predictions):
            if pred_index not in matched_prediction_indices:
                class_counts[prediction.class_name]["fp"] += 1
        for target_index in unmatched:
            class_counts[targets[target_index].class_name]["fn"] += 1

    macro = sum(
        _f1(values["tp"], values["fp"], values["fn"])
        for values in class_counts.values()
    ) / len(classes)
    result = SpecialistMetrics(
        class_aware_center_f1_4px=_f1(center_tp, center_fp, center_fn),
        class_aware_bbox_f1_iou50=_f1(bbox_tp, bbox_fp, bbox_fn),
        macro_class_f1=macro,
    )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (
        result.class_aware_center_f1_4px,
        result.class_aware_bbox_f1_iou50,
        result.macro_class_f1,
    )):
        _fail("D13 metric result is outside [0,1]")
    return result


def acceptance_passed(specialist: str, metrics: SpecialistMetrics) -> bool:
    threshold = ACCEPTANCE.get(specialist)
    if threshold is None:
        _fail("unknown D13 specialist acceptance gate")
    return (
        metrics.class_aware_center_f1_4px
        >= threshold.class_aware_center_f1_4px_milli / 1000.0
        and metrics.class_aware_bbox_f1_iou50
        >= threshold.class_aware_bbox_f1_iou50_milli / 1000.0
        and metrics.macro_class_f1
        >= threshold.macro_class_f1_milli / 1000.0
    )
