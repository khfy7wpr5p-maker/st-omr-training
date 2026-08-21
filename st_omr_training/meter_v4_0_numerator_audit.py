"""Bounded Meter V4-0 numerator-only representation audit.

V4-0 uses only positive Teacher Gold TRAIN records and derives a deterministic
numerator crop from the accepted full Meter bbox.  A tiny from-scratch 3-class
specialist is evaluated with family-disjoint 3-fold out-of-fold predictions.

D10, Teacher Gold adaptation-validation evaluation, sealed TEST, runtime,
Resolver, checkpoint replacement, and production promotion remain closed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Final


METER_V4_0_NUMERATOR_AUDIT: Final[str] = "meter-v4-0-numerator-representation-audit-v1"
NUMERATOR_CLASSES: Final[tuple[str, ...]] = ("2", "3", "4")
METER_TO_NUMERATOR: Final[dict[str, str]] = {"2/4": "2", "3/4": "3", "4/4": "4"}
ROI_WIDTH: Final[int] = 256
ROI_HEIGHT: Final[int] = 192


class MeterV4_0AuditError(RuntimeError):
    """Raised when the bounded V4-0 audit contract is violated."""


def _fail(message: str) -> None:
    raise MeterV4_0AuditError(message)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _hex64(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        _fail(f"{name} must be canonical lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class NumeratorAuditConfigV4_0:
    output_size: int = 64
    folds: int = 3
    master_seed: int = 840_001
    epochs: int = 120
    learning_rate_micros: int = 3_000
    weight_decay_micros: int = 100
    grad_clip_milli: int = 2_000
    horizontal_padding_milli: int = 150
    vertical_padding_milli: int = 50
    numerator_fraction_milli: int = 500
    shift_pixels: int = 2
    max_trainable_parameters: int = 50_000

    def __post_init__(self) -> None:
        bounds = {
            "output_size": (self.output_size, 32, 128),
            "folds": (self.folds, 2, 9),
            "master_seed": (self.master_seed, 0, 2**63 - 1),
            "epochs": (self.epochs, 1, 500),
            "learning_rate_micros": (self.learning_rate_micros, 1, 100_000),
            "weight_decay_micros": (self.weight_decay_micros, 0, 100_000),
            "grad_clip_milli": (self.grad_clip_milli, 1, 100_000),
            "horizontal_padding_milli": (self.horizontal_padding_milli, 0, 500),
            "vertical_padding_milli": (self.vertical_padding_milli, 0, 250),
            "numerator_fraction_milli": (self.numerator_fraction_milli, 350, 650),
            "shift_pixels": (self.shift_pixels, 0, 4),
            "max_trainable_parameters": (self.max_trainable_parameters, 1, 1_000_000),
        }
        for name, (value, low, high) in bounds.items():
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ValueError(f"{name} is outside Meter V4-0 bounds")
        frozen = {
            "output_size": 64,
            "folds": 3,
            "master_seed": 840_001,
            "epochs": 120,
            "learning_rate_micros": 3_000,
            "weight_decay_micros": 100,
            "grad_clip_milli": 2_000,
            "horizontal_padding_milli": 150,
            "vertical_padding_milli": 50,
            "numerator_fraction_milli": 500,
            "shift_pixels": 2,
            "max_trainable_parameters": 50_000,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} is frozen to {expected!r}")


FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0: Final[NumeratorAuditConfigV4_0] = NumeratorAuditConfigV4_0()


@dataclass(frozen=True, slots=True)
class AuditRecordIdentityV4_0:
    record_id: str
    family_id: str
    meter_class: str

    def __post_init__(self) -> None:
        _hex64("record_id", self.record_id)
        if not isinstance(self.family_id, str) or not self.family_id or len(self.family_id) > 256:
            raise ValueError("family_id must be bounded non-empty string")
        if self.meter_class not in METER_TO_NUMERATOR:
            raise ValueError("meter_class must be 2/4, 3/4, or 4/4")

    @property
    def numerator_class(self) -> str:
        return METER_TO_NUMERATOR[self.meter_class]

    @property
    def class_index(self) -> int:
        return NUMERATOR_CLASSES.index(self.numerator_class)


@dataclass(frozen=True, slots=True)
class FoldAssignmentV4_0:
    record_id: str
    family_id: str
    meter_class: str
    fold: int


@dataclass(frozen=True, slots=True)
class ClassificationSummaryV4_0:
    record_count: int
    accuracy: float
    macro_f1: float
    per_class_recall: dict[str, float]
    confusion: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class AuditDecisionV4_0:
    decision: str
    strong_signal: bool
    reasons: tuple[str, ...]


def _finite(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def numerator_crop_bounds_v4_0(
    meter_bbox: Mapping[str, object],
    config: NumeratorAuditConfigV4_0 = FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0,
) -> tuple[int, int, int, int]:
    """Return deterministic pixel bounds for the numerator half of a full Meter bbox."""
    if config != FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0:
        _fail("V4-0 requires the frozen audit configuration")
    if not isinstance(meter_bbox, Mapping) or set(meter_bbox) != {"x_min", "y_min", "x_max", "y_max"}:
        _fail("meter_bbox must contain exactly x_min/y_min/x_max/y_max")
    x_min = _finite("meter_bbox.x_min", meter_bbox.get("x_min"))
    y_min = _finite("meter_bbox.y_min", meter_bbox.get("y_min"))
    x_max = _finite("meter_bbox.x_max", meter_bbox.get("x_max"))
    y_max = _finite("meter_bbox.y_max", meter_bbox.get("y_max"))
    if not (0 <= x_min < x_max <= ROI_WIDTH and 0 <= y_min < y_max <= ROI_HEIGHT):
        _fail("meter_bbox is outside the canonical 256x192 Teacher Gold ROI")

    width = x_max - x_min
    height = y_max - y_min
    x_pad = width * (config.horizontal_padding_milli / 1000.0)
    y_pad = height * (config.vertical_padding_milli / 1000.0)
    numerator_bottom = y_min + height * (config.numerator_fraction_milli / 1000.0)

    left = max(0, int(math.floor(x_min - x_pad)))
    top = max(0, int(math.floor(y_min - y_pad)))
    right = min(ROI_WIDTH, int(math.ceil(x_max + x_pad)))
    bottom = min(ROI_HEIGHT, int(math.ceil(numerator_bottom + y_pad)))
    if not (0 <= left < right <= ROI_WIDTH and 0 <= top < bottom <= ROI_HEIGHT):
        _fail("derived numerator crop is empty or outside the Teacher Gold ROI")
    return left, top, right, bottom


def numerator_crop_tensor_v4_0(
    roi_tensor,
    meter_bbox: Mapping[str, object],
    config: NumeratorAuditConfigV4_0 = FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0,
):
    """Extract and aspect-fit a verified inverted-ink [1,192,256] tensor to [1,64,64]."""
    try:
        import torch
        from torch.nn import functional as F
    except ModuleNotFoundError as exc:
        raise MeterV4_0AuditError("torch is required only for V4-0 execution") from exc

    if config != FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0:
        _fail("V4-0 requires the frozen audit configuration")
    if not isinstance(roi_tensor, torch.Tensor) or roi_tensor.dtype != torch.float32:
        _fail("Teacher Gold ROI tensor must be float32")
    if tuple(roi_tensor.shape) != (1, ROI_HEIGHT, ROI_WIDTH):
        _fail("Teacher Gold ROI tensor must be exact [1,192,256]")
    if not bool(torch.isfinite(roi_tensor).all()) or bool((roi_tensor < 0).any()) or bool((roi_tensor > 1).any()):
        _fail("Teacher Gold ROI tensor must be finite ink values in [0,1]")

    left, top, right, bottom = numerator_crop_bounds_v4_0(meter_bbox, config)
    cropped = roi_tensor[:, top:bottom, left:right].unsqueeze(0)
    crop_h = bottom - top
    crop_w = right - left
    scale = min(config.output_size / crop_w, config.output_size / crop_h)
    resized_w = max(1, min(config.output_size, int(round(crop_w * scale))))
    resized_h = max(1, min(config.output_size, int(round(crop_h * scale))))
    resized = F.interpolate(cropped, size=(resized_h, resized_w), mode="bilinear", align_corners=False)[0]
    canvas = torch.zeros((1, config.output_size, config.output_size), dtype=torch.float32)
    x0 = (config.output_size - resized_w) // 2
    y0 = (config.output_size - resized_h) // 2
    canvas[:, y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    if not bool(torch.isfinite(canvas).all()) or bool((canvas < 0).any()) or bool((canvas > 1).any()):
        _fail("normalized numerator crop left the finite [0,1] boundary")
    return canvas.contiguous()


def fold_plan_v4_0(
    identities: Sequence[AuditRecordIdentityV4_0],
    config: NumeratorAuditConfigV4_0 = FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0,
) -> tuple[FoldAssignmentV4_0, ...]:
    """Create the frozen class-balanced family-disjoint 3-fold plan."""
    if config != FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0:
        _fail("V4-0 requires the frozen audit configuration")
    if len(identities) != 27:
        _fail("V4-0 requires exactly 27 positive Teacher Gold TRAIN records")
    if len({item.record_id for item in identities}) != 27:
        _fail("V4-0 record ids must be unique")
    if len({item.family_id for item in identities}) != 27:
        _fail("V4-0 requires one positive record per family")
    counts = Counter(item.meter_class for item in identities)
    if counts != Counter({"2/4": 9, "3/4": 9, "4/4": 9}):
        _fail("V4-0 positive TRAIN classes must be exactly balanced 9/9/9")

    assignments: list[FoldAssignmentV4_0] = []
    for meter_class in ("2/4", "3/4", "4/4"):
        class_items = [item for item in identities if item.meter_class == meter_class]
        ranked = sorted(
            class_items,
            key=lambda item: sha256(
                _canonical_json(
                    {
                        "version": METER_V4_0_NUMERATOR_AUDIT,
                        "fold_policy": "class-balanced-family-hash-rank-mod3-v1",
                        "meter_class": meter_class,
                        "family_id": item.family_id,
                    }
                )
            ).hexdigest(),
        )
        for rank, item in enumerate(ranked):
            assignments.append(
                FoldAssignmentV4_0(
                    record_id=item.record_id,
                    family_id=item.family_id,
                    meter_class=item.meter_class,
                    fold=rank % config.folds,
                )
            )

    assignments.sort(key=lambda item: (item.fold, item.meter_class, item.family_id, item.record_id))
    fold_class_counts = Counter((item.fold, item.meter_class) for item in assignments)
    expected = Counter((fold, meter_class) for fold in range(3) for meter_class in ("2/4", "3/4", "4/4") for _ in range(3))
    if fold_class_counts != expected:
        _fail("V4-0 fold plan must hold out exactly three families per class per fold")
    return tuple(assignments)


def _shift_tensor(image, dx: int, dy: int):
    import torch

    if tuple(image.shape) != (1, 64, 64):
        _fail("V4-0 shift input must be [1,64,64]")
    result = torch.zeros_like(image)
    src_x0 = max(0, -dx)
    src_x1 = min(64, 64 - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(64, 64 + dx)
    src_y0 = max(0, -dy)
    src_y1 = min(64, 64 - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(64, 64 + dy)
    if src_x0 < src_x1 and src_y0 < src_y1:
        result[:, dst_y0:dst_y1, dst_x0:dst_x1] = image[:, src_y0:src_y1, src_x0:src_x1]
    return result


def deterministic_shift_bank_v4_0(images, config: NumeratorAuditConfigV4_0 = FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0):
    """Return the fixed 3x3 integer-shift bank without wrap-around."""
    import torch

    if config != FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0:
        _fail("V4-0 requires the frozen audit configuration")
    if not isinstance(images, torch.Tensor) or images.dtype != torch.float32 or images.ndim != 4:
        _fail("V4-0 training images must be float32 [B,1,64,64]")
    if tuple(images.shape[1:]) != (1, 64, 64):
        _fail("V4-0 training images must use the exact 64x64 crop")
    shifts = (-config.shift_pixels, 0, config.shift_pixels)
    augmented = []
    for dy in shifts:
        for dx in shifts:
            augmented.append(torch.stack([_shift_tensor(image, dx, dy) for image in images], dim=0))
    return torch.cat(augmented, dim=0)


def build_numerator_specialist_v4_0(
    *,
    seed: int,
    config: NumeratorAuditConfigV4_0 = FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0,
):
    """Build the frozen tiny 3-class numerator specialist from scratch."""
    try:
        from torch import nn
    except ModuleNotFoundError as exc:
        raise MeterV4_0AuditError("torch is required only for V4-0 execution") from exc
    from .training_model import count_trainable_parameters, set_deterministic_cpu

    if config != FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0:
        _fail("V4-0 requires the frozen audit configuration")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
        raise ValueError("seed must be bounded non-negative integer")
    set_deterministic_cpu(seed)

    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
    ).cpu()
    count = count_trainable_parameters(model)
    if not 0 < count <= config.max_trainable_parameters:
        _fail("V4-0 numerator specialist exceeds the frozen parameter budget")
    return model


def train_fold_v4_0(
    images,
    labels,
    *,
    fold: int,
    config: NumeratorAuditConfigV4_0 = FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0,
):
    """Train one fixed-length fold model; no held-out data participates."""
    import torch
    from torch.nn import functional as F
    from .training_model import model_state_sha256

    if config != FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0:
        _fail("V4-0 requires the frozen audit configuration")
    if fold not in {0, 1, 2}:
        raise ValueError("fold must be 0, 1, or 2")
    if not isinstance(images, torch.Tensor) or images.dtype != torch.float32 or tuple(images.shape) != (18, 1, 64, 64):
        _fail("each V4-0 fold must train on exact float32 [18,1,64,64]")
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64 or tuple(labels.shape) != (18,):
        _fail("each V4-0 fold must carry exact int64 [18] labels")
    if Counter(labels.tolist()) != Counter({0: 6, 1: 6, 2: 6}):
        _fail("each V4-0 fold training set must contain six examples per numerator class")

    seed = config.master_seed + fold
    model = build_numerator_specialist_v4_0(seed=seed, config=config)
    augmented = deterministic_shift_bank_v4_0(images, config)
    augmented_labels = labels.repeat(9)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate_micros / 1_000_000.0,
        weight_decay=config.weight_decay_micros / 1_000_000.0,
    )
    final_loss = float("nan")
    for _epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(augmented)
        loss = F.cross_entropy(logits, augmented_labels)
        if not bool(torch.isfinite(loss)):
            _fail("V4-0 fold loss is not finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_milli / 1000.0)
        optimizer.step()
        final_loss = float(loss.detach().item())
    return model.eval(), final_loss, model_state_sha256(model)


def classification_summary_v4_0(
    truth: Sequence[int],
    predicted: Sequence[int],
) -> ClassificationSummaryV4_0:
    if len(truth) != len(predicted) or not truth:
        raise ValueError("truth/predicted must be non-empty equal-length sequences")
    if any(value not in {0, 1, 2} for value in truth) or any(value not in {0, 1, 2} for value in predicted):
        raise ValueError("V4-0 classes must be 0,1,2")
    confusion = [[0, 0, 0] for _ in range(3)]
    for actual, guess in zip(truth, predicted):
        confusion[actual][guess] += 1
    correct = sum(confusion[index][index] for index in range(3))
    recalls: dict[str, float] = {}
    f1s: list[float] = []
    for index, label in enumerate(NUMERATOR_CLASSES):
        tp = confusion[index][index]
        fn = sum(confusion[index]) - tp
        fp = sum(confusion[row][index] for row in range(3)) - tp
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls[label] = recall
        f1s.append(f1)
    return ClassificationSummaryV4_0(
        record_count=len(truth),
        accuracy=correct / len(truth),
        macro_f1=sum(f1s) / 3.0,
        per_class_recall=recalls,
        confusion=tuple(tuple(row) for row in confusion),
    )


def audit_decision_v4_0(summary: ClassificationSummaryV4_0) -> AuditDecisionV4_0:
    if summary.record_count != 27:
        raise ValueError("V4-0 OOF decision requires exactly 27 predictions")
    reasons: list[str] = []
    if summary.accuracy < 25 / 27:
        reasons.append("OOF_ACCURACY_BELOW_25_OF_27")
    for label in NUMERATOR_CLASSES:
        if summary.per_class_recall[label] < 8 / 9:
            reasons.append(f"OOF_{label}_RECALL_BELOW_8_OF_9")
    strong = not reasons
    return AuditDecisionV4_0(
        decision="REPRESENTATION_SIGNAL_STRONG" if strong else "REPRESENTATION_SIGNAL_WEAK_OR_DATA_LIMITED",
        strong_signal=strong,
        reasons=tuple(reasons),
    )


def sealed_test_access_allowed() -> bool:
    return False


def d10_access_allowed() -> bool:
    return False


def teacher_adaptation_validation_evaluation_allowed() -> bool:
    return False


def runtime_connection_allowed() -> bool:
    return False


def resolver_connection_allowed() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False
