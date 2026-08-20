"""Stage 7-D13-R2 Rest balanced-loss diagnostic primitives.

This module freezes the single-variable R2-2A experiment after D13-R1 Rest
collapsed its heatmap confidence toward zero.  It does not provide a full
training runner and it never authorizes TEST access.  The intended diagnostic
reuses the frozen R1 model/data/optimizer settings while changing only the
heatmap normalization from positive-count-normalized global loss to separately
normalized positive and negative focal terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Final, Mapping, Sequence

import torch


STAGE7D13_R2_REST_BALANCED_LOSS_VERSION: Final[str] = (
    "stage7d13-r2-rest-balanced-loss-v1"
)
R1_REPOSITORY_SHA: Final[str] = "cf82ecbc0ef8df3d635e6e1923b4c4000c40da5b"
R1_REST_EPOCH10_SHA256: Final[str] = (
    "91d1f471615058a61ddfa64bc774b3bd03bcbcfb3a594cf196ab5f9c465da1fa"
)

REST_CLASSES: Final[tuple[str, ...]] = ("half", "quarter", "eighth")
REST_TRAIN_CLASS_COUNTS: Final[dict[str, int]] = {
    "half": 1998,
    "quarter": 3417,
    "eighth": 5187,
}

# R1 settings deliberately preserved for the loss-only diagnostic.
INPUT_WIDTH: Final[int] = 512
INPUT_HEIGHT: Final[int] = 128
OUTPUT_STRIDE: Final[int] = 4
BATCH_SIZE: Final[int] = 16
LEARNING_RATE: Final[float] = 7e-4
WEIGHT_DECAY: Final[float] = 1e-4
GRAD_CLIP_NORM: Final[float] = 1.0
MASTER_SEED: Final[int] = 713_013
FOCAL_GAMMA: Final[float] = 2.0
NEGATIVE_TERM_WEIGHT: Final[float] = 1.0
DECODER_SCORE_THRESHOLD: Final[float] = 0.25

# Bounded TRAIN-only diagnostic budget.  No authoritative VALIDATION or TEST.
DIAGNOSTIC_OPTIMIZATION_RECORDS: Final[int] = 2048
DIAGNOSTIC_EVALUATION_RECORDS: Final[int] = 512
DIAGNOSTIC_EPOCHS: Final[int] = 3
DIAGNOSTIC_MAX_OPTIMIZER_STEPS: Final[int] = (
    DIAGNOSTIC_EPOCHS
    * math.ceil(DIAGNOSTIC_OPTIMIZATION_RECORDS / BATCH_SIZE)
)

_EPS: Final[float] = 1e-6
_PARTITION_SALT: Final[bytes] = b"stage7d13-r2-rest-balanced-diagnostic-v1:"
_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")


class Stage7D13R2RestBalancedLossError(RuntimeError):
    """Raised when the frozen R2-2A diagnostic contract is violated."""


@dataclass(frozen=True, slots=True)
class BalancedHeatmapLoss:
    total: torch.Tensor
    positive_term: torch.Tensor
    negative_term: torch.Tensor
    positive_count: int
    negative_count: int


@dataclass(frozen=True, slots=True)
class TrainOnlyDiagnosticPartition:
    optimization_record_ids: tuple[str, ...]
    evaluation_record_ids: tuple[str, ...]
    validation_seen: int
    test_opened: bool = False


def _fail(message: str) -> None:
    raise Stage7D13R2RestBalancedLossError(message)


def _sha64(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in _HEX for ch in value)
    ):
        _fail(f"{name} must be lowercase SHA-256")
    return value


def rest_positive_class_weights() -> dict[str, float]:
    """Return the exact R1 inverse-sqrt TRAIN-only positive class weights."""

    raw = {
        class_name: 1.0 / math.sqrt(count)
        for class_name, count in REST_TRAIN_CLASS_COUNTS.items()
    }
    mean = sum(raw.values()) / len(raw)
    return {
        class_name: min(3.0, max(0.5, raw[class_name] / mean))
        for class_name in REST_CLASSES
    }


def balanced_rest_heatmap_focal_loss(
    logits: torch.Tensor,
    heatmap: torch.Tensor,
    *,
    negative_weight: float = NEGATIVE_TERM_WEIGHT,
) -> BalancedHeatmapLoss:
    """Compute R2-2A separately normalized positive/negative focal heatmap loss.

    The R1 objective summed every negative cell and divided the complete heatmap
    loss by the number of positive cells.  For Rest that exposed roughly 15.5k
    negatives per positive in the observed VALIDATION audit.  R2-2A preserves
    the focal form and R1 positive class weights, but normalizes positive and
    negative terms independently before adding them.

    An all-negative batch remains valid: the positive term becomes a graph-safe
    zero and the negative mean is still optimized.  This is intentional because
    zero-Rest measures are legitimate examples rather than samples to discard.
    """

    if not isinstance(logits, torch.Tensor) or not isinstance(heatmap, torch.Tensor):
        raise TypeError("logits and heatmap must be torch tensors")
    if logits.shape != heatmap.shape or logits.ndim != 4:
        _fail("R2 Rest heatmap logits/targets must share [B,C,H,W] shape")
    if logits.shape[1] != len(REST_CLASSES):
        _fail("R2 Rest heatmap class dimension must be half|quarter|eighth")
    if not logits.dtype.is_floating_point:
        _fail("R2 Rest logits must be floating point")
    if not math.isfinite(float(negative_weight)) or negative_weight < 0.0:
        _fail("negative_weight must be finite and non-negative")
    if not bool(torch.isfinite(logits).all()):
        _fail("R2 Rest logits must be finite")

    target = heatmap.to(dtype=logits.dtype, device=logits.device)
    if not bool(torch.isfinite(target).all()):
        _fail("R2 Rest heatmap target must be finite")
    if not bool(((target == 0.0) | (target == 1.0)).all()):
        _fail("R2 Rest heatmap target must be binary")

    probabilities = torch.sigmoid(logits).clamp(_EPS, 1.0 - _EPS)
    positive = target.eq(1.0)
    negative = ~positive

    weights = rest_positive_class_weights()
    weight_tensor = torch.tensor(
        [weights[name] for name in REST_CLASSES],
        dtype=logits.dtype,
        device=logits.device,
    ).view(1, len(REST_CLASSES), 1, 1)

    positive_loss = (
        -weight_tensor
        * (1.0 - probabilities).pow(FOCAL_GAMMA)
        * torch.log(probabilities)
    )
    negative_loss = (
        -probabilities.pow(FOCAL_GAMMA)
        * torch.log(1.0 - probabilities)
    )

    positive_count = int(positive.sum().item())
    negative_count = int(negative.sum().item())

    if positive_count:
        positive_term = positive_loss.masked_select(positive).mean()
    else:
        positive_term = logits.sum() * 0.0

    if negative_count:
        negative_term = negative_loss.masked_select(negative).mean()
    else:
        negative_term = logits.sum() * 0.0

    total = positive_term + float(negative_weight) * negative_term
    if not bool(torch.isfinite(total)):
        _fail("R2 Rest balanced heatmap loss became non-finite")

    return BalancedHeatmapLoss(
        total=total,
        positive_term=positive_term,
        negative_term=negative_term,
        positive_count=positive_count,
        negative_count=negative_count,
    )


def r1_heatmap_focal_loss_for_diagnostic_comparison(
    logits: torch.Tensor,
    heatmap: torch.Tensor,
) -> torch.Tensor:
    """Reproduce only the R1 heatmap term for controlled unit comparison."""

    if logits.shape != heatmap.shape or logits.ndim != 4:
        _fail("R1 comparison logits/targets must share [B,C,H,W] shape")
    target = heatmap.to(dtype=logits.dtype, device=logits.device)
    probabilities = torch.sigmoid(logits).clamp(_EPS, 1.0 - _EPS)
    positive = target.eq(1.0)
    negative = ~positive
    weights = rest_positive_class_weights()
    weight_tensor = torch.tensor(
        [weights[name] for name in REST_CLASSES],
        dtype=logits.dtype,
        device=logits.device,
    ).view(1, len(REST_CLASSES), 1, 1)
    positive_loss = (
        -weight_tensor
        * (1.0 - probabilities).pow(FOCAL_GAMMA)
        * torch.log(probabilities)
    )
    negative_loss = (
        -probabilities.pow(FOCAL_GAMMA)
        * torch.log(1.0 - probabilities)
    )
    positive_count = positive.sum().clamp_min(1).to(dtype=logits.dtype)
    return (
        positive_loss.masked_select(positive).sum()
        + negative_loss.masked_select(negative).sum()
    ) / positive_count


def select_train_only_diagnostic_partition(
    rows: Sequence[Mapping[str, object]],
    *,
    optimization_records: int = DIAGNOSTIC_OPTIMIZATION_RECORDS,
    evaluation_records: int = DIAGNOSTIC_EVALUATION_RECORDS,
) -> TrainOnlyDiagnosticPartition:
    """Deterministically select disjoint TRAIN-only optimization/eval records.

    VALIDATION rows are counted but never selected.  Encountering TEST fails
    immediately before any TEST payload can be used.  Ranking depends only on a
    salted SHA-256 of record_id, not labels or model outcomes.
    """

    if not isinstance(optimization_records, int) or isinstance(optimization_records, bool):
        raise TypeError("optimization_records must be int")
    if not isinstance(evaluation_records, int) or isinstance(evaluation_records, bool):
        raise TypeError("evaluation_records must be int")
    if optimization_records <= 0 or evaluation_records <= 0:
        _fail("diagnostic partition counts must be positive")

    ranked: list[tuple[bytes, str]] = []
    validation_seen = 0
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"manifest row[{index}] must be an object")
        split = row.get("split")
        if split == "test":
            _fail("sealed TEST reached R2-2A TRAIN-only diagnostic partition")
        if split == "validation":
            validation_seen += 1
            continue
        if split != "train":
            _fail("R2-2A diagnostic split must be train or validation")
        record_id = _sha64(row.get("record_id"), "record_id")
        if record_id in seen:
            _fail("duplicate TRAIN record_id in R2-2A partition input")
        seen.add(record_id)
        rank = sha256(_PARTITION_SALT + record_id.encode("ascii")).digest()
        ranked.append((rank, record_id))

    required = optimization_records + evaluation_records
    if len(ranked) < required:
        _fail("insufficient TRAIN records for frozen R2-2A diagnostic partition")
    ranked.sort(key=lambda item: (item[0], item[1]))
    chosen = [record_id for _rank, record_id in ranked[:required]]
    return TrainOnlyDiagnosticPartition(
        optimization_record_ids=tuple(chosen[:optimization_records]),
        evaluation_record_ids=tuple(chosen[optimization_records:]),
        validation_seen=validation_seen,
        test_opened=False,
    )
