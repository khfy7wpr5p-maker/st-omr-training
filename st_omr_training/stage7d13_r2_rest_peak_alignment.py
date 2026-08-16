"""Stage 7-D13-R2 Rest exact-cell peak-alignment diagnostic primitive.

R2-2A fixed the Rest all-zero heatmap collapse by normalizing positive and
negative focal terms separately. R2-2E then showed that the remaining dominant
failure is a heatmap/regression-cell mismatch: exact GT-cell regression was
within 4 px for every audited target, while many same-class heatmap local maxima
shifted one or two output cells away.

This module freezes one new TRAIN-only diagnostic variable: a same-class local
peak-alignment loss that asks the exact GT heatmap cell to outrank non-target
neighbors in a radius-2 (5x5) window. Other positive cells are excluded from the
competitor set so legitimate nearby Rest targets never suppress each other.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch

from .stage7d13_r2_rest_balanced_loss import REST_CLASSES


STAGE7D13_R2_REST_PEAK_ALIGNMENT_VERSION: Final[str] = (
    "stage7d13-r2-rest-peak-alignment-v1"
)
R2_2A_REPOSITORY_SHA: Final[str] = "5daa7fe564db0270b37b3bd47fe68b6ede545bc8"
PEAK_ALIGNMENT_RADIUS: Final[int] = 2
PEAK_ALIGNMENT_WEIGHT: Final[float] = 1.0


class Stage7D13R2RestPeakAlignmentError(RuntimeError):
    """Raised when the frozen R2-3A peak-alignment contract is violated."""


@dataclass(frozen=True, slots=True)
class PeakAlignmentLoss:
    total: torch.Tensor
    target_count: int
    compared_target_count: int


def _fail(message: str) -> None:
    raise Stage7D13R2RestPeakAlignmentError(message)


def local_gt_cell_peak_alignment_loss(
    logits: torch.Tensor,
    heatmap: torch.Tensor,
    *,
    radius: int = PEAK_ALIGNMENT_RADIUS,
) -> PeakAlignmentLoss:
    """Make each exact GT cell outrank same-class non-target local neighbors.

    For every binary-positive heatmap cell, collect the same-class logits in the
    clipped `(2 * radius + 1)^2` neighborhood. The denominator contains the GT
    logit plus only non-target cells; other positive GT cells are excluded.
    The per-target term is therefore `logsumexp(target + local negatives) -
    target`. Terms are averaged across targets.

    This loss deliberately does not change center-offset or bbox regression and
    does not introduce cross-class competition because the R2-2D audit observed
    no wrong-class candidate within 4 px for the missed targets.
    """

    if not isinstance(logits, torch.Tensor) or not isinstance(heatmap, torch.Tensor):
        raise TypeError("logits and heatmap must be torch tensors")
    if logits.shape != heatmap.shape or logits.ndim != 4:
        _fail("R2 Rest peak logits/targets must share [B,C,H,W] shape")
    if logits.shape[1] != len(REST_CLASSES):
        _fail("R2 Rest peak class dimension must be half|quarter|eighth")
    if not logits.dtype.is_floating_point:
        _fail("R2 Rest peak logits must be floating point")
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 1:
        _fail("peak-alignment radius must be positive int")
    if not bool(torch.isfinite(logits).all()):
        _fail("R2 Rest peak logits must be finite")

    target = heatmap.to(dtype=logits.dtype, device=logits.device)
    if not bool(torch.isfinite(target).all()):
        _fail("R2 Rest peak target must be finite")
    if not bool(((target == 0.0) | (target == 1.0)).all()):
        _fail("R2 Rest peak target must be binary")

    positive = target.eq(1.0)
    positive_coordinates = torch.nonzero(positive, as_tuple=False)
    target_count = int(positive_coordinates.shape[0])

    if target_count == 0:
        return PeakAlignmentLoss(
            total=logits.sum() * 0.0,
            target_count=0,
            compared_target_count=0,
        )

    terms: list[torch.Tensor] = []
    height = logits.shape[2]
    width = logits.shape[3]

    for coordinate in positive_coordinates:
        batch_index, class_index, y, x = (int(value) for value in coordinate.tolist())
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)

        local_logits = logits[batch_index, class_index, y0:y1, x0:x1].reshape(-1)
        local_positive = positive[
            batch_index, class_index, y0:y1, x0:x1
        ].reshape(-1)

        center_logit = logits[batch_index, class_index, y, x]
        negative_neighbor_logits = local_logits.masked_select(~local_positive)

        if negative_neighbor_logits.numel() == 0:
            continue

        comparison_logits = torch.cat(
            (center_logit.reshape(1), negative_neighbor_logits),
            dim=0,
        )
        terms.append(torch.logsumexp(comparison_logits, dim=0) - center_logit)

    if not terms:
        total = logits.sum() * 0.0
        compared_target_count = 0
    else:
        total = torch.stack(terms).mean()
        compared_target_count = len(terms)

    if not bool(torch.isfinite(total)):
        _fail("R2 Rest peak-alignment loss became non-finite")

    return PeakAlignmentLoss(
        total=total,
        target_count=target_count,
        compared_target_count=compared_target_count,
    )


def peak_alignment_contract_payload() -> dict[str, object]:
    """Return the frozen R2-3A single-variable experiment constants."""

    return {
        "version": STAGE7D13_R2_REST_PEAK_ALIGNMENT_VERSION,
        "base_r2_2a_repository_sha": R2_2A_REPOSITORY_SHA,
        "classes": REST_CLASSES,
        "radius_cells": PEAK_ALIGNMENT_RADIUS,
        "window": 2 * PEAK_ALIGNMENT_RADIUS + 1,
        "weight": PEAK_ALIGNMENT_WEIGHT,
        "same_class_only": True,
        "exclude_other_positive_targets": True,
        "scheduler": None,
        "production_checkpoint": False,
        "test_authorized": False,
    }
