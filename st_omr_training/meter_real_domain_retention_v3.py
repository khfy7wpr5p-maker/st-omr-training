"""Deterministic learning-rate schedule for Meter real-domain retention V3.

This module is deliberately small.  It does not train a model, read TRAIN /
VALIDATION / TEST, load checkpoints, alter thresholds, or connect runtime.
It freezes the single-variable learning-rate intervention that will be wired
into the existing V2 shadow adaptation loop in a later commit.
"""

from __future__ import annotations

from typing import Final


METER_REAL_DOMAIN_RETENTION_V3: Final[str] = "meter-real-domain-retention-v3"
TOTAL_EPOCHS_V3: Final[int] = 20
MIDPOINT_DECAY_EPOCH_V3: Final[int] = 11
EARLY_LEARNING_RATE_MICROS_V3: Final[int] = 1000
LATE_LEARNING_RATE_MICROS_V3: Final[int] = 250


class MeterRealDomainRetentionV3Error(ValueError):
    """Raised when the frozen V3 schedule boundary is violated."""


def learning_rate_micros_for_epoch_v3(epoch: int) -> int:
    """Return the frozen V3 learning rate for a one-based epoch index."""
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise MeterRealDomainRetentionV3Error("epoch must be an integer")
    if not 1 <= epoch <= TOTAL_EPOCHS_V3:
        raise MeterRealDomainRetentionV3Error("epoch is outside the frozen 20-epoch V3 run")
    if epoch < MIDPOINT_DECAY_EPOCH_V3:
        return EARLY_LEARNING_RATE_MICROS_V3
    return LATE_LEARNING_RATE_MICROS_V3


def apply_learning_rate_v3(optimizer: object, epoch: int) -> int:
    """Apply the frozen schedule to every optimizer param group.

    The helper intentionally accepts a minimal optimizer-like object so unit
    tests do not require Torch.  Runtime integration must pass an optimizer
    exposing a mutable ``param_groups`` sequence of mappings with an ``lr``
    field, as PyTorch optimizers do.
    """
    micros = learning_rate_micros_for_epoch_v3(epoch)
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        raise MeterRealDomainRetentionV3Error("optimizer must expose non-empty param_groups")
    rate = micros / 1_000_000.0
    for group in groups:
        if not isinstance(group, dict) or "lr" not in group:
            raise MeterRealDomainRetentionV3Error("optimizer param group is malformed")
        group["lr"] = rate
    return micros


def schedule_fingerprint_payload_v3() -> dict[str, int | str]:
    """Return canonical primitive data that can be bound into later provenance."""
    return {
        "version": METER_REAL_DOMAIN_RETENTION_V3,
        "total_epochs": TOTAL_EPOCHS_V3,
        "midpoint_decay_epoch": MIDPOINT_DECAY_EPOCH_V3,
        "early_learning_rate_micros": EARLY_LEARNING_RATE_MICROS_V3,
        "late_learning_rate_micros": LATE_LEARNING_RATE_MICROS_V3,
    }
