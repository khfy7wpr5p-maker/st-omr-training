"""Strict read-only audit for the frozen real Meter specialist checkpoints.

This module verifies local checkpoint files before any runtime inference adapter
is allowed to consume their model state. It intentionally does not train, tune
thresholds, access data splits, invoke the Resolver, or claim that the current
runtime measure-start ROI is pixel-equivalent to the historical D11 Meter ROI.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
import re
from typing import Final, Mapping

import torch


METER_REAL_CHECKPOINT_AUDIT_V1_VERSION: Final[str] = "meter-real-checkpoint-audit-v1"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

PRESENCE_D11_SHA256: Final[str] = "cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3"
DIGIT2_SHA256: Final[str] = "92b985d989e4338e3ae39b0a984879f4188be32c0d281390839117e1e9a715fa"
DIGIT3_SHA256: Final[str] = "5ee45faf2efe0e2c83dbad716736d7ae16ad7251730431d368c10c4574836485"
DIGIT4_SHA256: Final[str] = "dcd582b60b39e65798aa77aacea3cc797cd7513b7925151f0573be4aec6af43f"

_DIGIT_STATE_SHAPES: Final[dict[str, tuple[int, ...]]] = {
    "features.0.weight": (16, 1, 3, 3),
    "features.0.bias": (16,),
    "features.3.weight": (32, 16, 3, 3),
    "features.3.bias": (32,),
    "features.6.weight": (64, 32, 3, 3),
    "features.6.bias": (64,),
    "head.weight": (1, 64),
    "head.bias": (1,),
}

_D11_METER_STATE_SHAPES: Final[dict[str, tuple[int, ...]]] = {
    "encoder.0.weight": (8, 1, 3, 3),
    "encoder.0.bias": (8,),
    "encoder.2.weight": (16, 8, 3, 3),
    "encoder.2.bias": (16,),
    "encoder.4.weight": (24, 16, 3, 3),
    "encoder.4.bias": (24,),
    "encoder.6.weight": (24, 24, 3, 3),
    "encoder.6.bias": (24,),
    "projection.1.weight": (64, 1152),
    "projection.1.bias": (64,),
    "classifier.weight": (4, 64),
    "classifier.bias": (4,),
    "bbox.weight": (4, 64),
    "bbox.bias": (4,),
}


class MeterRealCheckpointAuditError(RuntimeError):
    """Raised when frozen checkpoint identity/state cannot be trusted."""


@dataclass(frozen=True, slots=True)
class AuditedCheckpointStateV1:
    role: str
    checkpoint_sha256: str
    byte_length: int
    model_state: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("checkpoint role must be non-empty")
        if _SHA_RE.fullmatch(self.checkpoint_sha256) is None:
            raise ValueError("checkpoint SHA must be canonical lowercase SHA-256")
        if not isinstance(self.byte_length, int) or isinstance(self.byte_length, bool) or self.byte_length <= 0:
            raise ValueError("checkpoint byte length must be positive")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return digest.hexdigest(), size


def _regular_checkpoint(path: object, *, maximum_bytes: int) -> Path:
    if not isinstance(path, Path):
        raise TypeError("checkpoint path must be pathlib.Path")
    if path.is_symlink() or not path.is_file():
        raise MeterRealCheckpointAuditError("checkpoint must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= maximum_bytes:
        raise MeterRealCheckpointAuditError("checkpoint byte length is outside the frozen audit bound")
    return path


def _strict_state(
    state: object,
    expected_shapes: Mapping[str, tuple[int, ...]],
) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise MeterRealCheckpointAuditError("checkpoint model state must be a mapping")
    if set(state) != set(expected_shapes):
        raise MeterRealCheckpointAuditError("checkpoint model-state keys differ from the frozen architecture")
    checked: dict[str, torch.Tensor] = {}
    for name in expected_shapes:
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise MeterRealCheckpointAuditError("checkpoint model state contains a non-tensor value")
        if tuple(tensor.shape) != expected_shapes[name]:
            raise MeterRealCheckpointAuditError("checkpoint tensor shape differs from the frozen architecture")
        if not bool(torch.isfinite(tensor).all().item()):
            raise MeterRealCheckpointAuditError("checkpoint model state contains non-finite values")
        checked[name] = tensor.detach().cpu()
    return checked


def _load_weights_only(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # fail closed at the external binary boundary
        raise MeterRealCheckpointAuditError("checkpoint cannot be loaded by the weights-only CPU loader") from exc


def audit_digit_checkpoint_v1(
    path: Path,
    *,
    role: str,
    expected_sha256: str,
) -> AuditedCheckpointStateV1:
    """Verify one frozen 2/3/4 specialist and expose only its model state."""
    if role not in {"digit-2", "digit-3", "digit-4"}:
        raise ValueError("digit checkpoint role must be digit-2, digit-3, or digit-4")
    if _SHA_RE.fullmatch(expected_sha256) is None:
        raise ValueError("expected checkpoint SHA must be canonical lowercase SHA-256")
    file_path = _regular_checkpoint(path, maximum_bytes=5_000_000)
    observed_sha, byte_length = _sha256_file(file_path)
    if observed_sha != expected_sha256:
        raise MeterRealCheckpointAuditError("digit checkpoint SHA-256 mismatch")
    payload = _load_weights_only(file_path)
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise MeterRealCheckpointAuditError("digit checkpoint is missing model_state_dict")
    state = _strict_state(payload["model_state_dict"], _DIGIT_STATE_SHAPES)
    return AuditedCheckpointStateV1(role, observed_sha, byte_length, state)


def audit_presence_d11_checkpoint_v1(path: Path) -> AuditedCheckpointStateV1:
    """Verify the frozen D11 Meter checkpoint used only as temporary Presence bridge."""
    file_path = _regular_checkpoint(path, maximum_bytes=20_000_000)
    observed_sha, byte_length = _sha256_file(file_path)
    if observed_sha != PRESENCE_D11_SHA256:
        raise MeterRealCheckpointAuditError("D11 Presence-bridge checkpoint SHA-256 mismatch")
    payload = _load_weights_only(file_path)
    if not isinstance(payload, Mapping) or "meter_state_dict" not in payload:
        raise MeterRealCheckpointAuditError("D11 checkpoint is missing meter_state_dict")
    state = _strict_state(payload["meter_state_dict"], _D11_METER_STATE_SHAPES)
    return AuditedCheckpointStateV1("presence-d11-bridge", observed_sha, byte_length, state)


def conservative_probability_to_milli_v1(value: float) -> int:
    """Quantize without promoting a sub-threshold probability by normal rounding."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("probability must be a finite numeric value")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite in 0..1")
    return min(1000, max(0, int(math.floor(probability * 1000.0 + 1e-12))))


def checkpoint_loading_requires_explicit_local_paths() -> bool:
    return True


def training_or_threshold_tuning_allowed() -> bool:
    return False


def resolver_connection_allowed() -> bool:
    return False


def runtime_real_inference_promoted() -> bool:
    """Checkpoint audit is necessary but cannot close the current pixel/context HOLD."""
    return False
