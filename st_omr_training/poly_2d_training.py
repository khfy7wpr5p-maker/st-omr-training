"""Bounded training and provenance harness for the tiny Polyphonic 2D Transformer.

TR-POLY-08A adds the minimum training surface required to move the executable
TR-POLY-08 forward model from architecture-only research code to an implemented
training candidate. It does not load repository datasets, write checkpoints,
open the sealed TEST split, run benchmarks, or grant production authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
from typing import Final

import torch
from torch.nn import functional as F

from .dataset_manifest import DatasetSplit
from .poly_2d_transformer import (
    FROZEN_POLY_2D_CONFIG,
    Poly2DTransformerConfig,
    TinyPoly2DTransformer,
    build_tiny_poly_2d_transformer,
    poly_2d_config_fingerprint,
)
from .polyphonic_representation import POLYPHONIC_REPRESENTATION_VERSION
from .polyphonic_serialization import (
    BOS_TOKEN_ID,
    PAD_TOKEN_ID,
    POLYPHONIC_TOKENIZER_VERSION,
    VOCABULARY_SIZE,
    tokenizer_fingerprint,
)
from .training_model import (
    TORCH_PINNED_VERSION,
    TrainingRuntimeError,
    assert_finite_tensor,
    assert_model_finite,
    assert_optimizer_finite,
    count_trainable_parameters,
    model_state_sha256,
)


POLY_2D_TRAINER_VERSION: Final[str] = "st-omr-poly-2d-trainer-v1"
POLY_2D_TRAINING_BATCH_VERSION: Final[str] = "st-omr-poly-2d-training-batch-v1"
POLY_2D_PROVENANCE_VERSION: Final[str] = "st-omr-poly-2d-training-provenance-v1"
MAX_POLY_2D_TRAINING_BATCH: Final[int] = 8
MAX_POLY_2D_SMOKE_STEPS: Final[int] = 2
MAX_POLY_2D_TRAINING_PIXELS: Final[int] = 1_048_576
_MAX_SEED: Final[int] = 2**63 - 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA40 = re.compile(r"^[0-9a-f]{40}$")


class Poly2DTrainingError(TrainingRuntimeError):
    """Raised when the bounded polyphonic training surface fails closed."""


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Poly2DTrainingError(f"{name} must be lowercase SHA-256 hex")
    return value


def _require_git_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _GIT_SHA40.fullmatch(value) is None:
        raise Poly2DTrainingError(f"{name} must be lowercase git SHA-40 hex")
    return value


@dataclass(frozen=True, slots=True)
class Poly2DTrainingConfig:
    master_seed: int = 82_081
    learning_rate_micros: int = 500
    weight_decay_micros: int = 100
    grad_clip_milli: int = 1_000
    smoke_steps: int = 2
    objective: str = "cross_entropy_ignore_v2_pad"
    optimizer: str = "adamw"
    scheduler: str = "none"
    batch_policy: str = "prevalidated_v2_tensor_batch_v1"

    def __post_init__(self) -> None:
        bounds = {
            "master_seed": (self.master_seed, 0, _MAX_SEED),
            "learning_rate_micros": (self.learning_rate_micros, 1, 100_000),
            "weight_decay_micros": (self.weight_decay_micros, 0, 100_000),
            "grad_clip_milli": (self.grad_clip_milli, 1, 100_000),
            "smoke_steps": (self.smoke_steps, 1, MAX_POLY_2D_SMOKE_STEPS),
        }
        for name, (value, lower, upper) in bounds.items():
            if not _plain_int(value) or not lower <= value <= upper:
                raise ValueError(f"{name} is outside the TR-POLY-08A boundary")
        expected = {
            "objective": "cross_entropy_ignore_v2_pad",
            "optimizer": "adamw",
            "scheduler": "none",
            "batch_policy": "prevalidated_v2_tensor_batch_v1",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} is frozen to {value!r}")


FROZEN_POLY_2D_TRAINING_CONFIG: Final[Poly2DTrainingConfig] = Poly2DTrainingConfig()


def poly_2d_trainer_fingerprint(
    config: Poly2DTrainingConfig = FROZEN_POLY_2D_TRAINING_CONFIG,
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
) -> str:
    if not isinstance(config, Poly2DTrainingConfig):
        raise TypeError("config must be Poly2DTrainingConfig")
    if not isinstance(model_config, Poly2DTransformerConfig):
        raise TypeError("model_config must be Poly2DTransformerConfig")
    payload = {
        "trainer_version": POLY_2D_TRAINER_VERSION,
        "batch_version": POLY_2D_TRAINING_BATCH_VERSION,
        "torch_version": TORCH_PINNED_VERSION,
        "representation_version": POLYPHONIC_REPRESENTATION_VERSION,
        "tokenizer_version": POLYPHONIC_TOKENIZER_VERSION,
        "tokenizer_fingerprint": tokenizer_fingerprint(),
        "model_profile_sha256": poly_2d_config_fingerprint(model_config),
        "config": asdict(config),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class Poly2DTrainingProvenance:
    repository_sha: str
    dataset_manifest_sha256: str
    preprocess_fingerprint_sha256: str
    model_profile_sha256: str
    trainer_profile_sha256: str
    tokenizer_fingerprint_sha256: str
    representation_version: str = POLYPHONIC_REPRESENTATION_VERSION
    tokenizer_version: str = POLYPHONIC_TOKENIZER_VERSION
    torch_version: str = TORCH_PINNED_VERSION
    provenance_version: str = POLY_2D_PROVENANCE_VERSION

    def __post_init__(self) -> None:
        _require_git_sha(self.repository_sha, "repository_sha")
        for name, value in (
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
            ("preprocess_fingerprint_sha256", self.preprocess_fingerprint_sha256),
            ("model_profile_sha256", self.model_profile_sha256),
            ("trainer_profile_sha256", self.trainer_profile_sha256),
            ("tokenizer_fingerprint_sha256", self.tokenizer_fingerprint_sha256),
        ):
            _require_sha256(value, name)
        if self.tokenizer_fingerprint_sha256 != tokenizer_fingerprint():
            raise Poly2DTrainingError("provenance tokenizer fingerprint differs from frozen V2 tokenizer")
        if self.representation_version != POLYPHONIC_REPRESENTATION_VERSION:
            raise Poly2DTrainingError("provenance representation version mismatch")
        if self.tokenizer_version != POLYPHONIC_TOKENIZER_VERSION:
            raise Poly2DTrainingError("provenance tokenizer version mismatch")
        if self.torch_version != TORCH_PINNED_VERSION:
            raise Poly2DTrainingError("provenance torch version mismatch")
        if self.provenance_version != POLY_2D_PROVENANCE_VERSION:
            raise Poly2DTrainingError("provenance schema version mismatch")

    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        return sha256(_canonical_json_bytes(self.canonical_payload())).hexdigest()


def build_poly_2d_training_provenance(
    *,
    repository_sha: str,
    dataset_manifest_sha256: str,
    preprocess_fingerprint_sha256: str,
    training_config: Poly2DTrainingConfig = FROZEN_POLY_2D_TRAINING_CONFIG,
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
) -> Poly2DTrainingProvenance:
    return Poly2DTrainingProvenance(
        repository_sha=repository_sha,
        dataset_manifest_sha256=dataset_manifest_sha256,
        preprocess_fingerprint_sha256=preprocess_fingerprint_sha256,
        model_profile_sha256=poly_2d_config_fingerprint(model_config),
        trainer_profile_sha256=poly_2d_trainer_fingerprint(training_config, model_config),
        tokenizer_fingerprint_sha256=tokenizer_fingerprint(),
    )


def _is_right_padded(tensor: torch.Tensor) -> bool:
    for row in tensor:
        seen_pad = False
        for value in row.tolist():
            if value == PAD_TOKEN_ID:
                seen_pad = True
            elif seen_pad:
                return False
    return True


@dataclass(slots=True)
class Poly2DTrainingBatch:
    images: torch.Tensor
    decoder_input_ids: torch.Tensor
    labels: torch.Tensor
    split: DatasetSplit
    sample_ids: tuple[str, ...]
    dataset_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.split not in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}:
            raise Poly2DTrainingError("sealed TEST data cannot enter TR-POLY-08A")
        _require_sha256(self.dataset_manifest_sha256, "dataset_manifest_sha256")
        if (
            not isinstance(self.sample_ids, tuple)
            or not self.sample_ids
            or len(self.sample_ids) > MAX_POLY_2D_TRAINING_BATCH
            or len(set(self.sample_ids)) != len(self.sample_ids)
            or any(not isinstance(value, str) or not value for value in self.sample_ids)
        ):
            raise Poly2DTrainingError("sample_ids must be a bounded non-empty unique tuple")
        if not isinstance(self.images, torch.Tensor) or self.images.dtype != torch.float32:
            raise Poly2DTrainingError("images must be a float32 torch tensor")
        if self.images.ndim != 4 or self.images.shape[1] != 1:
            raise Poly2DTrainingError("images must have shape [batch,1,height,width]")
        batch_size, _channels, height, width = self.images.shape
        if batch_size != len(self.sample_ids) or not 1 <= batch_size <= MAX_POLY_2D_TRAINING_BATCH:
            raise Poly2DTrainingError("image batch size does not match sample_ids")
        if height <= 0 or width <= 0 or height * width > MAX_POLY_2D_TRAINING_PIXELS:
            raise Poly2DTrainingError("image dimensions exceed the TR-POLY-08A tensor boundary")
        if not bool(torch.isfinite(self.images).all()):
            raise Poly2DTrainingError("images contain NaN or Infinity")
        if bool((self.images < 0).any()) or bool((self.images > 1).any()):
            raise Poly2DTrainingError("images must stay in normalized [0,1] range")
        for tensor, name in (
            (self.decoder_input_ids, "decoder_input_ids"),
            (self.labels, "labels"),
        ):
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.long or tensor.ndim != 2:
                raise Poly2DTrainingError(f"{name} must be rank-2 torch.long")
            if tensor.shape[0] != batch_size or tensor.shape[1] < 1:
                raise Poly2DTrainingError(f"{name} batch shape is invalid")
            if bool((tensor < 0).any()) or bool((tensor >= VOCABULARY_SIZE).any()):
                raise Poly2DTrainingError(f"{name} contains an id outside the frozen V2 vocabulary")
            if not _is_right_padded(tensor):
                raise Poly2DTrainingError(f"{name} must use contiguous right padding")
        if self.decoder_input_ids.shape != self.labels.shape:
            raise Poly2DTrainingError("decoder inputs and labels must have identical shapes")
        if not bool((self.decoder_input_ids[:, 0] == BOS_TOKEN_ID).all()):
            raise Poly2DTrainingError("every teacher-forced decoder row must start with V2 BOS")
        if not bool((self.labels != PAD_TOKEN_ID).any()):
            raise Poly2DTrainingError("batch contains no unmasked V2 target labels")
        if not torch.equal(self.decoder_input_ids.eq(PAD_TOKEN_ID), self.labels.eq(PAD_TOKEN_ID)):
            raise Poly2DTrainingError("decoder inputs and labels must share the same padding mask")


def _validate_batch_for_model(batch: Poly2DTrainingBatch, model: TinyPoly2DTransformer) -> None:
    if not isinstance(batch, Poly2DTrainingBatch):
        raise TypeError("batch must be Poly2DTrainingBatch")
    if not isinstance(model, TinyPoly2DTransformer):
        raise TypeError("model must be TinyPoly2DTransformer")
    expected_image_shape = (1, model.config.input_height, model.config.input_width)
    if tuple(batch.images.shape[1:]) != expected_image_shape:
        raise Poly2DTrainingError("batch image geometry differs from model config")
    if batch.decoder_input_ids.shape[1] > model.config.max_target_tokens:
        raise Poly2DTrainingError("batch target length exceeds model config")


def _masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or labels.dtype != torch.long:
        raise Poly2DTrainingError("labels do not align with model logits")
    if not bool((labels != PAD_TOKEN_ID).any()):
        raise Poly2DTrainingError("loss has no unmasked V2 target positions")
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=PAD_TOKEN_ID,
    )
    assert_finite_tensor("TR-POLY-08A training loss", loss)
    return loss


def build_poly_2d_optimizer(
    model: TinyPoly2DTransformer,
    config: Poly2DTrainingConfig = FROZEN_POLY_2D_TRAINING_CONFIG,
) -> torch.optim.AdamW:
    if not isinstance(model, TinyPoly2DTransformer):
        raise TypeError("model must be TinyPoly2DTransformer")
    if not isinstance(config, Poly2DTrainingConfig):
        raise TypeError("config must be Poly2DTrainingConfig")
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate_micros / 1_000_000,
        weight_decay=config.weight_decay_micros / 1_000_000,
        foreach=False,
        fused=False,
    )


def train_poly_2d_one_step(
    model: TinyPoly2DTransformer,
    batch: Poly2DTrainingBatch,
    optimizer: torch.optim.Optimizer,
    config: Poly2DTrainingConfig = FROZEN_POLY_2D_TRAINING_CONFIG,
) -> float:
    if batch.split is not DatasetSplit.TRAIN:
        raise Poly2DTrainingError("gradient updates are allowed only from the TRAIN split")
    _validate_batch_for_model(batch, model)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer")
    if not isinstance(config, Poly2DTrainingConfig):
        raise TypeError("config must be Poly2DTrainingConfig")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch.images, batch.decoder_input_ids)
    loss = _masked_cross_entropy(logits, batch.labels)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert_finite_tensor(f"TR-POLY-08A gradient {name}", parameter.grad)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.grad_clip_milli / 1000.0,
        error_if_nonfinite=True,
    )
    if isinstance(grad_norm, torch.Tensor):
        assert_finite_tensor("TR-POLY-08A gradient norm", grad_norm)
    elif not math.isfinite(float(grad_norm)):
        raise Poly2DTrainingError("gradient norm is non-finite")
    optimizer.step()
    assert_model_finite(model)
    assert_optimizer_finite(optimizer)
    return float(loss.detach().cpu().item())


def evaluate_poly_2d_validation_loss(
    model: TinyPoly2DTransformer,
    batch: Poly2DTrainingBatch,
) -> float:
    if batch.split is not DatasetSplit.VALIDATION:
        raise Poly2DTrainingError("read-only evaluation requires the VALIDATION split")
    _validate_batch_for_model(batch, model)
    before = model_state_sha256(model)
    previous_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(batch.images, batch.decoder_input_ids)
            loss = _masked_cross_entropy(logits, batch.labels)
    finally:
        model.train(previous_training)
    after = model_state_sha256(model)
    if before != after:
        raise Poly2DTrainingError("VALIDATION evaluation mutated model state")
    return float(loss.detach().cpu().item())


@dataclass(frozen=True, slots=True)
class Poly2DSmokeTrainingResult:
    initial_state_sha256: str
    final_state_sha256: str
    train_losses: tuple[float, ...]
    validation_loss: float
    optimizer_steps: int
    parameter_count: int
    model_profile_sha256: str
    trainer_profile_sha256: str
    provenance_sha256: str
    dataset_manifest_sha256: str
    tokenizer_fingerprint_sha256: str
    torch_version: str
    checkpoint_written: bool = False
    authoritative_dataset_execution: bool = False
    test_split_accessed: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("initial_state_sha256", self.initial_state_sha256),
            ("final_state_sha256", self.final_state_sha256),
            ("model_profile_sha256", self.model_profile_sha256),
            ("trainer_profile_sha256", self.trainer_profile_sha256),
            ("provenance_sha256", self.provenance_sha256),
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
            ("tokenizer_fingerprint_sha256", self.tokenizer_fingerprint_sha256),
        ):
            _require_sha256(value, name)
        if self.initial_state_sha256 == self.final_state_sha256:
            raise Poly2DTrainingError("bounded training did not update model state")
        if not self.train_losses or any(not math.isfinite(value) for value in self.train_losses):
            raise Poly2DTrainingError("training result contains non-finite train loss")
        if not math.isfinite(self.validation_loss):
            raise Poly2DTrainingError("training result contains non-finite validation loss")
        if self.optimizer_steps != len(self.train_losses) or not 1 <= self.optimizer_steps <= MAX_POLY_2D_SMOKE_STEPS:
            raise Poly2DTrainingError("training result optimizer-step count is invalid")
        if not _plain_int(self.parameter_count) or self.parameter_count <= 0:
            raise Poly2DTrainingError("training result parameter count is invalid")
        if self.tokenizer_fingerprint_sha256 != tokenizer_fingerprint():
            raise Poly2DTrainingError("training result tokenizer fingerprint mismatch")
        if self.torch_version != TORCH_PINNED_VERSION:
            raise Poly2DTrainingError("training result torch version mismatch")
        if self.checkpoint_written or self.authoritative_dataset_execution or self.test_split_accessed:
            raise Poly2DTrainingError("TR-POLY-08A result may not claim checkpoint, authoritative data, or TEST access")


def run_bounded_poly_2d_smoke_training(
    *,
    train_batches: tuple[Poly2DTrainingBatch, ...],
    validation_batch: Poly2DTrainingBatch,
    provenance: Poly2DTrainingProvenance,
    training_config: Poly2DTrainingConfig = FROZEN_POLY_2D_TRAINING_CONFIG,
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
) -> Poly2DSmokeTrainingResult:
    if not isinstance(train_batches, tuple) or not train_batches:
        raise Poly2DTrainingError("train_batches must be a non-empty immutable tuple")
    if any(not isinstance(batch, Poly2DTrainingBatch) or batch.split is not DatasetSplit.TRAIN for batch in train_batches):
        raise Poly2DTrainingError("train_batches may contain only TRAIN batches")
    if not isinstance(validation_batch, Poly2DTrainingBatch) or validation_batch.split is not DatasetSplit.VALIDATION:
        raise Poly2DTrainingError("validation_batch must be a VALIDATION batch")
    if not isinstance(provenance, Poly2DTrainingProvenance):
        raise TypeError("provenance must be Poly2DTrainingProvenance")
    expected_model_profile = poly_2d_config_fingerprint(model_config)
    expected_trainer_profile = poly_2d_trainer_fingerprint(training_config, model_config)
    if provenance.model_profile_sha256 != expected_model_profile:
        raise Poly2DTrainingError("provenance model profile mismatch")
    if provenance.trainer_profile_sha256 != expected_trainer_profile:
        raise Poly2DTrainingError("provenance trainer profile mismatch")
    all_batches = train_batches + (validation_batch,)
    if any(batch.dataset_manifest_sha256 != provenance.dataset_manifest_sha256 for batch in all_batches):
        raise Poly2DTrainingError("batch dataset identity differs from provenance")

    model = build_tiny_poly_2d_transformer(model_config, seed=training_config.master_seed)
    optimizer = build_poly_2d_optimizer(model, training_config)
    initial_state = model_state_sha256(model)
    losses: list[float] = []
    for step in range(training_config.smoke_steps):
        batch = train_batches[step % len(train_batches)]
        losses.append(train_poly_2d_one_step(model, batch, optimizer, training_config))
    validation_loss = evaluate_poly_2d_validation_loss(model, validation_batch)
    final_state = model_state_sha256(model)

    return Poly2DSmokeTrainingResult(
        initial_state_sha256=initial_state,
        final_state_sha256=final_state,
        train_losses=tuple(losses),
        validation_loss=validation_loss,
        optimizer_steps=training_config.smoke_steps,
        parameter_count=count_trainable_parameters(model),
        model_profile_sha256=expected_model_profile,
        trainer_profile_sha256=expected_trainer_profile,
        provenance_sha256=provenance.fingerprint(),
        dataset_manifest_sha256=provenance.dataset_manifest_sha256,
        tokenizer_fingerprint_sha256=tokenizer_fingerprint(),
        torch_version=TORCH_PINNED_VERSION,
    )