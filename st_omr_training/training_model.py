"""Bounded from-scratch PyTorch baseline and deterministic CPU smoke trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from .dataset_manifest import DatasetSplit
from .training_data import TrainingBatch
from .training_tokens import PAD_TOKEN_ID, VOCABULARY_SIZE

TORCH_PINNED_VERSION: Final[str] = "2.13.0+cpu"
BASELINE_MODEL_VERSION: Final[str] = "st-omr-cnn-gru-baseline-v1"
TRAINER_VERSION: Final[str] = "st-omr-smoke-trainer-v1"
MAX_TRAINABLE_PARAMETERS: Final[int] = 25_000_000
_MAX_SEED: Final[int] = 2**63 - 1


class TrainingRuntimeError(RuntimeError):
    """Raised when the selected training runtime or numeric state fails closed."""


class TrainingConfigError(ValueError):
    """Raised when a Stage 7-B model/trainer configuration violates the contract."""


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class BaselineModelConfig:
    input_height: int = 64
    input_width: int = 512
    conv_channels: tuple[int, int] = (8, 16)
    encoder_steps: int = 32
    embedding_dim: int = 32
    hidden_dim: int = 64

    def __post_init__(self) -> None:
        if not _is_plain_int(self.input_height) or not 32 <= self.input_height <= 512:
            raise TrainingConfigError("input_height is outside the Stage 7-B boundary")
        if not _is_plain_int(self.input_width) or not 128 <= self.input_width <= 2048:
            raise TrainingConfigError("input_width is outside the Stage 7-B boundary")
        if (
            not isinstance(self.conv_channels, tuple)
            or len(self.conv_channels) != 2
            or any(not _is_plain_int(item) or not 4 <= item <= 256 for item in self.conv_channels)
        ):
            raise TrainingConfigError("conv_channels must contain exactly two bounded integers")
        for name, value, lower, upper in (
            ("encoder_steps", self.encoder_steps, 4, 256),
            ("embedding_dim", self.embedding_dim, 8, 512),
            ("hidden_dim", self.hidden_dim, 16, 1024),
        ):
            if not _is_plain_int(value) or not lower <= value <= upper:
                raise TrainingConfigError(f"{name} is outside the Stage 7-B boundary")


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    master_seed: int = 7_001
    learning_rate_micros: int = 1_000
    weight_decay_micros: int = 0
    grad_clip_milli: int = 1_000
    smoke_steps: int = 2
    objective: str = "cross_entropy_ignore_pad"
    optimizer: str = "adamw"
    scheduler: str = "none"
    checkpoint_selection: str = "min_validation_loss"
    batch_policy: str = "deterministic_prebatched_v1"

    def __post_init__(self) -> None:
        if not _is_plain_int(self.master_seed) or not 0 <= self.master_seed <= _MAX_SEED:
            raise TrainingConfigError("master_seed is outside the Stage 7-B seed range")
        if (
            not _is_plain_int(self.learning_rate_micros)
            or not 1 <= self.learning_rate_micros <= 1_000_000
        ):
            raise TrainingConfigError("learning_rate_micros is outside the Stage 7-B range")
        if (
            not _is_plain_int(self.weight_decay_micros)
            or not 0 <= self.weight_decay_micros <= 1_000_000
        ):
            raise TrainingConfigError("weight_decay_micros is outside the Stage 7-B range")
        if not _is_plain_int(self.grad_clip_milli) or not 1 <= self.grad_clip_milli <= 100_000:
            raise TrainingConfigError("grad_clip_milli is outside the Stage 7-B range")
        if not _is_plain_int(self.smoke_steps) or not 1 <= self.smoke_steps <= 4:
            raise TrainingConfigError("smoke_steps must be an integer from 1 through 4")
        expected = {
            "objective": "cross_entropy_ignore_pad",
            "optimizer": "adamw",
            "scheduler": "none",
            "checkpoint_selection": "min_validation_loss",
            "batch_policy": "deterministic_prebatched_v1",
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise TrainingConfigError(f"{field_name} is frozen to {expected_value!r}")


@dataclass(frozen=True, slots=True)
class SmokeTrainingResult:
    initial_state_sha256: str
    final_state_sha256: str
    losses: tuple[float, ...]
    parameter_count: int
    model_fingerprint: str
    trainer_fingerprint: str
    torch_version: str

    def __post_init__(self) -> None:
        if self.initial_state_sha256 == self.final_state_sha256:
            raise TrainingRuntimeError("smoke training did not update the model")
        if not self.losses or any(not math.isfinite(value) for value in self.losses):
            raise TrainingRuntimeError("smoke result contains a non-finite loss")
        if not 0 < self.parameter_count <= MAX_TRAINABLE_PARAMETERS:
            raise TrainingRuntimeError("smoke result parameter count is outside the V1 ceiling")
        if self.torch_version != TORCH_PINNED_VERSION:
            raise TrainingRuntimeError("smoke result was not produced by the pinned PyTorch runtime")


class BaselineSTOMRModel(nn.Module):
    """Small CNN-conditioned GRU decoder used only as the first from-scratch baseline."""

    def __init__(self, config: BaselineModelConfig) -> None:
        super().__init__()
        if not isinstance(config, BaselineModelConfig):
            raise TypeError("config must be BaselineModelConfig")
        self.config = config
        c1, c2 = config.conv_channels
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.sequence_pool = nn.AdaptiveAvgPool2d((1, config.encoder_steps))
        self.context_projection = nn.Linear(c2, config.hidden_dim)
        self.token_embedding = nn.Embedding(
            VOCABULARY_SIZE,
            config.embedding_dim,
            padding_idx=PAD_TOKEN_ID,
        )
        self.decoder = nn.GRU(
            input_size=config.embedding_dim + config.hidden_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
        )
        self.output_projection = nn.Linear(config.hidden_dim, VOCABULARY_SIZE)

    def forward(self, images: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(images, torch.Tensor) or not isinstance(decoder_input_ids, torch.Tensor):
            raise TrainingRuntimeError("model inputs must be torch tensors")
        if images.dtype != torch.float32 or images.ndim != 4:
            raise TrainingRuntimeError("images must be float32 [batch, channel, height, width]")
        if (
            images.shape[1] != 1
            or images.shape[2] != self.config.input_height
            or images.shape[3] != self.config.input_width
        ):
            raise TrainingRuntimeError("image tensor shape differs from the frozen model config")
        if decoder_input_ids.dtype != torch.long or decoder_input_ids.ndim != 2:
            raise TrainingRuntimeError("decoder input ids must be rank-2 torch.long")
        if decoder_input_ids.shape[0] != images.shape[0] or decoder_input_ids.shape[1] < 1:
            raise TrainingRuntimeError("decoder batch shape does not match image batch")
        if bool((decoder_input_ids < 0).any()) or bool((decoder_input_ids >= VOCABULARY_SIZE).any()):
            raise TrainingRuntimeError("decoder input id is outside the frozen vocabulary")
        assert_finite_tensor("model input images", images)

        features = self.visual_encoder(images)
        encoded = self.sequence_pool(features).squeeze(2).transpose(1, 2)
        context = encoded.mean(dim=1)
        hidden0 = torch.tanh(self.context_projection(context)).unsqueeze(0)
        embedded = self.token_embedding(decoder_input_ids)
        conditioned = torch.cat(
            (embedded, hidden0.squeeze(0).unsqueeze(1).expand(-1, embedded.shape[1], -1)),
            dim=-1,
        )
        decoded, _hidden = self.decoder(conditioned, hidden0)
        logits = self.output_projection(decoded)
        assert_finite_tensor("model logits", logits)
        return logits


def verify_torch_runtime() -> str:
    actual = torch.__version__
    if actual != TORCH_PINNED_VERSION:
        raise TrainingRuntimeError(
            f"PyTorch runtime mismatch: expected {TORCH_PINNED_VERSION}, got {actual}"
        )
    return actual


def set_deterministic_cpu(seed: int) -> None:
    if not _is_plain_int(seed) or not 0 <= seed <= _MAX_SEED:
        raise TrainingConfigError("seed is outside the Stage 7-B range")
    verify_torch_runtime()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def count_trainable_parameters(model: nn.Module) -> int:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_config_fingerprint(config: BaselineModelConfig) -> str:
    if not isinstance(config, BaselineModelConfig):
        raise TypeError("config must be BaselineModelConfig")
    payload = {
        "model_version": BASELINE_MODEL_VERSION,
        "torch_version": TORCH_PINNED_VERSION,
        "vocabulary_size": VOCABULARY_SIZE,
        "config": asdict(config),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def trainer_config_fingerprint(config: TrainerConfig) -> str:
    if not isinstance(config, TrainerConfig):
        raise TypeError("config must be TrainerConfig")
    payload = {
        "trainer_version": TRAINER_VERSION,
        "torch_version": TORCH_PINNED_VERSION,
        "config": asdict(config),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def build_baseline_model(
    config: BaselineModelConfig = BaselineModelConfig(),
    *,
    seed: int = 7_001,
) -> BaselineSTOMRModel:
    set_deterministic_cpu(seed)
    model = BaselineSTOMRModel(config).cpu()
    parameter_count = count_trainable_parameters(model)
    if parameter_count <= 0 or parameter_count > MAX_TRAINABLE_PARAMETERS:
        raise TrainingRuntimeError(
            f"trainable parameter count {parameter_count} violates the V1 ceiling"
        )
    assert_model_finite(model)
    return model


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be torch.Tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise TrainingRuntimeError(f"{name} contains NaN or Infinity")


def assert_model_finite(model: nn.Module) -> None:
    for name, value in model.state_dict().items():
        if torch.is_floating_point(value) or torch.is_complex(value):
            assert_finite_tensor(f"model state {name}", value)


def assert_optimizer_finite(optimizer: torch.optim.Optimizer) -> None:
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer")
    for parameter, state in optimizer.state.items():
        if not isinstance(parameter, torch.Tensor):
            raise TrainingRuntimeError("optimizer state key is not a tensor")
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and (torch.is_floating_point(value) or torch.is_complex(value)):
                assert_finite_tensor(f"optimizer state {key}", value)
            elif isinstance(value, float) and not math.isfinite(value):
                raise TrainingRuntimeError(f"optimizer scalar state {key} is non-finite")


def model_state_sha256(model: nn.Module) -> str:
    """Hash sorted state names, metadata, and exact CPU tensor bytes."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        raw = value.view(torch.uint8).flatten().tolist()
        digest.update(bytes(raw))
        digest.update(b"\0")
    return digest.hexdigest()


def _new_optimizer(model: nn.Module, config: TrainerConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate_micros / 1_000_000,
        weight_decay=config.weight_decay_micros / 1_000_000,
        foreach=False,
        fused=False,
    )


def _masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.dtype != torch.long or labels.shape != logits.shape[:2]:
        raise TrainingRuntimeError("labels do not align with model logits")
    if not bool((labels != PAD_TOKEN_ID).any()):
        raise TrainingRuntimeError("loss has no unmasked target positions")
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=PAD_TOKEN_ID,
    )
    assert_finite_tensor("training loss", loss)
    return loss


def train_one_smoke_step(
    model: BaselineSTOMRModel,
    batch: TrainingBatch,
    optimizer: torch.optim.Optimizer,
    config: TrainerConfig,
) -> float:
    if not isinstance(batch, TrainingBatch) or batch.split is not DatasetSplit.TRAIN:
        raise TrainingRuntimeError("gradient updates are allowed only from the train split")
    if not isinstance(config, TrainerConfig):
        raise TypeError("config must be TrainerConfig")
    verify_torch_runtime()
    assert_model_finite(model)
    assert_optimizer_finite(optimizer)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch.images.cpu(), batch.decoder_input_ids.cpu())
    loss = _masked_cross_entropy(logits, batch.labels.cpu())
    loss.backward()

    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert_finite_tensor(f"gradient {name}", parameter.grad)

    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=config.grad_clip_milli / 1_000,
        error_if_nonfinite=True,
    )
    if not math.isfinite(float(total_norm)):
        raise TrainingRuntimeError("gradient norm is NaN or Infinity")

    optimizer.step()
    assert_model_finite(model)
    assert_optimizer_finite(optimizer)
    value = float(loss.detach().cpu().item())
    if not math.isfinite(value):
        raise TrainingRuntimeError("reported training loss is NaN or Infinity")
    return value


def validation_loss(model: BaselineSTOMRModel, batch: TrainingBatch) -> float:
    if not isinstance(batch, TrainingBatch) or batch.split is not DatasetSplit.VALIDATION:
        raise TrainingRuntimeError("validation metrics require the validation split")
    before = model_state_sha256(model)
    model.eval()
    with torch.no_grad():
        logits = model(batch.images.cpu(), batch.decoder_input_ids.cpu())
        loss = _masked_cross_entropy(logits, batch.labels.cpu())
    after = model_state_sha256(model)
    if before != after:
        raise TrainingRuntimeError("validation path mutated model state")
    value = float(loss.detach().cpu().item())
    if not math.isfinite(value):
        raise TrainingRuntimeError("reported validation loss is NaN or Infinity")
    return value


def run_deterministic_cpu_smoke(
    batch: TrainingBatch,
    model_config: BaselineModelConfig = BaselineModelConfig(),
    trainer_config: TrainerConfig = TrainerConfig(),
) -> SmokeTrainingResult:
    if not isinstance(batch, TrainingBatch) or batch.split is not DatasetSplit.TRAIN:
        raise TrainingRuntimeError("CPU smoke training requires a train batch")
    set_deterministic_cpu(trainer_config.master_seed)
    model = build_baseline_model(model_config, seed=trainer_config.master_seed)
    optimizer = _new_optimizer(model, trainer_config)
    initial = model_state_sha256(model)
    losses: list[float] = []
    for _ in range(trainer_config.smoke_steps):
        losses.append(train_one_smoke_step(model, batch, optimizer, trainer_config))
    final = model_state_sha256(model)
    return SmokeTrainingResult(
        initial_state_sha256=initial,
        final_state_sha256=final,
        losses=tuple(losses),
        parameter_count=count_trainable_parameters(model),
        model_fingerprint=model_config_fingerprint(model_config),
        trainer_fingerprint=trainer_config_fingerprint(trainer_config),
        torch_version=verify_torch_runtime(),
    )
