"""Deterministic TRAIN-only quality-training regime for the Polyphonic V2 2D model.

TR-POLY-09B4 is additive to the frozen TR-POLY-08A smoke harness. It reuses the
already-hardened one-step optimizer/loss primitive but permits a separately
versioned multi-epoch training plan. VALIDATION remains read-only and TEST is
never admitted. Checkpoint persistence is intentionally owned by a later gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
from typing import Final

import torch

from .dataset_manifest import DatasetSplit
from .poly_2d_training import (
    Poly2DTrainingBatch,
    Poly2DTrainingConfig,
    Poly2DTrainingError,
    build_poly_2d_optimizer,
    evaluate_poly_2d_validation_loss,
    poly_2d_trainer_fingerprint,
    train_poly_2d_one_step,
)
from .poly_2d_transformer import (
    FROZEN_POLY_2D_CONFIG,
    Poly2DTransformerConfig,
    TinyPoly2DTransformer,
    build_tiny_poly_2d_transformer,
    poly_2d_config_fingerprint,
)
from .polyphonic_representation import POLYPHONIC_REPRESENTATION_VERSION
from .polyphonic_serialization import (
    POLYPHONIC_TOKENIZER_VERSION,
    tokenizer_fingerprint,
)
from .training_model import TORCH_PINNED_VERSION, model_state_sha256


POLY_2D_QUALITY_TRAINER_VERSION: Final[str] = "st-omr-poly-2d-quality-trainer-v1"
POLY_2D_QUALITY_PROVENANCE_VERSION: Final[str] = (
    "st-omr-poly-2d-quality-training-provenance-v1"
)
POLY_2D_QUALITY_PLAN_VERSION: Final[str] = "st-omr-poly-2d-quality-training-plan-v1"
MAX_QUALITY_EPOCHS: Final[int] = 128
MAX_QUALITY_OPTIMIZER_STEPS: Final[int] = 100_000
MAX_QUALITY_BATCHES_PER_SPLIT: Final[int] = 100_000
_MAX_SEED: Final[int] = 2**63 - 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA40 = re.compile(r"^[0-9a-f]{40}$")


class Poly2DQualityTrainingError(Poly2DTrainingError):
    """Raised when the quality-training contract fails closed."""


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Poly2DQualityTrainingError(
            "quality-training evidence is not canonical-JSON serializable"
        ) from exc


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Poly2DQualityTrainingError(f"{name} must be lowercase SHA-256 text")
    return value


def _require_git_sha(name: str, value: object) -> str:
    if not isinstance(value, str) or _GIT_SHA40.fullmatch(value) is None:
        raise Poly2DQualityTrainingError(f"{name} must be lowercase git SHA-40 text")
    return value


@dataclass(frozen=True, slots=True)
class Poly2DQualityTrainingConfig:
    """Versioned bounded training recipe; exact values are fingerprinted per run."""

    master_seed: int = 82_081
    learning_rate_micros: int = 500
    weight_decay_micros: int = 100
    grad_clip_milli: int = 1_000
    epochs: int = 8
    max_optimizer_steps: int = 8_192
    validation_interval_epochs: int = 1
    objective: str = "cross_entropy_ignore_v2_pad"
    optimizer: str = "adamw"
    scheduler: str = "none"
    batch_order_policy: str = "caller-frozen-tuple-order-v1"
    validation_policy: str = "full-read-only-validation-each-epoch-v1"
    checkpoint_selection: str = "min-mean-validation-loss-earliest-epoch-v1"

    def __post_init__(self) -> None:
        bounds = {
            "master_seed": (self.master_seed, 0, _MAX_SEED),
            "learning_rate_micros": (self.learning_rate_micros, 1, 100_000),
            "weight_decay_micros": (self.weight_decay_micros, 0, 100_000),
            "grad_clip_milli": (self.grad_clip_milli, 1, 100_000),
            "epochs": (self.epochs, 1, MAX_QUALITY_EPOCHS),
            "max_optimizer_steps": (
                self.max_optimizer_steps,
                1,
                MAX_QUALITY_OPTIMIZER_STEPS,
            ),
            "validation_interval_epochs": (
                self.validation_interval_epochs,
                1,
                1,
            ),
        }
        for name, (value, lower, upper) in bounds.items():
            if not _plain_int(value) or not lower <= value <= upper:
                raise ValueError(f"{name} is outside the TR-POLY-09B4 boundary")
        expected = {
            "objective": "cross_entropy_ignore_v2_pad",
            "optimizer": "adamw",
            "scheduler": "none",
            "batch_order_policy": "caller-frozen-tuple-order-v1",
            "validation_policy": "full-read-only-validation-each-epoch-v1",
            "checkpoint_selection": "min-mean-validation-loss-earliest-epoch-v1",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} is frozen to {value!r}")


FROZEN_POLY_2D_QUALITY_CONFIG: Final[Poly2DQualityTrainingConfig] = (
    Poly2DQualityTrainingConfig()
)


def quality_step_config(
    config: Poly2DQualityTrainingConfig,
) -> Poly2DTrainingConfig:
    """Map the quality recipe to the frozen one-step optimizer/loss primitive."""

    if not isinstance(config, Poly2DQualityTrainingConfig):
        raise TypeError("config must be Poly2DQualityTrainingConfig")
    return Poly2DTrainingConfig(
        master_seed=config.master_seed,
        learning_rate_micros=config.learning_rate_micros,
        weight_decay_micros=config.weight_decay_micros,
        grad_clip_milli=config.grad_clip_milli,
        smoke_steps=1,
    )


def poly_2d_quality_trainer_fingerprint(
    config: Poly2DQualityTrainingConfig = FROZEN_POLY_2D_QUALITY_CONFIG,
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
) -> str:
    if not isinstance(config, Poly2DQualityTrainingConfig):
        raise TypeError("config must be Poly2DQualityTrainingConfig")
    if not isinstance(model_config, Poly2DTransformerConfig):
        raise TypeError("model_config must be Poly2DTransformerConfig")
    step_config = quality_step_config(config)
    payload = {
        "quality_trainer_version": POLY_2D_QUALITY_TRAINER_VERSION,
        "quality_plan_version": POLY_2D_QUALITY_PLAN_VERSION,
        "torch_version": TORCH_PINNED_VERSION,
        "representation_version": POLYPHONIC_REPRESENTATION_VERSION,
        "tokenizer_version": POLYPHONIC_TOKENIZER_VERSION,
        "tokenizer_fingerprint_sha256": tokenizer_fingerprint(),
        "model_profile_sha256": poly_2d_config_fingerprint(model_config),
        "step_primitive_profile_sha256": poly_2d_trainer_fingerprint(
            step_config, model_config
        ),
        "config": asdict(config),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class Poly2DQualityTrainingProvenance:
    repository_sha: str
    dataset_manifest_sha256: str
    preprocess_fingerprint_sha256: str
    model_profile_sha256: str
    quality_trainer_profile_sha256: str
    tokenizer_fingerprint_sha256: str
    representation_version: str = POLYPHONIC_REPRESENTATION_VERSION
    tokenizer_version: str = POLYPHONIC_TOKENIZER_VERSION
    torch_version: str = TORCH_PINNED_VERSION
    provenance_version: str = POLY_2D_QUALITY_PROVENANCE_VERSION

    def __post_init__(self) -> None:
        _require_git_sha("repository_sha", self.repository_sha)
        for name in (
            "dataset_manifest_sha256",
            "preprocess_fingerprint_sha256",
            "model_profile_sha256",
            "quality_trainer_profile_sha256",
            "tokenizer_fingerprint_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.tokenizer_fingerprint_sha256 != tokenizer_fingerprint():
            raise Poly2DQualityTrainingError("quality provenance tokenizer mismatch")
        if self.representation_version != POLYPHONIC_REPRESENTATION_VERSION:
            raise Poly2DQualityTrainingError("quality provenance representation mismatch")
        if self.tokenizer_version != POLYPHONIC_TOKENIZER_VERSION:
            raise Poly2DQualityTrainingError("quality provenance tokenizer version mismatch")
        if self.torch_version != TORCH_PINNED_VERSION:
            raise Poly2DQualityTrainingError("quality provenance torch version mismatch")
        if self.provenance_version != POLY_2D_QUALITY_PROVENANCE_VERSION:
            raise Poly2DQualityTrainingError("quality provenance version mismatch")

    def fingerprint(self) -> str:
        return sha256(_canonical_json_bytes(asdict(self))).hexdigest()


def build_poly_2d_quality_training_provenance(
    *,
    repository_sha: str,
    dataset_manifest_sha256: str,
    preprocess_fingerprint_sha256: str,
    quality_config: Poly2DQualityTrainingConfig = FROZEN_POLY_2D_QUALITY_CONFIG,
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
) -> Poly2DQualityTrainingProvenance:
    return Poly2DQualityTrainingProvenance(
        repository_sha=repository_sha,
        dataset_manifest_sha256=dataset_manifest_sha256,
        preprocess_fingerprint_sha256=preprocess_fingerprint_sha256,
        model_profile_sha256=poly_2d_config_fingerprint(model_config),
        quality_trainer_profile_sha256=poly_2d_quality_trainer_fingerprint(
            quality_config, model_config
        ),
        tokenizer_fingerprint_sha256=tokenizer_fingerprint(),
    )


def _batch_plan_payload(batch: Poly2DTrainingBatch) -> dict[str, object]:
    return {
        "split": batch.split.value,
        "sample_ids": list(batch.sample_ids),
        "dataset_manifest_sha256": batch.dataset_manifest_sha256,
        "image_shape": list(batch.images.shape),
        "target_shape": list(batch.decoder_input_ids.shape),
    }


def quality_training_plan_fingerprint(
    train_batches: tuple[Poly2DTrainingBatch, ...],
    validation_batches: tuple[Poly2DTrainingBatch, ...],
) -> str:
    if not isinstance(train_batches, tuple) or not train_batches:
        raise Poly2DQualityTrainingError("train_batches must be a non-empty tuple")
    if not isinstance(validation_batches, tuple) or not validation_batches:
        raise Poly2DQualityTrainingError(
            "validation_batches must be a non-empty tuple"
        )
    payload = {
        "plan_version": POLY_2D_QUALITY_PLAN_VERSION,
        "train": [_batch_plan_payload(batch) for batch in train_batches],
        "validation": [_batch_plan_payload(batch) for batch in validation_batches],
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class Poly2DQualityEpochEvidence:
    epoch: int
    optimizer_steps_cumulative: int
    train_mean_loss: float
    validation_mean_loss: float
    model_state_sha256: str

    def __post_init__(self) -> None:
        if not _plain_int(self.epoch) or self.epoch < 1:
            raise Poly2DQualityTrainingError("epoch must be a positive integer")
        if (
            not _plain_int(self.optimizer_steps_cumulative)
            or self.optimizer_steps_cumulative < 1
        ):
            raise Poly2DQualityTrainingError(
                "optimizer_steps_cumulative must be positive"
            )
        for name in ("train_mean_loss", "validation_mean_loss"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise Poly2DQualityTrainingError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise Poly2DQualityTrainingError(f"{name} must be finite non-negative")
        _require_sha256("model_state_sha256", self.model_state_sha256)


@dataclass(frozen=True, slots=True)
class Poly2DQualityTrainingResult:
    initial_state_sha256: str
    selected_state_sha256: str
    selected_epoch: int
    selected_validation_loss: float
    optimizer_steps: int
    epoch_evidence: tuple[Poly2DQualityEpochEvidence, ...]
    training_plan_sha256: str
    model_profile_sha256: str
    quality_trainer_profile_sha256: str
    step_primitive_profile_sha256: str
    provenance_sha256: str
    dataset_manifest_sha256: str
    tokenizer_fingerprint_sha256: str
    repository_sha: str
    torch_version: str = TORCH_PINNED_VERSION
    test_split_accessed: bool = False
    production_authority: bool = False

    def __post_init__(self) -> None:
        for name in (
            "initial_state_sha256",
            "selected_state_sha256",
            "training_plan_sha256",
            "model_profile_sha256",
            "quality_trainer_profile_sha256",
            "step_primitive_profile_sha256",
            "provenance_sha256",
            "dataset_manifest_sha256",
            "tokenizer_fingerprint_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_git_sha("repository_sha", self.repository_sha)
        if self.initial_state_sha256 == self.selected_state_sha256:
            raise Poly2DQualityTrainingError("quality training did not update model state")
        if not _plain_int(self.selected_epoch) or not 1 <= self.selected_epoch <= len(
            self.epoch_evidence
        ):
            raise Poly2DQualityTrainingError("selected_epoch is outside evidence")
        if not math.isfinite(float(self.selected_validation_loss)):
            raise Poly2DQualityTrainingError("selected_validation_loss must be finite")
        if not _plain_int(self.optimizer_steps) or not 1 <= self.optimizer_steps <= MAX_QUALITY_OPTIMIZER_STEPS:
            raise Poly2DQualityTrainingError("optimizer_steps outside quality boundary")
        if not isinstance(self.epoch_evidence, tuple) or not self.epoch_evidence:
            raise Poly2DQualityTrainingError("epoch_evidence must be non-empty")
        if any(
            not isinstance(item, Poly2DQualityEpochEvidence)
            for item in self.epoch_evidence
        ):
            raise Poly2DQualityTrainingError("invalid epoch evidence")
        if tuple(item.epoch for item in self.epoch_evidence) != tuple(
            range(1, len(self.epoch_evidence) + 1)
        ):
            raise Poly2DQualityTrainingError("epoch evidence must be sequential")
        selected = self.epoch_evidence[self.selected_epoch - 1]
        if selected.model_state_sha256 != self.selected_state_sha256:
            raise Poly2DQualityTrainingError("selected state does not match selected epoch")
        if selected.validation_mean_loss != self.selected_validation_loss:
            raise Poly2DQualityTrainingError("selected validation loss mismatch")
        if self.optimizer_steps != self.epoch_evidence[-1].optimizer_steps_cumulative:
            raise Poly2DQualityTrainingError("optimizer step count differs from epoch evidence")
        if self.tokenizer_fingerprint_sha256 != tokenizer_fingerprint():
            raise Poly2DQualityTrainingError("quality result tokenizer mismatch")
        if self.torch_version != TORCH_PINNED_VERSION:
            raise Poly2DQualityTrainingError("quality result torch version mismatch")
        if self.test_split_accessed or self.production_authority:
            raise Poly2DQualityTrainingError(
                "quality training may not claim TEST access or production authority"
            )

    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["epoch_evidence"] = [asdict(item) for item in self.epoch_evidence]
        return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(slots=True)
class Poly2DQualityTrainingRun:
    model: TinyPoly2DTransformer
    result: Poly2DQualityTrainingResult

    def __post_init__(self) -> None:
        if not isinstance(self.model, TinyPoly2DTransformer):
            raise TypeError("model must be TinyPoly2DTransformer")
        if not isinstance(self.result, Poly2DQualityTrainingResult):
            raise TypeError("result must be Poly2DQualityTrainingResult")
        if model_state_sha256(self.model) != self.result.selected_state_sha256:
            raise Poly2DQualityTrainingError(
                "returned model state differs from selected quality state"
            )


def _validate_quality_batches(
    train_batches: object,
    validation_batches: object,
    provenance: Poly2DQualityTrainingProvenance,
    quality_config: Poly2DQualityTrainingConfig,
    model_config: Poly2DTransformerConfig,
) -> tuple[tuple[Poly2DTrainingBatch, ...], tuple[Poly2DTrainingBatch, ...]]:
    if not isinstance(train_batches, tuple) or not train_batches:
        raise Poly2DQualityTrainingError("train_batches must be a non-empty tuple")
    if not isinstance(validation_batches, tuple) or not validation_batches:
        raise Poly2DQualityTrainingError(
            "validation_batches must be a non-empty tuple"
        )
    if len(train_batches) > MAX_QUALITY_BATCHES_PER_SPLIT or len(
        validation_batches
    ) > MAX_QUALITY_BATCHES_PER_SPLIT:
        raise Poly2DQualityTrainingError("quality batch count exceeds bounded contract")
    if any(
        not isinstance(batch, Poly2DTrainingBatch)
        or batch.split is not DatasetSplit.TRAIN
        for batch in train_batches
    ):
        raise Poly2DQualityTrainingError("quality train_batches may contain TRAIN only")
    if any(
        not isinstance(batch, Poly2DTrainingBatch)
        or batch.split is not DatasetSplit.VALIDATION
        for batch in validation_batches
    ):
        raise Poly2DQualityTrainingError(
            "quality validation_batches may contain VALIDATION only"
        )
    expected_steps = quality_config.epochs * len(train_batches)
    if expected_steps > quality_config.max_optimizer_steps:
        raise Poly2DQualityTrainingError(
            "quality plan exceeds max_optimizer_steps before training starts"
        )
    all_batches = train_batches + validation_batches
    if any(
        batch.dataset_manifest_sha256 != provenance.dataset_manifest_sha256
        for batch in all_batches
    ):
        raise Poly2DQualityTrainingError(
            "quality batch dataset identity differs from provenance"
        )
    sample_ids = [sample for batch in all_batches for sample in batch.sample_ids]
    if len(set(sample_ids)) != len(sample_ids):
        raise Poly2DQualityTrainingError(
            "TRAIN/VALIDATION quality plan contains duplicate sample IDs"
        )
    expected_model = poly_2d_config_fingerprint(model_config)
    expected_trainer = poly_2d_quality_trainer_fingerprint(
        quality_config, model_config
    )
    if provenance.model_profile_sha256 != expected_model:
        raise Poly2DQualityTrainingError("quality provenance model profile mismatch")
    if provenance.quality_trainer_profile_sha256 != expected_trainer:
        raise Poly2DQualityTrainingError("quality provenance trainer profile mismatch")
    return train_batches, validation_batches


def _clone_state_dict(model: TinyPoly2DTransformer) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def run_poly_2d_quality_training(
    *,
    train_batches: tuple[Poly2DTrainingBatch, ...],
    validation_batches: tuple[Poly2DTrainingBatch, ...],
    provenance: Poly2DQualityTrainingProvenance,
    quality_config: Poly2DQualityTrainingConfig = FROZEN_POLY_2D_QUALITY_CONFIG,
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
) -> Poly2DQualityTrainingRun:
    """Run a deterministic fixed-order multi-epoch TRAIN/VALIDATION experiment."""

    if not isinstance(provenance, Poly2DQualityTrainingProvenance):
        raise TypeError("provenance must be Poly2DQualityTrainingProvenance")
    if not isinstance(quality_config, Poly2DQualityTrainingConfig):
        raise TypeError("quality_config must be Poly2DQualityTrainingConfig")
    if not isinstance(model_config, Poly2DTransformerConfig):
        raise TypeError("model_config must be Poly2DTransformerConfig")
    train_batches, validation_batches = _validate_quality_batches(
        train_batches,
        validation_batches,
        provenance,
        quality_config,
        model_config,
    )
    plan_sha = quality_training_plan_fingerprint(train_batches, validation_batches)
    step_config = quality_step_config(quality_config)
    step_profile = poly_2d_trainer_fingerprint(step_config, model_config)
    model = build_tiny_poly_2d_transformer(
        model_config, seed=quality_config.master_seed
    )
    optimizer = build_poly_2d_optimizer(model, step_config)
    initial_state = model_state_sha256(model)

    evidence: list[Poly2DQualityEpochEvidence] = []
    optimizer_steps = 0
    best_validation = float("inf")
    selected_epoch = 0
    selected_state: dict[str, torch.Tensor] | None = None
    selected_state_sha = ""

    for epoch in range(1, quality_config.epochs + 1):
        train_losses: list[float] = []
        for batch in train_batches:
            train_losses.append(
                train_poly_2d_one_step(model, batch, optimizer, step_config)
            )
            optimizer_steps += 1
            if optimizer_steps > quality_config.max_optimizer_steps:
                raise Poly2DQualityTrainingError(
                    "optimizer step bound exceeded during quality training"
                )

        validation_losses = [
            evaluate_poly_2d_validation_loss(model, batch)
            for batch in validation_batches
        ]
        train_mean = sum(train_losses) / len(train_losses)
        validation_mean = sum(validation_losses) / len(validation_losses)
        if not math.isfinite(train_mean) or not math.isfinite(validation_mean):
            raise Poly2DQualityTrainingError("quality epoch produced non-finite loss")
        state_sha = model_state_sha256(model)
        evidence.append(
            Poly2DQualityEpochEvidence(
                epoch=epoch,
                optimizer_steps_cumulative=optimizer_steps,
                train_mean_loss=train_mean,
                validation_mean_loss=validation_mean,
                model_state_sha256=state_sha,
            )
        )
        if validation_mean < best_validation:
            best_validation = validation_mean
            selected_epoch = epoch
            selected_state = _clone_state_dict(model)
            selected_state_sha = state_sha

    if selected_state is None or selected_epoch == 0:
        raise Poly2DQualityTrainingError("quality training selected no candidate state")
    model.load_state_dict(selected_state, strict=True)
    if model_state_sha256(model) != selected_state_sha:
        raise Poly2DQualityTrainingError("selected quality state failed exact reload")

    result = Poly2DQualityTrainingResult(
        initial_state_sha256=initial_state,
        selected_state_sha256=selected_state_sha,
        selected_epoch=selected_epoch,
        selected_validation_loss=best_validation,
        optimizer_steps=optimizer_steps,
        epoch_evidence=tuple(evidence),
        training_plan_sha256=plan_sha,
        model_profile_sha256=poly_2d_config_fingerprint(model_config),
        quality_trainer_profile_sha256=poly_2d_quality_trainer_fingerprint(
            quality_config, model_config
        ),
        step_primitive_profile_sha256=step_profile,
        provenance_sha256=provenance.fingerprint(),
        dataset_manifest_sha256=provenance.dataset_manifest_sha256,
        tokenizer_fingerprint_sha256=tokenizer_fingerprint(),
        repository_sha=provenance.repository_sha,
    )
    return Poly2DQualityTrainingRun(model=model, result=result)
