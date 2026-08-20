"""Shadow-only Meter V3-A1 classification recovery.

V3-A1 isolates classification from localization. The exact D11 Meter model is
fully frozen; a small glyph-zone adapter may change only class logits. Bbox
outputs are always the untouched D11 boxes. D10 replay receives both logit
distillation and an explicit residual-zero penalty so the adapter is pushed to
leave the accepted source decision surface unchanged.

This module intentionally does not implement V3-A2 positive-class margin loss.
TEST, runtime, Resolver, checkpoint replacement, and production promotion stay
closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

from .meter_real_domain_adaptation_v1 import MeterEvaluationV1, _canonical_json, _hex64, _sha
from .meter_real_domain_adaptation_v2 import AdaptationGateDecisionV2, adaptation_acceptance_v2
from .meter_real_domain_adaptation_v1 import PRESENCE_D11_SHA256


METER_REAL_DOMAIN_ADAPTATION_V3_A1: Final[str] = "meter-real-domain-adaptation-v3-a1"


class MeterRealDomainAdaptationV3A1Error(RuntimeError):
    """Raised when the bounded V3-A1 contract is violated."""


@dataclass(frozen=True, slots=True)
class MeterRealDomainAdaptationConfigV3A1:
    batch_size: int = 32
    epochs: int = 20
    learning_rate_micros: int = 750
    weight_decay_micros: int = 100
    grad_clip_milli: int = 1_000
    master_seed: int = 812_031
    real_balanced_repeat_factor: int = 4
    synthetic_replay_per_class: int = 128
    glyph_x0: int = 56
    glyph_x1: int = 248
    glyph_y0: int = 8
    glyph_y1: int = 184
    presence_loss_milli: int = 500
    digit_loss_milli: int = 750
    distillation_loss_milli: int = 2_000
    residual_zero_loss_milli: int = 5_000
    distillation_temperature_milli: int = 2_000
    augmentation_shift_px: int = 4
    trainable_surface: str = "classification-adapter-only-d11-and-bbox-fully-frozen"
    objective: str = "real-classification-plus-d10-logit-distillation-and-residual-zero-v3-a1"
    real_min_macro_f1_milli: int = 900
    real_min_accuracy_milli: int = 900
    real_min_none_recall_milli: int = 888
    real_min_positive_class_recall_milli: int = 999
    synthetic_max_macro_f1_drop_milli: int = 20
    synthetic_max_localization_drop_milli: int = 30

    def __post_init__(self) -> None:
        bounds = {
            "batch_size": (self.batch_size, 1, 64),
            "epochs": (self.epochs, 1, 40),
            "learning_rate_micros": (self.learning_rate_micros, 1, 100_000),
            "weight_decay_micros": (self.weight_decay_micros, 0, 100_000),
            "grad_clip_milli": (self.grad_clip_milli, 1, 100_000),
            "master_seed": (self.master_seed, 0, 2**63 - 1),
            "real_balanced_repeat_factor": (self.real_balanced_repeat_factor, 1, 32),
            "synthetic_replay_per_class": (self.synthetic_replay_per_class, 1, 1024),
            "augmentation_shift_px": (self.augmentation_shift_px, 0, 16),
        }
        for name, (value, low, high) in bounds.items():
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ValueError(f"{name} is outside Meter V3-A1 bounds")
        if not (0 <= self.glyph_x0 < self.glyph_x1 <= 256):
            raise ValueError("V3-A1 glyph x window must stay inside the 256px ROI")
        if not (0 <= self.glyph_y0 < self.glyph_y1 <= 192):
            raise ValueError("V3-A1 glyph y window must stay inside the 192px ROI")
        for name in (
            "presence_loss_milli",
            "digit_loss_milli",
            "distillation_loss_milli",
            "residual_zero_loss_milli",
            "distillation_temperature_milli",
            "real_min_macro_f1_milli",
            "real_min_accuracy_milli",
            "real_min_none_recall_milli",
            "real_min_positive_class_recall_milli",
            "synthetic_max_macro_f1_drop_milli",
            "synthetic_max_localization_drop_milli",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 10_000:
                raise ValueError(f"{name} must be bounded integer milli-units")
        frozen = {
            "trainable_surface": "classification-adapter-only-d11-and-bbox-fully-frozen",
            "objective": "real-classification-plus-d10-logit-distillation-and-residual-zero-v3-a1",
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} is frozen to {expected!r}")


FROZEN_ADAPTATION_CONFIG_V3_A1: Final[MeterRealDomainAdaptationConfigV3A1] = (
    MeterRealDomainAdaptationConfigV3A1()
)


def adaptation_acceptance_v3_a1(
    *,
    candidate_real: MeterEvaluationV1,
    baseline_synthetic: MeterEvaluationV1,
    candidate_synthetic: MeterEvaluationV1,
    config: MeterRealDomainAdaptationConfigV3A1 = FROZEN_ADAPTATION_CONFIG_V3_A1,
) -> AdaptationGateDecisionV2:
    """Apply the unchanged V2 quality and retention gate."""
    from .meter_real_domain_adaptation_v2 import MeterRealDomainAdaptationConfigV2

    gate_config = MeterRealDomainAdaptationConfigV2(
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate_micros=1_000,
        weight_decay_micros=config.weight_decay_micros,
        grad_clip_milli=config.grad_clip_milli,
        master_seed=812_021,
        real_balanced_repeat_factor=config.real_balanced_repeat_factor,
        synthetic_replay_per_class=config.synthetic_replay_per_class,
        glyph_x0=config.glyph_x0,
        glyph_x1=config.glyph_x1,
        glyph_y0=config.glyph_y0,
        glyph_y1=config.glyph_y1,
        bbox_max_delta_milli=120,
        presence_loss_milli=500,
        digit_loss_milli=750,
        bbox_loss_milli=1_000,
        distillation_loss_milli=2_000,
        bbox_anchor_loss_milli=2_000,
        distillation_temperature_milli=2_000,
        augmentation_shift_px=config.augmentation_shift_px,
        trainable_surface="glyph-adapter-only-d11-fully-frozen",
        objective="hierarchical-presence-digit-plus-synthetic-distillation-v2",
        checkpoint_selection="all-gates-then-real-macro-f1-then-synthetic-macro-f1",
        real_min_macro_f1_milli=config.real_min_macro_f1_milli,
        real_min_accuracy_milli=config.real_min_accuracy_milli,
        real_min_none_recall_milli=config.real_min_none_recall_milli,
        real_min_positive_class_recall_milli=config.real_min_positive_class_recall_milli,
        synthetic_max_macro_f1_drop_milli=config.synthetic_max_macro_f1_drop_milli,
        synthetic_max_localization_drop_milli=config.synthetic_max_localization_drop_milli,
    )
    return adaptation_acceptance_v2(
        candidate_real=candidate_real,
        baseline_synthetic=baseline_synthetic,
        candidate_synthetic=candidate_synthetic,
        config=gate_config,
    )


def meter_real_domain_adaptation_fingerprint_v3_a1(
    *,
    teacher_manifest_sha256: str,
    d10_manifest_sha256: str,
    d10_artifact_binding_sha256: str,
    config: MeterRealDomainAdaptationConfigV3A1 = FROZEN_ADAPTATION_CONFIG_V3_A1,
) -> str:
    return _sha(
        _canonical_json(
            {
                "version": METER_REAL_DOMAIN_ADAPTATION_V3_A1,
                "base_checkpoint_sha256": PRESENCE_D11_SHA256,
                "teacher_manifest_sha256": _hex64(
                    "teacher_manifest_sha256", teacher_manifest_sha256
                ),
                "d10_manifest_sha256": _hex64("d10_manifest_sha256", d10_manifest_sha256),
                "d10_artifact_binding_sha256": _hex64(
                    "d10_artifact_binding_sha256", d10_artifact_binding_sha256
                ),
                "config": asdict(config),
            }
        )
    )


def build_meter_classification_adapter_v3_a1(
    base_model,
    config: MeterRealDomainAdaptationConfigV3A1 = FROZEN_ADAPTATION_CONFIG_V3_A1,
):
    """Wrap frozen D11 with a zero-initialized class-only residual adapter."""
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise MeterRealDomainAdaptationV3A1Error("torch is required for V3-A1") from exc
    if config != FROZEN_ADAPTATION_CONFIG_V3_A1:
        raise MeterRealDomainAdaptationV3A1Error("V3-A1 requires the frozen configuration")

    class MeterClassificationAdapterV3A1(nn.Module):
        def __init__(self, frozen_base) -> None:
            super().__init__()
            self.base = frozen_base
            for parameter in self.base.parameters():
                parameter.requires_grad = False
            self.glyph_encoder = nn.Sequential(
                nn.Conv2d(1, 8, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 24, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((4, 8)),
                nn.Flatten(),
                nn.Linear(24 * 4 * 8, 48),
                nn.ReLU(),
            )
            self.presence_head = nn.Linear(48, 1)
            self.digit_head = nn.Linear(48, 3)
            for head in (self.presence_head, self.digit_head):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

        def components(self, images):
            with torch.no_grad():
                base_logits, base_boxes = self.base(images)
            glyph = images[
                :,
                :,
                config.glyph_y0 : config.glyph_y1,
                config.glyph_x0 : config.glyph_x1,
            ]
            hidden = self.glyph_encoder(glyph)
            presence = self.presence_head(hidden).squeeze(1)
            digits = self.digit_head(hidden)
            centered_digits = digits - digits.mean(dim=1, keepdim=True)
            adapter_logits = torch.cat(
                (-presence.unsqueeze(1), presence.unsqueeze(1) + centered_digits), dim=1
            )
            logits = base_logits + adapter_logits
            return logits, base_boxes, presence, digits, base_logits, adapter_logits

        def forward(self, images):
            logits, base_boxes, _presence, _digits, _base_logits, _adapter_logits = self.components(
                images
            )
            return logits, base_boxes

    return MeterClassificationAdapterV3A1(base_model).cpu()


def train_batch_v3_a1(
    model,
    images,
    classes,
    positive,
    *,
    real_count: int,
    optimizer,
    config: MeterRealDomainAdaptationConfigV3A1 = FROZEN_ADAPTATION_CONFIG_V3_A1,
) -> float:
    """Train real classification while forcing D10 adapter residuals toward zero."""
    import torch
    from torch.nn import functional as F

    if config != FROZEN_ADAPTATION_CONFIG_V3_A1:
        raise MeterRealDomainAdaptationV3A1Error("V3-A1 requires the frozen configuration")
    if not isinstance(real_count, int) or isinstance(real_count, bool):
        raise TypeError("real_count must be int")
    if not 0 <= real_count <= images.shape[0]:
        raise ValueError("real_count is outside the batch")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, _boxes, presence, digits, base_logits, adapter_logits = model.components(images)

    if real_count:
        real_slice = slice(0, real_count)
        real_positive = positive[real_slice]
        real_classes = classes[real_slice]
        classification = F.cross_entropy(logits[real_slice], real_classes)
        presence_targets = real_positive.to(dtype=torch.float32)
        presence_loss = F.binary_cross_entropy_with_logits(
            presence[real_slice], presence_targets
        )
        digit_loss = (
            F.cross_entropy(digits[real_slice][real_positive], real_classes[real_positive] - 1)
            if bool(real_positive.any())
            else digits[real_slice].sum() * 0.0
        )
    else:
        classification = logits.sum() * 0.0
        presence_loss = presence.sum() * 0.0
        digit_loss = digits.sum() * 0.0

    synthetic_mask = torch.arange(images.shape[0], device=images.device) >= real_count
    if bool(synthetic_mask.any()):
        temperature = config.distillation_temperature_milli / 1000.0
        distillation = F.kl_div(
            F.log_softmax(logits[synthetic_mask] / temperature, dim=1),
            F.softmax(base_logits[synthetic_mask] / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature * temperature)
        residual_zero = F.mse_loss(
            adapter_logits[synthetic_mask], torch.zeros_like(adapter_logits[synthetic_mask])
        )
    else:
        distillation = logits.sum() * 0.0
        residual_zero = adapter_logits.sum() * 0.0

    loss = (
        classification
        + (config.presence_loss_milli / 1000.0) * presence_loss
        + (config.digit_loss_milli / 1000.0) * digit_loss
        + (config.distillation_loss_milli / 1000.0) * distillation
        + (config.residual_zero_loss_milli / 1000.0) * residual_zero
    )
    if not bool(torch.isfinite(loss)):
        raise MeterRealDomainAdaptationV3A1Error("V3-A1 loss is not finite")
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip_milli / 1000.0)
    optimizer.step()
    return float(loss.detach().item())


def sealed_test_access_allowed() -> bool:
    return False


def runtime_connection_allowed() -> bool:
    return False


def resolver_connection_allowed() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False
