"""Shadow-only Meter V3-A2 positive-class margin recovery.

V3-A2 keeps the exact V3-A1 architecture and source-retention boundary:
D11 and bbox outputs remain fully frozen and only the bounded classification
adapter may update. The only new learning term is a fixed pairwise margin on
REAL TRAIN positive examples (2/4, 3/4, 4/4). It never special-cases held-out
validation records or family IDs.

TEST, runtime, Resolver, checkpoint replacement, and production promotion stay
closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

from .meter_real_domain_adaptation_v1 import (
    MeterEvaluationV1,
    PRESENCE_D11_SHA256,
    _canonical_json,
    _hex64,
    _sha,
)
from .meter_real_domain_adaptation_v3_a1 import (
    FROZEN_ADAPTATION_CONFIG_V3_A1,
    adaptation_acceptance_v3_a1,
    build_meter_classification_adapter_v3_a1,
)


METER_REAL_DOMAIN_ADAPTATION_V3_A2: Final[str] = "meter-real-domain-adaptation-v3-a2-positive-margin"


class MeterRealDomainAdaptationV3A2Error(RuntimeError):
    """Raised when the bounded V3-A2 contract is violated."""


@dataclass(frozen=True, slots=True)
class MeterRealDomainAdaptationConfigV3A2:
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
    positive_margin_loss_milli: int = 1_000
    positive_margin_milli: int = 2_000
    augmentation_shift_px: int = 4
    trainable_surface: str = "classification-adapter-only-d11-and-bbox-fully-frozen"
    objective: str = "v3-a1-plus-real-positive-pairwise-margin-v3-a2"
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
            "positive_margin_loss_milli": (self.positive_margin_loss_milli, 1, 10_000),
            "positive_margin_milli": (self.positive_margin_milli, 1, 20_000),
        }
        for name, (value, low, high) in bounds.items():
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ValueError(f"{name} is outside Meter V3-A2 bounds")
        if not (0 <= self.glyph_x0 < self.glyph_x1 <= 256):
            raise ValueError("V3-A2 glyph x window must stay inside the 256px ROI")
        if not (0 <= self.glyph_y0 < self.glyph_y1 <= 192):
            raise ValueError("V3-A2 glyph y window must stay inside the 192px ROI")
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
            "objective": "v3-a1-plus-real-positive-pairwise-margin-v3-a2",
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} is frozen to {expected!r}")


FROZEN_ADAPTATION_CONFIG_V3_A2: Final[MeterRealDomainAdaptationConfigV3A2] = (
    MeterRealDomainAdaptationConfigV3A2()
)


def meter_real_domain_adaptation_fingerprint_v3_a2(
    *,
    teacher_manifest_sha256: str,
    d10_manifest_sha256: str,
    d10_artifact_binding_sha256: str,
    config: MeterRealDomainAdaptationConfigV3A2 = FROZEN_ADAPTATION_CONFIG_V3_A2,
) -> str:
    return _sha(
        _canonical_json(
            {
                "version": METER_REAL_DOMAIN_ADAPTATION_V3_A2,
                "base_checkpoint_sha256": PRESENCE_D11_SHA256,
                "teacher_manifest_sha256": _hex64("teacher_manifest_sha256", teacher_manifest_sha256),
                "d10_manifest_sha256": _hex64("d10_manifest_sha256", d10_manifest_sha256),
                "d10_artifact_binding_sha256": _hex64(
                    "d10_artifact_binding_sha256", d10_artifact_binding_sha256
                ),
                "config": asdict(config),
                "validation_policy": "held-out-validation-never-used-by-positive-margin-loss",
                "test_policy": "sealed-test-never-enumerated-or-opened",
            }
        )
    )


def build_meter_classification_adapter_v3_a2(base_model):
    """Build the exact unchanged V3-A1 classification-only adapter."""
    return build_meter_classification_adapter_v3_a1(
        base_model,
        FROZEN_ADAPTATION_CONFIG_V3_A1,
    )


def adaptation_acceptance_v3_a2(
    *,
    candidate_real: MeterEvaluationV1,
    baseline_synthetic: MeterEvaluationV1,
    candidate_synthetic: MeterEvaluationV1,
):
    """Apply the unchanged V3-A1 real/source-retention gate."""
    return adaptation_acceptance_v3_a1(
        candidate_real=candidate_real,
        baseline_synthetic=baseline_synthetic,
        candidate_synthetic=candidate_synthetic,
    )


def real_positive_pairwise_margin_loss_v3_a2(
    logits,
    classes,
    positive,
    *,
    margin: float,
):
    """Require the true positive Meter class to beat both positive alternatives.

    Inputs are REAL TRAIN records only. Class index 0 (none) is intentionally
    excluded from this term so the V3-A1 none/source-retention behavior is not
    redesigned by V3-A2.
    """
    import torch
    from torch.nn import functional as F

    if margin <= 0:
        raise ValueError("V3-A2 positive margin must be positive")
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError("V3-A2 logits must be [B,4]")
    if classes.ndim != 1 or positive.ndim != 1 or classes.shape[0] != logits.shape[0]:
        raise ValueError("V3-A2 class/positive shapes must match logits")
    mask = positive.to(dtype=torch.bool)
    if not bool(mask.any()):
        return logits.sum() * 0.0

    positive_logits = logits[mask, 1:]
    positive_targets = classes[mask] - 1
    if bool(((positive_targets < 0) | (positive_targets > 2)).any()):
        raise ValueError("V3-A2 positive records must target 2/4, 3/4, or 4/4")

    true_scores = positive_logits.gather(1, positive_targets.unsqueeze(1)).squeeze(1)
    wrong_scores = positive_logits.clone()
    wrong_scores.scatter_(1, positive_targets.unsqueeze(1), float("-inf"))
    strongest_wrong = wrong_scores.max(dim=1).values
    return F.relu(margin - (true_scores - strongest_wrong)).mean()


def train_batch_v3_a2(
    model,
    images,
    classes,
    positive,
    *,
    real_count: int,
    optimizer,
    config: MeterRealDomainAdaptationConfigV3A2 = FROZEN_ADAPTATION_CONFIG_V3_A2,
) -> float:
    """V3-A1 objective plus one fixed REAL-positive pairwise margin term."""
    import torch
    from torch.nn import functional as F

    if config != FROZEN_ADAPTATION_CONFIG_V3_A2:
        raise MeterRealDomainAdaptationV3A2Error("V3-A2 requires the frozen configuration")
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
        presence_loss = F.binary_cross_entropy_with_logits(presence[real_slice], presence_targets)
        digit_loss = (
            F.cross_entropy(digits[real_slice][real_positive], real_classes[real_positive] - 1)
            if bool(real_positive.any())
            else digits[real_slice].sum() * 0.0
        )
        positive_margin = real_positive_pairwise_margin_loss_v3_a2(
            logits[real_slice],
            real_classes,
            real_positive,
            margin=config.positive_margin_milli / 1000.0,
        )
    else:
        classification = logits.sum() * 0.0
        presence_loss = presence.sum() * 0.0
        digit_loss = digits.sum() * 0.0
        positive_margin = logits.sum() * 0.0

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
        + (config.positive_margin_loss_milli / 1000.0) * positive_margin
        + (config.distillation_loss_milli / 1000.0) * distillation
        + (config.residual_zero_loss_milli / 1000.0) * residual_zero
    )
    if not bool(torch.isfinite(loss)):
        raise MeterRealDomainAdaptationV3A2Error("V3-A2 loss is not finite")
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
