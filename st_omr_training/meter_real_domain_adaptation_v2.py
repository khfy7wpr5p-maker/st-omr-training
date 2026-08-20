"""Shadow-only Meter real-domain recovery with a frozen D11 base.

V1 updated the shared D11 projection/classifier/bbox surface.  The real pilot
improved, but the shared update forgot too much of the accepted synthetic
surface.  V2 freezes every D11 parameter and trains only a small glyph-zone
adapter.  The adapter decomposes the class decision into presence and upper
digit (2/3/4), while synthetic logit/box distillation anchors the unchanged
D11 behaviour.

This module is deliberately shadow-only: TEST, runtime, Resolver, checkpoint
replacement, and production promotion remain closed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
from typing import Final

from .meter_real_domain_adaptation_v1 import (
    MeterEvaluationV1,
    PRESENCE_D11_SHA256,
    _canonical_json,
    _evaluate_synthetic,
    _evaluate_teacher,
    _evaluation_from_payload_v1,
    _hex64,
    _load_teacher_records,
    _prepare_output_root,
    _read_regular,
    _sha,
    _stack_teacher,
    _teacher_inference_fingerprint,
    _teacher_target,
    deterministic_replay_ids_v1,
)
from .meter_teacher_gold_admission_v1 import METER_CLASSES, verify_meter_teacher_gold_bundle_v1


METER_REAL_DOMAIN_ADAPTATION_V2: Final[str] = "meter-real-domain-adaptation-v2"
METRICS_SCHEMA_V2: Final[str] = "st-omr-meter-real-domain-adaptation-metrics-v2"
VERIFICATION_SCHEMA_V2: Final[str] = "st-omr-meter-real-domain-adaptation-verification-v2"
CHECKPOINT_ROLE_V2: Final[str] = "meter-real-domain-shadow-candidate-v2"
RESUME_ROLE_V2: Final[str] = "meter-real-domain-adaptation-resume-v2"


class MeterRealDomainAdaptationV2Error(RuntimeError):
    """Raised when a V2 provenance, data, numeric, or acceptance boundary fails."""


def _fail(message: str) -> None:
    raise MeterRealDomainAdaptationV2Error(message)


@dataclass(frozen=True, slots=True)
class MeterRealDomainAdaptationConfigV2:
    batch_size: int = 32
    epochs: int = 20
    learning_rate_micros: int = 1_000
    weight_decay_micros: int = 100
    grad_clip_milli: int = 1_000
    master_seed: int = 812_021
    real_balanced_repeat_factor: int = 4
    synthetic_replay_per_class: int = 128
    # Teacher Gold contains shifted 4/4 glyphs whose mapped right edge reaches
    # x=244.15. Keep a bounded search strip instead of assuming one fixed x.
    glyph_x0: int = 56
    glyph_x1: int = 248
    glyph_y0: int = 8
    glyph_y1: int = 184
    bbox_max_delta_milli: int = 120
    presence_loss_milli: int = 500
    digit_loss_milli: int = 750
    bbox_loss_milli: int = 1_000
    distillation_loss_milli: int = 2_000
    bbox_anchor_loss_milli: int = 2_000
    distillation_temperature_milli: int = 2_000
    augmentation_shift_px: int = 4
    trainable_surface: str = "glyph-adapter-only-d11-fully-frozen"
    objective: str = "hierarchical-presence-digit-plus-synthetic-distillation-v2"
    checkpoint_selection: str = "all-gates-then-real-macro-f1-then-synthetic-macro-f1"
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
            "bbox_max_delta_milli": (self.bbox_max_delta_milli, 0, 500),
            "augmentation_shift_px": (self.augmentation_shift_px, 0, 16),
        }
        for name, (value, low, high) in bounds.items():
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ValueError(f"{name} is outside Meter adaptation v2 bounds")
        if not (0 <= self.glyph_x0 < self.glyph_x1 <= 256):
            raise ValueError("V2 glyph x window must stay inside the 256px ROI")
        if not (0 <= self.glyph_y0 < self.glyph_y1 <= 192):
            raise ValueError("V2 glyph y window must stay inside the 192px ROI")
        for name in (
            "presence_loss_milli",
            "digit_loss_milli",
            "bbox_loss_milli",
            "distillation_loss_milli",
            "bbox_anchor_loss_milli",
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
            "trainable_surface": "glyph-adapter-only-d11-fully-frozen",
            "objective": "hierarchical-presence-digit-plus-synthetic-distillation-v2",
            "checkpoint_selection": "all-gates-then-real-macro-f1-then-synthetic-macro-f1",
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} is frozen to {expected!r}")


FROZEN_ADAPTATION_CONFIG_V2: Final[MeterRealDomainAdaptationConfigV2] = (
    MeterRealDomainAdaptationConfigV2()
)


@dataclass(frozen=True, slots=True)
class AdaptationGateDecisionV2:
    accepted: bool
    reasons: tuple[str, ...]


def adaptation_acceptance_v2(
    *,
    candidate_real: MeterEvaluationV1,
    baseline_synthetic: MeterEvaluationV1,
    candidate_synthetic: MeterEvaluationV1,
    config: MeterRealDomainAdaptationConfigV2 = FROZEN_ADAPTATION_CONFIG_V2,
) -> AdaptationGateDecisionV2:
    """Require pilot real metrics above 90% without synthetic forgetting."""
    reasons: list[str] = []
    threshold = lambda milli: milli / 1000.0
    if candidate_real.macro_f1 < threshold(config.real_min_macro_f1_milli):
        reasons.append("REAL_MACRO_F1_BELOW_90_PERCENT")
    if candidate_real.accuracy < threshold(config.real_min_accuracy_milli):
        reasons.append("REAL_ACCURACY_BELOW_90_PERCENT")
    if candidate_real.per_class_recall["none"] < threshold(config.real_min_none_recall_milli):
        reasons.append("REAL_NONE_RECALL_BELOW_MINIMUM")
    for label in METER_CLASSES[1:]:
        if candidate_real.per_class_recall[label] < threshold(
            config.real_min_positive_class_recall_milli
        ):
            reasons.append(f"REAL_{label.replace('/', '_')}_RECALL_NOT_3_OF_3")
    if (
        baseline_synthetic.macro_f1 - candidate_synthetic.macro_f1
        > threshold(config.synthetic_max_macro_f1_drop_milli)
    ):
        reasons.append("SYNTHETIC_MACRO_F1_REGRESSION")
    if (
        baseline_synthetic.positive_localization_f1_2px
        - candidate_synthetic.positive_localization_f1_2px
        > threshold(config.synthetic_max_localization_drop_milli)
    ):
        reasons.append("SYNTHETIC_LOCALIZATION_REGRESSION")
    return AdaptationGateDecisionV2(not reasons, tuple(reasons))


def meter_real_domain_adaptation_fingerprint_v2(
    *,
    teacher_manifest_sha256: str,
    d10_manifest_sha256: str,
    d10_artifact_binding_sha256: str,
    config: MeterRealDomainAdaptationConfigV2 = FROZEN_ADAPTATION_CONFIG_V2,
) -> str:
    return _sha(
        _canonical_json(
            {
                "version": METER_REAL_DOMAIN_ADAPTATION_V2,
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


def build_meter_glyph_adapter_v2(base_model, config=FROZEN_ADAPTATION_CONFIG_V2):
    """Wrap a fully frozen D11 model with a small spatial glyph adapter."""
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise MeterRealDomainAdaptationV2Error("torch is required for the V2 adapter") from exc
    if config != FROZEN_ADAPTATION_CONFIG_V2:
        _fail("Meter V2 adapter requires the frozen configuration")

    class MeterGlyphAdapterV2(nn.Module):
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
                # Retain horizontal evidence so shifted Meter glyphs remain
                # distinguishable from ordinary noteheads in the measure.
                nn.AdaptiveMaxPool2d((4, 8)),
                nn.Flatten(),
                nn.Linear(24 * 4 * 8, 48),
                nn.ReLU(),
            )
            self.presence_head = nn.Linear(48, 1)
            self.digit_head = nn.Linear(48, 3)
            self.bbox_delta_head = nn.Linear(48, 4)
            for head in (self.presence_head, self.digit_head, self.bbox_delta_head):
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
            maximum_delta = config.bbox_max_delta_milli / 1000.0
            adjusted = torch.clamp(
                base_boxes + maximum_delta * torch.tanh(self.bbox_delta_head(hidden)), 0.0, 1.0
            )
            boxes = torch.stack(
                (
                    torch.minimum(adjusted[:, 0], adjusted[:, 2]),
                    torch.minimum(adjusted[:, 1], adjusted[:, 3]),
                    torch.maximum(adjusted[:, 0], adjusted[:, 2]),
                    torch.maximum(adjusted[:, 1], adjusted[:, 3]),
                ),
                dim=1,
            )
            return logits, boxes, presence, digits, base_logits, base_boxes

        def forward(self, images):
            logits, boxes, _presence, _digits, _base_logits, _base_boxes = self.components(images)
            return logits, boxes

    return MeterGlyphAdapterV2(base_model).cpu()


def _clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _balanced_real_records(records: Sequence[object], repeat: int) -> tuple[object, ...]:
    by_class: defaultdict[str, list[object]] = defaultdict(list)
    for record in records:
        by_class[str(_teacher_target(record).get("meter_class"))].append(record)
    if {label: len(by_class[label]) for label in METER_CLASSES} != {
        "none": 27,
        "2/4": 9,
        "3/4": 9,
        "4/4": 9,
    }:
        _fail("V2 real TRAIN class balance changed")
    balanced: list[object] = []
    for label in METER_CLASSES:
        multiplier = 1 if label == "none" else 3
        balanced.extend(by_class[label] * multiplier)
    return tuple(balanced * repeat)


def _augment_real(images, *, seed: int, maximum_shift: int):
    if maximum_shift == 0 or images.shape[0] == 0:
        return images
    result = images.clone()
    for index in range(images.shape[0]):
        rng = random.Random(seed + index * 104_729)
        dx = rng.randint(-maximum_shift, maximum_shift)
        dy = rng.randint(-maximum_shift, maximum_shift)
        source = images[index]
        shifted = source.new_zeros(source.shape)
        source_x0, source_x1 = max(0, -dx), min(source.shape[2], source.shape[2] - dx)
        source_y0, source_y1 = max(0, -dy), min(source.shape[1], source.shape[1] - dy)
        target_x0, target_x1 = max(0, dx), min(source.shape[2], source.shape[2] + dx)
        target_y0, target_y1 = max(0, dy), min(source.shape[1], source.shape[1] + dy)
        shifted[:, target_y0:target_y1, target_x0:target_x1] = source[
            :, source_y0:source_y1, source_x0:source_x1
        ]
        gain = 0.90 + 0.20 * rng.random()
        result[index] = shifted.mul(gain).clamp(0.0, 1.0)
    return result


def _train_batch_v2(
    model,
    images,
    classes,
    boxes,
    positive,
    *,
    real_count: int,
    optimizer,
    config: MeterRealDomainAdaptationConfigV2,
) -> float:
    import torch
    from torch.nn import functional as F
    from .training_model import assert_finite_tensor

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, predicted_boxes, presence, digits, base_logits, base_boxes = model.components(images)
    classification = F.cross_entropy(logits, classes)
    presence_targets = positive.to(dtype=torch.float32)
    presence_loss = F.binary_cross_entropy_with_logits(presence, presence_targets)
    digit_loss = (
        F.cross_entropy(digits[positive], classes[positive] - 1)
        if bool(positive.any())
        else digits.sum() * 0.0
    )
    bbox_loss = (
        F.smooth_l1_loss(predicted_boxes[positive], boxes[positive])
        if bool(positive.any())
        else predicted_boxes.sum() * 0.0
    )
    synthetic = torch.arange(images.shape[0]) >= real_count
    if bool(synthetic.any()):
        temperature = config.distillation_temperature_milli / 1000.0
        distillation = F.kl_div(
            F.log_softmax(logits[synthetic] / temperature, dim=1),
            F.softmax(base_logits[synthetic] / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature * temperature)
        bbox_anchor = F.smooth_l1_loss(predicted_boxes[synthetic], base_boxes[synthetic])
    else:
        distillation = logits.sum() * 0.0
        bbox_anchor = predicted_boxes.sum() * 0.0
    loss = (
        classification
        + (config.presence_loss_milli / 1000.0) * presence_loss
        + (config.digit_loss_milli / 1000.0) * digit_loss
        + (config.bbox_loss_milli / 1000.0) * bbox_loss
        + (config.distillation_loss_milli / 1000.0) * distillation
        + (config.bbox_anchor_loss_milli / 1000.0) * bbox_anchor
    )
    assert_finite_tensor("Meter V2 loss", loss)
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip_milli / 1000.0)
    optimizer.step()
    return float(loss.detach().item())


def _better_candidate(real, synthetic, best_real, best_synthetic) -> bool:
    return (
        real.macro_f1 > best_real.macro_f1
        or (
            real.macro_f1 == best_real.macro_f1
            and synthetic.macro_f1 > best_synthetic.macro_f1
        )
        or (
            real.macro_f1 == best_real.macro_f1
            and synthetic.macro_f1 == best_synthetic.macro_f1
            and real.loss < best_real.loss
        )
    )


def run_meter_real_domain_adaptation_v2(
    *,
    teacher_bundle_root: str | Path,
    d10_root: str | Path,
    base_checkpoint_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    expected_d10_manifest_sha256: str,
    expected_d10_artifact_binding_sha256: str,
    config: MeterRealDomainAdaptationConfigV2 = FROZEN_ADAPTATION_CONFIG_V2,
    progress=None,
    resume: bool = False,
) -> dict[str, object]:
    """Run the bounded V2 shadow recovery and emit only a gate-passing candidate."""
    if not isinstance(config, MeterRealDomainAdaptationConfigV2):
        raise TypeError("config must be MeterRealDomainAdaptationConfigV2")
    if config != FROZEN_ADAPTATION_CONFIG_V2:
        _fail("Meter real-domain adaptation v2 requires the frozen configuration")
    if not isinstance(resume, bool):
        raise TypeError("resume must be bool")
    try:
        import torch
        from .runtime_meter_real_checkpoint_audit_v1 import audit_presence_d11_checkpoint_v1
        from .stage7c_execution import verify_authoritative_repository
        from .stage7d11_barline_meter_training import (
            FROZEN_D11_CONFIG,
            _load_d11_label,
            _stack_meter,
            build_meter_refiner,
            load_verified_stage7d11_records,
        )
        from .training_model import assert_model_finite, model_state_sha256, set_deterministic_cpu
    except ModuleNotFoundError as exc:
        raise MeterRealDomainAdaptationV2Error(
            "torch and the pinned training runtime are required for Meter V2"
        ) from exc

    teacher_root = Path(teacher_bundle_root)
    if progress is not None:
        progress("phase_started", {"phase": "teacher_gold_verify", "phase_index": 1, "phase_total": 7})
    teacher_receipt = verify_meter_teacher_gold_bundle_v1(teacher_root)
    teacher_records = _load_teacher_records(teacher_root)
    real_train = tuple(record for record in teacher_records if record.split == "train")
    real_validation = tuple(record for record in teacher_records if record.split == "validation")
    balanced_real = _balanced_real_records(real_train, config.real_balanced_repeat_factor)

    d10_manifest = _hex64("expected D10 manifest SHA", expected_d10_manifest_sha256)
    d10_binding = _hex64("expected D10 artifact binding SHA", expected_d10_artifact_binding_sha256)
    if progress is not None:
        progress(
            "phase_started",
            {"phase": "d10_full_integrity_verify", "phase_index": 2, "phase_total": 7, "records_total": 22_128},
        )
    d10_records = load_verified_stage7d11_records(
        d10_root,
        expected_manifest_sha256=d10_manifest,
        expected_artifact_binding_sha256=d10_binding,
    )
    synthetic_train_all = tuple(
        record for record in d10_records if record.kind == "meter" and record.split == "train"
    )
    synthetic_validation = tuple(
        record for record in d10_records if record.kind == "meter" and record.split == "validation"
    )
    if len(synthetic_train_all) != 9_840 or len(synthetic_validation) != 1_224:
        _fail("accepted D10 Meter surface cardinality changed")
    class_to_ids: defaultdict[str, list[str]] = defaultdict(list)
    record_by_id = {record.record_id: record for record in synthetic_train_all}
    for record in synthetic_train_all:
        target = _load_d11_label(record).get("target")
        if not isinstance(target, Mapping) or target.get("meter_class") not in METER_CLASSES:
            _fail("D10 replay target class is invalid")
        class_to_ids[str(target["meter_class"])].append(record.record_id)
    replay_ids = deterministic_replay_ids_v1(
        class_to_ids,
        per_class=config.synthetic_replay_per_class,
        seed=config.master_seed,
    )
    synthetic_replay = tuple(record_by_id[record_id] for record_id in replay_ids)

    base_path = Path(base_checkpoint_path)
    if progress is not None:
        progress("phase_started", {"phase": "d11_checkpoint_audit", "phase_index": 3, "phase_total": 7})
    if _sha(_read_regular(base_path, maximum=64 * 1024 * 1024, name="base D11 checkpoint")) != PRESENCE_D11_SHA256:
        _fail("base D11 checkpoint SHA-256 mismatch")
    audited = audit_presence_d11_checkpoint_v1(base_path)
    if audited.checkpoint_sha256 != PRESENCE_D11_SHA256 or audited.role != "presence-d11-bridge":
        _fail("base checkpoint audit did not return the exact D11 Meter state")

    repository_sha, repository_origin = verify_authoritative_repository(repository_root)
    profile = meter_real_domain_adaptation_fingerprint_v2(
        teacher_manifest_sha256=teacher_receipt.manifest_sha256,
        d10_manifest_sha256=d10_manifest,
        d10_artifact_binding_sha256=d10_binding,
        config=config,
    )
    root = Path(output_root)
    _prepare_output_root(root, Path(repository_root), resume=resume)

    set_deterministic_cpu(config.master_seed)
    base_model = build_meter_refiner(FROZEN_D11_CONFIG)
    base_model.load_state_dict(dict(audited.model_state), strict=True)
    for parameter in base_model.parameters():
        parameter.requires_grad = False
    base_state_sha = model_state_sha256(base_model)
    model = build_meter_glyph_adapter_v2(base_model, config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.requires_grad for parameter in model.base.parameters()):
        _fail("V2 must train only the glyph adapter while D11 remains fully frozen")
    assert_model_finite(model)

    if progress is not None:
        progress("phase_started", {"phase": "baseline_real_validation", "phase_index": 4, "phase_total": 7})
    baseline_real = _evaluate_teacher(base_model, real_validation, config)
    if progress is not None:
        progress(
            "phase_started",
            {"phase": "baseline_synthetic_validation", "phase_index": 5, "phase_total": 7, "records_total": 1_224},
        )
    baseline_synthetic = _evaluate_synthetic(base_model, synthetic_validation, config)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate_micros / 1_000_000.0,
        weight_decay=config.weight_decay_micros / 1_000_000.0,
    )

    training_items = [("real", record) for record in balanced_real]
    training_items += [("synthetic", record) for record in synthetic_replay]
    batches_per_epoch = math.ceil(len(training_items) / config.batch_size)
    history: list[dict[str, object]] = []
    best_state = None
    best_decision = AdaptationGateDecisionV2(False, ("NO_EPOCH_EVALUATED",))
    best_real = baseline_real
    best_synthetic = baseline_synthetic
    best_epoch = 0
    optimizer_steps = 0
    completed_epoch = 0

    resume_path = root / "resume.pt"
    if resume_path.exists():
        if resume_path.is_symlink() or not resume_path.is_file():
            _fail("V2 resume state must be a regular file")
        try:
            snapshot = torch.load(resume_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise MeterRealDomainAdaptationV2Error("V2 resume state cannot be loaded safely") from exc
        if not isinstance(snapshot, Mapping):
            _fail("V2 resume state must be a mapping")
        checks = {
            "role": RESUME_ROLE_V2,
            "adaptation_version": METER_REAL_DOMAIN_ADAPTATION_V2,
            "repository_sha": repository_sha,
            "profile_fingerprint": profile,
            "teacher_manifest_sha256": teacher_receipt.manifest_sha256,
            "d10_manifest_sha256": d10_manifest,
            "base_checkpoint_sha256": PRESENCE_D11_SHA256,
            "base_meter_state_sha256": base_state_sha,
            "baseline_real": asdict(baseline_real),
            "baseline_synthetic": asdict(baseline_synthetic),
        }
        for name, expected in checks.items():
            if snapshot.get(name) != expected:
                _fail(f"V2 resume state {name} mismatch")
        try:
            completed_epoch = int(snapshot["completed_epoch"])
            best_epoch = int(snapshot["best_epoch"])
            optimizer_steps = int(snapshot["optimizer_steps"])
            if not 1 <= completed_epoch <= config.epochs:
                _fail("V2 resume epoch is outside the frozen run")
            model.load_state_dict(snapshot["current_model_state"], strict=True)
            optimizer.load_state_dict(snapshot["optimizer_state_dict"])
            loaded_best = snapshot.get("best_model_state")
            best_state = None if loaded_best is None else dict(loaded_best)
            decision = snapshot["best_decision"]
            if not isinstance(decision, Mapping):
                _fail("V2 resume best decision is malformed")
            reasons = decision.get("reasons")
            if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes, bytearray)):
                _fail("V2 resume reasons are malformed")
            best_decision = AdaptationGateDecisionV2(bool(decision.get("accepted")), tuple(str(x) for x in reasons))
            best_real = _evaluation_from_payload_v1(snapshot["best_real"], name="V2 resume best real")
            best_synthetic = _evaluation_from_payload_v1(
                snapshot["best_synthetic"], name="V2 resume best synthetic"
            )
            history = list(snapshot["history"])
        except MeterRealDomainAdaptationV2Error:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise MeterRealDomainAdaptationV2Error("V2 resume state is malformed") from exc
        if progress is not None:
            progress(
                "resume_loaded",
                {"completed_epoch": completed_epoch, "epochs_total": config.epochs, "optimizer_steps": optimizer_steps},
            )
    if len(history) != completed_epoch or optimizer_steps != completed_epoch * batches_per_epoch:
        _fail("V2 resume history/optimizer counters are inconsistent")

    if progress is not None:
        progress(
            "phase_started",
            {
                "phase": "training_and_validation",
                "phase_index": 6,
                "phase_total": 7,
                "completed_epoch": completed_epoch,
                "epochs_total": config.epochs,
                "batches_per_epoch": batches_per_epoch,
            },
        )
    for epoch in range(completed_epoch + 1, config.epochs + 1):
        order = list(range(len(training_items)))
        random.Random(config.master_seed + epoch * 1_000_003).shuffle(order)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(order), config.batch_size):
            items = [training_items[index] for index in order[start : start + config.batch_size]]
            teacher_batch = [record for kind, record in items if kind == "real"]
            synthetic_batch = [record for kind, record in items if kind == "synthetic"]
            tensors = []
            if teacher_batch:
                real_tensors = list(_stack_teacher(teacher_batch))
                real_tensors[0] = _augment_real(
                    real_tensors[0],
                    seed=config.master_seed + epoch * 10_000_019 + batches * 1_009,
                    maximum_shift=config.augmentation_shift_px,
                )
                tensors.append(tuple(real_tensors))
            if synthetic_batch:
                tensors.append(_stack_meter(synthetic_batch))
            images = torch.cat([value[0] for value in tensors], dim=0)
            classes = torch.cat([value[1] for value in tensors], dim=0)
            boxes = torch.cat([value[2] for value in tensors], dim=0)
            positive = torch.cat([value[3] for value in tensors], dim=0)
            total_loss += _train_batch_v2(
                model,
                images,
                classes,
                boxes,
                positive,
                real_count=len(teacher_batch),
                optimizer=optimizer,
                config=config,
            )
            optimizer_steps += 1
            batches += 1
            if progress is not None:
                progress(
                    "training_batch",
                    {
                        "epoch": epoch,
                        "epochs_total": config.epochs,
                        "batch": batches,
                        "batches_total": batches_per_epoch,
                        "optimizer_steps": optimizer_steps,
                    },
                )
        real_metrics = _evaluate_teacher(model, real_validation, config)
        synthetic_metrics = _evaluate_synthetic(model, synthetic_validation, config)
        decision = adaptation_acceptance_v2(
            candidate_real=real_metrics,
            baseline_synthetic=baseline_synthetic,
            candidate_synthetic=synthetic_metrics,
            config=config,
        )
        event = {
            "epoch": epoch,
            "train_loss": total_loss / batches,
            "real_validation": asdict(real_metrics),
            "synthetic_validation": asdict(synthetic_metrics),
            "gate": asdict(decision),
        }
        history.append(event)
        if decision.accepted and (
            best_state is None
            or _better_candidate(real_metrics, synthetic_metrics, best_real, best_synthetic)
        ):
            best_state = _clone_state(model)
            best_decision = decision
            best_real = real_metrics
            best_synthetic = synthetic_metrics
            best_epoch = epoch
        elif best_state is None and (
            best_epoch == 0
            or _better_candidate(real_metrics, synthetic_metrics, best_real, best_synthetic)
        ):
            best_decision = decision
            best_real = real_metrics
            best_synthetic = synthetic_metrics
            best_epoch = epoch
        temporary_resume = root / "resume.tmp.pt"
        torch.save(
            {
                "role": RESUME_ROLE_V2,
                "adaptation_version": METER_REAL_DOMAIN_ADAPTATION_V2,
                "repository_sha": repository_sha,
                "profile_fingerprint": profile,
                "teacher_manifest_sha256": teacher_receipt.manifest_sha256,
                "d10_manifest_sha256": d10_manifest,
                "base_checkpoint_sha256": PRESENCE_D11_SHA256,
                "base_meter_state_sha256": base_state_sha,
                "baseline_real": asdict(baseline_real),
                "baseline_synthetic": asdict(baseline_synthetic),
                "completed_epoch": epoch,
                "current_model_state": _clone_state(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_model_state": best_state,
                "best_decision": asdict(best_decision),
                "best_real": asdict(best_real),
                "best_synthetic": asdict(best_synthetic),
                "best_epoch": best_epoch,
                "optimizer_steps": optimizer_steps,
                "history": history,
            },
            temporary_resume,
        )
        temporary_resume.replace(resume_path)
        if progress is not None:
            progress("epoch_checkpointed", {"epoch": epoch, "epochs_total": config.epochs, "resume_path": str(resume_path)})
            progress("epoch_complete", event)

    if progress is not None:
        progress("phase_started", {"phase": "final_verification", "phase_index": 7, "phase_total": 7})
    ending_sha, ending_origin = verify_authoritative_repository(repository_root)
    if (ending_sha, ending_origin) != (repository_sha, repository_origin):
        _fail("repository identity changed during Meter V2")
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    assert_model_finite(model)
    if model_state_sha256(model.base) != base_state_sha:
        _fail("V2 mutated the fully frozen D11 model")

    candidate_state_sha = model_state_sha256(model) if best_state is not None else None
    replay_fingerprints: tuple[str, ...] = ()
    if best_state is not None:
        replay_fingerprints = tuple(
            _teacher_inference_fingerprint(model, real_validation) for _ in range(10)
        )
        if len(set(replay_fingerprints)) != 1:
            _fail("V2 candidate inference is not deterministic 10/10")
    status = "SHADOW_CANDIDATE_ACCEPTED" if best_state is not None else "HOLD_NO_ACCEPTED_CANDIDATE"
    run_id = _sha(
        _canonical_json(
            {
                "version": METER_REAL_DOMAIN_ADAPTATION_V2,
                "repository_sha": repository_sha,
                "profile_fingerprint": profile,
                "teacher_manifest_sha256": teacher_receipt.manifest_sha256,
                "d10_manifest_sha256": d10_manifest,
            }
        )
    )

    checkpoint_path = None
    checkpoint_sha = None
    checkpoint_reload_verified = False
    if best_state is not None:
        temporary = root / "checkpoint.tmp.pt"
        torch.save(
            {
                "role": CHECKPOINT_ROLE_V2,
                "model_state_dict": best_state,
                "base_checkpoint_sha256": PRESENCE_D11_SHA256,
                "base_meter_state_sha256": base_state_sha,
                "candidate_meter_state_sha256": candidate_state_sha,
                "profile_fingerprint": profile,
                "teacher_manifest_sha256": teacher_receipt.manifest_sha256,
                "d10_manifest_sha256": d10_manifest,
                "best_epoch": best_epoch,
                "runtime_connected": False,
                "production_promotion_authorized": False,
            },
            temporary,
        )
        checkpoint_sha = _sha(temporary.read_bytes())
        checkpoint_path = root / f"checkpoint-{checkpoint_sha}.pt"
        temporary.rename(checkpoint_path)
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping) or payload.get("role") != CHECKPOINT_ROLE_V2:
                _fail("V2 checkpoint role changed during reload")
            reload_base = build_meter_refiner(FROZEN_D11_CONFIG)
            reload_base.load_state_dict(dict(audited.model_state), strict=True)
            reloaded = build_meter_glyph_adapter_v2(reload_base, config)
            reloaded.load_state_dict(payload["model_state_dict"], strict=True)
            assert_model_finite(reloaded)
        except MeterRealDomainAdaptationV2Error:
            raise
        except Exception as exc:
            raise MeterRealDomainAdaptationV2Error("V2 checkpoint cannot be strictly reloaded") from exc
        if model_state_sha256(reloaded) != candidate_state_sha or model_state_sha256(reloaded.base) != base_state_sha:
            _fail("V2 checkpoint state hash mismatch after reload")
        checkpoint_reload_verified = True

    metrics = {
        "schema_version": METRICS_SCHEMA_V2,
        "adaptation_version": METER_REAL_DOMAIN_ADAPTATION_V2,
        "status": status,
        "run_id": run_id,
        "repository_sha": repository_sha,
        "repository_origin": repository_origin,
        "profile_fingerprint": profile,
        "configuration": asdict(config),
        "base_checkpoint_sha256": PRESENCE_D11_SHA256,
        "base_meter_state_sha256": base_state_sha,
        "teacher_gold": {
            "manifest_sha256": teacher_receipt.manifest_sha256,
            "artifact_binding_sha256": teacher_receipt.artifact_binding_sha256,
            "train_records": 54,
            "validation_records": 18,
        },
        "synthetic_replay": {
            "d10_manifest_sha256": d10_manifest,
            "d10_artifact_binding_sha256": d10_binding,
            "train_records": len(synthetic_replay),
            "train_records_per_class": config.synthetic_replay_per_class,
            "validation_records": len(synthetic_validation),
        },
        "baseline": {
            "real_validation": asdict(baseline_real),
            "synthetic_validation": asdict(baseline_synthetic),
        },
        "best": {
            "epoch": best_epoch,
            "real_validation": asdict(best_real),
            "synthetic_validation": asdict(best_synthetic),
            "gate": asdict(best_decision),
            "candidate_meter_state_sha256": candidate_state_sha,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_filename": checkpoint_path.name if checkpoint_path is not None else None,
        },
        "history": history,
        "optimizer_steps": optimizer_steps,
        "d11_fully_frozen": True,
        "candidate_replay_10_of_10": best_state is not None and len(set(replay_fingerprints)) == 1,
        "candidate_replay_fingerprint": replay_fingerprints[0] if replay_fingerprints else None,
        "checkpoint_reload_verified": checkpoint_reload_verified,
        "test_records": 0,
        "test_opened": False,
        "runtime_connected": False,
        "resolver_connected": False,
        "production_promotion_authorized": False,
    }
    metrics_raw = _canonical_json(metrics)
    metrics_sha = _sha(metrics_raw)
    metrics_path = root / f"metrics-{metrics_sha}.json"
    metrics_path.write_bytes(metrics_raw)
    verification = {
        "schema_version": VERIFICATION_SCHEMA_V2,
        "adaptation_version": METER_REAL_DOMAIN_ADAPTATION_V2,
        "status": status,
        "run_id": run_id,
        "repository_sha": repository_sha,
        "profile_fingerprint": profile,
        "metrics_sha256": metrics_sha,
        "checkpoint_sha256": checkpoint_sha,
        "base_checkpoint_sha256": PRESENCE_D11_SHA256,
        "teacher_manifest_sha256": teacher_receipt.manifest_sha256,
        "d10_manifest_sha256": d10_manifest,
        "d10_artifact_binding_sha256": d10_binding,
        "optimizer_steps": optimizer_steps,
        "d11_fully_frozen": True,
        "checkpoint_is_shadow_only": True,
        "candidate_replay_10_of_10": best_state is not None and len(set(replay_fingerprints)) == 1,
        "checkpoint_reload_verified": checkpoint_reload_verified,
        "test_records": 0,
        "test_opened": False,
        "runtime_connected": False,
        "resolver_connected": False,
        "production_promotion_authorized": False,
        "repository_stable_during_run": True,
    }
    verification_raw = _canonical_json(verification)
    verification_sha = _sha(verification_raw)
    verification_path = root / f"verification-{verification_sha}.json"
    verification_path.write_bytes(verification_raw)
    lines = [f"{verification_sha}  {verification_path.name}", f"{metrics_sha}  {metrics_path.name}"]
    if checkpoint_path is not None and checkpoint_sha is not None:
        lines.append(f"{checkpoint_sha}  {checkpoint_path.name}")
    (root / "RUN_COMPLETE").write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    if progress is not None:
        progress(
            "run_complete",
            {"status": status, "best_epoch": best_epoch, "epochs_total": config.epochs, "run_root": str(root)},
        )
    return metrics


def sealed_test_access_allowed() -> bool:
    return False


def runtime_connection_allowed() -> bool:
    return False


def resolver_connection_allowed() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False
