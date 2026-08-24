"""Meter V5-3F tensor-only execution harness for the fixed rescue recipe.

V5-3E preregistered exactly one 64->8->1 tanh rescue recipe for 2-AI and 3-AI.
V5-3F implements that recipe only as an in-memory tensor harness. It does not
discover datasets, read protected evaluation surfaces, mutate frozen specialists,
write checkpoints, wire runtime decisions, or promote production state.

CI may exercise this harness on synthetic tensors. An authoritative data run
requires a later, separately authorized exact-SHA execution wrapper.
"""
from __future__ import annotations

import hashlib
import math
from typing import Final, Mapping

from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_3e_rescue_training_preregistration_v1 as v53e


SCHEMA: Final[str] = "st-omr-meter-v5-3f-rescue-training-execution-harness-v1"
V53E_HEAD_SHA: Final[str] = "f27d2334d9dfbdd8c6c70d3e214573765cee15c6"
V53E_MODULE_BLOB_SHA: Final[str] = "c6cf28e8ea7301b6b03f3a4d7d6b931444af3795"
V53E_DOC_BLOB_SHA: Final[str] = "46b92c2026b7da98f2fc6c84e6b6030cd86c994a"
APPROVAL_TOKEN: Final[str] = "V5_3F_SINGLE_FIXED_RESCUE_TENSOR_HARNESS_APPROVED"

GROUP_TARGETS: Final[dict[str, float]] = {
    "v5_frozen_false_negative_positive": 1.0,
    "v5_frozen_true_negative": 0.0,
    "historical_frozen_false_negative_positive": 1.0,
    "historical_frozen_true_negative": 0.0,
}


class MeterV5_3FError(RuntimeError):
    """Raised when V5-3F departs from the exact preregistered harness."""


def _fail(message: str) -> None:
    raise MeterV5_3FError(message)


def prerequisite_contract() -> dict[str, object]:
    return {
        "v5_3e_head_sha": V53E_HEAD_SHA,
        "v5_3e_module_blob_sha": V53E_MODULE_BLOB_SHA,
        "v5_3e_doc_blob_sha": V53E_DOC_BLOB_SHA,
        "v5_3e_schema": v53e.SCHEMA,
        "recipe_id": v53e.RECIPE_ID,
        "candidate_configuration_count": v53e.CANDIDATE_CONFIGURATION_COUNT,
        "feature_dim": v53e.FEATURE_DIM,
        "hidden_width": v53e.HIDDEN_WIDTH,
        "activation": v53e.ACTIVATION,
        "rescue_threshold": v53e.RESCUE_THRESHOLD,
        "fixed_optimizer_steps": v53e.FIXED_OPTIMIZER_STEPS,
    }


def safety_boundary() -> dict[str, object]:
    return {
        "tensor_harness_present": True,
        "authoritative_dataset_execution_present": False,
        "dataset_path_access": False,
        "checkpoint_load": False,
        "checkpoint_write": False,
        "rescue_artifact_write": False,
        "frozen_model_reference_accepted": False,
        "frozen_model_mutation_surface": False,
        "trainable_surface": "new-rescue-parameters-only",
        "digit4_frozen": True,
        "threshold_tuning": False,
        "hyperparameter_sweep": False,
        "automatic_second_configuration": False,
        "historical_validation_opened": False,
        "first30_opened": False,
        "v5_reserve_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "bbox_access_added": False,
        "crop_geometry_change": False,
        "spatial_heuristic_change": False,
        "resolver_wiring": False,
        "production_promotion": False,
        "colab_execution_wrapper_present": False,
    }


def execution_contract() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "prerequisite": prerequisite_contract(),
        "objective": v53e.objective_contract(),
        "recipe": v53e.fixed_training_recipe(),
        "numerical_guards": v53e.numerical_execution_guards(),
        "approval_token_required": True,
        "ci_execution_surface": "synthetic-tensors-only",
        "authoritative_execution_requires_later_exact_sha_wrapper": True,
        **safety_boundary(),
    }


def _state_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _build_rescue_model_v1():
    torch, nn = v52b._import_torch()

    class RescueSpecialist(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = nn.Linear(v53e.FEATURE_DIM, v53e.HIDDEN_WIDTH)
            self.activation = nn.Tanh()
            self.output = nn.Linear(v53e.HIDDEN_WIDTH, 1)

        def forward(self, features):
            return self.output(self.activation(self.hidden(features))).squeeze(1)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(v53e.SEED)
        model = RescueSpecialist().cpu().to(dtype=torch.float32)
        for layer in (model.hidden, model.output):
            nn.init.xavier_uniform_(layer.weight, gain=v53e.INITIALIZATION_GAIN)
            nn.init.zeros_(layer.bias)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != v53e.PARAMETERS_PER_RESCUE:
        _fail(f"rescue parameter count changed: {parameter_count}")
    for parameter in model.parameters():
        if parameter.dtype != torch.float32 or parameter.device.type != "cpu":
            _fail("rescue parameter device/dtype changed")
        if not bool(torch.isfinite(parameter).all().item()):
            _fail("rescue initialization produced non-finite parameters")
    return model


def _validate_group_features(
    *,
    digit: str,
    features_by_group: Mapping[str, object],
    enforce_preregistered_counts: bool,
) -> dict[str, object]:
    if digit not in v53e.RESCUE_SPECIALISTS:
        _fail("V5-3F supports only 2-AI and 3-AI rescue specialists")
    if type(enforce_preregistered_counts) is not bool:
        _fail("enforce_preregistered_counts must be bool")
    if not isinstance(features_by_group, Mapping):
        _fail("features_by_group must be a mapping")
    if tuple(features_by_group.keys()) != v53e.TRAIN_GROUPS:
        _fail("TRAIN groups must be present in exact preregistered order")

    torch, _nn = v52b._import_torch()
    expected_counts = v53e.EXPECTED_TRAIN_GROUP_COUNTS[digit]
    normalized: dict[str, object] = {}
    for group in v53e.TRAIN_GROUPS:
        value = features_by_group[group]
        if not isinstance(value, torch.Tensor):
            _fail(f"{group} features must be a torch.Tensor")
        features = value.detach().cpu().to(dtype=torch.float32).clone()
        if features.ndim != 2 or features.shape[1] != v53e.FEATURE_DIM:
            _fail(f"{group} feature shape changed: {tuple(features.shape)}")
        if features.shape[0] <= 0:
            _fail(f"{group} must not be empty")
        if enforce_preregistered_counts and features.shape[0] != expected_counts[group]:
            _fail(
                f"{digit}-AI {group} count changed: "
                f"{features.shape[0]} != {expected_counts[group]}"
            )
        if not bool(torch.isfinite(features).all().item()):
            _fail(f"{group} contains non-finite features")
        normalized[group] = features
    return normalized


def _four_group_objective_v1(model, groups: Mapping[str, object]):
    torch, _nn = v52b._import_torch()
    losses: dict[str, object] = {}
    total = None
    counts: dict[str, int] = {}
    for group in v53e.TRAIN_GROUPS:
        features = groups[group]
        logits = model(features)
        if logits.ndim != 1 or logits.numel() != features.shape[0]:
            _fail(f"{group} rescue logit shape changed")
        if not bool(torch.isfinite(logits).all().item()):
            _fail(f"{group} produced non-finite logits")
        targets = torch.full_like(logits, GROUP_TARGETS[group])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="mean",
        )
        if not bool(torch.isfinite(loss).item()):
            _fail(f"{group} loss became non-finite")
        losses[group] = loss
        counts[group] = int(features.shape[0])
        weighted = v53e.TRAIN_GROUP_WEIGHT * loss
        total = weighted if total is None else total + weighted
    if total is None or not bool(torch.isfinite(total).item()):
        _fail("four-group rescue objective became non-finite")
    return total, losses, counts


def execute_rescue_tensor_harness_v1(
    *,
    digit: str,
    features_by_group: Mapping[str, object],
    approval_token: str,
    enforce_preregistered_counts: bool = True,
):
    """Run exactly one preregistered rescue fit on already-bound feature tensors.

    This function performs no file or checkpoint I/O. The caller is responsible
    for a later exact-SHA wrapper that materializes authoritative TRAIN tensors.
    """
    if approval_token != APPROVAL_TOKEN:
        _fail("exact V5-3F approval token is required before tensor access")
    groups = _validate_group_features(
        digit=digit,
        features_by_group=features_by_group,
        enforce_preregistered_counts=enforce_preregistered_counts,
    )
    torch, nn = v52b._import_torch()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model = _build_rescue_model_v1()
        initial_state_fingerprint = _state_fingerprint(model)
        with torch.no_grad():
            initial_total, initial_groups, group_counts = _four_group_objective_v1(model, groups)
        initial_loss = float(initial_total.item())

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=v53e.LEARNING_RATE,
            betas=(v53e.BETA1, v53e.BETA2),
            eps=v53e.EPSILON,
            weight_decay=v53e.WEIGHT_DECAY,
        )

        max_preclip_gradient_norm = 0.0
        final_loss = initial_loss
        final_group_losses = {
            name: float(initial_groups[name].item()) for name in v53e.TRAIN_GROUPS
        }
        for step in range(1, v53e.FIXED_OPTIMIZER_STEPS + 1):
            optimizer.zero_grad(set_to_none=True)
            total, group_losses, observed_counts = _four_group_objective_v1(model, groups)
            if observed_counts != group_counts:
                _fail("objective group counts changed during execution")
            if not bool(torch.isfinite(total).item()):
                _fail(f"step {step} loss became non-finite")
            total.backward()

            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    _fail(f"step {step} missing gradient: {name}")
                if not bool(torch.isfinite(parameter.grad).all().item()):
                    _fail(f"step {step} non-finite gradient: {name}")
            preclip_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=v53e.GRADIENT_CLIP_GLOBAL_NORM,
                error_if_nonfinite=True,
            )
            preclip_value = float(preclip_norm.item())
            if not math.isfinite(preclip_value):
                _fail(f"step {step} gradient norm became non-finite")
            max_preclip_gradient_norm = max(max_preclip_gradient_norm, preclip_value)

            optimizer.step()
            for name, parameter in model.named_parameters():
                if not bool(torch.isfinite(parameter).all().item()):
                    _fail(f"step {step} produced non-finite parameter: {name}")

            final_loss = float(total.detach().item())
            final_group_losses = {
                name: float(group_losses[name].detach().item())
                for name in v53e.TRAIN_GROUPS
            }

        with torch.no_grad():
            final_total_post, final_groups_post, final_counts_post = _four_group_objective_v1(
                model, groups
            )
        if final_counts_post != group_counts:
            _fail("objective group counts changed after final step")
        final_loss = float(final_total_post.item())
        final_group_losses = {
            name: float(final_groups_post[name].item()) for name in v53e.TRAIN_GROUPS
        }
        final_state_fingerprint = _state_fingerprint(model)
        evidence = {
            "schema": SCHEMA,
            "digit": digit,
            "recipe_id": v53e.RECIPE_ID,
            "approval_token_verified": True,
            "authoritative_dataset_execution": False,
            "preregistered_count_enforcement": bool(enforce_preregistered_counts),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "initial_state_fingerprint": initial_state_fingerprint,
            "final_state_fingerprint": final_state_fingerprint,
            "optimizer": v53e.OPTIMIZER,
            "optimizer_steps": v53e.FIXED_OPTIMIZER_STEPS,
            "group_counts": group_counts,
            "group_weights": {
                group: v53e.TRAIN_GROUP_WEIGHT for group in v53e.TRAIN_GROUPS
            },
            "initial_total_loss": initial_loss,
            "final_total_loss": final_loss,
            "final_group_losses": final_group_losses,
            "max_preclip_gradient_norm": max_preclip_gradient_norm,
            "gradient_clip_global_norm": v53e.GRADIENT_CLIP_GLOBAL_NORM,
            "finite_initial_parameters": True,
            "finite_losses": True,
            "finite_gradients": True,
            "finite_post_step_parameters": True,
            "checkpoint_write": False,
            "protected_evaluation_opened": False,
            "runtime_authority_changed": False,
            "production_promotion": False,
        }
        return model, evidence
    finally:
        torch.use_deterministic_algorithms(previous_determinism)


def checkpoint_write_allowed() -> bool:
    return False


def protected_evaluation_access_allowed() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False


def authoritative_colab_execution_available() -> bool:
    return False


def future_gate_order() -> tuple[str, ...]:
    return (
        "v5_3f_exact_ci_green_sha",
        "separately_authorized_exact_train_tensor_materialization_and_single_execution",
        "single_candidate_numerical_and_state_isolation",
        "train_v5_f1_and_frozen_correct_retention",
        "historical_validation_retention_at_frozen_thresholds",
        "immutable_v5_first30_diagnostic",
        "separately_authorized_v5_validation",
        "separately_authorized_final_holdout",
    )
