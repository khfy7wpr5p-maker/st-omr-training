"""Training-disabled Meter V5-3E fixed rescue-training preregistration.

V5-3D fixed the gated 64->8->1 tanh rescue topology and its TRAIN-only
eligibility surface. V5-3E freezes exactly one optimization recipe for the two
independent rescue specialists. This module is declarative: it contains no
model construction, autograd, optimizer execution, checkpoint write, protected
evaluation access, threshold search, or runtime promotion path.
"""
from __future__ import annotations

from typing import Final

from . import meter_v5_3d_gated_rescue_architecture_contract_v1 as v53d


SCHEMA: Final[str] = "st-omr-meter-v5-3e-fixed-rescue-training-preregistration-v1"
V53D_HEAD_SHA: Final[str] = "7d50fbec4d730aa46c69f7dfa3a20917a3478ef8"
V53D_MODULE_BLOB_SHA: Final[str] = "f74ff71e9999f889086c3cf68a9c6ed5a0e69427"
V53D_DOC_BLOB_SHA: Final[str] = "3be6b10fb18cfee049de05a0c5e21a253990f4c8"

RECIPE_ID: Final[str] = "meter-v5-3e-rescue-fixed-recipe-v1"
CANDIDATE_CONFIGURATION_COUNT: Final[int] = 1

RESCUE_SPECIALISTS: Final[tuple[str, ...]] = v53d.RESCUE_SPECIALISTS
FEATURE_DIM: Final[int] = v53d.FEATURE_DIM
HIDDEN_WIDTH: Final[int] = v53d.HIDDEN_WIDTH
ACTIVATION: Final[str] = v53d.ACTIVATION
RESCUE_THRESHOLD: Final[float] = v53d.RESCUE_THRESHOLD
PARAMETERS_PER_RESCUE: Final[int] = v53d.PARAMETERS_PER_RESCUE
TRAIN_GROUPS: Final[tuple[str, ...]] = v53d.TRAIN_GROUPS
TRAIN_GROUP_WEIGHT: Final[float] = v53d.TRAIN_GROUP_WEIGHT
EXPECTED_TRAIN_GROUP_COUNTS: Final[dict[str, dict[str, int]]] = {
    digit: dict(counts) for digit, counts in v53d.EXPECTED_TRAIN_GROUP_COUNTS.items()
}

LOSS: Final[str] = "binary_cross_entropy_with_logits"
LOSS_REDUCTION: Final[str] = "mean_per_group_then_equal_weight_sum"
POS_WEIGHT: Final[float] = 1.0
LABEL_SMOOTHING: Final[float] = 0.0

OPTIMIZER: Final[str] = "AdamW"
LEARNING_RATE: Final[float] = 1e-3
WEIGHT_DECAY: Final[float] = 1e-4
BETA1: Final[float] = 0.9
BETA2: Final[float] = 0.999
EPSILON: Final[float] = 1e-8
FIXED_OPTIMIZER_STEPS: Final[int] = 110
GRADIENT_CLIP_GLOBAL_NORM: Final[float] = 1.0
SCHEDULER: Final[str] = "none"
WARMUP_STEPS: Final[int] = 0
EARLY_STOPPING: Final[bool] = False

SEED: Final[int] = 52023
INITIALIZATION: Final[str] = "xavier_uniform_weights_zero_bias"
INITIALIZATION_GAIN: Final[float] = 1.0
SAME_INITIAL_PARAMETER_REALIZATION_PER_SPECIALIST: Final[bool] = True
DEVICE: Final[str] = "cpu"
DTYPE: Final[str] = "float32"
DETERMINISTIC_ALGORITHMS: Final[bool] = True
AMP_ENABLED: Final[bool] = False

CANONICAL_ROW_ORDER: Final[str] = "stable_manifest_identity_ascending"
OBJECTIVE_ROWS: Final[str] = "all_exact_rows_in_each_of_four_frozen_negative_groups"
BATCHING: Final[str] = "full_group_objective_each_optimizer_step"
SHUFFLE: Final[bool] = False
SAMPLING_WITH_REPLACEMENT: Final[bool] = False

FIXED_FINAL_STEP_ONLY: Final[bool] = True
HYPERPARAMETER_SWEEP: Final[bool] = False
THRESHOLD_SEARCH: Final[bool] = False
ARCHITECTURE_SEARCH: Final[bool] = False
AUTOMATIC_SECOND_CONFIGURATION: Final[bool] = False
FALLBACK_OPTIMIZER: Final[bool] = False


class MeterV5_3EContractError(RuntimeError):
    """Raised when the fixed V5-3E training preregistration is violated."""


def prerequisite_contract() -> dict[str, object]:
    return {
        "v5_3d_head_sha": V53D_HEAD_SHA,
        "v5_3d_module_blob_sha": V53D_MODULE_BLOB_SHA,
        "v5_3d_doc_blob_sha": V53D_DOC_BLOB_SHA,
        "v5_3d_schema": v53d.SCHEMA,
        "architecture_contract_inherited_without_change": True,
        "rescue_specialists": RESCUE_SPECIALISTS,
        "feature_dim": FEATURE_DIM,
        "hidden_width": HIDDEN_WIDTH,
        "activation": ACTIVATION,
        "rescue_threshold": RESCUE_THRESHOLD,
        "parameters_per_rescue": PARAMETERS_PER_RESCUE,
        "digit4_frozen": True,
        "frozen_specialist_tensors_authoritative": True,
    }


def objective_contract() -> dict[str, object]:
    return {
        "eligible_rows": "same-specialist-frozen-negative-only",
        "data_surfaces": ("v5_train", "historical_train"),
        "groups": TRAIN_GROUPS,
        "group_weights": {group: TRAIN_GROUP_WEIGHT for group in TRAIN_GROUPS},
        "group_weight_sum": sum(TRAIN_GROUP_WEIGHT for _ in TRAIN_GROUPS),
        "expected_group_counts": {
            digit: dict(counts) for digit, counts in EXPECTED_TRAIN_GROUP_COUNTS.items()
        },
        "loss": LOSS,
        "loss_reduction": LOSS_REDUCTION,
        "positive_weight": POS_WEIGHT,
        "label_smoothing": LABEL_SMOOTHING,
        "objective_rows": OBJECTIVE_ROWS,
        "canonical_row_order": CANONICAL_ROW_ORDER,
        "batching": BATCHING,
        "shuffle": SHUFFLE,
        "sampling_with_replacement": SAMPLING_WITH_REPLACEMENT,
    }


def fixed_training_recipe() -> dict[str, object]:
    return {
        "recipe_id": RECIPE_ID,
        "candidate_configuration_count": CANDIDATE_CONFIGURATION_COUNT,
        "specialists_trained_independently": RESCUE_SPECIALISTS,
        "frozen_backbone": True,
        "frozen_head_weight": True,
        "frozen_head_bias": True,
        "digit4_frozen": True,
        "initialization": INITIALIZATION,
        "initialization_gain": INITIALIZATION_GAIN,
        "seed": SEED,
        "same_initial_parameter_realization_per_specialist": (
            SAME_INITIAL_PARAMETER_REALIZATION_PER_SPECIALIST
        ),
        "device": DEVICE,
        "dtype": DTYPE,
        "deterministic_algorithms": DETERMINISTIC_ALGORITHMS,
        "amp_enabled": AMP_ENABLED,
        "optimizer": OPTIMIZER,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "betas": (BETA1, BETA2),
        "epsilon": EPSILON,
        "fixed_optimizer_steps": FIXED_OPTIMIZER_STEPS,
        "gradient_clip_global_norm": GRADIENT_CLIP_GLOBAL_NORM,
        "scheduler": SCHEDULER,
        "warmup_steps": WARMUP_STEPS,
        "early_stopping": EARLY_STOPPING,
        "fixed_final_step_only": FIXED_FINAL_STEP_ONLY,
        "hyperparameter_sweep": HYPERPARAMETER_SWEEP,
        "threshold_search": THRESHOLD_SEARCH,
        "architecture_search": ARCHITECTURE_SEARCH,
        "automatic_second_configuration": AUTOMATIC_SECOND_CONFIGURATION,
        "fallback_optimizer": FALLBACK_OPTIMIZER,
    }


def numerical_execution_guards() -> dict[str, object]:
    return {
        "finite_input_features_required": True,
        "finite_initial_parameters_required": True,
        "finite_loss_each_step_required": True,
        "finite_gradients_each_step_required": True,
        "finite_parameters_after_step_required": True,
        "gradient_global_norm_clip": GRADIENT_CLIP_GLOBAL_NORM,
        "abort_on_nonfinite": True,
        "abort_writes_checkpoint": False,
        "frozen_tensor_bit_identity_required": True,
        "only_rescue_namespace_may_change": True,
    }


def protected_surface_contract() -> dict[str, object]:
    return {
        "historical_validation_opened": False,
        "first30_opened": False,
        "v5_reserve_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "bbox_access_added": False,
        "crop_geometry_change": False,
        "spatial_heuristic_change": False,
        "threshold_tuning": False,
        "resolver_wiring": False,
        "production_promotion": False,
    }


def safety_boundary() -> dict[str, object]:
    return {
        "preregistration_only": True,
        "training_implementation_present": False,
        "training_authorized": False,
        "training_executed": False,
        "autograd_used": False,
        "backward": False,
        "optimizer_steps_executed": 0,
        "checkpoint_write": False,
        "rescue_artifact_write": False,
        "frozen_model_mutation": False,
        "retention_executed": False,
        **protected_surface_contract(),
    }


def future_gate_order() -> tuple[str, ...]:
    return (
        "v5_3e_exact_ci_green_sha",
        "single_fixed_recipe_execution_harness",
        "single_candidate_numerical_and_state_isolation",
        "train_v5_f1_and_frozen_correct_retention",
        "historical_validation_retention_at_frozen_thresholds",
        "immutable_v5_first30_diagnostic",
        "separately_authorized_v5_validation",
        "separately_authorized_final_holdout",
    )


def training_entry_point_available() -> bool:
    return False


def checkpoint_write_allowed() -> bool:
    return False


def protected_evaluation_access_allowed() -> bool:
    return False
