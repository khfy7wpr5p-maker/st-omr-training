"""Stage 7-D13 NoteHead/Rest/Accidental specialist training contract.

D13 freezes the post-D12 data-readiness, deterministic measure derivative,
three-specialist model/optimizer, metric and acceptance policy *before* any
optimizer step is allowed.  The actual derivative builder, models and training
runner are implemented in later controlled D13 sub-stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Final


STAGE7D13_VERSION: Final[str] = "stage7d13-symbol-specialist-training-contract-v1"
STAGE7D13_SCHEMA: Final[str] = "stage7d13-symbol-specialist-training-contract-v1"

EXPECTED_D12_REPOSITORY_SHA: Final[str] = (
    "e2de6f64c27be2dd6d706a700553ef4f5c236e25"
)
EXPECTED_D12_DERIVATIVE_BUILD_ID: Final[str] = (
    "35323e831c5c693bf607808c5f846624445bf537f30e1d93db9ca949a7eed106"
)
EXPECTED_D12_MANIFEST_SHA256: Final[str] = (
    "a372eba640b38704020922ad4eb102738fc4492d278a38e4b51b8ad0b78d4ea1"
)
EXPECTED_D12_ARTIFACT_BINDING_SHA256: Final[str] = (
    "14c64e16ca2f993bf94f8009bf0bcd974b7ddee87c19bb748219ba3f774b229d"
)
EXPECTED_SOURCE_SAMPLE_COUNTS: Final[dict[str, int]] = {
    "train": 1230,
    "validation": 153,
}
EXPECTED_SOURCE_FAMILY_COUNTS: Final[dict[str, int]] = {
    "train": 410,
    "validation": 51,
}
TEST_SPECIALIST_RECORDS: Final[int] = 0

D12_CLASS_INVENTORY: Final[dict[str, dict[str, dict[str, int]]]] = {
    "train": {
        "notehead": {"open": 7935, "filled": 30399},
        "rest": {"half": 1998, "quarter": 3417, "eighth": 5187},
        "accidental": {"sharp": 10665, "flat": 10596, "natural": 1131},
    },
    "validation": {
        "notehead": {"open": 1128, "filled": 4104},
        "rest": {"half": 162, "quarter": 327, "eighth": 480},
        "accidental": {"sharp": 1566, "flat": 1575, "natural": 189},
    },
}

SPECIALIST_CLASSES: Final[dict[str, tuple[str, ...]]] = {
    "notehead": ("open", "filled"),
    "rest": ("half", "quarter", "eighth"),
    "accidental": ("sharp", "flat", "natural"),
}

MIN_TRAIN_INSTANCES_PER_CLASS: Final[int] = 1000
MIN_VALIDATION_INSTANCES_PER_CLASS: Final[int] = 150
CLASS_WEIGHT_MIN_MILLI: Final[int] = 500
CLASS_WEIGHT_MAX_MILLI: Final[int] = 3000

INPUT_WIDTH: Final[int] = 512
INPUT_HEIGHT: Final[int] = 128
OUTPUT_STRIDE: Final[int] = 4
WHITE_BACKGROUND: Final[int] = 255
MAX_PARAMETERS_PER_SPECIALIST: Final[int] = 1_500_000
MAX_PARAMETERS_COMBINED: Final[int] = 4_500_000


class Stage7D13ContractError(ValueError):
    """Raised when the frozen D13 pre-training contract is violated."""


@dataclass(frozen=True, slots=True)
class SpecialistAcceptance:
    class_aware_center_f1_4px_milli: int
    class_aware_bbox_f1_iou50_milli: int
    macro_class_f1_milli: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 1000:
                raise Stage7D13ContractError(
                    f"{name} must be an integer in [0, 1000]"
                )


ACCEPTANCE: Final[dict[str, SpecialistAcceptance]] = {
    "notehead": SpecialistAcceptance(850, 750, 900),
    "rest": SpecialistAcceptance(800, 700, 850),
    "accidental": SpecialistAcceptance(800, 700, 850),
}


@dataclass(frozen=True, slots=True)
class Stage7D13TrainingConfig:
    batch_size: int = 16
    epochs: int = 10
    learning_rate_micros: int = 700
    weight_decay_micros: int = 100
    grad_clip_milli: int = 1000
    master_seed: int = 713_013
    heartbeat_batches: int = 50
    optimizer: str = "adamw"
    objective: str = "weighted_positive_focal_plus_bbox_smoothl1_plus_offset_smoothl1_v1"
    checkpoint_selection: str = "min_validation_loss_per_specialist"
    decoder: str = "localmax3_score025_top256_greedy_v1"
    execution: str = "deterministic_pinned_cpu"

    def __post_init__(self) -> None:
        bounds = {
            "batch_size": (self.batch_size, 1, 64),
            "epochs": (self.epochs, 1, 64),
            "learning_rate_micros": (self.learning_rate_micros, 1, 100_000),
            "weight_decay_micros": (self.weight_decay_micros, 0, 100_000),
            "grad_clip_milli": (self.grad_clip_milli, 1, 100_000),
            "master_seed": (self.master_seed, 0, 2**63 - 1),
            "heartbeat_batches": (self.heartbeat_batches, 1, 10_000),
        }
        for name, (value, low, high) in bounds.items():
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise Stage7D13ContractError(f"{name} is outside the D13 boundary")
        expected = {
            "optimizer": "adamw",
            "objective": "weighted_positive_focal_plus_bbox_smoothl1_plus_offset_smoothl1_v1",
            "checkpoint_selection": "min_validation_loss_per_specialist",
            "decoder": "localmax3_score025_top256_greedy_v1",
            "execution": "deterministic_pinned_cpu",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise Stage7D13ContractError(f"{name} is frozen to {value!r}")


FROZEN_D13_CONFIG: Final[Stage7D13TrainingConfig] = Stage7D13TrainingConfig()


def class_readiness_violations(
    inventory: dict[str, dict[str, dict[str, int]]] = D12_CLASS_INVENTORY,
) -> tuple[str, ...]:
    """Return deterministic D13 class-readiness violations."""

    violations: list[str] = []
    for specialist, classes in SPECIALIST_CLASSES.items():
        for class_name in classes:
            train = inventory.get("train", {}).get(specialist, {}).get(class_name)
            validation = inventory.get("validation", {}).get(specialist, {}).get(class_name)
            if not isinstance(train, int) or isinstance(train, bool):
                violations.append(f"{specialist}.{class_name}: missing TRAIN count")
            elif train < MIN_TRAIN_INSTANCES_PER_CLASS:
                violations.append(
                    f"{specialist}.{class_name}: TRAIN {train} < {MIN_TRAIN_INSTANCES_PER_CLASS}"
                )
            if not isinstance(validation, int) or isinstance(validation, bool):
                violations.append(f"{specialist}.{class_name}: missing VALIDATION count")
            elif validation < MIN_VALIDATION_INSTANCES_PER_CLASS:
                violations.append(
                    f"{specialist}.{class_name}: VALIDATION {validation} < {MIN_VALIDATION_INSTANCES_PER_CLASS}"
                )
    return tuple(violations)


def positive_class_weights(
    specialist: str,
    inventory: dict[str, dict[str, dict[str, int]]] = D12_CLASS_INVENTORY,
) -> dict[str, float]:
    """Compute frozen TRAIN-only inverse-sqrt positive class weights.

    Weights are normalized to mean 1 within a specialist and clipped to the
    contract's [0.5, 3.0] interval.  Validation counts never participate.
    """

    classes = SPECIALIST_CLASSES.get(specialist)
    if classes is None:
        raise Stage7D13ContractError("unknown D13 specialist")
    train_counts = inventory.get("train", {}).get(specialist, {})
    raw: dict[str, float] = {}
    for class_name in classes:
        count = train_counts.get(class_name)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise Stage7D13ContractError(
                f"{specialist}.{class_name} requires a positive TRAIN count"
            )
        raw[class_name] = 1.0 / math.sqrt(count)
    mean = sum(raw.values()) / len(raw)
    lower = CLASS_WEIGHT_MIN_MILLI / 1000.0
    upper = CLASS_WEIGHT_MAX_MILLI / 1000.0
    return {
        class_name: min(upper, max(lower, raw[class_name] / mean))
        for class_name in classes
    }


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
        raise Stage7D13ContractError("D13 contract payload is not canonical JSON") from exc


def stage7d13_contract_payload() -> dict[str, object]:
    if class_readiness_violations():
        raise Stage7D13ContractError("accepted D12 inventory fails D13 readiness")
    weights = {
        specialist: positive_class_weights(specialist)
        for specialist in SPECIALIST_CLASSES
    }
    return {
        "schema": STAGE7D13_SCHEMA,
        "version": STAGE7D13_VERSION,
        "accepted_d12": {
            "repository_sha": EXPECTED_D12_REPOSITORY_SHA,
            "derivative_build_id": EXPECTED_D12_DERIVATIVE_BUILD_ID,
            "manifest_sha256": EXPECTED_D12_MANIFEST_SHA256,
            "artifact_binding_sha256": EXPECTED_D12_ARTIFACT_BINDING_SHA256,
            "sample_counts": EXPECTED_SOURCE_SAMPLE_COUNTS,
            "family_counts": EXPECTED_SOURCE_FAMILY_COUNTS,
            "class_inventory": D12_CLASS_INVENTORY,
            "test_specialist_records": TEST_SPECIALIST_RECORDS,
        },
        "readiness": {
            "min_train_instances_per_class": MIN_TRAIN_INSTANCES_PER_CLASS,
            "min_validation_instances_per_class": MIN_VALIDATION_INSTANCES_PER_CLASS,
        },
        "imbalance": {
            "policy": "train_only_inverse_sqrt_positive_weights_no_family_oversampling",
            "clip_milli": [CLASS_WEIGHT_MIN_MILLI, CLASS_WEIGHT_MAX_MILLI],
            "positive_class_weights": weights,
        },
        "measure_derivative": {
            "crop_authority": "accepted_d12_measure_bbox",
            "input_width": INPUT_WIDTH,
            "input_height": INPUT_HEIGHT,
            "letterbox_preserve_aspect_ratio": True,
            "white_background": WHITE_BACKGROUND,
            "test_derivatives": 0,
            "exact_record_counts": "freeze_after_independent_derivative_verification",
        },
        "model": {
            "family": "compact_fully_convolutional_center_detector_v1",
            "input_channels": 1,
            "output_stride": OUTPUT_STRIDE,
            "heads": ["class_heatmap", "bbox_size", "center_offset"],
            "classes": SPECIALIST_CLASSES,
            "separate_specialist_weights": True,
            "pretrained_external_backbone": False,
            "max_parameters_per_specialist": MAX_PARAMETERS_PER_SPECIALIST,
            "max_parameters_combined": MAX_PARAMETERS_COMBINED,
        },
        "training": asdict(FROZEN_D13_CONFIG),
        "validation_decoder": {
            "local_max_kernel": 3,
            "score_threshold_milli": 250,
            "top_k_per_measure": 256,
            "one_to_one_greedy_matching": True,
            "center_tolerance_px": 4,
            "bbox_iou_threshold_milli": 500,
        },
        "acceptance": {
            name: asdict(value) for name, value in ACCEPTANCE.items()
        },
        "split_policy": "family_exclusive_train_validation_test_forbidden",
        "optimizer_authorization": "blocked_until_d13_derivative_and_code_gates_pass",
    }


def stage7d13_contract_fingerprint() -> str:
    return sha256(_canonical_json_bytes(stage7d13_contract_payload())).hexdigest()
