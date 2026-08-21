"""Deterministic shadow-only Meter V3-A3 residual calibration screen.

V3-A3 never trains a network. It freezes the completed V3-A2 shadow model and
selects one bounded pair of class-conditional residual gains from REAL TRAIN
only. Held-out REAL validation is evaluated only after the gain pair is fixed.
D10 source-retention is deliberately deferred until the REAL-only screen passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

from .meter_teacher_gold_admission_v1 import METER_CLASSES


METER_REAL_DOMAIN_ADAPTATION_V3_A3: Final[str] = (
    "meter-real-domain-adaptation-v3-a3-residual-calibration-screen"
)
PARENT_ADAPTATION_VERSION_V3_A3: Final[str] = (
    "meter-real-domain-adaptation-v3-a2-positive-margin"
)
PARENT_REPOSITORY_SHA_V3_A3: Final[str] = (
    "2e3247d33d7d516a4def2aec87447ae7355e7e9d"
)
PRESENCE_D11_SHA256_V3_A3: Final[str] = (
    "cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3"
)
GAIN_MIN_MILLI_V3_A3: Final[int] = 1_000
GAIN_MAX_MILLI_V3_A3: Final[int] = 1_250
GAIN_STEP_MILLI_V3_A3: Final[int] = 25
GAIN_GRID_SIZE_V3_A3: Final[int] = 121


class MeterRealDomainAdaptationV3A3Error(RuntimeError):
    """Raised when the V3-A3 bounded calibration contract is violated."""


@dataclass(frozen=True, slots=True)
class ClassificationSummaryV3A3:
    record_count: int
    macro_f1: float
    accuracy: float
    per_class_recall: Mapping[str, float]
    confusion: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.record_count, int) or isinstance(self.record_count, bool) or self.record_count <= 0:
            raise ValueError("record_count must be a positive integer")
        if not 0.0 <= self.macro_f1 <= 1.0 or not 0.0 <= self.accuracy <= 1.0:
            raise ValueError("classification rates must be in [0,1]")
        if tuple(self.per_class_recall) != tuple(METER_CLASSES):
            raise ValueError("per_class_recall must use canonical Meter class order")
        if len(self.confusion) != 4 or any(len(row) != 4 for row in self.confusion):
            raise ValueError("confusion must be 4x4")


@dataclass(frozen=True, slots=True)
class GainSelectionV3A3:
    gain_2_4_milli: int
    gain_4_4_milli: int
    train_summary: ClassificationSummaryV3A3

    def __post_init__(self) -> None:
        grid = set(gain_grid_milli_v3_a3())
        if self.gain_2_4_milli not in grid or self.gain_4_4_milli not in grid:
            raise ValueError("selected V3-A3 gains must come from the frozen grid")


def gain_grid_milli_v3_a3() -> tuple[int, ...]:
    values = tuple(range(GAIN_MIN_MILLI_V3_A3, GAIN_MAX_MILLI_V3_A3 + 1, GAIN_STEP_MILLI_V3_A3))
    if len(values) != 11 or values[0] != 1_000 or values[-1] != 1_250:
        raise MeterRealDomainAdaptationV3A3Error("V3-A3 gain grid drifted")
    return values


def gain_pairs_milli_v3_a3() -> tuple[tuple[int, int], ...]:
    grid = gain_grid_milli_v3_a3()
    pairs = tuple((gain_2_4, gain_4_4) for gain_2_4 in grid for gain_4_4 in grid)
    if len(pairs) != GAIN_GRID_SIZE_V3_A3:
        raise MeterRealDomainAdaptationV3A3Error("V3-A3 gain-pair cardinality drifted")
    return pairs


def verify_parent_resume_metadata_v3_a3(snapshot: Mapping[str, object]) -> None:
    """Fail closed unless the supplied resume is the exact completed V3-A2 parent."""
    expected = {
        "adaptation_version": PARENT_ADAPTATION_VERSION_V3_A3,
        "repository_sha": PARENT_REPOSITORY_SHA_V3_A3,
        "base_checkpoint_sha256": PRESENCE_D11_SHA256_V3_A3,
        "completed_epoch": 20,
        "best_epoch": 20,
    }
    for name, value in expected.items():
        if snapshot.get(name) != value:
            raise MeterRealDomainAdaptationV3A3Error(
                f"V3-A3 parent resume {name} mismatch"
            )
    if "current_model_state" not in snapshot:
        raise MeterRealDomainAdaptationV3A3Error("V3-A3 parent resume lacks current_model_state")


def calibrated_logits_v3_a3(
    base_logits,
    adapter_logits,
    *,
    gain_2_4_milli: int,
    gain_4_4_milli: int,
):
    """Apply the frozen class-conditional residual gain rule.

    none and 3/4 remain exactly identity-gain. Only 2/4 and 4/4 residuals may
    increase within the preregistered [1.000, 1.250] grid.
    """
    import torch

    grid = set(gain_grid_milli_v3_a3())
    if gain_2_4_milli not in grid or gain_4_4_milli not in grid:
        raise ValueError("V3-A3 gains must come from the frozen grid")
    if base_logits.ndim != 2 or adapter_logits.ndim != 2 or base_logits.shape != adapter_logits.shape:
        raise ValueError("base_logits and adapter_logits must have matching [B,4] shape")
    if base_logits.shape[1] != 4:
        raise ValueError("V3-A3 logits must have four Meter classes")
    gains = torch.tensor(
        [1.0, gain_2_4_milli / 1000.0, 1.0, gain_4_4_milli / 1000.0],
        dtype=adapter_logits.dtype,
        device=adapter_logits.device,
    )
    return base_logits + adapter_logits * gains.unsqueeze(0)


def classification_summary_v3_a3(
    true_classes: Sequence[int],
    predicted_classes: Sequence[int],
) -> ClassificationSummaryV3A3:
    if len(true_classes) != len(predicted_classes) or not true_classes:
        raise ValueError("true/predicted class sequences must be non-empty and equal length")
    confusion = [[0, 0, 0, 0] for _ in range(4)]
    for true_class, predicted_class in zip(true_classes, predicted_classes):
        if true_class not in {0, 1, 2, 3} or predicted_class not in {0, 1, 2, 3}:
            raise ValueError("Meter class indices must be in [0,3]")
        confusion[true_class][predicted_class] += 1

    recalls: dict[str, float] = {}
    f1_values: list[float] = []
    correct = 0
    for class_index, label in enumerate(METER_CLASSES):
        true_positive = confusion[class_index][class_index]
        correct += true_positive
        actual = sum(confusion[class_index])
        predicted = sum(confusion[row][class_index] for row in range(4))
        recall = true_positive / actual if actual else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)
        recalls[label] = recall
        f1_values.append(f1)

    return ClassificationSummaryV3A3(
        record_count=len(true_classes),
        macro_f1=sum(f1_values) / 4.0,
        accuracy=correct / len(true_classes),
        per_class_recall=recalls,
        confusion=tuple(tuple(int(value) for value in row) for row in confusion),
    )


def _selection_rank_v3_a3(
    gain_2_4_milli: int,
    gain_4_4_milli: int,
    summary: ClassificationSummaryV3A3,
) -> tuple[float, ...]:
    min_positive_recall = min(summary.per_class_recall[label] for label in METER_CLASSES[1:])
    total_deviation = (gain_2_4_milli - 1_000) + (gain_4_4_milli - 1_000)
    maximum_gain = max(gain_2_4_milli, gain_4_4_milli)
    return (
        min_positive_recall,
        summary.macro_f1,
        summary.accuracy,
        -float(total_deviation),
        -float(maximum_gain),
        -float(gain_2_4_milli),
        -float(gain_4_4_milli),
    )


def select_gain_pair_v3_a3(
    *,
    parent_train_summary: ClassificationSummaryV3A3,
    candidate_train_summaries: Mapping[tuple[int, int], ClassificationSummaryV3A3],
) -> GainSelectionV3A3:
    expected_pairs = set(gain_pairs_milli_v3_a3())
    if set(candidate_train_summaries) != expected_pairs:
        raise MeterRealDomainAdaptationV3A3Error("V3-A3 candidate grid is incomplete or altered")
    parent_none = parent_train_summary.per_class_recall["none"]
    eligible = [
        (pair, summary)
        for pair, summary in candidate_train_summaries.items()
        if summary.per_class_recall["none"] + 1e-12 >= parent_none
    ]
    if not eligible:
        raise MeterRealDomainAdaptationV3A3Error("no V3-A3 gain pair preserves REAL TRAIN none recall")
    (gain_2_4, gain_4_4), summary = max(
        eligible,
        key=lambda item: _selection_rank_v3_a3(item[0][0], item[0][1], item[1]),
    )
    return GainSelectionV3A3(
        gain_2_4_milli=gain_2_4,
        gain_4_4_milli=gain_4_4,
        train_summary=summary,
    )


def real_phase0_gate_v3_a3(summary: ClassificationSummaryV3A3) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if summary.macro_f1 < 0.900:
        reasons.append("REAL_MACRO_F1_BELOW_90_PERCENT")
    if summary.accuracy < 0.900:
        reasons.append("REAL_ACCURACY_BELOW_90_PERCENT")
    if summary.per_class_recall["none"] < 8.0 / 9.0:
        reasons.append("REAL_NONE_RECALL_BELOW_8_OF_9")
    for label in METER_CLASSES[1:]:
        if summary.per_class_recall[label] < 0.999:
            reasons.append(f"REAL_{label.replace('/', '_')}_RECALL_NOT_3_OF_3")
    return (not reasons, tuple(reasons))


def sealed_test_access_allowed() -> bool:
    return False


def runtime_connection_allowed() -> bool:
    return False


def resolver_connection_allowed() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False
