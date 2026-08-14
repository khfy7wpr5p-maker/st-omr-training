"""Stage 7-D9 declarative Structure refinement contract.

D9 freezes the architecture selected from the accepted D8 validation-only
report before any new optimizer run.  It preserves the external D4
``StructureSet`` task contract while decomposing the weak sparse channels into
bounded local specialists.  This module is declarative only: it contains no
model, optimizer, trainer, checkpoint loader, dataloader, or TEST evaluation
path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Final

from .stage7d4_specialist_architecture import V1_SPECIALIST_TASKS


STAGE7D9_VERSION: Final[str] = "stage7d9-structure-refinement-contract-v1"
STAGE7D9_SCHEMA: Final[str] = "stage7d9-structure-refinement-contract-v1"

EXPECTED_D8_REPOSITORY_SHA: Final[str] = (
    "e0e721bf5a6d13025546fdf5eeb755647eef383f"
)
EXPECTED_D8_REPORT_SHA256: Final[str] = (
    "46de5f6766f78bb567f70794a364ccd44835d09af94ef29c3f1eab5cd13ce968"
)
EXPECTED_D7_STRUCTURE_STATE_SHA256: Final[str] = (
    "0d11b2ae414959b678ccc22a6b8cfcc1edc1ecadc3c73ed6ab5a0cda6e593907"
)

D8_BASELINE_DICE: Final[dict[str, float]] = {
    "system_region": 0.9304226398,
    "measure_region": 0.8449312699,
    "barline": 0.2736204205,
    "clef_g2": 0.8286431574,
    "meter_2_4": 0.3481060606,
    "meter_3_4": 0.3528485803,
    "meter_4_4": 0.3103351169,
}
D8_TOLERANT_F1_2PX: Final[dict[str, float]] = {
    "system_region": 0.9898584521,
    "measure_region": 0.9793445289,
    "barline": 0.3670878904,
    "clef_g2": 0.9441274383,
    "meter_2_4": 0.3892495018,
    "meter_3_4": 0.4025582667,
    "meter_4_4": 0.3547076746,
}

METER_CLASSES: Final[tuple[str, ...]] = ("none", "2/4", "3/4", "4/4")
STRUCTURE_CORE_CHANNELS: Final[tuple[str, ...]] = (
    "system_region",
    "measure_region",
    "clef_g2",
)


@dataclass(frozen=True, slots=True)
class LocalRoiPolicy:
    """Geometry-conditioned native-image crop policy for one local specialist."""

    policy_id: str
    anchor: str
    x_before_staff_spacings_milli: int
    x_after_staff_spacings_milli: int
    y_before_staff_spacings_milli: int
    y_after_staff_spacings_milli: int
    output_height: int
    output_width: int
    resize_mode: str = "fit-pad-preserve-aspect-v1"

    def __post_init__(self) -> None:
        if not self.policy_id or not self.policy_id.isascii():
            raise ValueError("policy_id must be non-empty ASCII")
        if self.anchor not in {"measure_start", "measure_end"}:
            raise ValueError("ROI anchor must be measure_start or measure_end")
        for name, value in (
            ("x_before_staff_spacings_milli", self.x_before_staff_spacings_milli),
            ("x_after_staff_spacings_milli", self.x_after_staff_spacings_milli),
            ("y_before_staff_spacings_milli", self.y_before_staff_spacings_milli),
            ("y_after_staff_spacings_milli", self.y_after_staff_spacings_milli),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 32_000:
                raise ValueError(f"{name} is outside D9 bounds")
        if not isinstance(self.output_height, int) or isinstance(self.output_height, bool):
            raise ValueError("output_height must be a plain integer")
        if not isinstance(self.output_width, int) or isinstance(self.output_width, bool):
            raise ValueError("output_width must be a plain integer")
        if not 64 <= self.output_height <= 512 or not 64 <= self.output_width <= 1024:
            raise ValueError("D9 ROI output dimensions are outside bounds")
        if self.resize_mode != "fit-pad-preserve-aspect-v1":
            raise ValueError("D9 ROI resize mode is frozen")


@dataclass(frozen=True, slots=True)
class RefinementComponentContract:
    """One independently auditable internal component of StructureSet."""

    component_id: str
    responsibility: str
    depends_on: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    trainable: bool
    accepted_d7_weights_frozen: bool
    max_trainable_parameters: int
    roi_policy_id: str | None

    def __post_init__(self) -> None:
        if not self.component_id or not self.component_id.isascii():
            raise ValueError("component_id must be non-empty ASCII")
        if not self.responsibility:
            raise ValueError("responsibility must be non-empty")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("depends_on entries must be unique")
        if not self.inputs or not self.outputs:
            raise ValueError("component inputs/outputs must be non-empty")
        if len(set(self.inputs)) != len(self.inputs) or len(set(self.outputs)) != len(self.outputs):
            raise ValueError("component inputs/outputs must be unique")
        if not isinstance(self.trainable, bool) or not isinstance(self.accepted_d7_weights_frozen, bool):
            raise ValueError("component train/freeze flags must be bool")
        if (
            not isinstance(self.max_trainable_parameters, int)
            or isinstance(self.max_trainable_parameters, bool)
            or not 0 <= self.max_trainable_parameters <= 2_000_000
        ):
            raise ValueError("component parameter budget is outside D9 bounds")
        if not self.trainable and self.max_trainable_parameters != 0:
            raise ValueError("non-trainable component must expose zero trainable-parameter budget")
        if self.roi_policy_id is not None and (
            not isinstance(self.roi_policy_id, str) or not self.roi_policy_id.isascii()
        ):
            raise ValueError("roi_policy_id must be ASCII or None")


BARLINE_ROI: Final[LocalRoiPolicy] = LocalRoiPolicy(
    policy_id="measure-end-barline-roi-v1",
    anchor="measure_end",
    x_before_staff_spacings_milli=5_000,
    x_after_staff_spacings_milli=1_500,
    y_before_staff_spacings_milli=3_000,
    y_after_staff_spacings_milli=3_000,
    output_height=192,
    output_width=128,
)

METER_ROI: Final[LocalRoiPolicy] = LocalRoiPolicy(
    policy_id="measure-start-meter-roi-v1",
    anchor="measure_start",
    x_before_staff_spacings_milli=500,
    x_after_staff_spacings_milli=12_000,
    y_before_staff_spacings_milli=3_000,
    y_after_staff_spacings_milli=3_000,
    output_height=192,
    output_width=256,
)

ROI_POLICIES: Final[tuple[LocalRoiPolicy, ...]] = (BARLINE_ROI, METER_ROI)

STRUCTURE_REFINEMENT_COMPONENTS: Final[tuple[RefinementComponentContract, ...]] = (
    RefinementComponentContract(
        component_id="structure_core",
        responsibility=(
            "preserve the accepted D7 system-region, measure-region, and G2-clef path without mutation"
        ),
        depends_on=("staff_geometry",),
        inputs=("grayscale_score_region", "staff_geometry_candidates"),
        outputs=("system_bboxes", "measure_bboxes", "clef_g2_candidate", "confidence"),
        trainable=False,
        accepted_d7_weights_frozen=True,
        max_trainable_parameters=0,
        roi_policy_id=None,
    ),
    RefinementComponentContract(
        component_id="barline_refiner",
        responsibility=(
            "localize the trailing barline segment from a high-resolution measure-end crop conditioned on staff geometry"
        ),
        depends_on=("structure_core", "staff_geometry"),
        inputs=("measure_end_roi", "five_staff_lines", "measure_bbox"),
        outputs=("barline_segment", "confidence"),
        trainable=True,
        accepted_d7_weights_frozen=True,
        max_trainable_parameters=500_000,
        roi_policy_id=BARLINE_ROI.policy_id,
    ),
    RefinementComponentContract(
        component_id="meter_refiner",
        responsibility=(
            "classify visible current-measure meter as none/2-4/3-4/4-4 and localize its bbox from a measure-start crop"
        ),
        depends_on=("structure_core", "staff_geometry"),
        inputs=("measure_start_roi", "five_staff_lines", "measure_bbox"),
        outputs=("meter_class", "meter_bbox", "confidence"),
        trainable=True,
        accepted_d7_weights_frozen=True,
        max_trainable_parameters=750_000,
        roi_policy_id=METER_ROI.policy_id,
    ),
    RefinementComponentContract(
        component_id="structure_fusion",
        responsibility=(
            "deterministically fuse frozen core plus local refiners into the unchanged external StructureSet outputs"
        ),
        depends_on=("structure_core", "barline_refiner", "meter_refiner"),
        inputs=(
            "system_bboxes",
            "measure_bboxes",
            "clef_g2_candidate",
            "barline_segment",
            "meter_class",
            "meter_bbox",
            "component_confidences",
        ),
        outputs=(
            "system_bboxes",
            "measure_bboxes",
            "barline_positions",
            "clef_g2_candidate",
            "meter_candidate",
            "confidence",
        ),
        trainable=False,
        accepted_d7_weights_frozen=True,
        max_trainable_parameters=0,
        roi_policy_id=None,
    ),
)


@dataclass(frozen=True, slots=True)
class D9AcceptancePolicy:
    """Frozen pre-training validation gates for the future D9 optimizer run."""

    test_records: int = 0
    core_model_mutation_allowed: bool = False
    barline_min_strict_dice_milli: int = 500
    barline_min_tolerant_f1_2px_milli: int = 700
    meter_min_macro_f1_milli: int = 800
    meter_min_positive_localization_f1_2px_milli: int = 600
    max_total_new_trainable_parameters: int = 1_250_000

    def __post_init__(self) -> None:
        if self.test_records != 0:
            raise ValueError("D9 TEST surface must stay sealed")
        if self.core_model_mutation_allowed is not False:
            raise ValueError("accepted D7 Structure core must remain frozen in D9")
        for name, value in (
            ("barline_min_strict_dice_milli", self.barline_min_strict_dice_milli),
            ("barline_min_tolerant_f1_2px_milli", self.barline_min_tolerant_f1_2px_milli),
            ("meter_min_macro_f1_milli", self.meter_min_macro_f1_milli),
            ("meter_min_positive_localization_f1_2px_milli", self.meter_min_positive_localization_f1_2px_milli),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 1000:
                raise ValueError(f"{name} must be bounded milli-units")
        if (
            not isinstance(self.max_total_new_trainable_parameters, int)
            or isinstance(self.max_total_new_trainable_parameters, bool)
            or not 1 <= self.max_total_new_trainable_parameters <= 2_000_000
        ):
            raise ValueError("D9 total parameter budget is outside bounds")


D9_ACCEPTANCE: Final[D9AcceptancePolicy] = D9AcceptancePolicy()


def _external_structure_outputs() -> tuple[str, ...]:
    structures = tuple(task for task in V1_SPECIALIST_TASKS if task.task_id == "structure")
    if len(structures) != 1:
        raise RuntimeError("D4 Structure task identity is not unique")
    return structures[0].outputs


def stage7d9_contract_payload() -> dict[str, object]:
    """Return the canonical frozen D9 architecture payload."""
    return {
        "schema_version": STAGE7D9_SCHEMA,
        "stage7d9_version": STAGE7D9_VERSION,
        "evidence": {
            "d8_repository_sha": EXPECTED_D8_REPOSITORY_SHA,
            "d8_report_sha256": EXPECTED_D8_REPORT_SHA256,
            "accepted_d7_structure_state_sha256": EXPECTED_D7_STRUCTURE_STATE_SHA256,
            "d8_baseline_dice": dict(sorted(D8_BASELINE_DICE.items())),
            "d8_tolerant_f1_2px": dict(sorted(D8_TOLERANT_F1_2PX.items())),
        },
        "external_contract": {
            "task_id": "structure",
            "dataset_name": "StructureSet",
            "outputs": _external_structure_outputs(),
            "contract_change": False,
        },
        "meter_classes": METER_CLASSES,
        "structure_core_channels": STRUCTURE_CORE_CHANNELS,
        "roi_policies": [asdict(policy) for policy in ROI_POLICIES],
        "components": [asdict(component) for component in STRUCTURE_REFINEMENT_COMPONENTS],
        "acceptance": asdict(D9_ACCEPTANCE),
        "split_policy": "train-new-refiners-validation-readonly-test-forbidden",
        "ground_truth_policy": (
            "synthetic labels remain canonical-music plus pinned-renderer geometry plus deterministic transform"
        ),
        "fusion_policy": "deterministic-fail-closed-no-invention-v1",
    }


def stage7d9_contract_fingerprint() -> str:
    raw = json.dumps(
        stage7d9_contract_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256(raw).hexdigest()
