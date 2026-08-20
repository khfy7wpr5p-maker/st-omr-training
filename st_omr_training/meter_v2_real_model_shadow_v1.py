"""Meter V2 real-model shadow evidence bridge.

This module binds the deterministic Meter V2 composer to the *identities* and
frozen validation thresholds of already-trained models, without loading model
binaries in CI and without wiring anything to the runtime Resolver.

The D11 four-class Meter baseline is used only as a temporary technical
presence bridge: an accepted D11 class is collapsed to ``present`` versus
``absent``.  The semantic D11 class is not passed to the V2 composer.

Digit model outputs are supplied per deterministic candidate slot as the
binary 2-AI / 3-AI / 4-AI scores.  Slot arbitration is deterministic: exactly
one passing specialist yields one visual digit observation; multiple passing
specialists yield AMBIGUOUS; malformed evidence yields REJECTED.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

from .meter_v2_deterministic_composer_v1 import (
    ACCEPTED,
    AMBIGUOUS,
    REJECTED,
    MeterBox,
    MeterCompositionResult,
    MeterDigitObservation,
    MeterPresenceObservation,
    compose_meter_v2,
)

D11_PRESENCE_BRIDGE_CHECKPOINT_SHA256: Final[str] = (
    "cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3"
)
D11_PRESENCE_BRIDGE_CLASSES: Final[tuple[str, ...]] = (
    "none",
    "2/4",
    "3/4",
    "4/4",
)
D11_PRESENCE_BRIDGE_STATUS: Final[str] = "TECHNICAL_BASELINE_ONLY"

R_D11_CLASS_INVALID: Final[str] = "METER_D11_PRESENCE_CLASS_INVALID"
R_D11_STATUS_AMBIGUOUS: Final[str] = "METER_D11_PRESENCE_AMBIGUOUS"
R_D11_STATUS_REJECTED: Final[str] = "METER_D11_PRESENCE_REJECTED"
R_SLOT_INVALID: Final[str] = "METER_DIGIT_SLOT_INVALID"
R_SLOT_SPECIALIST_CONFLICT: Final[str] = "METER_DIGIT_SLOT_SPECIALIST_CONFLICT"


@dataclass(frozen=True, slots=True)
class FrozenDigitSpecialistSpec:
    digit: int
    checkpoint_sha256: str
    threshold_milli: int
    validation_tp: int
    validation_fp: int
    validation_fn: int
    validation_tn: int

    def __post_init__(self) -> None:
        if self.digit not in (2, 3, 4):
            raise ValueError("frozen Meter V2 specialist digit must be 2, 3, or 4")
        if len(self.checkpoint_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.checkpoint_sha256
        ):
            raise ValueError("checkpoint_sha256 must be canonical lowercase SHA-256")
        if not isinstance(self.threshold_milli, int) or isinstance(self.threshold_milli, bool):
            raise ValueError("threshold_milli must be a plain integer")
        if not 0 <= self.threshold_milli <= 1000:
            raise ValueError("threshold_milli is outside [0,1000]")
        for value in (
            self.validation_tp,
            self.validation_fp,
            self.validation_fn,
            self.validation_tn,
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("validation confusion counts must be non-negative integers")

    @property
    def validation_recall(self) -> float:
        denominator = self.validation_tp + self.validation_fn
        return self.validation_tp / denominator if denominator else 0.0

    @property
    def validation_precision(self) -> float:
        denominator = self.validation_tp + self.validation_fp
        return self.validation_tp / denominator if denominator else 0.0

    @property
    def validation_f1(self) -> float:
        precision = self.validation_precision
        recall = self.validation_recall
        return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


FROZEN_DIGIT_SPECIALISTS: Final[tuple[FrozenDigitSpecialistSpec, ...]] = (
    FrozenDigitSpecialistSpec(
        digit=2,
        checkpoint_sha256="92b985d989e4338e3ae39b0a984879f4188be32c0d281390839117e1e9a715fa",
        threshold_milli=480,
        validation_tp=185,
        validation_fp=4,
        validation_fn=1,
        validation_tn=3182,
    ),
    FrozenDigitSpecialistSpec(
        digit=3,
        checkpoint_sha256="5ee45faf2efe0e2c83dbad716736d7ae16ad7251730431d368c10c4574836485",
        threshold_milli=600,
        validation_tp=203,
        validation_fp=0,
        validation_fn=1,
        validation_tn=3168,
    ),
    FrozenDigitSpecialistSpec(
        digit=4,
        checkpoint_sha256="dcd582b60b39e65798aa77aacea3cc797cd7513b7925151f0573be4aec6af43f",
        threshold_milli=470,
        validation_tp=788,
        validation_fp=23,
        validation_fn=4,
        validation_tn=2557,
    ),
)

_SPEC_BY_DIGIT: Final[dict[int, FrozenDigitSpecialistSpec]] = {
    spec.digit: spec for spec in FROZEN_DIGIT_SPECIALISTS
}


@dataclass(frozen=True, slots=True)
class MeterDigitSlotScores:
    """Externally computed real-model scores for one deterministic digit slot."""

    slot_id: str
    bbox: MeterBox
    score_2_milli: int
    score_3_milli: int
    score_4_milli: int


def _valid_milli(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= 1000
    )


def _valid_box(box: object) -> bool:
    if not isinstance(box, MeterBox):
        return False
    values = (box.x0, box.y0, box.x1, box.y1)
    return (
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        )
        and float(box.x0) < float(box.x1)
        and float(box.y0) < float(box.y1)
    )


def presence_from_d11_class(
    *,
    status: str,
    meter_class: str | None,
    confidence_milli: int,
) -> MeterPresenceObservation:
    """Collapse an already-accepted D11 class to visual presence only.

    No new presence threshold is invented here.  ``status`` is supplied by the
    external D11 shadow adapter.  This bridge is explicitly temporary and does
    not make D11 a product-quality Meter Presence specialist.
    """
    if not _valid_milli(confidence_milli):
        return MeterPresenceObservation(REJECTED, None, 0, (R_D11_CLASS_INVALID,))
    if status == REJECTED:
        return MeterPresenceObservation(REJECTED, None, confidence_milli, (R_D11_STATUS_REJECTED,))
    if status == AMBIGUOUS:
        return MeterPresenceObservation(AMBIGUOUS, None, confidence_milli, (R_D11_STATUS_AMBIGUOUS,))
    if status != ACCEPTED or meter_class not in D11_PRESENCE_BRIDGE_CLASSES:
        return MeterPresenceObservation(REJECTED, None, confidence_milli, (R_D11_CLASS_INVALID,))
    return MeterPresenceObservation(
        ACCEPTED,
        meter_class != "none",
        confidence_milli,
        (),
    )


def digit_observation_from_slot(
    slot: MeterDigitSlotScores,
) -> MeterDigitObservation | None:
    """Arbitrate one candidate slot across the frozen 2/3/4 binary specialists."""
    if not isinstance(slot, MeterDigitSlotScores) or not slot.slot_id or not _valid_box(slot.bbox):
        return MeterDigitObservation(
            "invalid-slot",
            REJECTED,
            None,
            0,
            None,
            (R_SLOT_INVALID,),
        )

    scores = {
        2: slot.score_2_milli,
        3: slot.score_3_milli,
        4: slot.score_4_milli,
    }
    if any(not _valid_milli(value) for value in scores.values()):
        return MeterDigitObservation(
            slot.slot_id,
            REJECTED,
            None,
            0,
            None,
            (R_SLOT_INVALID,),
        )

    passing = tuple(
        digit
        for digit in (2, 3, 4)
        if scores[digit] >= _SPEC_BY_DIGIT[digit].threshold_milli
    )
    if not passing:
        return None
    if len(passing) > 1:
        return MeterDigitObservation(
            slot.slot_id,
            AMBIGUOUS,
            None,
            max(scores[digit] for digit in passing),
            slot.bbox,
            (R_SLOT_SPECIALIST_CONFLICT,),
        )

    digit = passing[0]
    return MeterDigitObservation(
        slot.slot_id,
        ACCEPTED,
        digit,
        scores[digit],
        slot.bbox,
        (),
    )


def compose_meter_v2_shadow_from_model_evidence(
    *,
    d11_status: str,
    d11_meter_class: str | None,
    d11_confidence_milli: int,
    slots: tuple[MeterDigitSlotScores, ...],
) -> MeterCompositionResult:
    """Bridge supplied real-model evidence into the deterministic composer."""
    presence = presence_from_d11_class(
        status=d11_status,
        meter_class=d11_meter_class,
        confidence_milli=d11_confidence_milli,
    )
    if not isinstance(slots, tuple):
        rejected = MeterDigitObservation(
            "invalid-slot",
            REJECTED,
            None,
            0,
            None,
            (R_SLOT_INVALID,),
        )
        return compose_meter_v2(presence, (rejected,))

    observations: list[MeterDigitObservation] = []
    for slot in slots:
        observation = digit_observation_from_slot(slot)
        if observation is not None:
            observations.append(observation)
    return compose_meter_v2(presence, tuple(observations))


def checkpoint_loading_allowed_in_ci() -> bool:
    """Private Drive checkpoint binaries are never loaded by CI."""
    return False


def resolver_connection_allowed() -> bool:
    """This shadow package never authorizes runtime Resolver wiring."""
    return False
