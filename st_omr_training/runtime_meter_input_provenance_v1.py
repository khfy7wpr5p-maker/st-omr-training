"""Fail-closed provenance contract for real Meter runtime inputs.

This layer does not run models. It makes the remaining historical-input gaps
explicit and fingerprintable before any real checkpoint may be invoked.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Final

from .runtime_local_roi_v1 import RuntimeRoiArtifact
from .meter_v2_digit_crop_profile_v1 import meter_v2_digit_crop_profile_fingerprint_v1

METER_RUNTIME_INPUT_PROVENANCE_V1: Final[str] = "meter-runtime-input-provenance-v1"
HISTORICAL_PRESENCE_ROI_PROFILE: Final[str] = "measure-start-meter-roi-v1"
RUNTIME_MEASURE_START_PROFILE: Final[str] = "runtime-local-roi-v1:measure-start"


class MeterRuntimeInputProvenanceError(RuntimeError):
    pass


def _fingerprint(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class DigitSlotGeometryV1:
    numerator_box_roi: tuple[float, float, float, float]
    denominator_box_roi: tuple[float, float, float, float]
    localization_profile_fingerprint: str

    def __post_init__(self) -> None:
        if len(self.numerator_box_roi) != 4 or len(self.denominator_box_roi) != 4:
            raise ValueError("digit slot boxes must contain four coordinates")
        if not self.localization_profile_fingerprint:
            raise ValueError("digit localization profile identity is required")
        for box in (self.numerator_box_roi, self.denominator_box_roi):
            x0, y0, x1, y1 = box
            if not x0 < x1 or not y0 < y1:
                raise ValueError("digit slot boxes must have positive area")

    def fingerprint(self) -> str:
        return _fingerprint({
            "version": METER_RUNTIME_INPUT_PROVENANCE_V1,
            "numerator_box_roi": self.numerator_box_roi,
            "denominator_box_roi": self.denominator_box_roi,
            "localization_profile_fingerprint": self.localization_profile_fingerprint,
            "digit_crop_profile_fingerprint": meter_v2_digit_crop_profile_fingerprint_v1(),
        })


@dataclass(frozen=True, slots=True)
class MeterRuntimeInputContextV1:
    roi_image_sha256: str
    presence_input_profile: str
    digit_slot_geometry_fingerprint: str

    def fingerprint(self) -> str:
        return _fingerprint({
            "version": METER_RUNTIME_INPUT_PROVENANCE_V1,
            "roi_image_sha256": self.roi_image_sha256,
            "presence_input_profile": self.presence_input_profile,
            "digit_slot_geometry_fingerprint": self.digit_slot_geometry_fingerprint,
        })


def bind_runtime_meter_inputs_v1(
    roi: RuntimeRoiArtifact,
    *,
    presence_input_profile: str,
    digit_slots: DigitSlotGeometryV1,
) -> MeterRuntimeInputContextV1:
    if roi.kind != "measure-start":
        raise MeterRuntimeInputProvenanceError("Meter input binding requires measure-start ROI")
    if presence_input_profile != HISTORICAL_PRESENCE_ROI_PROFILE:
        raise MeterRuntimeInputProvenanceError("real Presence inference requires historical D9/D10 ROI profile")
    # Current RuntimeRoiArtifact is explicitly a different crop contract. Until a
    # source-image/geometry adapter can reconstruct the historical crop exactly,
    # accepting it as the historical profile would be false provenance.
    raise MeterRuntimeInputProvenanceError(
        "current runtime measure-start ROI cannot prove historical Presence pixel equivalence"
    )


def real_presence_inference_allowed_from_runtime_roi_v1() -> bool:
    return False


def digit_slot_provenance_required_v1() -> bool:
    return True
