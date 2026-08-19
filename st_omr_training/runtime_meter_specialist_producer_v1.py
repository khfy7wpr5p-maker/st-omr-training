"""Provenance-bound runtime Meter specialist producer boundary.

This module closes the byte-identity gap left by Meter Runtime Integration V3
without loading checkpoints itself.  The producer accepts exactly one
``RuntimeRoiArtifact`` and passes only that artifact's immutable PNG bytes to a
caller-supplied, audited inference runner.  The resulting Meter evidence is
bound to source-image, ROI-image and frozen specialist-profile identities.

The module does not train, tune thresholds, access TRAIN/VALIDATION/TEST, load
D10/D13 derivatives, or invoke the Deterministic Resolver.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Callable, Final

from .runtime_local_roi_v1 import RuntimeRoiArtifact
from .runtime_meter_integration_v3 import MeterDigitScoresV3, MeterModelEvidenceV3


METER_SPECIALIST_PRODUCER_V1_VERSION: Final[str] = "runtime-meter-specialist-producer-v1"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_STATUSES: Final[tuple[str, ...]] = ("accepted", "ambiguous", "rejected")


class MeterSpecialistProducerError(RuntimeError):
    """Raised when runtime Meter specialist provenance cannot be trusted."""


def _require_sha(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise MeterSpecialistProducerError(f"{name} must be canonical lowercase SHA-256")
    return value


def _canonical_sha(payload: object) -> str:
    try:
        raw = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise MeterSpecialistProducerError("producer payload is not canonical JSON") from exc
    return sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class MeterSpecialistProfileV1:
    """Immutable identity of the four frozen Meter specialists used at runtime."""

    presence_checkpoint_sha256: str
    digit2_checkpoint_sha256: str
    digit3_checkpoint_sha256: str
    digit4_checkpoint_sha256: str
    runner_contract: str = "meter-presence-plus-independent-2-3-4-v1"

    def __post_init__(self) -> None:
        for name in (
            "presence_checkpoint_sha256",
            "digit2_checkpoint_sha256",
            "digit3_checkpoint_sha256",
            "digit4_checkpoint_sha256",
        ):
            _require_sha(getattr(self, name), name)
        if self.runner_contract != "meter-presence-plus-independent-2-3-4-v1":
            raise ValueError("unsupported Meter specialist runner contract")

    def fingerprint(self) -> str:
        return _canonical_sha(
            {
                "version": METER_SPECIALIST_PRODUCER_V1_VERSION,
                "runner_contract": self.runner_contract,
                "presence_checkpoint_sha256": self.presence_checkpoint_sha256,
                "digit2_checkpoint_sha256": self.digit2_checkpoint_sha256,
                "digit3_checkpoint_sha256": self.digit3_checkpoint_sha256,
                "digit4_checkpoint_sha256": self.digit4_checkpoint_sha256,
                "train_access": False,
                "validation_access": False,
                "test_access": False,
                "threshold_tuning": False,
                "resolver_wiring": False,
            }
        )


@dataclass(frozen=True, slots=True)
class MeterRawInferenceV1:
    """Raw output returned by one audited runtime Meter inference runner."""

    presence_status: str
    presence_score: float | None
    refined_x_center_roi: float | None = None
    numerator_scores: MeterDigitScoresV3 | None = None
    denominator_scores: MeterDigitScoresV3 | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.presence_status not in _ALLOWED_STATUSES:
            raise ValueError("unsupported Meter presence status")
        if self.presence_score is not None:
            if (
                isinstance(self.presence_score, bool)
                or not isinstance(self.presence_score, (int, float))
                or not math.isfinite(float(self.presence_score))
                or not 0.0 <= float(self.presence_score) <= 1.0
            ):
                raise ValueError("Meter presence score must be finite in 0..1")
        if self.presence_status == "accepted":
            if self.presence_score is None or self.reasons:
                raise ValueError("accepted Meter raw inference needs score and no reasons")
        elif not self.reasons:
            raise ValueError("non-accepted Meter raw inference must explain why")
        if self.refined_x_center_roi is not None and (
            isinstance(self.refined_x_center_roi, bool)
            or not isinstance(self.refined_x_center_roi, (int, float))
            or not math.isfinite(float(self.refined_x_center_roi))
        ):
            raise ValueError("refined_x_center_roi must be finite when supplied")

    def canonical_payload(self) -> dict[str, object]:
        def score_payload(value: MeterDigitScoresV3 | None) -> object:
            if value is None:
                return None
            return {
                "2": value.score_2_milli,
                "3": value.score_3_milli,
                "4": value.score_4_milli,
            }

        return {
            "presence_status": self.presence_status,
            "presence_score": self.presence_score,
            "refined_x_center_roi": self.refined_x_center_roi,
            "numerator_scores": score_payload(self.numerator_scores),
            "denominator_scores": score_payload(self.denominator_scores),
            "reasons": list(self.reasons),
        }


MeterInferenceRunnerV1 = Callable[[bytes], MeterRawInferenceV1]


@dataclass(frozen=True, slots=True)
class ProvenanceBoundMeterEvidenceV1:
    evidence: MeterModelEvidenceV3
    source_image_sha256: str
    roi_image_sha256: str
    specialist_profile_fingerprint: str
    inference_request_fingerprint: str
    inference_output_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "source_image_sha256",
            "roi_image_sha256",
            "specialist_profile_fingerprint",
            "inference_request_fingerprint",
            "inference_output_fingerprint",
        ):
            _require_sha(getattr(self, name), name)

    def fingerprint(self) -> str:
        return _canonical_sha(
            {
                "version": METER_SPECIALIST_PRODUCER_V1_VERSION,
                "evidence_id": self.evidence.evidence_id,
                "system_id": self.evidence.system_id,
                "logical_measure_id": self.evidence.logical_measure_id,
                "measure_id": self.evidence.measure_id,
                "staff_id": self.evidence.staff_id,
                "roi_id": self.evidence.roi_id,
                "source_image_sha256": self.source_image_sha256,
                "roi_image_sha256": self.roi_image_sha256,
                "specialist_profile_fingerprint": self.specialist_profile_fingerprint,
                "inference_request_fingerprint": self.inference_request_fingerprint,
                "inference_output_fingerprint": self.inference_output_fingerprint,
            }
        )


def produce_meter_evidence_v1(
    roi: RuntimeRoiArtifact,
    *,
    system_id: str,
    logical_measure_id: str,
    profile: MeterSpecialistProfileV1,
    runner: MeterInferenceRunnerV1,
) -> ProvenanceBoundMeterEvidenceV1:
    """Run one Meter specialist request on the exact immutable ROI bytes.

    The only pixel payload supplied to ``runner`` is ``roi.png_bytes``.  The
    producer independently re-hashes those bytes before and after inference so
    a mutated/mismatched artifact fails closed.
    """
    if not isinstance(roi, RuntimeRoiArtifact):
        raise TypeError("roi must be RuntimeRoiArtifact")
    if not isinstance(profile, MeterSpecialistProfileV1):
        raise TypeError("profile must be MeterSpecialistProfileV1")
    if not callable(runner):
        raise TypeError("runner must be callable")
    if roi.kind != "measure-start" or roi.roi_id != f"{roi.measure_id}:measure-start":
        raise MeterSpecialistProducerError("Meter producer requires canonical measure-start ROI")
    if not system_id or not logical_measure_id:
        raise MeterSpecialistProducerError("system/logical-measure identities must be non-empty")

    before_sha = sha256(roi.png_bytes).hexdigest()
    if before_sha != roi.roi_image_sha256:
        raise MeterSpecialistProducerError("ROI byte identity mismatch before inference")

    profile_fp = profile.fingerprint()
    request_fp = _canonical_sha(
        {
            "version": METER_SPECIALIST_PRODUCER_V1_VERSION,
            "system_id": system_id,
            "logical_measure_id": logical_measure_id,
            "measure_id": roi.measure_id,
            "staff_id": roi.staff_id,
            "roi_id": roi.roi_id,
            "source_image_sha256": roi.source_image_sha256,
            "roi_image_sha256": roi.roi_image_sha256,
            "specialist_profile_fingerprint": profile_fp,
        }
    )

    raw = runner(roi.png_bytes)
    if not isinstance(raw, MeterRawInferenceV1):
        raise MeterSpecialistProducerError("runner returned wrong Meter raw inference contract")

    after_sha = sha256(roi.png_bytes).hexdigest()
    if after_sha != before_sha or after_sha != roi.roi_image_sha256:
        raise MeterSpecialistProducerError("ROI byte identity changed during inference")

    output_fp = _canonical_sha(
        {
            "request_fingerprint": request_fp,
            "raw_output": raw.canonical_payload(),
        }
    )
    evidence_id = f"meter-producer-v1:{roi.measure_id}:{output_fp[:16]}"
    evidence = MeterModelEvidenceV3(
        evidence_id=evidence_id,
        system_id=system_id,
        logical_measure_id=logical_measure_id,
        measure_id=roi.measure_id,
        staff_id=roi.staff_id,
        roi_id=roi.roi_id,
        presence_status=raw.presence_status,
        presence_score=raw.presence_score,
        refined_x_center_roi=raw.refined_x_center_roi,
        numerator_scores=raw.numerator_scores,
        denominator_scores=raw.denominator_scores,
        reasons=raw.reasons,
    )
    return ProvenanceBoundMeterEvidenceV1(
        evidence=evidence,
        source_image_sha256=roi.source_image_sha256,
        roi_image_sha256=roi.roi_image_sha256,
        specialist_profile_fingerprint=profile_fp,
        inference_request_fingerprint=request_fp,
        inference_output_fingerprint=output_fp,
    )


def checkpoint_loading_allowed() -> bool:
    return False


def train_validation_test_access_allowed() -> bool:
    return False


def resolver_connection_allowed() -> bool:
    return False
