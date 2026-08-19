"""Deterministic Meter evidence association on Measure/System v2 geometry.

The learned boundary stays external: callers supply already-computed presence
and frozen 2/3/4 specialist scores.  This module binds those observations to
exact system/logical-measure/measure/staff/ROI identities, derives staff-relative
digit slots from accepted upstream geometry, composes only none|2/4|3/4|4/4,
and emits model-agnostic SpecialistEvidenceBatch observations.

No model/checkpoint, optimizer, TRAIN/VALIDATION/TEST split, old measure-geometry
fallback, or Deterministic Resolver is imported or invoked here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from typing import Final

from .runtime_geometry_engine_contract import BoxContract, PageGeometryContract
from .runtime_local_roi_v1 import RuntimeRoiArtifact, RuntimeRoiBatch
from .runtime_measure_system_boundaries_v2 import MeasureSystemBoundaryReportV2
from .runtime_specialist_evidence_v1 import SpecialistEvidenceBatch, SpecialistObservation
from .runtime_system_grouper_v1 import page_geometry_fingerprint_v1


METER_RUNTIME_INTEGRATION_V3_VERSION: Final[str] = "runtime-meter-integration-v3"
METER_CLASSES: Final[tuple[str, ...]] = ("none", "2/4", "3/4", "4/4")
METER_DIGITS: Final[tuple[int, ...]] = (2, 3, 4)
STATUSES: Final[tuple[str, ...]] = ("accepted", "ambiguous", "rejected")

# Frozen prior development evidence.  V3 does not tune these values.
PRESENCE_THRESHOLD: Final[float] = 0.90
DIGIT_THRESHOLDS_MILLI: Final[dict[int, int]] = {2: 480, 3: 600, 4: 470}
TRAIN_WIDTH_OVER_STAFF_SPACING: Final[float] = 1.5960569245912566
TRAIN_HEIGHT_OVER_STAFF_SPACING: Final[float] = 2.0
NUMERATOR_STAFF_LINE_INDEX: Final[int] = 1
DENOMINATOR_STAFF_LINE_INDEX: Final[int] = 3
_EPS: Final[float] = 1e-9

M01_UPSTREAM_GEOMETRY_NOT_ACCEPTED: Final[str] = "M01_UPSTREAM_GEOMETRY_NOT_ACCEPTED"
M02_BOUNDARY_REPORT_MISMATCH: Final[str] = "M02_BOUNDARY_REPORT_MISMATCH"
M03_METER_EVIDENCE_MISSING: Final[str] = "M03_METER_EVIDENCE_MISSING"
M04_WRONG_PRESENCE_REGION: Final[str] = "M04_WRONG_PRESENCE_REGION"
M05_IDENTITY_MISMATCH: Final[str] = "M05_IDENTITY_MISMATCH"
M06_PRESENCE_AMBIGUOUS: Final[str] = "M06_PRESENCE_AMBIGUOUS"
M07_PRESENCE_REJECTED: Final[str] = "M07_PRESENCE_REJECTED"
M08_NO_DIGIT: Final[str] = "M08_NO_DIGIT"
M09_UPPER_DIGIT_NOT_FOUND: Final[str] = "M09_UPPER_DIGIT_NOT_FOUND"
M10_LOWER_DIGIT_NOT_FOUND: Final[str] = "M10_LOWER_DIGIT_NOT_FOUND"
M11_DIGIT_SPECIALIST_CONFLICT: Final[str] = "M11_DIGIT_SPECIALIST_CONFLICT"
M12_PRESENCE_DIGIT_CONFLICT: Final[str] = "M12_PRESENCE_DIGIT_CONFLICT"
M13_SLOT_GEOMETRY_AMBIGUOUS: Final[str] = "M13_SLOT_GEOMETRY_AMBIGUOUS"
M14_UNSUPPORTED_COMPOSITION: Final[str] = "M14_UNSUPPORTED_COMPOSITION"
M15_CROSS_STAFF_METER_MISMATCH: Final[str] = "M15_CROSS_STAFF_METER_MISMATCH"

METER_REASON_PRIORITY: Final[tuple[str, ...]] = (
    M01_UPSTREAM_GEOMETRY_NOT_ACCEPTED,
    M02_BOUNDARY_REPORT_MISMATCH,
    M03_METER_EVIDENCE_MISSING,
    M04_WRONG_PRESENCE_REGION,
    M05_IDENTITY_MISMATCH,
    M06_PRESENCE_AMBIGUOUS,
    M07_PRESENCE_REJECTED,
    M08_NO_DIGIT,
    M09_UPPER_DIGIT_NOT_FOUND,
    M10_LOWER_DIGIT_NOT_FOUND,
    M11_DIGIT_SPECIALIST_CONFLICT,
    M12_PRESENCE_DIGIT_CONFLICT,
    M13_SLOT_GEOMETRY_AMBIGUOUS,
    M14_UNSUPPORTED_COMPOSITION,
    M15_CROSS_STAFF_METER_MISMATCH,
)


@dataclass(frozen=True, slots=True)
class MeterDigitScoresV3:
    score_2_milli: int
    score_3_milli: int
    score_4_milli: int

    def __post_init__(self) -> None:
        for value in (self.score_2_milli, self.score_3_milli, self.score_4_milli):
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 1000:
                raise ValueError("Meter digit scores must be plain integer milli values in 0..1000")

    def as_dict(self) -> dict[int, int]:
        return {2: self.score_2_milli, 3: self.score_3_milli, 4: self.score_4_milli}


@dataclass(frozen=True, slots=True)
class MeterModelEvidenceV3:
    evidence_id: str
    system_id: str
    logical_measure_id: str
    measure_id: str
    staff_id: str
    roi_id: str
    presence_status: str
    presence_score: float | None
    refined_x_center_roi: float | None = None
    numerator_scores: MeterDigitScoresV3 | None = None
    denominator_scores: MeterDigitScoresV3 | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all((self.evidence_id, self.system_id, self.logical_measure_id, self.measure_id, self.staff_id, self.roi_id)):
            raise ValueError("Meter evidence identities must be non-empty")
        if self.presence_status not in STATUSES:
            raise ValueError("unsupported Meter presence status")
        if self.presence_status == "accepted":
            if self.reasons:
                raise ValueError("accepted Meter presence evidence cannot carry reasons")
            if (
                self.presence_score is None
                or isinstance(self.presence_score, bool)
                or not isinstance(self.presence_score, (int, float))
                or not math.isfinite(float(self.presence_score))
                or not 0.0 <= float(self.presence_score) <= 1.0
            ):
                raise ValueError("accepted Meter presence score must be finite in 0..1")
        else:
            if not self.reasons:
                raise ValueError("non-accepted Meter presence evidence must explain why")
            if self.presence_score is not None and (
                isinstance(self.presence_score, bool)
                or not isinstance(self.presence_score, (int, float))
                or not math.isfinite(float(self.presence_score))
                or not 0.0 <= float(self.presence_score) <= 1.0
            ):
                raise ValueError("optional Meter presence score must be finite in 0..1")
        if self.refined_x_center_roi is not None and (
            isinstance(self.refined_x_center_roi, bool)
            or not isinstance(self.refined_x_center_roi, (int, float))
            or not math.isfinite(float(self.refined_x_center_roi))
        ):
            raise ValueError("refined_x_center_roi must be finite when supplied")


@dataclass(frozen=True, slots=True)
class MeterIntegrationDecisionV3:
    evidence_id: str
    system_id: str
    logical_measure_id: str
    measure_id: str
    staff_id: str
    roi_id: str
    status: str
    meter_class: str | None
    confidence_milli: int
    bbox: BoxContract | None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError("unsupported Meter integration status")
        if not all((self.evidence_id, self.system_id, self.logical_measure_id, self.measure_id, self.staff_id, self.roi_id)):
            raise ValueError("Meter integration identities must be non-empty")
        if not isinstance(self.confidence_milli, int) or isinstance(self.confidence_milli, bool) or not 0 <= self.confidence_milli <= 1000:
            raise ValueError("Meter integration confidence must be integer milli in 0..1000")
        if tuple(sorted(set(self.reasons), key=METER_REASON_PRIORITY.index)) != self.reasons:
            raise ValueError("Meter integration reasons must be unique and canonical")
        if self.status == "accepted":
            if self.meter_class not in METER_CLASSES or self.reasons:
                raise ValueError("accepted Meter integration needs supported class and no reasons")
            if self.meter_class == "none" and self.bbox is not None:
                raise ValueError("none Meter cannot carry a visible bbox")
            if self.meter_class != "none" and self.bbox is None:
                raise ValueError("visible accepted Meter requires bbox")
        else:
            if self.meter_class is not None or not self.reasons:
                raise ValueError("non-accepted Meter integration cannot assign a class and must explain why")

    def canonical_payload(self) -> dict[str, object]:
        bbox = None if self.bbox is None else [self.bbox.x_min, self.bbox.y_min, self.bbox.x_max, self.bbox.y_max]
        return {
            "evidence_id": self.evidence_id,
            "system_id": self.system_id,
            "logical_measure_id": self.logical_measure_id,
            "measure_id": self.measure_id,
            "staff_id": self.staff_id,
            "roi_id": self.roi_id,
            "status": self.status,
            "meter_class": self.meter_class,
            "confidence_milli": self.confidence_milli,
            "bbox": bbox,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class MeterRuntimeIntegrationV3Result:
    config_fingerprint: str
    decisions: tuple[MeterIntegrationDecisionV3, ...]
    evidence_batch: SpecialistEvidenceBatch

    def __post_init__(self) -> None:
        _require_sha(self.config_fingerprint)
        measure_ids = tuple(item.measure_id for item in self.decisions)
        if len(measure_ids) != len(set(measure_ids)):
            raise ValueError("Meter integration decisions must have unique measure ids")
        if len(self.evidence_batch.observations) != len(self.decisions):
            raise ValueError("Meter integration emits exactly one specialist observation per measure")

    def fingerprint(self) -> str:
        return _canonical_sha(
            {
                "version": METER_RUNTIME_INTEGRATION_V3_VERSION,
                "config_fingerprint": self.config_fingerprint,
                "decisions": [item.canonical_payload() for item in self.decisions],
                "observations": [
                    {
                        "observation_id": item.observation_id,
                        "task": item.task,
                        "measure_id": item.measure_id,
                        "staff_id": item.staff_id,
                        "status": item.status,
                        "confidence_milli": item.confidence_milli,
                        "class_label": item.class_label,
                        "bbox": None if item.bbox is None else [item.bbox.x_min, item.bbox.y_min, item.bbox.x_max, item.bbox.y_max],
                        "reasons": list(item.reasons),
                        "source_kind": item.source_kind,
                    }
                    for item in self.evidence_batch.observations
                ],
            }
        )


def _require_sha(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("expected canonical lowercase SHA-256")


def _canonical_sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return sha256(raw).hexdigest()


def meter_runtime_integration_v3_config_fingerprint(geometry_fingerprint: str) -> str:
    _require_sha(geometry_fingerprint)
    return _canonical_sha(
        {
            "version": METER_RUNTIME_INTEGRATION_V3_VERSION,
            "geometry_fingerprint": geometry_fingerprint,
            "presence_threshold": PRESENCE_THRESHOLD,
            "digit_thresholds_milli": DIGIT_THRESHOLDS_MILLI,
            "width_over_staff_spacing": TRAIN_WIDTH_OVER_STAFF_SPACING,
            "height_over_staff_spacing": TRAIN_HEIGHT_OVER_STAFF_SPACING,
            "numerator_staff_line_index": NUMERATOR_STAFF_LINE_INDEX,
            "denominator_staff_line_index": DENOMINATOR_STAFF_LINE_INDEX,
            "measure_index_special_cases": False,
            "staff_redetection": False,
            "model_loading": False,
            "checkpoint_access": False,
            "optimizer_access": False,
            "train_validation_test_access": False,
            "resolver_wiring": False,
            "old_measure_geometry_fallback": False,
        }
    )


def _canonical_reasons(*codes: str) -> tuple[str, ...]:
    unknown = set(codes) - set(METER_REASON_PRIORITY)
    if unknown:
        raise ValueError("unknown Meter integration reason")
    chosen = set(codes)
    return tuple(code for code in METER_REASON_PRIORITY if code in chosen)


def _placeholder_decision(measure, logical_id: str, roi_id: str, reason: str) -> MeterIntegrationDecisionV3:
    return MeterIntegrationDecisionV3(
        evidence_id=f"missing:{measure.measure_id}",
        system_id=measure.system_id,
        logical_measure_id=logical_id,
        measure_id=measure.measure_id,
        staff_id=measure.staff_id,
        roi_id=roi_id,
        status="ambiguous",
        meter_class=None,
        confidence_milli=0,
        bbox=None,
        reasons=_canonical_reasons(reason),
    )


def _translation_matches_roi(roi: RuntimeRoiArtifact) -> bool:
    left = float(roi.crop_bbox.x_min)
    top = float(roi.crop_bbox.y_min)
    expected_f = (1.0, 0.0, -left, 0.0, 1.0, -top, 0.0, 0.0, 1.0)
    expected_i = (1.0, 0.0, left, 0.0, 1.0, top, 0.0, 0.0, 1.0)
    return all(abs(a - b) <= _EPS for a, b in zip(roi.source_to_roi.forward, expected_f)) and all(
        abs(a - b) <= _EPS for a, b in zip(roi.source_to_roi.inverse, expected_i)
    )


def _line_y_at_source_x(line, source_x: float) -> float | None:
    x0, x1 = float(line.start.x), float(line.end.x)
    y0, y1 = float(line.start.y), float(line.end.y)
    lo, hi = min(x0, x1), max(x0, x1)
    if hi - lo <= _EPS or source_x < lo - _EPS or source_x > hi + _EPS:
        return None
    value = y0 + ((source_x - x0) / (x1 - x0)) * (y1 - y0)
    return value if math.isfinite(value) else None


def _slot_box_source(geometry: PageGeometryContract, measure, roi: RuntimeRoiArtifact, x_roi: float, line_index: int) -> BoxContract | None:
    staffs = tuple(staff for staff in geometry.staffs if staff.staff_id == measure.staff_id)
    if len(staffs) != 1 or not _translation_matches_roi(roi):
        return None
    staff = staffs[0]
    roi_width = float(roi.crop_bbox.x_max - roi.crop_bbox.x_min)
    if not 0.0 <= x_roi <= roi_width:
        return None
    source_x = x_roi + float(roi.crop_bbox.x_min)
    ys: list[float] = []
    for line in staff.five_staff_lines:
        value = _line_y_at_source_x(line, source_x)
        if value is None:
            return None
        ys.append(value)
    gaps = tuple(b - a for a, b in zip(ys, ys[1:]))
    if len(gaps) != 4 or any(not math.isfinite(gap) or gap <= 0 for gap in gaps):
        return None
    spacing = sum(gaps) / 4.0
    width = spacing * TRAIN_WIDTH_OVER_STAFF_SPACING
    height = spacing * TRAIN_HEIGHT_OVER_STAFF_SPACING
    center_y = ys[line_index]
    box = BoxContract(source_x - width / 2.0, center_y - height / 2.0, source_x + width / 2.0, center_y + height / 2.0)
    if not _box_inside(box, roi.crop_bbox) or not _box_inside(box, measure.bbox):
        return None
    return box


def _box_inside(inner: BoxContract, outer: BoxContract) -> bool:
    return (
        inner.x_min >= outer.x_min - _EPS
        and inner.y_min >= outer.y_min - _EPS
        and inner.x_max <= outer.x_max + _EPS
        and inner.y_max <= outer.y_max + _EPS
    )


def _union_box(a: BoxContract, b: BoxContract) -> BoxContract:
    return BoxContract(min(a.x_min, b.x_min), min(a.y_min, b.y_min), max(a.x_max, b.x_max), max(a.y_max, b.y_max))


def _passing_digit(scores: MeterDigitScoresV3 | None) -> tuple[int, ...]:
    if scores is None:
        return ()
    values = scores.as_dict()
    return tuple(digit for digit in METER_DIGITS if values[digit] >= DIGIT_THRESHOLDS_MILLI[digit])


def _selected_score(scores: MeterDigitScoresV3, digit: int) -> int:
    return scores.as_dict()[digit]


def _decision_for_measure(geometry, measure, logical_id: str, roi: RuntimeRoiArtifact, evidence: MeterModelEvidenceV3) -> MeterIntegrationDecisionV3:
    identity_ok = (
        evidence.system_id == measure.system_id
        and evidence.logical_measure_id == logical_id
        and evidence.measure_id == measure.measure_id
        and evidence.staff_id == measure.staff_id
        and evidence.roi_id == roi.roi_id
        and roi.measure_id == measure.measure_id
        and roi.staff_id == measure.staff_id
        and roi.source_image_sha256 == geometry.normalized_image_sha256
    )
    if roi.kind != "measure-start":
        return replace(
            _placeholder_decision(measure, logical_id, roi.roi_id, M04_WRONG_PRESENCE_REGION),
            evidence_id=evidence.evidence_id,
        )
    if not identity_ok:
        return replace(
            _placeholder_decision(measure, logical_id, roi.roi_id, M05_IDENTITY_MISMATCH),
            evidence_id=evidence.evidence_id,
        )
    confidence = 0 if evidence.presence_score is None else max(0, min(1000, int(round(float(evidence.presence_score) * 1000.0))))
    if evidence.presence_status == "ambiguous":
        return MeterIntegrationDecisionV3(evidence.evidence_id, measure.system_id, logical_id, measure.measure_id, measure.staff_id, roi.roi_id, "ambiguous", None, confidence, None, _canonical_reasons(M06_PRESENCE_AMBIGUOUS))
    if evidence.presence_status == "rejected":
        return MeterIntegrationDecisionV3(evidence.evidence_id, measure.system_id, logical_id, measure.measure_id, measure.staff_id, roi.roi_id, "rejected", None, confidence, None, _canonical_reasons(M07_PRESENCE_REJECTED))
    assert evidence.presence_score is not None
    present = float(evidence.presence_score) >= PRESENCE_THRESHOLD
    upper = _passing_digit(evidence.numerator_scores)
    lower = _passing_digit(evidence.denominator_scores)

    if not present:
        if upper or lower:
            return MeterIntegrationDecisionV3(evidence.evidence_id, measure.system_id, logical_id, measure.measure_id, measure.staff_id, roi.roi_id, "ambiguous", None, confidence, None, _canonical_reasons(M12_PRESENCE_DIGIT_CONFLICT))
        return MeterIntegrationDecisionV3(evidence.evidence_id, measure.system_id, logical_id, measure.measure_id, measure.staff_id, roi.roi_id, "accepted", "none", confidence, None, ())

    if not upper and not lower:
        return MeterIntegrationDecisionV3(evidence.evidence_id, measure.system_id, logical_id, measure.measure_id, measure.staff_id, roi.roi_id, "ambiguous", None, confidence, None, _canonical_reasons(M08_NO_DIGIT))
    missing: list[str] = []
    if not upper:
        missing.append(M09_UPPER_DIGIT_NOT_FOUND)
    if not lower:
        missing.append(M10_LOWER_DIGIT_NOT_FOUND)
    if missing:
        return MeterIntegrationDecisionV3(evidence.evidence_id, measure.system_id, logical_id, measure.measure_id, measure.staff_id, roi.roi_id, "ambiguous", None, confidence, None, _canonical_reasons(*missing))
    if len(upper) != 1 or len(lower) != 1:
        return MeterIntegrationDecisionV3(evidence.evidence_id, measure.system_id, logical_id, measure.measure_id, measure.staff_id, roi.roi_id, "ambiguous", None, confidence, None, _canonical_reasons(M11_DIGIT_SPECIALIST_CONFLICT))
    numerator, denominator = upper[0], lower[0]
    if denominator != 4 or numerator not in (2, 3, 4):
        return MeterIntegrationDecisionV3(evidence.evidence_id, measure.system_id, logical_id, measure.measure_id, measure.staff_id, roi.roi_id, "ambiguous", None, confidence, None, _canonical_reasons(M14_UNSUPPORTED_COMPOSITION))
    if evidence.refined_x_center_roi is None:
        return MeterIntegrationDecisionV3(evidence.evidence_id, measure.system_id, logical_id, measure.measure_id, measure.staff_id, roi.roi_id, "ambiguous", None, confidence, None, _canonical_reasons(M13_SLOT_GEOMETRY_AMBIGUOUS))
    x_roi = float(evidence.refined_x_center_roi)
    upper_box = _slot_box_source(geometry, measure, roi, x_roi, NUMERATOR_STAFF_LINE_INDEX)
    lower_box = _slot_box_source(geometry, measure, roi, x_roi, DENOMINATOR_STAFF_LINE_INDEX)
    if upper_box is None or lower_box is None:
        return MeterIntegrationDecisionV3(evidence.evidence_id, measure.system_id, logical_id, measure.measure_id, measure.staff_id, roi.roi_id, "ambiguous", None, confidence, None, _canonical_reasons(M13_SLOT_GEOMETRY_AMBIGUOUS))
    assert evidence.numerator_scores is not None and evidence.denominator_scores is not None
    confidence = min(
        confidence,
        _selected_score(evidence.numerator_scores, numerator),
        _selected_score(evidence.denominator_scores, denominator),
    )
    return MeterIntegrationDecisionV3(
        evidence.evidence_id,
        measure.system_id,
        logical_id,
        measure.measure_id,
        measure.staff_id,
        roi.roi_id,
        "accepted",
        f"{numerator}/{denominator}",
        confidence,
        _union_box(upper_box, lower_box),
        (),
    )


def _to_specialist(decision: MeterIntegrationDecisionV3) -> SpecialistObservation:
    # Any failed Meter association must remain visible to the later Resolver;
    # rejected integration evidence is therefore propagated as ambiguous rather
    # than disappearing from the resolver input batch.
    status = "accepted" if decision.status == "accepted" else "ambiguous"
    return SpecialistObservation(
        observation_id=f"meter-v3:{decision.measure_id}",
        task="meter",
        measure_id=decision.measure_id,
        staff_id=decision.staff_id,
        status=status,
        confidence_milli=decision.confidence_milli,
        class_label=decision.meter_class if status == "accepted" else None,
        bbox=decision.bbox if status == "accepted" else None,
        reasons=() if status == "accepted" else decision.reasons,
        source_kind="specialist-adapter",
    )


def integrate_meter_evidence_v3(
    geometry: PageGeometryContract,
    boundary_report: MeasureSystemBoundaryReportV2,
    roi_batch: RuntimeRoiBatch,
    evidence: tuple[MeterModelEvidenceV3, ...],
) -> MeterRuntimeIntegrationV3Result:
    """Bind supplied Meter model evidence to exact deterministic measure ownership."""
    if not isinstance(geometry, PageGeometryContract):
        raise TypeError("geometry must be PageGeometryContract")
    if not isinstance(boundary_report, MeasureSystemBoundaryReportV2):
        raise TypeError("boundary_report must be MeasureSystemBoundaryReportV2")
    if not isinstance(roi_batch, RuntimeRoiBatch) or not isinstance(evidence, tuple):
        raise TypeError("roi_batch/evidence must use runtime contracts")

    geometry_fp = page_geometry_fingerprint_v1(geometry)
    config_fp = meter_runtime_integration_v3_config_fingerprint(geometry_fp)
    ordered_measures = tuple(sorted(geometry.measure_proposals, key=lambda item: (item.system_id, item.staff_id, item.bbox.x_min, item.measure_id)))

    logical_by_measure: dict[str, str] = {}
    valid_report = boundary_report.status == "accepted" and boundary_report.output_geometry_fingerprint == geometry_fp
    if valid_report:
        for logical in boundary_report.logical_measures:
            for measure_id in logical.member_measure_ids:
                if measure_id in logical_by_measure:
                    valid_report = False
                    break
                logical_by_measure[measure_id] = logical.logical_measure_id
            if not valid_report:
                break
        valid_report = valid_report and set(logical_by_measure) == {item.measure_id for item in ordered_measures}
        if valid_report:
            measure_by_id = {item.measure_id: item for item in ordered_measures}
            for logical in boundary_report.logical_measures:
                if any(measure_by_id[mid].system_id != logical.system_id for mid in logical.member_measure_ids):
                    valid_report = False
                    break

    roi_candidates: dict[str, list[RuntimeRoiArtifact]] = {}
    for artifact in roi_batch.artifacts:
        roi_candidates.setdefault(artifact.measure_id, []).append(artifact)
    evidence_candidates: dict[str, list[MeterModelEvidenceV3]] = {}
    for item in evidence:
        if not isinstance(item, MeterModelEvidenceV3):
            raise TypeError("evidence tuple contains non-MeterModelEvidenceV3")
        evidence_candidates.setdefault(item.measure_id, []).append(item)

    decisions: list[MeterIntegrationDecisionV3] = []
    for measure in ordered_measures:
        logical_id = logical_by_measure.get(measure.measure_id, f"unresolved:{measure.system_id}:{measure.measure_id}")
        expected_roi = f"{measure.measure_id}:measure-start"
        if geometry.status != "accepted":
            decisions.append(_placeholder_decision(measure, logical_id, expected_roi, M01_UPSTREAM_GEOMETRY_NOT_ACCEPTED))
            continue
        if not valid_report:
            decisions.append(_placeholder_decision(measure, logical_id, expected_roi, M02_BOUNDARY_REPORT_MISMATCH))
            continue
        if roi_batch.source_image_sha256 != geometry.normalized_image_sha256:
            decisions.append(_placeholder_decision(measure, logical_id, expected_roi, M05_IDENTITY_MISMATCH))
            continue
        inputs = evidence_candidates.get(measure.measure_id, [])
        if len(inputs) != 1:
            reason = M03_METER_EVIDENCE_MISSING if not inputs else M05_IDENTITY_MISMATCH
            decisions.append(_placeholder_decision(measure, logical_id, expected_roi, reason))
            continue
        rois = [item for item in roi_candidates.get(measure.measure_id, []) if item.roi_id == inputs[0].roi_id]
        if len(rois) != 1:
            decisions.append(replace(_placeholder_decision(measure, logical_id, inputs[0].roi_id, M05_IDENTITY_MISMATCH), evidence_id=inputs[0].evidence_id))
            continue
        decisions.append(_decision_for_measure(geometry, measure, logical_id, rois[0], inputs[0]))

    # Cross-staff logical measures are one musical Meter ownership unit.  Any
    # disagreement or unresolved member makes every member explicit ambiguous.
    decision_by_measure = {item.measure_id: item for item in decisions}
    for logical in boundary_report.logical_measures if valid_report else ():
        member_decisions = tuple(decision_by_measure[mid] for mid in logical.member_measure_ids)
        if len(member_decisions) <= 1:
            continue
        classes = {item.meter_class for item in member_decisions if item.status == "accepted"}
        mismatch = any(item.status != "accepted" for item in member_decisions) or len(classes) != 1
        if mismatch:
            for item in member_decisions:
                reasons = _canonical_reasons(*item.reasons, M15_CROSS_STAFF_METER_MISMATCH)
                decision_by_measure[item.measure_id] = replace(item, status="ambiguous", meter_class=None, bbox=None, reasons=reasons)

    final_decisions = tuple(decision_by_measure[item.measure_id] for item in decisions)
    observations = tuple(_to_specialist(item) for item in final_decisions)
    return MeterRuntimeIntegrationV3Result(
        config_fingerprint=config_fp,
        decisions=final_decisions,
        evidence_batch=SpecialistEvidenceBatch(observations),
    )


def checkpoint_loading_allowed() -> bool:
    return False


def train_validation_test_access_allowed() -> bool:
    return False


def resolver_connection_allowed() -> bool:
    return False
