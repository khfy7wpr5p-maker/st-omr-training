from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import unittest

from PIL import Image

from st_omr_training.runtime_geometry_engine_contract import (
    BoxContract,
    LineSegmentContract,
    MeasureProposalContract,
    PageGeometryContract,
    Point2DContract,
    StaffGeometryContract,
    SystemGeometryContract,
)
from st_omr_training.runtime_local_roi_v1 import RuntimeRoiArtifact, RuntimeRoiBatch
from st_omr_training.runtime_measure_system_boundaries_v2 import (
    LogicalMeasureV2,
    MeasureSystemBoundaryReportV2,
)
from st_omr_training.runtime_meter_integration_v3 import (
    M02_BOUNDARY_REPORT_MISMATCH,
    M03_METER_EVIDENCE_MISSING,
    M04_WRONG_PRESENCE_REGION,
    M05_IDENTITY_MISMATCH,
    M06_PRESENCE_AMBIGUOUS,
    M08_NO_DIGIT,
    M09_UPPER_DIGIT_NOT_FOUND,
    M10_LOWER_DIGIT_NOT_FOUND,
    M11_DIGIT_SPECIALIST_CONFLICT,
    M12_PRESENCE_DIGIT_CONFLICT,
    M15_CROSS_STAFF_METER_MISMATCH,
    MeterDigitScoresV3,
    MeterModelEvidenceV3,
    checkpoint_loading_allowed,
    integrate_meter_evidence_v3,
    resolver_connection_allowed,
    train_validation_test_access_allowed,
)
from st_omr_training.runtime_page_normalizer_contract import HomographyContract
from st_omr_training.runtime_system_grouper_v1 import page_geometry_fingerprint_v1


SOURCE_SHA = "a" * 64
CONFIG_SHA = "b" * 64


def _line(y: float) -> LineSegmentContract:
    return LineSegmentContract(Point2DContract(10.0, y), Point2DContract(190.0, y))


def _staff(staff_id: str, system_id: str, top: float) -> StaffGeometryContract:
    return StaffGeometryContract(
        staff_id=staff_id,
        system_id=system_id,
        five_staff_lines=tuple(_line(top + 10.0 * index) for index in range(5)),
        staff_bbox=BoxContract(10.0, top, 190.0, top + 40.0),
        staff_spacing=10.0,
    )


def _measure(staff_id: str, system_id: str, top: float) -> MeasureProposalContract:
    measure_id = f"{staff_id}-measure-1"
    return MeasureProposalContract(
        measure_id=measure_id,
        system_id=system_id,
        staff_id=staff_id,
        bbox=BoxContract(10.0, top, 190.0, top + 40.0),
        left_boundary=LineSegmentContract(Point2DContract(10.0, top), Point2DContract(10.0, top + 40.0)),
        right_boundary=LineSegmentContract(Point2DContract(190.0, top), Point2DContract(190.0, top + 40.0)),
        status="accepted",
    )


def _homography_identity() -> HomographyContract:
    return HomographyContract(
        forward=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        inverse=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )


def _single_geometry() -> tuple[PageGeometryContract, MeasureSystemBoundaryReportV2]:
    staff = _staff("staff-1", "system-1", 20.0)
    measure = _measure("staff-1", "system-1", 20.0)
    geometry = PageGeometryContract(
        normalized_image_sha256=SOURCE_SHA,
        geometry_config_fingerprint=CONFIG_SHA,
        page_width=200,
        page_height=140,
        transform=_homography_identity(),
        systems=(SystemGeometryContract("system-1", staff.staff_bbox, (staff.staff_id,)),),
        staffs=(staff,),
        measure_proposals=(measure,),
        status="accepted",
    )
    report = MeasureSystemBoundaryReportV2(
        status="accepted",
        input_geometry_fingerprint="c" * 64,
        output_geometry_fingerprint=page_geometry_fingerprint_v1(geometry),
        logical_measures=(
            LogicalMeasureV2(
                "system-1-measure-1",
                "system-1",
                1,
                10.0,
                190.0,
                "system_edge",
                "system_edge",
                (measure.measure_id,),
            ),
        ),
    )
    return geometry, report


def _two_staff_geometry() -> tuple[PageGeometryContract, MeasureSystemBoundaryReportV2]:
    upper = _staff("staff-1", "system-1", 20.0)
    lower = _staff("staff-2", "system-1", 70.0)
    m1 = _measure("staff-1", "system-1", 20.0)
    m2 = _measure("staff-2", "system-1", 70.0)
    system_bbox = BoxContract(10.0, 20.0, 190.0, 110.0)
    geometry = PageGeometryContract(
        normalized_image_sha256=SOURCE_SHA,
        geometry_config_fingerprint=CONFIG_SHA,
        page_width=200,
        page_height=140,
        transform=_homography_identity(),
        systems=(SystemGeometryContract("system-1", system_bbox, (upper.staff_id, lower.staff_id)),),
        staffs=(upper, lower),
        measure_proposals=(m1, m2),
        status="accepted",
    )
    report = MeasureSystemBoundaryReportV2(
        status="accepted",
        input_geometry_fingerprint="c" * 64,
        output_geometry_fingerprint=page_geometry_fingerprint_v1(geometry),
        logical_measures=(
            LogicalMeasureV2(
                "system-1-measure-1",
                "system-1",
                1,
                10.0,
                190.0,
                "system_edge",
                "system_edge",
                (m1.measure_id, m2.measure_id),
            ),
        ),
    )
    return geometry, report


def _png(width: int, height: int) -> bytes:
    image = Image.new("L", (width, height), 255)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _roi(measure_id: str, staff_id: str, top: float, kind: str = "measure-start") -> RuntimeRoiArtifact:
    if kind == "measure-start":
        box = BoxContract(10.0, top, 130.0, top + 40.0)
    else:
        box = BoxContract(10.0, top, 190.0, top + 40.0)
    width = int(box.x_max - box.x_min)
    height = int(box.y_max - box.y_min)
    data = _png(width, height)
    left, y0 = box.x_min, box.y_min
    transform = HomographyContract(
        forward=(1.0, 0.0, -left, 0.0, 1.0, -y0, 0.0, 0.0, 1.0),
        inverse=(1.0, 0.0, left, 0.0, 1.0, y0, 0.0, 0.0, 1.0),
    )
    return RuntimeRoiArtifact(
        roi_id=f"{measure_id}:{kind}",
        kind=kind,
        measure_id=measure_id,
        staff_id=staff_id,
        source_image_sha256=SOURCE_SHA,
        roi_image_sha256=sha256(data).hexdigest(),
        crop_bbox=box,
        source_to_roi=transform,
        png_bytes=data,
    )


def _batch(*artifacts: RuntimeRoiArtifact) -> RuntimeRoiBatch:
    return RuntimeRoiBatch(SOURCE_SHA, "d" * 64, tuple(artifacts))


def _scores(digit: int | None) -> MeterDigitScoresV3:
    if digit == 2:
        return MeterDigitScoresV3(500, 0, 0)
    if digit == 3:
        return MeterDigitScoresV3(0, 700, 0)
    if digit == 4:
        return MeterDigitScoresV3(0, 0, 500)
    return MeterDigitScoresV3(0, 0, 0)


def _evidence(
    measure_id: str = "staff-1-measure-1",
    staff_id: str = "staff-1",
    system_id: str = "system-1",
    logical_id: str = "system-1-measure-1",
    roi_kind: str = "measure-start",
    presence: float = 0.99,
    upper: int | None = 2,
    lower: int | None = 4,
    status: str = "accepted",
) -> MeterModelEvidenceV3:
    return MeterModelEvidenceV3(
        evidence_id=f"e:{measure_id}",
        system_id=system_id,
        logical_measure_id=logical_id,
        measure_id=measure_id,
        staff_id=staff_id,
        roi_id=f"{measure_id}:{roi_kind}",
        presence_status=status,
        presence_score=presence,
        refined_x_center_roi=30.0,
        numerator_scores=_scores(upper),
        denominator_scores=_scores(lower),
        reasons=() if status == "accepted" else ("upstream-presence",),
    )


class RuntimeMeterIntegrationV3Tests(unittest.TestCase):
    def test_accepts_2_4_with_exact_measure_system_and_roi_ownership(self) -> None:
        geometry, report = _single_geometry()
        roi = _roi("staff-1-measure-1", "staff-1", 20.0)
        result = integrate_meter_evidence_v3(geometry, report, _batch(roi), (_evidence(),))
        decision = result.decisions[0]
        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.meter_class, "2/4")
        self.assertIsNotNone(decision.bbox)
        self.assertEqual(result.evidence_batch.observations[0].class_label, "2/4")
        self.assertEqual(result.evidence_batch.observations[0].status, "accepted")

    def test_presence_absent_with_no_digit_accepts_none(self) -> None:
        geometry, report = _single_geometry()
        roi = _roi("staff-1-measure-1", "staff-1", 20.0)
        result = integrate_meter_evidence_v3(
            geometry,
            report,
            _batch(roi),
            (_evidence(presence=0.10, upper=None, lower=None),),
        )
        self.assertEqual((result.decisions[0].status, result.decisions[0].meter_class), ("accepted", "none"))
        self.assertIsNone(result.decisions[0].bbox)

    def test_presence_absent_with_passing_digit_is_ambiguous(self) -> None:
        geometry, report = _single_geometry()
        roi = _roi("staff-1-measure-1", "staff-1", 20.0)
        result = integrate_meter_evidence_v3(
            geometry,
            report,
            _batch(roi),
            (_evidence(presence=0.10, upper=2, lower=None),),
        )
        self.assertEqual(result.decisions[0].reasons, (M12_PRESENCE_DIGIT_CONFLICT,))

    def test_present_with_no_digit_reports_no_digit(self) -> None:
        geometry, report = _single_geometry()
        roi = _roi("staff-1-measure-1", "staff-1", 20.0)
        result = integrate_meter_evidence_v3(geometry, report, _batch(roi), (_evidence(upper=None, lower=None),))
        self.assertEqual(result.decisions[0].reasons, (M08_NO_DIGIT,))

    def test_missing_upper_and_lower_are_distinct(self) -> None:
        geometry, report = _single_geometry()
        roi = _roi("staff-1-measure-1", "staff-1", 20.0)
        upper_missing = integrate_meter_evidence_v3(geometry, report, _batch(roi), (_evidence(upper=None, lower=4),))
        lower_missing = integrate_meter_evidence_v3(geometry, report, _batch(roi), (_evidence(upper=2, lower=None),))
        self.assertEqual(upper_missing.decisions[0].reasons, (M09_UPPER_DIGIT_NOT_FOUND,))
        self.assertEqual(lower_missing.decisions[0].reasons, (M10_LOWER_DIGIT_NOT_FOUND,))

    def test_multiple_passing_digit_specialists_fail_ambiguous(self) -> None:
        geometry, report = _single_geometry()
        roi = _roi("staff-1-measure-1", "staff-1", 20.0)
        evidence = _evidence()
        evidence = MeterModelEvidenceV3(
            evidence_id=evidence.evidence_id,
            system_id=evidence.system_id,
            logical_measure_id=evidence.logical_measure_id,
            measure_id=evidence.measure_id,
            staff_id=evidence.staff_id,
            roi_id=evidence.roi_id,
            presence_status="accepted",
            presence_score=0.99,
            refined_x_center_roi=30.0,
            numerator_scores=MeterDigitScoresV3(500, 700, 0),
            denominator_scores=_scores(4),
        )
        result = integrate_meter_evidence_v3(geometry, report, _batch(roi), (evidence,))
        self.assertEqual(result.decisions[0].reasons, (M11_DIGIT_SPECIALIST_CONFLICT,))

    def test_wrong_presence_region_does_not_fallback_to_measure_start(self) -> None:
        geometry, report = _single_geometry()
        start = _roi("staff-1-measure-1", "staff-1", 20.0)
        full = _roi("staff-1-measure-1", "staff-1", 20.0, "measure-full")
        result = integrate_meter_evidence_v3(
            geometry,
            report,
            _batch(start, full),
            (_evidence(roi_kind="measure-full"),),
        )
        self.assertEqual(result.decisions[0].reasons, (M04_WRONG_PRESENCE_REGION,))

    def test_wrong_system_or_logical_measure_identity_fails_closed(self) -> None:
        geometry, report = _single_geometry()
        roi = _roi("staff-1-measure-1", "staff-1", 20.0)
        wrong_system = integrate_meter_evidence_v3(
            geometry,
            report,
            _batch(roi),
            (_evidence(system_id="system-99"),),
        )
        wrong_logical = integrate_meter_evidence_v3(
            geometry,
            report,
            _batch(roi),
            (_evidence(logical_id="system-99-measure-1"),),
        )
        self.assertEqual(wrong_system.decisions[0].reasons, (M05_IDENTITY_MISMATCH,))
        self.assertEqual(wrong_logical.decisions[0].reasons, (M05_IDENTITY_MISMATCH,))

    def test_boundary_report_must_bind_exact_output_geometry(self) -> None:
        geometry, report = _single_geometry()
        roi = _roi("staff-1-measure-1", "staff-1", 20.0)
        bad_report = MeasureSystemBoundaryReportV2(
            status="accepted",
            input_geometry_fingerprint=report.input_geometry_fingerprint,
            output_geometry_fingerprint="e" * 64,
            logical_measures=report.logical_measures,
        )
        result = integrate_meter_evidence_v3(geometry, bad_report, _batch(roi), (_evidence(),))
        self.assertEqual(result.decisions[0].reasons, (M02_BOUNDARY_REPORT_MISMATCH,))

    def test_missing_evidence_is_explicit_ambiguous_and_propagated(self) -> None:
        geometry, report = _single_geometry()
        roi = _roi("staff-1-measure-1", "staff-1", 20.0)
        result = integrate_meter_evidence_v3(geometry, report, _batch(roi), ())
        self.assertEqual(result.decisions[0].reasons, (M03_METER_EVIDENCE_MISSING,))
        observation = result.evidence_batch.observations[0]
        self.assertEqual(observation.status, "ambiguous")
        self.assertEqual(observation.reasons, (M03_METER_EVIDENCE_MISSING,))

    def test_ambiguous_presence_remains_ambiguous(self) -> None:
        geometry, report = _single_geometry()
        roi = _roi("staff-1-measure-1", "staff-1", 20.0)
        result = integrate_meter_evidence_v3(
            geometry,
            report,
            _batch(roi),
            (_evidence(status="ambiguous"),),
        )
        self.assertEqual(result.decisions[0].reasons, (M06_PRESENCE_AMBIGUOUS,))
        self.assertEqual(result.evidence_batch.observations[0].status, "ambiguous")

    def test_cross_staff_meter_disagreement_marks_entire_logical_measure_ambiguous(self) -> None:
        geometry, report = _two_staff_geometry()
        roi1 = _roi("staff-1-measure-1", "staff-1", 20.0)
        roi2 = _roi("staff-2-measure-1", "staff-2", 70.0)
        e1 = _evidence()
        e2 = _evidence(
            measure_id="staff-2-measure-1",
            staff_id="staff-2",
            upper=3,
        )
        result = integrate_meter_evidence_v3(geometry, report, _batch(roi1, roi2), (e1, e2))
        self.assertEqual(len(result.decisions), 2)
        self.assertTrue(all(item.status == "ambiguous" for item in result.decisions))
        self.assertTrue(all(M15_CROSS_STAFF_METER_MISMATCH in item.reasons for item in result.decisions))
        self.assertTrue(all(item.meter_class is None for item in result.decisions))

    def test_cross_staff_same_meter_is_accepted(self) -> None:
        geometry, report = _two_staff_geometry()
        roi1 = _roi("staff-1-measure-1", "staff-1", 20.0)
        roi2 = _roi("staff-2-measure-1", "staff-2", 70.0)
        e1 = _evidence()
        e2 = _evidence(measure_id="staff-2-measure-1", staff_id="staff-2")
        result = integrate_meter_evidence_v3(geometry, report, _batch(roi1, roi2), (e1, e2))
        self.assertEqual(tuple(item.meter_class for item in result.decisions), ("2/4", "2/4"))
        self.assertTrue(all(item.status == "accepted" for item in result.decisions))

    def test_replay_is_bit_stable_10_of_10(self) -> None:
        geometry, report = _single_geometry()
        roi = _roi("staff-1-measure-1", "staff-1", 20.0)
        evidence = (_evidence(),)
        fingerprints = tuple(
            integrate_meter_evidence_v3(geometry, report, _batch(roi), evidence).fingerprint()
            for _ in range(10)
        )
        self.assertEqual(len(set(fingerprints)), 1)

    def test_safety_gates_remain_closed(self) -> None:
        self.assertFalse(checkpoint_loading_allowed())
        self.assertFalse(train_validation_test_access_allowed())
        self.assertFalse(resolver_connection_allowed())


if __name__ == "__main__":
    unittest.main()
