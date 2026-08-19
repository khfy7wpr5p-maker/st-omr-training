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
from st_omr_training.runtime_local_roi_v1 import (
    RuntimeRoiArtifact,
    RuntimeRoiBatch,
    runtime_roi_v1_config_fingerprint,
)
from st_omr_training.runtime_measure_system_boundaries_v2 import (
    LogicalMeasureV2,
    MeasureSystemBoundaryReportV2,
)
from st_omr_training.runtime_meter_integration_v3 import (
    M02_BOUNDARY_REPORT_MISMATCH,
    M05_IDENTITY_MISMATCH,
    MeterDigitScoresV3,
    MeterModelEvidenceV3,
    integrate_meter_evidence_v3,
)
from st_omr_training.runtime_page_normalizer_contract import HomographyContract
from st_omr_training.runtime_system_grouper_v1 import page_geometry_fingerprint_v1


SOURCE_SHA = "a" * 64
CONFIG_SHA = "b" * 64


def _identity() -> HomographyContract:
    return HomographyContract(
        forward=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        inverse=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )


def _line(x0: float, x1: float, y: float) -> LineSegmentContract:
    return LineSegmentContract(Point2DContract(x0, y), Point2DContract(x1, y))


def _png(width: int, height: int) -> bytes:
    image = Image.new("L", (width, height), 255)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _geometry(
    *,
    x_min: float = 10.0,
    y_min: float = 20.0,
    x_max: float = 190.0,
    y_max: float = 60.0,
    measure_status: str = "accepted",
) -> tuple[PageGeometryContract, MeasureSystemBoundaryReportV2]:
    staff = StaffGeometryContract(
        staff_id="staff-1",
        system_id="system-1",
        five_staff_lines=tuple(_line(x_min, x_max, y_min + 10.0 * index) for index in range(5)),
        staff_bbox=BoxContract(x_min, y_min, x_max, y_max),
        staff_spacing=10.0,
    )
    measure = MeasureProposalContract(
        measure_id="measure-1",
        system_id="system-1",
        staff_id="staff-1",
        bbox=BoxContract(x_min, y_min, x_max, y_max),
        left_boundary=LineSegmentContract(Point2DContract(x_min, y_min), Point2DContract(x_min, y_max)),
        right_boundary=LineSegmentContract(Point2DContract(x_max, y_min), Point2DContract(x_max, y_max)),
        status=measure_status,
        reasons=() if measure_status == "accepted" else ("fixture-nonaccepted-measure",),
    )
    geometry = PageGeometryContract(
        normalized_image_sha256=SOURCE_SHA,
        geometry_config_fingerprint=CONFIG_SHA,
        page_width=220,
        page_height=140,
        transform=_identity(),
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
                x_min,
                x_max,
                "system_edge",
                "system_edge",
                (measure.measure_id,),
            ),
        ),
    )
    return geometry, report


def _roi(
    geometry: PageGeometryContract,
    *,
    measure_id: str = "measure-1",
    staff_id: str = "staff-1",
    box: BoxContract | None = None,
) -> RuntimeRoiArtifact:
    measure = geometry.measure_proposals[0]
    if box is None:
        desired_x_max = min(measure.bbox.x_max, measure.bbox.x_min + 12.0 * geometry.staffs[0].staff_spacing)
        left = max(0, int(measure.bbox.x_min // 1))
        top = max(0, int(measure.bbox.y_min // 1))
        # Values in these fixtures are positive; integer-ceil without importing
        # producer internals keeps the contract test independent of its helper.
        right = min(geometry.page_width, int(-(-desired_x_max // 1)))
        bottom = min(geometry.page_height, int(-(-measure.bbox.y_max // 1)))
        box = BoxContract(float(left), float(top), float(right), float(bottom))
    width = int(box.x_max - box.x_min)
    height = int(box.y_max - box.y_min)
    data = _png(width, height)
    left = float(box.x_min)
    top = float(box.y_min)
    return RuntimeRoiArtifact(
        roi_id=f"{measure_id}:measure-start",
        kind="measure-start",
        measure_id=measure_id,
        staff_id=staff_id,
        source_image_sha256=SOURCE_SHA,
        roi_image_sha256=sha256(data).hexdigest(),
        crop_bbox=box,
        source_to_roi=HomographyContract(
            forward=(1.0, 0.0, -left, 0.0, 1.0, -top, 0.0, 0.0, 1.0),
            inverse=(1.0, 0.0, left, 0.0, 1.0, top, 0.0, 0.0, 1.0),
        ),
        png_bytes=data,
    )


def _batch(geometry: PageGeometryContract, *artifacts: RuntimeRoiArtifact) -> RuntimeRoiBatch:
    return RuntimeRoiBatch(
        source_image_sha256=SOURCE_SHA,
        config_fingerprint=runtime_roi_v1_config_fingerprint(geometry.geometry_config_fingerprint),
        artifacts=tuple(artifacts),
    )


def _evidence(
    *,
    evidence_id: str = "meter-evidence-1",
    measure_id: str = "measure-1",
    staff_id: str = "staff-1",
    system_id: str = "system-1",
    logical_measure_id: str = "system-1-measure-1",
) -> MeterModelEvidenceV3:
    return MeterModelEvidenceV3(
        evidence_id=evidence_id,
        system_id=system_id,
        logical_measure_id=logical_measure_id,
        measure_id=measure_id,
        staff_id=staff_id,
        roi_id=f"{measure_id}:measure-start",
        presence_status="accepted",
        presence_score=0.99,
        refined_x_center_roi=30.0,
        numerator_scores=MeterDigitScoresV3(500, 0, 0),
        denominator_scores=MeterDigitScoresV3(0, 0, 500),
    )


class MeterRuntimeV3SecondShadowHardeningTests(unittest.TestCase):
    def test_fractional_measure_accepts_exact_producer_integer_rounding(self) -> None:
        geometry, report = _geometry(x_min=10.2, y_min=20.2, x_max=190.4, y_max=60.4)
        roi = _roi(geometry)
        # Runtime Local ROI floor/ceil rounding intentionally extends the crop
        # slightly outside the float measure bbox. That is valid producer output.
        self.assertLess(roi.crop_bbox.x_min, geometry.measure_proposals[0].bbox.x_min)
        self.assertLess(roi.crop_bbox.y_min, geometry.measure_proposals[0].bbox.y_min)
        result = integrate_meter_evidence_v3(geometry, report, _batch(geometry, roi), (_evidence(),))
        self.assertEqual(result.decisions[0].status, "accepted")
        self.assertEqual(result.decisions[0].meter_class, "2/4")

    def test_nonaccepted_measure_on_accepted_page_cannot_receive_meter(self) -> None:
        geometry, report = _geometry(measure_status="rejected")
        roi = _roi(geometry)
        result = integrate_meter_evidence_v3(geometry, report, _batch(geometry, roi), (_evidence(),))
        self.assertEqual(result.decisions[0].status, "ambiguous")
        self.assertEqual(result.decisions[0].reasons, (M02_BOUNDARY_REPORT_MISMATCH,))

    def test_extra_unowned_meter_evidence_invalidates_batch(self) -> None:
        geometry, report = _geometry()
        roi = _roi(geometry)
        ghost = _evidence(
            evidence_id="ghost-evidence",
            measure_id="ghost-measure",
            staff_id="staff-1",
            logical_measure_id="ghost-logical",
        )
        result = integrate_meter_evidence_v3(
            geometry,
            report,
            _batch(geometry, roi),
            (_evidence(), ghost),
        )
        self.assertEqual(result.decisions[0].status, "ambiguous")
        self.assertEqual(result.decisions[0].reasons, (M05_IDENTITY_MISMATCH,))

    def test_extra_unowned_roi_invalidates_batch(self) -> None:
        geometry, report = _geometry()
        roi = _roi(geometry)
        ghost = _roi(
            geometry,
            measure_id="ghost-measure",
            staff_id="staff-1",
            box=BoxContract(10.0, 20.0, 40.0, 60.0),
        )
        result = integrate_meter_evidence_v3(
            geometry,
            report,
            _batch(geometry, roi, ghost),
            (_evidence(),),
        )
        self.assertEqual(result.decisions[0].status, "ambiguous")
        self.assertEqual(result.decisions[0].reasons, (M05_IDENTITY_MISMATCH,))


if __name__ == "__main__":
    unittest.main()
