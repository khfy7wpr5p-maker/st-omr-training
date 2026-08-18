from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import inspect
import math
import unittest

from PIL import Image

from st_omr_training.meter_v2_slot_geometry_adapter_v1 import (
    ACCEPTED,
    AMBIGUOUS,
    REJECTED,
    R_BBOX_OUTSIDE_ROI,
    R_IDENTITY,
    R_NONFINITE,
    R_ROI_KIND,
    R_STAFF_NOT_FOUND,
    R_UPSTREAM_NOT_ACCEPTED,
    R_X_OUTSIDE_STAFF,
    adapt_meter_slots_from_upstream_geometry_v1,
    meter_slot_geometry_adapter_production_accepted,
    meter_slot_geometry_adapter_train_derived_only,
    meter_slot_geometry_adapter_validation_tuned,
    resolver_connection_allowed,
)
from st_omr_training.runtime_geometry_engine_contract import (
    BoxContract,
    LineSegmentContract,
    MeasureProposalContract,
    PageGeometryContract,
    Point2DContract,
    StaffGeometryContract,
    SystemGeometryContract,
)
from st_omr_training.runtime_local_roi_v1 import RuntimeRoiArtifact
from st_omr_training.runtime_page_normalizer_contract import HomographyContract


SOURCE_SHA = "a" * 64
GEOM_SHA = "b" * 64


def _roi_png(width: int = 60, height: int = 50) -> bytes:
    image = Image.new("L", (width, height), 255)
    out = BytesIO()
    image.save(out, format="PNG", optimize=False, compress_level=9)
    return out.getvalue()


def _geometry(*, line_x0: float = 0.0, line_x1: float = 100.0, status: str = "accepted") -> PageGeometryContract:
    lines = tuple(
        LineSegmentContract(Point2DContract(line_x0, y), Point2DContract(line_x1, y))
        for y in (10.0, 20.0, 30.0, 40.0, 50.0)
    )
    staff = StaffGeometryContract(
        staff_id="staff-1",
        system_id="system-1",
        five_staff_lines=lines,
        staff_bbox=BoxContract(0.0, 5.0, 100.0, 55.0),
        staff_spacing=10.0,
    )
    system = SystemGeometryContract(
        system_id="system-1",
        system_bbox=BoxContract(0.0, 5.0, 100.0, 55.0),
        staff_ids=("staff-1",),
    )
    measure = MeasureProposalContract(
        measure_id="staff-1-measure-1",
        system_id="system-1",
        staff_id="staff-1",
        bbox=BoxContract(20.0, 5.0, 100.0, 55.0),
        left_boundary=LineSegmentContract(Point2DContract(20.0, 5.0), Point2DContract(20.0, 55.0)),
        right_boundary=LineSegmentContract(Point2DContract(100.0, 5.0), Point2DContract(100.0, 55.0)),
        status="accepted",
    )
    return PageGeometryContract(
        normalized_image_sha256=SOURCE_SHA,
        geometry_config_fingerprint=GEOM_SHA,
        page_width=120,
        page_height=80,
        transform=HomographyContract(
            forward=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            inverse=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        ),
        systems=(system,),
        staffs=(staff,),
        measure_proposals=(measure,),
        status=status,
        reasons=() if status == "accepted" else ("synthetic-ambiguity",),
    )


def _roi(
    *,
    kind: str = "measure-start",
    source_sha: str = SOURCE_SHA,
    staff_id: str = "staff-1",
    crop_right: float = 80.0,
) -> RuntimeRoiArtifact:
    width = int(crop_right - 20.0)
    data = _roi_png(width=width, height=50)
    return RuntimeRoiArtifact(
        roi_id=f"staff-1-measure-1:{kind}",
        kind=kind,
        measure_id="staff-1-measure-1",
        staff_id=staff_id,
        source_image_sha256=source_sha,
        roi_image_sha256=sha256(data).hexdigest(),
        crop_bbox=BoxContract(20.0, 5.0, crop_right, 55.0),
        source_to_roi=HomographyContract(
            forward=(1.0, 0.0, -20.0, 0.0, 1.0, -5.0, 0.0, 0.0, 1.0),
            inverse=(1.0, 0.0, 20.0, 0.0, 1.0, 5.0, 0.0, 0.0, 1.0),
        ),
        png_bytes=data,
    )


class MeterSlotGeometryAdapterV1Tests(unittest.TestCase):
    def test_maps_upstream_staff_lines_to_two_staff_relative_slots(self) -> None:
        result = adapt_meter_slots_from_upstream_geometry_v1(
            _geometry(), _roi(), refined_x_center_roi=25.0
        )
        self.assertEqual(result.status, ACCEPTED)
        self.assertEqual(result.reasons, ())
        self.assertAlmostEqual(result.local_staff_spacing, 10.0)
        self.assertIsNotNone(result.numerator_bbox)
        self.assertIsNotNone(result.denominator_bbox)
        assert result.numerator_bbox is not None and result.denominator_bbox is not None
        self.assertAlmostEqual((result.numerator_bbox.y0 + result.numerator_bbox.y1) / 2.0, 15.0)
        self.assertAlmostEqual((result.denominator_bbox.y0 + result.denominator_bbox.y1) / 2.0, 35.0)
        self.assertAlmostEqual(result.numerator_bbox.x1 - result.numerator_bbox.x0, 15.960569245912566)
        self.assertAlmostEqual(result.numerator_bbox.y1 - result.numerator_bbox.y0, 20.0)

    def test_repeatability_is_exact_for_same_contracts(self) -> None:
        outputs = tuple(
            adapt_meter_slots_from_upstream_geometry_v1(
                _geometry(), _roi(), refined_x_center_roi=25.0
            )
            for _ in range(10)
        )
        self.assertEqual(len(set(outputs)), 1)

    def test_nonfinite_x_rejects(self) -> None:
        result = adapt_meter_slots_from_upstream_geometry_v1(
            _geometry(), _roi(), refined_x_center_roi=float("nan")
        )
        self.assertEqual((result.status, result.reasons), (REJECTED, (R_NONFINITE,)))

    def test_upstream_ambiguity_is_not_reinterpreted(self) -> None:
        result = adapt_meter_slots_from_upstream_geometry_v1(
            _geometry(status="ambiguous"), _roi(), refined_x_center_roi=25.0
        )
        self.assertEqual((result.status, result.reasons), (AMBIGUOUS, (R_UPSTREAM_NOT_ACCEPTED,)))

    def test_only_measure_start_roi_is_admitted(self) -> None:
        result = adapt_meter_slots_from_upstream_geometry_v1(
            _geometry(), _roi(kind="measure-full"), refined_x_center_roi=25.0
        )
        self.assertEqual((result.status, result.reasons), (REJECTED, (R_ROI_KIND,)))

    def test_source_identity_mismatch_rejects(self) -> None:
        result = adapt_meter_slots_from_upstream_geometry_v1(
            _geometry(), _roi(source_sha="c" * 64), refined_x_center_roi=25.0
        )
        self.assertEqual((result.status, result.reasons), (REJECTED, (R_IDENTITY,)))

    def test_unknown_staff_rejects(self) -> None:
        result = adapt_meter_slots_from_upstream_geometry_v1(
            _geometry(), _roi(staff_id="staff-x"), refined_x_center_roi=25.0
        )
        self.assertEqual((result.status, result.reasons), (REJECTED, (R_STAFF_NOT_FOUND,)))

    def test_x_outside_upstream_line_support_fails_closed(self) -> None:
        # ROI x=50 -> source x=70; staff lines stop at x=60.
        result = adapt_meter_slots_from_upstream_geometry_v1(
            _geometry(line_x0=0.0, line_x1=60.0),
            _roi(crop_right=100.0),
            refined_x_center_roi=50.0,
        )
        self.assertEqual((result.status, result.reasons), (AMBIGUOUS, (R_X_OUTSIDE_STAFF,)))

    def test_slot_that_would_cross_roi_boundary_fails_closed(self) -> None:
        result = adapt_meter_slots_from_upstream_geometry_v1(
            _geometry(), _roi(), refined_x_center_roi=1.0
        )
        self.assertEqual((result.status, result.reasons), (AMBIGUOUS, (R_BBOX_OUTSIDE_ROI,)))

    def test_safety_boundary_and_no_duplicate_staff_detector(self) -> None:
        import st_omr_training.meter_v2_slot_geometry_adapter_v1 as module

        source = inspect.getsource(module)
        self.assertNotIn("detect_multistaff_geometry_v2", source)
        self.assertNotIn("runtime_geometry_engine_v2", source)
        self.assertNotIn("runtime_deterministic_resolver_v1", source)
        self.assertNotIn("import torch", source)
        self.assertNotIn("from PIL", source)
        self.assertTrue(meter_slot_geometry_adapter_train_derived_only())
        self.assertFalse(meter_slot_geometry_adapter_validation_tuned())
        self.assertFalse(meter_slot_geometry_adapter_production_accepted())
        self.assertFalse(resolver_connection_allowed())


if __name__ == "__main__":
    unittest.main()
