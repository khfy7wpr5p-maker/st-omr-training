from __future__ import annotations

from dataclasses import fields
import inspect
import json
import math
import unittest

from st_omr_training.runtime_geometry_engine_contract import (
    BoxContract,
    GEOMETRY_FORBIDDEN_SEMANTICS,
    GeometryInputContract,
    LineSegmentContract,
    MeasureProposalContract,
    PageGeometryContract,
    Point2DContract,
    StaffGeometryContract,
    SystemGeometryContract,
    runtime_geometry_engine_contract_fingerprint,
    runtime_geometry_engine_contract_payload,
)
from st_omr_training.runtime_page_normalizer_contract import (
    HomographyContract,
    NormalizationOperationContract,
    NormalizedPageContract,
    RasterPageInputContract,
    runtime_page_normalizer_contract_fingerprint,
    runtime_page_normalizer_contract_payload,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
IDENTITY = HomographyContract(
    forward=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    inverse=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
)


def _line(y: float) -> LineSegmentContract:
    return LineSegmentContract(Point2DContract(10.0, y), Point2DContract(190.0, y))


def _valid_staff() -> StaffGeometryContract:
    return StaffGeometryContract(
        staff_id="staff-1",
        system_id="system-1",
        five_staff_lines=(_line(30), _line(40), _line(50), _line(60), _line(70)),
        staff_bbox=BoxContract(10, 25, 190, 75),
        staff_spacing=10.0,
    )


class PageNormalizerContractTests(unittest.TestCase):
    def test_contract_payload_and_fingerprint_are_canonical(self) -> None:
        payload_a = runtime_page_normalizer_contract_payload()
        payload_b = runtime_page_normalizer_contract_payload()
        raw_a = json.dumps(payload_a, sort_keys=True, separators=(",", ":"), allow_nan=False)
        raw_b = json.dumps(payload_b, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.assertEqual(raw_a, raw_b)
        fingerprint = runtime_page_normalizer_contract_fingerprint()
        self.assertEqual(len(fingerprint), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in fingerprint))

    def test_raster_input_is_bounded_and_fail_closed(self) -> None:
        page = RasterPageInputContract(
            source_id="sample",
            source_sha256=SHA_A,
            page_number=1,
            width=1200,
            height=1600,
            pixel_mode="rgb8",
            raster_sha256=SHA_B,
            dpi=300,
        )
        self.assertEqual(page.width, 1200)
        with self.assertRaises(ValueError):
            RasterPageInputContract(
                source_id="sample",
                source_sha256=SHA_A,
                page_number=1,
                width=0,
                height=1600,
                pixel_mode="rgb8",
                raster_sha256=SHA_B,
            )
        with self.assertRaises(ValueError):
            RasterPageInputContract(
                source_id="sample",
                source_sha256=SHA_A,
                page_number=1,
                width=1200,
                height=1600,
                pixel_mode="float32",
                raster_sha256=SHA_B,
            )

    def test_transform_round_trip_is_replayable(self) -> None:
        transform = HomographyContract(
            forward=(2.0, 0.0, 10.0, 0.0, 2.0, 20.0, 0.0, 0.0, 1.0),
            inverse=(0.5, 0.0, -5.0, 0.0, 0.5, -10.0, 0.0, 0.0, 1.0),
        )
        normalized = transform.original_to_normalized(13.25, 41.5)
        original = transform.normalized_to_original(*normalized)
        self.assertAlmostEqual(original[0], 13.25, places=9)
        self.assertAlmostEqual(original[1], 41.5, places=9)

    def test_invalid_or_nonfinite_transform_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            HomographyContract(
                forward=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                inverse=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            )
        with self.assertRaises(ValueError):
            HomographyContract(
                forward=(1.0, 0.0, math.inf, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                inverse=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            )

    def test_symbol_destructive_operation_is_forbidden(self) -> None:
        with self.assertRaises(ValueError):
            NormalizationOperationContract(
                operation_id="staff-removal",
                destructive_symbol_removal_allowed=True,
            )

    def test_accepted_page_requires_image_transform_and_no_rejection_reason(self) -> None:
        output = NormalizedPageContract(
            source_raster_sha256=SHA_A,
            normalizer_config_fingerprint=SHA_B,
            normalized_image_sha256=SHA_C,
            normalized_width=1200,
            normalized_height=1600,
            transform=IDENTITY,
            operations=(NormalizationOperationContract("deskew"),),
            status="accepted",
        )
        self.assertEqual(output.pixel_mode, "gray8")
        with self.assertRaises(ValueError):
            NormalizedPageContract(
                source_raster_sha256=SHA_A,
                normalizer_config_fingerprint=SHA_B,
                normalized_image_sha256=SHA_C,
                normalized_width=1200,
                normalized_height=1600,
                transform=IDENTITY,
                operations=(),
                status="accepted",
                rejection_reasons=("uncertain",),
            )

    def test_uncertain_page_must_explain_itself(self) -> None:
        ambiguous = NormalizedPageContract(
            source_raster_sha256=SHA_A,
            normalizer_config_fingerprint=SHA_B,
            normalized_image_sha256=None,
            normalized_width=None,
            normalized_height=None,
            transform=None,
            operations=(),
            status="ambiguous",
            rejection_reasons=("orientation-not-reliable",),
        )
        self.assertEqual(ambiguous.status, "ambiguous")
        with self.assertRaises(ValueError):
            NormalizedPageContract(
                source_raster_sha256=SHA_A,
                normalizer_config_fingerprint=SHA_B,
                normalized_image_sha256=None,
                normalized_width=None,
                normalized_height=None,
                transform=None,
                operations=(),
                status="rejected",
            )


class GeometryEngineContractTests(unittest.TestCase):
    def test_contract_payload_and_fingerprint_are_canonical(self) -> None:
        payload = runtime_geometry_engine_contract_payload()
        self.assertTrue(payload["output"]["measure_outputs_are_proposals"])
        self.assertTrue(payload["output"]["fail_closed_on_ambiguity"])
        fingerprint = runtime_geometry_engine_contract_fingerprint()
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(fingerprint, runtime_geometry_engine_contract_fingerprint())

    def test_geometry_input_is_bound_to_normalized_page_identity(self) -> None:
        item = GeometryInputContract(
            normalized_image_sha256=SHA_A,
            normalizer_config_fingerprint=SHA_B,
            normalized_width=200,
            normalized_height=100,
            transform=IDENTITY,
        )
        self.assertEqual(item.normalized_width, 200)

    def test_staff_requires_exactly_five_ordered_lines(self) -> None:
        staff = _valid_staff()
        self.assertEqual(len(staff.five_staff_lines), 5)
        with self.assertRaises(ValueError):
            StaffGeometryContract(
                staff_id="staff-1",
                system_id="system-1",
                five_staff_lines=(_line(30), _line(40), _line(50), _line(60)),
                staff_bbox=BoxContract(10, 25, 190, 75),
                staff_spacing=10.0,
            )
        with self.assertRaises(ValueError):
            StaffGeometryContract(
                staff_id="staff-1",
                system_id="system-1",
                five_staff_lines=(_line(30), _line(50), _line(40), _line(60), _line(70)),
                staff_bbox=BoxContract(10, 25, 190, 75),
                staff_spacing=10.0,
            )

    def test_measure_is_only_a_geometry_proposal(self) -> None:
        field_names = {field.name for field in fields(MeasureProposalContract)}
        for forbidden in GEOMETRY_FORBIDDEN_SEMANTICS:
            self.assertNotIn(forbidden, field_names)
        self.assertIn("status", field_names)
        self.assertIn("bbox", field_names)

    def test_ambiguous_measure_must_explain_itself(self) -> None:
        proposal = MeasureProposalContract(
            measure_id="m1",
            system_id="system-1",
            staff_id="staff-1",
            bbox=BoxContract(20, 25, 100, 75),
            left_boundary=LineSegmentContract(Point2DContract(20, 25), Point2DContract(20, 75)),
            right_boundary=LineSegmentContract(Point2DContract(100, 25), Point2DContract(100, 75)),
            status="ambiguous",
            reasons=("right-boundary-weak",),
        )
        self.assertEqual(proposal.status, "ambiguous")
        with self.assertRaises(ValueError):
            MeasureProposalContract(
                measure_id="m1",
                system_id="system-1",
                staff_id="staff-1",
                bbox=BoxContract(20, 25, 100, 75),
                left_boundary=LineSegmentContract(Point2DContract(20, 25), Point2DContract(20, 75)),
                right_boundary=LineSegmentContract(Point2DContract(100, 25), Point2DContract(100, 75)),
                status="ambiguous",
            )

    def test_page_geometry_rejects_unknown_links_and_out_of_bounds_boxes(self) -> None:
        staff = _valid_staff()
        system = SystemGeometryContract(
            system_id="system-1",
            system_bbox=BoxContract(0, 10, 200, 90),
            staff_ids=("staff-1",),
        )
        measure = MeasureProposalContract(
            measure_id="m1",
            system_id="system-1",
            staff_id="staff-1",
            bbox=BoxContract(20, 25, 100, 75),
            left_boundary=LineSegmentContract(Point2DContract(20, 25), Point2DContract(20, 75)),
            right_boundary=LineSegmentContract(Point2DContract(100, 25), Point2DContract(100, 75)),
            status="accepted",
        )
        page = PageGeometryContract(
            normalized_image_sha256=SHA_A,
            geometry_config_fingerprint=SHA_B,
            page_width=200,
            page_height=100,
            transform=IDENTITY,
            systems=(system,),
            staffs=(staff,),
            measure_proposals=(measure,),
            status="accepted",
        )
        self.assertEqual(page.measure_proposals[0].measure_id, "m1")

        bad_system = SystemGeometryContract(
            system_id="system-1",
            system_bbox=BoxContract(0, 10, 220, 90),
            staff_ids=("staff-1",),
        )
        with self.assertRaises(ValueError):
            PageGeometryContract(
                normalized_image_sha256=SHA_A,
                geometry_config_fingerprint=SHA_B,
                page_width=200,
                page_height=100,
                transform=IDENTITY,
                systems=(bad_system,),
                staffs=(staff,),
                measure_proposals=(measure,),
                status="accepted",
            )


class RuntimeIsolationTests(unittest.TestCase):
    def test_contracts_are_isolated_from_d10_d13_and_training_paths(self) -> None:
        import st_omr_training.runtime_geometry_engine_contract as geometry_module
        import st_omr_training.runtime_page_normalizer_contract as normalizer_module

        combined_source = inspect.getsource(normalizer_module) + inspect.getsource(geometry_module)
        forbidden_tokens = (
            "import stage7d10_",
            "import stage7d13_",
            "from .stage7d10_",
            "from .stage7d13_",
            "torch.optim",
            ".backward(",
            "DataLoader(",
            "torch.load(",
        )
        for token in forbidden_tokens:
            self.assertNotIn(token, combined_source)

        for payload in (
            runtime_page_normalizer_contract_payload(),
            runtime_geometry_engine_contract_payload(),
        ):
            isolation = payload["isolation"]
            self.assertFalse(isolation["stage7d10_read"])
            self.assertFalse(isolation["stage7d10_write"])
            self.assertFalse(isolation["stage7d13_read"])
            self.assertFalse(isolation["stage7d13_write"])
            self.assertFalse(isolation["checkpoint_access"])
            self.assertFalse(isolation["optimizer_access"])
            self.assertFalse(isolation["test_split_access"])


if __name__ == "__main__":
    unittest.main()
