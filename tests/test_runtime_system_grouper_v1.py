from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import unittest

from PIL import Image, ImageDraw

from st_omr_training.runtime_geometry_engine_contract import (
    BoxContract,
    GeometryInputContract,
    LineSegmentContract,
    MeasureProposalContract,
    PageGeometryContract,
    Point2DContract,
)
from st_omr_training.runtime_geometry_engine_v2 import detect_multistaff_geometry_v2
from st_omr_training.runtime_page_normalizer_contract import HomographyContract
from st_omr_training.runtime_system_grouper_v1 import (
    G01_UPSTREAM_GEOMETRY_NOT_ACCEPTED,
    G02_MEASURE_GEOMETRY_ALREADY_PRESENT,
    G03_DECLARED_POLICY_STAFF_COUNT_MISMATCH,
    G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP,
    G05_INVALID_STAFF_ORDER,
    G06_RASTER_EVIDENCE_REQUIRED,
    MIN_CONNECTOR_COVERAGE_MILLI,
    SYSTEM_GROUPER_REASON_PRIORITY,
    SystemGrouperV1Error,
    SystemGroupingReportV1,
    group_staffs_into_systems_v1,
    page_geometry_fingerprint_v1,
)


IDENTITY = HomographyContract(
    forward=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    inverse=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
)
NORMALIZER_FP = sha256(b"system-grouper-v1-normalizer").hexdigest()


def _png_with_staff_tops(
    tops: tuple[int, ...],
    *,
    connectors: tuple[int, ...] = (),
    height: int = 460,
) -> bytes:
    image = Image.new("L", (320, height), 255)
    draw = ImageDraw.Draw(image)
    for top in tops:
        for offset in (0, 10, 20, 30, 40):
            draw.line((20, top + offset, 300, top + offset), fill=0, width=1)
    for upper_index in connectors:
        if upper_index < 0 or upper_index + 1 >= len(tops):
            raise ValueError("connector index must reference one adjacent staff pair")
        draw.line((20, tops[upper_index] + 40, 20, tops[upper_index + 1]), fill=0, width=1)
    out = BytesIO()
    image.save(out, format="PNG", optimize=False, compress_level=9)
    return out.getvalue()


def _input(png: bytes, *, height: int = 460) -> GeometryInputContract:
    return GeometryInputContract(
        normalized_image_sha256=sha256(png).hexdigest(),
        normalizer_config_fingerprint=NORMALIZER_FP,
        normalized_width=320,
        normalized_height=height,
        transform=IDENTITY,
    )


def _accepted_staff_page_from_png(png: bytes, *, height: int = 460) -> PageGeometryContract:
    result = detect_multistaff_geometry_v2(png, _input(png, height=height))
    if result.page.status != "accepted":
        raise AssertionError(f"test fixture must be accepted: {result.page.reasons}")
    return result.page


def _accepted_staff_page(tops: tuple[int, ...], *, height: int = 460) -> PageGeometryContract:
    return _accepted_staff_page_from_png(_png_with_staff_tops(tops, height=height), height=height)


class SystemGrouperContractV1Tests(unittest.TestCase):
    def test_reason_priority_is_frozen_and_report_is_canonical(self) -> None:
        self.assertEqual(
            SYSTEM_GROUPER_REASON_PRIORITY,
            (
                G01_UPSTREAM_GEOMETRY_NOT_ACCEPTED,
                G02_MEASURE_GEOMETRY_ALREADY_PRESENT,
                G05_INVALID_STAFF_ORDER,
                G03_DECLARED_POLICY_STAFF_COUNT_MISMATCH,
                G06_RASTER_EVIDENCE_REQUIRED,
                G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP,
            ),
        )
        digest = sha256(b"input").hexdigest()
        report = SystemGroupingReportV1(
            status="ambiguous",
            policy="auto-v1",
            input_geometry_fingerprint=digest,
            output_geometry_fingerprint=None,
            primary_reason=G05_INVALID_STAFF_ORDER,
            secondary_reasons=(G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP,),
            adjacent_connector_coverages_milli=(0,),
        )
        self.assertEqual(
            report.active_reasons,
            (G05_INVALID_STAFF_ORDER, G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP),
        )
        self.assertEqual(report.adjacent_connector_coverages_milli, (0,))
        self.assertEqual(report.fingerprint(), report.fingerprint())

    def test_unknown_noncanonical_or_invalid_evidence_fails_closed(self) -> None:
        digest = sha256(b"input").hexdigest()
        with self.assertRaises(ValueError):
            SystemGroupingReportV1(
                status="ambiguous",
                policy="auto-v1",
                input_geometry_fingerprint=digest,
                output_geometry_fingerprint=None,
                primary_reason="G99_UNKNOWN",
            )
        with self.assertRaises(ValueError):
            SystemGroupingReportV1(
                status="ambiguous",
                policy="auto-v1",
                input_geometry_fingerprint=digest,
                output_geometry_fingerprint=None,
                primary_reason=G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP,
                secondary_reasons=(G05_INVALID_STAFF_ORDER,),
            )
        with self.assertRaises(ValueError):
            SystemGroupingReportV1(
                status="ambiguous",
                policy="auto-v1",
                input_geometry_fingerprint=digest,
                output_geometry_fingerprint=None,
                primary_reason=G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP,
                adjacent_connector_coverages_milli=(1001,),
            )


class RuntimeSystemGrouperV1Tests(unittest.TestCase):
    def test_auto_single_staff_is_unambiguous_and_accepted_without_raster(self) -> None:
        page = _accepted_staff_page((40,))
        result = group_staffs_into_systems_v1(page)
        self.assertEqual(result.report.status, "accepted")
        self.assertIsNotNone(result.page)
        assert result.page is not None
        self.assertEqual(tuple(system.system_id for system in result.page.systems), ("system-1",))
        self.assertEqual(result.page.systems[0].staff_ids, ("staff-1",))
        self.assertEqual(result.page.staffs[0].system_id, "system-1")

    def test_auto_multistaff_requires_hash_bound_raster(self) -> None:
        page = _accepted_staff_page((30, 140))
        result = group_staffs_into_systems_v1(page, policy="auto-v1")
        self.assertIsNone(result.page)
        self.assertEqual(result.report.status, "ambiguous")
        self.assertEqual(result.report.primary_reason, G06_RASTER_EVIDENCE_REQUIRED)

    def test_auto_multistaff_without_connector_fails_closed_instead_of_splitting(self) -> None:
        png = _png_with_staff_tops((30, 140))
        page = _accepted_staff_page_from_png(png)
        result = group_staffs_into_systems_v1(page, normalized_png=png, policy="auto-v1")
        self.assertIsNone(result.page)
        self.assertEqual(result.report.status, "ambiguous")
        self.assertEqual(result.report.primary_reason, G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP)
        self.assertEqual(result.report.adjacent_connector_coverages_milli, (0,))

    def test_auto_visible_grand_staff_connector_is_positive_raster_evidence(self) -> None:
        png = _png_with_staff_tops((30, 140), connectors=(0,))
        page = _accepted_staff_page_from_png(png)
        result = group_staffs_into_systems_v1(page, normalized_png=png, policy="auto-v1")
        self.assertEqual(result.report.status, "accepted")
        assert result.page is not None
        self.assertEqual(len(result.page.systems), 1)
        self.assertEqual(result.page.systems[0].staff_ids, ("staff-1", "staff-2"))
        self.assertEqual(len(result.report.adjacent_connector_coverages_milli), 1)
        self.assertGreaterEqual(
            result.report.adjacent_connector_coverages_milli[0],
            MIN_CONNECTOR_COVERAGE_MILLI,
        )

    def test_auto_partial_connector_pattern_is_ambiguous_not_partially_grouped(self) -> None:
        png = _png_with_staff_tops((30, 100, 240, 310), connectors=(0, 2))
        page = _accepted_staff_page_from_png(png)
        result = group_staffs_into_systems_v1(page, normalized_png=png, policy="auto-v1")
        self.assertIsNone(result.page)
        self.assertEqual(result.report.primary_reason, G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP)
        self.assertEqual(len(result.report.adjacent_connector_coverages_milli), 3)
        self.assertGreaterEqual(result.report.adjacent_connector_coverages_milli[0], MIN_CONNECTOR_COVERAGE_MILLI)
        self.assertLess(result.report.adjacent_connector_coverages_milli[1], MIN_CONNECTOR_COVERAGE_MILLI)
        self.assertGreaterEqual(result.report.adjacent_connector_coverages_milli[2], MIN_CONNECTOR_COVERAGE_MILLI)

    def test_auto_raster_identity_mismatch_is_never_ignored(self) -> None:
        png = _png_with_staff_tops((30, 140), connectors=(0,))
        page = _accepted_staff_page_from_png(png)
        wrong = _png_with_staff_tops((30, 150), connectors=(0,))
        with self.assertRaises(SystemGrouperV1Error):
            group_staffs_into_systems_v1(page, normalized_png=wrong, policy="auto-v1")

    def test_monostaff_policy_rebinds_every_detected_staff_to_its_own_system(self) -> None:
        page = _accepted_staff_page((30, 140, 260))
        # Geometry v2 is intentionally staff-only and still puts every detected
        # staff in one provisional system.  The grouper must not trust that.
        self.assertEqual(len(page.systems), 1)
        self.assertEqual(tuple(staff.system_id for staff in page.staffs), ("system-1",) * 3)

        result = group_staffs_into_systems_v1(page, policy="monostaff-v1")
        self.assertEqual(result.report.status, "accepted")
        assert result.page is not None
        self.assertEqual(
            tuple(system.staff_ids for system in result.page.systems),
            (("staff-1",), ("staff-2",), ("staff-3",)),
        )
        self.assertEqual(
            tuple(staff.system_id for staff in result.page.staffs),
            ("system-1", "system-2", "system-3"),
        )
        self.assertEqual(
            tuple(system.system_bbox for system in result.page.systems),
            tuple(staff.staff_bbox for staff in result.page.staffs),
        )

    def test_fixed_two_staff_policy_pairs_adjacent_staffs_top_to_bottom(self) -> None:
        page = _accepted_staff_page((30, 100, 240, 310))
        result = group_staffs_into_systems_v1(page, policy="fixed-two-staff-v1")
        self.assertEqual(result.report.status, "accepted")
        assert result.page is not None
        self.assertEqual(
            tuple(system.staff_ids for system in result.page.systems),
            (("staff-1", "staff-2"), ("staff-3", "staff-4")),
        )
        self.assertEqual(
            tuple(staff.system_id for staff in result.page.staffs),
            ("system-1", "system-1", "system-2", "system-2"),
        )
        for system in result.page.systems:
            members = tuple(staff for staff in result.page.staffs if staff.system_id == system.system_id)
            self.assertEqual(system.system_bbox.y_min, min(staff.staff_bbox.y_min for staff in members))
            self.assertEqual(system.system_bbox.y_max, max(staff.staff_bbox.y_max for staff in members))

    def test_fixed_two_staff_policy_odd_count_is_explicitly_ambiguous(self) -> None:
        page = _accepted_staff_page((30, 140, 260))
        result = group_staffs_into_systems_v1(page, policy="fixed-two-staff-v1")
        self.assertIsNone(result.page)
        self.assertEqual(result.report.primary_reason, G03_DECLARED_POLICY_STAFF_COUNT_MISMATCH)

    def test_upstream_ambiguity_propagates_fail_closed(self) -> None:
        png = _png_with_staff_tops(())
        upstream = detect_multistaff_geometry_v2(png, _input(png))
        self.assertEqual(upstream.page.status, "ambiguous")
        result = group_staffs_into_systems_v1(upstream.page, policy="monostaff-v1")
        self.assertIsNone(result.page)
        self.assertEqual(result.report.status, "rejected")
        self.assertEqual(result.report.primary_reason, G01_UPSTREAM_GEOMETRY_NOT_ACCEPTED)

    def test_measure_geometry_cannot_be_silently_regrouped_after_the_fact(self) -> None:
        page = _accepted_staff_page((40,))
        staff = page.staffs[0]
        measure = MeasureProposalContract(
            measure_id="measure-1",
            system_id="system-1",
            staff_id=staff.staff_id,
            bbox=BoxContract(20.0, staff.staff_bbox.y_min, 160.0, staff.staff_bbox.y_max),
            left_boundary=LineSegmentContract(
                Point2DContract(20.0, staff.staff_bbox.y_min),
                Point2DContract(20.0, staff.staff_bbox.y_max),
            ),
            right_boundary=LineSegmentContract(
                Point2DContract(160.0, staff.staff_bbox.y_min),
                Point2DContract(160.0, staff.staff_bbox.y_max),
            ),
            status="accepted",
        )
        after_measure = PageGeometryContract(
            normalized_image_sha256=page.normalized_image_sha256,
            geometry_config_fingerprint=page.geometry_config_fingerprint,
            page_width=page.page_width,
            page_height=page.page_height,
            transform=page.transform,
            systems=page.systems,
            staffs=page.staffs,
            measure_proposals=(measure,),
            status="accepted",
        )
        result = group_staffs_into_systems_v1(after_measure, policy="monostaff-v1")
        self.assertIsNone(result.page)
        self.assertEqual(result.report.primary_reason, G02_MEASURE_GEOMETRY_ALREADY_PRESENT)

    def test_noncanonical_staff_order_is_not_reordered_silently(self) -> None:
        page = _accepted_staff_page((30, 140))
        reversed_page = PageGeometryContract(
            normalized_image_sha256=page.normalized_image_sha256,
            geometry_config_fingerprint=page.geometry_config_fingerprint,
            page_width=page.page_width,
            page_height=page.page_height,
            transform=page.transform,
            systems=page.systems,
            staffs=tuple(reversed(page.staffs)),
            measure_proposals=(),
            status="accepted",
        )
        result = group_staffs_into_systems_v1(reversed_page, policy="monostaff-v1")
        self.assertIsNone(result.page)
        self.assertEqual(result.report.primary_reason, G05_INVALID_STAFF_ORDER)

    def test_same_declared_policy_produces_identical_page_and_report_10_of_10(self) -> None:
        page = _accepted_staff_page((30, 100, 240, 310))
        results = [
            group_staffs_into_systems_v1(page, policy="fixed-two-staff-v1")
            for _ in range(10)
        ]
        self.assertTrue(all(result.page is not None for result in results))
        page_fingerprints = [page_geometry_fingerprint_v1(result.page) for result in results if result.page is not None]
        report_fingerprints = [result.report.fingerprint() for result in results]
        self.assertEqual(len(set(page_fingerprints)), 1)
        self.assertEqual(len(set(report_fingerprints)), 1)

    def test_same_auto_raster_produces_identical_evidence_and_report_10_of_10(self) -> None:
        png = _png_with_staff_tops((30, 140), connectors=(0,))
        page = _accepted_staff_page_from_png(png)
        results = [
            group_staffs_into_systems_v1(page, normalized_png=png, policy="auto-v1")
            for _ in range(10)
        ]
        self.assertTrue(all(result.page is not None for result in results))
        self.assertEqual(len({result.report.fingerprint() for result in results}), 1)
        self.assertEqual(len({result.report.adjacent_connector_coverages_milli for result in results}), 1)

    def test_policy_changes_output_fingerprint_without_changing_staff_observations(self) -> None:
        page = _accepted_staff_page((30, 140))
        mono = group_staffs_into_systems_v1(page, policy="monostaff-v1")
        paired = group_staffs_into_systems_v1(page, policy="fixed-two-staff-v1")
        assert mono.page is not None and paired.page is not None
        self.assertNotEqual(mono.report.output_geometry_fingerprint, paired.report.output_geometry_fingerprint)
        self.assertEqual(
            tuple((staff.staff_id, staff.five_staff_lines, staff.staff_bbox, staff.staff_spacing) for staff in mono.page.staffs),
            tuple((staff.staff_id, staff.five_staff_lines, staff.staff_bbox, staff.staff_spacing) for staff in paired.page.staffs),
        )


if __name__ == "__main__":
    unittest.main()
