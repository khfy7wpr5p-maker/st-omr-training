from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import unittest

from PIL import Image, ImageDraw

from st_omr_training.runtime_geometry_engine_contract import (
    GeometryInputContract,
    PageGeometryContract,
    SystemGeometryContract,
)
from st_omr_training.runtime_geometry_engine_v2 import detect_multistaff_geometry_v2
from st_omr_training.runtime_measure_system_boundaries_v2 import (
    B02_MEASURE_GEOMETRY_ALREADY_PRESENT,
    B03_SYSTEM_STAFF_MEMBERSHIP_INVALID,
    B04_SYSTEM_ORDER_INVALID,
    B05_CROSS_STAFF_BOUNDARY_MISMATCH,
    B06_MEASURE_TOO_NARROW,
    BOUNDARY_REASON_PRIORITY,
    MeasureSystemBoundariesV2Error,
    propose_measure_system_boundaries_v2,
)
from st_omr_training.runtime_page_normalizer_contract import HomographyContract
from st_omr_training.runtime_system_grouper_v1 import group_staffs_into_systems_v1


IDENTITY = HomographyContract(
    forward=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    inverse=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
)
NORMALIZER_FP = sha256(b"measure-system-v2-normalizer").hexdigest()


def _page_png(
    tops: tuple[int, ...],
    *,
    verticals_by_staff: tuple[tuple[int, ...], ...] | None = None,
    x_start: int = 20,
    x_end: int = 300,
    height: int = 460,
) -> bytes:
    if verticals_by_staff is None:
        verticals_by_staff = tuple(() for _ in tops)
    if len(verticals_by_staff) != len(tops):
        raise ValueError("one vertical sequence is required per staff")
    image = Image.new("L", (320, height), 255)
    draw = ImageDraw.Draw(image)
    for top, verticals in zip(tops, verticals_by_staff):
        for offset in (0, 10, 20, 30, 40):
            draw.line((x_start, top + offset, x_end, top + offset), fill=0, width=1)
        for x in verticals:
            draw.line((x, top, x, top + 40), fill=0, width=1)
    out = BytesIO()
    image.save(out, format="PNG", optimize=False, compress_level=9)
    return out.getvalue()


def _detect(png: bytes, *, height: int = 460) -> PageGeometryContract:
    contract = GeometryInputContract(
        normalized_image_sha256=sha256(png).hexdigest(),
        normalizer_config_fingerprint=NORMALIZER_FP,
        normalized_width=320,
        normalized_height=height,
        transform=IDENTITY,
    )
    result = detect_multistaff_geometry_v2(png, contract)
    if result.page.status != "accepted":
        raise AssertionError(f"fixture must be accepted staff geometry: {result.page.reasons}")
    return result.page


def _group(
    png: bytes,
    *,
    policy: str,
    height: int = 460,
) -> PageGeometryContract:
    detected = _detect(png, height=height)
    grouped = group_staffs_into_systems_v1(detected, policy=policy)
    if grouped.page is None or grouped.report.status != "accepted":
        raise AssertionError(f"fixture must group cleanly: {grouped.report.active_reasons}")
    return grouped.page


class MeasureSystemBoundariesV2Tests(unittest.TestCase):
    def test_reason_priority_is_frozen(self) -> None:
        self.assertEqual(
            BOUNDARY_REASON_PRIORITY,
            (
                "B01_UPSTREAM_GEOMETRY_NOT_ACCEPTED",
                "B02_MEASURE_GEOMETRY_ALREADY_PRESENT",
                "B03_SYSTEM_STAFF_MEMBERSHIP_INVALID",
                "B04_SYSTEM_ORDER_INVALID",
                "B05_CROSS_STAFF_BOUNDARY_MISMATCH",
                "B06_MEASURE_TOO_NARROW",
            ),
        )

    def test_one_measure_system_uses_implicit_start_and_end_edges(self) -> None:
        png = _page_png((40,))
        grouped = _group(png, policy="monostaff-v1")
        result = propose_measure_system_boundaries_v2(png, grouped)
        self.assertEqual(result.report.status, "accepted")
        assert result.page is not None
        self.assertEqual(len(result.page.measure_proposals), 1)
        logical = result.report.logical_measures
        self.assertEqual(len(logical), 1)
        self.assertEqual((logical[0].left_kind, logical[0].right_kind), ("system_edge", "system_edge"))
        self.assertEqual((logical[0].left_x, logical[0].right_x), (20.0, 301.0))

    def test_first_last_and_pickup_shape_are_system_local(self) -> None:
        png = _page_png((40,), verticals_by_staff=((60, 180),))
        grouped = _group(png, policy="monostaff-v1")
        result = propose_measure_system_boundaries_v2(png, grouped)
        self.assertEqual(result.report.status, "accepted")
        logical = result.report.logical_measures
        self.assertEqual(len(logical), 3)
        self.assertEqual(
            tuple((item.left_x, item.right_x) for item in logical),
            ((20.0, 60.0), (60.0, 180.0), (180.0, 301.0)),
        )
        self.assertEqual(logical[0].left_kind, "system_edge")
        self.assertEqual(logical[0].right_kind, "vertical_cluster")
        self.assertEqual(logical[-1].right_kind, "system_edge")

    def test_double_and_final_barline_strokes_are_geometric_clusters(self) -> None:
        png = _page_png((40,), verticals_by_staff=((150, 154, 296, 300),))
        grouped = _group(png, policy="monostaff-v1")
        result = propose_measure_system_boundaries_v2(png, grouped)
        self.assertEqual(result.report.status, "accepted")
        evidence = result.report.staff_evidence[0]
        self.assertEqual(evidence.raw_run_centers, (150.0, 154.0, 296.0, 300.0))
        self.assertEqual(evidence.clustered_centers, (152.0, 298.0))
        self.assertEqual(evidence.boundary_x, (20.0, 152.0, 301.0))
        self.assertEqual(len(result.report.logical_measures), 2)

    def test_missing_internal_barline_on_one_grand_staff_member_fails_closed(self) -> None:
        png = _page_png(
            (40, 150),
            verticals_by_staff=((160,), ()),
        )
        grouped = _group(png, policy="fixed-two-staff-v1")
        result = propose_measure_system_boundaries_v2(png, grouped)
        self.assertIsNone(result.page)
        self.assertEqual(result.report.status, "ambiguous")
        self.assertEqual(result.report.primary_reason, B05_CROSS_STAFF_BOUNDARY_MISMATCH)

    def test_cross_staff_barline_position_mismatch_fails_closed(self) -> None:
        png = _page_png(
            (40, 150),
            verticals_by_staff=((150,), (180,)),
        )
        grouped = _group(png, policy="fixed-two-staff-v1")
        result = propose_measure_system_boundaries_v2(png, grouped)
        self.assertIsNone(result.page)
        self.assertEqual(result.report.primary_reason, B05_CROSS_STAFF_BOUNDARY_MISMATCH)

    def test_multiple_staffs_share_one_logical_measure_identity(self) -> None:
        png = _page_png(
            (40, 150),
            verticals_by_staff=((160,), (160,)),
        )
        grouped = _group(png, policy="fixed-two-staff-v1")
        result = propose_measure_system_boundaries_v2(png, grouped)
        self.assertEqual(result.report.status, "accepted")
        assert result.page is not None
        self.assertEqual(len(result.report.logical_measures), 2)
        self.assertEqual(len(result.page.measure_proposals), 4)
        self.assertEqual(
            result.report.logical_measures[0].member_measure_ids,
            ("staff-1-measure-1", "staff-2-measure-1"),
        )
        self.assertEqual(
            {measure.system_id for measure in result.page.measure_proposals},
            {"system-1"},
        )

    def test_system_breaks_have_independent_measure_layouts(self) -> None:
        png = _page_png(
            (40, 180),
            verticals_by_staff=((100,), (210,)),
        )
        grouped = _group(png, policy="monostaff-v1")
        self.assertEqual(len(grouped.systems), 2)
        result = propose_measure_system_boundaries_v2(png, grouped)
        self.assertEqual(result.report.status, "accepted")
        logical_by_system = {
            system_id: tuple(
                (item.left_x, item.right_x)
                for item in result.report.logical_measures
                if item.system_id == system_id
            )
            for system_id in ("system-1", "system-2")
        }
        self.assertEqual(logical_by_system["system-1"], ((20.0, 100.0), (100.0, 301.0)))
        self.assertEqual(logical_by_system["system-2"], ((20.0, 210.0), (210.0, 301.0)))

    def test_page_edge_staff_extent_is_a_valid_system_edge(self) -> None:
        png = _page_png((40,), verticals_by_staff=((160,),), x_start=0)
        grouped = _group(png, policy="monostaff-v1")
        result = propose_measure_system_boundaries_v2(png, grouped)
        self.assertEqual(result.report.status, "accepted")
        self.assertEqual(result.report.logical_measures[0].left_x, 0.0)
        self.assertEqual(result.report.logical_measures[0].left_kind, "system_edge")

    def test_too_narrow_measure_is_not_hidden_by_edge_logic(self) -> None:
        png = _page_png((40,), verticals_by_staff=((31,),))
        grouped = _group(png, policy="monostaff-v1")
        result = propose_measure_system_boundaries_v2(png, grouped)
        self.assertIsNone(result.page)
        self.assertEqual(result.report.primary_reason, B06_MEASURE_TOO_NARROW)

    def test_exact_raster_identity_mismatch_is_not_downgraded_to_ambiguity(self) -> None:
        png = _page_png((40,), verticals_by_staff=((160,),))
        grouped = _group(png, policy="monostaff-v1")
        wrong = _page_png((40,), verticals_by_staff=((170,),))
        with self.assertRaises(MeasureSystemBoundariesV2Error):
            propose_measure_system_boundaries_v2(wrong, grouped)

    def test_existing_measure_geometry_cannot_be_overwritten(self) -> None:
        png = _page_png((40,), verticals_by_staff=((160,),))
        grouped = _group(png, policy="monostaff-v1")
        first = propose_measure_system_boundaries_v2(png, grouped)
        assert first.page is not None
        second = propose_measure_system_boundaries_v2(png, first.page)
        self.assertIsNone(second.page)
        self.assertEqual(second.report.primary_reason, B02_MEASURE_GEOMETRY_ALREADY_PRESENT)

    def test_invalid_membership_fails_before_raster_processing(self) -> None:
        png = _page_png((40, 180))
        grouped = _group(png, policy="monostaff-v1")
        bad_systems = (
            SystemGeometryContract(
                system_id="system-1",
                system_bbox=grouped.systems[0].system_bbox,
                staff_ids=("staff-1",),
            ),
            SystemGeometryContract(
                system_id="system-2",
                system_bbox=grouped.systems[1].system_bbox,
                staff_ids=("staff-1", "staff-2"),
            ),
        )
        bad = PageGeometryContract(
            normalized_image_sha256=grouped.normalized_image_sha256,
            geometry_config_fingerprint=grouped.geometry_config_fingerprint,
            page_width=grouped.page_width,
            page_height=grouped.page_height,
            transform=grouped.transform,
            systems=bad_systems,
            staffs=grouped.staffs,
            measure_proposals=(),
            status="accepted",
        )
        result = propose_measure_system_boundaries_v2(b"not-even-a-png", bad)
        self.assertIsNone(result.page)
        self.assertEqual(result.report.primary_reason, B03_SYSTEM_STAFF_MEMBERSHIP_INVALID)

    def test_noncanonical_system_order_fails_closed(self) -> None:
        png = _page_png((40, 180))
        grouped = _group(png, policy="monostaff-v1")
        bad = PageGeometryContract(
            normalized_image_sha256=grouped.normalized_image_sha256,
            geometry_config_fingerprint=grouped.geometry_config_fingerprint,
            page_width=grouped.page_width,
            page_height=grouped.page_height,
            transform=grouped.transform,
            systems=tuple(reversed(grouped.systems)),
            staffs=grouped.staffs,
            measure_proposals=(),
            status="accepted",
        )
        result = propose_measure_system_boundaries_v2(b"not-even-a-png", bad)
        self.assertIsNone(result.page)
        self.assertEqual(result.report.primary_reason, B04_SYSTEM_ORDER_INVALID)

    def test_detector_grouper_measure_chain_is_deterministic_10_of_10(self) -> None:
        png = _page_png(
            (40, 180),
            verticals_by_staff=((100, 200), (120, 220)),
        )
        detected = _detect(png)
        grouped = group_staffs_into_systems_v1(detected, policy="monostaff-v1")
        assert grouped.page is not None
        results = [propose_measure_system_boundaries_v2(png, grouped.page) for _ in range(10)]
        self.assertTrue(all(result.page is not None for result in results))
        self.assertEqual(len({result.report.fingerprint() for result in results}), 1)
        identities = {
            tuple(
                (
                    item.logical_measure_id,
                    item.system_id,
                    item.left_x,
                    item.right_x,
                    item.member_measure_ids,
                )
                for item in result.report.logical_measures
            )
            for result in results
        }
        self.assertEqual(len(identities), 1)


if __name__ == "__main__":
    unittest.main()
