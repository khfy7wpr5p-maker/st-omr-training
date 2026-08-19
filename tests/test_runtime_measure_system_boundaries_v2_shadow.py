from __future__ import annotations

import unittest

from st_omr_training.runtime_geometry_engine_contract import (
    BoxContract,
    PageGeometryContract,
    SystemGeometryContract,
)
from st_omr_training.runtime_measure_system_boundaries_v2 import (
    B03_SYSTEM_STAFF_MEMBERSHIP_INVALID,
    propose_measure_system_boundaries_v2,
)
from test_runtime_measure_system_boundaries_v2 import _group, _page_png


class MeasureSystemBoundariesV2ShadowRegressions(unittest.TestCase):
    """Try to falsify the assumption that any ID-consistent grouping is trusted."""

    def test_reversed_member_order_is_not_accepted_as_grouped_predecessor_truth(self) -> None:
        png = _page_png((40, 150), verticals_by_staff=((160,), (160,)))
        grouped = _group(png, policy="fixed-two-staff-v1")
        original = grouped.systems[0]
        corrupted_system = SystemGeometryContract(
            system_id=original.system_id,
            system_bbox=original.system_bbox,
            staff_ids=tuple(reversed(original.staff_ids)),
        )
        corrupted = PageGeometryContract(
            normalized_image_sha256=grouped.normalized_image_sha256,
            geometry_config_fingerprint=grouped.geometry_config_fingerprint,
            page_width=grouped.page_width,
            page_height=grouped.page_height,
            transform=grouped.transform,
            systems=(corrupted_system,),
            staffs=grouped.staffs,
            measure_proposals=(),
            status="accepted",
        )
        result = propose_measure_system_boundaries_v2(b"invalid-is-never-read", corrupted)
        self.assertIsNone(result.page)
        self.assertEqual(result.report.primary_reason, B03_SYSTEM_STAFF_MEMBERSHIP_INVALID)

    def test_system_bbox_must_equal_exact_union_of_member_staff_boxes(self) -> None:
        png = _page_png((40, 150), verticals_by_staff=((160,), (160,)))
        grouped = _group(png, policy="fixed-two-staff-v1")
        original = grouped.systems[0]
        box = original.system_bbox
        corrupted_system = SystemGeometryContract(
            system_id=original.system_id,
            system_bbox=BoxContract(box.x_min, box.y_min, box.x_max - 1.0, box.y_max),
            staff_ids=original.staff_ids,
        )
        corrupted = PageGeometryContract(
            normalized_image_sha256=grouped.normalized_image_sha256,
            geometry_config_fingerprint=grouped.geometry_config_fingerprint,
            page_width=grouped.page_width,
            page_height=grouped.page_height,
            transform=grouped.transform,
            systems=(corrupted_system,),
            staffs=grouped.staffs,
            measure_proposals=(),
            status="accepted",
        )
        result = propose_measure_system_boundaries_v2(b"invalid-is-never-read", corrupted)
        self.assertIsNone(result.page)
        self.assertEqual(result.report.primary_reason, B03_SYSTEM_STAFF_MEMBERSHIP_INVALID)


if __name__ == "__main__":
    unittest.main()
