from __future__ import annotations

import unittest

from st_omr_training.runtime_deterministic_resolver_v1 import resolve_specialist_evidence_v1
from st_omr_training.runtime_geometry_engine_contract import (
    BoxContract,
    LineSegmentContract,
    MeasureProposalContract,
    PageGeometryContract,
    Point2DContract,
    StaffGeometryContract,
    SystemGeometryContract,
)
from st_omr_training.runtime_page_normalizer_contract import HomographyContract
from st_omr_training.runtime_specialist_evidence_v1 import SpecialistEvidenceBatch


def _identity() -> HomographyContract:
    return HomographyContract(
        forward=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        inverse=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )


def _staff(staff_id: str, system_id: str, top: float) -> StaffGeometryContract:
    lines = tuple(
        LineSegmentContract(Point2DContract(10.0, top + index * 2.0), Point2DContract(190.0, top + index * 2.0))
        for index in range(5)
    )
    return StaffGeometryContract(staff_id, system_id, lines, BoxContract(10.0, top, 190.0, top + 8.0), 2.0)


def _measure(staff_id: str, system_id: str, top: float, *, status: str = "accepted") -> MeasureProposalContract:
    bbox = BoxContract(10.0, top, 190.0, top + 8.0)
    return MeasureProposalContract(
        measure_id=f"{staff_id}-m1",
        system_id=system_id,
        staff_id=staff_id,
        bbox=bbox,
        left_boundary=LineSegmentContract(Point2DContract(10.0, top), Point2DContract(10.0, top + 8.0)),
        right_boundary=LineSegmentContract(Point2DContract(190.0, top), Point2DContract(190.0, top + 8.0)),
        status=status,
        reasons=() if status == "accepted" else ("fixture-nonaccepted",),
    )


def _page(*, swapped_system_membership: bool = False, measure_status: str = "accepted") -> PageGeometryContract:
    staff1 = _staff("staff-1", "system-1", 20.0)
    staff2 = _staff("staff-2", "system-2", 60.0)
    if swapped_system_membership:
        systems = (
            SystemGeometryContract("system-1", staff1.staff_bbox, ("staff-2",)),
            SystemGeometryContract("system-2", staff2.staff_bbox, ("staff-1",)),
        )
    else:
        systems = (
            SystemGeometryContract("system-1", staff1.staff_bbox, ("staff-1",)),
            SystemGeometryContract("system-2", staff2.staff_bbox, ("staff-2",)),
        )
    return PageGeometryContract(
        normalized_image_sha256="a" * 64,
        geometry_config_fingerprint="b" * 64,
        page_width=200,
        page_height=120,
        transform=_identity(),
        systems=systems,
        staffs=(staff1, staff2),
        measure_proposals=(
            _measure("staff-1", "system-1", 20.0, status=measure_status),
            _measure("staff-2", "system-2", 60.0),
        ),
        status="accepted",
    )


class ResolverMembershipHardeningTests(unittest.TestCase):
    def test_crossed_system_staff_membership_fails_closed(self) -> None:
        geometry = _page(swapped_system_membership=True)
        with self.assertRaises(ValueError):
            resolve_specialist_evidence_v1(geometry, SpecialistEvidenceBatch(()))

    def test_nonaccepted_measure_on_accepted_page_fails_closed(self) -> None:
        geometry = _page(measure_status="ambiguous")
        with self.assertRaises(ValueError):
            resolve_specialist_evidence_v1(geometry, SpecialistEvidenceBatch(()))


if __name__ == "__main__":
    unittest.main()
