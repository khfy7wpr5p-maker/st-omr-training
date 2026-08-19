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


def _page() -> PageGeometryContract:
    systems = []
    staffs = []
    measures = []
    for index in range(1, 11):
        system_id = f"system-{index}"
        staff_id = f"staff-{index}"
        measure_id = f"measure-{index}"
        top = float(10 + index * 20)
        lines = tuple(
            LineSegmentContract(Point2DContract(10.0, top + offset * 2.0), Point2DContract(190.0, top + offset * 2.0))
            for offset in range(5)
        )
        bbox = BoxContract(10.0, top, 190.0, top + 8.0)
        staffs.append(StaffGeometryContract(staff_id, system_id, lines, bbox, 2.0))
        systems.append(SystemGeometryContract(system_id, bbox, (staff_id,)))
        measures.append(
            MeasureProposalContract(
                measure_id=measure_id,
                system_id=system_id,
                staff_id=staff_id,
                bbox=bbox,
                left_boundary=LineSegmentContract(Point2DContract(10.0, top), Point2DContract(10.0, top + 8.0)),
                right_boundary=LineSegmentContract(Point2DContract(190.0, top), Point2DContract(190.0, top + 8.0)),
                status="accepted",
            )
        )
    # Deliberately reverse proposal tuple: resolver must recover system geometry order,
    # not preserve arbitrary proposal order or sort identifier strings lexically.
    return PageGeometryContract(
        normalized_image_sha256="a" * 64,
        geometry_config_fingerprint="b" * 64,
        page_width=200,
        page_height=240,
        transform=_identity(),
        systems=tuple(systems),
        staffs=tuple(staffs),
        measure_proposals=tuple(reversed(measures)),
        status="accepted",
    )


class ResolverGeometryOrderTests(unittest.TestCase):
    def test_system_10_stays_after_system_9(self) -> None:
        result = resolve_specialist_evidence_v1(_page(), SpecialistEvidenceBatch(()))
        self.assertEqual(tuple(item.measure_id for item in result.measures), tuple(f"measure-{index}" for index in range(1, 11)))


if __name__ == "__main__":
    unittest.main()
