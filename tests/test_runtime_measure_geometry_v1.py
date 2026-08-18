from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import unittest

from PIL import Image, ImageDraw

from st_omr_training.runtime_geometry_engine_contract import GeometryInputContract
from st_omr_training.runtime_geometry_engine_v2 import detect_multistaff_geometry_v2
from st_omr_training.runtime_measure_geometry_v1 import (
    M01_INSUFFICIENT_BOUNDARIES,
    M03_CROSS_STAFF_BOUNDARY_MISMATCH,
    propose_measure_geometry_v1,
)
from st_omr_training.runtime_page_normalizer_contract import HomographyContract


IDENTITY = HomographyContract(
    forward=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    inverse=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
)
NORMALIZER_FP = sha256(b"measure-geometry-test-normalizer").hexdigest()


def _page(*, lower_middle_x: int = 160, include_boundaries: bool = True) -> bytes:
    image = Image.new("L", (320, 220), 255)
    draw = ImageDraw.Draw(image)
    for y in (30, 40, 50, 60, 70, 140, 150, 160, 170, 180):
        draw.line((20, y, 300, y), fill=0, width=1)
    if include_boundaries:
        for x in (20, 160, 300):
            draw.line((x, 30, x, 70), fill=0, width=1)
        for x in (20, lower_middle_x, 300):
            draw.line((x, 140, x, 180), fill=0, width=1)
    out = BytesIO()
    image.save(out, format="PNG", optimize=False, compress_level=9)
    return out.getvalue()


def _staff_geometry(png: bytes):
    contract = GeometryInputContract(
        normalized_image_sha256=sha256(png).hexdigest(),
        normalizer_config_fingerprint=NORMALIZER_FP,
        normalized_width=320,
        normalized_height=220,
        transform=IDENTITY,
    )
    result = detect_multistaff_geometry_v2(png, contract)
    if result.page.status != "accepted":
        raise AssertionError(result.page.reasons)
    return result.page


class RuntimeMeasureGeometryV1Tests(unittest.TestCase):
    def test_two_staffs_with_three_aligned_boundaries_produce_four_measure_proposals(self) -> None:
        png = _page()
        staff_geometry = _staff_geometry(png)
        results = [propose_measure_geometry_v1(png, staff_geometry) for _ in range(10)]
        for result in results:
            self.assertEqual(result.page.status, "accepted")
            self.assertEqual(len(result.page.measure_proposals), 4)
            self.assertEqual(
                result.boundary_x_by_staff,
                (
                    ("staff-1", (20.0, 160.0, 300.0)),
                    ("staff-2", (20.0, 160.0, 300.0)),
                ),
            )
            ids = tuple(measure.measure_id for measure in result.page.measure_proposals)
            self.assertEqual(
                ids,
                (
                    "staff-1-measure-1",
                    "staff-1-measure-2",
                    "staff-2-measure-1",
                    "staff-2-measure-2",
                ),
            )
            for measure in result.page.measure_proposals:
                self.assertEqual(measure.status, "accepted")
                self.assertLess(measure.bbox.x_min, measure.bbox.x_max)
        identities = [
            tuple((m.measure_id, m.bbox) for m in result.page.measure_proposals)
            for result in results
        ]
        self.assertEqual(len(set(identities)), 1)

    def test_missing_vertical_boundaries_fails_closed(self) -> None:
        png = _page(include_boundaries=False)
        staff_geometry = _staff_geometry(png)
        result = propose_measure_geometry_v1(png, staff_geometry)
        self.assertEqual(result.page.status, "ambiguous")
        self.assertEqual(result.page.reasons, (M01_INSUFFICIENT_BOUNDARIES,))
        self.assertEqual(result.page.measure_proposals, ())

    def test_cross_staff_boundary_mismatch_fails_closed(self) -> None:
        png = _page(lower_middle_x=180)
        staff_geometry = _staff_geometry(png)
        result = propose_measure_geometry_v1(png, staff_geometry)
        self.assertEqual(result.page.status, "ambiguous")
        self.assertEqual(result.page.reasons, (M03_CROSS_STAFF_BOUNDARY_MISMATCH,))
        self.assertEqual(result.page.measure_proposals, ())


if __name__ == "__main__":
    unittest.main()
