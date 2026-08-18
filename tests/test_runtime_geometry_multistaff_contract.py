from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import unittest

from PIL import Image, ImageDraw

from st_omr_training.runtime_geometry_engine_contract import GeometryInputContract
from st_omr_training.runtime_geometry_engine_v1 import detect_single_staff_geometry_v1
from st_omr_training.runtime_page_normalizer_contract import HomographyContract


IDENTITY = HomographyContract(
    forward=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    inverse=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
)
NORMALIZER_FINGERPRINT = sha256(b"multistaff-test-normalizer").hexdigest()


def _two_clear_staffs_png() -> bytes:
    """Return a small deterministic page containing two clearly separated staffs."""

    image = Image.new("L", (320, 220), 255)
    draw = ImageDraw.Draw(image)

    for y in (30, 40, 50, 60, 70):
        draw.line((20, y, 300, y), fill=0, width=1)

    for y in (140, 150, 160, 170, 180):
        draw.line((20, y, 300, y), fill=0, width=1)

    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _geometry_input(png: bytes) -> GeometryInputContract:
    return GeometryInputContract(
        normalized_image_sha256=sha256(png).hexdigest(),
        normalizer_config_fingerprint=NORMALIZER_FINGERPRINT,
        normalized_width=320,
        normalized_height=220,
        transform=IDENTITY,
    )


class MultiStaffSafetyGateTests(unittest.TestCase):
    def test_current_single_staff_v1_does_not_merge_two_staffs_into_one(self) -> None:
        """Until multi-staff support exists, two staffs must fail closed, not be merged."""

        png = _two_clear_staffs_png()
        result = detect_single_staff_geometry_v1(png, _geometry_input(png))

        self.assertEqual(result.page.status, "ambiguous")
        self.assertEqual(result.page.staffs, ())
        self.assertIn(
            "multiple-or-overlapping-staff-candidates",
            result.page.reasons,
        )
        self.assertEqual(len(result.candidate_row_centers), 10)


class FutureMultiStaffSeparationGateTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_two_clear_staffs_are_returned_as_two_separate_staffs(self) -> None:
        """Frozen future gate: two obvious staffs must become two independent outputs.

        This is intentionally an expected failure while Geometry Engine V1 remains
        single-staff-only. Removing expectedFailure is allowed only when a future
        isolated multi-staff implementation can satisfy the assertions without
        changing D10/D13, Rest R2, or specialist behavior.
        """

        png = _two_clear_staffs_png()
        result = detect_single_staff_geometry_v1(png, _geometry_input(png))

        self.assertEqual(result.page.status, "accepted")
        self.assertEqual(len(result.page.staffs), 2)

        first, second = result.page.staffs
        self.assertEqual(len(first.five_staff_lines), 5)
        self.assertEqual(len(second.five_staff_lines), 5)
        self.assertLess(first.staff_bbox.y_max, second.staff_bbox.y_min)

        first_centers = tuple(
            (line.start.y + line.end.y) / 2.0 for line in first.five_staff_lines
        )
        second_centers = tuple(
            (line.start.y + line.end.y) / 2.0 for line in second.five_staff_lines
        )
        self.assertEqual(first_centers, (30.0, 40.0, 50.0, 60.0, 70.0))
        self.assertEqual(second_centers, (140.0, 150.0, 160.0, 170.0, 180.0))


if __name__ == "__main__":
    unittest.main()
