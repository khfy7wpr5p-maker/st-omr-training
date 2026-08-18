from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import unittest

from PIL import Image, ImageDraw

from st_omr_training.runtime_geometry_ambiguity_v1 import (
    A01_INCOMPLETE_STAFF,
    A02_STAFFS_TOO_CLOSE,
    A03_LOW_VISIBILITY,
    A04_PAGE_CROPPED,
    A05_OVERLAPPING_CANDIDATES,
    A06_IRREGULAR_SPACING,
    A07_EXTRA_LINE_CANDIDATES,
    AMBIGUITY_PRIORITY,
    build_geometry_ambiguity_report,
)
from st_omr_training.runtime_geometry_engine_contract import GeometryInputContract
from st_omr_training.runtime_geometry_engine_v2 import detect_multistaff_geometry_v2
from st_omr_training.runtime_page_normalizer_contract import HomographyContract


IDENTITY = HomographyContract(
    forward=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    inverse=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
)
NORMALIZER_FP = sha256(b"geometry-v2-test-normalizer").hexdigest()


def _png_with_lines(lines: tuple[float, ...], *, height: int = 240) -> bytes:
    image = Image.new("L", (320, height), 255)
    draw = ImageDraw.Draw(image)
    for y in lines:
        draw.line((20, int(y), 300, int(y)), fill=0, width=1)
    out = BytesIO()
    image.save(out, format="PNG", optimize=False, compress_level=9)
    return out.getvalue()


def _input(png: bytes, height: int = 240) -> GeometryInputContract:
    return GeometryInputContract(
        normalized_image_sha256=sha256(png).hexdigest(),
        normalizer_config_fingerprint=NORMALIZER_FP,
        normalized_width=320,
        normalized_height=height,
        transform=IDENTITY,
    )


class GeometryAmbiguityResolverTests(unittest.TestCase):
    def test_priority_is_frozen_and_primary_secondary_are_canonical(self) -> None:
        self.assertEqual(
            AMBIGUITY_PRIORITY,
            (
                A04_PAGE_CROPPED,
                A03_LOW_VISIBILITY,
                A01_INCOMPLETE_STAFF,
                A05_OVERLAPPING_CANDIDATES,
                A02_STAFFS_TOO_CLOSE,
                A07_EXTRA_LINE_CANDIDATES,
                A06_IRREGULAR_SPACING,
            ),
        )
        report = build_geometry_ambiguity_report(
            (A06_IRREGULAR_SPACING, A01_INCOMPLETE_STAFF, A03_LOW_VISIBILITY)
        )
        self.assertEqual(report.primary_reason, A03_LOW_VISIBILITY)
        self.assertEqual(
            report.secondary_reasons,
            (A01_INCOMPLETE_STAFF, A06_IRREGULAR_SPACING),
        )
        self.assertEqual(
            report.active_reasons,
            (A03_LOW_VISIBILITY, A01_INCOMPLETE_STAFF, A06_IRREGULAR_SPACING),
        )

    def test_codes_are_unique_and_unknown_codes_fail_closed(self) -> None:
        report = build_geometry_ambiguity_report(
            (A03_LOW_VISIBILITY, A03_LOW_VISIBILITY, A01_INCOMPLETE_STAFF)
        )
        self.assertEqual(report.active_reasons, (A03_LOW_VISIBILITY, A01_INCOMPLETE_STAFF))
        with self.assertRaises(ValueError):
            build_geometry_ambiguity_report(("A99_UNKNOWN",))

    def test_same_active_reasons_produce_identical_report_10_of_10(self) -> None:
        reports = [
            build_geometry_ambiguity_report(
                (A06_IRREGULAR_SPACING, A03_LOW_VISIBILITY, A01_INCOMPLETE_STAFF)
            )
            for _ in range(10)
        ]
        payloads = [report.canonical_payload() for report in reports]
        fingerprints = [report.fingerprint() for report in reports]
        self.assertEqual(len(set(str(payload) for payload in payloads)), 1)
        self.assertEqual(len(set(fingerprints)), 1)


class MultiStaffGeometryV2Tests(unittest.TestCase):
    def test_two_clear_staffs_are_two_separate_stable_outputs(self) -> None:
        png = _png_with_lines((30, 40, 50, 60, 70, 140, 150, 160, 170, 180))
        results = [detect_multistaff_geometry_v2(png, _input(png)) for _ in range(10)]
        for result in results:
            self.assertEqual(result.page.status, "accepted")
            self.assertEqual(len(result.page.staffs), 2)
            first, second = result.page.staffs
            self.assertEqual(first.staff_id, "staff-1")
            self.assertEqual(second.staff_id, "staff-2")
            self.assertEqual(len(first.five_staff_lines), 5)
            self.assertEqual(len(second.five_staff_lines), 5)
            self.assertLess(first.staff_bbox.y_max, second.staff_bbox.y_min)
            first_centers = tuple(line.start.y for line in first.five_staff_lines)
            second_centers = tuple(line.start.y for line in second.five_staff_lines)
            self.assertEqual(first_centers, (30.0, 40.0, 50.0, 60.0, 70.0))
            self.assertEqual(second_centers, (140.0, 150.0, 160.0, 170.0, 180.0))
        identities = [
            tuple((staff.staff_id, staff.staff_bbox, staff.staff_spacing) for staff in result.page.staffs)
            for result in results
        ]
        self.assertEqual(len(set(identities)), 1)

    def test_a01_incomplete_staff(self) -> None:
        png = _png_with_lines((30, 40, 50, 60))
        result = detect_multistaff_geometry_v2(png, _input(png))
        self.assertEqual(result.page.status, "ambiguous")
        self.assertEqual(result.ambiguity_report.primary_reason, A01_INCOMPLETE_STAFF)

    def test_a03_low_visibility_when_no_staff_lines_are_visible(self) -> None:
        png = _png_with_lines(())
        result = detect_multistaff_geometry_v2(png, _input(png))
        self.assertEqual(result.ambiguity_report.primary_reason, A03_LOW_VISIBILITY)

    def test_a04_page_cropped_has_highest_priority(self) -> None:
        png = _png_with_lines((3, 13, 23, 33, 43), height=100)
        result = detect_multistaff_geometry_v2(png, _input(png, height=100))
        self.assertEqual(result.page.status, "ambiguous")
        self.assertEqual(result.ambiguity_report.primary_reason, A04_PAGE_CROPPED)

    def test_a05_overlapping_candidates_fails_closed(self) -> None:
        png = _png_with_lines((30, 40, 50, 60, 70, 80))
        result = detect_multistaff_geometry_v2(png, _input(png))
        self.assertEqual(result.ambiguity_report.primary_reason, A05_OVERLAPPING_CANDIDATES)
        self.assertIn(A07_EXTRA_LINE_CANDIDATES, result.ambiguity_report.secondary_reasons)

    def test_a02_staffs_too_close(self) -> None:
        png = _png_with_lines((30, 40, 50, 60, 70, 90, 100, 110, 120, 130))
        result = detect_multistaff_geometry_v2(png, _input(png))
        self.assertEqual(result.ambiguity_report.primary_reason, A02_STAFFS_TOO_CLOSE)

    def test_a07_extra_line_candidates(self) -> None:
        png = _png_with_lines((30, 40, 50, 60, 70, 105, 140, 150, 160, 170, 180))
        result = detect_multistaff_geometry_v2(png, _input(png))
        self.assertEqual(result.ambiguity_report.primary_reason, A07_EXTRA_LINE_CANDIDATES)

    def test_a06_irregular_spacing(self) -> None:
        png = _png_with_lines((30, 40, 50, 78, 90))
        result = detect_multistaff_geometry_v2(png, _input(png))
        self.assertEqual(result.ambiguity_report.primary_reason, A06_IRREGULAR_SPACING)


if __name__ == "__main__":
    unittest.main()
