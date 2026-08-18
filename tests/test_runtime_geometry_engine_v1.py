from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import inspect
import unittest

from PIL import Image, ImageDraw

from st_omr_training.runtime_geometry_engine_contract import GeometryInputContract
from st_omr_training.runtime_geometry_engine_v1 import (
    GeometryEngineV1Error,
    detect_single_staff_geometry_v1,
    geometry_engine_v1_config_fingerprint,
)
from st_omr_training.runtime_page_normalizer_contract import HomographyContract, RasterPageInputContract
from st_omr_training.runtime_page_normalizer_v1 import normalize_raster_page_v1


IDENTITY = HomographyContract(
    forward=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    inverse=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
)
SOURCE_SHA = sha256(b"geometry-v1-test-source").hexdigest()


def _png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _geometry_contract(data: bytes, image: Image.Image) -> GeometryInputContract:
    return GeometryInputContract(
        normalized_image_sha256=sha256(data).hexdigest(),
        normalizer_config_fingerprint="a" * 64,
        normalized_width=image.width,
        normalized_height=image.height,
        transform=IDENTITY,
    )


def _staff_image(line_count: int = 5, *, spacing: int = 10) -> Image.Image:
    image = Image.new("L", (240, 120), 255)
    draw = ImageDraw.Draw(image)
    for index in range(line_count):
        y = 30 + index * spacing
        draw.line((20, y, 220, y), fill=0, width=1)
    if line_count >= 3:
        draw.ellipse((104, 46, 116, 56), fill=0)
        draw.line((116, 30, 116, 51), fill=0, width=2)
    return image


def _two_staff_image() -> Image.Image:
    image = Image.new("L", (260, 220), 255)
    draw = ImageDraw.Draw(image)
    for base in (30, 130):
        for offset in (0, 10, 20, 30, 40):
            draw.line((20, base + offset, 240, base + offset), fill=0, width=1)
    return image


class RuntimeGeometryEngineV1Tests(unittest.TestCase):
    def test_single_five_line_staff_is_detected_without_music_semantics(self) -> None:
        image = _staff_image()
        data = _png(image)
        result = detect_single_staff_geometry_v1(data, _geometry_contract(data, image))

        self.assertEqual(result.page.status, "accepted")
        self.assertEqual(len(result.page.staffs), 1)
        self.assertEqual(len(result.page.systems), 1)
        self.assertEqual(result.page.measure_proposals, ())
        staff = result.page.staffs[0]
        centers = tuple((line.start.y + line.end.y) / 2 for line in staff.five_staff_lines)
        self.assertEqual(centers, (30.0, 40.0, 50.0, 60.0, 70.0))
        self.assertAlmostEqual(staff.staff_spacing, 10.0, places=9)

    def test_four_lines_fail_closed_instead_of_becoming_a_staff(self) -> None:
        image = _staff_image(line_count=4)
        data = _png(image)
        result = detect_single_staff_geometry_v1(data, _geometry_contract(data, image))

        self.assertEqual(result.page.status, "ambiguous")
        self.assertEqual(result.page.staffs, ())
        self.assertEqual(result.page.reasons, ("no-unambiguous-five-line-staff",))

    def test_two_staffs_are_ambiguous_in_this_first_slice(self) -> None:
        image = _two_staff_image()
        data = _png(image)
        result = detect_single_staff_geometry_v1(data, _geometry_contract(data, image))

        self.assertEqual(result.page.status, "ambiguous")
        self.assertEqual(result.page.staffs, ())
        self.assertEqual(
            result.page.reasons,
            ("multiple-or-overlapping-staff-candidates",),
        )

    def test_low_contrast_staff_survives_normalizer_then_geometry_detection(self) -> None:
        source = Image.new("RGB", (240, 120), (220, 220, 220))
        draw = ImageDraw.Draw(source)
        for y in (30, 40, 50, 60, 70):
            draw.line((20, y, 220, y), fill=(145, 145, 145), width=1)
        draw.ellipse((104, 46, 116, 56), fill=(135, 135, 135))
        source_bytes = _png(source)
        raster_contract = RasterPageInputContract(
            source_id="low-contrast-staff",
            source_sha256=SOURCE_SHA,
            page_number=1,
            width=source.width,
            height=source.height,
            pixel_mode="rgb8",
            raster_sha256=sha256(source_bytes).hexdigest(),
            dpi=300,
        )
        normalized = normalize_raster_page_v1(source_bytes, raster_contract)
        self.assertEqual(normalized.page.status, "accepted")
        assert normalized.normalized_png is not None
        assert normalized.page.transform is not None
        geometry_contract = GeometryInputContract(
            normalized_image_sha256=normalized.page.normalized_image_sha256 or "",
            normalizer_config_fingerprint=normalized.page.normalizer_config_fingerprint,
            normalized_width=normalized.page.normalized_width or 0,
            normalized_height=normalized.page.normalized_height or 0,
            transform=normalized.page.transform,
        )
        result = detect_single_staff_geometry_v1(normalized.normalized_png, geometry_contract)
        self.assertEqual(result.page.status, "accepted")
        self.assertAlmostEqual(result.page.staffs[0].staff_spacing, 10.0, places=9)

    def test_same_input_produces_identical_geometry(self) -> None:
        image = _staff_image()
        data = _png(image)
        contract = _geometry_contract(data, image)
        first = detect_single_staff_geometry_v1(data, contract)
        second = detect_single_staff_geometry_v1(data, contract)

        self.assertEqual(first, second)
        self.assertEqual(
            first.page.geometry_config_fingerprint,
            geometry_engine_v1_config_fingerprint(),
        )

    def test_wrong_bytes_or_non_gray_png_are_rejected(self) -> None:
        image = _staff_image()
        data = _png(image)
        wrong = bytearray(data)
        wrong[-1] ^= 1
        with self.assertRaises(GeometryEngineV1Error):
            detect_single_staff_geometry_v1(bytes(wrong), _geometry_contract(data, image))

        rgb = image.convert("RGB")
        rgb_data = _png(rgb)
        with self.assertRaises(GeometryEngineV1Error):
            detect_single_staff_geometry_v1(rgb_data, _geometry_contract(rgb_data, rgb))

    def test_v1_stays_isolated_from_d10_d13_models_and_training(self) -> None:
        import st_omr_training.runtime_geometry_engine_v1 as module

        source = inspect.getsource(module)
        for token in (
            "stage7d10_",
            "stage7d13_",
            "torch.optim",
            ".backward(",
            "DataLoader(",
            "torch.load(",
            "meter_class",
            "notehead_class",
            "rest_class",
            "accidental_class",
        ):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
