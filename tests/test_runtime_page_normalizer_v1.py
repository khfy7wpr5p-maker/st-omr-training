from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import inspect
import unittest

from PIL import Image, ImageDraw

from st_omr_training.runtime_page_normalizer_contract import RasterPageInputContract
from st_omr_training.runtime_page_normalizer_v1 import (
    PageNormalizerV1Error,
    normalize_raster_page_v1,
    page_normalizer_v1_config_fingerprint,
)


SOURCE_SHA = sha256(b"runtime-normalizer-test-source").hexdigest()


def _encode_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _encode_jpeg_with_orientation(image: Image.Image, orientation: int) -> bytes:
    exif = Image.Exif()
    exif[274] = orientation
    output = BytesIO()
    image.save(output, format="JPEG", quality=95, exif=exif)
    return output.getvalue()


def _contract(data: bytes, image: Image.Image, pixel_mode: str) -> RasterPageInputContract:
    return RasterPageInputContract(
        source_id="fixture-page-1",
        source_sha256=SOURCE_SHA,
        page_number=1,
        width=image.width,
        height=image.height,
        pixel_mode=pixel_mode,
        raster_sha256=sha256(data).hexdigest(),
        dpi=300,
    )


def _low_contrast_staff_rgb() -> Image.Image:
    image = Image.new("RGB", (240, 120), (220, 220, 220))
    draw = ImageDraw.Draw(image)
    for y in (30, 40, 50, 60, 70):
        draw.line((20, y, 220, y), fill=(155, 155, 155), width=1)
    draw.ellipse((102, 46, 116, 56), fill=(140, 140, 140))
    draw.line((116, 30, 116, 51), fill=(145, 145, 145), width=2)
    return image


class RuntimePageNormalizerV1Tests(unittest.TestCase):
    def test_low_contrast_staff_is_preserved_and_made_readable(self) -> None:
        image = _low_contrast_staff_rgb()
        data = _encode_png(image)
        result = normalize_raster_page_v1(data, _contract(data, image, "rgb8"))

        self.assertEqual(result.page.status, "accepted")
        self.assertIsNotNone(result.normalized_png)
        self.assertEqual(result.page.normalized_width, 240)
        self.assertEqual(result.page.normalized_height, 120)
        operation_ids = tuple(item.operation_id for item in result.page.operations)
        self.assertEqual(
            operation_ids,
            ("grayscale_conversion", "contrast_normalization"),
        )

        assert result.normalized_png is not None
        with Image.open(BytesIO(result.normalized_png)) as normalized:
            self.assertEqual(normalized.mode, "L")
            self.assertEqual(normalized.size, (240, 120))
            background = normalized.getpixel((5, 5))
            staff_pixel = normalized.getpixel((50, 30))
            notehead_pixel = normalized.getpixel((108, 51))
            self.assertGreater(background, staff_pixel)
            self.assertGreater(background, notehead_pixel)
            for y in (30, 40, 50, 60, 70):
                dark_pixels = sum(
                    1 for x in range(20, 221) if normalized.getpixel((x, y)) < 128
                )
                self.assertGreaterEqual(dark_pixels, 180)

    def test_phone_orientation_metadata_is_applied_with_replayable_transform(self) -> None:
        image = Image.new("RGB", (40, 60), "white")
        draw = ImageDraw.Draw(image)
        draw.line((5, 10, 35, 10), fill="black", width=2)
        data = _encode_jpeg_with_orientation(image, 6)
        result = normalize_raster_page_v1(data, _contract(data, image, "rgb8"))

        self.assertEqual(result.page.status, "accepted")
        self.assertEqual(result.exif_orientation, 6)
        self.assertEqual(result.page.normalized_width, 60)
        self.assertEqual(result.page.normalized_height, 40)
        self.assertIn("orientation", tuple(item.operation_id for item in result.page.operations))
        assert result.page.transform is not None
        mapped = result.page.transform.original_to_normalized(10.0, 20.0)
        replayed = result.page.transform.normalized_to_original(*mapped)
        self.assertAlmostEqual(replayed[0], 10.0, places=9)
        self.assertAlmostEqual(replayed[1], 20.0, places=9)

    def test_mirrored_orientation_fails_closed_instead_of_guessing(self) -> None:
        image = Image.new("RGB", (40, 60), "white")
        data = _encode_jpeg_with_orientation(image, 2)
        result = normalize_raster_page_v1(data, _contract(data, image, "rgb8"))

        self.assertEqual(result.page.status, "rejected")
        self.assertIsNone(result.normalized_png)
        self.assertEqual(
            result.page.rejection_reasons,
            ("mirrored-exif-orientation-not-supported-v1",),
        )

    def test_same_input_produces_identical_normalized_bytes_and_identity(self) -> None:
        image = _low_contrast_staff_rgb()
        data = _encode_png(image)
        contract = _contract(data, image, "rgb8")
        first = normalize_raster_page_v1(data, contract)
        second = normalize_raster_page_v1(data, contract)

        self.assertEqual(first.normalized_png, second.normalized_png)
        self.assertEqual(
            first.page.normalized_image_sha256,
            second.page.normalized_image_sha256,
        )
        self.assertEqual(
            first.page.normalizer_config_fingerprint,
            page_normalizer_v1_config_fingerprint(),
        )

    def test_byte_identity_and_declared_raster_shape_are_enforced(self) -> None:
        image = _low_contrast_staff_rgb()
        data = _encode_png(image)
        forged = RasterPageInputContract(
            source_id="fixture-page-1",
            source_sha256=SOURCE_SHA,
            page_number=1,
            width=image.width,
            height=image.height,
            pixel_mode="rgb8",
            raster_sha256="0" * 64,
            dpi=300,
        )
        with self.assertRaises(PageNormalizerV1Error):
            normalize_raster_page_v1(data, forged)

        wrong_size = RasterPageInputContract(
            source_id="fixture-page-1",
            source_sha256=SOURCE_SHA,
            page_number=1,
            width=image.width + 1,
            height=image.height,
            pixel_mode="rgb8",
            raster_sha256=sha256(data).hexdigest(),
            dpi=300,
        )
        with self.assertRaises(PageNormalizerV1Error):
            normalize_raster_page_v1(data, wrong_size)

    def test_transparent_page_is_composited_on_white_before_gray_conversion(self) -> None:
        image = Image.new("RGBA", (80, 40), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.line((10, 20, 70, 20), fill=(0, 0, 0, 255), width=2)
        data = _encode_png(image)
        result = normalize_raster_page_v1(data, _contract(data, image, "rgba8"))

        assert result.normalized_png is not None
        with Image.open(BytesIO(result.normalized_png)) as normalized:
            self.assertGreater(normalized.getpixel((2, 2)), normalized.getpixel((40, 20)))

    def test_v1_implementation_stays_isolated_from_training_and_stage7_derivatives(self) -> None:
        import st_omr_training.runtime_page_normalizer_v1 as module

        source = inspect.getsource(module)
        forbidden_tokens = (
            "stage7d10_",
            "stage7d13_",
            "torch.optim",
            ".backward(",
            "DataLoader(",
            "torch.load(",
        )
        for token in forbidden_tokens:
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
