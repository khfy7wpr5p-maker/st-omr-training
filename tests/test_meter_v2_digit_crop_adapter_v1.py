from __future__ import annotations

import inspect
import math
import unittest

from PIL import Image

import st_omr_training.meter_v2_digit_crop_adapter_v1 as crop_adapter
from st_omr_training.meter_v2_digit_crop_adapter_v1 import (
    METER_V2_DIGIT_CROP_SIZE,
    MeterV2DigitCropError,
    crop_meter_digit_to_64_v1,
    meter_v2_digit_pixel_box_v1,
    meter_v2_digit_crop_profile_fingerprint,
    resolver_connection_allowed,
    runtime_digit_bbox_localization_frozen,
)


class MeterV2DigitCropAdapterV1Tests(unittest.TestCase):
    @staticmethod
    def _historical_reference(image: Image.Image, box) -> Image.Image:
        # Exact valid-input transform recovered from the M4 training worker.
        import math as _math

        w, h = image.size
        x0, y0, x1, y1 = map(float, box)
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.5:
            x0 *= w
            x1 *= w
            y0 *= h
            y1 *= h
        x0 = max(0, min(w - 1, int(_math.floor(x0))))
        y0 = max(0, min(h - 1, int(_math.floor(y0))))
        x1 = max(x0 + 1, min(w, int(_math.ceil(x1))))
        y1 = max(y0 + 1, min(h, int(_math.ceil(y1))))
        crop = image.crop((x0, y0, x1, y1)).convert("L")
        crop.thumbnail((64, 64), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (64, 64), color=255)
        ox = (64 - crop.width) // 2
        oy = (64 - crop.height) // 2
        canvas.paste(crop, (ox, oy))
        return canvas

    @staticmethod
    def _source_image() -> Image.Image:
        image = Image.new("L", (256, 192), 255)
        image.putdata(
            [
                (x * 7 + y * 11) % 256
                for y in range(192)
                for x in range(256)
            ]
        )
        return image

    def test_pixel_box_uses_floor_ceil_and_clipping(self) -> None:
        image = self._source_image()
        self.assertEqual(
            meter_v2_digit_pixel_box_v1(image, (10.9, 20.1, 40.01, 60.99)),
            (10, 20, 41, 61),
        )
        self.assertEqual(
            meter_v2_digit_pixel_box_v1(image, (-5.2, -4.8, 400.0, 300.0)),
            (0, 0, 256, 192),
        )

    def test_normalized_and_pixel_boxes_are_equivalent(self) -> None:
        image = self._source_image()
        pixel_box = (64.0, 48.0, 128.0, 96.0)
        normalized_box = (0.25, 0.25, 0.5, 0.5)
        self.assertEqual(
            crop_meter_digit_to_64_v1(image, pixel_box).tobytes(),
            crop_meter_digit_to_64_v1(image, normalized_box).tobytes(),
        )

    def test_matches_historical_training_transform_exactly(self) -> None:
        image = self._source_image()
        boxes = (
            (100.7797291145, 53.1777843972, 128.1071639724, 87.6255772285),
            (0.12, 0.21, 0.31, 0.49),
            (-3.2, 15.4, 32.2, 81.9),
        )
        for box in boxes:
            with self.subTest(box=box):
                actual = crop_meter_digit_to_64_v1(image, box)
                expected = self._historical_reference(image, box)
                self.assertEqual(actual.mode, "L")
                self.assertEqual(actual.size, (64, 64))
                self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_thumbnail_preserves_aspect_and_never_upscales(self) -> None:
        image = Image.new("L", (100, 100), 255)
        for y in range(20, 40):
            for x in range(30, 40):
                image.putpixel((x, y), 0)
        crop = crop_meter_digit_to_64_v1(image, (30, 20, 40, 40))
        dark_points = [
            (x, y)
            for y in range(64)
            for x in range(64)
            if crop.getpixel((x, y)) < 128
        ]
        self.assertTrue(dark_points)
        xs = [point[0] for point in dark_points]
        ys = [point[1] for point in dark_points]
        self.assertLessEqual(max(xs) - min(xs) + 1, 10)
        self.assertLessEqual(max(ys) - min(ys) + 1, 20)

    def test_output_is_10_of_10_deterministic(self) -> None:
        image = self._source_image()
        box = (34.572, 58.535304, 63.167982, 96.364121)
        reports = [crop_meter_digit_to_64_v1(image, box).tobytes() for _ in range(10)]
        self.assertEqual(len(set(reports)), 1)

    def test_malformed_boxes_fail_closed(self) -> None:
        image = self._source_image()
        bad_boxes = (
            (1, 2, 3),
            (10, 10, 10, 20),
            (10, 10, 20, 10),
            (0, 0, math.inf, 2),
            (0, 0, math.nan, 2),
            (True, 0, 1, 2),
        )
        for box in bad_boxes:
            with self.subTest(box=box):
                with self.assertRaises(MeterV2DigitCropError):
                    crop_meter_digit_to_64_v1(image, box)

    def test_profile_and_isolation_are_frozen(self) -> None:
        self.assertEqual(METER_V2_DIGIT_CROP_SIZE, 64)
        self.assertEqual(len(meter_v2_digit_crop_profile_fingerprint()), 64)
        self.assertFalse(runtime_digit_bbox_localization_frozen())
        self.assertFalse(resolver_connection_allowed())
        source = inspect.getsource(crop_adapter)
        self.assertNotIn("import torch", source)
        self.assertNotIn("import numpy", source)
        self.assertNotIn("runtime_deterministic_resolver", source)


if __name__ == "__main__":
    unittest.main()
