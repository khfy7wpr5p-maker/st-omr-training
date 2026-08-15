from __future__ import annotations

from collections.abc import Iterator, Mapping
from hashlib import sha256
from io import BytesIO
import json
import unittest

from PIL import Image

from st_omr_training.stage7d13_measure_derivatives import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    Stage7D13DerivativeError,
    _render_measure_image,
    _transform_box,
    _transform_point,
    development_d12_records,
    make_letterbox_plan,
    stage7d13_derivative_profile_fingerprint,
)


PNG_SHA = "a" * 64


class _HostileTestRow(Mapping[str, object]):
    """A TEST row that explodes if D13 touches anything beyond split."""

    def __getitem__(self, key: str) -> object:
        if key == "split":
            return "test"
        raise AssertionError(f"D13 touched forbidden TEST field: {key}")

    def __iter__(self) -> Iterator[str]:
        yield "split"
        yield "secret"

    def __len__(self) -> int:
        return 2


class Stage7D13MeasureDerivativeTests(unittest.TestCase):
    def test_test_row_is_rejected_after_split_only(self) -> None:
        with self.assertRaisesRegex(Stage7D13DerivativeError, "sealed TEST record"):
            development_d12_records([_HostileTestRow()])

    def test_letterbox_plan_is_deterministic_and_isotropic(self) -> None:
        box = {"x_min": 10.2, "y_min": 20.1, "x_max": 210.1, "y_max": 70.2}
        first = make_letterbox_plan(
            box,
            image_width=400,
            image_height=200,
            source_png_sha256=PNG_SHA,
        )
        second = make_letterbox_plan(
            box,
            image_width=400,
            image_height=200,
            source_png_sha256=PNG_SHA,
        )
        self.assertEqual(first, second)
        self.assertEqual((first.crop_left, first.crop_top), (10, 20))
        self.assertEqual((first.crop_right, first.crop_bottom), (211, 71))
        self.assertGreater(first.scale, 0.0)
        self.assertGreaterEqual(first.pad_x, 0.0)
        self.assertGreaterEqual(first.pad_y, 0.0)
        self.assertEqual(len(first.transform_fingerprint), 64)

    def test_geometry_uses_exact_same_scale_and_pad(self) -> None:
        measure = {"x_min": 10.0, "y_min": 20.0, "x_max": 210.0, "y_max": 70.0}
        plan = make_letterbox_plan(
            measure,
            image_width=400,
            image_height=200,
            source_png_sha256=PNG_SHA,
        )
        point = _transform_point({"x": 60.0, "y": 45.0}, plan)
        box = _transform_box(
            {"x_min": 50.0, "y_min": 40.0, "x_max": 70.0, "y_max": 50.0},
            plan,
        )
        expected_x = (60.0 - plan.crop_left) * plan.scale + plan.pad_x
        expected_y = (45.0 - plan.crop_top) * plan.scale + plan.pad_y
        self.assertAlmostEqual(point["x"], expected_x)
        self.assertAlmostEqual(point["y"], expected_y)
        self.assertLessEqual(box["x_min"], point["x"])
        self.assertGreaterEqual(box["x_max"], point["x"])
        self.assertLessEqual(box["y_min"], point["y"])
        self.assertGreaterEqual(box["y_max"], point["y"])

    def test_rendered_measure_is_fixed_grayscale_and_repeatable(self) -> None:
        source = Image.new("L", (300, 120), 0)
        measure = {"x_min": 20.0, "y_min": 30.0, "x_max": 220.0, "y_max": 70.0}
        plan = make_letterbox_plan(
            measure,
            image_width=300,
            image_height=120,
            source_png_sha256=PNG_SHA,
        )
        first = _render_measure_image(source, plan)
        second = _render_measure_image(source, plan)
        self.assertEqual(sha256(first).hexdigest(), sha256(second).hexdigest())
        with Image.open(BytesIO(first)) as opened:
            self.assertEqual(opened.format, "PNG")
            self.assertEqual(opened.mode, "L")
            self.assertEqual(opened.size, (INPUT_WIDTH, INPUT_HEIGHT))
            opened.load()
            self.assertEqual(opened.getpixel((0, 0)), 255)

    def test_derivative_profile_fingerprint_is_canonical_sha256(self) -> None:
        value = stage7d13_derivative_profile_fingerprint()
        self.assertEqual(len(value), 64)
        int(value, 16)
        self.assertEqual(value, stage7d13_derivative_profile_fingerprint())


if __name__ == "__main__":
    unittest.main()
