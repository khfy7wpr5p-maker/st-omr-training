from __future__ import annotations

import inspect
import math
import unittest

from PIL import Image, ImageDraw

from st_omr_training.m4_e3k_boundary_proposals import FROZEN_E3K_CONFIG
from st_omr_training.m4_e3k_r2_inward_endpoint_geometry import (
    STAGE,
    _column_evidence_inward,
    propose_measure_boundaries_r2,
)
from st_omr_training.m4_e3k_boundary_proposals import _staff_lines, _otsu_threshold


STAFF_BBOX = {"x_min": 20.0, "y_min": 40.0, "x_max": 380.0, "y_max": 80.0}
SYSTEM_BBOX = {"x_min": 15.0, "y_min": 30.0, "x_max": 385.0, "y_max": 90.0}
STAFF_SPACING = 10.0
STAFF_LINES = tuple(
    {
        "start": {"x": 20.0, "y": float(y)},
        "end": {"x": 380.0, "y": float(y)},
    }
    for y in (40, 50, 60, 70, 80)
)


def _staff_image() -> Image.Image:
    image = Image.new("L", (400, 120), 255)
    draw = ImageDraw.Draw(image)
    for y in (40, 50, 60, 70, 80):
        draw.line((20, y, 380, y), fill=0, width=1)
    return image


class M4E3KR2InwardEndpointTests(unittest.TestCase):
    def test_exact_staff_span_barline_has_full_inward_endpoint_support(self) -> None:
        image = _staff_image()
        draw = ImageDraw.Draw(image)
        draw.line((100, 40, 100, 80), fill=0, width=1)
        lines, slope, left, right = _staff_lines(
            STAFF_LINES,
            config=FROZEN_E3K_CONFIG,
        )
        threshold = _otsu_threshold(image, (int(left), 35, int(right), 85))
        coverage, top, bottom, score = _column_evidence_inward(
            image,
            anchor_x=100,
            x_left=int(left),
            x_right=int(right),
            staff_lines=lines,
            staff_slope=slope,
            staff_spacing=STAFF_SPACING,
            threshold=threshold,
            config=FROZEN_E3K_CONFIG,
        )
        self.assertEqual(top, 1.0)
        self.assertEqual(bottom, 1.0)
        self.assertEqual(coverage, 1.0)
        self.assertEqual(score, 1.0)

    def test_short_stem_is_still_rejected(self) -> None:
        image = _staff_image()
        draw = ImageDraw.Draw(image)
        draw.line((100, 40, 100, 80), fill=0, width=2)
        draw.line((170, 47, 170, 74), fill=0, width=2)

        result = propose_measure_boundaries_r2(
            image,
            staff_bbox=STAFF_BBOX,
            five_staff_lines=STAFF_LINES,
            staff_spacing=STAFF_SPACING,
            system_bbox=SYSTEM_BBOX,
        )
        xs = [item.x for item in result.proposals]
        self.assertTrue(any(abs(x - 100.0) <= 2.0 for x in xs))
        self.assertFalse(any(abs(x - 170.0) <= 2.0 for x in xs))
        self.assertEqual(result.stage, STAGE)

    def test_rotated_true_boundary_remains_supported(self) -> None:
        image = Image.new("L", (420, 150), 255)
        draw = ImageDraw.Draw(image)
        slope = 0.05
        lines = []
        for base_y in (40.0, 50.0, 60.0, 70.0, 80.0):
            end_y = base_y + slope * 360.0
            lines.append(
                {
                    "start": {"x": 20.0, "y": base_y},
                    "end": {"x": 380.0, "y": end_y},
                }
            )
            draw.line((20, round(base_y), 380, round(end_y)), fill=0, width=1)

        anchor_x = 200.0
        top_y = 40.0 + slope * (anchor_x - 20.0)
        bottom_y = 80.0 + slope * (anchor_x - 20.0)
        center_y = (top_y + bottom_y) / 2.0
        top_x = anchor_x - slope * (top_y - center_y)
        bottom_x = anchor_x - slope * (bottom_y - center_y)
        draw.line(
            (round(top_x), round(top_y), round(bottom_x), round(bottom_y)),
            fill=0,
            width=2,
        )

        result = propose_measure_boundaries_r2(
            image,
            staff_bbox={"x_min": 20.0, "y_min": 40.0, "x_max": 380.0, "y_max": 98.0},
            five_staff_lines=tuple(lines),
            staff_spacing=STAFF_SPACING,
            system_bbox={"x_min": 15.0, "y_min": 30.0, "x_max": 385.0, "y_max": 110.0},
        )
        self.assertTrue(any(abs(item.x - anchor_x) <= 2.0 for item in result.proposals))
        self.assertTrue(math.isclose(result.staff_slope, slope, abs_tol=1e-9))

    def test_r2_reuses_frozen_non_endpoint_policy(self) -> None:
        self.assertEqual(FROZEN_E3K_CONFIG.minimum_vertical_coverage, 0.45)
        self.assertEqual(FROZEN_E3K_CONFIG.minimum_endpoint_coverage, 0.50)
        self.assertEqual(FROZEN_E3K_CONFIG.horizontal_probe_radius_staff_spaces, 0.10)
        self.assertEqual(FROZEN_E3K_CONFIG.endpoint_half_window_staff_spaces, 0.30)
        self.assertEqual(FROZEN_E3K_CONFIG.cluster_gap_staff_spaces, 0.20)
        self.assertEqual(FROZEN_E3K_CONFIG.maximum_proposals_per_system, 128)

    def test_module_has_no_training_or_model_loading_path(self) -> None:
        source = inspect.getsource(
            __import__(
                "st_omr_training.m4_e3k_r2_inward_endpoint_geometry",
                fromlist=["*"],
            )
        ).lower()
        for forbidden in (
            "import torch",
            "torch.",
            ".backward(",
            "load_state_dict",
            "optimizer.step",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
