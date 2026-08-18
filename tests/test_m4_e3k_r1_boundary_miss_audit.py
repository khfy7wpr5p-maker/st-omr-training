from __future__ import annotations

import inspect
import unittest

from PIL import Image, ImageDraw

from st_omr_training.degradation import sample_degradation_config
from st_omr_training.m4_e3k_boundary_proposals import propose_measure_boundaries
from st_omr_training.m4_e3k_r1_boundary_miss_audit import (
    DIAGNOSTIC_WINDOW_STAFF_SPACES,
    MISS_TOLERANCE_STAFF_SPACES,
    _classify_degradation_profile,
    _diagnose_truth_boundary,
)
import st_omr_training.m4_e3k_r1_boundary_miss_audit as audit_module


STAFF_BBOX = {"x_min": 20.0, "y_min": 40.0, "x_max": 380.0, "y_max": 80.0}
SYSTEM_BBOX = {"x_min": 15.0, "y_min": 30.0, "x_max": 385.0, "y_max": 90.0}
STAFF_SPACING = 10.0
FIVE_LINES = tuple(
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


class M4E3KR1AuditTests(unittest.TestCase):
    def test_profile_recovery_is_exact_for_all_frozen_profiles(self) -> None:
        for index, profile in enumerate(("clean", "light", "medium"), start=1):
            config = sample_degradation_config(index * 101, profile, raster_width=1400)
            self.assertEqual(_classify_degradation_profile(config), profile)

    def test_frozen_diagnostic_surface_is_train_root_cause_only(self) -> None:
        self.assertEqual(MISS_TOLERANCE_STAFF_SPACES, 1.0)
        self.assertEqual(DIAGNOSTIC_WINDOW_STAFF_SPACES, 2.0)
        source = inspect.getsource(audit_module)
        self.assertNotIn("import torch", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("load_state_dict(", source)
        self.assertIn('"validation_opened": False', source)
        self.assertIn('"test_opened": False', source)
        self.assertIn('"authorizes_e3k_b": False', source)
        self.assertIn('"authorizes_d11_validator": False', source)

    def test_true_full_height_barline_is_diagnosed_as_hit(self) -> None:
        image = _staff_image()
        draw = ImageDraw.Draw(image)
        draw.line((120, 40, 120, 80), fill=0, width=2)
        draw.line((300, 40, 300, 80), fill=0, width=2)
        proposals = propose_measure_boundaries(
            image,
            staff_bbox=STAFF_BBOX,
            five_staff_lines=FIVE_LINES,
            staff_spacing=STAFF_SPACING,
            system_bbox=SYSTEM_BBOX,
        )
        diag = _diagnose_truth_boundary(
            image,
            truth_x=120.0,
            staff_bbox=STAFF_BBOX,
            five_staff_lines=FIVE_LINES,
            staff_spacing=STAFF_SPACING,
            system_bbox=SYSTEM_BBOX,
            proposals=proposals.proposals,
            threshold=proposals.otsu_threshold,
        )
        self.assertEqual(diag["reason"], "HIT")
        self.assertLessEqual(diag["nearest_proposal_error_staff_spaces"], 1.0)

    def test_short_truth_stroke_is_fail_classified_not_silently_hit(self) -> None:
        image = _staff_image()
        draw = ImageDraw.Draw(image)
        # Keep one valid proposal elsewhere so the diagnostic surface is finite.
        draw.line((300, 40, 300, 80), fill=0, width=2)
        # Truth-like short stroke does not span both staff endpoints.
        draw.line((170, 49, 170, 72), fill=0, width=2)
        proposals = propose_measure_boundaries(
            image,
            staff_bbox=STAFF_BBOX,
            five_staff_lines=FIVE_LINES,
            staff_spacing=STAFF_SPACING,
            system_bbox=SYSTEM_BBOX,
        )
        diag = _diagnose_truth_boundary(
            image,
            truth_x=170.0,
            staff_bbox=STAFF_BBOX,
            five_staff_lines=FIVE_LINES,
            staff_spacing=STAFF_SPACING,
            system_bbox=SYSTEM_BBOX,
            proposals=proposals.proposals,
            threshold=proposals.otsu_threshold,
        )
        self.assertNotEqual(diag["reason"], "HIT")
        self.assertIn(
            diag["reason"],
            {
                "VERTICAL_COVERAGE_FAIL",
                "TOP_ENDPOINT_FAIL",
                "BOTTOM_ENDPOINT_FAIL",
                "BOTH_ENDPOINTS_FAIL",
                "COMBINED_GATE_FAIL",
            },
        )
        self.assertGreater(diag["nearest_proposal_error_staff_spaces"], 1.0)

    def test_truth_outside_search_surface_has_explicit_reason(self) -> None:
        image = _staff_image()
        draw = ImageDraw.Draw(image)
        draw.line((300, 40, 300, 80), fill=0, width=2)
        proposals = propose_measure_boundaries(
            image,
            staff_bbox=STAFF_BBOX,
            five_staff_lines=FIVE_LINES,
            staff_spacing=STAFF_SPACING,
            system_bbox=SYSTEM_BBOX,
        )
        diag = _diagnose_truth_boundary(
            image,
            truth_x=5.0,
            staff_bbox=STAFF_BBOX,
            five_staff_lines=FIVE_LINES,
            staff_spacing=STAFF_SPACING,
            system_bbox=SYSTEM_BBOX,
            proposals=proposals.proposals,
            threshold=proposals.otsu_threshold,
        )
        self.assertEqual(diag["reason"], "TRUTH_OUTSIDE_SEARCH_X")
        self.assertIsNone(diag["truth_best_vertical_coverage"])


if __name__ == "__main__":
    unittest.main()
