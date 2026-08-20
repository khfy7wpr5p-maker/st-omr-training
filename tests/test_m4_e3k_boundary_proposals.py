from __future__ import annotations

import math
import unittest

from PIL import Image, ImageDraw

from st_omr_training.m4_e3k_boundary_proposals import (
    BoundaryProposalConfig,
    M4E3KBoundaryProposalError,
    evaluate_boundary_recall,
    propose_measure_boundaries,
)


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


def _staff_image(*, faint: bool = False) -> Image.Image:
    image = Image.new("L", (400, 120), 255)
    draw = ImageDraw.Draw(image)
    ink = 120 if faint else 0
    for y in (40, 50, 60, 70, 80):
        draw.line((20, y, 380, y), fill=ink, width=1)
    return image


def _nearest_xs(result: object) -> list[float]:
    return [proposal.x for proposal in result.proposals]


class M4E3KBoundaryProposalTests(unittest.TestCase):
    def test_recovers_full_height_barlines_and_rejects_short_stem(self) -> None:
        image = _staff_image()
        draw = ImageDraw.Draw(image)
        draw.line((100, 40, 100, 80), fill=0, width=2)
        draw.line((250, 40, 250, 80), fill=0, width=2)
        # A note-like short stem does not have vertical continuity through both
        # endpoint windows even though horizontal staff lines cross its x.
        draw.line((170, 47, 170, 74), fill=0, width=2)

        result = propose_measure_boundaries(
            image,
            staff_bbox=STAFF_BBOX,
            five_staff_lines=STAFF_LINES,
            system_bbox=SYSTEM_BBOX,
            staff_spacing=STAFF_SPACING,
        )
        xs = _nearest_xs(result)
        self.assertTrue(any(abs(x - 100.0) <= 2.0 for x in xs))
        self.assertTrue(any(abs(x - 250.0) <= 2.0 for x in xs))
        self.assertFalse(any(abs(x - 170.0) <= 2.0 for x in xs))
        self.assertEqual(result.stage, "M4-E3K-DETERMINISTIC-MEASURE-BOUNDARY-PROPOSALS")
        self.assertTrue(math.isclose(result.staff_slope, 0.0, abs_tol=1e-12))

    def test_thick_barline_is_clustered_to_one_proposal(self) -> None:
        image = _staff_image()
        draw = ImageDraw.Draw(image)
        draw.rectangle((199, 40, 203, 80), fill=0)

        result = propose_measure_boundaries(
            image,
            staff_bbox=STAFF_BBOX,
            five_staff_lines=STAFF_LINES,
            staff_spacing=STAFF_SPACING,
        )
        near = [proposal for proposal in result.proposals if abs(proposal.x - 201.0) <= 4.0]
        self.assertEqual(len(near), 1)
        self.assertLessEqual(near[0].cluster_left, 199)
        self.assertGreaterEqual(near[0].cluster_right, 203)

    def test_otsu_path_handles_faint_staff_and_barlines(self) -> None:
        image = _staff_image(faint=True)
        draw = ImageDraw.Draw(image)
        draw.line((125, 40, 125, 80), fill=90, width=2)

        result = propose_measure_boundaries(
            image,
            staff_bbox=STAFF_BBOX,
            five_staff_lines=STAFF_LINES,
            staff_spacing=STAFF_SPACING,
        )
        self.assertTrue(any(abs(proposal.x - 125.0) <= 2.0 for proposal in result.proposals))
        self.assertGreaterEqual(result.otsu_threshold, 90)
        self.assertLess(result.otsu_threshold, 255)

    def test_rotated_staff_uses_perpendicular_probe(self) -> None:
        image = Image.new("L", (420, 150), 255)
        draw = ImageDraw.Draw(image)
        slope = 0.05
        lines = []
        for base_y in (40.0, 50.0, 60.0, 70.0, 80.0):
            start = {"x": 20.0, "y": base_y}
            end_y = base_y + slope * (380.0 - 20.0)
            end = {"x": 380.0, "y": end_y}
            lines.append({"start": start, "end": end})
            draw.line((20, round(base_y), 380, round(end_y)), fill=0, width=1)

        anchor_x = 200.0
        top_y = 40.0 + slope * (anchor_x - 20.0)
        bottom_y = 80.0 + slope * (anchor_x - 20.0)
        center_y = (top_y + bottom_y) / 2.0
        # A true boundary is perpendicular to the staff: dx/dy = -slope.
        top_x = anchor_x - slope * (top_y - center_y)
        bottom_x = anchor_x - slope * (bottom_y - center_y)
        draw.line(
            (round(top_x), round(top_y), round(bottom_x), round(bottom_y)),
            fill=0,
            width=2,
        )

        result = propose_measure_boundaries(
            image,
            staff_bbox={"x_min": 20.0, "y_min": 40.0, "x_max": 380.0, "y_max": 98.0},
            five_staff_lines=tuple(lines),
            staff_spacing=STAFF_SPACING,
            system_bbox={"x_min": 15.0, "y_min": 30.0, "x_max": 385.0, "y_max": 110.0},
        )
        self.assertTrue(any(abs(proposal.x - anchor_x) <= 2.0 for proposal in result.proposals))
        self.assertTrue(math.isclose(result.staff_slope, slope, abs_tol=1e-9))

    def test_candidate_bound_fails_closed_instead_of_top_k_pruning(self) -> None:
        image = _staff_image()
        draw = ImageDraw.Draw(image)
        for x in (80, 140, 200):
            draw.line((x, 40, x, 80), fill=0, width=1)
        config = BoundaryProposalConfig(maximum_proposals_per_system=2)
        with self.assertRaises(M4E3KBoundaryProposalError):
            propose_measure_boundaries(
                image,
                staff_bbox=STAFF_BBOX,
                five_staff_lines=STAFF_LINES,
                staff_spacing=STAFF_SPACING,
                config=config,
            )

    def test_recall_audit_reports_fixed_staff_space_tolerances(self) -> None:
        image = _staff_image()
        draw = ImageDraw.Draw(image)
        draw.line((100, 40, 100, 80), fill=0, width=1)
        draw.line((250, 40, 250, 80), fill=0, width=1)
        result = propose_measure_boundaries(
            image,
            staff_bbox=STAFF_BBOX,
            five_staff_lines=STAFF_LINES,
            staff_spacing=STAFF_SPACING,
        )
        metrics = evaluate_boundary_recall(
            result.proposals,
            (102.0, 257.0),
            staff_spacing=STAFF_SPACING,
        )
        self.assertEqual(metrics.truth_count, 2)
        self.assertEqual(metrics.recall_by_tolerance[0.5], 0.5)
        self.assertEqual(metrics.recall_by_tolerance[1.0], 1.0)
        self.assertTrue(math.isclose(metrics.p50_error_staff_spaces, 0.45, abs_tol=0.11))

    def test_invalid_geometry_and_image_mode_fail_closed(self) -> None:
        rgb = Image.new("RGB", (100, 100), "white")
        with self.assertRaises(M4E3KBoundaryProposalError):
            propose_measure_boundaries(
                rgb,
                staff_bbox=STAFF_BBOX,
                five_staff_lines=STAFF_LINES,
                staff_spacing=STAFF_SPACING,
            )
        with self.assertRaises(M4E3KBoundaryProposalError):
            propose_measure_boundaries(
                _staff_image(),
                staff_bbox=STAFF_BBOX,
                five_staff_lines=STAFF_LINES,
                staff_spacing=0.0,
            )
        with self.assertRaises(M4E3KBoundaryProposalError):
            propose_measure_boundaries(
                _staff_image(),
                staff_bbox={"x_min": 20, "y_min": 80, "x_max": 380, "y_max": 40},
                five_staff_lines=STAFF_LINES,
                staff_spacing=STAFF_SPACING,
            )
        inconsistent_lines = list(STAFF_LINES)
        inconsistent_lines[-1] = {
            "start": {"x": 20.0, "y": 80.0},
            "end": {"x": 380.0, "y": 130.0},
        }
        with self.assertRaises(M4E3KBoundaryProposalError):
            propose_measure_boundaries(
                _staff_image(),
                staff_bbox=STAFF_BBOX,
                five_staff_lines=tuple(inconsistent_lines),
                staff_spacing=STAFF_SPACING,
            )


if __name__ == "__main__":
    unittest.main()
