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
        # A note-like short stem does not touch both outer staff lines.
        draw.line((170, 47, 170, 74), fill=0, width=2)

        result = propose_measure_boundaries(
            image,
            staff_bbox=STAFF_BBOX,
            system_bbox=SYSTEM_BBOX,
            staff_spacing=STAFF_SPACING,
        )
        xs = _nearest_xs(result)
        self.assertTrue(any(abs(x - 100.0) <= 2.0 for x in xs))
        self.assertTrue(any(abs(x - 250.0) <= 2.0 for x in xs))
        self.assertFalse(any(abs(x - 170.0) <= 2.0 for x in xs))
        self.assertEqual(result.stage, "M4-E3K-DETERMINISTIC-MEASURE-BOUNDARY-PROPOSALS")

    def test_thick_barline_is_clustered_to_one_proposal(self) -> None:
        image = _staff_image()
        draw = ImageDraw.Draw(image)
        draw.rectangle((199, 40, 203, 80), fill=0)

        result = propose_measure_boundaries(
            image,
            staff_bbox=STAFF_BBOX,
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
            staff_spacing=STAFF_SPACING,
        )
        self.assertTrue(any(abs(proposal.x - 125.0) <= 2.0 for proposal in result.proposals))
        self.assertGreaterEqual(result.otsu_threshold, 90)
        self.assertLess(result.otsu_threshold, 255)

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
                staff_spacing=STAFF_SPACING,
            )
        with self.assertRaises(M4E3KBoundaryProposalError):
            propose_measure_boundaries(
                _staff_image(),
                staff_bbox=STAFF_BBOX,
                staff_spacing=0.0,
            )
        with self.assertRaises(M4E3KBoundaryProposalError):
            propose_measure_boundaries(
                _staff_image(),
                staff_bbox={"x_min": 20, "y_min": 80, "x_max": 380, "y_max": 40},
                staff_spacing=STAFF_SPACING,
            )


if __name__ == "__main__":
    unittest.main()
