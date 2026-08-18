from __future__ import annotations

import inspect
import unittest

from st_omr_training.m4_e3k_b1_d7_staff_geometry import PredictedStaffGeometry
from st_omr_training.m4_e3k_b1_scoring import (
    EXPECTED_TRAIN_INTERIOR_BOUNDARIES,
    EXPECTED_TRAIN_RECORDS,
    EXPECTED_TRAIN_SYSTEMS,
    MINIMUM_BOUNDARY_RECALL_AT_ONE_STAFF_SPACE,
    MINIMUM_SYSTEM_STAFF_MATCH_COVERAGE,
    _profile_payload,
    match_predicted_staffs_to_truth_systems,
    persist_e3k_b1_train_report,
    score_e3k_b1_train,
)


def _predicted(y0: float, y1: float) -> PredictedStaffGeometry:
    lines = tuple(
        {
            "start": {"x": 20.0, "y": y0 + 2.0 + index * 4.0},
            "end": {"x": 380.0, "y": y0 + 2.0 + index * 4.0},
        }
        for index in range(5)
    )
    return PredictedStaffGeometry(
        staff_bbox={"x_min": 20.0, "y_min": y0, "x_max": 380.0, "y_max": y1},
        five_staff_lines=lines,
        staff_spacing=4.0,
        model_component_bbox=(10, int(y0), 190, int(y1)),
        model_staff_slope=0.0,
        line_template_score=100.0,
    )


def _bundle(system_id: str, sy0: float, sy1: float, staff_y0: float, staff_y1: float):
    return {
        "system": {
            "system_id": system_id,
            "system_bbox": {"x_min": 10.0, "y_min": sy0, "x_max": 390.0, "y_max": sy1},
        },
        "staff": {
            "staff_instance_bbox": {
                "x_min": 20.0,
                "y_min": staff_y0,
                "x_max": 380.0,
                "y_max": staff_y1,
            }
        },
        "measures": (),
    }


class M4E3KB1ScoringTests(unittest.TestCase):
    def test_one_to_one_matching_uses_truth_system_vertical_span_only_for_association(self) -> None:
        predicted = (_predicted(20.0, 40.0), _predicted(70.0, 90.0))
        bundles = (
            _bundle("s1", 10.0, 50.0, 19.0, 41.0),
            _bundle("s2", 60.0, 100.0, 69.0, 91.0),
        )
        matches = match_predicted_staffs_to_truth_systems(predicted, bundles)
        self.assertIs(matches[0][0], predicted[0])
        self.assertIs(matches[1][0], predicted[1])
        self.assertGreater(matches[0][1], 0.0)
        self.assertGreater(matches[1][1], 0.0)

    def test_unmatched_system_is_preserved_instead_of_forcing_far_staff(self) -> None:
        predicted = (_predicted(70.0, 90.0),)
        bundles = (
            _bundle("s1", 10.0, 50.0, 19.0, 41.0),
            _bundle("s2", 60.0, 100.0, 69.0, 91.0),
        )
        matches = match_predicted_staffs_to_truth_systems(predicted, bundles)
        self.assertIsNone(matches[0][0])
        self.assertIs(matches[1][0], predicted[0])

    def test_frozen_b1_gates_and_surface_cardinalities(self) -> None:
        self.assertEqual(MINIMUM_SYSTEM_STAFF_MATCH_COVERAGE, 0.98)
        self.assertEqual(MINIMUM_BOUNDARY_RECALL_AT_ONE_STAFF_SPACE, 0.98)
        self.assertEqual(EXPECTED_TRAIN_RECORDS, 1230)
        self.assertEqual(EXPECTED_TRAIN_SYSTEMS, 2346)
        self.assertEqual(EXPECTED_TRAIN_INTERIOR_BOUNDARIES, 7494)

    def test_profile_declares_single_variable_and_no_d7_system_region(self) -> None:
        profile = _profile_payload()
        self.assertEqual(
            profile["single_changed_variable"],
            "D6_truth_staff_geometry_to_frozen_D7_StaffSet_geometry",
        )
        self.assertFalse(profile["d7_system_region_used"])
        self.assertEqual(profile["surface"], "TRAIN_only")
        self.assertEqual(profile["d7_dense_threshold"], 0.5)

    def test_public_scoring_api_has_no_split_selector(self) -> None:
        self.assertNotIn("split", inspect.signature(score_e3k_b1_train).parameters)
        self.assertNotIn("split", inspect.signature(persist_e3k_b1_train_report).parameters)


if __name__ == "__main__":
    unittest.main()
