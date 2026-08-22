import inspect
import unittest

from st_omr_training import meter_v5_2b_train_replacement_v1 as m


class TestMeterV52BTrainReplacementV1(unittest.TestCase):
    def test_approved_hold_identities_are_exact_and_non_seed_indices(self):
        self.assertEqual(
            m.APPROVED_HOLDS,
            {
                "150201200-1_1_1": {
                    "meter": "2/4",
                    "index": 63,
                    "required_reason": "A04_PAGE_CROPPED",
                },
                "110003725-1_1_1": {
                    "meter": "3/4",
                    "index": 125,
                    "required_reason": "A04_PAGE_CROPPED",
                },
            },
        )
        self.assertTrue(all(int(item["index"]) >= 30 for item in m.APPROVED_HOLDS.values()))

    def test_next_unused_is_frozen_rank_then_sample_id(self):
        rows = [
            {"Split": "train", "Meter": "2/4", "SplitRank": "002", "SampleId": "b", "FamilyId": "fb"},
            {"Split": "train", "Meter": "2/4", "SplitRank": "001", "SampleId": "z", "FamilyId": "fz"},
            {"Split": "train", "Meter": "2/4", "SplitRank": "001", "SampleId": "a", "FamilyId": "fa"},
        ]
        picked = m.select_next_unused_train_row_v1(
            rows,
            meter="2/4",
            existing_sample_ids=set(),
            existing_family_ids=set(),
        )
        self.assertEqual(picked["SampleId"], "a")

    def test_next_unused_skips_existing_sample_and_family_only(self):
        rows = [
            {"Split": "train", "Meter": "3/4", "SplitRank": "001", "SampleId": "a", "FamilyId": "fa"},
            {"Split": "train", "Meter": "3/4", "SplitRank": "002", "SampleId": "b", "FamilyId": "fb"},
            {"Split": "train", "Meter": "3/4", "SplitRank": "003", "SampleId": "c", "FamilyId": "fc"},
        ]
        picked = m.select_next_unused_train_row_v1(
            rows,
            meter="3/4",
            existing_sample_ids={"a"},
            existing_family_ids={"fb"},
        )
        self.assertEqual(picked["SampleId"], "c")

    def test_wrong_meter_rows_are_not_candidates(self):
        rows = [
            {"Split": "train", "Meter": "2/4", "SplitRank": "001", "SampleId": "a", "FamilyId": "fa"},
            {"Split": "train", "Meter": "3/4", "SplitRank": "002", "SampleId": "b", "FamilyId": "fb"},
        ]
        picked = m.select_next_unused_train_row_v1(
            rows,
            meter="3/4",
            existing_sample_ids=set(),
            existing_family_ids=set(),
        )
        self.assertEqual(picked["SampleId"], "b")

    def test_no_geometry_driven_skip_ahead_in_selector(self):
        source = inspect.getsource(m.select_next_unused_train_row_v1).lower()
        self.assertNotIn("geometry", source)
        self.assertNotIn("accepted", source)
        self.assertNotIn("a04", source)
        self.assertIn("selection_rank", m.REPLACEMENT_RULE)

    def test_apply_keeps_training_closed_and_two_human_boxes_required(self):
        source = inspect.getsource(m.apply_approved_train_replacements_v1)
        self.assertIn('"new_human_bboxes_required": 2', source)
        self.assertIn('"training_authorized": False', source)
        self.assertIn('"validation_opened": False', source)
        self.assertIn('"final_holdout_locked": True', source)
        self.assertIn('retained_annotations', source)

    def test_apply_invalidates_stale_qa_and_preflight_evidence(self):
        source = inspect.getsource(m.apply_approved_train_replacements_v1)
        self.assertIn("v52b.HUMAN_QA_NAME", source)
        self.assertIn("v52b2.PREFLIGHT_CSV_NAME", source)
        self.assertIn("v52b2.PREFLIGHT_AUDIT_NAME", source)
        self.assertIn("path.unlink()", source)


if __name__ == "__main__":
    unittest.main()
