import json
import tempfile
import unittest
from pathlib import Path

from tests.test_meter_v5_1_bbox_pilot import make_clean_dataset
from st_omr_training import meter_v5_1_bbox_pilot as pilot
from st_omr_training import meter_v5_1_bbox_human_review as review


class TestMeterV51BBoxHumanReview(unittest.TestCase):
    def _complete_pilot_with_one_large_flag(self, root: Path):
        session = pilot.AnnotationSession(data_root=root)
        for i in range(30):
            p = session.sample_payload(i)
            if i == 0:
                session.save_from_preview(
                    token=p["binding_token"],
                    x0=10, y0=5, x1=40, y1=95,
                    preview_width=p["preview_width"],
                    preview_height=p["preview_height"],
                )
            else:
                session.save_from_preview(
                    token=p["binding_token"],
                    x0=10, y0=20, x1=40, y1=70,
                    preview_width=p["preview_width"],
                    preview_height=p["preview_height"],
                )
        audit_path = pilot.write_pilot_audit(root)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(audit["mechanical_gate"], "PASS")
        self.assertEqual(audit["suspicious_too_large_count"], 1)
        self.assertFalse(audit["annotation_contract_freeze_ready"])
        return session.samples[0].sample_id

    def test_accept_as_drawn_resolves_frozen_flag_without_editing_bbox(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            flagged_id = self._complete_pilot_with_one_large_flag(root)
            annotation_path = root / pilot.ANNOTATIONS_DIR / pilot.PILOT_CSV_NAME
            before = annotation_path.read_bytes()

            session = review.HumanReviewSession(data_root=root)
            self.assertEqual(len(session.items), 1)
            self.assertEqual(session.items[0].sample_id, flagged_id)
            result = session.accept_as_drawn(review_index=0)
            self.assertEqual(result["action"], "ACCEPT_AS_DRAWN")
            self.assertEqual(annotation_path.read_bytes(), before)

            audit_path = review.write_human_review_audit(root)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["human_review_gate"], "PASS")
            self.assertTrue(audit["annotation_contract_freeze_ready"])
            self.assertFalse(audit["training_authorized"])
            self.assertEqual(audit["inference_count"], 0)

    def test_resolution_fails_closed_if_bbox_changes_after_acceptance(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            self._complete_pilot_with_one_large_flag(root)
            session = review.HumanReviewSession(data_root=root)
            session.accept_as_drawn(review_index=0)

            p = session.session.sample_payload(session.items[0].pilot_index)
            session.session.save_from_preview(
                token=p["binding_token"],
                x0=12, y0=5, x1=42, y1=95,
                preview_width=p["preview_width"],
                preview_height=p["preview_height"],
            )
            with self.assertRaises(pilot.MeterV5_1PilotError):
                review.HumanReviewSession(data_root=root)

    def test_review_selection_is_train_only_and_keeps_safety_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = make_clean_dataset(Path(td))
            self._complete_pilot_with_one_large_flag(root)
            session = review.HumanReviewSession(data_root=root)
            payload = session.sample_payload(0)
            self.assertEqual(payload["split"], "train")
            self.assertTrue(payload["final_holdout_locked"])
            selection = json.loads(session.selection_path.read_text(encoding="utf-8"))
            self.assertEqual(selection["scope"], "train_pilot_flagged_only")
            self.assertTrue(selection["final_holdout_locked"])
            self.assertFalse(selection["training_authorized"])


if __name__ == "__main__":
    unittest.main()
