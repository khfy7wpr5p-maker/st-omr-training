import inspect
import unittest

from st_omr_training import meter_v5_2b_post_replacement_human_qa_v1 as m


class TestMeterV52BPostReplacementHumanQA(unittest.TestCase):
    def test_confirmation_token_is_exact_and_specific(self):
        self.assertEqual(
            m.POST_HUMAN_QA_CONFIRMATION,
            "V5_2B_REPLACEMENT_QA_2_OF_2_PASS",
        )

    def test_legacy_payload_is_exact_existing_contract(self):
        payload = m.legacy_current_human_qa_payload_v1(
            selection_sha256="a" * 64,
            annotation_sha256="b" * 64,
            mechanical_audit_sha256="c" * 64,
        )
        self.assertEqual(payload["schema"], m.v52b.HUMAN_QA_SCHEMA)
        self.assertEqual(payload["selection_sha256"], "a" * 64)
        self.assertEqual(payload["annotation_sha256"], "b" * 64)
        self.assertEqual(payload["mechanical_audit_sha256"], "c" * 64)
        self.assertEqual(payload["contact_sheets_reviewed"], 15)
        self.assertEqual(payload["contact_sheet_visual_errors_reported"], 0)
        self.assertEqual(payload["human_visual_qa"], "PASS")
        self.assertTrue(payload["slot_derivation_authorized"])
        self.assertTrue(payload["adaptation_training_boundary_authorized"])
        self.assertEqual(payload["trainable_specialists"], ["2-AI", "3-AI"])
        self.assertEqual(payload["frozen_control_specialist"], "4-AI")
        self.assertFalse(payload["threshold_tuning_allowed"])
        self.assertFalse(payload["validation_opened"])
        self.assertTrue(payload["final_holdout_locked"])

    def test_archived_payload_validator_fails_on_any_mutation(self):
        expected = {
            "schema": "x",
            "human_visual_qa": "PASS",
            "contact_sheets_reviewed": 15,
        }
        m.validate_archived_original_qa_payload_v1(dict(expected), expected)
        mutated = dict(expected)
        mutated["contact_sheets_reviewed"] = 14
        with self.assertRaises(Exception):
            m.validate_archived_original_qa_payload_v1(mutated, expected)

    def test_module_has_no_geometry_slot_training_or_inference_execution(self):
        source = inspect.getsource(m).lower()
        self.assertNotIn("detect_multistaff_geometry", source)
        self.assertNotIn("normalize_raster_page", source)
        self.assertNotIn("derive_staff_relative_slots", source)
        self.assertNotIn("optimizer =", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("train_adapted_specialists", source)
        self.assertNotIn("evaluate_diagnostic_gate", source)


if __name__ == "__main__":
    unittest.main()
