from __future__ import annotations

import unittest

from st_omr_training.external_dataset_registry import (
    DataUseClass,
    ExternalDatasetRegistryError,
    RegistryState,
    validate_training_admission,
)
from st_omr_training.poly_v2_corpus_source_selection import (
    B8K_SOURCE_SELECTION_VERSION,
    FIRST_REAL_CORPUS_SOURCE_CANDIDATES,
    OSSQ_OMR_CAMERA_READY_SHA,
    OSSQ_OMR_SCANNED_TRACK,
    OSSQ_OMR_SYMBOLIC_SYNTHETIC,
    first_real_corpus_source_selection_fingerprint,
)


class PolyV2CorpusSourceSelectionTests(unittest.TestCase):
    def test_camera_ready_source_is_exactly_pinned(self) -> None:
        self.assertEqual(
            OSSQ_OMR_CAMERA_READY_SHA,
            "7a17e45cddc0b7064fc3a179b62caeb57595e993",
        )
        for record in FIRST_REAL_CORPUS_SOURCE_CANDIDATES:
            self.assertIn(OSSQ_OMR_CAMERA_READY_SHA, record.source)
            self.assertIn(OSSQ_OMR_CAMERA_READY_SHA, record.version)

    def test_symbolic_synthetic_component_is_license_verified_but_not_installed(self) -> None:
        record = OSSQ_OMR_SYMBOLIC_SYNTHETIC
        self.assertEqual(record.license_id, "CC0-1.0")
        self.assertEqual(record.data_use_class, DataUseClass.COMMERCIAL_CLEAN)
        self.assertEqual(record.registry_state, RegistryState.LICENSE_VERIFIED)
        self.assertTrue(record.commercial_use_allowed)
        self.assertTrue(record.training_allowed)
        self.assertTrue(record.evaluation_allowed)
        self.assertFalse(record.research_training_ready)
        self.assertFalse(record.commercial_candidate_training_ready)
        self.assertIsNone(record.artifact_sha256)

    def test_scanned_track_stays_fail_closed_pending_upstream_rights_review(self) -> None:
        record = OSSQ_OMR_SCANNED_TRACK
        self.assertEqual(record.data_use_class, DataUseClass.LICENSE_REVIEW_REQUIRED)
        self.assertEqual(record.registry_state, RegistryState.CANDIDATE)
        self.assertIsNone(record.redistribution_allowed)
        self.assertIsNone(record.commercial_use_allowed)
        self.assertIsNone(record.training_allowed)
        self.assertIsNone(record.evaluation_allowed)
        self.assertFalse(record.research_training_ready)
        self.assertFalse(record.evaluation_ready)
        self.assertFalse(record.commercial_candidate_training_ready)
        self.assertIsNone(record.artifact_sha256)
        with self.assertRaises(ExternalDatasetRegistryError):
            validate_training_admission((record,), commercial_candidate=False)

    def test_b8k_does_not_make_any_external_source_training_ready(self) -> None:
        self.assertEqual(len(FIRST_REAL_CORPUS_SOURCE_CANDIDATES), 2)
        for record in FIRST_REAL_CORPUS_SOURCE_CANDIDATES:
            self.assertFalse(record.research_training_ready)
            self.assertFalse(record.evaluation_ready)
            self.assertIsNone(record.artifact_sha256)

    def test_source_selection_fingerprint_is_deterministic(self) -> None:
        self.assertEqual(B8K_SOURCE_SELECTION_VERSION, "st-omr-poly-v2-corpus-source-selection-v1")
        first = first_real_corpus_source_selection_fingerprint()
        second = first_real_corpus_source_selection_fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
