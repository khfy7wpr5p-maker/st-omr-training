from __future__ import annotations

from dataclasses import replace
import unittest

from st_omr_training.external_dataset_registry import (
    DataUseClass,
    EXTERNAL_DATASET_CANDIDATES,
    ExternalDatasetRecord,
    ExternalDatasetRegistryError,
    RegistryState,
    validate_evaluation_admission,
    validate_registry,
    validate_training_admission,
)


SHA_A = "a" * 64


def _verified_commercial() -> ExternalDatasetRecord:
    return ExternalDatasetRecord(
        dataset_name="Fixture",
        dataset_component="component",
        source="https://example.test/source",
        version="1",
        license_id="CC0-1.0",
        license_evidence="https://example.test/license",
        redistribution_allowed=True,
        commercial_use_allowed=True,
        training_allowed=True,
        evaluation_allowed=True,
        derivative_restrictions="none",
        data_use_class=DataUseClass.COMMERCIAL_CLEAN,
        registry_state=RegistryState.LICENSE_VERIFIED,
    )


class ExternalDatasetRegistryTests(unittest.TestCase):
    def test_seed_registry_contains_expected_candidates(self) -> None:
        identities = {
            (record.dataset_name, record.dataset_component)
            for record in EXTERNAL_DATASET_CANDIDATES
        }
        self.assertEqual(
            identities,
            {
                ("Muse OMR Benchmark", "1077 symbolic-score + augmented-PDF pairs"),
                ("DeepScoresV2", "complete/dense music-object detection dataset"),
                ("MUSCIMA++", "MUSCIMA++ annotations"),
                ("OLiMPiC", "synthetic 1.0"),
                ("OLiMPiC", "scanned 1.0"),
                ("GrandStaff-LMX", "added .lmx and .musicxml annotations only"),
                ("GrandStaff", "original pianoform dataset"),
                ("DoReMi", "published openly distributable score subset"),
            },
        )

    def test_seed_registry_is_conservative_about_use_classes(self) -> None:
        by_name = {
            (record.dataset_name, record.dataset_component): record
            for record in EXTERNAL_DATASET_CANDIDATES
        }
        self.assertEqual(
            by_name[("Muse OMR Benchmark", "1077 symbolic-score + augmented-PDF pairs")].data_use_class,
            DataUseClass.COMMERCIAL_CLEAN,
        )
        self.assertEqual(
            by_name[("DeepScoresV2", "complete/dense music-object detection dataset")].data_use_class,
            DataUseClass.COMMERCIAL_CLEAN,
        )
        self.assertEqual(
            by_name[("MUSCIMA++", "MUSCIMA++ annotations")].data_use_class,
            DataUseClass.RESEARCH_ONLY,
        )
        for identity in (
            ("OLiMPiC", "synthetic 1.0"),
            ("OLiMPiC", "scanned 1.0"),
            ("GrandStaff-LMX", "added .lmx and .musicxml annotations only"),
            ("GrandStaff", "original pianoform dataset"),
            ("DoReMi", "published openly distributable score subset"),
        ):
            self.assertEqual(
                by_name[identity].data_use_class,
                DataUseClass.LICENSE_REVIEW_REQUIRED,
            )

    def test_no_seed_candidate_is_training_or_evaluation_ready(self) -> None:
        self.assertTrue(EXTERNAL_DATASET_CANDIDATES)
        for record in EXTERNAL_DATASET_CANDIDATES:
            self.assertFalse(record.research_training_ready)
            self.assertFalse(record.evaluation_ready)
            self.assertFalse(record.commercial_candidate_training_ready)
            self.assertIsNone(record.artifact_sha256)

    def test_license_review_required_cannot_assert_permissions(self) -> None:
        with self.assertRaises(ExternalDatasetRegistryError):
            ExternalDatasetRecord(
                dataset_name="Unknown",
                dataset_component="component",
                source="https://example.test/source",
                version="1",
                license_id="UNKNOWN",
                license_evidence="https://example.test/license",
                redistribution_allowed=None,
                commercial_use_allowed=True,
                training_allowed=None,
                evaluation_allowed=None,
                derivative_restrictions="review required",
                data_use_class=DataUseClass.LICENSE_REVIEW_REQUIRED,
                registry_state=RegistryState.CANDIDATE,
            )

    def test_license_review_required_cannot_be_install_pinned(self) -> None:
        with self.assertRaises(ExternalDatasetRegistryError):
            ExternalDatasetRecord(
                dataset_name="Unknown",
                dataset_component="component",
                source="https://example.test/source",
                version="1",
                license_id="UNKNOWN",
                license_evidence="https://example.test/license",
                redistribution_allowed=None,
                commercial_use_allowed=None,
                training_allowed=None,
                evaluation_allowed=None,
                derivative_restrictions="review required",
                data_use_class=DataUseClass.LICENSE_REVIEW_REQUIRED,
                registry_state=RegistryState.INSTALL_PINNED,
                artifact_sha256=SHA_A,
            )

    def test_verified_record_is_not_ready_until_sha256_install_pin(self) -> None:
        record = _verified_commercial()
        self.assertFalse(record.research_training_ready)
        self.assertFalse(record.evaluation_ready)
        with self.assertRaises(ExternalDatasetRegistryError):
            replace(record, registry_state=RegistryState.INSTALL_PINNED)

        pinned = replace(
            record,
            registry_state=RegistryState.INSTALL_PINNED,
            artifact_sha256=SHA_A,
        )
        self.assertTrue(pinned.research_training_ready)
        self.assertTrue(pinned.evaluation_ready)
        self.assertTrue(pinned.commercial_candidate_training_ready)

    def test_research_only_can_never_enter_commercial_candidate_training(self) -> None:
        research = ExternalDatasetRecord(
            dataset_name="Research",
            dataset_component="component",
            source="https://example.test/source",
            version="1",
            license_id="CC-BY-NC-SA-4.0",
            license_evidence="https://example.test/license",
            redistribution_allowed=True,
            commercial_use_allowed=False,
            training_allowed=True,
            evaluation_allowed=True,
            derivative_restrictions="NC-SA",
            data_use_class=DataUseClass.RESEARCH_ONLY,
            registry_state=RegistryState.INSTALL_PINNED,
            artifact_sha256=SHA_A,
        )
        self.assertTrue(research.research_training_ready)
        self.assertFalse(research.commercial_candidate_training_ready)
        validate_training_admission((research,), commercial_candidate=False)
        with self.assertRaises(ExternalDatasetRegistryError):
            validate_training_admission((research,), commercial_candidate=True)

    def test_evaluation_only_cannot_be_used_for_training(self) -> None:
        evaluation = ExternalDatasetRecord(
            dataset_name="Eval",
            dataset_component="component",
            source="https://example.test/source",
            version="1",
            license_id="evaluation-license",
            license_evidence="https://example.test/license",
            redistribution_allowed=False,
            commercial_use_allowed=False,
            training_allowed=False,
            evaluation_allowed=True,
            derivative_restrictions="evaluation only",
            data_use_class=DataUseClass.EVALUATION_ONLY,
            registry_state=RegistryState.INSTALL_PINNED,
            artifact_sha256=SHA_A,
        )
        validate_evaluation_admission((evaluation,))
        with self.assertRaises(ExternalDatasetRegistryError):
            validate_training_admission((evaluation,), commercial_candidate=False)

    def test_registry_rejects_duplicate_identity_and_raw_entries(self) -> None:
        record = _verified_commercial()
        with self.assertRaises(ExternalDatasetRegistryError):
            validate_registry((record, record))
        with self.assertRaises(ExternalDatasetRegistryError):
            validate_registry(({"dataset_name": "raw"},))

    def test_record_hash_is_deterministic(self) -> None:
        record = _verified_commercial()
        self.assertEqual(record.canonical_sha256(), record.canonical_sha256())
        self.assertEqual(len(record.canonical_sha256()), 64)


if __name__ == "__main__":
    unittest.main()
