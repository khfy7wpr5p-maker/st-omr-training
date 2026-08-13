from __future__ import annotations

from dataclasses import replace
import unittest

from st_omr_training.real_data_contract import (
    AdmissionState,
    PairingState,
    REAL_DATA_SCHEMA_VERSION,
    RealDataContractError,
    RealDataManifest,
    RealDataOrigin,
    RealDataSample,
    RealDataSplit,
    ReviewState,
    RightsBasis,
    STAGE7C_ARTIFACT_EXPIRES_AT,
    STAGE7C_ARTIFACT_ID,
    STAGE7C_CHECKPOINT_SHA256,
    SealedTestAccessError,
    canonical_real_data_manifest_bytes,
    real_data_manifest_sha256,
    real_data_sample_id,
    select_stage8_development_records,
    validate_quarantine_record,
    validate_real_data_manifest,
    validate_real_data_sample,
)


def h(ch: str) -> str:
    return ch * 64


def make_sample(
    *,
    family: str,
    split: RealDataSplit,
    page: int,
    source: str,
    image: str,
    target: str,
    semantic: str,
    origin: RealDataOrigin = RealDataOrigin.CURATED,
    rights_basis: RightsBasis = RightsBasis.OPEN_LICENSE,
    permission: str | None = None,
    privacy: str | None = None,
    admission: AdmissionState = AdmissionState.ADMITTED,
    rights_review: ReviewState = ReviewState.APPROVED,
    pairing_review: PairingState = PairingState.VERIFIED,
) -> RealDataSample:
    sample_id = real_data_sample_id(
        family_id=family,
        page_number=page,
        source_document_sha256=source,
        image_sha256=image,
        musicxml_sha256=target,
        semantic_fingerprint=semantic,
    )
    return RealDataSample(
        sample_id=sample_id,
        family_id=family,
        split=split,
        page_number=page,
        origin=origin,
        rights_basis=rights_basis,
        source_document_sha256=source,
        image_sha256=image,
        musicxml_sha256=target,
        semantic_fingerprint=semantic,
        provenance_evidence_sha256=h("a"),
        rights_evidence_sha256=h("b"),
        pairing_evidence_sha256=h("c"),
        explicit_training_permission_sha256=permission,
        privacy_review_evidence_sha256=privacy,
        rights_review=rights_review,
        pairing_review=pairing_review,
        admission_state=admission,
    )


def make_manifest() -> RealDataManifest:
    return RealDataManifest(
        dataset_name="stage8-real-v1",
        dataset_version="v1",
        samples=(
            make_sample(family="fam-train", split=RealDataSplit.TRAIN, page=1, source=h("1"), image=h("2"), target=h("3"), semantic=h("4")),
            make_sample(family="fam-val", split=RealDataSplit.VALIDATION, page=1, source=h("5"), image=h("6"), target=h("7"), semantic=h("8")),
            make_sample(family="fam-test", split=RealDataSplit.TEST, page=1, source=h("9"), image=h("d"), target=h("e"), semantic=h("f")),
        ),
    )


class RealDataContractTests(unittest.TestCase):
    def test_stage7c_checkpoint_identity_is_frozen(self) -> None:
        self.assertEqual(STAGE7C_ARTIFACT_ID, 9177923796)
        self.assertEqual(STAGE7C_ARTIFACT_EXPIRES_AT, "2026-09-12T10:44:21Z")
        self.assertEqual(STAGE7C_CHECKPOINT_SHA256, "75c33cefeb970305f5f9171b3274dbe8b785cbcf1e8d6851de7848e66d24efa4")
        self.assertEqual(REAL_DATA_SCHEMA_VERSION, "st-real-data-admission-v1")

    def test_valid_manifest_passes_and_is_canonical(self) -> None:
        manifest = make_manifest()
        self.assertTrue(validate_real_data_manifest(manifest).is_valid)
        first = canonical_real_data_manifest_bytes(manifest)
        reordered = replace(manifest, samples=tuple(reversed(manifest.samples)))
        self.assertEqual(first, canonical_real_data_manifest_bytes(reordered))
        self.assertEqual(real_data_manifest_sha256(manifest), real_data_manifest_sha256(reordered))

    def test_sample_identity_does_not_change_when_split_changes(self) -> None:
        sample = make_manifest().samples[0]
        moved = replace(sample, split=RealDataSplit.VALIDATION)
        self.assertEqual(sample.sample_id, moved.sample_id)
        self.assertTrue(validate_real_data_sample(moved).is_valid)

    def test_quarantine_record_is_not_training_eligible(self) -> None:
        sample = replace(make_manifest().samples[0], admission_state=AdmissionState.QUARANTINED, rights_review=ReviewState.PENDING, pairing_review=PairingState.PENDING, rights_evidence_sha256=None, pairing_evidence_sha256=None)
        self.assertTrue(validate_quarantine_record(sample).is_valid)
        result = validate_real_data_sample(sample)
        self.assertFalse(result.is_valid)
        self.assertIn("admission.state", {issue.code for issue in result.issues})

    def test_tampered_sample_identity_fails(self) -> None:
        sample = replace(make_manifest().samples[0], sample_id=h("0"))
        result = validate_real_data_sample(sample)
        self.assertFalse(result.is_valid)
        self.assertIn("lineage.sample_id", {issue.code for issue in result.issues})

    def test_user_submission_requires_explicit_training_permission_and_privacy_review(self) -> None:
        sample = replace(make_manifest().samples[0], origin=RealDataOrigin.USER_SUBMISSION)
        result = validate_real_data_sample(sample)
        codes = {issue.code for issue in result.issues}
        self.assertIn("permission.explicit_training", codes)
        self.assertIn("privacy.review", codes)

    def test_teacher_correction_with_explicit_evidence_can_be_admitted(self) -> None:
        sample = replace(
            make_manifest().samples[0],
            origin=RealDataOrigin.TEACHER_CORRECTION,
            rights_basis=RightsBasis.EXPLICIT_PERMISSION,
            explicit_training_permission_sha256=h("a"),
            privacy_review_evidence_sha256=h("b"),
        )
        self.assertTrue(validate_real_data_sample(sample).is_valid)

    def test_scoremosaic_upload_never_becomes_eligible_without_explicit_permission(self) -> None:
        sample = replace(make_manifest().samples[0], origin=RealDataOrigin.SCOREMOSAIC_UPLOAD)
        self.assertFalse(validate_real_data_sample(sample).is_valid)

    def test_duplicate_image_is_rejected(self) -> None:
        manifest = make_manifest()
        duplicate = replace(manifest.samples[1], image_sha256=manifest.samples[0].image_sha256)
        duplicate = replace(
            duplicate,
            sample_id=real_data_sample_id(
                family_id=duplicate.family_id,
                page_number=duplicate.page_number,
                source_document_sha256=duplicate.source_document_sha256,
                image_sha256=duplicate.image_sha256,
                musicxml_sha256=duplicate.musicxml_sha256,
                semantic_fingerprint=duplicate.semantic_fingerprint,
            ),
        )
        result = validate_real_data_manifest(replace(manifest, samples=(manifest.samples[0], duplicate, manifest.samples[2])))
        self.assertIn("duplicate.image_sha256", {issue.code for issue in result.issues})

    def test_family_split_leakage_is_rejected(self) -> None:
        manifest = make_manifest()
        leaked = replace(manifest.samples[1], family_id=manifest.samples[0].family_id)
        leaked = replace(
            leaked,
            sample_id=real_data_sample_id(
                family_id=leaked.family_id,
                page_number=leaked.page_number,
                source_document_sha256=leaked.source_document_sha256,
                image_sha256=leaked.image_sha256,
                musicxml_sha256=leaked.musicxml_sha256,
                semantic_fingerprint=leaked.semantic_fingerprint,
            ),
        )
        result = validate_real_data_manifest(replace(manifest, samples=(manifest.samples[0], leaked, manifest.samples[2])))
        self.assertIn("leakage.family_split", {issue.code for issue in result.issues})

    def test_target_and_semantic_split_leakage_are_rejected(self) -> None:
        manifest = make_manifest()
        leaked = replace(
            manifest.samples[1],
            musicxml_sha256=manifest.samples[0].musicxml_sha256,
            semantic_fingerprint=manifest.samples[0].semantic_fingerprint,
        )
        leaked = replace(
            leaked,
            sample_id=real_data_sample_id(
                family_id=leaked.family_id,
                page_number=leaked.page_number,
                source_document_sha256=leaked.source_document_sha256,
                image_sha256=leaked.image_sha256,
                musicxml_sha256=leaked.musicxml_sha256,
                semantic_fingerprint=leaked.semantic_fingerprint,
            ),
        )
        result = validate_real_data_manifest(replace(manifest, samples=(manifest.samples[0], leaked, manifest.samples[2])))
        codes = {issue.code for issue in result.issues}
        self.assertIn("leakage.target_split", codes)
        self.assertIn("leakage.semantic_split", codes)

    def test_target_alias_under_different_family_is_rejected_even_in_same_split(self) -> None:
        manifest = make_manifest()
        alias = make_sample(
            family="other-train-family",
            split=RealDataSplit.TRAIN,
            page=2,
            source=h("a"),
            image=h("b"),
            target=manifest.samples[0].musicxml_sha256,
            semantic=manifest.samples[0].semantic_fingerprint,
        )
        result = validate_real_data_manifest(
            replace(manifest, samples=(manifest.samples[0], alias, manifest.samples[1], manifest.samples[2]))
        )
        codes = {issue.code for issue in result.issues}
        self.assertIn("leakage.target_family", codes)
        self.assertIn("leakage.semantic_family", codes)

    def test_manifest_requires_all_three_splits(self) -> None:
        manifest = make_manifest()
        result = validate_real_data_manifest(replace(manifest, samples=manifest.samples[:2]))
        self.assertIn("manifest.missing_split", {issue.code for issue in result.issues})

    def test_test_split_is_sealed_for_stage8_development(self) -> None:
        manifest = make_manifest()
        with self.assertRaises(SealedTestAccessError):
            select_stage8_development_records(manifest, RealDataSplit.TEST)
        self.assertEqual(len(select_stage8_development_records(manifest, RealDataSplit.TRAIN)), 1)
        self.assertEqual(len(select_stage8_development_records(manifest, RealDataSplit.VALIDATION)), 1)

    def test_invalid_manifest_cannot_be_selected(self) -> None:
        manifest = replace(make_manifest(), samples=make_manifest().samples[:2])
        with self.assertRaises(RealDataContractError):
            select_stage8_development_records(manifest, RealDataSplit.TRAIN)


if __name__ == "__main__":
    unittest.main()
