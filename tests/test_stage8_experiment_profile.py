from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import unittest

from st_omr_training.real_data_contract import (
    AdmissionState,
    PairingState,
    RealDataManifest,
    RealDataOrigin,
    RealDataSample,
    RealDataSplit,
    ReviewState,
    RightsBasis,
    STAGE7C_CHECKPOINT_SHA256,
    STAGE7C_MODEL_STATE_SHA256,
    real_data_sample_id,
)
from st_omr_training.stage8_experiment_profile import (
    Stage8Candidate,
    Stage8ExperimentBinding,
    Stage8ManifestSummary,
    Stage8PairedRunProfile,
    Stage8ProfileError,
    make_stage8_candidate_binding,
    stage8_paired_profile_fingerprint,
    stage8_receipt_set_sha256,
    summarize_stage8_pilot_manifest,
    validate_paired_experiment_bindings,
)
from st_omr_training.training_model import BaselineModelConfig


def _digest(text: str) -> str:
    return sha256(text.encode("ascii")).hexdigest()


def _sample(index: int, split: RealDataSplit) -> RealDataSample:
    family_id = f"family-{index:03d}"
    source_sha = _digest(f"source-{index}")
    image_sha = _digest(f"image-{index}")
    musicxml_sha = _digest(f"musicxml-{index}")
    semantic_sha = _digest(f"semantic-{index}")
    sample_id = real_data_sample_id(
        family_id=family_id,
        page_number=1,
        source_document_sha256=source_sha,
        image_sha256=image_sha,
        musicxml_sha256=musicxml_sha,
        semantic_fingerprint=semantic_sha,
    )
    return RealDataSample(
        sample_id=sample_id,
        family_id=family_id,
        split=split,
        page_number=1,
        origin=RealDataOrigin.CURATED,
        rights_basis=RightsBasis.PUBLIC_DOMAIN,
        source_document_sha256=source_sha,
        image_sha256=image_sha,
        musicxml_sha256=musicxml_sha,
        semantic_fingerprint=semantic_sha,
        provenance_evidence_sha256=_digest(f"provenance-{index}"),
        rights_evidence_sha256=_digest(f"rights-{index}"),
        pairing_evidence_sha256=_digest(f"pairing-{index}"),
        explicit_training_permission_sha256=None,
        privacy_review_evidence_sha256=None,
        rights_review=ReviewState.APPROVED,
        pairing_review=PairingState.VERIFIED,
        admission_state=AdmissionState.ADMITTED,
    )


def _manifest(train_count: int = 40, validation_count: int = 10) -> RealDataManifest:
    samples = tuple(
        [_sample(index, RealDataSplit.TRAIN) for index in range(train_count)]
        + [
            _sample(train_count + index, RealDataSplit.VALIDATION)
            for index in range(validation_count)
        ]
    )
    return RealDataManifest(
        dataset_name="stage8-pilot",
        dataset_version="v1",
        samples=samples,
        sealed_test_manifest_sha256=_digest("sealed-real-test-manifest"),
    )


class Stage8PairedProfileTests(unittest.TestCase):
    def test_default_profile_is_exact_50_pair_cpu_pilot(self) -> None:
        profile = Stage8PairedRunProfile()
        self.assertEqual(profile.total_samples, 50)
        self.assertEqual(profile.train_samples, 40)
        self.assertEqual(profile.validation_samples, 10)
        self.assertEqual(profile.epochs, 40)
        self.assertEqual(profile.batch_size, 4)
        self.assertEqual(profile.device, "cpu")
        self.assertEqual(profile.cpu_threads, 1)
        self.assertEqual(profile.retained_checkpoints, 1)
        self.assertEqual(profile.validation_cadence, "every-epoch")
        self.assertEqual(
            profile.acceptance_policy,
            "best-validation-loss-strictly-below-preupdate-v1",
        )
        self.assertEqual(profile.numeric_policy, "finite-fail-closed-v1")

    def test_profile_drift_fails_closed(self) -> None:
        with self.assertRaises(Stage8ProfileError):
            Stage8PairedRunProfile(epochs=41)
        with self.assertRaises(Stage8ProfileError):
            Stage8PairedRunProfile(device="cuda")
        with self.assertRaises(Stage8ProfileError):
            Stage8PairedRunProfile(model_config=BaselineModelConfig(hidden_dim=128))
        with self.assertRaises(Stage8ProfileError):
            Stage8PairedRunProfile(validation_cadence="final-only")
        with self.assertRaises(Stage8ProfileError):
            Stage8PairedRunProfile(acceptance_policy="accept-any-finite-run")
        with self.assertRaises(Stage8ProfileError):
            Stage8PairedRunProfile(numeric_policy="allow-nan")

    def test_profile_fingerprint_is_deterministic(self) -> None:
        first = stage8_paired_profile_fingerprint()
        second = stage8_paired_profile_fingerprint(Stage8PairedRunProfile())
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_exact_40_10_admitted_manifest_passes(self) -> None:
        summary = summarize_stage8_pilot_manifest(_manifest())
        self.assertEqual(summary.train_samples, 40)
        self.assertEqual(summary.validation_samples, 10)
        self.assertEqual(summary.train_families, 40)
        self.assertEqual(summary.validation_families, 10)
        self.assertEqual(len(summary.manifest_sha256), 64)

    def test_manifest_count_drift_is_rejected(self) -> None:
        with self.assertRaises(Stage8ProfileError):
            summarize_stage8_pilot_manifest(_manifest(train_count=39, validation_count=10))
        with self.assertRaises(Stage8ProfileError):
            summarize_stage8_pilot_manifest(_manifest(train_count=40, validation_count=11))

    def test_manifest_summary_fails_closed_on_tampered_bounds(self) -> None:
        with self.assertRaises(Stage8ProfileError):
            Stage8ManifestSummary(
                manifest_sha256=_digest("manifest"),
                train_samples=39,
                validation_samples=10,
                train_families=39,
                validation_families=10,
            )
        with self.assertRaises(Stage8ProfileError):
            Stage8ManifestSummary(
                manifest_sha256=_digest("manifest"),
                train_samples=40,
                validation_samples=10,
                train_families=41,
                validation_families=10,
            )

    def test_manifest_summary_requires_profile_type(self) -> None:
        with self.assertRaises(TypeError):
            summarize_stage8_pilot_manifest(_manifest(), profile=object())

    def test_receipt_set_is_order_independent_and_exactly_50(self) -> None:
        receipts = tuple(_digest(f"receipt-{index}") for index in range(50))
        first = stage8_receipt_set_sha256(receipts)
        second = stage8_receipt_set_sha256(tuple(reversed(receipts)))
        self.assertEqual(first, second)
        with self.assertRaises(Stage8ProfileError):
            stage8_receipt_set_sha256(receipts[:-1])

    def test_duplicate_receipt_identity_is_rejected(self) -> None:
        receipts = tuple(_digest(f"receipt-{index}") for index in range(49))
        duplicated = receipts + (receipts[0],)
        with self.assertRaises(Stage8ProfileError):
            stage8_receipt_set_sha256(duplicated)

    def test_candidate_a_binding_is_exact_stage7c_checkpoint(self) -> None:
        manifest = _manifest()
        summary = summarize_stage8_pilot_manifest(manifest)
        receipt_set = stage8_receipt_set_sha256(
            tuple(_digest(f"receipt-{index}") for index in range(50))
        )
        binding = make_stage8_candidate_binding(
            Stage8Candidate.CHECKPOINT_FINE_TUNE,
            development_manifest_sha256=summary.manifest_sha256,
            receipt_set_sha256=receipt_set,
            sealed_test_manifest_sha256=manifest.sealed_test_manifest_sha256,
        )
        self.assertEqual(binding.initialization_checkpoint_sha256, STAGE7C_CHECKPOINT_SHA256)
        self.assertEqual(binding.initialization_model_state_sha256, STAGE7C_MODEL_STATE_SHA256)

    def test_candidate_b_cannot_load_checkpoint(self) -> None:
        with self.assertRaises(Stage8ProfileError):
            Stage8ExperimentBinding(
                candidate=Stage8Candidate.FROM_SCRATCH,
                profile_fingerprint=_digest("profile"),
                development_manifest_sha256=_digest("manifest"),
                receipt_set_sha256=_digest("receipts"),
                sealed_test_manifest_sha256=_digest("sealed"),
                initialization_checkpoint_sha256=STAGE7C_CHECKPOINT_SHA256,
                initialization_model_state_sha256=STAGE7C_MODEL_STATE_SHA256,
            )

    def test_test_access_and_online_learning_fail_closed(self) -> None:
        common = dict(
            candidate=Stage8Candidate.FROM_SCRATCH,
            profile_fingerprint=_digest("profile"),
            development_manifest_sha256=_digest("manifest"),
            receipt_set_sha256=_digest("receipts"),
            sealed_test_manifest_sha256=_digest("sealed"),
            initialization_checkpoint_sha256=None,
            initialization_model_state_sha256=None,
        )
        with self.assertRaises(Stage8ProfileError):
            Stage8ExperimentBinding(**common, test_accessed=True)
        with self.assertRaises(Stage8ProfileError):
            Stage8ExperimentBinding(**common, online_learning=True)

    def test_pair_requires_same_manifest_receipts_profile_and_sealed_commitment(self) -> None:
        manifest = _manifest()
        summary = summarize_stage8_pilot_manifest(manifest)
        receipt_set = stage8_receipt_set_sha256(
            tuple(_digest(f"receipt-{index}") for index in range(50))
        )
        candidate_a = make_stage8_candidate_binding(
            Stage8Candidate.CHECKPOINT_FINE_TUNE,
            development_manifest_sha256=summary.manifest_sha256,
            receipt_set_sha256=receipt_set,
            sealed_test_manifest_sha256=manifest.sealed_test_manifest_sha256,
        )
        candidate_b = make_stage8_candidate_binding(
            Stage8Candidate.FROM_SCRATCH,
            development_manifest_sha256=summary.manifest_sha256,
            receipt_set_sha256=receipt_set,
            sealed_test_manifest_sha256=manifest.sealed_test_manifest_sha256,
        )
        validated_a, validated_b = validate_paired_experiment_bindings(candidate_a, candidate_b)
        self.assertIs(validated_a.candidate, Stage8Candidate.CHECKPOINT_FINE_TUNE)
        self.assertIs(validated_b.candidate, Stage8Candidate.FROM_SCRATCH)

        drifted_b = replace(candidate_b, development_manifest_sha256=_digest("different-manifest"))
        with self.assertRaises(Stage8ProfileError):
            validate_paired_experiment_bindings(candidate_a, drifted_b)

    def test_pair_rejects_two_same_candidates(self) -> None:
        manifest_sha = _digest("manifest")
        receipt_sha = _digest("receipts")
        sealed_sha = _digest("sealed")
        candidate_b = make_stage8_candidate_binding(
            Stage8Candidate.FROM_SCRATCH,
            development_manifest_sha256=manifest_sha,
            receipt_set_sha256=receipt_sha,
            sealed_test_manifest_sha256=sealed_sha,
        )
        with self.assertRaises(Stage8ProfileError):
            validate_paired_experiment_bindings(candidate_b, candidate_b)


if __name__ == "__main__":
    unittest.main()
