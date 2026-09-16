from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from st_omr_training.dataset_manifest import DatasetSplit
from st_omr_training.polyphonic_serialization import serialize_polyphonic_score
from st_omr_training.poly_v2_dataset_materialization import (
    NativePolyV2ArtifactInput,
    build_native_poly_v2_dataset,
    persist_native_poly_v2_dataset,
)
from st_omr_training.poly_v2_dataset_reload import load_and_verify_native_poly_v2_dataset
from st_omr_training.poly_v2_real_corpus_admission import (
    NativePolyV2RealCorpusAdmissionError,
    NativePolyV2RealDataBinding,
    admit_native_poly_v2_real_corpus,
)
from st_omr_training.real_data_contract import (
    AdmissionState,
    RealDataManifest,
    RealDataOrigin,
    RealDataSplit,
)
from st_omr_training.real_data_intake import validate_quarantined_sample_bytes

from test_poly_v2_dataset_reload import _score, _sealed_test_sample
from test_real_data_intake import make_musicxml, make_png, make_quarantined_sample


class PolyV2RealCorpusAdmissionTests(unittest.TestCase):
    def _fixture(self, parent: Path):
        train_image = make_png("vertical")
        validation_image = make_png("horizontal")
        artifacts = (
            NativePolyV2ArtifactInput(
                family_id="fam-train",
                split=DatasetSplit.TRAIN,
                target_json=serialize_polyphonic_score(_score("native-train")).encode("ascii"),
                image_png=train_image,
            ),
            NativePolyV2ArtifactInput(
                family_id="fam-validation",
                split=DatasetSplit.VALIDATION,
                target_json=serialize_polyphonic_score(_score("native-validation")).encode("ascii"),
                image_png=validation_image,
            ),
        )
        build = build_native_poly_v2_dataset(
            artifacts,
            sealed_test_samples=(_sealed_test_sample(),),
        )
        root = persist_native_poly_v2_dataset(build, parent / "native-v2")
        loaded = load_and_verify_native_poly_v2_dataset(root)

        train_xml = make_musicxml(8801)
        validation_xml = make_musicxml(8802)
        train_source = b"b8i-synthetic-train-source"
        validation_source = b"b8i-synthetic-validation-source"

        train_quarantine = make_quarantined_sample(
            family="fam-train",
            split=RealDataSplit.TRAIN,
            source_bytes=train_source,
            image_bytes=train_image,
            musicxml_bytes=train_xml,
        )
        validation_quarantine = make_quarantined_sample(
            family="fam-validation",
            split=RealDataSplit.VALIDATION,
            source_bytes=validation_source,
            image_bytes=validation_image,
            musicxml_bytes=validation_xml,
        )
        train_receipt = validate_quarantined_sample_bytes(
            train_quarantine,
            source_document_bytes=train_source,
            training_image_png_bytes=train_image,
            musicxml_bytes=train_xml,
        )
        validation_receipt = validate_quarantined_sample_bytes(
            validation_quarantine,
            source_document_bytes=validation_source,
            training_image_png_bytes=validation_image,
            musicxml_bytes=validation_xml,
        )
        train_admitted = replace(
            train_quarantine,
            admission_state=AdmissionState.ADMITTED,
        )
        validation_admitted = replace(
            validation_quarantine,
            admission_state=AdmissionState.ADMITTED,
        )
        manifest = RealDataManifest(
            dataset_name="b8i-fixture",
            dataset_version="v1",
            samples=(train_admitted, validation_admitted),
            sealed_test_manifest_sha256="f" * 64,
        )

        native_train = next(
            item
            for item in loaded.build.manifest.samples
            if item.split is DatasetSplit.TRAIN
        )
        native_validation = next(
            item
            for item in loaded.build.manifest.samples
            if item.split is DatasetSplit.VALIDATION
        )
        bindings = (
            NativePolyV2RealDataBinding(
                native_sample_id=native_train.sample_id,
                real_sample_id=train_admitted.sample_id,
                v2_conversion_profile_sha256="d" * 64,
                v2_target_review_evidence_sha256="e" * 64,
            ),
            NativePolyV2RealDataBinding(
                native_sample_id=native_validation.sample_id,
                real_sample_id=validation_admitted.sample_id,
                v2_conversion_profile_sha256="d" * 64,
                v2_target_review_evidence_sha256="e" * 64,
            ),
        )
        return (
            loaded,
            manifest,
            (train_receipt, validation_receipt),
            bindings,
        )

    def test_complete_real_lineage_emits_deterministic_training_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loaded, manifest, receipts, bindings = self._fixture(Path(directory))
            first = admit_native_poly_v2_real_corpus(
                loaded=loaded,
                real_manifest=manifest,
                byte_receipts=receipts,
                bindings=bindings,
            )
            second = admit_native_poly_v2_real_corpus(
                loaded=loaded,
                real_manifest=manifest,
                byte_receipts=reversed(receipts),
                bindings=reversed(bindings),
            )

            self.assertEqual(first.fingerprint(), second.fingerprint())
            self.assertEqual(first.dataset_manifest_sha256, loaded.build.manifest_sha256)
            self.assertEqual(first.dataset_build_id, loaded.build.build_id)
            self.assertEqual(first.b8a_preflight_receipt_sha256, loaded.receipt.fingerprint())
            self.assertEqual(first.sealed_real_test_manifest_sha256, "f" * 64)
            self.assertEqual(len(first.train_native_sample_ids), 1)
            self.assertEqual(len(first.validation_native_sample_ids), 1)
            self.assertEqual(len(first.admitted_real_sample_ids), 2)
            self.assertTrue(first.quality_training_eligible)
            self.assertFalse(first.test_artifact_bytes_accessed)
            self.assertFalse(first.production_authority)
            self.assertFalse(first.commercial_use_authority)

    def test_missing_binding_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loaded, manifest, receipts, bindings = self._fixture(Path(directory))
            with self.assertRaisesRegex(
                NativePolyV2RealCorpusAdmissionError,
                "exact Native V2 development population",
            ):
                admit_native_poly_v2_real_corpus(
                    loaded=loaded,
                    real_manifest=manifest,
                    byte_receipts=receipts,
                    bindings=bindings[:1],
                )

    def test_swapped_real_sources_fail_split_or_family_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loaded, manifest, receipts, bindings = self._fixture(Path(directory))
            swapped = (
                replace(bindings[0], real_sample_id=bindings[1].real_sample_id),
                replace(bindings[1], real_sample_id=bindings[0].real_sample_id),
            )
            with self.assertRaisesRegex(
                NativePolyV2RealCorpusAdmissionError,
                "split mismatch|family mismatch",
            ):
                admit_native_poly_v2_real_corpus(
                    loaded=loaded,
                    real_manifest=manifest,
                    byte_receipts=receipts,
                    bindings=swapped,
                )

    def test_teacher_correction_without_explicit_permission_is_rejected_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loaded, manifest, receipts, bindings = self._fixture(Path(directory))
            unsafe_train = replace(
                manifest.samples[0],
                origin=RealDataOrigin.TEACHER_CORRECTION,
            )
            unsafe_manifest = replace(
                manifest,
                samples=(unsafe_train, manifest.samples[1]),
            )
            with self.assertRaisesRegex(
                NativePolyV2RealCorpusAdmissionError,
                "development handoff is not admitted",
            ):
                admit_native_poly_v2_real_corpus(
                    loaded=loaded,
                    real_manifest=unsafe_manifest,
                    byte_receipts=receipts,
                    bindings=bindings,
                )

    def test_binding_requires_exact_hash_evidence(self) -> None:
        with self.assertRaises(NativePolyV2RealCorpusAdmissionError):
            NativePolyV2RealDataBinding(
                native_sample_id="a" * 64,
                real_sample_id="b" * 64,
                v2_conversion_profile_sha256="not-a-sha",
                v2_target_review_evidence_sha256="c" * 64,
            )


if __name__ == "__main__":
    unittest.main()
