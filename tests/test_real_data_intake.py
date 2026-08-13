from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import unittest

from PIL import Image

from st_omr_training.generator import GeneratorConfig, generate_score
from st_omr_training.musicxml_writer import write_musicxml
from st_omr_training.real_data_contract import (
    AdmissionState,
    PairingState,
    RealDataManifest,
    RealDataOrigin,
    RealDataSample,
    RealDataSplit,
    ReviewState,
    RightsBasis,
    SealedTestAccessError,
    real_data_sample_id,
)
from st_omr_training.real_data_intake import (
    NEAR_DUPLICATE_MAX_HAMMING_DISTANCE,
    RealDataIntakeError,
    find_near_duplicate_candidates,
    intake_policy_fingerprint,
    semantic_fingerprint_from_musicxml,
    validate_byte_receipt,
    validate_quarantined_sample_bytes,
    validate_stage8_development_handoff,
)


def h(ch: str) -> str:
    return ch * 64


def make_png(pattern: str, *, compress_level: int = 6, mode: str = "L") -> bytes:
    image = Image.new(mode, (64, 64), 255 if mode == "L" else (255, 255, 255))
    pixels = image.load()
    for y in range(64):
        for x in range(64):
            if pattern == "vertical":
                value = 0 if x < 32 else 255
            elif pattern == "horizontal":
                value = 0 if y < 32 else 255
            else:
                raise ValueError("unknown synthetic pattern")
            pixels[x, y] = value if mode == "L" else (value, value, value)
    output = BytesIO()
    image.save(output, format="PNG", compress_level=compress_level)
    return output.getvalue()


def make_musicxml(seed: int) -> bytes:
    return write_musicxml(generate_score(GeneratorConfig(measure_count=2), seed))


def make_quarantined_sample(
    *,
    family: str,
    split: RealDataSplit,
    source_bytes: bytes,
    image_bytes: bytes,
    musicxml_bytes: bytes,
) -> RealDataSample:
    source_hash = sha256(source_bytes).hexdigest()
    image_hash = sha256(image_bytes).hexdigest()
    musicxml_hash = sha256(musicxml_bytes).hexdigest()
    semantic = semantic_fingerprint_from_musicxml(musicxml_bytes)
    sample_id = real_data_sample_id(
        family_id=family,
        page_number=1,
        source_document_sha256=source_hash,
        image_sha256=image_hash,
        musicxml_sha256=musicxml_hash,
        semantic_fingerprint=semantic,
    )
    return RealDataSample(
        sample_id=sample_id,
        family_id=family,
        split=split,
        page_number=1,
        origin=RealDataOrigin.CURATED,
        rights_basis=RightsBasis.OPEN_LICENSE,
        source_document_sha256=source_hash,
        image_sha256=image_hash,
        musicxml_sha256=musicxml_hash,
        semantic_fingerprint=semantic,
        provenance_evidence_sha256=h("a"),
        rights_evidence_sha256=h("b"),
        pairing_evidence_sha256=h("c"),
        explicit_training_permission_sha256=None,
        privacy_review_evidence_sha256=None,
        rights_review=ReviewState.APPROVED,
        pairing_review=PairingState.VERIFIED,
        admission_state=AdmissionState.QUARANTINED,
    )


class RealDataIntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.xml_train = make_musicxml(8101)
        cls.xml_validation = make_musicxml(8102)
        cls.source_train = b"synthetic-source-train"
        cls.source_validation = b"synthetic-source-validation"

    def test_policy_and_semantic_fingerprint_are_deterministic(self) -> None:
        self.assertEqual(intake_policy_fingerprint(), intake_policy_fingerprint())
        self.assertEqual(len(intake_policy_fingerprint()), 64)
        self.assertEqual(
            semantic_fingerprint_from_musicxml(self.xml_train),
            semantic_fingerprint_from_musicxml(self.xml_train),
        )

    def test_valid_quarantined_bytes_emit_deterministic_hash_only_receipt(self) -> None:
        image = make_png("vertical")
        sample = make_quarantined_sample(
            family="fam-train",
            split=RealDataSplit.TRAIN,
            source_bytes=self.source_train,
            image_bytes=image,
            musicxml_bytes=self.xml_train,
        )
        first = validate_quarantined_sample_bytes(
            sample,
            source_document_bytes=self.source_train,
            training_image_png_bytes=image,
            musicxml_bytes=self.xml_train,
        )
        second = validate_quarantined_sample_bytes(
            sample,
            source_document_bytes=self.source_train,
            training_image_png_bytes=image,
            musicxml_bytes=self.xml_train,
        )
        self.assertEqual(first, second)
        self.assertEqual((first.image_width, first.image_height), (64, 64))
        self.assertGreater(first.token_count, 0)
        self.assertEqual(len(first.receipt_sha256), 64)
        validate_byte_receipt(sample, first)

    def test_test_split_fails_before_any_byte_access(self) -> None:
        image = make_png("vertical")
        sample = make_quarantined_sample(
            family="fam-test",
            split=RealDataSplit.TEST,
            source_bytes=self.source_train,
            image_bytes=image,
            musicxml_bytes=self.xml_train,
        )
        with self.assertRaises(SealedTestAccessError):
            validate_quarantined_sample_bytes(
                sample,
                source_document_bytes=object(),
                training_image_png_bytes=object(),
                musicxml_bytes=object(),
            )

    def test_source_image_and_musicxml_hash_mismatch_are_rejected(self) -> None:
        image = make_png("vertical")
        other_image = make_png("horizontal")
        sample = make_quarantined_sample(
            family="fam-train",
            split=RealDataSplit.TRAIN,
            source_bytes=self.source_train,
            image_bytes=image,
            musicxml_bytes=self.xml_train,
        )
        cases = (
            (b"different-source", image, self.xml_train, "source_document_sha256"),
            (self.source_train, other_image, self.xml_train, "image_sha256"),
            (self.source_train, image, self.xml_validation, "musicxml_sha256"),
        )
        for source, png, xml, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RealDataIntakeError, message):
                    validate_quarantined_sample_bytes(
                        sample,
                        source_document_bytes=source,
                        training_image_png_bytes=png,
                        musicxml_bytes=xml,
                    )

    def test_non_grayscale_and_truncated_png_are_rejected(self) -> None:
        for image in (make_png("vertical", mode="RGB"), make_png("vertical")[:40]):
            sample = make_quarantined_sample(
                family="fam-image",
                split=RealDataSplit.TRAIN,
                source_bytes=self.source_train,
                image_bytes=image,
                musicxml_bytes=self.xml_train,
            )
            with self.assertRaises(RealDataIntakeError):
                validate_quarantined_sample_bytes(
                    sample,
                    source_document_bytes=self.source_train,
                    training_image_png_bytes=image,
                    musicxml_bytes=self.xml_train,
                )

    def test_invalid_musicxml_is_rejected_after_exact_hash_binding(self) -> None:
        image = make_png("vertical")
        invalid_xml = b"<not-musicxml/>"
        source_hash = sha256(self.source_train).hexdigest()
        image_hash = sha256(image).hexdigest()
        xml_hash = sha256(invalid_xml).hexdigest()
        semantic = h("d")
        sample = RealDataSample(
            sample_id=real_data_sample_id(
                family_id="fam-invalid",
                page_number=1,
                source_document_sha256=source_hash,
                image_sha256=image_hash,
                musicxml_sha256=xml_hash,
                semantic_fingerprint=semantic,
            ),
            family_id="fam-invalid",
            split=RealDataSplit.TRAIN,
            page_number=1,
            origin=RealDataOrigin.CURATED,
            rights_basis=RightsBasis.OPEN_LICENSE,
            source_document_sha256=source_hash,
            image_sha256=image_hash,
            musicxml_sha256=xml_hash,
            semantic_fingerprint=semantic,
            provenance_evidence_sha256=h("a"),
            rights_evidence_sha256=h("b"),
            pairing_evidence_sha256=h("c"),
            explicit_training_permission_sha256=None,
            privacy_review_evidence_sha256=None,
            rights_review=ReviewState.APPROVED,
            pairing_review=PairingState.VERIFIED,
            admission_state=AdmissionState.QUARANTINED,
        )
        with self.assertRaisesRegex(RealDataIntakeError, "MusicXML failed"):
            validate_quarantined_sample_bytes(
                sample,
                source_document_bytes=self.source_train,
                training_image_png_bytes=image,
                musicxml_bytes=invalid_xml,
            )

    def test_semantic_fingerprint_mismatch_is_rejected(self) -> None:
        image = make_png("vertical")
        original = make_quarantined_sample(
            family="fam-train",
            split=RealDataSplit.TRAIN,
            source_bytes=self.source_train,
            image_bytes=image,
            musicxml_bytes=self.xml_train,
        )
        wrong_semantic = h("0")
        tampered = replace(
            original,
            semantic_fingerprint=wrong_semantic,
            sample_id=real_data_sample_id(
                family_id=original.family_id,
                page_number=original.page_number,
                source_document_sha256=original.source_document_sha256,
                image_sha256=original.image_sha256,
                musicxml_sha256=original.musicxml_sha256,
                semantic_fingerprint=wrong_semantic,
            ),
        )
        with self.assertRaisesRegex(RealDataIntakeError, "semantic_fingerprint"):
            validate_quarantined_sample_bytes(
                tampered,
                source_document_bytes=self.source_train,
                training_image_png_bytes=image,
                musicxml_bytes=self.xml_train,
            )

    def test_perceptual_near_duplicate_detects_different_png_bytes(self) -> None:
        image_a = make_png("vertical", compress_level=1)
        image_b = make_png("vertical", compress_level=9)
        self.assertNotEqual(sha256(image_a).hexdigest(), sha256(image_b).hexdigest())
        sample_a = make_quarantined_sample(
            family="fam-a",
            split=RealDataSplit.TRAIN,
            source_bytes=b"source-a",
            image_bytes=image_a,
            musicxml_bytes=self.xml_train,
        )
        sample_b = make_quarantined_sample(
            family="fam-b",
            split=RealDataSplit.VALIDATION,
            source_bytes=b"source-b",
            image_bytes=image_b,
            musicxml_bytes=self.xml_validation,
        )
        receipts = (
            validate_quarantined_sample_bytes(
                sample_a,
                source_document_bytes=b"source-a",
                training_image_png_bytes=image_a,
                musicxml_bytes=self.xml_train,
            ),
            validate_quarantined_sample_bytes(
                sample_b,
                source_document_bytes=b"source-b",
                training_image_png_bytes=image_b,
                musicxml_bytes=self.xml_validation,
            ),
        )
        candidates = find_near_duplicate_candidates(receipts)
        self.assertEqual(len(candidates), 1)
        self.assertLessEqual(candidates[0].hamming_distance, NEAR_DUPLICATE_MAX_HAMMING_DISTANCE)

    def _handoff_fixture(self, *, near_duplicate: bool = False):
        train_image = make_png("vertical", compress_level=1)
        validation_image = make_png(
            "vertical" if near_duplicate else "horizontal",
            compress_level=9,
        )
        train = make_quarantined_sample(
            family="fam-train",
            split=RealDataSplit.TRAIN,
            source_bytes=self.source_train,
            image_bytes=train_image,
            musicxml_bytes=self.xml_train,
        )
        validation = make_quarantined_sample(
            family="fam-validation",
            split=RealDataSplit.VALIDATION,
            source_bytes=self.source_validation,
            image_bytes=validation_image,
            musicxml_bytes=self.xml_validation,
        )
        receipts = (
            validate_quarantined_sample_bytes(
                train,
                source_document_bytes=self.source_train,
                training_image_png_bytes=train_image,
                musicxml_bytes=self.xml_train,
            ),
            validate_quarantined_sample_bytes(
                validation,
                source_document_bytes=self.source_validation,
                training_image_png_bytes=validation_image,
                musicxml_bytes=self.xml_validation,
            ),
        )
        manifest = RealDataManifest(
            dataset_name="stage8-real-v1",
            dataset_version="v1",
            samples=(
                replace(train, admission_state=AdmissionState.ADMITTED),
                replace(validation, admission_state=AdmissionState.ADMITTED),
            ),
            sealed_test_manifest_sha256=h("9"),
        )
        return manifest, receipts

    def test_valid_development_handoff_requires_matching_receipts(self) -> None:
        manifest, receipts = self._handoff_fixture()
        self.assertEqual(validate_stage8_development_handoff(manifest, receipts), ())
        with self.assertRaisesRegex(RealDataIntakeError, "exactly one byte receipt"):
            validate_stage8_development_handoff(manifest, receipts[:1])

    def test_cross_family_perceptual_near_duplicate_blocks_handoff(self) -> None:
        manifest, receipts = self._handoff_fixture(near_duplicate=True)
        with self.assertRaisesRegex(RealDataIntakeError, "different families"):
            validate_stage8_development_handoff(manifest, receipts)


if __name__ == "__main__":
    unittest.main()
