from __future__ import annotations

import hashlib
from io import BytesIO
import unittest
from unittest.mock import patch

from PIL import Image

from st_omr_training.real_data_contract import (
    AdmissionState,
    PairingState,
    RealDataOrigin,
    RealDataSample,
    RealDataSplit,
    ReviewState,
    RightsBasis,
    real_data_sample_id,
)
from st_omr_training.real_data_intake import (
    RealDataIntakeError,
    validate_quarantined_sample_bytes,
)


def _hex(ch: str) -> str:
    return ch * 64


def _png_bytes() -> bytes:
    image = Image.new("L", (8, 8), 255)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _sample(*, source: bytes, image: bytes, musicxml: bytes) -> RealDataSample:
    source_sha = hashlib.sha256(source).hexdigest()
    image_sha = hashlib.sha256(image).hexdigest()
    musicxml_sha = hashlib.sha256(musicxml).hexdigest()
    semantic = _hex("d")
    sample_id = real_data_sample_id(
        family_id="size-guard-family",
        page_number=1,
        source_document_sha256=source_sha,
        image_sha256=image_sha,
        musicxml_sha256=musicxml_sha,
        semantic_fingerprint=semantic,
    )
    return RealDataSample(
        sample_id=sample_id,
        family_id="size-guard-family",
        split=RealDataSplit.TRAIN,
        page_number=1,
        origin=RealDataOrigin.CURATED,
        rights_basis=RightsBasis.OPEN_LICENSE,
        source_document_sha256=source_sha,
        image_sha256=image_sha,
        musicxml_sha256=musicxml_sha,
        semantic_fingerprint=semantic,
        provenance_evidence_sha256=_hex("a"),
        rights_evidence_sha256=_hex("b"),
        pairing_evidence_sha256=_hex("c"),
        explicit_training_permission_sha256=None,
        privacy_review_evidence_sha256=None,
        rights_review=ReviewState.APPROVED,
        pairing_review=PairingState.VERIFIED,
        admission_state=AdmissionState.QUARANTINED,
    )


class RealDataIntakeSizeGuardTests(unittest.TestCase):
    def test_oversized_image_is_rejected_before_digest(self) -> None:
        source = b"synthetic-source"
        image = _png_bytes()
        musicxml = b"synthetic-not-reached"
        sample = _sample(source=source, image=image, musicxml=musicxml)
        original_sha256 = hashlib.sha256

        def guarded_sha256(data: bytes):
            if data is image:
                raise AssertionError("oversized image reached sha256")
            return original_sha256(data)

        with patch(
            "st_omr_training.real_data_intake.MAX_TRAINING_IMAGE_BYTES",
            len(image) - 1,
        ), patch(
            "st_omr_training.real_data_intake.sha256",
            side_effect=guarded_sha256,
        ):
            with self.assertRaisesRegex(RealDataIntakeError, "training image exceeds"):
                validate_quarantined_sample_bytes(
                    sample,
                    source_document_bytes=source,
                    training_image_png_bytes=image,
                    musicxml_bytes=musicxml,
                )

    def test_oversized_musicxml_is_rejected_before_digest(self) -> None:
        source = b"synthetic-source"
        image = _png_bytes()
        musicxml = b"oversized-synthetic-musicxml"
        sample = _sample(source=source, image=image, musicxml=musicxml)
        original_sha256 = hashlib.sha256

        def guarded_sha256(data: bytes):
            if data is musicxml:
                raise AssertionError("oversized MusicXML reached sha256")
            return original_sha256(data)

        with patch(
            "st_omr_training.real_data_intake.MAX_MUSICXML_BYTES",
            len(musicxml) - 1,
        ), patch(
            "st_omr_training.real_data_intake.sha256",
            side_effect=guarded_sha256,
        ):
            with self.assertRaisesRegex(RealDataIntakeError, "MusicXML exceeds"):
                validate_quarantined_sample_bytes(
                    sample,
                    source_document_bytes=source,
                    training_image_png_bytes=image,
                    musicxml_bytes=musicxml,
                )


if __name__ == "__main__":
    unittest.main()
