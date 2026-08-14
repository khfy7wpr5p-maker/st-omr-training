from __future__ import annotations

from hashlib import sha256
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from st_omr_training.dataset_manifest import DatasetSplit
from st_omr_training.stage7d2_execution import (
    Stage7D2ExecutionError,
    _load_development_refs,
    _verify_d1_receipt,
)
from st_omr_training.synthetic_curriculum_corpus_gate import (
    SyntheticCurriculumCorpusReceipt,
)


def canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def accepted_receipt() -> SyntheticCurriculumCorpusReceipt:
    return SyntheticCurriculumCorpusReceipt(
        source_commit="adc8139539d3c8cd6a2e3ee4ce4de6db4dcfeb90",
        build_id="d9320e362f162cd2ace2a830a7b93e0c21ceba2d51a4e95ef1c7a9b11a108352",
        config_fingerprint="154bf1c3e6dfe4e6db096f8b668f29df0623cfd38352b89a04d295764c7458cb",
        manifest_sha256="44a963cd7dbc612fa29c2953ea8b2c8776d89ce470074e8f8b3fe25c6e165f34",
        transport_sha256="4a9f3bb337ef99386081dff29c4c1fc3047dc3ada4db13c93b6254e680918e2b",
        transport_archive="st-omr-synthetic-curriculum-v1-d9320e362f162cd2.tar.gz",
        archive_size_bytes=494006801,
        sample_count=1536,
        target_count=512,
        image_count=1536,
        family_split_counts={"test": 51, "train": 410, "validation": 51},
        sample_split_counts={"test": 153, "train": 1230, "validation": 153},
        target_bytes_total=3506839,
        image_bytes_total=494937881,
        artifact_binding_sha256="e603b945c6dc60cf7e618ae28a7734dee97cf0e05a81891479107b18a87af540",
    )


class Stage7D2ExecutionTests(unittest.TestCase):
    def test_accepted_d1_receipt_is_required_exactly(self) -> None:
        _verify_d1_receipt(accepted_receipt())
        receipt = accepted_receipt()
        object.__setattr__(receipt, "artifact_binding_sha256", "0" * 64)
        with self.assertRaises(Stage7D2ExecutionError):
            _verify_d1_receipt(receipt)

    def test_loader_skips_test_before_artifact_path_or_byte_access(self) -> None:
        target_bytes = b"target"
        image_bytes = b"image"
        target_sha = sha256(target_bytes).hexdigest()
        image_sha = sha256(image_bytes).hexdigest()

        samples: list[dict[str, object]] = []
        for index in range(1230):
            samples.append(
                {
                    "split": "train",
                    "sample_id": sha256(f"train-{index}".encode()).hexdigest(),
                    "family_id": f"train-family-{index % 410}",
                    "png_sha256": image_sha,
                    "source_musicxml_sha256": target_sha,
                    "width": 512,
                    "height": 64,
                }
            )
        for index in range(153):
            samples.append(
                {
                    "split": "validation",
                    "sample_id": sha256(f"validation-{index}".encode()).hexdigest(),
                    "family_id": f"validation-family-{index % 51}",
                    "png_sha256": image_sha,
                    "source_musicxml_sha256": target_sha,
                    "width": 512,
                    "height": 64,
                }
            )
        for _index in range(153):
            # Deliberately invalid hash metadata that remains valid JSON. If D2
            # validates/derives TEST artifact identity before skipping TEST, this
            # fixture fails at the loader boundary.
            samples.append({"split": "test", "png_sha256": "not-a-sha"})

        manifest = {"samples": samples}
        manifest_bytes = canonical(manifest)
        build = {
            "build_id": "d9320e362f162cd2ace2a830a7b93e0c21ceba2d51a4e95ef1c7a9b11a108352",
            "config_fingerprint": "154bf1c3e6dfe4e6db096f8b668f29df0623cfd38352b89a04d295764c7458cb",
            "manifest_sha256": sha256(manifest_bytes).hexdigest(),
            "sample_count": 1536,
            "target_count": 512,
            "image_count": 1536,
        }

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "targets").mkdir()
            (root / "images").mkdir()
            (root / "manifest.json").write_bytes(manifest_bytes)
            (root / "build.json").write_bytes(canonical(build))
            (root / "targets" / f"{target_sha}.musicxml").write_bytes(target_bytes)
            (root / "images" / f"{image_sha}.png").write_bytes(image_bytes)

            class Tokenized:
                token_ids = (1, 2)

            with (
                patch(
                    "st_omr_training.stage7d2_execution.EXPECTED_MANIFEST_SHA256",
                    sha256(manifest_bytes).hexdigest(),
                ),
                patch(
                    "st_omr_training.stage7d2_execution.tokenize_musicxml",
                    return_value=Tokenized(),
                ),
                patch(
                    "st_omr_training.stage7d2_execution.preprocess_grayscale_png"
                ),
            ):
                train, validation = _load_development_refs(root)

        self.assertEqual(len(train), 1230)
        self.assertEqual(len(validation), 153)
        self.assertTrue(all(item.split is DatasetSplit.TRAIN for item in train))
        self.assertTrue(all(item.split is DatasetSplit.VALIDATION for item in validation))
        self.assertFalse(any(item.split is DatasetSplit.TEST for item in train + validation))


if __name__ == "__main__":
    unittest.main()
