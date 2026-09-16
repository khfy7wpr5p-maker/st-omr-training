from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

from st_omr_training.poly_v2_ossq_pairing_preflight import B8O_PAIRING_SOURCE_SPECS
from st_omr_training.poly_v2_ossq_system_pair_materialization import (
    B8P_BLOCKED_SCORE_IDS,
    B8P_EXPECTED_B8N_RECEIPT_SHA256,
    B8P_EXPECTED_B8O_RECEIPT_SHA256,
    B8P_READY_SCORE_IDS,
    OssqSystemPairMaterializationError,
    build_b8p_materialization_receipt,
    receipt_to_json,
)


def fake_png(width: int = 16, height: int = 8) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + b"\x08\x00\x00\x00\x00"


def source_hashes() -> dict[str, str]:
    return {score_id: (hex(index + 1)[2:] * 64)[:64] for index, score_id in enumerate(B8P_READY_SCORE_IDS)}


def populate(root: Path) -> None:
    specs = {item.score_id: item for item in B8O_PAIRING_SOURCE_SPECS}
    for index, score_id in enumerate(B8P_READY_SCORE_IDS, start=1):
        score_dir = root / "scores" / specs[score_id].work_path
        image_root = score_dir / "images" / "scanned"
        image_dir = image_root / "systemwise"
        xml_dir = score_dir / "musicxml" / "scanned" / "systemwise"
        image_dir.mkdir(parents=True, exist_ok=True)
        xml_dir.mkdir(parents=True, exist_ok=True)
        (image_root / f"sq{score_id}_yolo_infos.yaml").write_text("model: pinned\n", encoding="utf-8")
        segment = f"sq{score_id}:0001:0001"
        (image_dir / f"{segment}.png").write_bytes(fake_png(10 + index, 20 + index))
        (xml_dir / f"{segment}.musicxml").write_bytes(b'<score-partwise version="4.0"></score-partwise>')


class B8PMaterializationTests(unittest.TestCase):
    def build(self, root: Path):
        return build_b8p_materialization_receipt(
            dataset_root=root,
            b8n_source_sha256_by_score=source_hashes(),
            b8n_receipt_sha256=B8P_EXPECTED_B8N_RECEIPT_SHA256,
            b8o_receipt_sha256=B8P_EXPECTED_B8O_RECEIPT_SHA256,
        )

    def test_exact_population_and_authority_boundary(self) -> None:
        self.assertEqual(len(B8P_READY_SCORE_IDS), 7)
        self.assertEqual(B8P_BLOCKED_SCORE_IDS, ("7397765",))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            populate(root)
            receipt = self.build(root)
        self.assertEqual(len(receipt.pairs), 7)
        self.assertEqual(dict(receipt.pair_count_by_score), {score_id: 1 for score_id in B8P_READY_SCORE_IDS})
        self.assertFalse(receipt.independent_pairing_review_authority)
        self.assertFalse(receipt.stage8_admission_authority)
        self.assertFalse(receipt.train_validation_assignment_authority)
        self.assertFalse(receipt.test_artifact_bytes_accessed)
        self.assertFalse(receipt.production_authority)
        self.assertFalse(receipt.commercial_use_authority)
        self.assertNotIn("7397765", {pair.score_id for pair in receipt.pairs})
        self.assertEqual(len(receipt.receipt_sha256), 64)
        self.assertIn(receipt.receipt_sha256, receipt_to_json(receipt))

    def test_source_population_must_be_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            populate(root)
            hashes = source_hashes()
            hashes.pop(B8P_READY_SCORE_IDS[0])
            with self.assertRaisesRegex(OssqSystemPairMaterializationError, "exact B8P READY population"):
                build_b8p_materialization_receipt(
                    dataset_root=root,
                    b8n_source_sha256_by_score=hashes,
                    b8n_receipt_sha256=B8P_EXPECTED_B8N_RECEIPT_SHA256,
                    b8o_receipt_sha256=B8P_EXPECTED_B8O_RECEIPT_SHA256,
                )

    def test_segment_population_mismatch_fails_closed(self) -> None:
        specs = {item.score_id: item for item in B8O_PAIRING_SOURCE_SPECS}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            populate(root)
            score_id = B8P_READY_SCORE_IDS[0]
            score_dir = root / "scores" / specs[score_id].work_path
            extra = score_dir / "musicxml" / "scanned" / "systemwise" / f"sq{score_id}:0001:0002.musicxml"
            extra.write_bytes(b'<score-partwise version="4.0"></score-partwise>')
            with self.assertRaisesRegex(OssqSystemPairMaterializationError, "population mismatch"):
                self.build(root)

    def test_non_png_fails_closed(self) -> None:
        specs = {item.score_id: item for item in B8O_PAIRING_SOURCE_SPECS}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            populate(root)
            score_id = B8P_READY_SCORE_IDS[0]
            path = root / "scores" / specs[score_id].work_path / "images" / "scanned" / "systemwise" / f"sq{score_id}:0001:0001.png"
            path.write_bytes(b"not a png")
            with self.assertRaisesRegex(OssqSystemPairMaterializationError, "valid PNG"):
                self.build(root)

    def test_wrong_upstream_receipt_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            populate(root)
            with self.assertRaisesRegex(OssqSystemPairMaterializationError, "B8O receipt identity"):
                build_b8p_materialization_receipt(
                    dataset_root=root,
                    b8n_source_sha256_by_score=source_hashes(),
                    b8n_receipt_sha256=B8P_EXPECTED_B8N_RECEIPT_SHA256,
                    b8o_receipt_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
