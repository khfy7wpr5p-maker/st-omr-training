from __future__ import annotations

from dataclasses import replace
import unittest

from st_omr_training.poly_v2_ossq_rights_evidence_batch1 import B8M_BATCH1_EVIDENCE
from st_omr_training.poly_v2_ossq_source_byte_pin import (
    B8N_EXPECTED_SCORE_COUNT,
    B8N_EXPECTED_SOURCE_COUNT,
    OssqSourceBytePinError,
    fetch_and_pin_b8m_sources,
    pin_source_payloads,
    receipt_from_json,
    receipt_to_json,
    verify_live_receipt,
)


def payloads() -> dict[str, bytes]:
    return {
        evidence.imslp_id: (
            b"%PDF-1.7\n"
            + f"fixture:{evidence.imslp_id}:{','.join(evidence.score_ids)}".encode("ascii")
            + b"\n%%EOF\n"
        )
        for evidence in B8M_BATCH1_EVIDENCE
    }


class OssqSourceBytePinTests(unittest.TestCase):
    def test_exact_b8m_population_emits_hash_only_receipt(self) -> None:
        receipt = pin_source_payloads(payloads())
        self.assertEqual(len(receipt.pins), B8N_EXPECTED_SOURCE_COUNT)
        self.assertEqual(
            len({score for pin in receipt.pins for score in pin.score_ids}),
            B8N_EXPECTED_SCORE_COUNT,
        )
        self.assertFalse(receipt.raw_source_bytes_persisted)
        self.assertFalse(receipt.raw_source_bytes_redistributed)
        self.assertFalse(receipt.stage8_admission_authority)
        self.assertFalse(receipt.test_artifact_bytes_accessed)
        self.assertFalse(receipt.production_authority)
        self.assertFalse(receipt.commercial_use_authority)
        for pin in receipt.pins:
            self.assertEqual(len(pin.source_sha256), 64)
            self.assertGreater(pin.byte_count, 5)

    def test_non_pdf_or_population_drift_fails_closed(self) -> None:
        values = payloads()
        first = next(iter(values))
        values[first] = b"<html>bot check</html>"
        with self.assertRaisesRegex(OssqSourceBytePinError, "not a PDF"):
            pin_source_payloads(values)

        values = payloads()
        values.pop(first)
        with self.assertRaisesRegex(OssqSourceBytePinError, "population differs"):
            pin_source_payloads(values)

    def test_fetcher_is_called_only_for_exact_reviewed_urls(self) -> None:
        seen: list[str] = []
        by_url = {
            evidence.source_pdf_url: payloads()[evidence.imslp_id]
            for evidence in B8M_BATCH1_EVIDENCE
        }

        def fake_fetch(url: str) -> bytes:
            seen.append(url)
            return by_url[url]

        receipt = fetch_and_pin_b8m_sources(fetcher=fake_fetch)
        self.assertEqual(seen, [item.source_pdf_url for item in B8M_BATCH1_EVIDENCE])
        self.assertEqual(len(receipt.pins), 5)

    def test_receipt_json_round_trip_is_exact(self) -> None:
        receipt = pin_source_payloads(payloads())
        text = receipt_to_json(receipt)
        restored = receipt_from_json(text)
        self.assertEqual(receipt_to_json(restored), text)
        self.assertEqual(restored.receipt_sha256, receipt.receipt_sha256)

    def test_receipt_tampering_fails_integrity(self) -> None:
        receipt = pin_source_payloads(payloads())
        with self.assertRaisesRegex(OssqSourceBytePinError, "receipt_sha256"):
            replace(receipt, receipt_sha256="0" * 64)

    def test_live_verification_detects_changed_bytes(self) -> None:
        expected = pin_source_payloads(payloads())
        changed = payloads()
        first = next(iter(changed))
        changed[first] += b"changed"
        actual = pin_source_payloads(changed)
        with self.assertRaisesRegex(OssqSourceBytePinError, "live OSSQ source bytes differ"):
            verify_live_receipt(expected, actual)


if __name__ == "__main__":
    unittest.main()
