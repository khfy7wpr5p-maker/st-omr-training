from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from st_omr_training.poly_v2_ossq_pair_review_admission import (
    B8Q_EXPECTED_B8P_RECEIPT_SHA256,
    B8Q_EXPECTED_PAIR_COUNT,
    OssqPairReviewError,
    PairReviewDecision,
    PairReviewMethod,
    OssqPairReviewRecord,
    apply_independent_pair_reviews,
    build_unreviewed_b8q_receipt,
    review_receipt_to_json,
    verified_review_records,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
B8P_EVIDENCE = REPO_ROOT / "evidence" / "ossq_b8p_system_pair_materialization.json"


def b8p_payload() -> dict[str, object]:
    return json.loads(B8P_EVIDENCE.read_text(encoding="ascii"))


class B8QPairReviewAdmissionTests(unittest.TestCase):
    def test_real_b8p_evidence_starts_entirely_review_required(self) -> None:
        receipt = build_unreviewed_b8q_receipt(b8p_payload())
        self.assertEqual(receipt.b8p_receipt_sha256, B8Q_EXPECTED_B8P_RECEIPT_SHA256)
        self.assertEqual(len(receipt.records), B8Q_EXPECTED_PAIR_COUNT)
        self.assertEqual(
            dict(receipt.review_count_by_decision),
            {"verified": 0, "review-required": 412, "blocked": 0},
        )
        self.assertTrue(all(item.decision is PairReviewDecision.REVIEW_REQUIRED for item in receipt.records))
        self.assertFalse(receipt.independent_pairing_review_authority)
        self.assertFalse(receipt.stage8_admission_authority)
        self.assertFalse(receipt.train_validation_assignment_authority)
        self.assertFalse(receipt.test_artifact_bytes_accessed)
        self.assertFalse(receipt.production_authority)
        self.assertFalse(receipt.commercial_use_authority)
        self.assertEqual(verified_review_records(receipt), ())
        self.assertIn(receipt.receipt_sha256, review_receipt_to_json(receipt))

    def test_b8p_identity_must_be_exact(self) -> None:
        payload = b8p_payload()
        payload["receipt_sha256"] = "0" * 64
        with self.assertRaisesRegex(OssqPairReviewError, "identity differs"):
            build_unreviewed_b8q_receipt(payload)

    def test_verified_cannot_be_forged_without_independent_evidence(self) -> None:
        base = build_unreviewed_b8q_receipt(b8p_payload()).records[0]
        with self.assertRaisesRegex(OssqPairReviewError, "require review evidence"):
            OssqPairReviewRecord(
                score_id=base.score_id,
                segment_id=base.segment_id,
                image_sha256=base.image_sha256,
                musicxml_sha256=base.musicxml_sha256,
                decision=PairReviewDecision.VERIFIED,
                method=PairReviewMethod.INDEPENDENT_VISUAL,
                review_evidence_sha256=None,
                reviewer_identity_sha256="2" * 64,
                reason_code="independent-pair-review-passed",
            )

    def test_one_independent_verified_pair_does_not_grant_stage8_authority(self) -> None:
        payload = b8p_payload()
        unreviewed = build_unreviewed_b8q_receipt(payload)
        first = unreviewed.records[0]
        verified = replace(
            first,
            decision=PairReviewDecision.VERIFIED,
            method=PairReviewMethod.INDEPENDENT_VISUAL,
            review_evidence_sha256="1" * 64,
            reviewer_identity_sha256="2" * 64,
            reason_code="independent-pair-review-passed",
        )
        reviews = (verified, *unreviewed.records[1:])
        receipt = apply_independent_pair_reviews(b8p_payload=payload, reviews=reviews)
        self.assertEqual(
            dict(receipt.review_count_by_decision),
            {"verified": 1, "review-required": 411, "blocked": 0},
        )
        self.assertTrue(receipt.independent_pairing_review_authority)
        self.assertFalse(receipt.stage8_admission_authority)
        self.assertFalse(receipt.train_validation_assignment_authority)
        self.assertEqual(verified_review_records(receipt), (verified,))

    def test_review_hashes_must_match_exact_b8p_pair(self) -> None:
        payload = b8p_payload()
        unreviewed = build_unreviewed_b8q_receipt(payload)
        forged = replace(unreviewed.records[0], image_sha256="f" * 64)
        with self.assertRaisesRegex(OssqPairReviewError, "identity differs"):
            apply_independent_pair_reviews(
                b8p_payload=payload,
                reviews=(forged, *unreviewed.records[1:]),
            )

    def test_partial_review_population_fails_closed(self) -> None:
        payload = b8p_payload()
        unreviewed = build_unreviewed_b8q_receipt(payload)
        with self.assertRaisesRegex(OssqPairReviewError, "exact 412-pair"):
            apply_independent_pair_reviews(
                b8p_payload=payload,
                reviews=unreviewed.records[:-1],
            )


if __name__ == "__main__":
    unittest.main()
