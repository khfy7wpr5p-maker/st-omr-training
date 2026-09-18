from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from st_omr_training.poly_v2_ossq_b8r_review_bridge import (
    B8R_TO_B8Q_EXPECTED_B8R_RECEIPT_SHA256,
    B8R_TO_B8Q_EXPECTED_POLICY_FINGERPRINT,
    B8R_TO_B8Q_EXPECTED_REVIEWER_IDENTITY_SHA256,
    OssqB8RReviewBridgeError,
    build_b8q_from_b8r,
)
from st_omr_training.poly_v2_ossq_pair_review_admission import (
    PairReviewDecision,
    PairReviewMethod,
    verified_review_records,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
B8P_EVIDENCE = REPO_ROOT / "evidence" / "ossq_b8p_system_pair_materialization.json"
B8R_EVIDENCE = REPO_ROOT / "evidence" / "ossq_b8r_cross_render_audit.json"


def b8p_payload() -> dict[str, object]:
    return json.loads(B8P_EVIDENCE.read_text(encoding="ascii"))


def b8r_payload() -> dict[str, object]:
    return json.loads(B8R_EVIDENCE.read_text(encoding="ascii"))


class B8RToB8QReviewBridgeTests(unittest.TestCase):
    def test_real_b8r_receipt_promotes_exactly_14_and_keeps_398_review_required(self) -> None:
        receipt = build_b8q_from_b8r(b8p_payload=b8p_payload(), b8r_payload=b8r_payload())

        self.assertEqual(
            dict(receipt.review_count_by_decision),
            {"verified": 14, "review-required": 398, "blocked": 0},
        )
        verified = verified_review_records(receipt)
        self.assertEqual(len(verified), 14)
        self.assertTrue(
            all(item.method is PairReviewMethod.INDEPENDENT_CROSS_RENDER for item in verified)
        )
        self.assertTrue(
            all(item.reason_code == "independent-pair-review-passed" for item in verified)
        )
        self.assertTrue(
            all(
                item.reviewer_identity_sha256
                == B8R_TO_B8Q_EXPECTED_REVIEWER_IDENTITY_SHA256
                for item in verified
            )
        )
        self.assertTrue(receipt.independent_pairing_review_authority)
        self.assertFalse(receipt.stage8_admission_authority)
        self.assertFalse(receipt.train_validation_assignment_authority)
        self.assertFalse(receipt.test_artifact_bytes_accessed)
        self.assertFalse(receipt.production_authority)
        self.assertFalse(receipt.commercial_use_authority)

    def test_b8r_frozen_receipt_identity_is_required(self) -> None:
        payload = b8r_payload()
        payload["receipt_sha256"] = "0" * 64
        with self.assertRaisesRegex(OssqB8RReviewBridgeError, "receipt identity"):
            build_b8q_from_b8r(b8p_payload=b8p_payload(), b8r_payload=payload)

    def test_b8r_policy_fingerprint_drift_fails_closed(self) -> None:
        payload = b8r_payload()
        self.assertEqual(payload["policy_fingerprint"], B8R_TO_B8Q_EXPECTED_POLICY_FINGERPRINT)
        payload["policy_fingerprint"] = "1" * 64
        with self.assertRaisesRegex(OssqB8RReviewBridgeError, "policy fingerprint"):
            build_b8q_from_b8r(b8p_payload=b8p_payload(), b8r_payload=payload)

    def test_b8r_reviewer_identity_drift_fails_closed(self) -> None:
        payload = b8r_payload()
        self.assertEqual(
            payload["reviewer_identity_sha256"],
            B8R_TO_B8Q_EXPECTED_REVIEWER_IDENTITY_SHA256,
        )
        payload["reviewer_identity_sha256"] = "2" * 64
        with self.assertRaisesRegex(OssqB8RReviewBridgeError, "reviewer identity"):
            build_b8q_from_b8r(b8p_payload=b8p_payload(), b8r_payload=payload)

    def test_b8r_image_identity_must_match_b8p(self) -> None:
        payload = b8r_payload()
        payload["observations"][0]["image_sha256"] = "3" * 64
        with self.assertRaisesRegex(OssqB8RReviewBridgeError, "exact B8P pair identity"):
            build_b8q_from_b8r(b8p_payload=b8p_payload(), b8r_payload=payload)

    def test_b8r_musicxml_identity_must_match_b8p(self) -> None:
        payload = b8r_payload()
        payload["observations"][0]["musicxml_sha256"] = "4" * 64
        with self.assertRaisesRegex(OssqB8RReviewBridgeError, "exact B8P pair identity"):
            build_b8q_from_b8r(b8p_payload=b8p_payload(), b8r_payload=payload)

    def test_b8p_image_identity_drift_is_rejected(self) -> None:
        payload = b8p_payload()
        payload["pairs"][0]["image_sha256"] = "5" * 64
        with self.assertRaisesRegex(OssqB8RReviewBridgeError, "B8P input rejected"):
            build_b8q_from_b8r(b8p_payload=payload, b8r_payload=b8r_payload())

    def test_b8p_musicxml_identity_drift_is_rejected(self) -> None:
        payload = b8p_payload()
        payload["pairs"][0]["musicxml_sha256"] = "6" * 64
        with self.assertRaisesRegex(OssqB8RReviewBridgeError, "B8P input rejected"):
            build_b8q_from_b8r(b8p_payload=payload, b8r_payload=b8r_payload())

    def test_b8r_non_identity_payload_drift_fails_canonical_fingerprint(self) -> None:
        payload = b8r_payload()
        payload["observations"][0]["primary_similarity_ppm"] += 1
        with self.assertRaisesRegex(OssqB8RReviewBridgeError, "canonical receipt fingerprint"):
            build_b8q_from_b8r(b8p_payload=b8p_payload(), b8r_payload=payload)

    def test_review_required_b8r_observation_never_becomes_verified(self) -> None:
        source = b8r_payload()
        review_required = next(
            item for item in source["observations"] if item["decision"] == "review-required"
        )
        receipt = build_b8q_from_b8r(
            b8p_payload=b8p_payload(),
            b8r_payload=source,
        )
        record = next(
            item for item in receipt.records if item.segment_id == review_required["segment_id"]
        )
        self.assertIs(record.decision, PairReviewDecision.REVIEW_REQUIRED)
        self.assertIs(record.method, PairReviewMethod.UNREVIEWED)
        self.assertIsNone(record.review_evidence_sha256)
        self.assertIsNone(record.reviewer_identity_sha256)

    def test_forged_extra_verified_candidate_fails_closed(self) -> None:
        payload = b8r_payload()
        item = next(
            item for item in payload["observations"] if item["decision"] == "review-required"
        )
        item["decision"] = "verified-candidate"
        with self.assertRaisesRegex(OssqB8RReviewBridgeError, "decision counts"):
            build_b8q_from_b8r(b8p_payload=b8p_payload(), b8r_payload=payload)

    def test_partial_or_extra_b8r_populations_fail_closed(self) -> None:
        partial = b8r_payload()
        partial["observations"] = partial["observations"][:-1]
        with self.assertRaisesRegex(OssqB8RReviewBridgeError, "exactly 412 observations"):
            build_b8q_from_b8r(b8p_payload=b8p_payload(), b8r_payload=partial)

        extra = b8r_payload()
        extra["observations"].append(copy.deepcopy(extra["observations"][0]))
        with self.assertRaisesRegex(OssqB8RReviewBridgeError, "exactly 412 observations"):
            build_b8q_from_b8r(b8p_payload=b8p_payload(), b8r_payload=extra)

    def test_nonzero_score_cross_conflict_fails_closed(self) -> None:
        payload = b8r_payload()
        payload["score_cross_conflicts"][0][1] = 1
        with self.assertRaisesRegex(OssqB8RReviewBridgeError, "cross-conflict"):
            build_b8q_from_b8r(b8p_payload=b8p_payload(), b8r_payload=payload)

    def test_b8r_authority_boundaries_must_remain_false(self) -> None:
        payload = b8r_payload()
        payload["stage8_admission_authority"] = True
        with self.assertRaisesRegex(OssqB8RReviewBridgeError, "authority boundary"):
            build_b8q_from_b8r(b8p_payload=b8p_payload(), b8r_payload=payload)

    def test_expected_b8r_receipt_identity_constant_is_frozen(self) -> None:
        self.assertEqual(
            B8R_TO_B8Q_EXPECTED_B8R_RECEIPT_SHA256,
            "4b880a6348897189e9851d74258ba8a07cb210bb7695d159319adad633e237c1",
        )


if __name__ == "__main__":
    unittest.main()
