from __future__ import annotations

from dataclasses import replace
import unittest

from st_omr_training.poly_v2_ossq_score_rights_audit import (
    OssqRightsAuditReceipt,
    OssqScannedInventory,
    OssqScoreRightsAuditError,
    OssqScoreRightsReview,
    OssqScoreSourceRecord,
    _parse_ossq_scanned_inventory,
    audit_ossq_scanned_score_rights,
    git_blob_sha1,
    load_pinned_ossq_scanned_inventory,
)
from st_omr_training.real_data_contract import ReviewState, RightsBasis


def h(ch: str) -> str:
    return ch * 64


def make_inventory() -> OssqScannedInventory:
    records: list[OssqScoreSourceRecord] = []
    for index in range(122):
        score_id = str(1000000 + index)
        if index < 7:
            path_index = 0
        else:
            path_index = index - 6
        source_type = "2" if index == 121 else "1"
        imslp_id = "#9000" if index < 2 else f"#{10000 + index}"
        records.append(
            OssqScoreSourceRecord(
                score_id=score_id,
                path=f"Composer/Work_{path_index}",
                name=f"Work {index}",
                source_type=source_type,
                musescore_url=(
                    "https://musescore.com/openscore-string-quartets/scores/" + score_id
                ),
                imslp_id=imslp_id,
                set_id=str(2000000 + index),
            )
        )
    return OssqScannedInventory(
        records=tuple(records),
        source_file_sha256=h("a"),
    )


def pending_reviews(inventory: OssqScannedInventory) -> list[OssqScoreRightsReview]:
    reviews: list[OssqScoreRightsReview] = []
    for index, source in enumerate(inventory.scanned_review_records):
        shared_source = source.imslp_id == "#9000"
        provenance = h("b") if shared_source else h("c")
        reviews.append(
            OssqScoreRightsReview(
                score_id=source.score_id,
                imslp_id=source.imslp_id,
                source_type=source.source_type,
                upstream_metadata_evidence_sha256=h("d"),
                provenance_evidence_sha256=provenance,
                rights_evidence_sha256=None,
                rights_basis=None,
                rights_review=ReviewState.PENDING,
                independent_rights_review=False,
                training_allowed=None,
                commercial_use_allowed=None,
                redistribution_allowed=None,
                notes=f"pending fixture {index}",
            )
        )
    return reviews


class OssqScoreRightsAuditTests(unittest.TestCase):
    def test_parser_verifies_git_blob_and_sorts_rows(self) -> None:
        payload = (
            "id\tpath\tname\ttype\tlink\timslp\tset_id\n"
            "3\tC/W2\tW2\t4\thttps://musescore.com/openscore-string-quartets/scores/3\t#30\t300\n"
            "1\tA/W1\tW1\t1\thttps://musescore.com/openscore-string-quartets/scores/1\t#10\t100\n"
            "2\tB/W1\tW1b\t2\thttps://musescore.com/openscore-string-quartets/scores/2\t#20\t200\n"
        ).encode("utf-8")
        records = _parse_ossq_scanned_inventory(
            payload,
            expected_git_blob_sha=git_blob_sha1(payload),
            expected_entry_count=3,
            expected_unique_path_count=3,
        )
        self.assertEqual(tuple(item.score_id for item in records), ("1", "2", "3"))
        self.assertFalse(records[1].requires_scanned_rights_review)
        self.assertTrue(records[2].requires_scanned_rights_review)

    def test_public_loader_rejects_non_pinned_metadata_bytes(self) -> None:
        payload = b"id\tpath\tname\ttype\tlink\timslp\tset_id\n"
        with self.assertRaisesRegex(OssqScoreRightsAuditError, "pinned Git blob"):
            load_pinned_ossq_scanned_inventory(payload)

    def test_upstream_metadata_cannot_be_its_own_independent_rights_evidence(self) -> None:
        inventory = make_inventory()
        source = inventory.scanned_review_records[2]
        upstream = h("e")
        with self.assertRaisesRegex(OssqScoreRightsAuditError, "metadata alone"):
            OssqScoreRightsReview(
                score_id=source.score_id,
                imslp_id=source.imslp_id,
                source_type=source.source_type,
                upstream_metadata_evidence_sha256=upstream,
                provenance_evidence_sha256=h("f"),
                rights_evidence_sha256=upstream,
                rights_basis=RightsBasis.PUBLIC_DOMAIN,
                rights_review=ReviewState.APPROVED,
                independent_rights_review=True,
                training_allowed=True,
                commercial_use_allowed=True,
                redistribution_allowed=True,
            )

    def test_complete_pending_audit_keeps_every_scanned_score_blocked(self) -> None:
        inventory = make_inventory()
        reviews = pending_reviews(inventory)
        receipt = audit_ossq_scanned_score_rights(inventory=inventory, reviews=reviews)
        self.assertIsInstance(receipt, OssqRightsAuditReceipt)
        self.assertEqual(receipt.reviewed_score_count, 121)
        self.assertEqual(receipt.training_candidate_score_ids, ())
        self.assertEqual(receipt.commercial_candidate_score_ids, ())
        self.assertEqual(len(receipt.blocked_score_ids), 121)
        self.assertEqual(receipt.shared_imslp_source_count, 1)
        self.assertFalse(receipt.test_artifact_bytes_accessed)
        self.assertFalse(receipt.production_authority)
        self.assertFalse(receipt.commercial_use_authority)
        self.assertEqual(len(receipt.fingerprint()), 64)

    def test_missing_score_review_fails_closed(self) -> None:
        inventory = make_inventory()
        reviews = pending_reviews(inventory)
        reviews.pop()
        with self.assertRaisesRegex(OssqScoreRightsAuditError, "coverage differs"):
            audit_ossq_scanned_score_rights(inventory=inventory, reviews=reviews)

    def test_independently_approved_score_can_be_training_candidate_only(self) -> None:
        inventory = make_inventory()
        reviews = pending_reviews(inventory)
        target = reviews[2]
        reviews[2] = OssqScoreRightsReview(
            score_id=target.score_id,
            imslp_id=target.imslp_id,
            source_type=target.source_type,
            upstream_metadata_evidence_sha256=target.upstream_metadata_evidence_sha256,
            provenance_evidence_sha256=target.provenance_evidence_sha256,
            rights_evidence_sha256=h("9"),
            rights_basis=RightsBasis.PUBLIC_DOMAIN,
            rights_review=ReviewState.APPROVED,
            independent_rights_review=True,
            training_allowed=True,
            commercial_use_allowed=False,
            redistribution_allowed=True,
            notes="independently reviewed fixture",
        )
        receipt = audit_ossq_scanned_score_rights(inventory=inventory, reviews=reviews)
        self.assertEqual(receipt.training_candidate_score_ids, (target.score_id,))
        self.assertEqual(receipt.commercial_candidate_score_ids, ())
        self.assertNotIn(target.score_id, receipt.blocked_score_ids)

    def test_shared_imslp_file_cannot_receive_contradictory_rights_decisions(self) -> None:
        inventory = make_inventory()
        reviews = pending_reviews(inventory)
        first = reviews[0]
        reviews[0] = OssqScoreRightsReview(
            score_id=first.score_id,
            imslp_id=first.imslp_id,
            source_type=first.source_type,
            upstream_metadata_evidence_sha256=first.upstream_metadata_evidence_sha256,
            provenance_evidence_sha256=h("b"),
            rights_evidence_sha256=h("8"),
            rights_basis=RightsBasis.PUBLIC_DOMAIN,
            rights_review=ReviewState.APPROVED,
            independent_rights_review=True,
            training_allowed=True,
            commercial_use_allowed=True,
            redistribution_allowed=True,
        )
        with self.assertRaisesRegex(OssqScoreRightsAuditError, "contradictory"):
            audit_ossq_scanned_score_rights(inventory=inventory, reviews=reviews)

    def test_pending_or_rejected_review_cannot_grant_training(self) -> None:
        inventory = make_inventory()
        source = inventory.scanned_review_records[2]
        with self.assertRaisesRegex(OssqScoreRightsAuditError, "may not grant"):
            OssqScoreRightsReview(
                score_id=source.score_id,
                imslp_id=source.imslp_id,
                source_type=source.source_type,
                upstream_metadata_evidence_sha256=h("1"),
                provenance_evidence_sha256=h("2"),
                rights_evidence_sha256=None,
                rights_basis=None,
                rights_review=ReviewState.PENDING,
                independent_rights_review=False,
                training_allowed=True,
                commercial_use_allowed=None,
                redistribution_allowed=None,
            )


if __name__ == "__main__":
    unittest.main()
