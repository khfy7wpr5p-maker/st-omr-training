from __future__ import annotations

from dataclasses import replace
import unittest

from st_omr_training.poly_v2_ossq_rights_evidence_batch1 import (
    B8M_BATCH1_EVIDENCE,
    B8M_BATCH1_SCORE_IDS,
    audit_b8m_batch1,
    build_b8m_batch1_reviews,
)
from st_omr_training.poly_v2_ossq_score_rights_audit import (
    OssqScannedInventory,
    OssqScoreRightsAuditError,
    OssqScoreSourceRecord,
)
from st_omr_training.real_data_contract import ReviewState, RightsBasis


def h(ch: str) -> str:
    return ch * 64


def make_inventory() -> OssqScannedInventory:
    selected = [
        OssqScoreSourceRecord(
            score_id="7070781",
            path="Mozart/K464",
            name="K464 I",
            source_type="1",
            musescore_url="https://musescore.com/openscore-string-quartets/scores/7070781",
            imslp_id="#64141",
            set_id="5108515",
        ),
        OssqScoreSourceRecord(
            score_id="7075297",
            path="Mozart/K464",
            name="K464 IV",
            source_type="1",
            musescore_url="https://musescore.com/openscore-string-quartets/scores/7075297",
            imslp_id="#64141",
            set_id="5108515",
        ),
        OssqScoreSourceRecord(
            score_id="7078259",
            path="Mozart/K464",
            name="K464 III",
            source_type="1",
            musescore_url="https://musescore.com/openscore-string-quartets/scores/7078259",
            imslp_id="#64141",
            set_id="5108515",
        ),
        OssqScoreSourceRecord(
            score_id="7093885",
            path="Mozart/K464",
            name="K464 II",
            source_type="1",
            musescore_url="https://musescore.com/openscore-string-quartets/scores/7093885",
            imslp_id="#64141",
            set_id="5108515",
        ),
        OssqScoreSourceRecord(
            score_id="7103818",
            path="Mozart/K387",
            name="K387",
            source_type="1",
            musescore_url="https://musescore.com/openscore-string-quartets/scores/7103818",
            imslp_id="#64136",
            set_id="5108511",
        ),
        OssqScoreSourceRecord(
            score_id="7108150",
            path="Brahms/Op51No1",
            name="Brahms Op51 No1",
            source_type="1",
            musescore_url="https://musescore.com/openscore-string-quartets/scores/7108150",
            imslp_id="#242305",
            set_id="5108531",
        ),
        OssqScoreSourceRecord(
            score_id="7397765",
            path="Schubert/D810",
            name="Schubert D810",
            source_type="1",
            musescore_url="https://musescore.com/openscore-string-quartets/scores/7397765",
            imslp_id="#04047",
            set_id="5108909",
        ),
        OssqScoreSourceRecord(
            score_id="8071278",
            path="Beethoven/Op18No1",
            name="Beethoven Op18 No1",
            source_type="1",
            musescore_url="https://musescore.com/openscore-string-quartets/scores/8071278",
            imslp_id="#04755",
            set_id="5108454",
        ),
    ]

    filler: list[OssqScoreSourceRecord] = []
    for index in range(114):
        score_id = str(10_000_000 + index)
        path = "Filler/Shared" if index < 4 else f"Filler/Work_{index}"
        source_type = "2" if index == 113 else "1"
        filler.append(
            OssqScoreSourceRecord(
                score_id=score_id,
                path=path,
                name=f"Filler {index}",
                source_type=source_type,
                musescore_url=(
                    "https://musescore.com/openscore-string-quartets/scores/" + score_id
                ),
                imslp_id="#99000" if index < 4 else f"#{100000 + index}",
                set_id=str(20_000_000 + index),
            )
        )

    records = tuple(sorted((*selected, *filler), key=lambda item: int(item.score_id)))
    return OssqScannedInventory(records=records, source_file_sha256=h("a"))


class OssqRightsEvidenceBatch1Tests(unittest.TestCase):
    def test_batch_identity_is_exact_and_conservative(self) -> None:
        self.assertEqual(
            B8M_BATCH1_SCORE_IDS,
            (
                "7070781",
                "7075297",
                "7078259",
                "7093885",
                "7103818",
                "7108150",
                "7397765",
                "8071278",
            ),
        )
        self.assertEqual(len(B8M_BATCH1_EVIDENCE), 5)
        for evidence in B8M_BATCH1_EVIDENCE:
            self.assertEqual(evidence.copyright_label, "Public Domain")
            self.assertEqual(evidence.training_scope, "research-training-candidate-only")
            self.assertEqual(len(evidence.canonical_sha256()), 64)
            self.assertEqual(len(evidence.provenance_sha256()), 64)
            self.assertNotEqual(evidence.canonical_sha256(), evidence.provenance_sha256())

    def test_full_population_reviews_keep_non_batch_scores_pending(self) -> None:
        inventory = make_inventory()
        reviews = build_b8m_batch1_reviews(inventory)
        self.assertEqual(len(reviews), 121)
        approved = tuple(item for item in reviews if item.rights_review is ReviewState.APPROVED)
        pending = tuple(item for item in reviews if item.rights_review is ReviewState.PENDING)
        self.assertEqual(tuple(sorted((item.score_id for item in approved), key=int)), B8M_BATCH1_SCORE_IDS)
        self.assertEqual(len(approved), 8)
        self.assertEqual(len(pending), 113)
        for item in approved:
            self.assertIs(item.rights_basis, RightsBasis.PUBLIC_DOMAIN)
            self.assertTrue(item.independent_rights_review)
            self.assertTrue(item.training_allowed)
            self.assertFalse(item.commercial_use_allowed)
            self.assertFalse(item.redistribution_allowed)
            self.assertIsNotNone(item.rights_evidence_sha256)
            self.assertNotEqual(item.rights_evidence_sha256, item.upstream_metadata_evidence_sha256)
        for item in pending:
            self.assertFalse(item.independent_rights_review)
            self.assertIsNone(item.training_allowed)
            self.assertIsNone(item.commercial_use_allowed)
            self.assertIsNone(item.redistribution_allowed)

    def test_shared_k464_source_has_one_identical_policy(self) -> None:
        inventory = make_inventory()
        reviews = build_b8m_batch1_reviews(inventory)
        shared = tuple(item for item in reviews if item.imslp_id == "#64141")
        self.assertEqual(len(shared), 4)
        self.assertEqual(len({item.rights_evidence_sha256 for item in shared}), 1)
        self.assertEqual(len({item.provenance_evidence_sha256 for item in shared}), 1)
        self.assertEqual(len({item.policy_tuple() for item in shared}), 1)

    def test_audit_receipt_grants_research_candidates_but_no_commercial_authority(self) -> None:
        inventory = make_inventory()
        receipt = audit_b8m_batch1(inventory)
        self.assertEqual(receipt.training_candidate_score_ids, B8M_BATCH1_SCORE_IDS)
        self.assertEqual(receipt.commercial_candidate_score_ids, ())
        self.assertEqual(len(receipt.blocked_score_ids), 113)
        self.assertFalse(receipt.test_artifact_bytes_accessed)
        self.assertFalse(receipt.production_authority)
        self.assertFalse(receipt.commercial_use_authority)

    def test_inventory_identity_drift_fails_closed(self) -> None:
        inventory = make_inventory()
        records = list(inventory.records)
        index = next(i for i, item in enumerate(records) if item.score_id == "8071278")
        records[index] = replace(records[index], imslp_id="#999999")
        tampered = replace(inventory, records=tuple(records))
        with self.assertRaisesRegex(OssqScoreRightsAuditError, "identity mismatch"):
            build_b8m_batch1_reviews(tampered)


if __name__ == "__main__":
    unittest.main()
