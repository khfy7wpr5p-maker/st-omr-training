"""TR-POLY-09B8M first independent rights-evidence batch for OSSQ-OMR.

B8L created a fail-closed per-score audit gate. B8M supplies the first small,
independently checked evidence batch from the official IMSLP work pages for five
exact OSSQ source files. The batch is intentionally conservative:

* eight OSSQ score records are eligible only as research-training candidates;
* no score receives commercial-use or redistribution approval;
* all remaining scanned-track records stay PENDING;
* no PDF/image bytes are downloaded, opened, install-pinned, or admitted here;
* TEST and production authority remain untouched.

The evidence snapshot records what the official IMSLP work page showed on
2026-09-16 for the exact IMSLP file identifier used by OSSQ. This is an
engineering admission record, not legal advice and not a global copyright
conclusion for every jurisdiction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Final

from .poly_v2_ossq_score_rights_audit import (
    OSSQ_PUBLICATION_METADATA_GIT_BLOB_SHA,
    OssqRightsAuditReceipt,
    OssqScannedInventory,
    OssqScoreRightsAuditError,
    OssqScoreRightsReview,
    audit_ossq_scanned_score_rights,
)
from .poly_v2_corpus_source_selection import OSSQ_OMR_CAMERA_READY_SHA
from .real_data_contract import ReviewState, RightsBasis


B8M_OSSQ_RIGHTS_EVIDENCE_VERSION: Final[str] = (
    "st-omr-poly-v2-ossq-rights-evidence-batch1-v1"
)
B8M_EVIDENCE_RETRIEVED_DATE: Final[str] = "2026-09-16"


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(payload: object) -> str:
    return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class OssqIndependentRightsEvidence:
    imslp_id: str
    score_ids: tuple[str, ...]
    source_type: str
    imslp_work_url: str
    source_pdf_url: str
    publisher_statement: str
    copyright_label: str
    retrieved_date: str = B8M_EVIDENCE_RETRIEVED_DATE
    training_scope: str = "research-training-candidate-only"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.imslp_id.startswith("#") or not self.imslp_id[1:].isdigit():
            raise OssqScoreRightsAuditError("B8M imslp_id must be an IMSLP file identifier")
        if not self.score_ids or any(not item.isdigit() for item in self.score_ids):
            raise OssqScoreRightsAuditError("B8M score_ids must contain decimal score IDs")
        if tuple(sorted(self.score_ids, key=int)) != self.score_ids:
            raise OssqScoreRightsAuditError("B8M score_ids must be numerically sorted")
        if self.source_type != "1":
            raise OssqScoreRightsAuditError("B8M batch 1 admits only full scanned score type 1")
        for name in ("imslp_work_url", "source_pdf_url"):
            value = getattr(self, name)
            if not value.startswith("https://"):
                raise OssqScoreRightsAuditError(f"{name} must use https")
        if "imslp.org/wiki/" not in self.imslp_work_url:
            raise OssqScoreRightsAuditError("B8M work evidence must come from an IMSLP work page")
        if "imslp" not in self.source_pdf_url.lower():
            raise OssqScoreRightsAuditError("B8M source PDF URL must identify the IMSLP source")
        if not self.publisher_statement.strip():
            raise OssqScoreRightsAuditError("B8M publisher_statement must be non-empty")
        if self.copyright_label != "Public Domain":
            raise OssqScoreRightsAuditError("B8M batch 1 accepts only exact Public Domain labels")
        if self.retrieved_date != B8M_EVIDENCE_RETRIEVED_DATE:
            raise OssqScoreRightsAuditError("B8M evidence retrieval date mismatch")
        if self.training_scope != "research-training-candidate-only":
            raise OssqScoreRightsAuditError("B8M training scope may not escalate")
        if not isinstance(self.notes, str):
            raise OssqScoreRightsAuditError("B8M notes must be text")

    def canonical_sha256(self) -> str:
        return _digest(asdict(self))

    def provenance_sha256(self) -> str:
        return _digest(
            {
                "version": B8M_OSSQ_RIGHTS_EVIDENCE_VERSION,
                "source_commit_sha": OSSQ_OMR_CAMERA_READY_SHA,
                "publication_metadata_git_blob_sha": OSSQ_PUBLICATION_METADATA_GIT_BLOB_SHA,
                "imslp_id": self.imslp_id,
                "source_type": self.source_type,
                "source_pdf_url": self.source_pdf_url,
                "imslp_work_url": self.imslp_work_url,
            }
        )


B8M_BATCH1_EVIDENCE: Final[tuple[OssqIndependentRightsEvidence, ...]] = (
    OssqIndependentRightsEvidence(
        imslp_id="#04047",
        score_ids=("7397765",),
        source_type="1",
        imslp_work_url=(
            "https://imslp.org/wiki/String_Quartet_No.14,_D.810_(Schubert,_Franz)"
        ),
        source_pdf_url=(
            "https://vmirror.imslp.org/files/imglnks/usimg/6/67/"
            "IMSLP04047-SchubertStringQuartetNo14.pdf"
        ),
        publisher_statement=(
            "Franz Schubert's Werke, Serie V, No.14; Leipzig: Breitkopf & Hartel, 1890"
        ),
        copyright_label="Public Domain",
        notes="Exact IMSLP #04047 work-page entry independently checked on 2026-09-16.",
    ),
    OssqIndependentRightsEvidence(
        imslp_id="#04755",
        score_ids=("8071278",),
        source_type="1",
        imslp_work_url=(
            "https://imslp.org/wiki/String_Quartet_No.1_(Beethoven,_Ludwig_van)"
        ),
        source_pdf_url=(
            "https://vmirror.imslp.org/files/imglnks/usimg/7/7f/"
            "IMSLP04755-Beethoven_-_String_Quartet_No.1_Dover.pdf"
        ),
        publisher_statement=(
            "Ludwig van Beethovens Werke, Serie 6, Bd.1, No.37; "
            "Leipzig: Breitkopf und Hartel, 1862; Dover reprint 1970"
        ),
        copyright_label="Public Domain",
        notes="Exact IMSLP #04755 work-page entry independently checked on 2026-09-16.",
    ),
    OssqIndependentRightsEvidence(
        imslp_id="#64136",
        score_ids=("7103818",),
        source_type="1",
        imslp_work_url=(
            "https://imslp.org/wiki/String_Quartet_No.14_in_G_major,_K.387_"
            "(Mozart,_Wolfgang_Amadeus)"
        ),
        source_pdf_url=(
            "https://vmirror.imslp.org/files/imglnks/usimg/d/da/"
            "IMSLP64136-PMLP05221-Mozart_Werke_Breitkopf_Serie_14_KV387.pdf"
        ),
        publisher_statement=(
            "Mozarts Werke, Serie XIV, Bd.2, No.14; Leipzig: Breitkopf & Hartel, 1882"
        ),
        copyright_label="Public Domain",
        notes="Exact IMSLP #64136 work-page entry independently checked on 2026-09-16.",
    ),
    OssqIndependentRightsEvidence(
        imslp_id="#64141",
        score_ids=("7070781", "7075297", "7078259", "7093885"),
        source_type="1",
        imslp_work_url=(
            "https://imslp.org/wiki/String_Quartet_No.18_in_A_major,_K.464_"
            "(Mozart,_Wolfgang_Amadeus)"
        ),
        source_pdf_url=(
            "https://vmirror.imslp.org/files/imglnks/usimg/8/8e/"
            "IMSLP64141-PMLP05225-Mozart_Werke_Breitkopf_Serie_14_KV464.pdf"
        ),
        publisher_statement=(
            "Mozarts Werke, Serie XIV, Bd.2, No.18; Leipzig: Breitkopf & Hartel, 1882"
        ),
        copyright_label="Public Domain",
        notes=(
            "Exact IMSLP #64141 work-page entry independently checked on 2026-09-16; "
            "four OSSQ movement records intentionally share one identical source policy."
        ),
    ),
    OssqIndependentRightsEvidence(
        imslp_id="#242305",
        score_ids=("7108150",),
        source_type="1",
        imslp_work_url=(
            "https://imslp.org/wiki/String_Quartet_No.1,_Op.51_No.1_(Brahms,_Johannes)"
        ),
        source_pdf_url=(
            "https://ks15.imslp.org/files/imglnks/usimg/5/5f/"
            "IMSLP242305-PMLP13793-Brahms_Werke_Band_7_Breitkopf_JB_22_Op_51_No_1_scan.pdf"
        ),
        publisher_statement=(
            "Samtliche Werke, Band 7; Leipzig: Breitkopf & Hartel, 1926-27; "
            "editor Hans Gal"
        ),
        copyright_label="Public Domain",
        notes="Exact IMSLP #242305 work-page entry independently checked on 2026-09-16.",
    ),
)


B8M_BATCH1_SCORE_IDS: Final[tuple[str, ...]] = tuple(
    sorted(
        (score_id for evidence in B8M_BATCH1_EVIDENCE for score_id in evidence.score_ids),
        key=int,
    )
)


def _upstream_metadata_evidence_sha256(*, score_id: str, imslp_id: str, source_type: str) -> str:
    return _digest(
        {
            "source_commit_sha": OSSQ_OMR_CAMERA_READY_SHA,
            "publication_metadata_git_blob_sha": OSSQ_PUBLICATION_METADATA_GIT_BLOB_SHA,
            "score_id": score_id,
            "imslp_id": imslp_id,
            "source_type": source_type,
            "evidence_class": "ossq-upstream-metadata-only",
        }
    )


def _pending_provenance_sha256(*, imslp_id: str, source_type: str) -> str:
    return _digest(
        {
            "source_commit_sha": OSSQ_OMR_CAMERA_READY_SHA,
            "publication_metadata_git_blob_sha": OSSQ_PUBLICATION_METADATA_GIT_BLOB_SHA,
            "imslp_id": imslp_id,
            "source_type": source_type,
            "evidence_class": "ossq-source-identity-pending-independent-rights-review",
        }
    )


def build_b8m_batch1_reviews(inventory: OssqScannedInventory) -> tuple[OssqScoreRightsReview, ...]:
    """Return full B8L review coverage with only batch-1 sources approved for research training."""

    if not isinstance(inventory, OssqScannedInventory):
        raise TypeError("inventory must be OssqScannedInventory")

    evidence_by_score: dict[str, OssqIndependentRightsEvidence] = {}
    for evidence in B8M_BATCH1_EVIDENCE:
        for score_id in evidence.score_ids:
            if score_id in evidence_by_score:
                raise OssqScoreRightsAuditError("B8M evidence contains duplicate score coverage")
            evidence_by_score[score_id] = evidence

    source_by_score = {item.score_id: item for item in inventory.scanned_review_records}
    missing = sorted(set(evidence_by_score) - set(source_by_score), key=int)
    if missing:
        raise OssqScoreRightsAuditError(f"B8M evidence refers to absent OSSQ score IDs: {missing}")

    reviews: list[OssqScoreRightsReview] = []
    for source in inventory.scanned_review_records:
        upstream_sha = _upstream_metadata_evidence_sha256(
            score_id=source.score_id,
            imslp_id=source.imslp_id,
            source_type=source.source_type,
        )
        evidence = evidence_by_score.get(source.score_id)
        if evidence is None:
            reviews.append(
                OssqScoreRightsReview(
                    score_id=source.score_id,
                    imslp_id=source.imslp_id,
                    source_type=source.source_type,
                    upstream_metadata_evidence_sha256=upstream_sha,
                    provenance_evidence_sha256=_pending_provenance_sha256(
                        imslp_id=source.imslp_id,
                        source_type=source.source_type,
                    ),
                    rights_evidence_sha256=None,
                    rights_basis=None,
                    rights_review=ReviewState.PENDING,
                    independent_rights_review=False,
                    training_allowed=None,
                    commercial_use_allowed=None,
                    redistribution_allowed=None,
                    notes="Not independently reviewed in B8M batch 1; remains blocked.",
                )
            )
            continue

        if source.imslp_id != evidence.imslp_id or source.source_type != evidence.source_type:
            raise OssqScoreRightsAuditError(
                f"B8M evidence identity mismatch for OSSQ score {source.score_id}"
            )
        reviews.append(
            OssqScoreRightsReview(
                score_id=source.score_id,
                imslp_id=source.imslp_id,
                source_type=source.source_type,
                upstream_metadata_evidence_sha256=upstream_sha,
                provenance_evidence_sha256=evidence.provenance_sha256(),
                rights_evidence_sha256=evidence.canonical_sha256(),
                rights_basis=RightsBasis.PUBLIC_DOMAIN,
                rights_review=ReviewState.APPROVED,
                independent_rights_review=True,
                training_allowed=True,
                commercial_use_allowed=False,
                redistribution_allowed=False,
                notes=(
                    "Independent IMSLP work-page evidence supports research-training candidacy only; "
                    "no commercial-use, redistribution, production, or jurisdiction-global authority granted."
                ),
            )
        )

    return tuple(reviews)


def audit_b8m_batch1(inventory: OssqScannedInventory) -> OssqRightsAuditReceipt:
    """Run the B8L full-population audit using the B8M evidence batch."""

    receipt = audit_ossq_scanned_score_rights(
        inventory=inventory,
        reviews=build_b8m_batch1_reviews(inventory),
    )
    if receipt.training_candidate_score_ids != B8M_BATCH1_SCORE_IDS:
        raise OssqScoreRightsAuditError("B8M training-candidate population drifted")
    if receipt.commercial_candidate_score_ids:
        raise OssqScoreRightsAuditError("B8M may not grant commercial-candidate status")
    if receipt.production_authority or receipt.commercial_use_authority:
        raise OssqScoreRightsAuditError("B8M may not grant production/commercial authority")
    return receipt
