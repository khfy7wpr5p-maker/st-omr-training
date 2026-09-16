"""TR-POLY-09B8Q independent pair-review boundary for real OSSQ candidates.

B8P proves reproducible materialization. It does not prove that a scanned system
image is semantically paired with the correct MusicXML segment. B8Q therefore
starts every B8P pair as REVIEW_REQUIRED and allows VERIFIED only when separate,
independent review evidence is supplied.

This module is hash-only. It stores no PDF, PNG, MusicXML, license text, personal
data, model bytes, or TEST bytes. B8Q review evidence is necessary but not
sufficient for Stage 8 admission; Stage 8-0/8-1 rights, split, byte, leakage and
sealed-TEST contracts remain authoritative.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
import re
from typing import Final, Mapping, Sequence


B8Q_REVIEW_VERSION: Final[str] = "st-omr-poly-v2-ossq-pair-review-v1"
B8Q_EXPECTED_B8P_RECEIPT_SHA256: Final[str] = (
    "3f9f2df43b287dd95e03eaf1ffc4c5a3c9e25bdced8f09675918bc9b1f99e0cf"
)
B8Q_READY_SCORE_IDS: Final[tuple[str, ...]] = (
    "7070781", "7075297", "7078259", "7093885", "7103818", "7108150", "8071278"
)
B8Q_BLOCKED_SCORE_IDS: Final[tuple[str, ...]] = ("7397765",)
B8Q_EXPECTED_PAIR_COUNT_BY_SCORE: Final[tuple[tuple[str, int], ...]] = (
    ("7070781", 9),
    ("7075297", 28),
    ("7078259", 26),
    ("7093885", 26),
    ("7103818", 87),
    ("7108150", 117),
    ("8071278", 119),
)
B8Q_EXPECTED_PAIR_COUNT: Final[int] = sum(count for _, count in B8Q_EXPECTED_PAIR_COUNT_BY_SCORE)

_SEGMENT_RE = re.compile(r"^sq(?P<score>[0-9]+):(?P<page>[0-9]{4}):(?P<system>[0-9]{4})$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_REASON_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")


class OssqPairReviewError(ValueError):
    """Raised when B8Q review evidence fails closed."""


class PairReviewDecision(str, Enum):
    VERIFIED = "verified"
    REVIEW_REQUIRED = "review-required"
    BLOCKED = "blocked"


class PairReviewMethod(str, Enum):
    UNREVIEWED = "unreviewed-v1"
    INDEPENDENT_VISUAL = "independent-visual-v1"
    INDEPENDENT_CROSS_RENDER = "independent-cross-render-v1"


@dataclass(frozen=True, slots=True)
class OssqPairReviewRecord:
    score_id: str
    segment_id: str
    image_sha256: str
    musicxml_sha256: str
    decision: PairReviewDecision
    method: PairReviewMethod
    review_evidence_sha256: str | None
    reviewer_identity_sha256: str | None
    reason_code: str

    def __post_init__(self) -> None:
        match = _SEGMENT_RE.fullmatch(self.segment_id)
        if match is None or match.group("score") != self.score_id:
            raise OssqPairReviewError("segment_id does not bind the declared score")
        if self.score_id not in B8Q_READY_SCORE_IDS:
            raise OssqPairReviewError("review record score is outside the exact B8Q READY population")
        _require_sha256("image_sha256", self.image_sha256)
        _require_sha256("musicxml_sha256", self.musicxml_sha256)
        _require_optional_sha256("review_evidence_sha256", self.review_evidence_sha256)
        _require_optional_sha256("reviewer_identity_sha256", self.reviewer_identity_sha256)
        if not isinstance(self.decision, PairReviewDecision):
            raise OssqPairReviewError("decision must be PairReviewDecision")
        if not isinstance(self.method, PairReviewMethod):
            raise OssqPairReviewError("method must be PairReviewMethod")
        if not isinstance(self.reason_code, str) or _REASON_RE.fullmatch(self.reason_code) is None:
            raise OssqPairReviewError("reason_code must match the bounded reason contract")

        if self.decision is PairReviewDecision.REVIEW_REQUIRED:
            if self.method is not PairReviewMethod.UNREVIEWED:
                raise OssqPairReviewError("REVIEW_REQUIRED must remain unreviewed")
            if self.review_evidence_sha256 is not None or self.reviewer_identity_sha256 is not None:
                raise OssqPairReviewError("unreviewed pairs may not carry independent-review authority")
            if self.reason_code != "independent-review-not-yet-performed":
                raise OssqPairReviewError("unreviewed pairs must use the canonical REVIEW_REQUIRED reason")
            return

        if self.method is PairReviewMethod.UNREVIEWED:
            raise OssqPairReviewError("VERIFIED/BLOCKED decisions require an independent review method")
        if self.review_evidence_sha256 is None:
            raise OssqPairReviewError("VERIFIED/BLOCKED decisions require review evidence")
        if self.reviewer_identity_sha256 is None:
            raise OssqPairReviewError("VERIFIED/BLOCKED decisions require reviewer identity evidence")
        if self.decision is PairReviewDecision.VERIFIED and self.reason_code != "independent-pair-review-passed":
            raise OssqPairReviewError("VERIFIED must use the canonical passed-review reason")
        if self.decision is PairReviewDecision.BLOCKED and self.reason_code == "independent-pair-review-passed":
            raise OssqPairReviewError("BLOCKED must carry a blocking reason")


@dataclass(frozen=True, slots=True)
class OssqPairReviewReceipt:
    version: str
    b8p_receipt_sha256: str
    ready_score_ids: tuple[str, ...]
    blocked_score_ids: tuple[str, ...]
    records: tuple[OssqPairReviewRecord, ...]
    review_count_by_decision: tuple[tuple[str, int], ...]
    independent_pairing_review_authority: bool
    stage8_admission_authority: bool
    train_validation_assignment_authority: bool
    test_artifact_bytes_accessed: bool
    production_authority: bool
    commercial_use_authority: bool
    receipt_sha256: str

    def payload_without_fingerprint(self) -> dict[str, object]:
        return {
            "version": self.version,
            "b8p_receipt_sha256": self.b8p_receipt_sha256,
            "ready_score_ids": list(self.ready_score_ids),
            "blocked_score_ids": list(self.blocked_score_ids),
            "records": [_record_payload(item) for item in self.records],
            "review_count_by_decision": [list(item) for item in self.review_count_by_decision],
            "independent_pairing_review_authority": self.independent_pairing_review_authority,
            "stage8_admission_authority": self.stage8_admission_authority,
            "train_validation_assignment_authority": self.train_validation_assignment_authority,
            "test_artifact_bytes_accessed": self.test_artifact_bytes_accessed,
            "production_authority": self.production_authority,
            "commercial_use_authority": self.commercial_use_authority,
        }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise OssqPairReviewError(f"{name} must be lowercase SHA-256")
    return value


def _require_optional_sha256(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _require_sha256(name, value)


def _record_payload(record: OssqPairReviewRecord) -> dict[str, object]:
    payload = asdict(record)
    payload["decision"] = record.decision.value
    payload["method"] = record.method.value
    return payload


def _recompute_embedded_b8p_sha(payload: Mapping[str, object]) -> str:
    copied = dict(payload)
    embedded = copied.pop("receipt_sha256", None)
    _require_sha256("B8P receipt_sha256", embedded)
    return sha256(_canonical_bytes(copied)).hexdigest()


def _validate_b8p_payload(payload: object) -> tuple[dict[str, object], ...]:
    if not isinstance(payload, Mapping):
        raise OssqPairReviewError("B8P payload must be a mapping")
    embedded = payload.get("receipt_sha256")
    if embedded != B8Q_EXPECTED_B8P_RECEIPT_SHA256:
        raise OssqPairReviewError("B8P receipt identity differs from frozen B8Q input")
    if _recompute_embedded_b8p_sha(payload) != B8Q_EXPECTED_B8P_RECEIPT_SHA256:
        raise OssqPairReviewError("B8P canonical payload fingerprint mismatch")
    if tuple(payload.get("ready_score_ids", ())) != B8Q_READY_SCORE_IDS:
        raise OssqPairReviewError("B8P READY score population differs from B8Q")
    if tuple(payload.get("blocked_score_ids", ())) != B8Q_BLOCKED_SCORE_IDS:
        raise OssqPairReviewError("B8P blocked score population differs from B8Q")
    expected_counts = tuple((str(score), int(count)) for score, count in payload.get("pair_count_by_score", ()))
    if expected_counts != B8Q_EXPECTED_PAIR_COUNT_BY_SCORE:
        raise OssqPairReviewError("B8P pair counts differ from the frozen B8Q population")
    for authority in (
        "independent_pairing_review_authority",
        "stage8_admission_authority",
        "train_validation_assignment_authority",
        "test_artifact_bytes_accessed",
        "production_authority",
        "commercial_use_authority",
    ):
        if payload.get(authority) is not False:
            raise OssqPairReviewError(f"B8P authority boundary changed: {authority}")

    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != B8Q_EXPECTED_PAIR_COUNT:
        raise OssqPairReviewError("B8P pair population must contain exactly 412 pairs")
    pairs: list[dict[str, object]] = []
    seen: set[str] = set()
    count_by_score = {score: 0 for score in B8Q_READY_SCORE_IDS}
    for index, raw in enumerate(raw_pairs):
        if not isinstance(raw, Mapping):
            raise OssqPairReviewError(f"B8P pair {index} must be a mapping")
        score_id = raw.get("score_id")
        segment_id = raw.get("segment_id")
        if not isinstance(score_id, str) or score_id not in B8Q_READY_SCORE_IDS:
            raise OssqPairReviewError(f"B8P pair {index} has an unexpected score")
        if not isinstance(segment_id, str):
            raise OssqPairReviewError(f"B8P pair {index} lacks segment_id")
        match = _SEGMENT_RE.fullmatch(segment_id)
        if match is None or match.group("score") != score_id:
            raise OssqPairReviewError(f"B8P pair {index} segment does not bind score")
        if segment_id in seen:
            raise OssqPairReviewError("B8P pair population contains duplicate segment ids")
        seen.add(segment_id)
        _require_sha256("B8P image_sha256", raw.get("image_sha256"))
        _require_sha256("B8P musicxml_sha256", raw.get("musicxml_sha256"))
        count_by_score[score_id] += 1
        pairs.append(dict(raw))
    if tuple((score, count_by_score[score]) for score in B8Q_READY_SCORE_IDS) != B8Q_EXPECTED_PAIR_COUNT_BY_SCORE:
        raise OssqPairReviewError("B8P actual pair counts differ from declared frozen counts")
    return tuple(sorted(pairs, key=lambda item: str(item["segment_id"])))


def _new_receipt(records: Sequence[OssqPairReviewRecord]) -> OssqPairReviewReceipt:
    ordered = tuple(sorted(records, key=lambda item: item.segment_id))
    if len(ordered) != B8Q_EXPECTED_PAIR_COUNT:
        raise OssqPairReviewError("B8Q receipt must cover all 412 B8P pairs exactly once")
    if len({item.segment_id for item in ordered}) != len(ordered):
        raise OssqPairReviewError("B8Q receipt contains duplicate segment ids")
    counts = tuple(
        (decision.value, sum(item.decision is decision for item in ordered))
        for decision in PairReviewDecision
    )
    has_independent_review = any(item.decision is not PairReviewDecision.REVIEW_REQUIRED for item in ordered)
    provisional = OssqPairReviewReceipt(
        version=B8Q_REVIEW_VERSION,
        b8p_receipt_sha256=B8Q_EXPECTED_B8P_RECEIPT_SHA256,
        ready_score_ids=B8Q_READY_SCORE_IDS,
        blocked_score_ids=B8Q_BLOCKED_SCORE_IDS,
        records=ordered,
        review_count_by_decision=counts,
        independent_pairing_review_authority=has_independent_review,
        stage8_admission_authority=False,
        train_validation_assignment_authority=False,
        test_artifact_bytes_accessed=False,
        production_authority=False,
        commercial_use_authority=False,
        receipt_sha256="0" * 64,
    )
    fingerprint = sha256(_canonical_bytes(provisional.payload_without_fingerprint())).hexdigest()
    return OssqPairReviewReceipt(
        version=provisional.version,
        b8p_receipt_sha256=provisional.b8p_receipt_sha256,
        ready_score_ids=provisional.ready_score_ids,
        blocked_score_ids=provisional.blocked_score_ids,
        records=provisional.records,
        review_count_by_decision=provisional.review_count_by_decision,
        independent_pairing_review_authority=provisional.independent_pairing_review_authority,
        stage8_admission_authority=False,
        train_validation_assignment_authority=False,
        test_artifact_bytes_accessed=False,
        production_authority=False,
        commercial_use_authority=False,
        receipt_sha256=fingerprint,
    )


def build_unreviewed_b8q_receipt(b8p_payload: object) -> OssqPairReviewReceipt:
    """Bind all exact B8P pairs while granting no pairing or Stage 8 authority."""

    pairs = _validate_b8p_payload(b8p_payload)
    return _new_receipt(tuple(
        OssqPairReviewRecord(
            score_id=str(pair["score_id"]),
            segment_id=str(pair["segment_id"]),
            image_sha256=str(pair["image_sha256"]),
            musicxml_sha256=str(pair["musicxml_sha256"]),
            decision=PairReviewDecision.REVIEW_REQUIRED,
            method=PairReviewMethod.UNREVIEWED,
            review_evidence_sha256=None,
            reviewer_identity_sha256=None,
            reason_code="independent-review-not-yet-performed",
        )
        for pair in pairs
    ))


def apply_independent_pair_reviews(
    *,
    b8p_payload: object,
    reviews: Sequence[OssqPairReviewRecord],
) -> OssqPairReviewReceipt:
    """Apply exact review decisions without allowing B8P identity alone to verify a pair.

    ``reviews`` must cover the complete 412-pair population. Callers can keep any
    not-yet-reviewed pair as REVIEW_REQUIRED. VERIFIED/BLOCKED entries require
    separate evidence and reviewer identity hashes by construction.
    """

    pairs = _validate_b8p_payload(b8p_payload)
    expected = {
        str(pair["segment_id"]): (
            str(pair["score_id"]),
            str(pair["image_sha256"]),
            str(pair["musicxml_sha256"]),
        )
        for pair in pairs
    }
    if len(reviews) != len(expected):
        raise OssqPairReviewError("review population must equal the exact 412-pair B8P population")
    seen: set[str] = set()
    for review in reviews:
        if not isinstance(review, OssqPairReviewRecord):
            raise OssqPairReviewError("every review must be OssqPairReviewRecord")
        if review.segment_id in seen:
            raise OssqPairReviewError("duplicate review segment id")
        seen.add(review.segment_id)
        identity = expected.get(review.segment_id)
        if identity is None:
            raise OssqPairReviewError("review references a segment outside B8P")
        if identity != (review.score_id, review.image_sha256, review.musicxml_sha256):
            raise OssqPairReviewError("review identity differs from exact B8P pair hashes")
    if seen != set(expected):
        raise OssqPairReviewError("review population does not cover exact B8P segment ids")
    return _new_receipt(reviews)


def verified_review_records(receipt: OssqPairReviewReceipt) -> tuple[OssqPairReviewRecord, ...]:
    """Return only independently VERIFIED records for later Stage 8 quarantine preparation."""

    if not isinstance(receipt, OssqPairReviewReceipt):
        raise OssqPairReviewError("receipt must be OssqPairReviewReceipt")
    return tuple(item for item in receipt.records if item.decision is PairReviewDecision.VERIFIED)


def review_receipt_to_json(receipt: OssqPairReviewReceipt) -> str:
    payload = receipt.payload_without_fingerprint()
    payload["receipt_sha256"] = receipt.receipt_sha256
    return _canonical_bytes(payload).decode("ascii")
