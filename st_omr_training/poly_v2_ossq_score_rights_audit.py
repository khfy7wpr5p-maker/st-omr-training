"""TR-POLY-09B8L per-score rights/provenance audit for OSSQ-OMR scans.

B8K selected OSSQ-OMR as the primary external source family but deliberately
kept the IMSLP-derived scanned track blocked. B8L turns the pinned OSSQ
score-source table into an immutable inventory and requires one independent
rights-review record for every scanned/manuscript/part-book score entry before
any subset can be considered for Stage 8 real-data admission.

Upstream copyright labels are metadata only. They never grant training,
commercial, redistribution, production, or TEST authority by themselves.
This module reads no external bytes other than caller-supplied metadata text and
never downloads score PDFs/images.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from hashlib import sha1, sha256
import io
import json
import re
from typing import Final, Iterable

from .poly_v2_corpus_source_selection import (
    OSSQ_OMR_CAMERA_READY_SHA,
    first_real_corpus_source_selection_fingerprint,
)
from .real_data_contract import ReviewState, RightsBasis


B8L_OSSQ_RIGHTS_AUDIT_VERSION: Final[str] = "st-omr-poly-v2-ossq-score-rights-audit-v1"
OSSQ_SCANNED_TYPES_PATH: Final[str] = "data/scanned_score_types.tsv"
OSSQ_SCANNED_TYPES_GIT_BLOB_SHA: Final[str] = "35ebe84f0c9f03a41231c92fa21183242bf12774"
OSSQ_PUBLICATION_METADATA_PATH: Final[str] = "data/scores_w_pub.yaml"
OSSQ_PUBLICATION_METADATA_GIT_BLOB_SHA: Final[str] = "7a72b220f4faa897900f51bcb85e49b845cb1304"
OSSQ_EXPECTED_SCORE_ENTRY_COUNT: Final[int] = 122
OSSQ_EXPECTED_UNIQUE_PATH_COUNT: Final[int] = 116
OSSQ_SCANNED_SOURCE_TYPES: Final[frozenset[str]] = frozenset({"0", "1", "1`", "1``", "3", "4"})
OSSQ_ALL_SOURCE_TYPES: Final[frozenset[str]] = frozenset({*OSSQ_SCANNED_SOURCE_TYPES, "2"})
OSSQ_SCANNED_TYPES_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "path",
    "name",
    "type",
    "link",
    "imslp",
    "set_id",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_IMSLP_RE = re.compile(r"^#[0-9]+$")
_DIGITS_RE = re.compile(r"^[0-9]+$")


class OssqScoreRightsAuditError(ValueError):
    """Raised when the B8L inventory or rights-review chain fails closed."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OssqScoreRightsAuditError(f"{name} must be lowercase SHA-256 text")
    return value


def _require_optional_sha256(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _require_sha256(name, value)


def git_blob_sha1(payload: bytes) -> str:
    """Compute the Git blob object SHA-1 for exact upstream-file verification."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    header = f"blob {len(payload)}\0".encode("ascii")
    return sha1(header + payload).hexdigest()


@dataclass(frozen=True, slots=True)
class OssqScoreSourceRecord:
    score_id: str
    path: str
    name: str
    source_type: str
    musescore_url: str
    imslp_id: str
    set_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.score_id, str) or _DIGITS_RE.fullmatch(self.score_id) is None:
            raise OssqScoreRightsAuditError("score_id must be decimal text")
        for field_name in ("path", "name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise OssqScoreRightsAuditError(f"{field_name} must be non-empty text")
        if self.source_type not in OSSQ_ALL_SOURCE_TYPES:
            raise OssqScoreRightsAuditError("source_type is outside the pinned OSSQ codebook")
        if not isinstance(self.musescore_url, str) or not self.musescore_url.startswith(
            "https://musescore.com/openscore-string-quartets/scores/"
        ):
            raise OssqScoreRightsAuditError("musescore_url is outside the OSSQ source namespace")
        if not isinstance(self.imslp_id, str) or _IMSLP_RE.fullmatch(self.imslp_id) is None:
            raise OssqScoreRightsAuditError("imslp_id must be an OSSQ IMSLP file identifier")
        if not isinstance(self.set_id, str) or _DIGITS_RE.fullmatch(self.set_id) is None:
            raise OssqScoreRightsAuditError("set_id must be decimal text")

    @property
    def requires_scanned_rights_review(self) -> bool:
        return self.source_type in OSSQ_SCANNED_SOURCE_TYPES

    def canonical_sha256(self) -> str:
        return sha256(_canonical_json_bytes(asdict(self))).hexdigest()


@dataclass(frozen=True, slots=True)
class OssqScannedInventory:
    records: tuple[OssqScoreSourceRecord, ...]
    source_file_sha256: str
    source_git_blob_sha: str = OSSQ_SCANNED_TYPES_GIT_BLOB_SHA
    source_commit_sha: str = OSSQ_OMR_CAMERA_READY_SHA
    source_path: str = OSSQ_SCANNED_TYPES_PATH
    publication_metadata_git_blob_sha: str = OSSQ_PUBLICATION_METADATA_GIT_BLOB_SHA
    publication_metadata_path: str = OSSQ_PUBLICATION_METADATA_PATH

    def __post_init__(self) -> None:
        if not self.records:
            raise OssqScoreRightsAuditError("OSSQ inventory must contain score records")
        if len(self.records) != OSSQ_EXPECTED_SCORE_ENTRY_COUNT:
            raise OssqScoreRightsAuditError("OSSQ inventory entry count differs from pinned camera-ready metadata")
        if len({item.path for item in self.records}) != OSSQ_EXPECTED_UNIQUE_PATH_COUNT:
            raise OssqScoreRightsAuditError("OSSQ inventory unique-path count differs from camera-ready metadata")
        if len({item.score_id for item in self.records}) != len(self.records):
            raise OssqScoreRightsAuditError("OSSQ inventory contains duplicate score IDs")
        if tuple(sorted(self.records, key=lambda item: int(item.score_id))) != self.records:
            raise OssqScoreRightsAuditError("OSSQ inventory records must remain score-id sorted")
        _require_sha256("source_file_sha256", self.source_file_sha256)
        for name, value in (
            ("source_git_blob_sha", self.source_git_blob_sha),
            ("source_commit_sha", self.source_commit_sha),
            ("publication_metadata_git_blob_sha", self.publication_metadata_git_blob_sha),
        ):
            if not isinstance(value, str) or _GIT_SHA40_RE.fullmatch(value) is None:
                raise OssqScoreRightsAuditError(f"{name} must be lowercase Git SHA-40 text")
        if self.source_git_blob_sha != OSSQ_SCANNED_TYPES_GIT_BLOB_SHA:
            raise OssqScoreRightsAuditError("OSSQ scanned-score metadata blob identity changed")
        if self.source_commit_sha != OSSQ_OMR_CAMERA_READY_SHA:
            raise OssqScoreRightsAuditError("OSSQ source commit differs from B8K camera-ready pin")
        if self.source_path != OSSQ_SCANNED_TYPES_PATH:
            raise OssqScoreRightsAuditError("OSSQ source path changed")
        if self.publication_metadata_git_blob_sha != OSSQ_PUBLICATION_METADATA_GIT_BLOB_SHA:
            raise OssqScoreRightsAuditError("OSSQ publication metadata blob identity changed")
        if self.publication_metadata_path != OSSQ_PUBLICATION_METADATA_PATH:
            raise OssqScoreRightsAuditError("OSSQ publication metadata path changed")

    @property
    def scanned_review_records(self) -> tuple[OssqScoreSourceRecord, ...]:
        return tuple(item for item in self.records if item.requires_scanned_rights_review)

    def fingerprint(self) -> str:
        payload = {
            "version": B8L_OSSQ_RIGHTS_AUDIT_VERSION,
            "source_commit_sha": self.source_commit_sha,
            "source_path": self.source_path,
            "source_git_blob_sha": self.source_git_blob_sha,
            "source_file_sha256": self.source_file_sha256,
            "publication_metadata_path": self.publication_metadata_path,
            "publication_metadata_git_blob_sha": self.publication_metadata_git_blob_sha,
            "records": [item.canonical_sha256() for item in self.records],
        }
        return sha256(_canonical_json_bytes(payload)).hexdigest()


def _parse_ossq_scanned_inventory(
    payload: bytes,
    *,
    expected_git_blob_sha: str,
    expected_entry_count: int,
    expected_unique_path_count: int,
) -> tuple[OssqScoreSourceRecord, ...]:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not isinstance(expected_git_blob_sha, str) or _GIT_SHA40_RE.fullmatch(expected_git_blob_sha) is None:
        raise OssqScoreRightsAuditError("expected_git_blob_sha must be Git SHA-40 text")
    actual_git_sha = git_blob_sha1(payload)
    if actual_git_sha != expected_git_blob_sha:
        raise OssqScoreRightsAuditError("OSSQ scanned-score metadata bytes do not match the pinned Git blob")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OssqScoreRightsAuditError("OSSQ scanned-score metadata must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if tuple(reader.fieldnames or ()) != OSSQ_SCANNED_TYPES_COLUMNS:
        raise OssqScoreRightsAuditError("OSSQ scanned-score metadata header changed")
    records: list[OssqScoreSourceRecord] = []
    for row_index, row in enumerate(reader, start=2):
        if None in row:
            raise OssqScoreRightsAuditError(f"OSSQ row {row_index} contains extra columns")
        if any(value is None for value in row.values()):
            raise OssqScoreRightsAuditError(f"OSSQ row {row_index} is missing columns")
        try:
            record = OssqScoreSourceRecord(
                score_id=row["id"],
                path=row["path"],
                name=row["name"],
                source_type=row["type"],
                musescore_url=row["link"],
                imslp_id=row["imslp"],
                set_id=row["set_id"],
            )
        except OssqScoreRightsAuditError as exc:
            raise OssqScoreRightsAuditError(f"invalid OSSQ row {row_index}: {exc}") from exc
        records.append(record)
    if len(records) != expected_entry_count:
        raise OssqScoreRightsAuditError("OSSQ parsed entry count differs from the expected pinned population")
    if len({item.path for item in records}) != expected_unique_path_count:
        raise OssqScoreRightsAuditError("OSSQ parsed unique-path count differs from expected pinned population")
    if len({item.score_id for item in records}) != len(records):
        raise OssqScoreRightsAuditError("OSSQ parsed inventory contains duplicate score IDs")
    return tuple(sorted(records, key=lambda item: int(item.score_id)))


def load_pinned_ossq_scanned_inventory(payload: bytes) -> OssqScannedInventory:
    """Parse only the exact B8K-pinned camera-ready scanned-score TSV bytes."""

    records = _parse_ossq_scanned_inventory(
        payload,
        expected_git_blob_sha=OSSQ_SCANNED_TYPES_GIT_BLOB_SHA,
        expected_entry_count=OSSQ_EXPECTED_SCORE_ENTRY_COUNT,
        expected_unique_path_count=OSSQ_EXPECTED_UNIQUE_PATH_COUNT,
    )
    return OssqScannedInventory(
        records=records,
        source_file_sha256=sha256(payload).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class OssqScoreRightsReview:
    score_id: str
    imslp_id: str
    source_type: str
    upstream_metadata_evidence_sha256: str
    provenance_evidence_sha256: str
    rights_evidence_sha256: str | None
    rights_basis: RightsBasis | None
    rights_review: ReviewState
    independent_rights_review: bool
    training_allowed: bool | None
    commercial_use_allowed: bool | None
    redistribution_allowed: bool | None
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.score_id, str) or _DIGITS_RE.fullmatch(self.score_id) is None:
            raise OssqScoreRightsAuditError("review score_id must be decimal text")
        if not isinstance(self.imslp_id, str) or _IMSLP_RE.fullmatch(self.imslp_id) is None:
            raise OssqScoreRightsAuditError("review imslp_id must be an IMSLP file identifier")
        if self.source_type not in OSSQ_SCANNED_SOURCE_TYPES:
            raise OssqScoreRightsAuditError("rights review is allowed only for the scanned-track source types")
        _require_sha256("upstream_metadata_evidence_sha256", self.upstream_metadata_evidence_sha256)
        _require_sha256("provenance_evidence_sha256", self.provenance_evidence_sha256)
        _require_optional_sha256("rights_evidence_sha256", self.rights_evidence_sha256)
        if self.rights_basis is not None and not isinstance(self.rights_basis, RightsBasis):
            raise OssqScoreRightsAuditError("rights_basis must be RightsBasis or None")
        if not isinstance(self.rights_review, ReviewState):
            raise OssqScoreRightsAuditError("rights_review must be ReviewState")
        if not isinstance(self.independent_rights_review, bool):
            raise OssqScoreRightsAuditError("independent_rights_review must be boolean")
        for name in ("training_allowed", "commercial_use_allowed", "redistribution_allowed"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise OssqScoreRightsAuditError(f"{name} must be bool or None")
        if not isinstance(self.notes, str):
            raise OssqScoreRightsAuditError("notes must be text")

        if self.rights_review is ReviewState.APPROVED:
            if not self.independent_rights_review:
                raise OssqScoreRightsAuditError("approved OSSQ rights review must be independently verified")
            if self.rights_evidence_sha256 is None or self.rights_basis is None:
                raise OssqScoreRightsAuditError("approved OSSQ rights review requires rights evidence and basis")
            if any(
                value is None
                for value in (
                    self.training_allowed,
                    self.commercial_use_allowed,
                    self.redistribution_allowed,
                )
            ):
                raise OssqScoreRightsAuditError("approved OSSQ rights review requires explicit permission booleans")
            if self.rights_evidence_sha256 == self.upstream_metadata_evidence_sha256:
                raise OssqScoreRightsAuditError(
                    "upstream OSSQ metadata alone cannot serve as independent rights evidence"
                )
        else:
            if any(
                value is True
                for value in (
                    self.training_allowed,
                    self.commercial_use_allowed,
                    self.redistribution_allowed,
                )
            ):
                raise OssqScoreRightsAuditError("pending/rejected OSSQ review may not grant usage permission")

    @property
    def training_candidate(self) -> bool:
        return self.rights_review is ReviewState.APPROVED and self.training_allowed is True

    @property
    def commercial_candidate(self) -> bool:
        return self.training_candidate and self.commercial_use_allowed is True

    def policy_tuple(self) -> tuple[object, ...]:
        return (
            self.rights_review,
            self.rights_basis,
            self.rights_evidence_sha256,
            self.provenance_evidence_sha256,
            self.independent_rights_review,
            self.training_allowed,
            self.commercial_use_allowed,
            self.redistribution_allowed,
        )

    def canonical_sha256(self) -> str:
        payload = asdict(self)
        payload["rights_basis"] = self.rights_basis.value if self.rights_basis is not None else None
        payload["rights_review"] = self.rights_review.value
        return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class OssqRightsAuditReceipt:
    inventory_fingerprint_sha256: str
    b8k_source_selection_sha256: str
    review_manifest_sha256: str
    reviewed_score_count: int
    training_candidate_score_ids: tuple[str, ...]
    commercial_candidate_score_ids: tuple[str, ...]
    blocked_score_ids: tuple[str, ...]
    shared_imslp_source_count: int
    audit_version: str = B8L_OSSQ_RIGHTS_AUDIT_VERSION
    test_artifact_bytes_accessed: bool = False
    production_authority: bool = False
    commercial_use_authority: bool = False

    def __post_init__(self) -> None:
        for name in (
            "inventory_fingerprint_sha256",
            "b8k_source_selection_sha256",
            "review_manifest_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if not isinstance(self.reviewed_score_count, int) or isinstance(self.reviewed_score_count, bool) or self.reviewed_score_count < 1:
            raise OssqScoreRightsAuditError("reviewed_score_count must be positive")
        for name in (
            "training_candidate_score_ids",
            "commercial_candidate_score_ids",
            "blocked_score_ids",
        ):
            values = getattr(self, name)
            if tuple(sorted(values, key=int)) != values or len(set(values)) != len(values):
                raise OssqScoreRightsAuditError(f"{name} must be unique and numerically sorted")
        if not set(self.commercial_candidate_score_ids).issubset(self.training_candidate_score_ids):
            raise OssqScoreRightsAuditError("commercial candidates must be a subset of training candidates")
        if set(self.training_candidate_score_ids) & set(self.blocked_score_ids):
            raise OssqScoreRightsAuditError("training and blocked score populations overlap")
        if not isinstance(self.shared_imslp_source_count, int) or isinstance(self.shared_imslp_source_count, bool) or self.shared_imslp_source_count < 0:
            raise OssqScoreRightsAuditError("shared_imslp_source_count must be non-negative")
        if self.audit_version != B8L_OSSQ_RIGHTS_AUDIT_VERSION:
            raise OssqScoreRightsAuditError("B8L audit version mismatch")
        if self.test_artifact_bytes_accessed or self.production_authority or self.commercial_use_authority:
            raise OssqScoreRightsAuditError("B8L may not access TEST bytes or grant product/commercial authority")

    def fingerprint(self) -> str:
        payload = asdict(self)
        for name in (
            "training_candidate_score_ids",
            "commercial_candidate_score_ids",
            "blocked_score_ids",
        ):
            payload[name] = list(payload[name])
        return sha256(_canonical_json_bytes(payload)).hexdigest()


def audit_ossq_scanned_score_rights(
    *,
    inventory: OssqScannedInventory,
    reviews: Iterable[OssqScoreRightsReview],
) -> OssqRightsAuditReceipt:
    """Audit every B8K scanned-track score without opening score/test bytes."""

    if not isinstance(inventory, OssqScannedInventory):
        raise TypeError("inventory must be OssqScannedInventory")
    review_values = tuple(reviews)
    if any(not isinstance(item, OssqScoreRightsReview) for item in review_values):
        raise OssqScoreRightsAuditError("reviews must contain OssqScoreRightsReview values")

    candidates = inventory.scanned_review_records
    candidate_by_id = {item.score_id: item for item in candidates}
    review_by_id: dict[str, OssqScoreRightsReview] = {}
    for review in review_values:
        if review.score_id in review_by_id:
            raise OssqScoreRightsAuditError("duplicate OSSQ rights review score_id")
        review_by_id[review.score_id] = review
    if set(review_by_id) != set(candidate_by_id):
        missing = sorted(set(candidate_by_id) - set(review_by_id), key=int)
        extra = sorted(set(review_by_id) - set(candidate_by_id), key=int)
        raise OssqScoreRightsAuditError(
            f"OSSQ rights-review coverage differs from the scanned candidate population; missing={missing[:3]} extra={extra[:3]}"
        )

    grouped_reviews: dict[str, list[OssqScoreRightsReview]] = {}
    for score_id, source in candidate_by_id.items():
        review = review_by_id[score_id]
        if review.imslp_id != source.imslp_id:
            raise OssqScoreRightsAuditError(f"OSSQ rights review IMSLP identity mismatch for score {score_id}")
        if review.source_type != source.source_type:
            raise OssqScoreRightsAuditError(f"OSSQ rights review source-type mismatch for score {score_id}")
        grouped_reviews.setdefault(source.imslp_id, []).append(review)

    shared_source_count = 0
    for imslp_id, group in grouped_reviews.items():
        if len(group) <= 1:
            continue
        shared_source_count += 1
        policy = group[0].policy_tuple()
        if any(item.policy_tuple() != policy for item in group[1:]):
            raise OssqScoreRightsAuditError(
                f"scores sharing {imslp_id} carry contradictory rights/provenance decisions"
            )

    ordered_reviews = tuple(sorted(review_values, key=lambda item: int(item.score_id)))
    training_ids = tuple(item.score_id for item in ordered_reviews if item.training_candidate)
    commercial_ids = tuple(item.score_id for item in ordered_reviews if item.commercial_candidate)
    blocked_ids = tuple(item.score_id for item in ordered_reviews if not item.training_candidate)
    manifest_payload = {
        "version": B8L_OSSQ_RIGHTS_AUDIT_VERSION,
        "inventory_fingerprint_sha256": inventory.fingerprint(),
        "b8k_source_selection_sha256": first_real_corpus_source_selection_fingerprint(),
        "publication_metadata_git_blob_sha": OSSQ_PUBLICATION_METADATA_GIT_BLOB_SHA,
        "reviews": [item.canonical_sha256() for item in ordered_reviews],
    }
    return OssqRightsAuditReceipt(
        inventory_fingerprint_sha256=inventory.fingerprint(),
        b8k_source_selection_sha256=first_real_corpus_source_selection_fingerprint(),
        review_manifest_sha256=sha256(_canonical_json_bytes(manifest_payload)).hexdigest(),
        reviewed_score_count=len(ordered_reviews),
        training_candidate_score_ids=training_ids,
        commercial_candidate_score_ids=commercial_ids,
        blocked_score_ids=blocked_ids,
        shared_imslp_source_count=shared_source_count,
    )
