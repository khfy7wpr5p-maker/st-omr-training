"""Stage 8-0 real-data admission metadata contract and fail-closed validator.

This module intentionally does not ingest files, open a test split, load a model,
or run training.  It freezes the metadata and leakage rules that a later Stage 8
intake implementation must satisfy before any real sample can become eligible
for train/validation use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
import re
from typing import Final


REAL_DATA_SCHEMA_VERSION: Final[str] = "st-real-data-admission-v1"
REAL_DATA_SOURCE_CLASS: Final[str] = "real"
REAL_DATA_SPLIT_POLICY: Final[str] = "real-family-exclusive-v1"
MAX_REAL_DATA_SAMPLES: Final[int] = 100_000
MAX_PAGE_NUMBER: Final[int] = 256

STAGE7C_ARTIFACT_ID: Final[int] = 9177923796
STAGE7C_ARTIFACT_EXPIRES_AT: Final[str] = "2026-09-12T10:44:21Z"
STAGE7C_CHECKPOINT_SHA256: Final[str] = (
    "75c33cefeb970305f5f9171b3274dbe8b785cbcf1e8d6851de7848e66d24efa4"
)
STAGE7C_MODEL_STATE_SHA256: Final[str] = (
    "79d354f2582f3f7cc106564b40f07a6027b62b2a74c9efe13a7b2437a6c3f7a0"
)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")


class RealDataContractError(ValueError):
    """Raised when a Stage 8-0 contract input fails closed."""


class SealedTestAccessError(RealDataContractError):
    """Raised when Stage 8 development code attempts to select test records."""


class RealDataSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class RealDataOrigin(str, Enum):
    CURATED = "curated"
    USER_SUBMISSION = "user-submission"
    SCOREMOSAIC_UPLOAD = "scoremosaic-upload"
    TEACHER_CORRECTION = "teacher-correction"


class RightsBasis(str, Enum):
    PUBLIC_DOMAIN = "public-domain"
    OPEN_LICENSE = "open-license"
    EXPLICIT_PERMISSION = "explicit-permission"
    COPYRIGHT_OWNER = "copyright-owner"


class ReviewState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class PairingState(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"


class AdmissionState(str, Enum):
    QUARANTINED = "quarantined"
    ADMITTED = "admitted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class RealDataValidationIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class RealDataValidationResult:
    issues: tuple[RealDataValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class RealDataSample:
    """Immutable Stage 8-0 metadata for one real image/MusicXML pair.

    The record contains hashes and review evidence only. It does not contain the
    image, MusicXML, personal data, license text, or model bytes.
    """

    sample_id: str
    family_id: str
    split: RealDataSplit
    page_number: int
    origin: RealDataOrigin
    rights_basis: RightsBasis
    source_document_sha256: str
    image_sha256: str
    musicxml_sha256: str
    semantic_fingerprint: str
    provenance_evidence_sha256: str
    rights_evidence_sha256: str | None
    pairing_evidence_sha256: str | None
    explicit_training_permission_sha256: str | None
    privacy_review_evidence_sha256: str | None
    rights_review: ReviewState
    pairing_review: PairingState
    admission_state: AdmissionState

    def __post_init__(self) -> None:
        _require_hex64("sample_id", self.sample_id)
        _require_identifier("family_id", self.family_id)
        if not isinstance(self.split, RealDataSplit):
            raise RealDataContractError("split must be RealDataSplit")
        if not isinstance(self.page_number, int) or isinstance(self.page_number, bool):
            raise RealDataContractError("page_number must be an integer")
        if not 1 <= self.page_number <= MAX_PAGE_NUMBER:
            raise RealDataContractError("page_number is outside the Stage 8-0 range")
        if not isinstance(self.origin, RealDataOrigin):
            raise RealDataContractError("origin must be RealDataOrigin")
        if not isinstance(self.rights_basis, RightsBasis):
            raise RealDataContractError("rights_basis must be RightsBasis")
        for name in (
            "source_document_sha256",
            "image_sha256",
            "musicxml_sha256",
            "semantic_fingerprint",
            "provenance_evidence_sha256",
        ):
            _require_hex64(name, getattr(self, name))
        for name in (
            "rights_evidence_sha256",
            "pairing_evidence_sha256",
            "explicit_training_permission_sha256",
            "privacy_review_evidence_sha256",
        ):
            _require_optional_hex64(name, getattr(self, name))
        if not isinstance(self.rights_review, ReviewState):
            raise RealDataContractError("rights_review must be ReviewState")
        if not isinstance(self.pairing_review, PairingState):
            raise RealDataContractError("pairing_review must be PairingState")
        if not isinstance(self.admission_state, AdmissionState):
            raise RealDataContractError("admission_state must be AdmissionState")


@dataclass(frozen=True, slots=True)
class RealDataManifest:
    dataset_name: str
    dataset_version: str
    samples: tuple[RealDataSample, ...]
    sealed_test_manifest_sha256: str
    schema_version: str = REAL_DATA_SCHEMA_VERSION
    source_class: str = REAL_DATA_SOURCE_CLASS
    split_policy: str = REAL_DATA_SPLIT_POLICY

    def __post_init__(self) -> None:
        _require_identifier("dataset_name", self.dataset_name)
        if not isinstance(self.dataset_version, str) or _VERSION_RE.fullmatch(self.dataset_version) is None:
            raise RealDataContractError("dataset_version must match the bounded version contract")
        if not isinstance(self.samples, tuple):
            raise RealDataContractError("samples must be an immutable tuple")
        _require_hex64("sealed_test_manifest_sha256", self.sealed_test_manifest_sha256)
        if len(self.samples) > MAX_REAL_DATA_SAMPLES:
            raise RealDataContractError("manifest exceeds the Stage 8-0 sample-count limit")
        if self.schema_version != REAL_DATA_SCHEMA_VERSION:
            raise RealDataContractError("unsupported real-data schema version")
        if self.source_class != REAL_DATA_SOURCE_CLASS:
            raise RealDataContractError("Stage 8-0 accepts real source_class only")
        if self.split_policy != REAL_DATA_SPLIT_POLICY:
            raise RealDataContractError("unsupported real-data split policy")


def _require_hex64(name: str, value: object) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise RealDataContractError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_optional_hex64(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _require_hex64(name, value)


def _require_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise RealDataContractError(f"{name} must match the bounded identifier contract")
    return value


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def real_data_sample_id(
    *,
    family_id: str,
    page_number: int,
    source_document_sha256: str,
    image_sha256: str,
    musicxml_sha256: str,
    semantic_fingerprint: str,
) -> str:
    """Compute content identity independently of split and review-state changes."""

    _require_identifier("family_id", family_id)
    if not isinstance(page_number, int) or isinstance(page_number, bool) or not 1 <= page_number <= MAX_PAGE_NUMBER:
        raise RealDataContractError("page_number is outside the Stage 8-0 range")
    for name, value in (
        ("source_document_sha256", source_document_sha256),
        ("image_sha256", image_sha256),
        ("musicxml_sha256", musicxml_sha256),
        ("semantic_fingerprint", semantic_fingerprint),
    ):
        _require_hex64(name, value)
    payload = {
        "schema_version": REAL_DATA_SCHEMA_VERSION,
        "family_id": family_id,
        "page_number": page_number,
        "source_document_sha256": source_document_sha256,
        "image_sha256": image_sha256,
        "musicxml_sha256": musicxml_sha256,
        "semantic_fingerprint": semantic_fingerprint,
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _issue(code: str, path: str, message: str) -> RealDataValidationIssue:
    return RealDataValidationIssue(code=code, path=path, message=message)


def _sorted_result(issues: list[RealDataValidationIssue]) -> RealDataValidationResult:
    return RealDataValidationResult(tuple(sorted(issues, key=lambda item: (item.code, item.path, item.message))))


def validate_quarantine_record(record: object, *, path: str = "sample") -> RealDataValidationResult:
    """Validate structural quarantine metadata without making it training-eligible."""

    issues: list[RealDataValidationIssue] = []
    if not isinstance(record, RealDataSample):
        return _sorted_result([_issue("sample.type", path, "sample must be RealDataSample")])

    expected_id = real_data_sample_id(
        family_id=record.family_id,
        page_number=record.page_number,
        source_document_sha256=record.source_document_sha256,
        image_sha256=record.image_sha256,
        musicxml_sha256=record.musicxml_sha256,
        semantic_fingerprint=record.semantic_fingerprint,
    )
    if record.sample_id != expected_id:
        issues.append(_issue("lineage.sample_id", f"{path}.sample_id", "does not match immutable content identity"))
    if record.admission_state is not AdmissionState.QUARANTINED:
        issues.append(_issue("quarantine.state", f"{path}.admission_state", "quarantine record must remain quarantined"))
    return _sorted_result(issues)


def validate_real_data_sample(record: object, *, path: str = "sample") -> RealDataValidationResult:
    """Validate one training-eligible Stage 8 real-data metadata record."""

    issues: list[RealDataValidationIssue] = []
    if not isinstance(record, RealDataSample):
        return _sorted_result([_issue("sample.type", path, "sample must be RealDataSample")])

    expected_id = real_data_sample_id(
        family_id=record.family_id,
        page_number=record.page_number,
        source_document_sha256=record.source_document_sha256,
        image_sha256=record.image_sha256,
        musicxml_sha256=record.musicxml_sha256,
        semantic_fingerprint=record.semantic_fingerprint,
    )
    if record.sample_id != expected_id:
        issues.append(_issue("lineage.sample_id", f"{path}.sample_id", "does not match immutable content identity"))
    if record.admission_state is not AdmissionState.ADMITTED:
        issues.append(_issue("admission.state", f"{path}.admission_state", "training manifest accepts admitted samples only"))
    if record.rights_review is not ReviewState.APPROVED:
        issues.append(_issue("rights.review", f"{path}.rights_review", "rights review must be approved"))
    if record.rights_evidence_sha256 is None:
        issues.append(_issue("rights.evidence", f"{path}.rights_evidence_sha256", "approved rights evidence is required"))
    if record.pairing_review is not PairingState.VERIFIED:
        issues.append(_issue("pairing.review", f"{path}.pairing_review", "image/MusicXML pairing must be independently verified"))
    if record.pairing_evidence_sha256 is None:
        issues.append(_issue("pairing.evidence", f"{path}.pairing_evidence_sha256", "pairing evidence is required"))

    user_derived = record.origin in {
        RealDataOrigin.USER_SUBMISSION,
        RealDataOrigin.SCOREMOSAIC_UPLOAD,
        RealDataOrigin.TEACHER_CORRECTION,
    }
    if user_derived and record.explicit_training_permission_sha256 is None:
        issues.append(
            _issue(
                "permission.explicit_training",
                f"{path}.explicit_training_permission_sha256",
                "user-derived material requires separate explicit training permission",
            )
        )
    if user_derived and record.privacy_review_evidence_sha256 is None:
        issues.append(
            _issue(
                "privacy.review",
                f"{path}.privacy_review_evidence_sha256",
                "user-derived material requires privacy review evidence",
            )
        )
    if record.rights_basis is RightsBasis.EXPLICIT_PERMISSION and record.explicit_training_permission_sha256 is None:
        issues.append(
            _issue(
                "permission.missing",
                f"{path}.explicit_training_permission_sha256",
                "explicit-permission rights basis requires permission evidence",
            )
        )

    return _sorted_result(issues)


def validate_real_data_manifest(manifest: object) -> RealDataValidationResult:
    """Validate admitted real-data metadata and veto duplicate/split leakage."""

    issues: list[RealDataValidationIssue] = []
    if not isinstance(manifest, RealDataManifest):
        return _sorted_result([_issue("manifest.type", "manifest", "manifest must be RealDataManifest")])
    if not manifest.samples:
        issues.append(_issue("manifest.empty", "manifest.samples", "admitted real-data manifest must contain samples"))
    if len(manifest.samples) > MAX_REAL_DATA_SAMPLES:
        issues.append(_issue("manifest.sample_limit", "manifest.samples", "manifest exceeds sample-count limit"))

    seen_sample_ids: dict[str, int] = {}
    seen_image_hashes: dict[str, int] = {}
    seen_pair_hashes: dict[tuple[str, str], int] = {}
    family_split: dict[str, RealDataSplit] = {}
    source_family: dict[str, str] = {}
    source_split: dict[str, RealDataSplit] = {}
    target_family: dict[str, str] = {}
    target_split: dict[str, RealDataSplit] = {}
    semantic_family: dict[str, str] = {}
    semantic_split: dict[str, RealDataSplit] = {}
    split_counts = {RealDataSplit.TRAIN: 0, RealDataSplit.VALIDATION: 0}

    for index, sample in enumerate(manifest.samples):
        path = f"manifest.samples[{index}]"
        if not isinstance(sample, RealDataSample):
            issues.append(_issue("sample.type", path, "sample must be RealDataSample"))
            continue
        if sample.split is RealDataSplit.TEST:
            issues.append(_issue("test.sealed", f"{path}.split", "test sample metadata must remain outside the Stage 8 development manifest"))
            continue
        sample_result = validate_real_data_sample(sample, path=path)
        issues.extend(sample_result.issues)
        split_counts[sample.split] += 1

        for code, value, seen in (
            ("duplicate.sample_id", sample.sample_id, seen_sample_ids),
            ("duplicate.image_sha256", sample.image_sha256, seen_image_hashes),
        ):
            if value in seen:
                issues.append(_issue(code, path, f"duplicates sample at index {seen[value]}"))
            else:
                seen[value] = index

        pair = (sample.image_sha256, sample.musicxml_sha256)
        if pair in seen_pair_hashes:
            issues.append(_issue("duplicate.image_target_pair", path, f"duplicates sample at index {seen_pair_hashes[pair]}"))
        else:
            seen_pair_hashes[pair] = index

        prior_family_split = family_split.get(sample.family_id)
        if prior_family_split is not None and prior_family_split is not sample.split:
            issues.append(_issue("leakage.family_split", path, "one family appears in multiple splits"))
        else:
            family_split.setdefault(sample.family_id, sample.split)

        for family_code, split_code, key, family_mapping, split_mapping in (
            ("leakage.source_family", "leakage.source_split", sample.source_document_sha256, source_family, source_split),
            ("leakage.target_family", "leakage.target_split", sample.musicxml_sha256, target_family, target_split),
            ("leakage.semantic_family", "leakage.semantic_split", sample.semantic_fingerprint, semantic_family, semantic_split),
        ):
            prior_family = family_mapping.get(key)
            prior_split = split_mapping.get(key)
            if prior_family is not None and prior_family != sample.family_id:
                issues.append(_issue(family_code, path, "equivalent content appears under multiple families"))
            if prior_split is not None and prior_split is not sample.split:
                issues.append(_issue(split_code, path, "equivalent content appears in multiple splits"))
            family_mapping.setdefault(key, sample.family_id)
            split_mapping.setdefault(key, sample.split)

    for split, count in split_counts.items():
        if count == 0:
            issues.append(_issue("manifest.missing_split", f"manifest.split.{split.value}", "development manifest requires both train and validation"))

    return _sorted_result(issues)


def _sample_payload(sample: RealDataSample) -> dict[str, object]:
    payload = asdict(sample)
    payload["split"] = sample.split.value
    payload["origin"] = sample.origin.value
    payload["rights_basis"] = sample.rights_basis.value
    payload["rights_review"] = sample.rights_review.value
    payload["pairing_review"] = sample.pairing_review.value
    payload["admission_state"] = sample.admission_state.value
    return payload


def canonical_real_data_manifest_bytes(manifest: RealDataManifest) -> bytes:
    result = validate_real_data_manifest(manifest)
    if not result.is_valid:
        first = result.issues[0]
        raise RealDataContractError(f"manifest is invalid: {first.code} at {first.path}: {first.message}")
    samples = sorted(manifest.samples, key=lambda item: (item.family_id, item.page_number, item.sample_id, item.split.value))
    payload = {
        "schema_version": manifest.schema_version,
        "source_class": manifest.source_class,
        "split_policy": manifest.split_policy,
        "dataset_name": manifest.dataset_name,
        "dataset_version": manifest.dataset_version,
        "sealed_test_manifest_sha256": manifest.sealed_test_manifest_sha256,
        "samples": [_sample_payload(sample) for sample in samples],
    }
    return _canonical_json_bytes(payload)


def real_data_manifest_sha256(manifest: RealDataManifest) -> str:
    return sha256(canonical_real_data_manifest_bytes(manifest)).hexdigest()


def select_stage8_development_records(
    manifest: RealDataManifest,
    split: RealDataSplit,
) -> tuple[RealDataSample, ...]:
    """Expose only admitted train/validation metadata; Stage 8 test stays sealed."""

    if split is RealDataSplit.TEST:
        raise SealedTestAccessError("Stage 8 test records are sealed until the Stage 9 benchmark gate")
    result = validate_real_data_manifest(manifest)
    if not result.is_valid:
        first = result.issues[0]
        raise RealDataContractError(f"manifest is invalid: {first.code} at {first.path}: {first.message}")
    if split not in (RealDataSplit.TRAIN, RealDataSplit.VALIDATION):
        raise RealDataContractError("split must be train or validation during Stage 8 development")
    return tuple(sample for sample in manifest.samples if sample.split is split)
