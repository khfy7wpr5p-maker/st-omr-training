"""Fail-closed candidate boundary for probing B8Q VERIFIED OSSQ pairs through Stage 8.

This module does not fetch or inspect source bytes and does not admit, split, or
train on data. It binds the frozen B8P/B8R/B8Q evidence to the already reviewed
B8M/B8N rights/source identities and returns the only 14 pairs eligible for a
live Stage 8 quarantine/intake probe.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Final, Mapping

from st_omr_training.poly_v2_ossq_b8r_review_bridge import (
    OssqB8RReviewBridgeError,
    build_b8q_from_b8r,
)
from st_omr_training.poly_v2_ossq_pair_review_admission import review_receipt_to_json
from st_omr_training.musicxml_validator import diagnose_musicxml_xsd_failure
from st_omr_training.poly_v2_ossq_rights_evidence_batch1 import B8M_BATCH1_EVIDENCE
from st_omr_training.real_data_contract import RightsBasis
from st_omr_training.validator import ValidationResult


B8S_STAGE8_PROBE_VERSION: Final[str] = "st-omr-poly-v2-ossq-stage8-verified-probe-v1"
B8S_EXPECTED_B8P_RECEIPT_SHA256: Final[str] = (
    "3f9f2df43b287dd95e03eaf1ffc4c5a3c9e25bdced8f09675918bc9b1f99e0cf"
)
B8S_EXPECTED_B8R_RECEIPT_SHA256: Final[str] = (
    "4b880a6348897189e9851d74258ba8a07cb210bb7695d159319adad633e237c1"
)
B8S_EXPECTED_B8Q_RECEIPT_SHA256: Final[str] = (
    "c9345c106217c52ecdb4cbc325e5ef4e1b5d23dfbfc44190ee6124dd655b86fa"
)
B8S_EXPECTED_B8N_RECEIPT_SHA256: Final[str] = (
    "9f9b678e2365ec849cc19424b28d8dbdb435af3a5a9ef5b47cf7a460e72a801c"
)
B8S_EXPECTED_VERIFIED_COUNT: Final[int] = 14
B8S_EXCLUDED_SCORE_IDS: Final[frozenset[str]] = frozenset({"7397765"})


class OssqStage8ProbeError(ValueError):
    """Raised when Stage 8 probe candidacy cannot be proven exactly."""


@dataclass(frozen=True, slots=True)
class SemanticGateDiagnostic:
    issue_code: str
    issue_path: str
    source: str


def semantic_gate_diagnostic_from_error(
    error: BaseException,
    *,
    musicxml_bytes: bytes | None = None,
) -> SemanticGateDiagnostic:
    """Recover the first structured validation issue without exposing source bytes."""

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        validation = getattr(current, "validation", None)
        issues = getattr(validation, "issues", ())
        if isinstance(issues, tuple) and issues:
            first = issues[0]
            code = getattr(first, "code", None)
            path = getattr(first, "path", None)
            if isinstance(code, str) and code and isinstance(path, str) and path:
                if code == "musicxml.xsd_invalid" and musicxml_bytes is not None:
                    xsd = diagnose_musicxml_xsd_failure(musicxml_bytes)
                    if xsd is not None:
                        return SemanticGateDiagnostic(
                            issue_code=xsd.issue_code,
                            issue_path=xsd.issue_path,
                            source="xsd-error-log",
                        )
                return SemanticGateDiagnostic(
                    issue_code=code,
                    issue_path=path,
                    source="validation",
                )
        current = current.__cause__ or current.__context__

    return SemanticGateDiagnostic(
        issue_code="semantic-gate-unclassified",
        issue_path="$",
        source="exception-chain",
    )


@dataclass(frozen=True, slots=True)
class OssqStage8ProbeCandidate:
    score_id: str
    segment_id: str
    page_number: int
    imslp_id: str
    family_id: str
    source_document_sha256: str
    image_sha256: str
    musicxml_sha256: str
    provenance_evidence_sha256: str
    rights_evidence_sha256: str
    pairing_evidence_sha256: str
    pairing_method: str
    rights_basis: str
    rights_review: str
    training_allowed: bool
    commercial_use_allowed: bool
    redistribution_allowed: bool


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _page_number(segment_id: str) -> int:
    pieces = segment_id.split(":")
    if len(pieces) != 3 or not pieces[0].startswith("sq"):
        raise OssqStage8ProbeError("segment identity is outside the frozen OSSQ system contract")
    try:
        page = int(pieces[1])
    except ValueError as exc:
        raise OssqStage8ProbeError("segment page identity is invalid") from exc
    if page < 1:
        raise OssqStage8ProbeError("segment page identity must be positive")
    return page


def _rights_by_score() -> dict[str, object]:
    result: dict[str, object] = {}
    for evidence in B8M_BATCH1_EVIDENCE:
        for score_id in evidence.score_ids:
            if score_id in result:
                raise OssqStage8ProbeError("B8M rights evidence contains duplicate score coverage")
            result[score_id] = evidence
    return result


def _validate_b8n_pins(b8n_payload: object | None) -> dict[str, Mapping[str, object]]:
    if b8n_payload is None:
        # B8P already binds exact B8N identity; live runner supplies B8N again.
        return {}
    if not isinstance(b8n_payload, Mapping):
        raise OssqStage8ProbeError("B8N payload must be a mapping")
    if b8n_payload.get("receipt_sha256") != B8S_EXPECTED_B8N_RECEIPT_SHA256:
        raise OssqStage8ProbeError("B8N receipt identity differs from frozen Stage 8 probe")
    if any(
        b8n_payload.get(name) is not False
        for name in (
            "stage8_admission_authority",
            "test_artifact_bytes_accessed",
            "production_authority",
            "commercial_use_authority",
        )
    ):
        raise OssqStage8ProbeError("B8N authority boundary changed")
    pins = b8n_payload.get("pins")
    if not isinstance(pins, list):
        raise OssqStage8ProbeError("B8N pins must be a list")
    result: dict[str, Mapping[str, object]] = {}
    for pin in pins:
        if not isinstance(pin, Mapping) or not isinstance(pin.get("imslp_id"), str):
            raise OssqStage8ProbeError("B8N pin identity is invalid")
        result[str(pin["imslp_id"])] = pin
    return result


def build_stage8_probe_candidates(
    *,
    b8p_payload: object,
    b8r_payload: object,
    b8q_payload: object,
    b8n_payload: object | None = None,
) -> tuple[OssqStage8ProbeCandidate, ...]:
    """Return only exact B8Q VERIFIED pairs eligible for a non-authoritative live probe."""

    if not isinstance(b8p_payload, Mapping):
        raise OssqStage8ProbeError("B8P payload must be a mapping")
    if b8p_payload.get("receipt_sha256") != B8S_EXPECTED_B8P_RECEIPT_SHA256:
        raise OssqStage8ProbeError("B8P receipt identity differs from frozen Stage 8 probe")
    if not isinstance(b8r_payload, Mapping):
        raise OssqStage8ProbeError("B8R payload must be a mapping")
    if b8r_payload.get("receipt_sha256") != B8S_EXPECTED_B8R_RECEIPT_SHA256:
        raise OssqStage8ProbeError("B8R receipt identity differs from frozen Stage 8 probe")
    if not isinstance(b8q_payload, Mapping):
        raise OssqStage8ProbeError("B8Q payload must be a mapping")
    if b8q_payload.get("receipt_sha256") != B8S_EXPECTED_B8Q_RECEIPT_SHA256:
        raise OssqStage8ProbeError("B8Q receipt identity differs from frozen Stage 8 probe")

    raw_pairs = b8p_payload.get("pairs")
    raw_records = b8q_payload.get("records")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != 412:
        raise OssqStage8ProbeError("B8P pair population differs from exact 412 candidates")
    if not isinstance(raw_records, list) or len(raw_records) != 412:
        raise OssqStage8ProbeError("B8Q review population differs from exact 412 candidates")

    pair_by_segment: dict[str, Mapping[str, object]] = {}
    for pair in raw_pairs:
        if not isinstance(pair, Mapping) or not isinstance(pair.get("segment_id"), str):
            raise OssqStage8ProbeError("B8P pair identity is invalid")
        segment_id = str(pair["segment_id"])
        if segment_id in pair_by_segment:
            raise OssqStage8ProbeError("B8P pair identity contains duplicate segments")
        pair_by_segment[segment_id] = pair

    verified_records: list[Mapping[str, object]] = []
    for record in raw_records:
        if not isinstance(record, Mapping) or not isinstance(record.get("segment_id"), str):
            raise OssqStage8ProbeError("B8Q review record identity is invalid")
        segment_id = str(record["segment_id"])
        pair = pair_by_segment.get(segment_id)
        if pair is None:
            raise OssqStage8ProbeError("B8Q record is outside exact B8P pair identity")
        if (
            record.get("score_id") != pair.get("score_id")
            or record.get("image_sha256") != pair.get("image_sha256")
            or record.get("musicxml_sha256") != pair.get("musicxml_sha256")
        ):
            raise OssqStage8ProbeError("B8Q/B8P pair identity differs")
        if record.get("decision") == "verified":
            verified_records.append(record)

    if len(verified_records) != B8S_EXPECTED_VERIFIED_COUNT:
        raise OssqStage8ProbeError("B8Q VERIFIED population differs from exact 14 candidates")

    # Rebuild B8Q from frozen B8P+B8R and require byte-for-byte canonical equality.
    try:
        rebuilt = build_b8q_from_b8r(
            b8p_payload=b8p_payload,
            b8r_payload=b8r_payload,
        )
    except OssqB8RReviewBridgeError as exc:
        raise OssqStage8ProbeError(f"frozen B8P/B8R bridge evidence is invalid: {exc}") from exc
    if review_receipt_to_json(rebuilt) != _canonical_json(b8q_payload):
        raise OssqStage8ProbeError("B8Q canonical receipt differs from frozen B8P/B8R evidence")

    rights = _rights_by_score()
    b8n_pins = _validate_b8n_pins(b8n_payload)
    candidates: list[OssqStage8ProbeCandidate] = []

    for record in verified_records:
        segment_id = str(record["segment_id"])
        pair = pair_by_segment[segment_id]
        score_id = str(record["score_id"])
        if score_id in B8S_EXCLUDED_SCORE_IDS:
            raise OssqStage8ProbeError("upstream blocked score entered Stage 8 probe candidates")
        evidence = rights.get(score_id)
        if evidence is None:
            raise OssqStage8ProbeError("B8Q VERIFIED score lacks independent B8M rights evidence")

        imslp_id = str(pair.get("imslp_id"))
        if evidence.imslp_id != imslp_id:
            raise OssqStage8ProbeError("B8M/B8P source identity differs")
        source_sha = str(pair.get("source_document_sha256"))
        if b8n_pins:
            pin = b8n_pins.get(imslp_id)
            if pin is None or pin.get("source_sha256") != source_sha or score_id not in pin.get("score_ids", ()):
                raise OssqStage8ProbeError("B8N/B8P source byte identity differs")

        if evidence.training_scope != "research-training-candidate-only":
            raise OssqStage8ProbeError("B8M training scope escalated unexpectedly")

        family_id = f"ossq-source-{source_sha}"
        candidates.append(
            OssqStage8ProbeCandidate(
                score_id=score_id,
                segment_id=segment_id,
                page_number=_page_number(segment_id),
                imslp_id=imslp_id,
                family_id=family_id,
                source_document_sha256=source_sha,
                image_sha256=str(record["image_sha256"]),
                musicxml_sha256=str(record["musicxml_sha256"]),
                provenance_evidence_sha256=evidence.provenance_sha256(),
                rights_evidence_sha256=evidence.canonical_sha256(),
                pairing_evidence_sha256=str(record["review_evidence_sha256"]),
                pairing_method=str(record["method"]),
                rights_basis=RightsBasis.PUBLIC_DOMAIN.value,
                rights_review="approved",
                training_allowed=True,
                commercial_use_allowed=False,
                redistribution_allowed=False,
            )
        )

    ordered = tuple(sorted(candidates, key=lambda item: item.segment_id))
    if len({item.segment_id for item in ordered}) != len(ordered):
        raise OssqStage8ProbeError("Stage 8 probe candidate population contains duplicate segments")
    if len({item.family_id for item in ordered}) != 4:
        raise OssqStage8ProbeError("Stage 8 probe source-family population differs from frozen four families")
    return ordered
