"""Deterministic TR-POLY-09B8R -> B8Q independent-review bridge.

The bridge consumes only the committed hash-only B8P materialization receipt and
the frozen B8R independent cross-render receipt. It may promote a B8Q record to
VERIFIED only when the corresponding frozen B8R observation is a
VERIFIED_CANDIDATE. REVIEW_REQUIRED B8R observations remain unreviewed in B8Q.

This module grants no Stage 8 admission, TRAIN/VALIDATION assignment, TEST,
production, or commercial-use authority.
"""
from __future__ import annotations

from hashlib import sha256
import json
import re
from typing import Final, Mapping, Sequence

from st_omr_training.poly_v2_ossq_pair_review_admission import (
    B8Q_EXPECTED_B8P_RECEIPT_SHA256,
    B8Q_EXPECTED_PAIR_COUNT,
    B8Q_READY_SCORE_IDS,
    OssqPairReviewError,
    OssqPairReviewReceipt,
    OssqPairReviewRecord,
    PairReviewDecision,
    PairReviewMethod,
    apply_independent_pair_reviews,
    build_unreviewed_b8q_receipt,
)


B8R_TO_B8Q_BRIDGE_VERSION: Final[str] = "st-omr-poly-v2-ossq-b8r-to-b8q-review-bridge-v1"
B8R_TO_B8Q_EXPECTED_B8R_VERSION: Final[str] = (
    "st-omr-poly-v2-ossq-independent-cross-render-audit-v1"
)
B8R_TO_B8Q_EXPECTED_B8R_RECEIPT_SHA256: Final[str] = (
    "4b880a6348897189e9851d74258ba8a07cb210bb7695d159319adad633e237c1"
)
B8R_TO_B8Q_EXPECTED_POLICY_FINGERPRINT: Final[str] = (
    "9d6a6d25843bdd9bbcfd1a9b6dccec59471646d0b59361e034e2e959323b8397"
)
B8R_TO_B8Q_EXPECTED_REVIEWER_IDENTITY_SHA256: Final[str] = (
    "35fd9e0abb378cd6bbd131a31468b248d13a9353b700193bc13ac0fcea6cdaac"
)
B8R_TO_B8Q_EXPECTED_DECISION_COUNTS: Final[tuple[tuple[str, int], ...]] = (
    ("verified-candidate", 14),
    ("review-required", 398),
)
B8R_TO_B8Q_EXPECTED_SCORE_CROSS_CONFLICTS: Final[tuple[tuple[str, int], ...]] = tuple(
    (score_id, 0) for score_id in B8Q_READY_SCORE_IDS
)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class OssqB8RReviewBridgeError(ValueError):
    """Raised when frozen B8R evidence cannot safely become B8Q review evidence."""


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
        raise OssqB8RReviewBridgeError(f"{name} must be lowercase SHA-256")
    return value


def _pairs_tuple(name: str, value: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, list):
        raise OssqB8RReviewBridgeError(f"{name} must be a list")
    parsed: list[tuple[str, int]] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], int)
            or isinstance(item[1], bool)
        ):
            raise OssqB8RReviewBridgeError(f"{name} contains an invalid entry")
        parsed.append((item[0], item[1]))
    return tuple(parsed)


def _recompute_b8r_receipt_sha256(payload: Mapping[str, object]) -> str:
    copied = dict(payload)
    embedded = copied.pop("receipt_sha256", None)
    _require_sha256("B8R receipt_sha256", embedded)
    return sha256(_canonical_bytes(copied)).hexdigest()


def _validate_b8r_payload(
    *,
    b8r_payload: object,
    expected_by_segment: Mapping[str, OssqPairReviewRecord],
) -> tuple[Mapping[str, object], ...]:
    if not isinstance(b8r_payload, Mapping):
        raise OssqB8RReviewBridgeError("B8R payload must be a mapping")

    if b8r_payload.get("version") != B8R_TO_B8Q_EXPECTED_B8R_VERSION:
        raise OssqB8RReviewBridgeError("B8R version differs from frozen bridge input")
    if b8r_payload.get("b8p_receipt_sha256") != B8Q_EXPECTED_B8P_RECEIPT_SHA256:
        raise OssqB8RReviewBridgeError("B8R B8P receipt identity differs from frozen B8Q input")
    if b8r_payload.get("policy_fingerprint") != B8R_TO_B8Q_EXPECTED_POLICY_FINGERPRINT:
        raise OssqB8RReviewBridgeError("B8R policy fingerprint drift")
    if (
        b8r_payload.get("reviewer_identity_sha256")
        != B8R_TO_B8Q_EXPECTED_REVIEWER_IDENTITY_SHA256
    ):
        raise OssqB8RReviewBridgeError("B8R reviewer identity drift")

    declared_counts = _pairs_tuple("B8R decision counts", b8r_payload.get("decision_counts"))
    if declared_counts != B8R_TO_B8Q_EXPECTED_DECISION_COUNTS:
        raise OssqB8RReviewBridgeError("B8R decision counts differ from frozen bridge input")

    declared_conflicts = _pairs_tuple(
        "B8R score cross-conflicts",
        b8r_payload.get("score_cross_conflicts"),
    )
    if declared_conflicts != B8R_TO_B8Q_EXPECTED_SCORE_CROSS_CONFLICTS:
        raise OssqB8RReviewBridgeError("B8R score cross-conflict evidence is non-zero or drifted")

    for boundary in (
        "raw_pair_bytes_persisted_as_evidence",
        "raw_pdf_bytes_persisted_as_evidence",
        "stage8_admission_authority",
        "train_validation_assignment_authority",
        "test_artifact_bytes_accessed",
        "production_authority",
        "commercial_use_authority",
    ):
        if b8r_payload.get(boundary) is not False:
            raise OssqB8RReviewBridgeError(f"B8R authority boundary changed: {boundary}")

    observations = b8r_payload.get("observations")
    if not isinstance(observations, list) or len(observations) != B8Q_EXPECTED_PAIR_COUNT:
        raise OssqB8RReviewBridgeError("B8R must contain exactly 412 observations")

    seen: set[str] = set()
    actual_counts = {"verified-candidate": 0, "review-required": 0}
    validated: list[Mapping[str, object]] = []

    for index, raw in enumerate(observations):
        if not isinstance(raw, Mapping):
            raise OssqB8RReviewBridgeError(f"B8R observation {index} must be a mapping")

        score_id = raw.get("score_id")
        segment_id = raw.get("segment_id")
        image_sha256 = raw.get("image_sha256")
        musicxml_sha256 = raw.get("musicxml_sha256")
        decision = raw.get("decision")

        if not isinstance(score_id, str) or not isinstance(segment_id, str):
            raise OssqB8RReviewBridgeError(f"B8R observation {index} lacks score/segment identity")
        if segment_id in seen:
            raise OssqB8RReviewBridgeError("B8R observation population contains duplicate segments")
        seen.add(segment_id)

        expected = expected_by_segment.get(segment_id)
        if expected is None or (
            expected.score_id,
            expected.image_sha256,
            expected.musicxml_sha256,
        ) != (score_id, image_sha256, musicxml_sha256):
            raise OssqB8RReviewBridgeError(
                "B8R observation differs from exact B8P pair identity"
            )

        _require_sha256("B8R image_sha256", image_sha256)
        _require_sha256("B8R musicxml_sha256", musicxml_sha256)
        _require_sha256(
            "B8R independent_render_png_sha256",
            raw.get("independent_render_png_sha256"),
        )
        _require_sha256("B8R evidence_sha256", raw.get("evidence_sha256"))

        if decision not in actual_counts:
            raise OssqB8RReviewBridgeError("B8R observation has an unsupported decision")
        actual_counts[str(decision)] += 1

        if raw.get("score_cross_conflict_count") != 0:
            raise OssqB8RReviewBridgeError(
                "B8R observation with non-zero cross-conflict cannot be promoted"
            )
        validated.append(raw)

    if seen != set(expected_by_segment):
        raise OssqB8RReviewBridgeError("B8R observation population differs from exact B8P segments")
    if tuple(actual_counts.items()) != B8R_TO_B8Q_EXPECTED_DECISION_COUNTS:
        raise OssqB8RReviewBridgeError("B8R actual decision counts differ from frozen bridge input")

    embedded_receipt = b8r_payload.get("receipt_sha256")
    if embedded_receipt != B8R_TO_B8Q_EXPECTED_B8R_RECEIPT_SHA256:
        raise OssqB8RReviewBridgeError("B8R receipt identity differs from frozen bridge input")
    if _recompute_b8r_receipt_sha256(b8r_payload) != B8R_TO_B8Q_EXPECTED_B8R_RECEIPT_SHA256:
        raise OssqB8RReviewBridgeError("B8R canonical receipt fingerprint mismatch")

    return tuple(sorted(validated, key=lambda item: str(item["segment_id"])))


def build_b8q_from_b8r(
    *,
    b8p_payload: object,
    b8r_payload: object,
) -> OssqPairReviewReceipt:
    """Convert only frozen B8R VERIFIED_CANDIDATE evidence into B8Q VERIFIED records."""

    try:
        baseline = build_unreviewed_b8q_receipt(b8p_payload)
    except OssqPairReviewError as exc:
        raise OssqB8RReviewBridgeError(f"B8P input rejected by B8Q contract: {exc}") from exc

    expected_by_segment = {item.segment_id: item for item in baseline.records}
    observations = _validate_b8r_payload(
        b8r_payload=b8r_payload,
        expected_by_segment=expected_by_segment,
    )
    observation_by_segment = {
        str(item["segment_id"]): item
        for item in observations
    }

    reviews: list[OssqPairReviewRecord] = []
    for base in baseline.records:
        observation = observation_by_segment[base.segment_id]
        if observation["decision"] == "verified-candidate":
            reviews.append(
                OssqPairReviewRecord(
                    score_id=base.score_id,
                    segment_id=base.segment_id,
                    image_sha256=base.image_sha256,
                    musicxml_sha256=base.musicxml_sha256,
                    decision=PairReviewDecision.VERIFIED,
                    method=PairReviewMethod.INDEPENDENT_CROSS_RENDER,
                    review_evidence_sha256=str(observation["evidence_sha256"]),
                    reviewer_identity_sha256=B8R_TO_B8Q_EXPECTED_REVIEWER_IDENTITY_SHA256,
                    reason_code="independent-pair-review-passed",
                )
            )
        else:
            reviews.append(base)

    try:
        receipt = apply_independent_pair_reviews(
            b8p_payload=b8p_payload,
            reviews=tuple(reviews),
        )
    except OssqPairReviewError as exc:
        raise OssqB8RReviewBridgeError(f"B8Q review contract rejected bridge output: {exc}") from exc

    if dict(receipt.review_count_by_decision) != {
        "verified": 14,
        "review-required": 398,
        "blocked": 0,
    }:
        raise OssqB8RReviewBridgeError("B8Q bridge output counts differ from frozen expectation")
    if not receipt.independent_pairing_review_authority:
        raise OssqB8RReviewBridgeError("B8Q independent review authority was not established")
    if any(
        (
            receipt.stage8_admission_authority,
            receipt.train_validation_assignment_authority,
            receipt.test_artifact_bytes_accessed,
            receipt.production_authority,
            receipt.commercial_use_authority,
        )
    ):
        raise OssqB8RReviewBridgeError("B8Q bridge widened an authority boundary")

    return receipt
