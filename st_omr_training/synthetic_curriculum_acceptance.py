"""Validate the canonical export receipt for the frozen synthetic curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Final

EXPECTED_SOURCE_COMMIT: Final[str] = "adc8139539d3c8cd6a2e3ee4ce4de6db4dcfeb90"
EXPECTED_CONFIG_FINGERPRINT: Final[str] = "154bf1c3e6dfe4e6db096f8b668f29df0623cfd38352b89a04d295764c7458cb"
EXPECTED_BUILD_ID: Final[str] = "d9320e362f162cd2ace2a830a7b93e0c21ceba2d51a4e95ef1c7a9b11a108352"
EXPECTED_MANIFEST_SHA256: Final[str] = "44a963cd7dbc612fa29c2953ea8b2c8776d89ce470074e8f8b3fe25c6e165f34"
EXPECTED_TRANSPORT_SHA256: Final[str] = "4a9f3bb337ef99386081dff29c4c1fc3047dc3ada4db13c93b6254e680918e2b"
MAX_EVIDENCE_BYTES: Final[int] = 256 * 1024


class SyntheticCurriculumAcceptanceError(RuntimeError):
    pass


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"


def _reject_constant(value: str) -> None:
    raise SyntheticCurriculumAcceptanceError(f"non-finite JSON constant: {value}")


@dataclass(frozen=True, slots=True)
class SyntheticCurriculumExportReceipt:
    build_id: str
    manifest_sha256: str
    config_fingerprint: str
    transport_sha256: str
    evidence_sha256: str


def verify_synthetic_curriculum_export_evidence(evidence_bytes: object) -> SyntheticCurriculumExportReceipt:
    if not isinstance(evidence_bytes, bytes):
        raise TypeError("evidence_bytes must be bytes")
    if not evidence_bytes or len(evidence_bytes) > MAX_EVIDENCE_BYTES:
        raise SyntheticCurriculumAcceptanceError("invalid evidence byte length")
    try:
        payload = json.loads(evidence_bytes.decode("ascii"), parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SyntheticCurriculumAcceptanceError("evidence is not valid ASCII JSON") from exc
    if not isinstance(payload, dict) or _canonical(payload) != evidence_bytes:
        raise SyntheticCurriculumAcceptanceError("evidence is not canonical JSON")

    if payload.get("schema_version") != "st-omr-colab-synthetic-export-v1":
        raise SyntheticCurriculumAcceptanceError("schema mismatch")
    if payload.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise SyntheticCurriculumAcceptanceError("source commit mismatch")

    plan = payload.get("plan")
    build = payload.get("build")
    runtime = payload.get("runtime")
    if not isinstance(plan, dict) or not isinstance(build, dict) or not isinstance(runtime, dict):
        raise SyntheticCurriculumAcceptanceError("missing structured evidence")

    if plan.get("config_fingerprint") != EXPECTED_CONFIG_FINGERPRINT:
        raise SyntheticCurriculumAcceptanceError("plan fingerprint mismatch")
    if plan.get("family_count") != 512:
        raise SyntheticCurriculumAcceptanceError("family count mismatch")
    if plan.get("family_split_counts") != {"test": 51, "train": 410, "validation": 51}:
        raise SyntheticCurriculumAcceptanceError("family split mismatch")
    profile_counts = plan.get("family_profile_counts")
    if not isinstance(profile_counts, dict) or set(profile_counts.values()) != {64} or len(profile_counts) != 8:
        raise SyntheticCurriculumAcceptanceError("family profile balance mismatch")

    if build.get("config_fingerprint") != EXPECTED_CONFIG_FINGERPRINT:
        raise SyntheticCurriculumAcceptanceError("build fingerprint mismatch")
    if build.get("build_id") != EXPECTED_BUILD_ID:
        raise SyntheticCurriculumAcceptanceError("build id mismatch")
    if build.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise SyntheticCurriculumAcceptanceError("manifest mismatch")
    if (build.get("sample_count"), build.get("target_count"), build.get("image_count")) != (1536, 512, 1536):
        raise SyntheticCurriculumAcceptanceError("artifact count mismatch")

    packages = runtime.get("packages")
    if packages != {"CairoSVG": "2.8.2", "Pillow": "12.3.0", "lxml": "6.1.1", "verovio": "6.2.1"}:
        raise SyntheticCurriculumAcceptanceError("runtime package mismatch")
    if payload.get("transport_sha256") != EXPECTED_TRANSPORT_SHA256:
        raise SyntheticCurriculumAcceptanceError("transport mismatch")

    return SyntheticCurriculumExportReceipt(
        build_id=EXPECTED_BUILD_ID,
        manifest_sha256=EXPECTED_MANIFEST_SHA256,
        config_fingerprint=EXPECTED_CONFIG_FINGERPRINT,
        transport_sha256=EXPECTED_TRANSPORT_SHA256,
        evidence_sha256=sha256(evidence_bytes).hexdigest(),
    )
