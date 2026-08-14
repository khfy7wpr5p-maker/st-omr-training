"""Strict Stage 7-D export gate layered on the canonical evidence validator."""

from __future__ import annotations

import json

from .synthetic_curriculum_acceptance import (
    SyntheticCurriculumAcceptanceError,
    SyntheticCurriculumExportReceipt,
    verify_synthetic_curriculum_export_evidence,
)

_EXPECTED_REPOSITORY = "https://github.com/khfy7wpr5p-maker/st-omr-training.git"
_EXPECTED_ARCHIVE = "st-omr-synthetic-curriculum-v1-d9320e362f162cd2.tar.gz"
_EXPECTED_PROFILE = "st-synthetic-curriculum-v1"
_EXPECTED_PROFILE_COUNTS = {
    "chord-only": 64,
    "mixed": 64,
    "no-accidentals": 64,
    "note-only": 64,
    "rest-only": 64,
    "time-2-4": 64,
    "time-3-4": 64,
    "time-4-4": 64,
}


def verify_stage7d_export_evidence(evidence_bytes: object) -> SyntheticCurriculumExportReceipt:
    receipt = verify_synthetic_curriculum_export_evidence(evidence_bytes)
    if not isinstance(evidence_bytes, bytes):
        raise TypeError("evidence_bytes must be bytes")
    payload = json.loads(evidence_bytes.decode("ascii"))
    if payload.get("source_repository") != _EXPECTED_REPOSITORY:
        raise SyntheticCurriculumAcceptanceError("source repository mismatch")
    if payload.get("transport_archive") != _EXPECTED_ARCHIVE:
        raise SyntheticCurriculumAcceptanceError("transport archive mismatch")
    plan = payload.get("plan")
    build = payload.get("build")
    if not isinstance(plan, dict) or not isinstance(build, dict):
        raise SyntheticCurriculumAcceptanceError("missing plan/build evidence")
    if plan.get("schema_version") != "st-synthetic-curriculum-plan-v1":
        raise SyntheticCurriculumAcceptanceError("plan schema mismatch")
    if plan.get("profile_version") != _EXPECTED_PROFILE:
        raise SyntheticCurriculumAcceptanceError("plan profile mismatch")
    if (plan.get("measure_count"), plan.get("raster_width")) != (8, 1000):
        raise SyntheticCurriculumAcceptanceError("plan dimensions mismatch")
    if plan.get("degradation_profiles") != ["clean", "light", "medium"]:
        raise SyntheticCurriculumAcceptanceError("degradation profile mismatch")
    if plan.get("family_profile_counts") != _EXPECTED_PROFILE_COUNTS:
        raise SyntheticCurriculumAcceptanceError("profile balance mismatch")
    if build.get("profile_version") != _EXPECTED_PROFILE:
        raise SyntheticCurriculumAcceptanceError("build profile mismatch")
    return receipt
