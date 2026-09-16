"""TR-POLY-09B8K external source selection for the first real Native V2 corpus.

This module records source/licensing facts only. It does not download, open,
install, admit, materialize, train on, or evaluate any external dataset bytes.

The selected public source family is OSSQ-OMR (ISMIR 2026). Its tracked
annotation sources and publisher-created synthetic/derived artifacts are
recorded separately from the scanned-image track because the latter is sourced
from IMSLP PDFs and therefore requires per-score upstream-rights review before
ST-OMR may admit those bytes.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Final

from .external_dataset_registry import (
    DataUseClass,
    ExternalDatasetRecord,
    RegistryState,
    validate_registry,
)


B8K_SOURCE_SELECTION_VERSION: Final[str] = "st-omr-poly-v2-corpus-source-selection-v1"
OSSQ_OMR_CAMERA_READY_SHA: Final[str] = "7a17e45cddc0b7064fc3a179b62caeb57595e993"
OSSQ_OMR_SOURCE_URL: Final[str] = (
    "https://github.com/MALerLab/ossq-omr/tree/"
    + OSSQ_OMR_CAMERA_READY_SHA
)
OSSQ_OMR_LICENSE_URL: Final[str] = (
    "https://github.com/MALerLab/ossq-omr/blob/"
    + OSSQ_OMR_CAMERA_READY_SHA
    + "/LICENSE.txt"
)
OSSQ_OMR_README_URL: Final[str] = (
    "https://github.com/MALerLab/ossq-omr/blob/"
    + OSSQ_OMR_CAMERA_READY_SHA
    + "/README.md"
)


OSSQ_OMR_SYMBOLIC_SYNTHETIC: Final[ExternalDatasetRecord] = ExternalDatasetRecord(
    dataset_name="OSSQ-OMR",
    dataset_component="annotation sources + publisher-created synthetic/derived artifacts",
    source=OSSQ_OMR_SOURCE_URL,
    version=f"camera-ready {OSSQ_OMR_CAMERA_READY_SHA}",
    license_id="CC0-1.0",
    license_evidence=OSSQ_OMR_LICENSE_URL,
    redistribution_allowed=True,
    commercial_use_allowed=True,
    training_allowed=True,
    evaluation_allowed=True,
    derivative_restrictions=(
        "CC0-1.0 for rights held by the affirmer; third-party rights are not "
        "automatically cleared by the CC0 disclaimer"
    ),
    data_use_class=DataUseClass.COMMERCIAL_CLEAN,
    registry_state=RegistryState.LICENSE_VERIFIED,
    notes=(
        "The camera-ready OSSQ-OMR README states that annotation sources and "
        "bulk-distributed derived formats are released under CC0. This record "
        "does not include IMSLP-derived scanned-image bytes and is not install-pinned."
    ),
)


OSSQ_OMR_SCANNED_TRACK: Final[ExternalDatasetRecord] = ExternalDatasetRecord(
    dataset_name="OSSQ-OMR",
    dataset_component="IMSLP-derived scanned-image track",
    source=OSSQ_OMR_SOURCE_URL,
    version=f"camera-ready {OSSQ_OMR_CAMERA_READY_SHA}",
    license_id="CC0-1.0 publisher claim + upstream-rights review required",
    license_evidence=OSSQ_OMR_LICENSE_URL,
    redistribution_allowed=None,
    commercial_use_allowed=None,
    training_allowed=None,
    evaluation_allowed=None,
    derivative_restrictions=(
        "Scanned PDFs/images originate from third-party IMSLP sources. Per-score "
        "source/provenance and rights evidence must be independently reviewed "
        "before any scanned bytes can be admitted."
    ),
    data_use_class=DataUseClass.LICENSE_REVIEW_REQUIRED,
    registry_state=RegistryState.CANDIDATE,
    notes=(
        "Primary candidate for the first real scanned OMR corpus because the "
        "camera-ready release provides score-level provenance/alignment metadata, "
        "paired symbolic targets, and score-level split design. No scanned bytes "
        "are admitted by this record."
    ),
)


FIRST_REAL_CORPUS_SOURCE_CANDIDATES: Final[tuple[ExternalDatasetRecord, ...]] = (
    validate_registry((OSSQ_OMR_SYMBOLIC_SYNTHETIC, OSSQ_OMR_SCANNED_TRACK))
)


def first_real_corpus_source_selection_fingerprint() -> str:
    """Return a deterministic hash of the B8K source-policy decision surface."""

    payload = {
        "version": B8K_SOURCE_SELECTION_VERSION,
        "ossq_camera_ready_sha": OSSQ_OMR_CAMERA_READY_SHA,
        "records": [
            {
                "dataset_name": item.dataset_name,
                "dataset_component": item.dataset_component,
                "record_sha256": item.canonical_sha256(),
            }
            for item in FIRST_REAL_CORPUS_SOURCE_CANDIDATES
        ],
        "scanned_track_policy": "per-score-rights-review-before-install-pin",
        "test_policy": "no-external-test-bytes-opened-by-b8k",
        "production_authority": False,
        "commercial_use_authority": False,
    }
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
