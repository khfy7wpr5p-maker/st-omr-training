"""Traceability binding from accepted D3 evidence to the D4 architecture."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Final

from .stage7d4_specialist_architecture import stage7d4_architecture_fingerprint


D3_MAIN_MERGE_SHA: Final[str] = "168c03755f0e06e8042fc0a391a357c71c6288fe"
D3_PR_HEAD_SHA: Final[str] = "c25caddeaa897df5eeaad545e68f51aafc19c1f6"
D3_RUN_ID: Final[str] = "22b7d63f5112fb9d41fa72d502c7a3648781d692949bedf5fbbad8142e910ab7"
D3_DIAGNOSTICS_SHA256: Final[str] = "b5843f896a2f75f8c0b111a8d1dd562a74b15cf67d48c0d4e1dfa8655ed41a6b"
D3_VERIFICATION_SHA256: Final[str] = "558fb0a6e0bfe7e7f461361773a9f8a08b48c5dc4613bd1a3d3a73da7e5186e9"
D3_POST_MERGE_CI_RUN: Final[int] = 146
D3_POST_MERGE_TESTS: Final[int] = 483
D3_DECISION: Final[str] = "specialist_musical_task_decomposition"
D4_DECISION_BINDING_SCHEMA: Final[str] = "stage7d4-d3-decision-binding-v1"


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def stage7d4_decision_binding_payload() -> dict[str, object]:
    return {
        "schema": D4_DECISION_BINDING_SCHEMA,
        "d3": {
            "main_merge_sha": D3_MAIN_MERGE_SHA,
            "pr_head_sha": D3_PR_HEAD_SHA,
            "run_id": D3_RUN_ID,
            "diagnostics_sha256": D3_DIAGNOSTICS_SHA256,
            "verification_sha256": D3_VERIFICATION_SHA256,
            "post_merge_ci_run": D3_POST_MERGE_CI_RUN,
            "post_merge_tests": D3_POST_MERGE_TESTS,
            "accepted_decision": D3_DECISION,
        },
        "d4_architecture_fingerprint": stage7d4_architecture_fingerprint(),
    }


def stage7d4_decision_binding_fingerprint() -> str:
    return sha256(_canonical_json_bytes(stage7d4_decision_binding_payload())).hexdigest()
