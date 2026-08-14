from __future__ import annotations

import re
import unittest

from st_omr_training.stage7d4_decision_provenance import (
    D3_DECISION,
    D3_DIAGNOSTICS_SHA256,
    D3_MAIN_MERGE_SHA,
    D3_POST_MERGE_CI_RUN,
    D3_POST_MERGE_TESTS,
    D3_PR_HEAD_SHA,
    D3_RUN_ID,
    D3_VERIFICATION_SHA256,
    stage7d4_decision_binding_fingerprint,
    stage7d4_decision_binding_payload,
)
from st_omr_training.stage7d4_specialist_architecture import stage7d4_architecture_fingerprint


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class Stage7D4DecisionProvenanceTests(unittest.TestCase):
    def test_exact_accepted_d3_evidence_is_bound(self) -> None:
        self.assertEqual(D3_MAIN_MERGE_SHA, "168c03755f0e06e8042fc0a391a357c71c6288fe")
        self.assertEqual(D3_PR_HEAD_SHA, "c25caddeaa897df5eeaad545e68f51aafc19c1f6")
        self.assertEqual(D3_RUN_ID, "22b7d63f5112fb9d41fa72d502c7a3648781d692949bedf5fbbad8142e910ab7")
        self.assertEqual(D3_DIAGNOSTICS_SHA256, "b5843f896a2f75f8c0b111a8d1dd562a74b15cf67d48c0d4e1dfa8655ed41a6b")
        self.assertEqual(D3_VERIFICATION_SHA256, "558fb0a6e0bfe7e7f461361773a9f8a08b48c5dc4613bd1a3d3a73da7e5186e9")
        self.assertEqual(D3_POST_MERGE_CI_RUN, 146)
        self.assertEqual(D3_POST_MERGE_TESTS, 483)
        self.assertEqual(D3_DECISION, "specialist_musical_task_decomposition")

    def test_binding_contains_current_architecture_fingerprint(self) -> None:
        payload = stage7d4_decision_binding_payload()
        self.assertEqual(payload["d4_architecture_fingerprint"], stage7d4_architecture_fingerprint())
        self.assertEqual(payload["d3"]["accepted_decision"], D3_DECISION)

    def test_binding_fingerprint_is_deterministic_sha256(self) -> None:
        first = stage7d4_decision_binding_fingerprint()
        second = stage7d4_decision_binding_fingerprint()
        self.assertEqual(first, second)
        self.assertIsNotNone(_HEX64_RE.fullmatch(first))


if __name__ == "__main__":
    unittest.main()
