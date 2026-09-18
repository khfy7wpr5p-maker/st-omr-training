from __future__ import annotations

import json
from pathlib import Path
import unittest

from st_omr_training.poly_v2_ossq_stage8_probe import (
    B8S_EXPECTED_B8P_RECEIPT_SHA256,
    B8S_EXPECTED_B8Q_RECEIPT_SHA256,
    OssqStage8ProbeError,
    build_stage8_probe_candidates,
    semantic_gate_diagnostic_from_error,
)
from st_omr_training.musicxml_roundtrip import SupportedV1RoundTripError
from st_omr_training.real_data_intake import RealDataIntakeError
from st_omr_training.validator import ValidationIssue, ValidationResult


REPO_ROOT = Path(__file__).resolve().parents[1]
B8P = REPO_ROOT / "evidence" / "ossq_b8p_system_pair_materialization.json"
B8R = REPO_ROOT / "evidence" / "ossq_b8r_cross_render_audit.json"
B8Q = REPO_ROOT / "evidence" / "ossq_b8q_pair_review_admission.json"


def payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


class OssqStage8VerifiedProbeTests(unittest.TestCase):
    def test_semantic_gate_diagnostic_recovers_wrapped_validation_issue_code_and_path(self) -> None:
        validation = ValidationResult(
            (
                ValidationIssue(
                    code="musicxml.part_count",
                    path="$.parts",
                    message="V1 requires exactly one score part",
                ),
            )
        )
        try:
            try:
                raise SupportedV1RoundTripError("Stage 2-C validation failed", validation)
            except SupportedV1RoundTripError as cause:
                raise RealDataIntakeError("generic Stage 8 semantic rejection") from cause
        except RealDataIntakeError as error:
            diagnostic = semantic_gate_diagnostic_from_error(error)

        self.assertEqual(diagnostic.issue_code, "musicxml.part_count")
        self.assertEqual(diagnostic.issue_path, "$.parts")
        self.assertEqual(diagnostic.source, "validation")

    def test_semantic_gate_diagnostic_refines_generic_xsd_failure_from_musicxml_bytes(self) -> None:
        validation = ValidationResult(
            (
                ValidationIssue(
                    code="musicxml.xsd_invalid",
                    path="$",
                    message="document is not valid MusicXML 4.0 XSD",
                ),
            )
        )
        try:
            try:
                raise SupportedV1RoundTripError("Stage 2-C validation failed", validation)
            except SupportedV1RoundTripError as cause:
                raise RealDataIntakeError("generic Stage 8 semantic rejection") from cause
        except RealDataIntakeError as error:
            diagnostic = semantic_gate_diagnostic_from_error(
                error,
                musicxml_bytes=b'<?xml version="1.0"?><other/>',
            )

        self.assertNotEqual(diagnostic.issue_code, "musicxml.xsd_invalid")
        self.assertTrue(diagnostic.issue_code.startswith("musicxml.xsd."))
        self.assertEqual(diagnostic.issue_path, "/other")
        self.assertEqual(diagnostic.source, "xsd-error-log")

    def test_exact_b8q_verified_population_yields_14_probe_candidates(self) -> None:
        candidates = build_stage8_probe_candidates(
            b8p_payload=payload(B8P),
            b8r_payload=payload(B8R),
            b8q_payload=payload(B8Q),
        )
        self.assertEqual(len(candidates), 14)
        self.assertEqual(
            {item.score_id for item in candidates},
            {"7075297", "7078259", "7093885", "7103818", "7108150", "8071278"},
        )
        self.assertNotIn("7397765", {item.score_id for item in candidates})
        self.assertEqual(len({item.family_id for item in candidates}), 4)

    def test_every_candidate_has_approved_research_training_rights_only(self) -> None:
        candidates = build_stage8_probe_candidates(
            b8p_payload=payload(B8P),
            b8r_payload=payload(B8R),
            b8q_payload=payload(B8Q),
        )
        self.assertTrue(all(item.training_allowed for item in candidates))
        self.assertTrue(all(not item.commercial_use_allowed for item in candidates))
        self.assertTrue(all(not item.redistribution_allowed for item in candidates))
        self.assertTrue(all(item.rights_basis == "public-domain" for item in candidates))
        self.assertTrue(all(item.rights_review == "approved" for item in candidates))
        self.assertTrue(all(item.pairing_method == "independent-cross-render-v1" for item in candidates))

    def test_review_required_b8q_records_never_enter_probe_candidates(self) -> None:
        b8q = payload(B8Q)
        review_required = {
            item["segment_id"]
            for item in b8q["records"]
            if item["decision"] == "review-required"
        }
        candidates = build_stage8_probe_candidates(
            b8p_payload=payload(B8P),
            b8r_payload=payload(B8R),
            b8q_payload=b8q,
        )
        self.assertTrue(review_required.isdisjoint({item.segment_id for item in candidates}))

    def test_b8q_receipt_identity_is_frozen(self) -> None:
        b8q = payload(B8Q)
        b8q["receipt_sha256"] = "0" * 64
        with self.assertRaisesRegex(OssqStage8ProbeError, "B8Q receipt identity"):
            build_stage8_probe_candidates(
                b8p_payload=payload(B8P),
                b8r_payload=payload(B8R),
                b8q_payload=b8q,
            )

    def test_b8p_receipt_identity_is_frozen(self) -> None:
        b8p = payload(B8P)
        b8p["receipt_sha256"] = "0" * 64
        with self.assertRaisesRegex(OssqStage8ProbeError, "B8P receipt identity"):
            build_stage8_probe_candidates(
                b8p_payload=b8p,
                b8r_payload=payload(B8R),
                b8q_payload=payload(B8Q),
            )

    def test_pair_identity_drift_fails_closed(self) -> None:
        b8p = payload(B8P)
        verified_segment = next(
            item["segment_id"]
            for item in payload(B8Q)["records"]
            if item["decision"] == "verified"
        )
        pair = next(item for item in b8p["pairs"] if item["segment_id"] == verified_segment)
        pair["musicxml_sha256"] = "f" * 64
        with self.assertRaisesRegex(OssqStage8ProbeError, "pair identity"):
            build_stage8_probe_candidates(
                b8p_payload=b8p,
                b8r_payload=payload(B8R),
                b8q_payload=payload(B8Q),
            )

    def test_source_family_is_bound_to_exact_source_document_hash(self) -> None:
        candidates = build_stage8_probe_candidates(
            b8p_payload=payload(B8P),
            b8r_payload=payload(B8R),
            b8q_payload=payload(B8Q),
        )
        by_source: dict[str, set[str]] = {}
        for item in candidates:
            by_source.setdefault(item.source_document_sha256, set()).add(item.family_id)
        self.assertTrue(all(len(families) == 1 for families in by_source.values()))
        self.assertEqual(len(by_source), 4)

    def test_expected_receipt_constants_are_frozen(self) -> None:
        self.assertEqual(
            B8S_EXPECTED_B8P_RECEIPT_SHA256,
            "3f9f2df43b287dd95e03eaf1ffc4c5a3c9e25bdced8f09675918bc9b1f99e0cf",
        )
        self.assertEqual(
            B8S_EXPECTED_B8Q_RECEIPT_SHA256,
            "c9345c106217c52ecdb4cbc325e5ef4e1b5d23dfbfc44190ee6124dd655b86fa",
        )


if __name__ == "__main__":
    unittest.main()
