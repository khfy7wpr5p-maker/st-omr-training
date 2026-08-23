from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ArchitectureDocumentCompatibilityTests(unittest.TestCase):
    def _read(self, name: str) -> str:
        return (ROOT / name).read_text(encoding="utf-8")

    def test_current_overlay_distinguishes_main_from_shadow_evidence(self) -> None:
        text = self._read("ARCHITECTURE_CURRENT.md")
        self.assertIn("Merged `main` authority", text)
        self.assertIn("Shadow / experimental overlay", text)
        self.assertIn("Rest R4", text)
        self.assertIn("Resolver connection remains CLOSED", text)
        self.assertIn("Meter V5-2B", text)
        self.assertIn("FINAL_HOLDOUT locked", text)

    def test_rest_shadow_state_preserves_frozen_v1_scope_and_fail_closed_authority(self) -> None:
        text = self._read("ARCHITECTURE_CURRENT.md")
        for rest_class in ("half", "quarter", "eighth"):
            self.assertIn(rest_class, text)
        self.assertIn("AMBIGUOUS", text)
        self.assertIn("REJECTED", text)
        self.assertIn("shadow PASS is not production PASS", text)

    def test_status_no_longer_claims_d10_as_current_active_lane(self) -> None:
        text = self._read("STATUS.md")
        self.assertIn("a6a40b218a95c72349984ee2aee7262f467021fc", text)
        self.assertIn("Rest R4", text)
        self.assertIn("Meter V5-2B", text)
        self.assertNotIn("The current active lane is **Stage 7-D10", text)

    def test_historical_stage_architecture_documents_point_to_current_overlay(self) -> None:
        for name in (
            "ARCHITECTURE_STAGE8_3A.md",
            "ARCHITECTURE_STAGE8_3A_ADAPTER.md",
            "STAGE7D4_SPECIALIST_ARCHITECTURE.md",
            "STAGE7D13_R2_REFINEMENT_PLAN.md",
        ):
            text = self._read(name)
            self.assertIn("ARCHITECTURE_CURRENT.md", text, name)
            self.assertTrue(
                "historical" in text.lower() or "frozen" in text.lower(),
                name,
            )

    def test_long_form_architecture_declares_status_authority_order(self) -> None:
        text = self._read("ARCHITECTURE.md")
        self.assertIn("Current-status authority", text)
        self.assertIn("ARCHITECTURE_CURRENT.md", text)
        self.assertIn("open draft PR", text)
        self.assertIn("sealed TEST", text)

    def test_audit_records_pass_with_documentation_drift_correction(self) -> None:
        text = self._read("ARCHITECTURE_COMPATIBILITY_AUDIT_2026-08-23.md")
        self.assertIn("Architecture compatibility: PASS", text)
        self.assertIn("Documentation consistency before this synchronization: FAIL", text)
        self.assertIn("AI does not create ground truth", text)
        self.assertIn("shadow PASS is not production PASS", text)


if __name__ == "__main__":
    unittest.main()
