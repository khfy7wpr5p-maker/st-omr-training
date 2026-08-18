from __future__ import annotations

import inspect
from pathlib import Path
import re
import unittest

from st_omr_training import accidental_r2_key_local_acceptance_v1 as acc


class AccidentalR2KeyLocalAcceptanceV1Tests(unittest.TestCase):
    def test_frozen_case_inventory_matches_markdown_contract(self) -> None:
        cases = acc.acceptance_cases()
        expected_ids = {
            *(f"K{i:02d}" for i in range(1, 7)),
            *(f"L{i:02d}" for i in range(1, 11)),
            *(f"A{i:02d}" for i in range(1, 7)),
        }
        self.assertEqual(set(cases), expected_ids)
        self.assertEqual(len(cases), 22)

        contract = Path("ACCIDENTAL_R2_KEY_LOCAL_ACCEPTANCE_V1.md").read_text(encoding="utf-8")
        documented_ids = set(re.findall(r"^\| ([KLA]\d{2}) \|", contract, flags=re.MULTILINE))
        self.assertEqual(documented_ids, expected_ids)

    def test_all_cases_match_expected_deterministic_outputs(self) -> None:
        failures: list[str] = []
        for case_id, (groups, expected) in acc.acceptance_cases().items():
            actual = acc.evaluate(groups)
            if actual != expected:
                failures.append(
                    f"{case_id}: actual={actual!r} expected={expected!r}"
                )
        self.assertFalse(
            failures,
            "Accidental R2 acceptance mismatches:\n" + "\n".join(failures),
        )

    def test_key_signature_and_double_accidental_are_not_conflated(self) -> None:
        two_sharps = acc.evaluate((acc.EvidenceGroup(("sharp", "sharp"), "staff_start"),))
        double_sharp = acc.evaluate((
            acc.EvidenceGroup(("double_sharp",), "measure", local_targets=("note_1",)),
        ))
        two_flats = acc.evaluate((acc.EvidenceGroup(("flat", "flat"), "staff_start"),))
        double_flat = acc.evaluate((
            acc.EvidenceGroup(("double_flat",), "measure", local_targets=("note_1",)),
        ))

        self.assertEqual(two_sharps.key_signature.fifths, 2)
        self.assertEqual(two_sharps.local_alterations, ())
        self.assertIsNone(double_sharp.key_signature)
        self.assertEqual(double_sharp.local_alterations[0].alter, 2)

        self.assertEqual(two_flats.key_signature.fifths, -2)
        self.assertEqual(two_flats.local_alterations, ())
        self.assertIsNone(double_flat.key_signature)
        self.assertEqual(double_flat.local_alterations[0].alter, -2)

    def test_key_signature_never_forces_mode(self) -> None:
        for case_id in ("K01", "K02", "K03", "K04", "K05", "K06"):
            groups, _ = acc.acceptance_cases()[case_id]
            outcome = acc.evaluate(groups)
            self.assertIsNotNone(outcome.key_signature, case_id)
            self.assertEqual(outcome.key_signature.mode, "UNKNOWN", case_id)
            self.assertEqual(len(outcome.key_signature.possible_keys), 2, case_id)

    def test_fail_closed_cases_never_mutate_music(self) -> None:
        for case_id in ("A01", "A02", "A03", "A04", "A05", "A06"):
            groups, expected = acc.acceptance_cases()[case_id]
            actual = acc.evaluate(groups)
            self.assertEqual(actual, expected, case_id)
            self.assertIsNone(actual.key_signature, case_id)
            self.assertEqual(actual.local_alterations, (), case_id)

    def test_shadow_package_cannot_authorize_resolver_or_training(self) -> None:
        self.assertFalse(acc.resolver_connection_allowed())
        source = inspect.getsource(acc)
        lowered = source.lower()
        self.assertNotIn("torch", lowered)
        self.assertNotIn("optimizer", lowered)
        self.assertNotIn("runtime_deterministic_resolver", lowered)
        self.assertNotIn("stage7d10", lowered)
        self.assertNotIn("test_open", lowered)


if __name__ == "__main__":
    unittest.main()
