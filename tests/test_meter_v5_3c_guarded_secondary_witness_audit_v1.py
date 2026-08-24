from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from st_omr_training import meter_v5_3c_guarded_secondary_witness_audit_v1 as v


def _row(first: float, second: float = 0.0) -> np.ndarray:
    value = np.zeros(64, dtype=np.float64)
    value[0] = first
    value[1] = second
    return value


class TestMeterV53CGuardedSecondaryWitnessAuditV1(unittest.TestCase):
    def test_exact_v5_3a_hold_evidence_is_bound(self):
        self.assertEqual(
            v.V53A_REPORT_SHA256,
            "a483173353b9e425a4a3eb8d177376c15a7c5fa1d13c62689356c04b3fffd92e",
        )
        self.assertEqual(
            v.V53A_RECOVERY_ENVELOPE_SHA256,
            "6514983e886c9ba41398f2a0c1888d3088455ab612cd6ad91614bcd8d7db4d40",
        )
        self.assertEqual(v.V53A_SOURCE_IMPLEMENTATION_HEAD, "cdc6683a556c16b00e7b154fca8e89ba5dd848b7")
        self.assertEqual(v.V53A_SOURCE_HARNESS_HEAD, "c2d5f1652adac52387e33b9d2f33078f864f980b")

    def test_contract_changes_only_secondary_cap_numerics(self):
        contract = v.solver_contract()
        self.assertEqual(contract["method"], "highs-ds")
        self.assertFalse(contract["presolve"])
        self.assertEqual(contract["witness_tolerance"], 1e-7)
        self.assertEqual(contract["primary_l1_absolute_slack"], 1e-6)
        self.assertEqual(contract["internal_cap_guard"], 5e-7)
        self.assertEqual(contract["internal_cap_guard_in_witness_tolerances"], 5.0)
        self.assertTrue(contract["secondary_l1_cap_row_normalized_to_rhs_one"])
        self.assertTrue(contract["external_acceptance_cap_unchanged"])
        self.assertTrue(contract["primary_optimum_reused_from_exact_v5_3a"])
        self.assertFalse(contract["primary_lp_rerun"])
        self.assertFalse(contract["solver_sweep"])
        self.assertFalse(contract["fallback_solver"])
        self.assertFalse(contract["tolerance_changed"])
        self.assertFalse(contract["objective_changed"])
        self.assertFalse(contract["margin_changed"])
        self.assertFalse(contract["threshold_or_bias_changed"])

    def test_stage_is_diagnostic_only_and_keeps_surfaces_closed(self):
        boundary = v.safety_boundary()
        self.assertTrue(boundary["secondary_linear_program_witness_fit"])
        self.assertFalse(boundary["candidate_checkpoint_write_authorized"])
        self.assertFalse(boundary["model_parameter_mutation_executed"])
        self.assertFalse(boundary["autograd_grad_used"])
        self.assertFalse(boundary["backward"])
        self.assertEqual(boundary["optimizer_steps"], 0)
        self.assertFalse(boundary["historical_retention_executed"])
        self.assertFalse(boundary["first30_opened"])
        self.assertFalse(boundary["v5_validation_opened"])
        self.assertTrue(boundary["final_holdout_locked"])
        self.assertTrue(boundary["digit4_frozen"])

        source = inspect.getsource(v)
        for forbidden in (
            ".backward(",
            "torch.optim",
            "optimizer.step",
            "run_historical_retention_v1(",
            "run_first30_diagnostic_v1(",
            "torch.save(",
        ):
            self.assertNotIn(forbidden, source)

    def test_known_guarded_secondary_witness_is_independently_verified(self):
        historical_features = np.stack((_row(1.0), _row(-1.0)))
        historical_targets = np.asarray([1.0, 0.0], dtype=np.float64)
        v5_features = np.stack((_row(1.0), _row(-1.0)))
        v5_targets = np.asarray([1.0, 0.0], dtype=np.float64)
        primary_l1 = v.ROBUST_DECISION_MARGIN + v.SOLVER_MARGIN_BUFFER
        external_cap = primary_l1 + v.PRIMARY_L1_ABSOLUTE_SLACK

        result, witness = v.solve_guarded_secondary_witness_v1(
            historical_features=historical_features,
            historical_targets=historical_targets,
            v5_features=v5_features,
            v5_targets=v5_targets,
            frozen_weight=np.zeros(64, dtype=np.float64),
            frozen_bias=0.0,
            threshold=0.5,
            exact_primary_l1_optimum=primary_l1,
            exact_external_l1_cap=external_cap,
        )

        self.assertEqual(result["witness_claim"], "GUARDED_SECONDARY_WITNESS_VERIFIED")
        self.assertIsNotNone(witness)
        self.assertEqual(result["external_l1_cap_violations"], 0)
        self.assertEqual(result["internal_guarded_l1_cap_violations"], 0)
        self.assertEqual(result["v5_constraint_violations"], 0)
        self.assertEqual(result["historical_margin_constraint_violations"], 0)
        self.assertTrue(result["functional_delta_identity_verified"])
        self.assertEqual(result["diagnostic_v5_train_metrics"]["f1"], 1.0)
        self.assertFalse(result["witness_weight_values_emitted"])

    def test_invalid_exact_cap_fails_closed_before_solver(self):
        surface = np.stack((_row(1.0), _row(-1.0)))
        targets = np.asarray([1.0, 0.0], dtype=np.float64)
        with self.assertRaisesRegex(v.MeterV5_3CError, "exact external L1 cap"):
            v.solve_guarded_secondary_witness_v1(
                historical_features=surface,
                historical_targets=targets,
                v5_features=surface,
                v5_targets=targets,
                frozen_weight=np.zeros(64, dtype=np.float64),
                frozen_bias=0.0,
                threshold=0.5,
                exact_primary_l1_optimum=1.0,
                exact_external_l1_cap=1.0,
            )

    def test_execution_requires_token_and_refuses_existing_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "annotations"
            annotations.mkdir()
            with self.assertRaisesRegex(v.MeterV5_3CError, "approval token"):
                v.run_guarded_secondary_witness_audit_v1(
                    root,
                    m4a_root=root,
                    d10_root=root,
                    digit2_frozen=root / "d2.pt",
                    digit3_frozen=root / "d3.pt",
                    v5_3a_report=root / "a.json",
                    v5_3a_recovery_envelope=root / "e.json",
                    confirmation="WRONG",
                )
            (annotations / v.REPORT_NAME).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(v.MeterV5_3CError, "overwrite/rerun"):
                v.run_guarded_secondary_witness_audit_v1(
                    root,
                    m4a_root=root,
                    d10_root=root,
                    digit2_frozen=root / "d2.pt",
                    digit3_frozen=root / "d3.pt",
                    v5_3a_report=root / "a.json",
                    v5_3a_recovery_envelope=root / "e.json",
                    confirmation=v.APPROVAL_TOKEN,
                )


if __name__ == "__main__":
    unittest.main()
