from __future__ import annotations

import inspect
import unittest

from st_omr_training import meter_v5_2t_bounded_class_balanced_head_repair_v1 as t
from st_omr_training import meter_v5_2u_v5_2t_historical_retention_v1 as u


class TestMeterV52UV52THistoricalRetentionV1(unittest.TestCase):
    def _valid_training_payload(self):
        per = {}
        for digit in ("2", "3"):
            per[digit] = {
                "candidate": {
                    "candidate_sha256": u.V52T_CANDIDATE_SHA256[digit],
                    "reload_verified": True,
                },
                "state_invariants": {
                    "changed_state_keys": ["head.weight"],
                    "only_head_weight_changed": True,
                    "backbone_bit_identical": True,
                    "head_bias_bit_identical": True,
                },
                "fit": {
                    "finite_non_increasing_objective": True,
                    "geometry_float32_copy_back": {"gate": "PASS"},
                    "lbfgs_termination": {"final_gradient_finite": True},
                },
            }
        return {
            "schema": t.SCHEMA,
            "numerical_integrity_gate": {"gate": "PASS", "reasons": []},
            "historical_preservation_claimed": False,
            "historical_validation_opened": False,
            "first30_opened": False,
            "v5_validation_opened": False,
            "final_holdout_locked": True,
            "digit4_frozen": True,
            "per_specialist": per,
        }

    def test_exact_completed_execution_is_frozen(self):
        self.assertEqual(
            u.V52T_IMPLEMENTATION_HEAD,
            "8d98c1f6ad66ee896d28c02fb7ff1afafab23be9",
        )
        self.assertEqual(
            u.V52T_TRAINING_REPORT_SHA256,
            "18851e86d9e2aa7d0d55ccbafdb2983c96c9276913419b928512ea76e3a2bc57",
        )
        self.assertEqual(
            u.V52T_EXECUTION_ENVELOPE_SHA256,
            "1a043631118612a85e6d2a78baaa7f26aeb0742254684b96d8b3a4d25c031382",
        )
        self.assertEqual(
            u.V52T_CANDIDATE_SHA256,
            {
                "2": "13fb7dd0af1faa8a762433df2b27d6c82553fca398b84b060cb8d573d2d228de",
                "3": "8ec37448af27f57bdd7840eeeee8a61cee5d21e6c21156ccfb4c1510d3542514",
            },
        )

    def test_exact_numerical_payload_is_accepted(self):
        u._validate_training_payload_v1(self._valid_training_payload())

    def test_payload_fails_closed_on_candidate_or_integrity_drift(self):
        payload = self._valid_training_payload()
        payload["per_specialist"]["3"]["candidate"]["candidate_sha256"] = "0" * 64
        with self.assertRaisesRegex(u.MeterV5_2UError, "candidate SHA"):
            u._validate_training_payload_v1(payload)

        payload = self._valid_training_payload()
        payload["per_specialist"]["2"]["state_invariants"][
            "backbone_bit_identical"
        ] = False
        with self.assertRaisesRegex(u.MeterV5_2UError, "invariant"):
            u._validate_training_payload_v1(payload)

    def test_stage_is_read_only_and_later_surfaces_are_closed(self):
        boundary = u.safety_boundary()
        self.assertFalse(boundary["training"])
        self.assertFalse(boundary["autograd_grad_used"])
        self.assertFalse(boundary["backward"])
        self.assertEqual(boundary["optimizer_steps"], 0)
        self.assertFalse(boundary["checkpoint_write"])
        self.assertTrue(boundary["historical_validation_opened"])
        self.assertTrue(boundary["historical_retention_executed"])
        self.assertFalse(boundary["first30_opened"])
        self.assertFalse(boundary["v5_validation_opened"])
        self.assertTrue(boundary["final_holdout_locked"])
        self.assertTrue(boundary["digit4_frozen"])
        self.assertFalse(u.validation_opened_by_this_module())
        self.assertFalse(u.production_promotion_allowed())

        source = inspect.getsource(u)
        for forbidden in (
            ".backward(",
            "torch.autograd",
            "torch.optim",
            "optimizer.step",
            "train_bounded_class_balanced_head_repair_v1(",
            "run_first30_diagnostic_v1(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_first30_is_authorized_only_by_exact_pass_report(self):
        self.assertFalse(u.first30_authorized({"schema": u.SCHEMA, "gate": "HOLD"}))
        self.assertFalse(u.first30_authorized({"schema": "wrong", "gate": "PASS"}))
        self.assertTrue(u.first30_authorized({"schema": u.SCHEMA, "gate": "PASS"}))


if __name__ == "__main__":
    unittest.main()
