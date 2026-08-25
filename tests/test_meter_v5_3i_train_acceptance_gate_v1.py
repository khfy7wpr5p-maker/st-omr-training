import hashlib
from pathlib import Path
import tempfile
import unittest

from st_omr_training import meter_v5_2b_specialist_adaptation as v52b
from st_omr_training import meter_v5_3e_rescue_training_preregistration_v1 as v53e
from st_omr_training import meter_v5_3g_authoritative_rescue_training_v1 as v53g
from st_omr_training import meter_v5_3i_train_acceptance_gate_v1 as gate


class TestMeterV53ITrainAcceptanceGate(unittest.TestCase):
    def _receipt_fixture(self):
        groups = {
            "2": {name: f"2-{name}-fingerprint" for name in v53e.TRAIN_GROUPS},
            "3": {name: f"3-{name}-fingerprint" for name in v53e.TRAIN_GROUPS},
        }
        per = {}
        for digit in ("2", "3"):
            per[digit] = {
                "frozen_state_before": f"frozen-{digit}",
                "frozen_state_after": f"frozen-{digit}",
                "frozen_state_bit_identical": True,
                "materialization": {
                    "group_counts": dict(v53e.EXPECTED_TRAIN_GROUP_COUNTS[digit]),
                    "group_fingerprints": dict(groups[digit]),
                },
                "execution": {
                    "optimizer_steps": v53e.FIXED_OPTIMIZER_STEPS,
                    "authoritative_dataset_execution": True,
                    "checkpoint_write": False,
                    "protected_evaluation_opened": False,
                    "initial_state_fingerprint": "initial",
                    "final_state_fingerprint": f"rescue-{digit}",
                },
                "artifact": {
                    "artifact_sha256": gate.EXPECTED_RESCUE_ARTIFACT_SHA256[digit],
                    "reload_verified": True,
                    "state_fingerprint": f"rescue-{digit}",
                },
            }
        report = {
            "schema": v53g.SCHEMA,
            "single_authoritative_execution_completed": True,
            "candidate_configuration_count": 1,
            "train_performance_gate_executed": False,
            "historical_validation_retention_executed": False,
            "first30_opened": False,
            "v5_validation_opened": False,
            "final_holdout_locked": True,
            "runtime_authority_changed": False,
            "production_promotion": False,
            "v5_3f_head_sha": v53g.V53F_HEAD_SHA,
            "numerical_integrity_gate": {"gate": "PASS", "reasons": []},
            "frozen_state_isolation_gate": {"gate": "PASS", "reasons": []},
            "source_checkpoint_sha256": {
                "2": v52b.DIGIT2_SHA256,
                "3": v52b.DIGIT3_SHA256,
            },
            "per_specialist": per,
        }
        envelope = {
            "schema": gate.V53H_ENVELOPE_SCHEMA,
            "repository": "khfy7wpr5p-maker/st-omr-training",
            "expected_head": gate.V53G_HEAD_SHA,
            "actual_head": gate.V53G_HEAD_SHA,
            "ci_run_id": 32769348282,
            "single_authoritative_execution_completed": True,
            "candidate_configuration_count": 1,
            "numerical_integrity_gate": "PASS",
            "frozen_state_isolation_gate": "PASS",
            "train_performance_gate_executed": False,
            "historical_validation_opened": False,
            "first30_opened": False,
            "v5_reserve_opened": False,
            "v5_validation_opened": False,
            "final_holdout_locked": True,
            "digit4_frozen": True,
            "threshold_tuning": False,
            "hyperparameter_sweep": False,
            "automatic_second_configuration": False,
            "runtime_authority_changed": False,
            "production_promotion": False,
            "isolated_runtime": True,
            "python_no_user_site": True,
            "venv_bootstrap": "stdlib-venv-without-pip+host-pip--python",
            "report_sha256": gate.EXPECTED_V53G_REPORT_SHA256,
            "artifact_sha256": dict(gate.EXPECTED_RESCUE_ARTIFACT_SHA256),
            "group_fingerprints": groups,
        }
        return report, envelope

    def test_contract_is_read_only_and_keeps_protected_surfaces_closed(self):
        boundary = gate.safety_boundary()
        self.assertFalse(boundary["training"])
        self.assertFalse(boundary["backward"])
        self.assertEqual(boundary["optimizer_steps"], 0)
        self.assertFalse(boundary["checkpoint_write"])
        self.assertFalse(boundary["rescue_artifact_write"])
        self.assertFalse(boundary["historical_validation_opened"])
        self.assertFalse(boundary["first30_opened"])
        self.assertFalse(boundary["v5_validation_opened"])
        self.assertTrue(boundary["final_holdout_locked"])
        self.assertFalse(boundary["retraining_authorized_on_hold"])
        self.assertFalse(gate.retraining_allowed_after_hold())
        self.assertFalse(gate.historical_validation_access_allowed())
        self.assertFalse(gate.first30_access_allowed())
        self.assertFalse(gate.v5_validation_access_allowed())
        self.assertFalse(gate.final_holdout_access_allowed())

    def test_acceptance_contract_is_exact(self):
        contract = gate.acceptance_contract()
        self.assertEqual(contract["v5_train_required_f1"], {"2": 1.0, "3": 1.0})
        self.assertEqual(
            contract["frozen_correct_regression_count_max"]["historical_train"],
            {"2": 0, "3": 0},
        )
        self.assertEqual(contract["rescue_threshold"], 0.50)
        self.assertEqual(contract["frozen_thresholds"], {"2": 0.48, "3": 0.60})
        self.assertTrue(contract["only_rescue_parameters_changed_required"])
        self.assertTrue(contract["hold_does_not_authorize_retraining"])

    def test_exact_completed_execution_receipt_is_admitted(self):
        report, envelope = self._receipt_fixture()
        gate._validate_execution_receipt(report=report, envelope=envelope)

    def test_execution_receipt_tamper_fails_closed(self):
        report, envelope = self._receipt_fixture()
        envelope["v5_validation_opened"] = True
        with self.assertRaises(gate.MeterV5_3IError):
            gate._validate_execution_receipt(report=report, envelope=envelope)

    def test_binary_metrics_exact_perfect_case(self):
        torch, _nn = v52b._import_torch()
        prediction = torch.tensor([True, False, True, False])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        metrics = gate._binary_metrics_from_predictions(prediction, target)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["tn"], 2)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_combined_prediction_repairs_frozen_negative_only(self):
        torch, nn = v52b._import_torch()

        class Frozen(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(v53e.FEATURE_DIM, 1)
                with torch.no_grad():
                    self.head.weight.zero_()
                    self.head.bias.zero_()
                    self.head.weight[0, 0] = 1.0

        class Rescue(nn.Module):
            def forward(self, x):
                return x[:, 1]

        features = torch.zeros((4, v53e.FEATURE_DIM), dtype=torch.float32)
        features[:, 0] = torch.tensor([-2.0, -2.0, 2.0, 2.0])
        features[:, 1] = torch.tensor([2.0, -2.0, -2.0, 2.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 1.0])

        evidence = gate._combined_prediction_evidence(
            digit="2",
            frozen_model=Frozen(),
            rescue_model=Rescue(),
            features=features,
            targets=targets,
        )
        self.assertEqual(evidence["rescue_eligible_count"], 2)
        self.assertEqual(evidence["frozen_correct_regression_count"], 0)
        self.assertEqual(evidence["frozen_incorrect_correction_count"], 1)
        self.assertEqual(evidence["combined_metrics"]["f1"], 1.0)

    def test_sha_bound_json_reader_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            raw = b'{"ok":true}\n'
            path.write_bytes(raw)
            expected = hashlib.sha256(raw).hexdigest()
            self.assertEqual(
                gate._read_json_bound(path, expected_sha256=expected, label="fixture"),
                {"ok": True},
            )
            path.write_text('{"ok":false}\n', encoding="utf-8")
            with self.assertRaises(gate.MeterV5_3IError):
                gate._read_json_bound(path, expected_sha256=expected, label="fixture")

    def test_future_gate_order_does_not_skip_historical_validation_gate(self):
        self.assertEqual(
            gate.future_gate_order(),
            (
                "v5_3i_train_acceptance",
                "separately_staged_historical_validation_retention",
                "immutable_v5_first30_diagnostic",
                "separately_authorized_v5_validation",
                "separately_authorized_final_holdout",
            ),
        )

    def test_module_contains_no_training_entry(self):
        source = Path(gate.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "torch.optim.",
            ".backward(",
            "optimizer.step(",
            "run_authoritative_rescue_training_v1(",
            "run_historical_retention_gate(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
