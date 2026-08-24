from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from st_omr_training import meter_v5_1_bbox_pilot as v51
from st_omr_training import meter_v5_2b_specialist_adaptation as v52b
from st_omr_training import meter_v5_3e_rescue_training_preregistration_v1 as v53e
from st_omr_training import meter_v5_3f_rescue_training_execution_harness_v1 as v53f
from st_omr_training import meter_v5_3g_authoritative_rescue_training_v1 as v53g


class TestMeterV53GAuthoritativeRescueTrainingV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, cls.nn = v52b._import_torch()

    def _frozen_model(self):
        torch, nn = self.torch, self.nn

        class Frozen(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(64, 1)
                with torch.no_grad():
                    self.head.weight.zero_()
                    self.head.weight[0, 0] = 1.0
                    self.head.bias.zero_()

        return Frozen().cpu()

    def _features(self):
        torch = self.torch
        values = torch.zeros((4, 64), dtype=torch.float32)
        values[:, 0] = torch.tensor([-2.0, -2.0, 2.0, 2.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        return values, targets

    def test_binds_exact_v5_3f_head_and_blobs(self):
        contract = v53g.prerequisite_contract()
        self.assertEqual(
            contract["v5_3f_head_sha"],
            "7ed41f2872058ac5e3e52df756b9098a1d60052d",
        )
        self.assertEqual(
            contract["v5_3f_module_blob_sha"],
            "908b5b7f83fc5a5358261b7dc04ab606ee66e063",
        )
        self.assertEqual(
            contract["v5_3f_doc_blob_sha"],
            "164f84b3dc89230024c8a62ef189204adfe4ebed",
        )
        self.assertEqual(contract["v5_3f_schema"], v53f.SCHEMA)

    def test_safety_boundary_keeps_protected_surfaces_closed(self):
        boundary = v53g.safety_boundary()
        self.assertEqual(boundary["data_surfaces"], ("v5_train", "historical_train"))
        self.assertTrue(boundary["digit4_frozen"])
        self.assertFalse(boundary["digit4_loaded"])
        self.assertFalse(boundary["threshold_tuning"])
        self.assertFalse(boundary["hyperparameter_sweep"])
        self.assertFalse(boundary["automatic_second_configuration"])
        self.assertFalse(boundary["historical_validation_opened"])
        self.assertFalse(boundary["first30_opened"])
        self.assertFalse(boundary["v5_reserve_opened"])
        self.assertFalse(boundary["v5_validation_opened"])
        self.assertTrue(boundary["final_holdout_locked"])
        self.assertFalse(boundary["resolver_wiring"])
        self.assertFalse(boundary["production_promotion"])

    def test_frozen_probability_uses_head_on_64d_features(self):
        torch = self.torch
        model = self._frozen_model()
        features, _targets = self._features()
        probabilities = v53g._frozen_probabilities_from_features(
            model, features, digit="2"
        )
        expected = torch.sigmoid(features[:, 0])
        self.assertTrue(torch.equal(probabilities, expected))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_materializer_selects_only_frozen_negative_rows(self):
        torch = self.torch
        model = self._frozen_model()
        v5_x, v5_y = self._features()
        hist_x, hist_y = self._features()
        v5_before = v5_x.clone()
        hist_before = hist_x.clone()

        groups, evidence = v53g._materialize_frozen_negative_groups_v1(
            digit="2",
            model=model,
            v5_features=v5_x,
            v5_targets=v5_y,
            historical_features=hist_x,
            historical_targets=hist_y,
            enforce_preregistered_counts=False,
        )

        self.assertEqual(tuple(groups), v53e.TRAIN_GROUPS)
        self.assertEqual(
            evidence["group_counts"],
            {
                "v5_frozen_false_negative_positive": 1,
                "v5_frozen_true_negative": 1,
                "historical_frozen_false_negative_positive": 1,
                "historical_frozen_true_negative": 1,
            },
        )
        for tensor in groups.values():
            self.assertEqual(tuple(tensor.shape), (1, 64))
            self.assertLess(float(tensor[0, 0].item()), 0.0)
        self.assertTrue(torch.equal(v5_x, v5_before))
        self.assertTrue(torch.equal(hist_x, hist_before))

    def test_materializer_exact_count_guard_fails_closed_on_synthetic_surface(self):
        model = self._frozen_model()
        v5_x, v5_y = self._features()
        hist_x, hist_y = self._features()
        with self.assertRaises(v53g.MeterV5_3GError):
            v53g._materialize_frozen_negative_groups_v1(
                digit="2",
                model=model,
                v5_features=v5_x,
                v5_targets=v5_y,
                historical_features=hist_x,
                historical_targets=hist_y,
                enforce_preregistered_counts=True,
            )

    def test_tensor_fingerprint_is_deterministic_and_value_sensitive(self):
        torch = self.torch
        x = torch.zeros((2, 64), dtype=torch.float32)
        first = v53g._tensor_fingerprint(x, name="2:test")
        second = v53g._tensor_fingerprint(x.clone(), name="2:test")
        changed = x.clone()
        changed[0, 0] = 1.0
        third = v53g._tensor_fingerprint(changed, name="2:test")
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_rescue_artifact_roundtrip_contains_only_rescue_state(self):
        model = v53f._build_rescue_model_v1()
        state = v53f._state_fingerprint(model)
        materialization = {
            "group_counts": {name: 1 for name in v53e.TRAIN_GROUPS},
            "group_fingerprints": {name: "a" * 64 for name in v53e.TRAIN_GROUPS},
        }
        execution = {
            "optimizer_steps": v53e.FIXED_OPTIMIZER_STEPS,
            "initial_state_fingerprint": state,
            "final_state_fingerprint": state,
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "digit_2_rescue.pt"
            saved = v53g._save_rescue_artifact(
                model=model,
                path=path,
                digit="2",
                source_checkpoint_sha256=v52b.DIGIT2_SHA256,
                slot_manifest_sha256="b" * 64,
                materialization=materialization,
                execution=execution,
            )
            self.assertTrue(path.is_file())
            self.assertTrue(saved["reload_verified"])
            payload = self.torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(set(payload), {"metadata", "rescue_state_dict"})
            self.assertEqual(payload["metadata"]["trainable_surface"], "new-rescue-parameters-only")

    def test_preflight_refuses_any_existing_execution_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ann = root / v51.ANNOTATIONS_DIR
            ann.mkdir(parents=True)
            report = ann / v53g.REPORT_NAME
            report.write_text("{}", encoding="utf-8")
            with self.assertRaises(v53g.MeterV5_3GError):
                v53g._preflight_outputs(root)

    def test_wrong_approval_token_fails_before_data_access(self):
        with self.assertRaisesRegex(v53g.MeterV5_3GError, "before data access"):
            v53g.run_authoritative_rescue_training_v1(
                "/definitely/not/a/v5/root",
                m4a_root="/not/read",
                d10_root="/not/read",
                digit2_frozen="/not/read",
                digit3_frozen="/not/read",
                confirmation="WRONG",
            )

    def test_authoritative_entry_reuses_v52n_and_v53f_without_alternate_optimizer(self):
        source = inspect.getsource(v53g.run_authoritative_rescue_training_v1)
        self.assertIn("v52n._frozen_models", source)
        self.assertIn("v52n._v5_surface", source)
        self.assertIn("v52n._historical_surface", source)
        self.assertIn("v53f.execute_rescue_tensor_harness_v1", source)
        self.assertNotIn("torch.optim.", source)
        self.assertNotIn("threshold =", source)
        self.assertNotIn("randperm", source)

    def test_stage_stops_before_train_performance_and_protected_gates(self):
        self.assertFalse(v53g.train_performance_gate_executed_by_this_module())
        self.assertFalse(v53g.protected_evaluation_access_allowed())
        self.assertFalse(v53g.production_promotion_allowed())
        contract = v53g.execution_contract()
        self.assertTrue(contract["one_shot_non_overwriting"])
        self.assertTrue(contract["exact_sha_colab_wrapper_required_for_external_execution"])
        self.assertFalse(contract["colab_execution_wrapper_present"])

    def test_future_gate_order_requires_external_exact_sha_wrapper_before_receipt(self):
        self.assertEqual(
            v53g.future_gate_order(),
            (
                "v5_3g_exact_ci_green_sha",
                "exact_sha_external_execution_wrapper",
                "single_authoritative_execution_receipt",
                "train_v5_f1_and_frozen_correct_retention",
                "historical_validation_retention_at_frozen_thresholds",
                "immutable_v5_first30_diagnostic",
                "separately_authorized_v5_validation",
                "separately_authorized_final_holdout",
            ),
        )


if __name__ == "__main__":
    unittest.main()
