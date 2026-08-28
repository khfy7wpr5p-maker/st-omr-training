from pathlib import Path
import unittest

from st_omr_training import meter_v5_4_unified_numerator_shadow_contract_v1 as v54


class TestMeterV54UnifiedNumeratorShadowContract(unittest.TestCase):
    def test_exactly_one_shared_numerator_classifier_with_three_classes(self):
        contract = v54.architecture_contract()
        self.assertEqual(contract["trainable_numerator_model_count"], 1)
        self.assertEqual(contract["numerator_classes"], ("2", "3", "4"))
        self.assertTrue(contract["legacy_specialists_remain_controls"])
        self.assertTrue(contract["shadow_only"])
        self.assertFalse(contract["legacy_path_replacement_authorized"])

    def test_contract_keeps_all_protected_surfaces_closed(self):
        boundary = v54.safety_boundary()
        self.assertFalse(boundary["training"])
        self.assertFalse(boundary["fitting"])
        self.assertEqual(boundary["optimizer_steps"], 0)
        self.assertFalse(boundary["checkpoint_write"])
        self.assertFalse(boundary["model_mutation"])
        self.assertFalse(boundary["threshold_tuning"])
        self.assertFalse(boundary["crop_bbox_tuning"])
        self.assertFalse(boundary["historical_validation_opened"])
        self.assertFalse(boundary["first30_opened"])
        self.assertFalse(boundary["v5_reserve_opened"])
        self.assertFalse(boundary["v5_validation_opened"])
        self.assertTrue(boundary["final_holdout_locked"])
        self.assertFalse(boundary["resolver_wiring"])
        self.assertFalse(boundary["production_promotion"])
        self.assertFalse(v54.production_promotion_allowed())
        self.assertFalse(v54.final_holdout_access_allowed())

    def test_training_preregistration_requires_bound_v53k_external_report(self):
        self.assertFalse(
            v54.training_preregistration_allowed(v5_3k_external_report_bound=False)
        )
        self.assertTrue(
            v54.training_preregistration_allowed(v5_3k_external_report_bound=True)
        )

    def test_adapter_only_composes_admitted_2_3_4_over_denominator_4(self):
        for digit in ("2", "3", "4"):
            with self.subTest(digit=digit):
                result = v54.compose_shadow_meter_candidate(
                    digit,
                    "4",
                    numerator_admitted=True,
                    denominator_admitted=True,
                )
                self.assertIsNotNone(result)
                self.assertEqual(result.meter, f"{digit}/4")
                self.assertEqual(result.status, "SHADOW_ONLY")

    def test_adapter_abstains_on_unadmitted_or_unsupported_evidence(self):
        self.assertIsNone(
            v54.compose_shadow_meter_candidate(
                "2", "4", numerator_admitted=False, denominator_admitted=True
            )
        )
        self.assertIsNone(
            v54.compose_shadow_meter_candidate(
                "3", "4", numerator_admitted=True, denominator_admitted=False
            )
        )
        self.assertIsNone(
            v54.compose_shadow_meter_candidate(
                "6", "8", numerator_admitted=True, denominator_admitted=True
            )
        )
        self.assertIsNone(
            v54.compose_shadow_meter_candidate(
                "2", "8", numerator_admitted=True, denominator_admitted=True
            )
        )

    def test_future_gate_order_keeps_final_holdout_last(self):
        gates = v54.future_gate_order()
        self.assertEqual(gates[0], "complete_and_hash_bind_v5_3k_external_forensics")
        self.assertEqual(gates[-1], "one_time_untouched_final_holdout")
        self.assertIn("shadow_adapter_and_meter_validator", gates)

    def test_source_contains_no_training_entry_points(self):
        source = Path(v54.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "torch.optim.",
            ".backward(",
            "optimizer.step(",
            "torch.save(",
            "run_authoritative_rescue_training_v1(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
