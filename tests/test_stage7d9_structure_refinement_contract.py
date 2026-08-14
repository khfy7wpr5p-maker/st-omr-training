from __future__ import annotations

import inspect
import json
import unittest

from st_omr_training.stage7d4_specialist_architecture import V1_SPECIALIST_TASKS
from st_omr_training.stage7d9_structure_refinement_contract import (
    BARLINE_ROI,
    D8_BASELINE_DICE,
    D8_TOLERANT_F1_2PX,
    D9_ACCEPTANCE,
    EXPECTED_D8_REPORT_SHA256,
    EXPECTED_D8_REPOSITORY_SHA,
    METER_CLASSES,
    METER_ROI,
    ROI_POLICIES,
    STAGE7D9_SCHEMA,
    STAGE7D9_VERSION,
    STRUCTURE_CORE_CHANNELS,
    STRUCTURE_REFINEMENT_COMPONENTS,
    D9AcceptancePolicy,
    LocalRoiPolicy,
    RefinementComponentContract,
    stage7d9_contract_fingerprint,
    stage7d9_contract_payload,
)


class Stage7D9EvidenceTests(unittest.TestCase):
    def test_exact_d8_evidence_is_frozen(self) -> None:
        self.assertEqual(
            EXPECTED_D8_REPOSITORY_SHA,
            "e0e721bf5a6d13025546fdf5eeb755647eef383f",
        )
        self.assertEqual(
            EXPECTED_D8_REPORT_SHA256,
            "46de5f6766f78bb567f70794a364ccd44835d09af94ef29c3f1eab5cd13ce968",
        )
        self.assertLess(D8_BASELINE_DICE["barline"], 0.30)
        self.assertLess(D8_TOLERANT_F1_2PX["barline"], 0.40)
        for channel in ("meter_2_4", "meter_3_4", "meter_4_4"):
            self.assertLess(D8_BASELINE_DICE[channel], 0.36)
            self.assertLess(D8_TOLERANT_F1_2PX[channel], 0.41)
        self.assertGreater(D8_BASELINE_DICE["system_region"], 0.90)
        self.assertGreater(D8_BASELINE_DICE["measure_region"], 0.80)
        self.assertGreater(D8_BASELINE_DICE["clef_g2"], 0.80)

    def test_contract_payload_and_fingerprint_are_canonical(self) -> None:
        payload = stage7d9_contract_payload()
        self.assertEqual(payload["schema_version"], STAGE7D9_SCHEMA)
        self.assertEqual(payload["stage7d9_version"], STAGE7D9_VERSION)
        raw_a = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        raw_b = json.dumps(
            stage7d9_contract_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        self.assertEqual(raw_a, raw_b)
        fingerprint = stage7d9_contract_fingerprint()
        self.assertEqual(len(fingerprint), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in fingerprint))


class Stage7D9ExternalContractTests(unittest.TestCase):
    def test_d4_structure_external_outputs_are_unchanged(self) -> None:
        original = tuple(task for task in V1_SPECIALIST_TASKS if task.task_id == "structure")
        self.assertEqual(len(original), 1)
        external = stage7d9_contract_payload()["external_contract"]
        self.assertEqual(external["task_id"], "structure")
        self.assertEqual(external["dataset_name"], "StructureSet")
        self.assertEqual(tuple(external["outputs"]), original[0].outputs)
        self.assertFalse(external["contract_change"])

    def test_core_preserves_only_accepted_strong_channels(self) -> None:
        self.assertEqual(
            STRUCTURE_CORE_CHANNELS,
            ("system_region", "measure_region", "clef_g2"),
        )
        components = {item.component_id: item for item in STRUCTURE_REFINEMENT_COMPONENTS}
        core = components["structure_core"]
        self.assertFalse(core.trainable)
        self.assertTrue(core.accepted_d7_weights_frozen)
        self.assertEqual(core.max_trainable_parameters, 0)
        self.assertEqual(core.roi_policy_id, None)

    def test_weak_channels_are_not_left_in_shared_core(self) -> None:
        self.assertNotIn("barline", STRUCTURE_CORE_CHANNELS)
        self.assertNotIn("meter_2_4", STRUCTURE_CORE_CHANNELS)
        components = {item.component_id: item for item in STRUCTURE_REFINEMENT_COMPONENTS}
        self.assertIn("barline_segment", components["barline_refiner"].outputs)
        self.assertIn("meter_class", components["meter_refiner"].outputs)
        self.assertIn("meter_bbox", components["meter_refiner"].outputs)


class Stage7D9RoiTests(unittest.TestCase):
    def test_barline_is_measure_end_high_resolution_roi(self) -> None:
        self.assertEqual(BARLINE_ROI.anchor, "measure_end")
        self.assertEqual(BARLINE_ROI.output_height, 192)
        self.assertEqual(BARLINE_ROI.output_width, 128)
        self.assertGreater(BARLINE_ROI.x_before_staff_spacings_milli, 0)
        components = {item.component_id: item for item in STRUCTURE_REFINEMENT_COMPONENTS}
        self.assertEqual(
            components["barline_refiner"].roi_policy_id,
            BARLINE_ROI.policy_id,
        )
        self.assertIn("five_staff_lines", components["barline_refiner"].inputs)
        self.assertIn("measure_bbox", components["barline_refiner"].inputs)

    def test_meter_is_measure_start_roi_with_explicit_none_class(self) -> None:
        self.assertEqual(METER_ROI.anchor, "measure_start")
        self.assertEqual(METER_CLASSES, ("none", "2/4", "3/4", "4/4"))
        self.assertGreater(METER_ROI.x_after_staff_spacings_milli, 0)
        components = {item.component_id: item for item in STRUCTURE_REFINEMENT_COMPONENTS}
        meter = components["meter_refiner"]
        self.assertEqual(meter.roi_policy_id, METER_ROI.policy_id)
        self.assertIn("meter_class", meter.outputs)
        self.assertIn("meter_bbox", meter.outputs)

    def test_roi_policies_are_unique_and_bounded(self) -> None:
        self.assertEqual(len({policy.policy_id for policy in ROI_POLICIES}), len(ROI_POLICIES))
        for policy in ROI_POLICIES:
            self.assertGreaterEqual(policy.output_height, 64)
            self.assertLessEqual(policy.output_height, 512)
            self.assertGreaterEqual(policy.output_width, 64)
            self.assertLessEqual(policy.output_width, 1024)
            self.assertEqual(policy.resize_mode, "fit-pad-preserve-aspect-v1")

    def test_invalid_roi_policy_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            LocalRoiPolicy(
                policy_id="bad",
                anchor="whole_page",
                x_before_staff_spacings_milli=0,
                x_after_staff_spacings_milli=1000,
                y_before_staff_spacings_milli=1000,
                y_after_staff_spacings_milli=1000,
                output_height=192,
                output_width=128,
            )
        with self.assertRaises(ValueError):
            LocalRoiPolicy(
                policy_id="bad",
                anchor="measure_end",
                x_before_staff_spacings_milli=-1,
                x_after_staff_spacings_milli=1000,
                y_before_staff_spacings_milli=1000,
                y_after_staff_spacings_milli=1000,
                output_height=192,
                output_width=128,
            )


class Stage7D9SafetyTests(unittest.TestCase):
    def test_future_refinement_acceptance_keeps_test_sealed_and_core_frozen(self) -> None:
        self.assertEqual(D9_ACCEPTANCE.test_records, 0)
        self.assertFalse(D9_ACCEPTANCE.core_model_mutation_allowed)
        self.assertEqual(D9_ACCEPTANCE.max_total_new_trainable_parameters, 1_250_000)
        trainable = tuple(item for item in STRUCTURE_REFINEMENT_COMPONENTS if item.trainable)
        self.assertEqual({item.component_id for item in trainable}, {"barline_refiner", "meter_refiner"})
        self.assertLessEqual(
            sum(item.max_trainable_parameters for item in trainable),
            D9_ACCEPTANCE.max_total_new_trainable_parameters,
        )

    def test_pretraining_acceptance_gates_are_frozen_before_optimizer_run(self) -> None:
        self.assertEqual(D9_ACCEPTANCE.barline_min_strict_dice_milli, 500)
        self.assertEqual(D9_ACCEPTANCE.barline_min_tolerant_f1_2px_milli, 700)
        self.assertEqual(D9_ACCEPTANCE.meter_min_macro_f1_milli, 800)
        self.assertEqual(D9_ACCEPTANCE.meter_min_positive_localization_f1_2px_milli, 600)

    def test_invalid_acceptance_policy_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            D9AcceptancePolicy(test_records=1)
        with self.assertRaises(ValueError):
            D9AcceptancePolicy(core_model_mutation_allowed=True)
        with self.assertRaises(ValueError):
            D9AcceptancePolicy(barline_min_strict_dice_milli=1001)

    def test_component_contract_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            RefinementComponentContract(
                component_id="bad",
                responsibility="bad",
                depends_on=(),
                inputs=("x",),
                outputs=("y",),
                trainable=False,
                accepted_d7_weights_frozen=True,
                max_trainable_parameters=1,
                roi_policy_id=None,
            )

    def test_contract_module_is_declarative_only(self) -> None:
        import st_omr_training.stage7d9_structure_refinement_contract as module

        source = inspect.getsource(module)
        forbidden = (
            "torch.optim",
            ".backward(",
            ".step(",
            "DataLoader(",
            "torch.load(",
        )
        for token in forbidden:
            self.assertNotIn(token, source)
        self.assertIn("test-forbidden", stage7d9_contract_payload()["split_policy"])


if __name__ == "__main__":
    unittest.main()
