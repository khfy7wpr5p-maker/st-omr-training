from __future__ import annotations

from copy import deepcopy
import unittest

from st_omr_training.stage7d13_symbol_training_contract import (
    ACCEPTANCE,
    D12_CLASS_INVENTORY,
    EXPECTED_D12_ARTIFACT_BINDING_SHA256,
    EXPECTED_D12_DERIVATIVE_BUILD_ID,
    EXPECTED_D12_MANIFEST_SHA256,
    EXPECTED_D12_REPOSITORY_SHA,
    EXPECTED_SOURCE_FAMILY_COUNTS,
    EXPECTED_SOURCE_SAMPLE_COUNTS,
    FROZEN_D13_CONFIG,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    MAX_PARAMETERS_COMBINED,
    MAX_PARAMETERS_PER_SPECIALIST,
    MIN_TRAIN_INSTANCES_PER_CLASS,
    MIN_VALIDATION_INSTANCES_PER_CLASS,
    SPECIALIST_CLASSES,
    TEST_SPECIALIST_RECORDS,
    Stage7D13ContractError,
    class_readiness_violations,
    positive_class_weights,
    stage7d13_contract_fingerprint,
    stage7d13_contract_payload,
)


class Stage7D13SymbolTrainingContractTests(unittest.TestCase):
    def test_exact_accepted_d12_identity_is_frozen(self) -> None:
        self.assertEqual(
            EXPECTED_D12_REPOSITORY_SHA,
            "e2de6f64c27be2dd6d706a700553ef4f5c236e25",
        )
        self.assertEqual(
            EXPECTED_D12_DERIVATIVE_BUILD_ID,
            "35323e831c5c693bf607808c5f846624445bf537f30e1d93db9ca949a7eed106",
        )
        self.assertEqual(
            EXPECTED_D12_MANIFEST_SHA256,
            "a372eba640b38704020922ad4eb102738fc4492d278a38e4b51b8ad0b78d4ea1",
        )
        self.assertEqual(
            EXPECTED_D12_ARTIFACT_BINDING_SHA256,
            "14c64e16ca2f993bf94f8009bf0bcd974b7ddee87c19bb748219ba3f774b229d",
        )
        self.assertEqual(EXPECTED_SOURCE_SAMPLE_COUNTS, {"train": 1230, "validation": 153})
        self.assertEqual(EXPECTED_SOURCE_FAMILY_COUNTS, {"train": 410, "validation": 51})
        self.assertEqual(TEST_SPECIALIST_RECORDS, 0)

    def test_verified_d12_inventory_satisfies_frozen_readiness(self) -> None:
        self.assertEqual(MIN_TRAIN_INSTANCES_PER_CLASS, 1000)
        self.assertEqual(MIN_VALIDATION_INSTANCES_PER_CLASS, 150)
        self.assertEqual(class_readiness_violations(), ())

        too_small = deepcopy(D12_CLASS_INVENTORY)
        too_small["train"]["accidental"]["natural"] = 999
        too_small["validation"]["rest"]["half"] = 149
        violations = class_readiness_violations(too_small)
        self.assertTrue(any("accidental.natural" in row for row in violations))
        self.assertTrue(any("rest.half" in row for row in violations))

    def test_training_only_class_weights_are_deterministic_and_bounded(self) -> None:
        first = positive_class_weights("accidental")
        second = positive_class_weights("accidental")
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(SPECIALIST_CLASSES["accidental"]))
        self.assertTrue(all(0.5 <= value <= 3.0 for value in first.values()))
        self.assertGreater(first["natural"], first["sharp"])
        self.assertGreater(first["natural"], first["flat"])

        changed_validation = deepcopy(D12_CLASS_INVENTORY)
        changed_validation["validation"]["accidental"]["natural"] = 10_000_000
        self.assertEqual(
            first,
            positive_class_weights("accidental", changed_validation),
        )

        with self.assertRaises(Stage7D13ContractError):
            positive_class_weights("unknown")

    def test_measure_derivative_and_model_boundaries_are_frozen(self) -> None:
        payload = stage7d13_contract_payload()
        derivative = payload["measure_derivative"]
        model = payload["model"]
        self.assertEqual((INPUT_WIDTH, INPUT_HEIGHT), (512, 128))
        self.assertEqual(derivative["crop_authority"], "accepted_d12_measure_bbox")
        self.assertIs(derivative["letterbox_preserve_aspect_ratio"], True)
        self.assertEqual(derivative["test_derivatives"], 0)
        self.assertIs(model["separate_specialist_weights"], True)
        self.assertIs(model["pretrained_external_backbone"], False)
        self.assertEqual(model["max_parameters_per_specialist"], MAX_PARAMETERS_PER_SPECIALIST)
        self.assertEqual(model["max_parameters_combined"], MAX_PARAMETERS_COMBINED)

    def test_optimizer_profile_and_checkpoint_policy_are_frozen(self) -> None:
        self.assertEqual(FROZEN_D13_CONFIG.batch_size, 16)
        self.assertEqual(FROZEN_D13_CONFIG.epochs, 10)
        self.assertEqual(FROZEN_D13_CONFIG.learning_rate_micros, 700)
        self.assertEqual(FROZEN_D13_CONFIG.weight_decay_micros, 100)
        self.assertEqual(FROZEN_D13_CONFIG.grad_clip_milli, 1000)
        self.assertEqual(FROZEN_D13_CONFIG.optimizer, "adamw")
        self.assertEqual(
            FROZEN_D13_CONFIG.checkpoint_selection,
            "min_validation_loss_per_specialist",
        )
        self.assertEqual(
            stage7d13_contract_payload()["optimizer_authorization"],
            "blocked_until_d13_derivative_and_code_gates_pass",
        )

    def test_acceptance_thresholds_are_frozen_per_specialist(self) -> None:
        self.assertEqual(
            (
                ACCEPTANCE["notehead"].class_aware_center_f1_4px_milli,
                ACCEPTANCE["notehead"].class_aware_bbox_f1_iou50_milli,
                ACCEPTANCE["notehead"].macro_class_f1_milli,
            ),
            (850, 750, 900),
        )
        self.assertEqual(
            (
                ACCEPTANCE["rest"].class_aware_center_f1_4px_milli,
                ACCEPTANCE["rest"].class_aware_bbox_f1_iou50_milli,
                ACCEPTANCE["rest"].macro_class_f1_milli,
            ),
            (800, 700, 850),
        )
        self.assertEqual(
            (
                ACCEPTANCE["accidental"].class_aware_center_f1_4px_milli,
                ACCEPTANCE["accidental"].class_aware_bbox_f1_iou50_milli,
                ACCEPTANCE["accidental"].macro_class_f1_milli,
            ),
            (800, 700, 850),
        )

    def test_contract_fingerprint_is_deterministic_sha256(self) -> None:
        first = stage7d13_contract_fingerprint()
        second = stage7d13_contract_fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertLessEqual(set(first), set("0123456789abcdef"))


if __name__ == "__main__":
    unittest.main()
