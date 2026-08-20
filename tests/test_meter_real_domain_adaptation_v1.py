from __future__ import annotations

import unittest

from st_omr_training.meter_real_domain_adaptation_v1 import (
    FROZEN_ADAPTATION_CONFIG_V1,
    MeterEvaluationV1,
    MeterRealDomainAdaptationConfigV1,
    adaptation_acceptance_v1,
    balanced_class_weight_values_v1,
    deterministic_replay_ids_v1,
    meter_real_domain_adaptation_fingerprint_v1,
    production_promotion_allowed,
    resolver_connection_allowed,
    run_meter_real_domain_adaptation_v1,
    runtime_connection_allowed,
    sealed_test_access_allowed,
)


def _evaluation(
    *,
    macro: float,
    accuracy: float,
    localization: float,
    none_recall: float,
    positive_recall: float,
) -> MeterEvaluationV1:
    return MeterEvaluationV1(
        loss=1.0 - macro,
        macro_f1=macro,
        accuracy=accuracy,
        positive_localization_f1_2px=localization,
        class_counts={"none": 9, "2/4": 3, "3/4": 3, "4/4": 3},
        per_class_recall={
            "none": none_recall,
            "2/4": positive_recall,
            "3/4": positive_recall,
            "4/4": positive_recall,
        },
        confusion=((8, 1, 0, 0), (0, 3, 0, 0), (0, 0, 3, 0), (0, 0, 1, 2)),
    )


class MeterRealDomainAdaptationGateV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline_real = _evaluation(
            macro=0.16,
            accuracy=0.50,
            localization=0.20,
            none_recall=1.0,
            positive_recall=0.0,
        )
        self.baseline_synthetic = _evaluation(
            macro=0.909,
            accuracy=0.91,
            localization=0.711,
            none_recall=0.91,
            positive_recall=0.91,
        )
        self.good_real = _evaluation(
            macro=0.90,
            accuracy=0.89,
            localization=0.80,
            none_recall=8 / 9,
            positive_recall=2 / 3,
        )
        self.good_synthetic = _evaluation(
            macro=0.900,
            accuracy=0.90,
            localization=0.690,
            none_recall=0.90,
            positive_recall=0.90,
        )

    def test_accepts_real_gain_only_when_synthetic_regression_is_bounded(self) -> None:
        decision = adaptation_acceptance_v1(
            baseline_real=self.baseline_real,
            candidate_real=self.good_real,
            baseline_synthetic=self.baseline_synthetic,
            candidate_synthetic=self.good_synthetic,
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reasons, ())

    def test_rejects_candidate_that_still_collapses_a_positive_class(self) -> None:
        collapsed = MeterEvaluationV1(
            loss=self.good_real.loss,
            macro_f1=self.good_real.macro_f1,
            accuracy=self.good_real.accuracy,
            positive_localization_f1_2px=self.good_real.positive_localization_f1_2px,
            class_counts=self.good_real.class_counts,
            per_class_recall={**self.good_real.per_class_recall, "3/4": 0.0},
            confusion=self.good_real.confusion,
        )
        decision = adaptation_acceptance_v1(
            baseline_real=self.baseline_real,
            candidate_real=collapsed,
            baseline_synthetic=self.baseline_synthetic,
            candidate_synthetic=self.good_synthetic,
        )
        self.assertFalse(decision.accepted)
        self.assertIn("REAL_3_4_RECALL_BELOW_MINIMUM", decision.reasons)

    def test_rejects_catastrophic_synthetic_forgetting(self) -> None:
        regressed = _evaluation(
            macro=0.70,
            accuracy=0.72,
            localization=0.50,
            none_recall=0.75,
            positive_recall=0.70,
        )
        decision = adaptation_acceptance_v1(
            baseline_real=self.baseline_real,
            candidate_real=self.good_real,
            baseline_synthetic=self.baseline_synthetic,
            candidate_synthetic=regressed,
        )
        self.assertFalse(decision.accepted)
        self.assertIn("SYNTHETIC_MACRO_F1_REGRESSION", decision.reasons)
        self.assertIn("SYNTHETIC_LOCALIZATION_REGRESSION", decision.reasons)

    def test_balanced_replay_sampler_is_unique_and_deterministic(self) -> None:
        classes = {
            label: [f"{label}-{index:03d}" for index in range(100)]
            for label in ("none", "2/4", "3/4", "4/4")
        }
        first = deterministic_replay_ids_v1(classes, per_class=8, seed=123)
        second = deterministic_replay_ids_v1(classes, per_class=8, seed=123)
        other = deterministic_replay_ids_v1(classes, per_class=8, seed=124)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 32)
        self.assertEqual(len(set(first)), 32)
        for label in classes:
            self.assertEqual(sum(value.startswith(label + "-") for value in first), 8)

    def test_effective_training_weights_counter_real_none_imbalance(self) -> None:
        weights = balanced_class_weight_values_v1(
            {"none": 172, "2/4": 100, "3/4": 100, "4/4": 100}
        )
        self.assertEqual(weights, balanced_class_weight_values_v1(
            {"none": 172, "2/4": 100, "3/4": 100, "4/4": 100}
        ))
        self.assertLess(weights[0], weights[1])
        self.assertEqual(weights[1], weights[2])
        self.assertEqual(weights[2], weights[3])
        self.assertAlmostEqual(sum(weights) / 4, 1.0)
        with self.assertRaisesRegex(RuntimeError, "exactly the four Meter classes"):
            balanced_class_weight_values_v1({"none": 1})

    def test_profile_is_deterministic_and_binds_all_data_identities(self) -> None:
        kwargs = {
            "teacher_manifest_sha256": "1" * 64,
            "d10_manifest_sha256": "2" * 64,
            "d10_artifact_binding_sha256": "3" * 64,
        }
        first = meter_real_domain_adaptation_fingerprint_v1(**kwargs)
        self.assertEqual(first, meter_real_domain_adaptation_fingerprint_v1(**kwargs))
        self.assertEqual(len(first), 64)
        self.assertNotEqual(
            first,
            meter_real_domain_adaptation_fingerprint_v1(**{**kwargs, "teacher_manifest_sha256": "4" * 64}),
        )

    def test_config_and_later_gates_are_frozen(self) -> None:
        self.assertEqual(FROZEN_ADAPTATION_CONFIG_V1.trainable_surface, "projection-classifier-bbox-encoder-frozen")
        with self.assertRaises(ValueError):
            MeterRealDomainAdaptationConfigV1(trainable_surface="full-model")
        self.assertFalse(sealed_test_access_allowed())
        self.assertFalse(runtime_connection_allowed())
        self.assertFalse(resolver_connection_allowed())
        self.assertFalse(production_promotion_allowed())
        with self.assertRaisesRegex(RuntimeError, "requires the frozen configuration"):
            run_meter_real_domain_adaptation_v1(
                teacher_bundle_root="unused",
                d10_root="unused",
                base_checkpoint_path="unused",
                output_root="unused",
                repository_root="unused",
                expected_d10_manifest_sha256="0" * 64,
                expected_d10_artifact_binding_sha256="0" * 64,
                config=MeterRealDomainAdaptationConfigV1(epochs=7),
            )
        with self.assertRaisesRegex(TypeError, "resume must be bool"):
            run_meter_real_domain_adaptation_v1(
                teacher_bundle_root="unused",
                d10_root="unused",
                base_checkpoint_path="unused",
                output_root="unused",
                repository_root="unused",
                expected_d10_manifest_sha256="0" * 64,
                expected_d10_artifact_binding_sha256="0" * 64,
                resume="yes",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
