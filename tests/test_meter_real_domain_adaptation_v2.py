from __future__ import annotations

import importlib.util
import unittest

from st_omr_training.meter_real_domain_adaptation_v1 import MeterEvaluationV1
from st_omr_training.meter_real_domain_adaptation_v2 import (
    FROZEN_ADAPTATION_CONFIG_V2,
    MeterRealDomainAdaptationConfigV2,
    adaptation_acceptance_v2,
    build_meter_glyph_adapter_v2,
    meter_real_domain_adaptation_fingerprint_v2,
    production_promotion_allowed,
    resolver_connection_allowed,
    runtime_connection_allowed,
    sealed_test_access_allowed,
)


def _evaluation(
    *,
    macro: float,
    accuracy: float,
    localization: float,
    none_recall: float = 1.0,
    positive_recall: float = 1.0,
) -> MeterEvaluationV1:
    return MeterEvaluationV1(
        loss=0.2,
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
        confusion=((9, 0, 0, 0), (0, 3, 0, 0), (0, 0, 3, 0), (0, 0, 0, 3)),
    )


class MeterRealDomainAdaptationV2ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.synthetic_baseline = _evaluation(macro=0.91, accuracy=0.94, localization=0.71)

    def test_accepts_only_real_result_above_90_without_synthetic_forgetting(self) -> None:
        decision = adaptation_acceptance_v2(
            candidate_real=_evaluation(macro=0.96, accuracy=17 / 18, localization=0.70),
            baseline_synthetic=self.synthetic_baseline,
            candidate_synthetic=_evaluation(macro=0.90, accuracy=0.93, localization=0.70),
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reasons, ())

    def test_rejects_16_of_18_and_any_positive_class_error(self) -> None:
        candidate = _evaluation(
            macro=0.89,
            accuracy=16 / 18,
            localization=0.70,
            positive_recall=2 / 3,
        )
        decision = adaptation_acceptance_v2(
            candidate_real=candidate,
            baseline_synthetic=self.synthetic_baseline,
            candidate_synthetic=self.synthetic_baseline,
        )
        self.assertFalse(decision.accepted)
        self.assertIn("REAL_ACCURACY_BELOW_90_PERCENT", decision.reasons)
        self.assertIn("REAL_3_4_RECALL_NOT_3_OF_3", decision.reasons)

    def test_rejects_synthetic_regression_even_when_real_is_perfect(self) -> None:
        decision = adaptation_acceptance_v2(
            candidate_real=_evaluation(macro=1.0, accuracy=1.0, localization=0.90),
            baseline_synthetic=self.synthetic_baseline,
            candidate_synthetic=_evaluation(macro=0.88, accuracy=0.90, localization=0.67),
        )
        self.assertFalse(decision.accepted)
        self.assertIn("SYNTHETIC_MACRO_F1_REGRESSION", decision.reasons)
        self.assertIn("SYNTHETIC_LOCALIZATION_REGRESSION", decision.reasons)

    def test_config_freezes_small_adapter_surface_and_closed_boundaries(self) -> None:
        self.assertEqual(
            FROZEN_ADAPTATION_CONFIG_V2.trainable_surface,
            "glyph-adapter-only-d11-fully-frozen",
        )
        self.assertEqual(FROZEN_ADAPTATION_CONFIG_V2.real_min_accuracy_milli, 900)
        self.assertEqual(FROZEN_ADAPTATION_CONFIG_V2.real_min_macro_f1_milli, 900)
        self.assertEqual(
            (FROZEN_ADAPTATION_CONFIG_V2.glyph_x0, FROZEN_ADAPTATION_CONFIG_V2.glyph_x1),
            (56, 248),
        )
        with self.assertRaises(ValueError):
            MeterRealDomainAdaptationConfigV2(trainable_surface="full-model")
        with self.assertRaises(ValueError):
            MeterRealDomainAdaptationConfigV2(glyph_x0=145, glyph_x1=144)
        self.assertFalse(sealed_test_access_allowed())
        self.assertFalse(runtime_connection_allowed())
        self.assertFalse(resolver_connection_allowed())
        self.assertFalse(production_promotion_allowed())

    def test_profile_is_deterministic_and_binds_configuration(self) -> None:
        arguments = {
            "teacher_manifest_sha256": "1" * 64,
            "d10_manifest_sha256": "2" * 64,
            "d10_artifact_binding_sha256": "3" * 64,
        }
        first = meter_real_domain_adaptation_fingerprint_v2(**arguments)
        self.assertEqual(first, meter_real_domain_adaptation_fingerprint_v2(**arguments))
        self.assertEqual(len(first), 64)
        self.assertNotEqual(
            first,
            meter_real_domain_adaptation_fingerprint_v2(
                **{**arguments, "teacher_manifest_sha256": "4" * 64}
            ),
        )


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed")
class MeterGlyphAdapterV2TorchTests(unittest.TestCase):
    def test_zero_initialized_adapter_preserves_base_and_freezes_it(self) -> None:
        import torch
        from torch import nn

        class FakeBase(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))

            def forward(self, images):
                batch = images.shape[0]
                logits = torch.tensor([[3.0, 1.0, 0.0, -1.0]], dtype=torch.float32).repeat(batch, 1)
                boxes = torch.tensor([[0.2, 0.2, 0.4, 0.6]], dtype=torch.float32).repeat(batch, 1)
                return logits * self.weight, boxes

        base = FakeBase()
        model = build_meter_glyph_adapter_v2(base)
        images = torch.zeros((2, 1, 192, 256), dtype=torch.float32)
        logits, boxes = model(images)
        expected_logits, expected_boxes = base(images)
        self.assertTrue(torch.equal(logits, expected_logits))
        self.assertTrue(torch.equal(boxes, expected_boxes))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.base.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.glyph_encoder.parameters()))

    def test_presence_and_digit_heads_compose_deterministic_meter_classes(self) -> None:
        import torch
        from torch import nn

        class ZeroBase(nn.Module):
            def forward(self, images):
                batch = images.shape[0]
                return torch.zeros((batch, 4)), torch.zeros((batch, 4))

        model = build_meter_glyph_adapter_v2(ZeroBase())
        with torch.no_grad():
            model.presence_head.bias.fill_(2.0)
            model.digit_head.bias.copy_(torch.tensor([2.0, 0.0, -2.0]))
        logits, _boxes = model(torch.zeros((1, 1, 192, 256), dtype=torch.float32))
        self.assertEqual(int(logits.argmax(1).item()), 1)
        self.assertGreater(float(logits[0, 1]), float(logits[0, 0]))


if __name__ == "__main__":
    unittest.main()
