from __future__ import annotations

import importlib.util
import unittest

from st_omr_training.meter_real_domain_adaptation_v1 import MeterEvaluationV1
from st_omr_training.meter_real_domain_adaptation_v3_a1 import (
    FROZEN_ADAPTATION_CONFIG_V3_A1,
    MeterRealDomainAdaptationConfigV3A1,
    adaptation_acceptance_v3_a1,
    build_meter_classification_adapter_v3_a1,
    meter_real_domain_adaptation_fingerprint_v3_a1,
    production_promotion_allowed,
    resolver_connection_allowed,
    runtime_connection_allowed,
    sealed_test_access_allowed,
    train_batch_v3_a1,
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


class MeterRealDomainAdaptationV3A1ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.synthetic_baseline = _evaluation(macro=0.91, accuracy=0.94, localization=0.71)

    def test_acceptance_gate_is_unchanged(self) -> None:
        accepted = adaptation_acceptance_v3_a1(
            candidate_real=_evaluation(macro=0.96, accuracy=17 / 18, localization=0.71),
            baseline_synthetic=self.synthetic_baseline,
            candidate_synthetic=_evaluation(macro=0.90, accuracy=0.93, localization=0.71),
        )
        self.assertTrue(accepted.accepted)
        rejected = adaptation_acceptance_v3_a1(
            candidate_real=_evaluation(
                macro=0.89,
                accuracy=16 / 18,
                localization=0.71,
                positive_recall=2 / 3,
            ),
            baseline_synthetic=self.synthetic_baseline,
            candidate_synthetic=self.synthetic_baseline,
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("REAL_ACCURACY_BELOW_90_PERCENT", rejected.reasons)
        self.assertIn("REAL_2_4_RECALL_NOT_3_OF_3", rejected.reasons)

    def test_config_is_classification_only_and_fail_closed(self) -> None:
        config = FROZEN_ADAPTATION_CONFIG_V3_A1
        self.assertEqual(
            config.trainable_surface,
            "classification-adapter-only-d11-and-bbox-fully-frozen",
        )
        self.assertEqual(config.residual_zero_loss_milli, 5_000)
        self.assertEqual((config.glyph_x0, config.glyph_x1), (56, 248))
        with self.assertRaises(ValueError):
            MeterRealDomainAdaptationConfigV3A1(trainable_surface="full-model")
        with self.assertRaises(ValueError):
            MeterRealDomainAdaptationConfigV3A1(glyph_x0=200, glyph_x1=100)
        self.assertFalse(sealed_test_access_allowed())
        self.assertFalse(runtime_connection_allowed())
        self.assertFalse(resolver_connection_allowed())
        self.assertFalse(production_promotion_allowed())

    def test_profile_is_deterministic_and_binds_v3_a1_config(self) -> None:
        kwargs = {
            "teacher_manifest_sha256": "1" * 64,
            "d10_manifest_sha256": "2" * 64,
            "d10_artifact_binding_sha256": "3" * 64,
        }
        first = meter_real_domain_adaptation_fingerprint_v3_a1(**kwargs)
        self.assertEqual(first, meter_real_domain_adaptation_fingerprint_v3_a1(**kwargs))
        self.assertEqual(len(first), 64)
        changed = meter_real_domain_adaptation_fingerprint_v3_a1(
            **{**kwargs, "teacher_manifest_sha256": "4" * 64}
        )
        self.assertNotEqual(first, changed)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed")
class MeterClassificationAdapterV3A1TorchTests(unittest.TestCase):
    def _fake_base(self):
        import torch
        from torch import nn

        class FakeBase(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))

            def forward(self, images):
                batch = images.shape[0]
                logits = torch.tensor([[3.0, 1.0, 0.0, -1.0]], dtype=torch.float32).repeat(
                    batch, 1
                )
                boxes = torch.tensor([[0.2, 0.2, 0.4, 0.6]], dtype=torch.float32).repeat(
                    batch, 1
                )
                return logits * self.weight, boxes

        return FakeBase()

    def test_zero_initialized_adapter_preserves_logits_and_exact_bbox(self) -> None:
        import torch

        base = self._fake_base()
        model = build_meter_classification_adapter_v3_a1(base)
        images = torch.zeros((2, 1, 192, 256), dtype=torch.float32)
        logits, boxes = model(images)
        expected_logits, expected_boxes = base(images)
        self.assertTrue(torch.equal(logits, expected_logits))
        self.assertTrue(torch.equal(boxes, expected_boxes))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.base.parameters()))
        self.assertFalse(hasattr(model, "bbox_delta_head"))

    def test_class_head_changes_never_change_bbox(self) -> None:
        import torch

        base = self._fake_base()
        model = build_meter_classification_adapter_v3_a1(base)
        images = torch.zeros((1, 1, 192, 256), dtype=torch.float32)
        _base_logits, expected_boxes = base(images)
        with torch.no_grad():
            model.presence_head.bias.fill_(3.0)
            model.digit_head.bias.copy_(torch.tensor([2.0, 0.0, -2.0]))
        logits, boxes = model(images)
        self.assertEqual(int(logits.argmax(1).item()), 1)
        self.assertTrue(torch.equal(boxes, expected_boxes))

    def test_source_only_step_penalizes_nonzero_adapter_residual_and_keeps_base_frozen(self) -> None:
        import torch

        base = self._fake_base()
        model = build_meter_classification_adapter_v3_a1(base)
        with torch.no_grad():
            model.presence_head.bias.fill_(1.0)
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01
        )
        images = torch.zeros((4, 1, 192, 256), dtype=torch.float32)
        classes = torch.zeros((4,), dtype=torch.long)
        positive = torch.zeros((4,), dtype=torch.bool)
        before = base.weight.detach().clone()
        loss = train_batch_v3_a1(
            model,
            images,
            classes,
            positive,
            real_count=0,
            optimizer=optimizer,
        )
        self.assertGreater(loss, 0.0)
        self.assertTrue(torch.equal(base.weight.detach(), before))


if __name__ == "__main__":
    unittest.main()
