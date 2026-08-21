from __future__ import annotations

from dataclasses import asdict
import unittest

import torch

from st_omr_training.meter_real_domain_adaptation_v3_a1 import (
    FROZEN_ADAPTATION_CONFIG_V3_A1,
)
from st_omr_training.meter_real_domain_adaptation_v3_a2 import (
    FROZEN_ADAPTATION_CONFIG_V3_A2,
    METER_REAL_DOMAIN_ADAPTATION_V3_A2,
    MeterRealDomainAdaptationConfigV3A2,
    production_promotion_allowed,
    real_positive_pairwise_margin_loss_v3_a2,
    resolver_connection_allowed,
    runtime_connection_allowed,
    sealed_test_access_allowed,
)
from st_omr_training.meter_real_domain_adaptation_v3_a2_run import (
    CHECKPOINT_ROLE_V3_A2,
    METRICS_SCHEMA_V3_A2,
    RESUME_ROLE_V3_A2,
    VERIFICATION_SCHEMA_V3_A2,
)


class MeterRealDomainAdaptationV3A2Tests(unittest.TestCase):
    def test_v3_a2_changes_only_positive_margin_objective_from_v3_a1(self) -> None:
        a1 = asdict(FROZEN_ADAPTATION_CONFIG_V3_A1)
        a2 = asdict(FROZEN_ADAPTATION_CONFIG_V3_A2)
        self.assertEqual(a2.pop("positive_margin_loss_milli"), 1_000)
        self.assertEqual(a2.pop("positive_margin_milli"), 2_000)
        self.assertEqual(a2.pop("objective"), "v3-a1-plus-real-positive-pairwise-margin-v3-a2")
        self.assertEqual(
            a1.pop("objective"),
            "real-classification-plus-d10-logit-distillation-and-residual-zero-v3-a1",
        )
        self.assertEqual(a2, a1)

    def test_pairwise_margin_uses_only_real_positive_classes(self) -> None:
        logits = torch.tensor(
            [
                [50.0, 5.0, 1.0, 0.0],
                [50.0, 0.0, 1.0, 3.0],
                [50.0, 100.0, 100.0, 100.0],
            ],
            dtype=torch.float32,
        )
        classes = torch.tensor([1, 2, 0], dtype=torch.long)
        positive = torch.tensor([True, True, False], dtype=torch.bool)
        loss = real_positive_pairwise_margin_loss_v3_a2(
            logits,
            classes,
            positive,
            margin=2.0,
        )
        self.assertAlmostEqual(float(loss.item()), 2.0, places=7)

    def test_pairwise_margin_is_zero_when_true_positive_classes_clear_margin(self) -> None:
        logits = torch.tensor(
            [
                [99.0, 6.0, 1.0, 0.0],
                [99.0, 0.0, 7.0, 1.0],
                [99.0, 0.0, 1.0, 8.0],
            ],
            dtype=torch.float32,
        )
        classes = torch.tensor([1, 2, 3], dtype=torch.long)
        positive = torch.tensor([True, True, True], dtype=torch.bool)
        loss = real_positive_pairwise_margin_loss_v3_a2(
            logits,
            classes,
            positive,
            margin=2.0,
        )
        self.assertAlmostEqual(float(loss.item()), 0.0, places=7)

    def test_pairwise_margin_rejects_malformed_positive_target(self) -> None:
        logits = torch.zeros((1, 4), dtype=torch.float32)
        classes = torch.tensor([0], dtype=torch.long)
        positive = torch.tensor([True], dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "positive records"):
            real_positive_pairwise_margin_loss_v3_a2(
                logits,
                classes,
                positive,
                margin=2.0,
            )

    def test_v3_a2_contract_is_shadow_only_and_separately_versioned(self) -> None:
        self.assertEqual(
            METER_REAL_DOMAIN_ADAPTATION_V3_A2,
            "meter-real-domain-adaptation-v3-a2-positive-margin",
        )
        self.assertTrue(METRICS_SCHEMA_V3_A2.endswith("v3-a2"))
        self.assertTrue(VERIFICATION_SCHEMA_V3_A2.endswith("v3-a2"))
        self.assertTrue(CHECKPOINT_ROLE_V3_A2.endswith("v3-a2"))
        self.assertTrue(RESUME_ROLE_V3_A2.endswith("v3-a2"))
        self.assertFalse(sealed_test_access_allowed())
        self.assertFalse(runtime_connection_allowed())
        self.assertFalse(resolver_connection_allowed())
        self.assertFalse(production_promotion_allowed())

    def test_v3_a2_margin_is_frozen_and_bounded(self) -> None:
        self.assertEqual(FROZEN_ADAPTATION_CONFIG_V3_A2.positive_margin_loss_milli, 1_000)
        self.assertEqual(FROZEN_ADAPTATION_CONFIG_V3_A2.positive_margin_milli, 2_000)
        with self.assertRaises(ValueError):
            MeterRealDomainAdaptationConfigV3A2(positive_margin_milli=0)
        with self.assertRaises(ValueError):
            MeterRealDomainAdaptationConfigV3A2(positive_margin_loss_milli=0)


if __name__ == "__main__":
    unittest.main()
