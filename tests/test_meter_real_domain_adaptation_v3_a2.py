from __future__ import annotations

from dataclasses import asdict

import pytest
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


def test_v3_a2_changes_only_positive_margin_objective_from_v3_a1() -> None:
    a1 = asdict(FROZEN_ADAPTATION_CONFIG_V3_A1)
    a2 = asdict(FROZEN_ADAPTATION_CONFIG_V3_A2)
    assert a2.pop("positive_margin_loss_milli") == 1_000
    assert a2.pop("positive_margin_milli") == 2_000
    assert a2.pop("objective") == "v3-a1-plus-real-positive-pairwise-margin-v3-a2"
    assert a1.pop("objective") == "real-classification-plus-d10-logit-distillation-and-residual-zero-v3-a1"
    assert a2 == a1


def test_pairwise_margin_uses_only_real_positive_classes() -> None:
    logits = torch.tensor(
        [
            [50.0, 5.0, 1.0, 0.0],  # true 2/4 already exceeds alternatives by >2
            [50.0, 0.0, 1.0, 3.0],  # true 3/4 trails 4/4 by 2 -> penalty 4
            [50.0, 100.0, 100.0, 100.0],  # none row must be ignored by this term
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
    assert float(loss.item()) == pytest.approx(2.0)


def test_pairwise_margin_is_zero_when_true_positive_classes_clear_margin() -> None:
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
    assert float(loss.item()) == pytest.approx(0.0)


def test_pairwise_margin_rejects_malformed_positive_target() -> None:
    logits = torch.zeros((1, 4), dtype=torch.float32)
    classes = torch.tensor([0], dtype=torch.long)
    positive = torch.tensor([True], dtype=torch.bool)
    with pytest.raises(ValueError, match="positive records"):
        real_positive_pairwise_margin_loss_v3_a2(
            logits,
            classes,
            positive,
            margin=2.0,
        )


def test_v3_a2_contract_is_shadow_only_and_separately_versioned() -> None:
    assert METER_REAL_DOMAIN_ADAPTATION_V3_A2 == "meter-real-domain-adaptation-v3-a2-positive-margin"
    assert METRICS_SCHEMA_V3_A2.endswith("v3-a2")
    assert VERIFICATION_SCHEMA_V3_A2.endswith("v3-a2")
    assert CHECKPOINT_ROLE_V3_A2.endswith("v3-a2")
    assert RESUME_ROLE_V3_A2.endswith("v3-a2")
    assert sealed_test_access_allowed() is False
    assert runtime_connection_allowed() is False
    assert resolver_connection_allowed() is False
    assert production_promotion_allowed() is False


def test_v3_a2_margin_is_frozen_and_bounded() -> None:
    assert FROZEN_ADAPTATION_CONFIG_V3_A2.positive_margin_loss_milli == 1_000
    assert FROZEN_ADAPTATION_CONFIG_V3_A2.positive_margin_milli == 2_000
    with pytest.raises(ValueError):
        MeterRealDomainAdaptationConfigV3A2(positive_margin_milli=0)
    with pytest.raises(ValueError):
        MeterRealDomainAdaptationConfigV3A2(positive_margin_loss_milli=0)
