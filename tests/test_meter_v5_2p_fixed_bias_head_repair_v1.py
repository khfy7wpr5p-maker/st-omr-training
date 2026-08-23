from __future__ import annotations

import math
from pathlib import Path

import pytest

from st_omr_training import meter_v5_2m_retention_contract_v3 as ret_v3
from st_omr_training import meter_v5_2p_fixed_bias_head_repair_v1 as repair


def test_contract_is_narrow_and_fail_closed() -> None:
    safety = repair.safety_boundary()
    assert safety["single_fixed_repair_authorized"] is True
    assert safety["automatic_second_configuration"] is False
    assert safety["trainable_surface"] == "head.weight-only-64-parameters"
    assert safety["frozen_backbone"] is True
    assert safety["frozen_head_bias"] is True
    assert safety["runtime_threshold_tuning"] is False
    assert safety["alternative_threshold_evaluated"] is False
    assert safety["new_bbox"] is False
    assert safety["new_crop_geometry"] is False
    assert safety["new_spatial_heuristic"] is False
    assert safety["reserve_v5_train_opened"] is False
    assert safety["v5_validation_opened"] is False
    assert safety["final_holdout_locked"] is True
    assert safety["digit4_frozen"] is True
    assert safety["runtime_domain_routing"] is False
    assert safety["production_promotion"] is False
    assert repair.gate_order() == ("historical_retention_v3", "v5_first30_diagnostic")
    assert repair.production_promotion_allowed() is False
    assert repair.validation_opened_by_this_module() is False
    assert repair.final_holdout_locked() is True


def test_objective_and_solver_are_single_fixed_configuration() -> None:
    objective = repair.objective_contract()
    assert objective == {
        "formula": "0.5*mean(V5_BCE_w1)+0.5*mean(HISTORICAL_BCE_w1)",
        "v5_domain_weight": 0.5,
        "historical_domain_weight": 0.5,
        "positive_weight": 1.0,
        "class_reweighting": False,
        "replay_ratio": None,
        "full_batch": True,
        "head_bias_trainable": False,
        "backbone_trainable": False,
    }
    solver = repair.solver_contract()
    assert solver["optimizer"] == "LBFGS"
    assert solver["lr"] == 1.0
    assert solver["max_iter"] == 100
    assert solver["max_eval"] == 125
    assert solver["history_size"] == 20
    assert solver["line_search_fn"] == "strong_wolfe"
    assert solver["initialization"] == "exact-frozen-head-weight"
    assert solver["checkpoint_selection"] == "single-final-solver-state-no-sweep"
    assert solver["weight_decay"] == 0.0
    assert solver["momentum"] == 0.0


def test_equal_domain_objective_is_invariant_to_within_domain_duplication() -> None:
    torch = pytest.importorskip("torch")
    v5_logits = torch.tensor([-2.0, 1.5], dtype=torch.float64)
    v5_targets = torch.tensor([0.0, 1.0], dtype=torch.float64)
    hist_logits = torch.tensor([-3.0, -1.0, 2.0], dtype=torch.float64)
    hist_targets = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

    total1, v51, hist1 = repair._balanced_domain_bce_v1(
        v5_logits=v5_logits,
        v5_targets=v5_targets,
        historical_logits=hist_logits,
        historical_targets=hist_targets,
    )
    total2, v52, hist2 = repair._balanced_domain_bce_v1(
        v5_logits=v5_logits.repeat(7),
        v5_targets=v5_targets.repeat(7),
        historical_logits=hist_logits.repeat(11),
        historical_targets=hist_targets.repeat(11),
    )
    assert math.isclose(float(v51), float(v52), rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(hist1), float(hist2), rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(total1), float(total2), rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(total1), 0.5 * float(v51) + 0.5 * float(hist1), abs_tol=1e-12)


def test_state_guard_allows_only_head_weight_change() -> None:
    torch = pytest.importorskip("torch")
    model = repair.v52b._build_digit_model().cpu()
    frozen = repair._frozen_state_snapshot(model)
    with torch.no_grad():
        model.head.weight.add_(0.01)
    result = repair._verify_only_head_weight_changed(model, frozen)
    assert result["changed_state_keys"] == ["head.weight"]
    assert result["only_head_weight_changed"] is True
    assert result["backbone_bit_identical"] is True
    assert result["head_bias_bit_identical"] is True


def test_state_guard_rejects_bias_or_backbone_mutation() -> None:
    torch = pytest.importorskip("torch")
    model = repair.v52b._build_digit_model().cpu()
    frozen = repair._frozen_state_snapshot(model)
    with torch.no_grad():
        model.head.bias.add_(0.01)
    with pytest.raises(repair.MeterV5_2PError, match="frozen tensor changed"):
        repair._verify_only_head_weight_changed(model, frozen)

    model = repair.v52b._build_digit_model().cpu()
    frozen = repair._frozen_state_snapshot(model)
    with torch.no_grad():
        next(model.features.parameters()).add_(0.01)
    with pytest.raises(repair.MeterV5_2PError, match="frozen tensor changed"):
        repair._verify_only_head_weight_changed(model, frozen)


def test_corrected_retention_contract_is_relative_only() -> None:
    frozen = {
        "2": {"f1": 0.92, "recall": 0.99, "precision": 0.86},
        "3": {"f1": 0.995, "recall": 0.995, "precision": 0.995},
    }
    candidate = {
        "2": {"f1": 0.919, "recall": 0.989, "precision": 0.859},
        "3": {"f1": 0.994, "recall": 0.994, "precision": 0.994},
    }
    result = ret_v3.evaluate_retention_gate_v3(
        frozen_metrics=frozen,
        candidate_metrics=candidate,
    )
    assert result["gate"] == "PASS"
    assert result["absolute_precision_floor_used"] is False
    assert result["absolute_recall_floor_used"] is False


def test_approval_token_is_explicit() -> None:
    assert repair.APPROVAL_TOKEN == "V5_2P_FIXED_BIAS_HEAD_REPAIR_APPROVED"
    assert repair.EXPECTED_V5_COUNT == 540
    assert repair.EXPECTED_HISTORICAL_COUNT == 26_964
    assert repair.EXPECTED_FEATURE_DIM == 64


def test_contract_document_exists() -> None:
    path = Path("METER_V5_2P_FIXED_BIAS_HEAD_REPAIR_V1.md")
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "freeze the complete convolutional feature extractor" in text
    assert "freeze the existing scalar head bias" in text
    assert "0.5 * mean(BCE_w1(V5_adaptation_train))" in text
    assert "V5 VALIDATION opening" in text
    assert "There is no automatic second configuration" in text
