from pathlib import Path

import pytest

from st_omr_training import meter_v5_2b_specialist_adaptation as v52b
from st_omr_training import meter_v5_3j_rescue_failure_forensics_v1 as v53j


def test_v5_3j_contract_is_read_only_and_keeps_protected_surfaces_closed():
    boundary = v53j.safety_boundary()
    assert boundary["training"] is False
    assert boundary["backward"] is False
    assert boundary["optimizer_steps"] == 0
    assert boundary["checkpoint_write"] is False
    assert boundary["rescue_artifact_write"] is False
    assert boundary["threshold_tuning"] is False
    assert boundary["threshold_sweep"] is False
    assert boundary["automatic_second_configuration"] is False
    assert boundary["retraining_authorized"] is False
    assert boundary["historical_validation_opened"] is False
    assert boundary["first30_opened"] is False
    assert boundary["v5_reserve_opened"] is False
    assert boundary["v5_validation_opened"] is False
    assert boundary["final_holdout_locked"] is True
    assert boundary["digit4_loaded"] is False
    assert v53j.retraining_allowed_after_forensics() is False
    assert v53j.threshold_tuning_allowed() is False
    assert v53j.historical_validation_access_allowed() is False
    assert v53j.first30_access_allowed() is False
    assert v53j.v5_validation_access_allowed() is False
    assert v53j.final_holdout_access_allowed() is False


def test_v5_3j_is_bound_to_exact_hold_receipt_and_witness():
    contract = v53j.forensic_contract()
    assert contract["prerequisite_v5_3i_head"] == "88c7acc551fa2b00b1f877f6a839704d58825adb"
    assert contract["prerequisite_v5_3i_module_blob"] == "abb5f1ae4c42b0c5f3ae26b80f2a467f47582197"
    assert contract["bound_v5_3i_report_sha256"] == "448b807086bc9ee66d090fdf173ce54e3c5e2a133e60cf6ae0a791aed2717434"
    assert contract["required_v5_3i_decision"] == "HOLD"
    assert contract["bound_hold_reasons"] == list(v53j.EXPECTED_HOLD_REASONS)
    assert v53j.EXPECTED_ACCEPTANCE_WITNESS["2"]["historical_regressions"] == 5307
    assert v53j.EXPECTED_ACCEPTANCE_WITNESS["3"]["v5_fn"] == 90
    assert v53j.EXPECTED_ACCEPTANCE_WITNESS["3"]["historical_regressions"] == 15775


def test_probability_distribution_and_rank_fraction_are_descriptive_only():
    torch, _nn = v52b._import_torch()
    values = torch.tensor([0.1, 0.2, 0.8, 0.9], dtype=torch.float32)
    dist = v53j._probability_distribution(values, label="unit")
    assert dist["count"] == 4
    assert dist["min"] == pytest.approx(0.1)
    assert dist["max"] == pytest.approx(0.9)
    assert dist["mean"] == pytest.approx(0.5)

    pos = torch.tensor([0.8, 0.9], dtype=torch.float32)
    neg = torch.tensor([0.1, 0.2], dtype=torch.float32)
    assert v53j._pairwise_rank_fraction(pos, neg, label="perfect") == pytest.approx(1.0)
    assert v53j._pairwise_rank_fraction(neg, pos, label="reverse") == pytest.approx(0.0)
    ties = torch.tensor([0.5], dtype=torch.float32)
    assert v53j._pairwise_rank_fraction(ties, ties, label="tie") == pytest.approx(0.5)


def test_score_group_diagnostics_reports_corrections_and_regressions_at_fixed_threshold():
    torch, _nn = v52b._import_torch()
    frozen = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float32)
    rescue = torch.tensor([0.9, 0.8, 0.9, 0.1], dtype=torch.float32)
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    evidence = v53j._score_group_diagnostics(
        frozen_probability=frozen,
        rescue_probability=rescue,
        targets=targets,
        frozen_threshold=0.48,
        rescue_threshold=0.50,
        label="unit",
    )
    assert evidence["eligible_positive_count"] == 2
    assert evidence["eligible_negative_count"] == 2
    assert evidence["eligible_positive_rescue_above_threshold"] == 2
    assert evidence["eligible_positive_rescue_below_threshold"] == 0
    assert evidence["eligible_negative_rescue_above_threshold"] == 1
    assert evidence["eligible_negative_rescue_below_threshold"] == 1
    assert evidence["positive_over_negative_rank_fraction"] == pytest.approx(0.625)


def test_failure_signature_does_not_select_repair_recipe():
    torch, _nn = v52b._import_torch()
    signature = v53j._failure_signature(
        digit="2",
        v5={"eligible_positive_count": 2, "eligible_positive_rescue_above_threshold": 2},
        historical={"eligible_negative_rescue_above_threshold": 1},
        v5_positive_scores=torch.tensor([0.8, 0.9]),
        historical_negative_scores=torch.tensor([0.1, 0.7]),
    )
    assert signature["signature"] == "V5_RECOVERED_HISTORICAL_TN_COLLAPSE"
    assert signature["v5_positive_recovery_fraction"] == 1.0
    assert signature["historical_true_negative_regression_count"] == 1
    assert signature["fixed_threshold_separates_required_groups"] is False
    assert "no threshold" in signature["interpretation_scope"]


def test_v5_3j_source_contains_no_training_or_protected_gate_entry_points():
    source = Path(v53j.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "run_authoritative_rescue_training_v1(",
        "execute_rescue_tensor_harness_v1(",
        "torch.optim.",
        ".backward(",
        "optimizer.step(",
        "run_historical_retention_gate(",
    ):
        assert forbidden not in source
    assert '"threshold_tuning": False' in source
    assert '"historical_validation_opened": False' in source
    assert '"final_holdout_locked": True' in source
