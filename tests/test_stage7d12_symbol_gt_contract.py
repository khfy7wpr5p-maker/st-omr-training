from __future__ import annotations

from collections.abc import Mapping

import pytest

from st_omr_training.stage7d12_symbol_gt_contract import (
    ACCIDENTAL_CLASSES,
    D12_ACCEPTANCE_GATES,
    EXPECTED_DEVELOPMENT_FAMILY_COUNTS,
    EXPECTED_DEVELOPMENT_SAMPLE_COUNTS,
    FUTURE_TRAINING_BOUNDARY,
    NOTEHEAD_FILL_CLASSES,
    OPTIMIZER_STEPS,
    REST_CLASSES,
    SYMBOL_TARGETS,
    TEST_SPECIALIST_RECORDS,
    Stage7D12ContractError,
    accidental_class,
    canonical_event_id,
    development_split,
    notehead_fill_class,
    require_link_cardinality,
    rest_class,
    stage7d12_contract_fingerprint,
    validate_canonical_event_id,
)


class _HostileTestRow(Mapping[str, object]):
    """TEST row that fails if D12 touches anything except the split field."""

    def __getitem__(self, key: str) -> object:
        if key == "split":
            return "test"
        raise AssertionError(f"D12 touched sealed TEST field: {key}")

    def __iter__(self):
        yield "split"

    def __len__(self) -> int:
        return 1

    def get(self, key: str, default=None):
        if key == "split":
            return "test"
        raise AssertionError(f"D12 touched sealed TEST field: {key}")


def test_target_surface_matches_frozen_d4_specialists() -> None:
    assert [target.set_name for target in SYMBOL_TARGETS] == [
        "NoteHeadSet",
        "RestSet",
        "AccidentalSet",
    ]
    assert [target.task_id for target in SYMBOL_TARGETS] == [
        "notehead",
        "rest",
        "accidental",
    ]
    assert NOTEHEAD_FILL_CLASSES == ("open", "filled")
    assert REST_CLASSES == ("half", "quarter", "eighth")
    assert ACCIDENTAL_CLASSES == ("sharp", "flat", "natural")


def test_d12_is_data_only_and_test_sealed() -> None:
    assert TEST_SPECIALIST_RECORDS == 0
    assert OPTIMIZER_STEPS == 0
    assert D12_ACCEPTANCE_GATES["test_specialist_records"] == 0
    assert D12_ACCEPTANCE_GATES["optimizer_steps"] == 0
    assert development_split(_HostileTestRow()) is None


def test_development_surface_is_frozen_to_accepted_d6_counts() -> None:
    assert EXPECTED_DEVELOPMENT_SAMPLE_COUNTS == {
        "train": 1230,
        "validation": 153,
    }
    assert EXPECTED_DEVELOPMENT_FAMILY_COUNTS == {
        "train": 410,
        "validation": 51,
    }
    assert development_split({"split": "train", "path": "ignored"}) == "train"
    assert development_split({"split": "validation"}) == "validation"
    with pytest.raises(Stage7D12ContractError):
        development_split({"split": "dev"})


def test_future_training_keeps_three_separate_specialists() -> None:
    assert FUTURE_TRAINING_BOUNDARY["shared_derivative_bundle"] is True
    assert FUTURE_TRAINING_BOUNDARY["separate_specialist_models"] is True
    assert FUTURE_TRAINING_BOUNDARY["joint_optimizer"] is False


def test_canonical_event_ids_are_stable_and_bounded() -> None:
    assert canonical_event_id(measure_number=1, event_index=0) == "m1-e0"
    assert (
        canonical_event_id(measure_number=12, event_index=3, chord_member_index=2)
        == "m12-e3-n2"
    )
    assert validate_canonical_event_id("m12-e3-n2") == "m12-e3-n2"
    for value in ("m0-e0", "m1-e-1", "measure1-event1", ""):
        with pytest.raises(Stage7D12ContractError):
            validate_canonical_event_id(value)


def test_symbol_class_mappings_stay_inside_v1() -> None:
    assert notehead_fill_class("whole") == "open"
    assert notehead_fill_class("half") == "open"
    assert notehead_fill_class("quarter") == "filled"
    assert notehead_fill_class("eighth") == "filled"
    assert rest_class("quarter") == "quarter"
    assert accidental_class("natural") == "natural"

    with pytest.raises(Stage7D12ContractError):
        notehead_fill_class("sixteenth")
    with pytest.raises(Stage7D12ContractError):
        rest_class("whole")
    with pytest.raises(Stage7D12ContractError):
        accidental_class("double-sharp")


def test_renderer_and_canonical_cardinality_must_match_exactly() -> None:
    require_link_cardinality(kind="notehead", canonical_count=4, renderer_count=4)
    require_link_cardinality(kind="rest", canonical_count=0, renderer_count=0)
    with pytest.raises(Stage7D12ContractError):
        require_link_cardinality(
            kind="accidental", canonical_count=2, renderer_count=1
        )
    with pytest.raises(Stage7D12ContractError):
        require_link_cardinality(kind="unknown", canonical_count=1, renderer_count=1)


def test_contract_fingerprint_is_deterministic_sha256() -> None:
    first = stage7d12_contract_fingerprint()
    second = stage7d12_contract_fingerprint()
    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")
