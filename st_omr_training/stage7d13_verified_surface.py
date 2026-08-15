"""Frozen authoritative Stage 7-D13 measure-derivative identity.

This module is intentionally downstream of the D13-1/D13-2 derivative profile.
It does not participate in derivative construction.  Later training code imports
these constants so model creation and optimizer authorization are bound to the
externally produced, independently verified development-only bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping


D13_DERIVATIVE_EXECUTABLE_SHA: Final[str] = (
    "d5fe4d2c120202ec7f962ef6d849b6e36af224ef"
)
D13_DERIVATIVE_BUILD_ID: Final[str] = (
    "44f1932532fb511dfa59a164f94be6b899f3aa0594c0ac0a6f499a38e5fb5697"
)
D13_DERIVATIVE_MANIFEST_SHA256: Final[str] = (
    "8cfb87b5c6135be14b4c9ad488868c0edb0d37bb3bb18ad1b5e79d04fdf24f7b"
)
D13_DERIVATIVE_ARTIFACT_BINDING_SHA256: Final[str] = (
    "c42c1f69e21d61d3eefdacfc40dabf2f0fcd6ac2ceb4d5cf88d8e158246dd33e"
)
D13_EXTERNAL_RECEIPT_SHA256: Final[str] = (
    "4e644c5a110c738fd99b905f093a9acb0ca07cd6bd1b7b52c4904aba7964466b"
)

D13_RECORD_SPLIT_COUNTS: Final[dict[str, int]] = {
    "train": 9_840,
    "validation": 1_224,
}
D13_RECORD_COUNT: Final[int] = 11_064
D13_IMAGE_COUNT: Final[int] = 11_062
D13_LABEL_COUNT: Final[int] = 11_064
D13_TEST_SPECIALIST_RECORDS: Final[int] = 0
D13_OPTIMIZER_STEPS_EXECUTED_DERIVATIVE_STAGE: Final[int] = 0

D13_TARGET_INSTANCE_COUNTS: Final[dict[str, dict[str, int]]] = {
    "train": {
        "notehead": 38_334,
        "rest": 10_602,
        "accidental": 22_392,
    },
    "validation": {
        "notehead": 5_232,
        "rest": 969,
        "accidental": 3_330,
    },
}

D13_EXPECTED_OPTIMIZER_STEPS: Final[dict[str, int]] = {
    "notehead": 6_150,
    "rest": 6_150,
    "accidental": 6_150,
}
D13_EXPECTED_OPTIMIZER_STEPS_TOTAL: Final[int] = 18_450


class Stage7D13VerifiedSurfaceError(ValueError):
    """Raised when a purported authoritative D13 derivative receipt drifts."""


@dataclass(frozen=True, slots=True)
class VerifiedSurfaceSummary:
    derivative_build_id: str
    manifest_sha256: str
    artifact_binding_sha256: str
    record_count: int
    image_count: int
    label_count: int
    record_split_counts: dict[str, int]
    target_instance_counts: dict[str, dict[str, int]]
    test_specialist_records: int
    optimizer_steps_executed: int
    expected_optimizer_steps: dict[str, int]
    expected_optimizer_steps_total: int
    complete_marker_present: bool
    independent_verification_passed: bool


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Stage7D13VerifiedSurfaceError(f"{name} must be a mapping")
    return value


def assert_verified_surface(receipt: object) -> VerifiedSurfaceSummary:
    """Fail closed unless an authoritative receipt matches the frozen D13 surface."""

    values = {
        "derivative_build_id": D13_DERIVATIVE_BUILD_ID,
        "manifest_sha256": D13_DERIVATIVE_MANIFEST_SHA256,
        "artifact_binding_sha256": D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
        "record_count": D13_RECORD_COUNT,
        "image_count": D13_IMAGE_COUNT,
        "label_count": D13_LABEL_COUNT,
        "test_specialist_records": D13_TEST_SPECIALIST_RECORDS,
        "optimizer_steps_executed": D13_OPTIMIZER_STEPS_EXECUTED_DERIVATIVE_STAGE,
        "expected_optimizer_steps_total": D13_EXPECTED_OPTIMIZER_STEPS_TOTAL,
        "complete_marker_present": False,
        "independent_verification_passed": True,
    }
    for name, expected in values.items():
        if getattr(receipt, name, object()) != expected:
            raise Stage7D13VerifiedSurfaceError(
                f"authoritative D13 receipt {name} mismatch"
            )

    split_counts = dict(_mapping(getattr(receipt, "record_split_counts", None), "record_split_counts"))
    if split_counts != D13_RECORD_SPLIT_COUNTS:
        raise Stage7D13VerifiedSurfaceError("authoritative D13 split counts mismatch")

    target_counts_raw = _mapping(
        getattr(receipt, "target_instance_counts", None),
        "target_instance_counts",
    )
    target_counts = {
        split: dict(_mapping(target_counts_raw.get(split), f"target_instance_counts.{split}"))
        for split in ("train", "validation")
    }
    if target_counts != D13_TARGET_INSTANCE_COUNTS:
        raise Stage7D13VerifiedSurfaceError("authoritative D13 target inventory mismatch")

    step_counts = dict(
        _mapping(getattr(receipt, "expected_optimizer_steps", None), "expected_optimizer_steps")
    )
    if step_counts != D13_EXPECTED_OPTIMIZER_STEPS:
        raise Stage7D13VerifiedSurfaceError("authoritative D13 optimizer-step expectation mismatch")

    return VerifiedSurfaceSummary(
        derivative_build_id=D13_DERIVATIVE_BUILD_ID,
        manifest_sha256=D13_DERIVATIVE_MANIFEST_SHA256,
        artifact_binding_sha256=D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
        record_count=D13_RECORD_COUNT,
        image_count=D13_IMAGE_COUNT,
        label_count=D13_LABEL_COUNT,
        record_split_counts=dict(D13_RECORD_SPLIT_COUNTS),
        target_instance_counts={
            split: dict(values) for split, values in D13_TARGET_INSTANCE_COUNTS.items()
        },
        test_specialist_records=D13_TEST_SPECIALIST_RECORDS,
        optimizer_steps_executed=D13_OPTIMIZER_STEPS_EXECUTED_DERIVATIVE_STAGE,
        expected_optimizer_steps=dict(D13_EXPECTED_OPTIMIZER_STEPS),
        expected_optimizer_steps_total=D13_EXPECTED_OPTIMIZER_STEPS_TOTAL,
        complete_marker_present=False,
        independent_verification_passed=True,
    )
