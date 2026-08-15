from __future__ import annotations

from dataclasses import dataclass
import unittest

from st_omr_training.stage7d13_verified_surface import (
    D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
    D13_DERIVATIVE_BUILD_ID,
    D13_DERIVATIVE_MANIFEST_SHA256,
    D13_EXPECTED_OPTIMIZER_STEPS,
    D13_EXPECTED_OPTIMIZER_STEPS_TOTAL,
    D13_IMAGE_COUNT,
    D13_LABEL_COUNT,
    D13_RECORD_COUNT,
    D13_RECORD_SPLIT_COUNTS,
    D13_TARGET_INSTANCE_COUNTS,
    Stage7D13VerifiedSurfaceError,
    assert_verified_surface,
)


@dataclass
class Receipt:
    derivative_build_id: str = D13_DERIVATIVE_BUILD_ID
    manifest_sha256: str = D13_DERIVATIVE_MANIFEST_SHA256
    artifact_binding_sha256: str = D13_DERIVATIVE_ARTIFACT_BINDING_SHA256
    record_count: int = D13_RECORD_COUNT
    image_count: int = D13_IMAGE_COUNT
    label_count: int = D13_LABEL_COUNT
    record_split_counts: dict[str, int] | None = None
    target_instance_counts: dict[str, dict[str, int]] | None = None
    test_specialist_records: int = 0
    optimizer_steps_executed: int = 0
    expected_optimizer_steps: dict[str, int] | None = None
    expected_optimizer_steps_total: int = D13_EXPECTED_OPTIMIZER_STEPS_TOTAL
    complete_marker_present: bool = False
    independent_verification_passed: bool = True

    def __post_init__(self) -> None:
        if self.record_split_counts is None:
            self.record_split_counts = dict(D13_RECORD_SPLIT_COUNTS)
        if self.target_instance_counts is None:
            self.target_instance_counts = {
                split: dict(values)
                for split, values in D13_TARGET_INSTANCE_COUNTS.items()
            }
        if self.expected_optimizer_steps is None:
            self.expected_optimizer_steps = dict(D13_EXPECTED_OPTIMIZER_STEPS)


class Stage7D13VerifiedSurfaceTests(unittest.TestCase):
    def test_exact_authoritative_receipt_passes(self) -> None:
        result = assert_verified_surface(Receipt())
        self.assertEqual(result.record_count, 11_064)
        self.assertEqual(result.image_count, 11_062)
        self.assertEqual(result.label_count, 11_064)
        self.assertEqual(result.record_split_counts, {"train": 9840, "validation": 1224})
        self.assertEqual(result.expected_optimizer_steps_total, 18_450)

    def test_hash_drift_fails_closed(self) -> None:
        receipt = Receipt()
        receipt.manifest_sha256 = "0" * 64
        with self.assertRaises(Stage7D13VerifiedSurfaceError):
            assert_verified_surface(receipt)

    def test_split_or_step_drift_fails_closed(self) -> None:
        receipt = Receipt()
        assert receipt.record_split_counts is not None
        receipt.record_split_counts["train"] = 9839
        with self.assertRaises(Stage7D13VerifiedSurfaceError):
            assert_verified_surface(receipt)

        receipt = Receipt()
        assert receipt.expected_optimizer_steps is not None
        receipt.expected_optimizer_steps["notehead"] = 6149
        with self.assertRaises(Stage7D13VerifiedSurfaceError):
            assert_verified_surface(receipt)

    def test_complete_or_failed_verification_is_rejected(self) -> None:
        with self.assertRaises(Stage7D13VerifiedSurfaceError):
            assert_verified_surface(Receipt(complete_marker_present=True))
        with self.assertRaises(Stage7D13VerifiedSurfaceError):
            assert_verified_surface(Receipt(independent_verification_passed=False))


if __name__ == "__main__":
    unittest.main()
