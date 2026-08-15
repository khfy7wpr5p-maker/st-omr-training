from __future__ import annotations

import unittest
from unittest.mock import patch

from st_omr_training.stage7d13_derivative_authoritative import (
    Stage7D13DerivativeAuthoritativeError,
    derive_optimizer_steps,
    run_verified_stage7d13_derivative_bundle,
)
from st_omr_training.stage7d13_measure_derivative_verifier import (
    STAGE7D13_VERIFIER_VERSION,
    Stage7D13VerificationReceipt,
)
from st_omr_training.stage7d13_measure_derivatives import Stage7D13DerivativeReceipt


HEAD = "a" * 40
ORIGIN = "https://github.com/khfy7wpr5p-maker/st-omr-training.git"
H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
INVENTORY = {
    "train": {
        "notehead": {"open": 7935, "filled": 30399},
        "rest": {"half": 1998, "quarter": 3417, "eighth": 5187},
        "accidental": {"sharp": 10665, "flat": 10596, "natural": 1131},
    },
    "validation": {
        "notehead": {"open": 1128, "filled": 4104},
        "rest": {"half": 162, "quarter": 327, "eighth": 480},
        "accidental": {"sharp": 1566, "flat": 1575, "natural": 189},
    },
}
TARGETS = {
    "train": {"notehead": 38334, "rest": 10602, "accidental": 22392},
    "validation": {"notehead": 5232, "rest": 969, "accidental": 3330},
}


def _build() -> Stage7D13DerivativeReceipt:
    return Stage7D13DerivativeReceipt(
        derivative_build_id=H1,
        manifest_sha256=H2,
        artifact_binding_sha256=H3,
        record_count=5000,
        image_count=5000,
        label_count=5000,
        source_sample_count=1383,
        family_count=461,
        record_split_counts={"train": 4500, "validation": 500},
        source_sample_split_counts={"train": 1230, "validation": 153},
        family_split_counts={"train": 410, "validation": 51},
        observed_class_inventory=INVENTORY,
        target_instance_counts=TARGETS,
        image_bytes_total=123,
        label_bytes_total=456,
        test_specialist_records=0,
        optimizer_steps=0,
        complete_marker_written=False,
    )


def _verified(*, manifest_sha: str = H2) -> Stage7D13VerificationReceipt:
    return Stage7D13VerificationReceipt(
        verifier_version=STAGE7D13_VERIFIER_VERSION,
        derivative_build_id=H1,
        manifest_sha256=manifest_sha,
        artifact_binding_sha256=H3,
        record_count=5000,
        image_count=5000,
        label_count=5000,
        source_sample_count=1383,
        family_count=461,
        record_split_counts={"train": 4500, "validation": 500},
        source_sample_split_counts={"train": 1230, "validation": 153},
        family_split_counts={"train": 410, "validation": 51},
        observed_class_inventory=INVENTORY,
        target_instance_counts=TARGETS,
        image_bytes_total=123,
        label_bytes_total=456,
        test_specialist_records=0,
        optimizer_steps=0,
        complete_marker_present=False,
        verification_passed=True,
    )


class Stage7D13DerivativeAuthoritativeTests(unittest.TestCase):
    def test_optimizer_steps_are_derived_from_verified_train_records(self) -> None:
        # ceil(4500 / 16) = 282 batches/epoch; x10 epochs = 2820.
        self.assertEqual(
            derive_optimizer_steps(4500),
            {"notehead": 2820, "rest": 2820, "accidental": 2820},
        )
        with self.assertRaises(ValueError):
            derive_optimizer_steps(0)

    @patch("st_omr_training.stage7d13_derivative_authoritative.verify_stage7c_runtime")
    @patch("st_omr_training.stage7d13_derivative_authoritative.verify_stage7d13_measure_derivatives")
    @patch("st_omr_training.stage7d13_derivative_authoritative.build_stage7d13_measure_derivatives")
    @patch("st_omr_training.stage7d13_derivative_authoritative.verify_authoritative_repository")
    def test_exact_head_build_verify_then_freeze_steps(
        self,
        repository_identity,
        builder,
        verifier,
        runtime,
    ) -> None:
        repository_identity.return_value = (HEAD, ORIGIN)
        builder.return_value = _build()
        verifier.return_value = _verified()
        result = run_verified_stage7d13_derivative_bundle(
            corpus_root="/tmp/corpus",
            d6_root="/tmp/d6",
            d12_root="/tmp/d12",
            output_root="/tmp/d13",
            repository_root="/tmp/repo",
            expected_repository_sha=HEAD,
        )
        self.assertEqual(result.repository_sha, HEAD)
        self.assertEqual(result.record_split_counts["train"], 4500)
        self.assertEqual(result.expected_optimizer_steps["notehead"], 2820)
        self.assertEqual(result.expected_optimizer_steps_total, 8460)
        self.assertEqual(result.optimizer_steps_executed, 0)
        self.assertIs(result.independent_verification_passed, True)
        self.assertEqual(repository_identity.call_count, 3)
        self.assertEqual(runtime.call_count, 3)

    @patch("st_omr_training.stage7d13_derivative_authoritative.verify_authoritative_repository")
    def test_wrong_head_fails_before_build(self, repository_identity) -> None:
        repository_identity.return_value = ("b" * 40, ORIGIN)
        with self.assertRaises(Stage7D13DerivativeAuthoritativeError):
            run_verified_stage7d13_derivative_bundle(
                corpus_root="/tmp/corpus",
                d6_root="/tmp/d6",
                d12_root="/tmp/d12",
                output_root="/tmp/d13",
                repository_root="/tmp/repo",
                expected_repository_sha=HEAD,
            )

    @patch("st_omr_training.stage7d13_derivative_authoritative.verify_stage7c_runtime")
    @patch("st_omr_training.stage7d13_derivative_authoritative.verify_stage7d13_measure_derivatives")
    @patch("st_omr_training.stage7d13_derivative_authoritative.build_stage7d13_measure_derivatives")
    @patch("st_omr_training.stage7d13_derivative_authoritative.verify_authoritative_repository")
    def test_builder_verifier_mismatch_fails_closed(
        self,
        repository_identity,
        builder,
        verifier,
        runtime,
    ) -> None:
        repository_identity.return_value = (HEAD, ORIGIN)
        builder.return_value = _build()
        verifier.return_value = _verified(manifest_sha="f" * 64)
        with self.assertRaises(Stage7D13DerivativeAuthoritativeError):
            run_verified_stage7d13_derivative_bundle(
                corpus_root="/tmp/corpus",
                d6_root="/tmp/d6",
                d12_root="/tmp/d12",
                output_root="/tmp/d13",
                repository_root="/tmp/repo",
                expected_repository_sha=HEAD,
            )


if __name__ == "__main__":
    unittest.main()
