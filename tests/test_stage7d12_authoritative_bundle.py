from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from st_omr_training.stage7d12_authoritative_bundle import (
    Stage7D12AuthoritativeError,
    run_verified_stage7d12_authoritative_bundle,
)
from st_omr_training.stage7d12_symbol_derivative_verifier import (
    STAGE7D12_VERIFIER_VERSION,
    Stage7D12VerificationReceipt,
)
from st_omr_training.stage7d12_symbol_derivatives import Stage7D12DerivativeReceipt


HEAD = "a" * 40
ORIGIN = "https://github.com/khfy7wpr5p-maker/st-omr-training.git"
H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
INVENTORY = {
    "train": {
        "notehead": {"open": 10, "filled": 20},
        "rest": {"half": 3, "quarter": 4, "eighth": 5},
        "accidental": {"sharp": 6, "flat": 7, "natural": 8},
    },
    "validation": {
        "notehead": {"open": 2, "filled": 3},
        "rest": {"half": 1, "quarter": 1, "eighth": 1},
        "accidental": {"sharp": 1, "flat": 1, "natural": 1},
    },
}


def _build_receipt() -> Stage7D12DerivativeReceipt:
    return Stage7D12DerivativeReceipt(
        derivative_build_id=H1,
        manifest_sha256=H2,
        sample_count=1383,
        family_count=461,
        sample_split_counts={"train": 1230, "validation": 153},
        family_split_counts={"train": 410, "validation": 51},
        label_count=1383,
        label_bytes_total=123456,
        artifact_binding_sha256=H3,
        observed_class_inventory=INVENTORY,
        test_specialist_records=0,
        optimizer_steps=0,
        complete_marker_written=False,
    )


def _verification_receipt(*, manifest_sha: str = H2) -> Stage7D12VerificationReceipt:
    return Stage7D12VerificationReceipt(
        verifier_version=STAGE7D12_VERIFIER_VERSION,
        derivative_build_id=H1,
        manifest_sha256=manifest_sha,
        artifact_binding_sha256=H3,
        sample_count=1383,
        family_count=461,
        label_count=1383,
        label_bytes_total=123456,
        sample_split_counts={"train": 1230, "validation": 153},
        family_split_counts={"train": 410, "validation": 51},
        observed_class_inventory=INVENTORY,
        test_specialist_records=0,
        optimizer_steps=0,
        complete_marker_present=False,
        verification_passed=True,
    )


class Stage7D12AuthoritativeBundleTests(unittest.TestCase):
    @patch("st_omr_training.stage7d12_authoritative_bundle.verify_stage7c_runtime")
    @patch("st_omr_training.stage7d12_authoritative_bundle.verify_stage7d12_symbol_derivatives")
    @patch("st_omr_training.stage7d12_authoritative_bundle.build_stage7d12_symbol_derivatives")
    @patch("st_omr_training.stage7d12_authoritative_bundle.verify_authoritative_repository")
    def test_exact_head_build_then_independent_verify_passes(
        self,
        repository_identity,
        builder,
        verifier,
        runtime,
    ) -> None:
        repository_identity.return_value = (HEAD, ORIGIN)
        builder.return_value = _build_receipt()
        verifier.return_value = _verification_receipt()

        with tempfile.TemporaryDirectory() as temporary:
            result = run_verified_stage7d12_authoritative_bundle(
                corpus_root=Path(temporary) / "corpus",
                d6_root=Path(temporary) / "d6",
                output_root=Path(temporary) / "d12",
                repository_root=Path(temporary) / "repo",
                expected_repository_sha=HEAD,
            )

        self.assertEqual(result.repository_sha, HEAD)
        self.assertEqual(result.repository_origin, ORIGIN)
        self.assertEqual(result.sample_count, 1383)
        self.assertEqual(result.sample_split_counts, {"train": 1230, "validation": 153})
        self.assertEqual(result.test_specialist_records, 0)
        self.assertEqual(result.optimizer_steps, 0)
        self.assertIs(result.complete_marker_present, False)
        self.assertIs(result.independent_verification_passed, True)
        self.assertEqual(repository_identity.call_count, 3)
        self.assertEqual(runtime.call_count, 3)
        builder.assert_called_once()
        verifier.assert_called_once()

    @patch("st_omr_training.stage7d12_authoritative_bundle.verify_authoritative_repository")
    def test_wrong_repository_head_fails_before_build(self, repository_identity) -> None:
        repository_identity.return_value = ("b" * 40, ORIGIN)
        with self.assertRaises(Stage7D12AuthoritativeError):
            run_verified_stage7d12_authoritative_bundle(
                corpus_root="/tmp/corpus",
                d6_root="/tmp/d6",
                output_root="/tmp/d12",
                repository_root="/tmp/repo",
                expected_repository_sha=HEAD,
            )

    @patch("st_omr_training.stage7d12_authoritative_bundle.verify_stage7c_runtime")
    @patch("st_omr_training.stage7d12_authoritative_bundle.verify_stage7d12_symbol_derivatives")
    @patch("st_omr_training.stage7d12_authoritative_bundle.build_stage7d12_symbol_derivatives")
    @patch("st_omr_training.stage7d12_authoritative_bundle.verify_authoritative_repository")
    def test_builder_verifier_identity_mismatch_fails_closed(
        self,
        repository_identity,
        builder,
        verifier,
        runtime,
    ) -> None:
        repository_identity.return_value = (HEAD, ORIGIN)
        builder.return_value = _build_receipt()
        verifier.return_value = _verification_receipt(manifest_sha="f" * 64)

        with self.assertRaises(Stage7D12AuthoritativeError):
            run_verified_stage7d12_authoritative_bundle(
                corpus_root="/tmp/corpus",
                d6_root="/tmp/d6",
                output_root="/tmp/d12",
                repository_root="/tmp/repo",
                expected_repository_sha=HEAD,
            )


if __name__ == "__main__":
    unittest.main()
