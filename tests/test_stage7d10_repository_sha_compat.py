from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import st_omr_training.stage7d10_local_roi_derivatives as d10


class Stage7D10RepositoryShaCompatibilityTests(unittest.TestCase):
    def test_repository_sha_fields_accept_exact_git_sha40(self) -> None:
        value = "a" * 40
        for name in (
            "repository_sha",
            "manifest.repository_sha",
            "expected_repository_sha",
        ):
            with self.subTest(name=name):
                self.assertEqual(d10._hex64(name, value), value)

    def test_repository_sha_fields_reject_wrong_length_and_case(self) -> None:
        for value in ("a" * 39, "a" * 41, "A" * 40, "g" * 40, "a" * 64):
            with self.subTest(value=value):
                with self.assertRaises(d10.Stage7D10DerivativeError):
                    d10._hex64("expected_repository_sha", value)

    def test_sha256_artifact_fields_remain_strict_64_hex(self) -> None:
        sha256_value = "b" * 64
        self.assertEqual(d10._hex64("image_sha256", sha256_value), sha256_value)
        with self.assertRaises(d10.Stage7D10DerivativeError):
            d10._hex64("image_sha256", "b" * 40)

    def test_stage7c_repository_tuple_is_reduced_to_exact_head(self) -> None:
        head = "c" * 40
        with patch.object(
            d10,
            "_stage7d10_original_verify_authoritative_repository",
            return_value=(head, "https://github.com/khfy7wpr5p-maker/st-omr-training.git"),
        ):
            self.assertEqual(
                d10.verify_authoritative_repository(Path("/tmp/placeholder")),
                head,
            )

    def test_stage7c_repository_adapter_rejects_non_git_head(self) -> None:
        with patch.object(
            d10,
            "_stage7d10_original_verify_authoritative_repository",
            return_value=("d" * 64, "origin"),
        ):
            with self.assertRaises(d10.Stage7D10DerivativeError):
                d10.verify_authoritative_repository(Path("/tmp/placeholder"))


if __name__ == "__main__":
    unittest.main()
