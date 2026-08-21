from __future__ import annotations

from hashlib import sha256
import unittest

from st_omr_training.meter_v4_1_numerator_specialist import MeterV4_1Error
from st_omr_training.meter_v4_1_numerator_specialist_run import repository_binding_v4_1


class MeterV41NumeratorSpecialistRunTests(unittest.TestCase):
    def test_repository_binding_is_exact_and_deterministic(self) -> None:
        git_sha = "a" * 40
        expected = sha256(("git-commit-sha1:" + git_sha).encode("ascii")).hexdigest()
        self.assertEqual(repository_binding_v4_1(git_sha), expected)
        self.assertEqual(repository_binding_v4_1(git_sha), repository_binding_v4_1(git_sha))

    def test_repository_binding_rejects_noncanonical_git_sha(self) -> None:
        for value in ("a" * 39, "A" * 40, "g" * 40, ""):
            with self.subTest(value=value):
                with self.assertRaises(MeterV4_1Error):
                    repository_binding_v4_1(value)


if __name__ == "__main__":
    unittest.main()
