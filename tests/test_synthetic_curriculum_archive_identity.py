from __future__ import annotations

import unittest

from st_omr_training.synthetic_curriculum_corpus_gate import EXPECTED_ARCHIVE_NAME
from st_omr_training.synthetic_curriculum_export_gate import _EXPECTED_ARCHIVE


ACTUAL_FROZEN_ARCHIVE = "st-omr-synthetic-curriculum-v1-d9320e362f162cd2.tar.gz"


class SyntheticCurriculumArchiveIdentityTests(unittest.TestCase):
    def test_d0_and_d1_share_the_actual_frozen_archive_name(self):
        self.assertEqual(_EXPECTED_ARCHIVE, ACTUAL_FROZEN_ARCHIVE)
        self.assertEqual(EXPECTED_ARCHIVE_NAME, ACTUAL_FROZEN_ARCHIVE)
        self.assertEqual(_EXPECTED_ARCHIVE, EXPECTED_ARCHIVE_NAME)


if __name__ == "__main__":
    unittest.main()
