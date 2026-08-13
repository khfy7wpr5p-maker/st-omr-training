from __future__ import annotations

import copy
import unittest

from st_omr_training.synthetic_curriculum_acceptance import SyntheticCurriculumAcceptanceError
from st_omr_training.synthetic_curriculum_export_gate import verify_stage7d_export_evidence
from test_synthetic_curriculum_acceptance import _canonical, _payload


class SyntheticCurriculumExportGateTests(unittest.TestCase):
    def test_frozen_surface_passes(self):
        receipt = verify_stage7d_export_evidence(_canonical(_payload()))
        self.assertEqual(len(receipt.evidence_sha256), 64)

    def test_repository_drift_fails(self):
        payload = copy.deepcopy(_payload())
        payload["source_repository"] = "changed"
        with self.assertRaises(SyntheticCurriculumAcceptanceError):
            verify_stage7d_export_evidence(_canonical(payload))

    def test_profile_key_drift_fails(self):
        payload = copy.deepcopy(_payload())
        payload["plan"]["family_profile_counts"] = {f"p{index}": 64 for index in range(8)}
        with self.assertRaises(SyntheticCurriculumAcceptanceError):
            verify_stage7d_export_evidence(_canonical(payload))


if __name__ == "__main__":
    unittest.main()
