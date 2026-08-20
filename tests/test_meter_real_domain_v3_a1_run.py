from __future__ import annotations

from pathlib import Path
import unittest

from st_omr_training.meter_real_domain_adaptation_v3_a1_run import (
    CHECKPOINT_ROLE_V3_A1,
    METRICS_SCHEMA_V3_A1,
    RESUME_ROLE_V3_A1,
    VERIFICATION_SCHEMA_V3_A1,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "meter_real_domain_background_runner_v3_a1.py"
RUN_MODULE = ROOT / "st_omr_training" / "meter_real_domain_adaptation_v3_a1_run.py"


class MeterRealDomainV3A1RunTests(unittest.TestCase):
    def test_roles_are_v3_a1_specific(self) -> None:
        self.assertIn("v3-a1", METRICS_SCHEMA_V3_A1)
        self.assertIn("v3-a1", VERIFICATION_SCHEMA_V3_A1)
        self.assertIn("v3-a1", CHECKPOINT_ROLE_V3_A1)
        self.assertIn("v3-a1", RESUME_ROLE_V3_A1)

    def test_runner_uses_v3_a1_and_keeps_closed_boundaries(self) -> None:
        source = RUNNER.read_text("utf-8")
        self.assertIn("meter_real_domain_adaptation_v3_a1_run", source)
        self.assertIn("FROZEN_ADAPTATION_CONFIG_V3_A1", source)
        self.assertIn("bbox_frozen_exact", source)
        self.assertNotIn("meter_real_domain_background_runner_v2.py", source)
        self.assertNotIn("/01_REVIEW/test", source)

    def test_run_module_hard_fails_on_localization_change(self) -> None:
        source = RUN_MODULE.read_text("utf-8")
        self.assertIn("V3-A1 changed synthetic localization despite exact frozen bbox output", source)
        self.assertIn('"bbox_frozen_exact": True', source)
        self.assertIn('"test_opened": False', source)
        self.assertIn('"runtime_connected": False', source)
        self.assertIn('"resolver_connected": False', source)
        self.assertIn('"production_promotion_authorized": False', source)


if __name__ == "__main__":
    unittest.main()
