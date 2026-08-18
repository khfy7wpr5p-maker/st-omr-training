from __future__ import annotations

import inspect
import unittest

import torch

from st_omr_training.m4_e3k_b1_d7_staff_geometry import D7_DENSE_THRESHOLD
from st_omr_training.m4_e3k_b1_r1_decoder_audit import (
    audit_d7_staff_probabilities,
    profile_fingerprint,
)
from st_omr_training.stage7d7_specialist_training import FROZEN_D7_CONFIG, STAFF_CHANNELS


class B1R1DecoderAuditTests(unittest.TestCase):
    def _blank(self) -> torch.Tensor:
        return torch.zeros(
            (
                len(STAFF_CHANNELS),
                FROZEN_D7_CONFIG.input_height,
                FROZEN_D7_CONFIG.input_width,
            ),
            dtype=torch.float32,
        )

    def test_frozen_threshold_is_not_changed(self) -> None:
        self.assertEqual(D7_DENSE_THRESHOLD, 0.50)
        self.assertEqual(len(profile_fingerprint()), 64)

    def test_perfect_staff_decodes_with_trace_parity(self) -> None:
        probabilities = self._blank()
        line_index = STAFF_CHANNELS.index("staff_lines")
        region_index = STAFF_CHANNELS.index("staff_region")
        probabilities[region_index, 20:35, 50:451] = 0.90
        for y in (22, 25, 28, 31, 34):
            probabilities[line_index, y, 60:441] = 0.90

        result = audit_d7_staff_probabilities(
            probabilities,
            original_width=1024,
            original_height=192,
            truth_system_count=1,
        )

        self.assertEqual(result["raw_region_components"], 1)
        self.assertEqual(result["qualifying_region_components"], 1)
        self.assertEqual(result["slope_pass_components"], 1)
        self.assertEqual(result["five_line_template_pass_components"], 1)
        self.assertEqual(result["x_support_pass_components"], 1)
        self.assertEqual(result["decoded_staff_count"], 1)
        self.assertTrue(result["decoder_parity_pass"])
        self.assertEqual(
            result["terminal_reason"],
            "DECODED_STAFF_COUNT_EQUALS_TRUTH_SYSTEM_COUNT",
        )

    def test_no_region_support_is_attributed_before_component_decode(self) -> None:
        result = audit_d7_staff_probabilities(
            self._blank(),
            original_width=1024,
            original_height=192,
            truth_system_count=2,
        )
        self.assertEqual(result["region_active_pixels"], 0)
        self.assertEqual(result["raw_region_components"], 0)
        self.assertEqual(result["decoded_staff_count"], 0)
        self.assertEqual(
            result["terminal_reason"],
            "NO_REGION_SUPPORT_AT_FROZEN_THRESHOLD",
        )

    def test_small_region_component_is_rejected_by_existing_size_gate(self) -> None:
        probabilities = self._blank()
        region_index = STAFF_CHANNELS.index("staff_region")
        probabilities[region_index, 20:22, 10:20] = 0.90

        result = audit_d7_staff_probabilities(
            probabilities,
            original_width=1024,
            original_height=192,
            truth_system_count=1,
        )

        self.assertEqual(result["raw_region_components"], 1)
        self.assertEqual(result["qualifying_region_components"], 0)
        failures = result["component_failure_counts"]
        self.assertEqual(failures.get("SIZE_WIDTH_AND_AREA_FAIL"), 1)
        self.assertEqual(
            result["terminal_reason"],
            "ALL_REGION_COMPONENTS_REJECTED_BY_FROZEN_SIZE_GATES",
        )

    def test_four_line_component_is_attributed_to_five_line_template(self) -> None:
        probabilities = self._blank()
        line_index = STAFF_CHANNELS.index("staff_lines")
        region_index = STAFF_CHANNELS.index("staff_region")
        probabilities[region_index, 20:35, 50:451] = 0.90
        for y in (22, 26, 30, 34):
            probabilities[line_index, y, 60:441] = 0.90

        result = audit_d7_staff_probabilities(
            probabilities,
            original_width=1024,
            original_height=192,
            truth_system_count=1,
        )

        self.assertEqual(result["qualifying_region_components"], 1)
        self.assertEqual(result["slope_pass_components"], 1)
        self.assertEqual(result["five_line_template_pass_components"], 0)
        self.assertEqual(result["five_line_template_fail_components"], 1)
        self.assertEqual(result["decoded_staff_count"], 0)
        self.assertEqual(
            result["terminal_reason"],
            "ALL_SLOPE_PASS_COMPONENTS_FAILED_FIVE_LINE_TEMPLATE",
        )

    def test_diagnostic_source_contains_no_training_or_threshold_sweep_path(self) -> None:
        import st_omr_training.m4_e3k_b1_r1_decoder_audit as module

        source = inspect.getsource(module)
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn("backward(", source)
        self.assertNotIn("load_state_dict(", source)
        self.assertNotIn("for threshold in", source)
        self.assertIn('"threshold_tuning": False', source)
        self.assertIn('"decoder_behavior_changed": False', source)
        self.assertIn('"validation_opened": False', source)
        self.assertIn('"test_opened": False', source)


if __name__ == "__main__":
    unittest.main()
