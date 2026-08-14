from __future__ import annotations

import inspect
import math
import tempfile
from pathlib import Path
import unittest

import torch

from st_omr_training import stage7d8_structure_diagnostics as d8


class Stage7D8ProbabilityDiagnosticTests(unittest.TestCase):
    def test_threshold_sweep_can_localize_calibration_error(self) -> None:
        probabilities = torch.tensor(
            [[[[0.30, 0.40, 0.10, 0.20]]]],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [[[[1.0, 1.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        )
        report = d8.diagnose_probability_tensor(
            probabilities,
            targets,
            ("barline",),
            (250, 500, 750),
        )
        channel = report["barline"]
        self.assertEqual(channel["positive_records"], 1)
        self.assertEqual(channel["positive_pixels"], 2)
        self.assertEqual(channel["total_pixels"], 4)
        self.assertEqual(channel["positive_pixel_fraction"], 0.5)
        self.assertEqual(channel["best_threshold"], 0.25)
        self.assertEqual(channel["best_threshold_metrics"]["dice"], 1.0)
        self.assertEqual(channel["threshold_0_50"]["dice"], 0.0)
        self.assertEqual(channel["best_threshold_dice_gain_over_0_50"], 1.0)

    def test_tie_break_prefers_threshold_closest_to_half(self) -> None:
        probabilities = torch.tensor(
            [[[[0.90, 0.10]]]],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [[[[1.0, 0.0]]]],
            dtype=torch.float32,
        )
        report = d8.diagnose_probability_tensor(
            probabilities,
            targets,
            ("system_region",),
            (250, 500, 750),
        )
        self.assertEqual(report["system_region"]["best_threshold"], 0.5)

    def test_nan_and_out_of_range_inputs_fail_closed(self) -> None:
        valid = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        nan_probabilities = valid.clone()
        nan_probabilities[0, 0, 0, 0] = float("nan")
        with self.assertRaises(Exception):
            d8.diagnose_probability_tensor(nan_probabilities, valid, ("x",), (500,))

        bad_targets = valid.clone()
        bad_targets[0, 0, 0, 0] = 1.1
        with self.assertRaises(d8.Stage7D8DiagnosticError):
            d8.diagnose_probability_tensor(valid, bad_targets, ("x",), (500,))

    def test_shape_and_channel_mismatch_fail_closed(self) -> None:
        probabilities = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
        targets = probabilities.clone()
        with self.assertRaises(d8.Stage7D8DiagnosticError):
            d8.diagnose_probability_tensor(probabilities, targets, ("only_one",), (500,))
        with self.assertRaises(d8.Stage7D8DiagnosticError):
            d8.diagnose_probability_tensor(probabilities, targets[:, :1], ("a", "b"), (500,))


class Stage7D8ToleranceTests(unittest.TestCase):
    def test_one_pixel_tolerance_exposes_near_miss_localization(self) -> None:
        probabilities = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
        targets = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
        probabilities[0, 0, 2, 3] = 1.0
        targets[0, 0, 2, 2] = 1.0

        exact = d8.diagnose_probability_tensor(
            probabilities,
            targets,
            ("barline",),
            (500,),
        )["barline"]["threshold_0_50"]["dice"]
        tolerant = d8.tolerant_f1_for_probabilities(
            probabilities,
            targets,
            ("barline",),
            {"barline": 500},
            1,
        )["barline"]["f1"]

        self.assertEqual(exact, 0.0)
        self.assertEqual(tolerant, 1.0)

    def test_tolerance_radius_is_bounded(self) -> None:
        probabilities = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        targets = probabilities.clone()
        with self.assertRaises(d8.Stage7D8DiagnosticError):
            d8.tolerant_f1_for_probabilities(
                probabilities,
                targets,
                ("x",),
                {"x": 500},
                0,
            )


class Stage7D8ContractTests(unittest.TestCase):
    def test_exact_d7_external_identity_is_frozen(self) -> None:
        self.assertEqual(
            d8.EXPECTED_D7_RUN_ID,
            "4ce2903206c7965471bb9569d379d8d9d1022d9248d80886638acfe0bd822598",
        )
        self.assertEqual(
            d8.EXPECTED_D7_CHECKPOINT_SHA256,
            "5f009ca8ba68d38497a7dd25590d4dd98c537f20c5d5525bf66e288afbf417dc",
        )
        self.assertEqual(d8.DEFAULT_THRESHOLD_MILLIS, 500)
        self.assertIn(500, d8.THRESHOLD_MILLIS)
        self.assertEqual(d8.TOLERANCE_RADII, (1, 2))

    def test_d8_module_contains_no_optimizer_construction_or_step(self) -> None:
        source = inspect.getsource(d8)
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("optimizer.step", source)
        self.assertNotIn("loss.backward", source)

    def test_output_root_must_be_fresh_and_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            repo = base / "repo"
            repo.mkdir()
            with self.assertRaises(d8.Stage7D8DiagnosticError):
                d8._fresh_output_root(repo / "nested", repo)

            external = base / "external"
            d8._fresh_output_root(external, repo)
            self.assertTrue(external.is_dir())
            with self.assertRaises(d8.Stage7D8DiagnosticError):
                d8._fresh_output_root(external, repo)

    def test_expected_structure_metrics_are_finite_and_bounded(self) -> None:
        self.assertEqual(set(d8.EXPECTED_D7_STRUCTURE_DICE), set(d8.STRUCTURE_CHANNELS))
        for value in d8.EXPECTED_D7_STRUCTURE_DICE.values():
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
