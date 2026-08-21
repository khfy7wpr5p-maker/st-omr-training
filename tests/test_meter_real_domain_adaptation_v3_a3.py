from __future__ import annotations

import unittest

import torch

from st_omr_training.meter_real_domain_adaptation_v3_a3 import (
    GAIN_GRID_SIZE_V3_A3,
    METER_REAL_DOMAIN_ADAPTATION_V3_A3,
    PARENT_ADAPTATION_VERSION_V3_A3,
    PARENT_REPOSITORY_SHA_V3_A3,
    PRESENCE_D11_SHA256_V3_A3,
    ClassificationSummaryV3A3,
    MeterRealDomainAdaptationV3A3Error,
    calibrated_logits_v3_a3,
    classification_summary_v3_a3,
    gain_grid_milli_v3_a3,
    gain_pairs_milli_v3_a3,
    production_promotion_allowed,
    real_phase0_gate_v3_a3,
    resolver_connection_allowed,
    runtime_connection_allowed,
    sealed_test_access_allowed,
    select_gain_pair_v3_a3,
    verify_parent_resume_metadata_v3_a3,
)
from st_omr_training.meter_teacher_gold_admission_v1 import METER_CLASSES


class MeterRealDomainAdaptationV3A3Tests(unittest.TestCase):
    def _summary(self, *, none=1.0, two=1.0, three=1.0, four=1.0, macro=1.0, accuracy=1.0):
        return ClassificationSummaryV3A3(
            record_count=54,
            macro_f1=macro,
            accuracy=accuracy,
            per_class_recall={
                "none": none,
                "2/4": two,
                "3/4": three,
                "4/4": four,
            },
            confusion=(
                (27, 0, 0, 0),
                (0, 9, 0, 0),
                (0, 0, 9, 0),
                (0, 0, 0, 9),
            ),
        )

    def test_contract_is_shadow_only(self) -> None:
        self.assertEqual(
            METER_REAL_DOMAIN_ADAPTATION_V3_A3,
            "meter-real-domain-adaptation-v3-a3-residual-calibration-screen",
        )
        self.assertFalse(sealed_test_access_allowed())
        self.assertFalse(runtime_connection_allowed())
        self.assertFalse(resolver_connection_allowed())
        self.assertFalse(production_promotion_allowed())

    def test_gain_grid_is_exact_and_bounded(self) -> None:
        grid = gain_grid_milli_v3_a3()
        self.assertEqual(grid, tuple(range(1000, 1251, 25)))
        pairs = gain_pairs_milli_v3_a3()
        self.assertEqual(len(pairs), GAIN_GRID_SIZE_V3_A3)
        self.assertEqual(len(set(pairs)), GAIN_GRID_SIZE_V3_A3)
        self.assertIn((1000, 1000), pairs)
        self.assertIn((1250, 1250), pairs)

    def test_parent_resume_is_fail_closed(self) -> None:
        good = {
            "adaptation_version": PARENT_ADAPTATION_VERSION_V3_A3,
            "repository_sha": PARENT_REPOSITORY_SHA_V3_A3,
            "base_checkpoint_sha256": PRESENCE_D11_SHA256_V3_A3,
            "completed_epoch": 20,
            "best_epoch": 20,
            "current_model_state": {"x": torch.tensor([1.0])},
        }
        verify_parent_resume_metadata_v3_a3(good)
        bad = dict(good)
        bad["best_epoch"] = 19
        with self.assertRaises(MeterRealDomainAdaptationV3A3Error):
            verify_parent_resume_metadata_v3_a3(bad)

    def test_calibration_changes_only_two_and_four_residuals(self) -> None:
        base = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        residual = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        calibrated = calibrated_logits_v3_a3(
            base,
            residual,
            gain_2_4_milli=1250,
            gain_4_4_milli=1100,
        )
        expected = torch.tensor([[11.0, 22.5, 33.0, 44.4]])
        self.assertTrue(torch.allclose(calibrated, expected))
        with self.assertRaises(ValueError):
            calibrated_logits_v3_a3(
                base,
                residual,
                gain_2_4_milli=1300,
                gain_4_4_milli=1000,
            )

    def test_classification_summary_uses_all_four_classes(self) -> None:
        summary = classification_summary_v3_a3(
            [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 0, 1, 2, 2, 2, 3, 2],
        )
        self.assertEqual(tuple(summary.per_class_recall), tuple(METER_CLASSES))
        self.assertEqual(summary.per_class_recall["none"], 1.0)
        self.assertEqual(summary.per_class_recall["2/4"], 0.5)
        self.assertEqual(summary.per_class_recall["3/4"], 1.0)
        self.assertEqual(summary.per_class_recall["4/4"], 0.5)
        self.assertEqual(summary.accuracy, 0.75)

    def test_selection_uses_train_rank_and_smallest_identity_deviation(self) -> None:
        parent = self._summary(two=8 / 9, four=8 / 9, macro=0.90, accuracy=52 / 54)
        candidates = {
            pair: self._summary(two=8 / 9, four=8 / 9, macro=0.90, accuracy=52 / 54)
            for pair in gain_pairs_milli_v3_a3()
        }
        candidates[(1175, 1100)] = self._summary(macro=0.99, accuracy=1.0)
        candidates[(1200, 1075)] = self._summary(macro=0.99, accuracy=1.0)
        selected = select_gain_pair_v3_a3(
            parent_train_summary=parent,
            candidate_train_summaries=candidates,
        )
        # Same quality and total deviation; smaller maximum gain wins.
        self.assertEqual((selected.gain_2_4_milli, selected.gain_4_4_milli), (1175, 1100))

    def test_selection_rejects_none_recall_regression(self) -> None:
        parent = self._summary(none=1.0, two=8 / 9, four=8 / 9, macro=0.90, accuracy=52 / 54)
        candidates = {
            pair: self._summary(none=1.0, two=8 / 9, four=8 / 9, macro=0.90, accuracy=52 / 54)
            for pair in gain_pairs_milli_v3_a3()
        }
        candidates[(1250, 1250)] = self._summary(
            none=26 / 27,
            two=1.0,
            three=1.0,
            four=1.0,
            macro=0.999,
            accuracy=53 / 54,
        )
        selected = select_gain_pair_v3_a3(
            parent_train_summary=parent,
            candidate_train_summaries=candidates,
        )
        self.assertNotEqual((selected.gain_2_4_milli, selected.gain_4_4_milli), (1250, 1250))

    def test_real_phase0_gate_requires_all_positive_three_of_three(self) -> None:
        accepted, reasons = real_phase0_gate_v3_a3(self._summary())
        self.assertTrue(accepted)
        self.assertEqual(reasons, ())
        failed, reasons = real_phase0_gate_v3_a3(
            self._summary(two=2 / 3, macro=0.84, accuracy=16 / 18)
        )
        self.assertFalse(failed)
        self.assertIn("REAL_2_4_RECALL_NOT_3_OF_3", reasons)


if __name__ == "__main__":
    unittest.main()
