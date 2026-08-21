from __future__ import annotations

from collections import Counter
import unittest

import torch

from st_omr_training.meter_v4_0_numerator_audit import (
    AuditRecordIdentityV4_0,
    ClassificationSummaryV4_0,
    FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0,
    MeterV4_0AuditError,
    NumeratorAuditConfigV4_0,
    audit_decision_v4_0,
    build_numerator_specialist_v4_0,
    classification_summary_v4_0,
    d10_access_allowed,
    deterministic_shift_bank_v4_0,
    fold_plan_v4_0,
    numerator_crop_bounds_v4_0,
    numerator_crop_tensor_v4_0,
    production_promotion_allowed,
    resolver_connection_allowed,
    runtime_connection_allowed,
    sealed_test_access_allowed,
    teacher_adaptation_validation_evaluation_allowed,
)
from st_omr_training.training_model import count_trainable_parameters


class MeterV40NumeratorAuditTests(unittest.TestCase):
    def _identities(self) -> tuple[AuditRecordIdentityV4_0, ...]:
        rows = []
        counter = 1
        for meter_class in ("2/4", "3/4", "4/4"):
            for index in range(9):
                rows.append(
                    AuditRecordIdentityV4_0(
                        record_id=f"{counter:064x}",
                        family_id=f"family-{meter_class.replace('/', '-')}-{index}",
                        meter_class=meter_class,
                    )
                )
                counter += 1
        return tuple(rows)

    def test_config_is_frozen(self) -> None:
        self.assertEqual(FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0.output_size, 64)
        self.assertEqual(FROZEN_NUMERATOR_AUDIT_CONFIG_V4_0.folds, 3)
        with self.assertRaises(ValueError):
            NumeratorAuditConfigV4_0(horizontal_padding_milli=149)

    def test_crop_bounds_use_top_half_with_bounded_padding(self) -> None:
        bbox = {"x_min": 100.0, "y_min": 40.0, "x_max": 140.0, "y_max": 120.0}
        self.assertEqual(numerator_crop_bounds_v4_0(bbox), (94, 36, 146, 84))
        with self.assertRaises(MeterV4_0AuditError):
            numerator_crop_bounds_v4_0({"x_min": -1, "y_min": 0, "x_max": 10, "y_max": 20})

    def test_crop_tensor_is_exact_64_square_and_finite(self) -> None:
        image = torch.zeros((1, 192, 256), dtype=torch.float32)
        image[:, 45:75, 105:125] = 1.0
        bbox = {"x_min": 100.0, "y_min": 40.0, "x_max": 140.0, "y_max": 120.0}
        crop = numerator_crop_tensor_v4_0(image, bbox)
        self.assertEqual(tuple(crop.shape), (1, 64, 64))
        self.assertTrue(bool(torch.isfinite(crop).all()))
        self.assertGreater(float(crop.sum().item()), 0.0)

    def test_fold_plan_is_balanced_and_family_disjoint(self) -> None:
        identities = self._identities()
        first = fold_plan_v4_0(identities)
        second = fold_plan_v4_0(tuple(reversed(identities)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 27)
        self.assertEqual(len({row.family_id for row in first}), 27)
        counts = Counter((row.fold, row.meter_class) for row in first)
        for fold in range(3):
            for meter_class in ("2/4", "3/4", "4/4"):
                self.assertEqual(counts[(fold, meter_class)], 3)

    def test_fold_plan_rejects_wrong_cardinality(self) -> None:
        with self.assertRaises(MeterV4_0AuditError):
            fold_plan_v4_0(self._identities()[:-1])

    def test_shift_bank_is_deterministic_and_no_wraparound(self) -> None:
        image = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
        image[0, 0, 0, 0] = 1.0
        first = deterministic_shift_bank_v4_0(image)
        second = deterministic_shift_bank_v4_0(image)
        self.assertEqual(tuple(first.shape), (9, 1, 64, 64))
        self.assertTrue(torch.equal(first, second))
        self.assertLessEqual(float(first.sum().item()), 9.0)

    def test_tiny_specialist_is_bounded(self) -> None:
        model = build_numerator_specialist_v4_0(seed=840_001)
        self.assertLessEqual(count_trainable_parameters(model), 50_000)
        logits = model(torch.zeros((2, 1, 64, 64), dtype=torch.float32))
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_summary_and_strong_decision(self) -> None:
        truth = [0] * 9 + [1] * 9 + [2] * 9
        predicted = truth.copy()
        predicted[0] = 1
        predicted[9] = 0
        summary = classification_summary_v4_0(truth, predicted)
        self.assertEqual(summary.record_count, 27)
        self.assertAlmostEqual(summary.accuracy, 25 / 27)
        self.assertAlmostEqual(summary.per_class_recall["2"], 8 / 9)
        self.assertAlmostEqual(summary.per_class_recall["3"], 8 / 9)
        self.assertAlmostEqual(summary.per_class_recall["4"], 1.0)
        decision = audit_decision_v4_0(summary)
        self.assertTrue(decision.strong_signal)
        self.assertEqual(decision.decision, "REPRESENTATION_SIGNAL_STRONG")
        self.assertEqual(decision.reasons, ())

    def test_weak_decision_is_fail_closed(self) -> None:
        summary = ClassificationSummaryV4_0(
            record_count=27,
            accuracy=24 / 27,
            macro_f1=0.8,
            per_class_recall={"2": 7 / 9, "3": 8 / 9, "4": 9 / 9},
            confusion=((7, 2, 0), (1, 8, 0), (0, 0, 9)),
        )
        decision = audit_decision_v4_0(summary)
        self.assertFalse(decision.strong_signal)
        self.assertIn("OOF_ACCURACY_BELOW_25_OF_27", decision.reasons)
        self.assertIn("OOF_2_RECALL_BELOW_8_OF_9", decision.reasons)

    def test_external_surfaces_are_closed(self) -> None:
        self.assertFalse(sealed_test_access_allowed())
        self.assertFalse(d10_access_allowed())
        self.assertFalse(teacher_adaptation_validation_evaluation_allowed())
        self.assertFalse(runtime_connection_allowed())
        self.assertFalse(resolver_connection_allowed())
        self.assertFalse(production_promotion_allowed())


if __name__ == "__main__":
    unittest.main()
