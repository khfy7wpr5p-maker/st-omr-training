from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
import unittest

import torch

from st_omr_training.meter_v4_1_numerator_specialist import NumeratorRecordV4_1, VerifiedParentV4_1
from st_omr_training.meter_v4_2_full_train_dev_screen import (
    DevSummaryV4_2,
    FINAL_SEED_V4_2,
    build_full_train_batch_v4_2,
    dev_decision_v4_2,
)
from st_omr_training.meter_v4_2_full_train_dev_screen_run import repository_binding_v4_2


class MeterV42FullTrainDevScreenTests(unittest.TestCase):
    def _parent(self) -> VerifiedParentV4_1:
        records = []
        counter = 0
        for class_name in ("2", "3", "4"):
            for index in range(9):
                records.append(NumeratorRecordV4_1(
                    record_id=sha256(f"record-{counter}".encode("ascii")).hexdigest(),
                    family_id=f"family-{counter:02d}",
                    numerator_class=class_name,
                    fold=index % 3,
                    crop_png_sha256=sha256(f"crop-{counter}".encode("ascii")).hexdigest(),
                ))
                counter += 1
        return VerifiedParentV4_1(
            root=Path("/nonexistent-fixture"),
            result_sha256="a" * 64,
            repository_binding="b" * 64,
            records=tuple(records),
            result={},
        )

    def test_full_train_batch_is_exact_balanced_243(self) -> None:
        parent = self._parent()
        crops = {}
        for index, row in enumerate(parent.records):
            image = torch.zeros((1, 64, 64), dtype=torch.float32)
            image[0, 8 + (index % 20), 8 + (index % 20)] = 1.0
            crops[row.record_id] = image
        batch, labels, origins = build_full_train_batch_v4_2(parent, crops)
        self.assertEqual(tuple(batch.shape), (243, 1, 64, 64))
        self.assertEqual(tuple(labels.shape), (243,))
        self.assertEqual(Counter(labels.tolist()), Counter({0: 81, 1: 81, 2: 81}))
        self.assertEqual(len(set(origins)), 27)
        self.assertEqual(len(origins), 243)

    def test_full_train_batch_rejects_missing_record(self) -> None:
        parent = self._parent()
        crops = {row.record_id: torch.zeros((1, 64, 64), dtype=torch.float32) for row in parent.records[:-1]}
        with self.assertRaises(RuntimeError):
            build_full_train_batch_v4_2(parent, crops)

    def test_dev_gate_is_strict_nine_of_nine(self) -> None:
        passed = DevSummaryV4_2(
            record_count=9,
            accuracy=1.0,
            macro_f1=1.0,
            per_class_recall={"2": 1.0, "3": 1.0, "4": 1.0},
            confusion=((3,0,0),(0,3,0),(0,0,3)),
        )
        decision = dev_decision_v4_2(passed, deterministic_repeat_pass=True)
        self.assertEqual(decision["name"], "FULL_TRAIN_DEV_SCREEN_PASS")
        self.assertTrue(decision["accepted_for_shadow_planning"])
        self.assertTrue(decision["fresh_independent_holdout_required"])
        self.assertFalse(decision["production_promotion_authorized"])

        failed = DevSummaryV4_2(
            record_count=9,
            accuracy=8/9,
            macro_f1=0.88,
            per_class_recall={"2": 2/3, "3": 1.0, "4": 1.0},
            confusion=((2,1,0),(0,3,0),(0,0,3)),
        )
        rejected = dev_decision_v4_2(failed, deterministic_repeat_pass=True)
        self.assertEqual(rejected["name"], "FULL_TRAIN_DEV_SCREEN_HOLD")
        self.assertFalse(rejected["accepted_for_shadow_planning"])
        self.assertIn("DEV_ACCURACY_NOT_9_OF_9", rejected["reasons"])
        self.assertIn("DEV_2_RECALL_NOT_3_OF_3", rejected["reasons"])

    def test_repository_binding_and_final_seed_are_frozen(self) -> None:
        git_sha = "1" * 40
        expected = sha256(("git-commit-sha1:" + git_sha).encode("ascii")).hexdigest()
        self.assertEqual(repository_binding_v4_2(git_sha), expected)
        self.assertEqual(FINAL_SEED_V4_2, 812042)
        with self.assertRaises(RuntimeError):
            repository_binding_v4_2("bad")


if __name__ == "__main__":
    unittest.main()
