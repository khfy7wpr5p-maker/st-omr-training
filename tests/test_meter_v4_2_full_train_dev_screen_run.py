from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from st_omr_training.meter_v4_2_full_train_dev_screen import DevSummaryV4_2, FullTrainResultV4_2
from st_omr_training.meter_v4_2_full_train_dev_screen_run import (
    repository_binding_v4_2,
    run_meter_v4_2_full_train_dev_screen,
)


class MeterV42RunOrderingTests(unittest.TestCase):
    def test_validation_selection_occurs_only_after_two_full_train_runs(self) -> None:
        order = []
        dummy_model = torch.nn.Linear(1, 1)
        train_result = FullTrainResultV4_2(
            final_loss=0.1,
            model_state_sha256="c" * 64,
            optimizer_steps=160,
            model=dummy_model,
        )
        parent = SimpleNamespace(
            records=(SimpleNamespace(record_id="r1"),),
            result_sha256="a" * 64,
            repository_binding="b" * 64,
        )
        v4_1_result = {
            "decision": {"name": "LEARNED_NUMERATOR_SIGNAL_STRONG", "strong_signal": True, "reasons": []},
            "oof_summary": {"record_count": 27, "accuracy": 1.0, "macro_f1": 1.0},
        }
        summary = DevSummaryV4_2(
            record_count=9,
            accuracy=1.0,
            macro_f1=1.0,
            per_class_recall={"2": 1.0, "3": 1.0, "4": 1.0},
            confusion=((3,0,0),(0,3,0),(0,0,3)),
        )

        def fake_train(*_args, **_kwargs):
            order.append("train")
            return train_result

        def fake_select(**_kwargs):
            order.append("select_validation")
            self.assertEqual(order[:2], ["train", "train"])
            return tuple(range(9)), {"pilot_sha256": "p"}

        git_sha = "1" * 40
        binding = repository_binding_v4_2(git_sha)
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "st_omr_training.meter_v4_2_full_train_dev_screen_run._verify_v4_1_result",
            return_value=v4_1_result,
        ), mock.patch(
            "st_omr_training.meter_v4_2_full_train_dev_screen_run.verify_parent_artifact_v4_1",
            return_value=parent,
        ), mock.patch(
            "st_omr_training.meter_v4_2_full_train_dev_screen_run.load_crop_tensor_v4_1",
            return_value=torch.zeros((1,64,64), dtype=torch.float32),
        ), mock.patch(
            "st_omr_training.meter_v4_2_full_train_dev_screen_run.train_full_candidate_v4_2",
            side_effect=fake_train,
        ), mock.patch(
            "st_omr_training.meter_v4_2_full_train_dev_screen_run.select_validation_positives_v4_2",
            side_effect=fake_select,
        ), mock.patch(
            "st_omr_training.meter_v4_2_full_train_dev_screen_run.evaluate_validation_positives_v4_2",
            return_value=(tuple(), summary),
        ):
            result = run_meter_v4_2_full_train_dev_screen(
                parent_v4_0_root="unused-v4-0",
                parent_v4_1_root="unused-v4-1",
                pilot_path="unused-pilot",
                choices_path="unused-choices",
                permission_path="unused-permission",
                privacy_path="unused-privacy",
                output_root=Path(tmp)/"out",
                git_commit_sha=git_sha,
                repository_sha=binding,
            )
        self.assertEqual(order, ["train", "train", "select_validation"])
        self.assertEqual(result["decision"]["name"], "FULL_TRAIN_DEV_SCREEN_PASS")
        self.assertTrue(result["safety"]["fresh_independent_holdout_required"])
        self.assertFalse(result["safety"]["production_promotion_authorized"])


if __name__ == "__main__":
    unittest.main()
