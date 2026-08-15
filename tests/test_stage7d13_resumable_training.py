from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from st_omr_training.stage7d13_resumable_training import (
    _load_snapshot,
    _save_snapshot,
)
from st_omr_training.stage7d13_symbol_models import build_symbol_model
from st_omr_training.stage7d13_training import (
    Stage7D13TrainingError,
    _clone_state,
    training_profile_fingerprint,
)
from st_omr_training.stage7d13_verified_surface import D13_EXPECTED_OPTIMIZER_STEPS
from st_omr_training.training_model import count_trainable_parameters, model_state_sha256


REPOSITORY_SHA = "a" * 40


class Stage7D13ResumableTrainingTests(unittest.TestCase):
    def _model_optimizer(self):
        model = build_symbol_model("notehead", seed=713013)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.0007, weight_decay=0.0001,
            foreach=False, fused=False,
        )
        return model, optimizer

    def test_epoch_zero_snapshot_round_trips_with_safe_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, optimizer = self._model_optimizer()
            state_before = model_state_sha256(model)
            best_state = _clone_state(model)
            profile = training_profile_fingerprint()
            _save_snapshot(
                root=root,
                specialist="notehead",
                epoch=0,
                repository_sha=REPOSITORY_SHA,
                profile=profile,
                model=model,
                optimizer=optimizer,
                parameter_count=count_trainable_parameters(model),
                initial_validation_loss=1.25,
                best_loss=1.25,
                best_epoch=0,
                best_state=best_state,
                history=[],
                optimizer_steps=0,
                heartbeat=None,
            )

            restored_model, restored_optimizer = self._model_optimizer()
            loaded = _load_snapshot(
                root=root,
                specialist="notehead",
                repository_sha=REPOSITORY_SHA,
                profile=profile,
                model=restored_model,
                optimizer=restored_optimizer,
                heartbeat=None,
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            epoch, steps, initial_loss, best_loss, best_epoch, _, history = loaded
            self.assertEqual(epoch, 0)
            self.assertEqual(steps, 0)
            self.assertEqual(initial_loss, 1.25)
            self.assertEqual(best_loss, 1.25)
            self.assertEqual(best_epoch, 0)
            self.assertEqual(history, [])
            self.assertEqual(model_state_sha256(restored_model), state_before)

    def test_orphan_newer_checkpoint_does_not_override_committed_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, optimizer = self._model_optimizer()
            profile = training_profile_fingerprint()
            _save_snapshot(
                root=root,
                specialist="notehead",
                epoch=0,
                repository_sha=REPOSITORY_SHA,
                profile=profile,
                model=model,
                optimizer=optimizer,
                parameter_count=count_trainable_parameters(model),
                initial_validation_loss=1.0,
                best_loss=1.0,
                best_epoch=0,
                best_state=_clone_state(model),
                history=[],
                optimizer_steps=0,
                heartbeat=None,
            )

            # Simulates a runtime loss after the next checkpoint bytes were written
            # but before the JSON commit marker was atomically published.
            orphan = root / "notehead" / "epoch-01.pt"
            torch.save({"incomplete": True}, orphan)

            restored_model, restored_optimizer = self._model_optimizer()
            loaded = _load_snapshot(
                root=root,
                specialist="notehead",
                repository_sha=REPOSITORY_SHA,
                profile=profile,
                model=restored_model,
                optimizer=restored_optimizer,
                heartbeat=None,
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded[0], 0)
            self.assertEqual(loaded[1], 0)

    def test_repository_identity_drift_rejects_committed_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, optimizer = self._model_optimizer()
            profile = training_profile_fingerprint()
            _save_snapshot(
                root=root,
                specialist="notehead",
                epoch=0,
                repository_sha=REPOSITORY_SHA,
                profile=profile,
                model=model,
                optimizer=optimizer,
                parameter_count=count_trainable_parameters(model),
                initial_validation_loss=1.0,
                best_loss=1.0,
                best_epoch=0,
                best_state=_clone_state(model),
                history=[],
                optimizer_steps=0,
                heartbeat=None,
            )
            restored_model, restored_optimizer = self._model_optimizer()
            with self.assertRaises(Stage7D13TrainingError):
                _load_snapshot(
                    root=root,
                    specialist="notehead",
                    repository_sha="b" * 40,
                    profile=profile,
                    model=restored_model,
                    optimizer=restored_optimizer,
                    heartbeat=None,
                )

    def test_resume_step_boundary_is_epoch_atomic(self) -> None:
        self.assertEqual(D13_EXPECTED_OPTIMIZER_STEPS["notehead"] // 10, 615)


if __name__ == "__main__":
    unittest.main()
