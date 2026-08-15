from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import torch

from st_omr_training.stage7d13_run_verifier import verify_stage7d13_run
from st_omr_training.stage7d13_symbol_models import (
    SpecialistMetrics,
    build_symbol_model,
    model_profile_fingerprint,
)
from st_omr_training.stage7d13_symbol_training_contract import SPECIALIST_CLASSES
from st_omr_training.stage7d13_training import (
    STAGE7D13_TRAINING_VERSION,
    training_profile_fingerprint,
)
from st_omr_training.stage7d13_verified_surface import (
    D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
    D13_DERIVATIVE_BUILD_ID,
    D13_DERIVATIVE_MANIFEST_SHA256,
    D13_EXPECTED_OPTIMIZER_STEPS,
    D13_EXPECTED_OPTIMIZER_STEPS_TOTAL,
    D13_RECORD_SPLIT_COUNTS,
)
from st_omr_training.training_model import count_trainable_parameters, model_state_sha256


def canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


class Stage7D13RunVerifierTests(unittest.TestCase):
    def _write_run(self, root: Path) -> None:
        checkpoint = {}
        specialist_rows = {}
        for specialist in SPECIALIST_CLASSES:
            model = build_symbol_model(specialist)
            checkpoint[specialist] = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            metrics = SpecialistMetrics(0.0, 0.0, 0.0)
            history = [
                {
                    "epoch": epoch,
                    "train_loss": 2.0 - epoch * 0.05,
                    "validation_loss": 1.0 - epoch * 0.05,
                    "center_f1_4px": 0.0,
                    "bbox_f1_iou50": 0.0,
                    "macro_class_f1": 0.0,
                }
                for epoch in range(1, 11)
            ]
            specialist_rows[specialist] = {
                "parameter_count": count_trainable_parameters(model),
                "model_fingerprint": model_profile_fingerprint(specialist),
                "final_state_sha256": model_state_sha256(model),
                "optimizer_steps": D13_EXPECTED_OPTIMIZER_STEPS[specialist],
                "best_epoch": 10,
                "initial_validation_loss": 1.5,
                "final_validation_loss": 0.5,
                "metrics": asdict(metrics),
                "accepted": False,
                "history": history,
            }

        checkpoint_path = root / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        checkpoint_sha = sha256(checkpoint_path.read_bytes()).hexdigest()
        run_id = "a" * 64
        metrics_payload = {
            "version": STAGE7D13_TRAINING_VERSION,
            "run_id": run_id,
            "training_profile_fingerprint": training_profile_fingerprint(),
            "specialists": specialist_rows,
            "acceptance": False,
        }
        metrics_raw = canonical(metrics_payload)
        (root / "metrics.json").write_bytes(metrics_raw)
        run_payload = {
            "version": STAGE7D13_TRAINING_VERSION,
            "run_id": run_id,
            "repository_sha": "b" * 40,
            "repository_origin": "https://github.com/khfy7wpr5p-maker/st-omr-training.git",
            "training_profile_fingerprint": training_profile_fingerprint(),
            "derivative": {
                "build_id": D13_DERIVATIVE_BUILD_ID,
                "manifest_sha256": D13_DERIVATIVE_MANIFEST_SHA256,
                "artifact_binding_sha256": D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
                "train_records": D13_RECORD_SPLIT_COUNTS["train"],
                "validation_records": D13_RECORD_SPLIT_COUNTS["validation"],
                "test_records": 0,
            },
            "optimizer_steps": D13_EXPECTED_OPTIMIZER_STEPS,
            "optimizer_steps_total": D13_EXPECTED_OPTIMIZER_STEPS_TOTAL,
            "checkpoint_sha256": checkpoint_sha,
            "metrics_sha256": sha256(metrics_raw).hexdigest(),
            "test_opened": False,
            "complete_marker_written": False,
        }
        (root / "run.json").write_bytes(canonical(run_payload))

    def test_uncompleted_run_reopens_with_safe_checkpoint_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root)
            receipt = verify_stage7d13_run(root)
            self.assertTrue(receipt.verification_passed)
            self.assertFalse(receipt.acceptance)
            self.assertFalse(receipt.test_opened)
            self.assertEqual(receipt.optimizer_steps_total, 18450)
            self.assertEqual(receipt.best_epochs, {"notehead": 10, "rest": 10, "accidental": 10})

    def test_tampered_checkpoint_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root)
            with (root / "checkpoint.pt").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaises(Exception):
                verify_stage7d13_run(root)


if __name__ == "__main__":
    unittest.main()
