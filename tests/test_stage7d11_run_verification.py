from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import torch

from st_omr_training.stage7d11_barline_meter_training import (
    EXPECTED_D10_REPOSITORY_SHA,
    EXPECTED_D7_STRUCTURE_STATE_SHA256,
    FROZEN_D11_CONFIG,
    STAGE7D11_METRICS_SCHEMA,
    STAGE7D11_VERIFICATION_SCHEMA,
    STAGE7D11_VERSION,
    Stage7D11TrainingError,
    build_barline_refiner,
    build_meter_refiner,
    stage7d11_profile_fingerprint,
)
from st_omr_training.stage7d11_run_verification import (
    EXPECTED_OPTIMIZER_STEPS_PER_REFINER,
    _expected_run_id,
    verify_stage7d11_run,
)
from st_omr_training.stage7d9_structure_refinement_contract import D9_ACCEPTANCE
from st_omr_training.training_model import count_trainable_parameters, model_state_sha256


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


class Stage7D11PersistedVerificationTests(unittest.TestCase):
    REPOSITORY_SHA = "1" * 40
    MANIFEST_SHA = "2" * 64
    BINDING_SHA = "3" * 64

    def _fixture(self, root: Path) -> Path:
        run_id = _expected_run_id(self.REPOSITORY_SHA, self.MANIFEST_SHA, self.BINDING_SHA)
        run = root / run_id
        run.mkdir()

        barline_model = build_barline_refiner(FROZEN_D11_CONFIG)
        meter_model = build_meter_refiner(FROZEN_D11_CONFIG)
        barline_state = model_state_sha256(barline_model)
        meter_state = model_state_sha256(meter_model)

        checkpoint_tmp = run / "checkpoint.tmp.pt"
        torch.save(
            {
                "barline_state_dict": {name: value.detach().cpu().clone() for name, value in barline_model.state_dict().items()},
                "meter_state_dict": {name: value.detach().cpu().clone() for name, value in meter_model.state_dict().items()},
            },
            checkpoint_tmp,
        )
        checkpoint_sha = sha256(checkpoint_tmp.read_bytes()).hexdigest()
        checkpoint = run / f"checkpoint-{checkpoint_sha}.pt"
        checkpoint_tmp.rename(checkpoint)

        metrics = {
            "schema_version": STAGE7D11_METRICS_SCHEMA,
            "stage7d11_version": STAGE7D11_VERSION,
            "repository_sha": self.REPOSITORY_SHA,
            "profile_fingerprint": stage7d11_profile_fingerprint(FROZEN_D11_CONFIG),
            "d10": {
                "repository_sha": EXPECTED_D10_REPOSITORY_SHA,
                "manifest_sha256": self.MANIFEST_SHA,
                "artifact_binding_sha256": self.BINDING_SHA,
                "roi_records": 22_128,
                "test_records": 0,
            },
            "accepted_d7_structure_state_sha256": EXPECTED_D7_STRUCTURE_STATE_SHA256,
            "accepted_d7_structure_core_loaded": False,
            "barline": {
                "task": "barline",
                "optimizer_steps": EXPECTED_OPTIMIZER_STEPS_PER_REFINER,
                "state_sha256": barline_state,
                "parameter_count": count_trainable_parameters(barline_model),
                "validation_metrics": {"strict_dice": 0.60, "tolerant_f1_2px": 0.80},
            },
            "meter": {
                "task": "meter",
                "optimizer_steps": EXPECTED_OPTIMIZER_STEPS_PER_REFINER,
                "state_sha256": meter_state,
                "parameter_count": count_trainable_parameters(meter_model),
                "validation_metrics": {"macro_f1": 0.85, "positive_localization_f1_2px": 0.70},
            },
            "acceptance_thresholds": asdict(D9_ACCEPTANCE),
            "acceptance_passed": True,
            "checkpoint": {"filename": checkpoint.name, "sha256": checkpoint_sha},
            "sealed_test_split_opened": False,
        }
        metrics_raw = _canonical(metrics)
        metrics_sha = sha256(metrics_raw).hexdigest()
        metrics_path = run / f"metrics-{metrics_sha}.json"
        metrics_path.write_bytes(metrics_raw)

        verification = {
            "schema_version": STAGE7D11_VERIFICATION_SCHEMA,
            "stage7d11_version": STAGE7D11_VERSION,
            "repository_sha": self.REPOSITORY_SHA,
            "profile_fingerprint": stage7d11_profile_fingerprint(FROZEN_D11_CONFIG),
            "d10_manifest_sha256": self.MANIFEST_SHA,
            "d10_artifact_binding_sha256": self.BINDING_SHA,
            "metrics_sha256": metrics_sha,
            "checkpoint_sha256": checkpoint_sha,
            "barline_state_sha256": barline_state,
            "meter_state_sha256": meter_state,
            "barline_optimizer_steps": EXPECTED_OPTIMIZER_STEPS_PER_REFINER,
            "meter_optimizer_steps": EXPECTED_OPTIMIZER_STEPS_PER_REFINER,
            "train_records": 19_680,
            "validation_records": 2_448,
            "test_records": 0,
            "test_opened": False,
            "accepted_d7_structure_core_loaded": False,
            "accepted_d7_structure_core_mutated": False,
            "checkpoint_reload_verified": True,
            "repository_stable_during_run": True,
            "runtime_stable_during_run": True,
            "acceptance_passed": True,
        }
        verification_raw = _canonical(verification)
        verification_sha = sha256(verification_raw).hexdigest()
        verification_path = run / f"verification-{verification_sha}.json"
        verification_path.write_bytes(verification_raw)

        (run / "COMPLETE").write_bytes(
            (
                f"{verification_sha}  {verification_path.name}\n"
                f"{metrics_sha}  {metrics_path.name}\n"
                f"{checkpoint_sha}  {checkpoint.name}\n"
            ).encode("ascii")
        )
        return run

    def test_independent_verifier_reopens_complete_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._fixture(Path(tmp))
            receipt = verify_stage7d11_run(
                run,
                expected_repository_sha=self.REPOSITORY_SHA,
                expected_d10_manifest_sha256=self.MANIFEST_SHA,
                expected_d10_artifact_binding_sha256=self.BINDING_SHA,
            )
            self.assertTrue(receipt.acceptance_passed)
            self.assertEqual(receipt.barline_optimizer_steps, 2464)
            self.assertEqual(receipt.meter_optimizer_steps, 2464)
            self.assertEqual(receipt.test_records, 0)
            self.assertFalse(receipt.core_loaded)
            self.assertFalse(receipt.core_mutated)

    def test_complete_marker_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._fixture(Path(tmp))
            (run / "COMPLETE").write_bytes(b"tampered\n")
            with self.assertRaisesRegex(Stage7D11TrainingError, "COMPLETE"):
                verify_stage7d11_run(
                    run,
                    expected_repository_sha=self.REPOSITORY_SHA,
                    expected_d10_manifest_sha256=self.MANIFEST_SHA,
                    expected_d10_artifact_binding_sha256=self.BINDING_SHA,
                )

    def test_unexpected_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._fixture(Path(tmp))
            (run / "extra.bin").write_bytes(b"x")
            with self.assertRaisesRegex(Stage7D11TrainingError, "unexpected"):
                verify_stage7d11_run(
                    run,
                    expected_repository_sha=self.REPOSITORY_SHA,
                    expected_d10_manifest_sha256=self.MANIFEST_SHA,
                    expected_d10_artifact_binding_sha256=self.BINDING_SHA,
                )


if __name__ == "__main__":
    unittest.main()
