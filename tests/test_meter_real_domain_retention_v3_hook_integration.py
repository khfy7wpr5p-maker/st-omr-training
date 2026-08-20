from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from st_omr_training.meter_real_domain_adaptation_v2 import FROZEN_ADAPTATION_CONFIG_V2
from st_omr_training.meter_real_domain_retention_v3 import (
    run_meter_real_domain_retention_v3,
)


class MeterRealDomainRetentionV3HookIntegrationTests(unittest.TestCase):
    def test_scoped_hook_applies_exact_midpoint_schedule_and_writes_receipt(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in this test runtime")

        captured: dict[str, object] = {}

        class Handle:
            removed = False

            def remove(self) -> None:
                self.removed = True

        handle = Handle()

        def fake_register(hook):
            captured["hook"] = hook
            return handle

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def fake_v2_run(**kwargs):
                self.assertEqual(kwargs["config"], FROZEN_ADAPTATION_CONFIG_V2)
                progress = kwargs["progress"]
                progress(
                    "phase_started",
                    {
                        "phase": "training_and_validation",
                        "phase_index": 6,
                        "phase_total": 7,
                        "completed_epoch": 0,
                        "epochs_total": 20,
                        "batches_per_epoch": 30,
                    },
                )
                parameter = torch.nn.Parameter(torch.tensor([1.0]))
                optimizer = torch.optim.AdamW([parameter], lr=0.001)
                observed_rates: dict[int, float] = {}
                hook = captured["hook"]
                for step in range(1, 601):
                    parameter.grad = torch.ones_like(parameter)
                    hook(optimizer, (), {})
                    if step in {1, 300, 301, 600}:
                        observed_rates[step] = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                captured["rates"] = observed_rates
                metrics = {
                    "adaptation_version": "meter-real-domain-adaptation-v2",
                    "run_id": "a" * 64,
                    "repository_sha": "b" * 40,
                    "profile_fingerprint": "c" * 64,
                    "configuration": {
                        "epochs": 20,
                        "learning_rate_micros": 1000,
                    },
                    "optimizer_steps": 600,
                    "status": "HOLD_NO_ACCEPTED_CANDIDATE",
                    "test_opened": False,
                    "runtime_connected": False,
                    "resolver_connected": False,
                    "production_promotion_authorized": False,
                    "history": [{"epoch": epoch} for epoch in range(1, 21)],
                    "best": {"epoch": 13},
                }
                (root / "metrics-fake.json").write_text(
                    json.dumps(metrics, sort_keys=True, separators=(",", ":")),
                    encoding="ascii",
                )
                return metrics

            with patch(
                "torch.optim.optimizer.register_optimizer_step_pre_hook",
                side_effect=fake_register,
            ), patch(
                "st_omr_training.meter_real_domain_adaptation_v2.run_meter_real_domain_adaptation_v2",
                side_effect=fake_v2_run,
            ):
                result = run_meter_real_domain_retention_v3(
                    teacher_bundle_root=root / "teacher",
                    d10_root=root / "d10",
                    base_checkpoint_path=root / "d11.pt",
                    output_root=root,
                    repository_root=root / "repo",
                    expected_d10_manifest_sha256="d" * 64,
                    expected_d10_artifact_binding_sha256="e" * 64,
                    progress=lambda event, payload: None,
                    resume=False,
                )

            self.assertTrue(handle.removed)
            self.assertEqual(
                captured["rates"],
                {1: 0.001, 300: 0.001, 301: 0.00025, 600: 0.00025},
            )
            self.assertEqual(result["retention_version"], "meter-real-domain-retention-v3")
            self.assertEqual(result["retention_schedule"]["midpoint_decay_epoch"], 11)
            receipt = root / result["retention_receipt_filename"]
            self.assertTrue(receipt.is_file())
            payload = json.loads(receipt.read_text("ascii"))
            self.assertEqual(payload["schedule"], result["retention_schedule"])
            self.assertEqual(payload["underlying_run_id"], "a" * 64)
            self.assertFalse(payload["test_opened"])
            self.assertFalse(payload["runtime_connected"])
            self.assertFalse(payload["production_promotion_authorized"])


if __name__ == "__main__":
    unittest.main()
