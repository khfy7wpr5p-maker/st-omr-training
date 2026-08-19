from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from st_omr_training.runtime_meter_real_checkpoint_audit_v1 import (
    DIGIT2_SHA256,
    MeterRealCheckpointAuditError,
    _D11_METER_STATE_SHAPES,
    _DIGIT_STATE_SHAPES,
    audit_digit_checkpoint_v1,
    conservative_probability_to_milli_v1,
    resolver_connection_allowed,
    runtime_real_inference_promoted,
    training_or_threshold_tuning_allowed,
)


def _state(shapes: dict[str, tuple[int, ...]]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros(shape, dtype=torch.float32) for name, shape in shapes.items()}


class RuntimeMeterRealCheckpointAuditV1Tests(unittest.TestCase):
    def test_conservative_milli_does_not_round_up_subthreshold_score(self) -> None:
        self.assertEqual(conservative_probability_to_milli_v1(0.4799), 479)
        self.assertEqual(conservative_probability_to_milli_v1(0.48), 480)
        self.assertEqual(conservative_probability_to_milli_v1(0.5999), 599)
        self.assertEqual(conservative_probability_to_milli_v1(0.60), 600)
        self.assertEqual(conservative_probability_to_milli_v1(0.4699), 469)
        self.assertEqual(conservative_probability_to_milli_v1(0.47), 470)

    def test_digit_state_shape_contract_is_exact(self) -> None:
        self.assertEqual(_DIGIT_STATE_SHAPES["features.0.weight"], (16, 1, 3, 3))
        self.assertEqual(_DIGIT_STATE_SHAPES["features.3.weight"], (32, 16, 3, 3))
        self.assertEqual(_DIGIT_STATE_SHAPES["features.6.weight"], (64, 32, 3, 3))
        self.assertEqual(_DIGIT_STATE_SHAPES["head.weight"], (1, 64))

    def test_d11_meter_state_shape_contract_is_exact(self) -> None:
        self.assertEqual(_D11_METER_STATE_SHAPES["projection.1.weight"], (64, 1152))
        self.assertEqual(_D11_METER_STATE_SHAPES["classifier.weight"], (4, 64))
        self.assertEqual(_D11_METER_STATE_SHAPES["bbox.weight"], (4, 64))

    def test_wrong_sha_fails_closed_before_state_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "digit.pt"
            torch.save({"model_state_dict": _state(_DIGIT_STATE_SHAPES)}, path)
            with self.assertRaises(MeterRealCheckpointAuditError):
                audit_digit_checkpoint_v1(
                    path,
                    role="digit-2",
                    expected_sha256=DIGIT2_SHA256,
                )

    def test_symlink_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target.pt"
            target.write_bytes(b"not-a-real-checkpoint")
            link = root / "link.pt"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable on this platform")
            with self.assertRaises(MeterRealCheckpointAuditError):
                audit_digit_checkpoint_v1(
                    link,
                    role="digit-2",
                    expected_sha256=DIGIT2_SHA256,
                )

    def test_nonfinite_probability_is_rejected(self) -> None:
        for value in (float("nan"), float("inf"), -0.01, 1.01):
            with self.assertRaises(ValueError):
                conservative_probability_to_milli_v1(value)

    def test_checkpoint_audit_does_not_authorize_training_resolver_or_promotion(self) -> None:
        self.assertFalse(training_or_threshold_tuning_allowed())
        self.assertFalse(resolver_connection_allowed())
        self.assertFalse(runtime_real_inference_promoted())


if __name__ == "__main__":
    unittest.main()
