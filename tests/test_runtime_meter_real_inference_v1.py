from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import unittest

from PIL import Image
import torch

from st_omr_training.runtime_meter_historical_roi_adapter_v1 import (
    HistoricalMeterRoiArtifactV1,
    historical_meter_roi_profile_fingerprint_v1,
)
from st_omr_training.runtime_meter_real_checkpoint_audit_v1 import (
    AuditedCheckpointStateV1,
    PRESENCE_D11_SHA256,
)
from st_omr_training.runtime_meter_real_inference_v1 import (
    RuntimeD11PresenceBridge,
    digit_specialist_execution_allowed_in_this_stage,
    infer_presence_from_audited_state_v1,
    meter_composition_allowed_in_this_stage,
    resolver_connection_allowed_in_this_stage,
)


def _artifact() -> HistoricalMeterRoiArtifactV1:
    image = Image.new("L", (256, 192), 255)
    for y in range(55, 138):
        image.putpixel((80, y), 0)
        image.putpixel((104, y), 0)
    out = BytesIO(); image.save(out, format="PNG", optimize=False, compress_level=9)
    raw = out.getvalue()
    return HistoricalMeterRoiArtifactV1(
        measure_id="m1", staff_id="s1", source_image_sha256=sha256(b"source").hexdigest(),
        image_sha256=sha256(raw).hexdigest(), crop_box=(10, 20, 210, 170),
        resized_size=(256, 192), pad_left=0, pad_top=0,
        profile_fingerprint=historical_meter_roi_profile_fingerprint_v1(), png_bytes=raw,
    )


def _audited() -> AuditedCheckpointStateV1:
    torch.manual_seed(12345)
    model = RuntimeD11PresenceBridge().cpu()
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    return AuditedCheckpointStateV1(
        role="presence-d11-bridge", checkpoint_sha256=PRESENCE_D11_SHA256,
        byte_length=1234, model_state=state,
    )


class RealMeterInferenceV1Tests(unittest.TestCase):
    def test_output_is_finite_and_identity_bound(self) -> None:
        result = infer_presence_from_audited_state_v1(_artifact(), _audited())
        self.assertEqual(result.measure_id, "m1")
        self.assertEqual(result.staff_id, "s1")
        self.assertEqual(result.checkpoint_sha256, PRESENCE_D11_SHA256)
        self.assertAlmostEqual(sum(result.class_probabilities), 1.0, places=6)
        self.assertAlmostEqual(result.presence_score, 1.0 - result.class_probabilities[0], places=7)
        self.assertEqual(len(result.inference_fingerprint), 64)

    def test_replay_is_deterministic_10_of_10(self) -> None:
        artifact = _artifact(); audited = _audited()
        outputs = [infer_presence_from_audited_state_v1(artifact, audited) for _ in range(10)]
        self.assertEqual(len({item.inference_fingerprint for item in outputs}), 1)
        self.assertEqual(len({item.class_probabilities for item in outputs}), 1)
        self.assertEqual(len({item.predicted_bbox_normalized for item in outputs}), 1)

    def test_wrong_role_fails_closed(self) -> None:
        audited = _audited()
        wrong = AuditedCheckpointStateV1("digit-2", audited.checkpoint_sha256, audited.byte_length, audited.model_state)
        with self.assertRaises(Exception):
            infer_presence_from_audited_state_v1(_artifact(), wrong)

    def test_later_gates_remain_closed(self) -> None:
        self.assertFalse(digit_specialist_execution_allowed_in_this_stage())
        self.assertFalse(meter_composition_allowed_in_this_stage())
        self.assertFalse(resolver_connection_allowed_in_this_stage())


if __name__ == "__main__":
    unittest.main()
