from __future__ import annotations

from hashlib import sha256
import unittest

from st_omr_training.meter_v2_digit_crop_adapter_v1 import meter_v2_digit_crop_profile_fingerprint
from st_omr_training.runtime_geometry_engine_contract import BoxContract
from st_omr_training.runtime_local_roi_v1 import RuntimeRoiArtifact
from st_omr_training.runtime_meter_input_provenance_v1 import (
    DigitSlotGeometryV1,
    HISTORICAL_PRESENCE_ROI_PROFILE,
    MeterRuntimeInputProvenanceError,
    bind_runtime_meter_inputs_v1,
    digit_slot_provenance_required_v1,
    real_presence_inference_allowed_from_runtime_roi_v1,
)
from st_omr_training.runtime_page_normalizer_contract import HomographyContract


def _roi() -> RuntimeRoiArtifact:
    data = b"runtime-measure-start"
    return RuntimeRoiArtifact(
        roi_id="m1:measure-start", kind="measure-start", measure_id="m1", staff_id="s1",
        source_image_sha256=sha256(b"source").hexdigest(), roi_image_sha256=sha256(data).hexdigest(),
        crop_bbox=BoxContract(10.0, 20.0, 80.0, 100.0),
        source_to_roi=HomographyContract(forward=(1,0,-10,0,1,-20,0,0,1), inverse=(1,0,10,0,1,20,0,0,1)),
        png_bytes=data,
    )


class MeterRuntimeInputProvenanceV1Tests(unittest.TestCase):
    def test_slot_fingerprint_binds_both_boxes_and_crop_profile(self) -> None:
        slots = DigitSlotGeometryV1((1, 2, 10, 12), (1, 15, 10, 25), "localizer-sha")
        changed = DigitSlotGeometryV1((2, 2, 10, 12), (1, 15, 10, 25), "localizer-sha")
        self.assertNotEqual(slots.fingerprint(), changed.fingerprint())
        self.assertEqual(len(meter_v2_digit_crop_profile_fingerprint()), 64)

    def test_current_runtime_roi_cannot_masquerade_as_historical_presence_crop(self) -> None:
        slots = DigitSlotGeometryV1((1, 2, 10, 12), (1, 15, 10, 25), "localizer-sha")
        with self.assertRaises(MeterRuntimeInputProvenanceError):
            bind_runtime_meter_inputs_v1(
                _roi(), presence_input_profile=HISTORICAL_PRESENCE_ROI_PROFILE, digit_slots=slots
            )

    def test_real_inference_gate_stays_closed(self) -> None:
        self.assertFalse(real_presence_inference_allowed_from_runtime_roi_v1())
        self.assertTrue(digit_slot_provenance_required_v1())


if __name__ == "__main__":
    unittest.main()
