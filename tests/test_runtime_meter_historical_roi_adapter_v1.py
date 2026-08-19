from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import unittest

from PIL import Image

from st_omr_training.runtime_geometry_engine_contract import (
    BoxContract, LineSegmentContract, MeasureProposalContract, PageGeometryContract,
    Point2DContract, StaffGeometryContract, SystemGeometryContract,
)
from st_omr_training.runtime_meter_historical_roi_adapter_v1 import (
    HistoricalMeterRoiError, historical_meter_roi_profile_fingerprint_v1,
    reconstruct_historical_meter_roi_v1,
)
from st_omr_training.runtime_page_normalizer_contract import HomographyContract


def _source() -> bytes:
    image = Image.new("L", (400, 220), 255)
    for x in range(400):
        for y in range(220):
            if (x + y) % 17 == 0:
                image.putpixel((x, y), 0)
    out = BytesIO(); image.save(out, format="PNG", optimize=False, compress_level=9)
    return out.getvalue()


def _geometry(raw: bytes) -> PageGeometryContract:
    lines = tuple(LineSegmentContract(Point2DContract(20, y), Point2DContract(380, y)) for y in (80, 90, 100, 110, 120))
    staff = StaffGeometryContract("s1", "sys1", lines, BoxContract(20, 78, 380, 122), 10.0)
    system = SystemGeometryContract("sys1", BoxContract(10, 50, 390, 150), ("s1",))
    measure = MeasureProposalContract(
        "m1", "sys1", "s1", BoxContract(100, 60, 300, 145),
        LineSegmentContract(Point2DContract(100, 60), Point2DContract(100, 145)),
        LineSegmentContract(Point2DContract(300, 60), Point2DContract(300, 145)),
        "accepted",
    )
    identity = HomographyContract(forward=(1,0,0,0,1,0,0,0,1), inverse=(1,0,0,0,1,0,0,0,1))
    return PageGeometryContract(
        normalized_image_sha256=sha256(raw).hexdigest(), geometry_config_fingerprint=sha256(b"g").hexdigest(),
        page_width=400, page_height=220, transform=identity, systems=(system,), staffs=(staff,),
        measure_proposals=(measure,), status="accepted",
    )


class HistoricalMeterRoiAdapterV1Tests(unittest.TestCase):
    def test_reconstructs_frozen_d10_geometry_and_output_shape(self) -> None:
        raw = _source(); artifact = reconstruct_historical_meter_roi_v1(raw, _geometry(raw), measure_id="m1")
        self.assertEqual(artifact.crop_box, (95, 48, 220, 152))
        with Image.open(BytesIO(artifact.png_bytes)) as image:
            self.assertEqual(image.mode, "L")
            self.assertEqual(image.size, (256, 192))
        self.assertEqual(artifact.profile_fingerprint, historical_meter_roi_profile_fingerprint_v1())

    def test_replay_is_byte_deterministic_10_of_10(self) -> None:
        raw = _source(); geometry = _geometry(raw)
        outputs = [reconstruct_historical_meter_roi_v1(raw, geometry, measure_id="m1") for _ in range(10)]
        self.assertEqual(len({item.image_sha256 for item in outputs}), 1)
        self.assertEqual(len({item.png_bytes for item in outputs}), 1)

    def test_wrong_source_sha_fails_closed(self) -> None:
        raw = _source(); geometry = _geometry(raw)
        with self.assertRaises(HistoricalMeterRoiError):
            reconstruct_historical_meter_roi_v1(raw + b"x", geometry, measure_id="m1")


if __name__ == "__main__":
    unittest.main()
