from __future__ import annotations

import inspect
import unittest
from PIL import Image, ImageDraw

import st_omr_training.meter_v2_runtime_slot_localizer_v1 as localizer
from st_omr_training.meter_v2_runtime_slot_localizer_v1 import (
    M3C2_ANCHOR_MODES_SHA256,
    M4A_DATASET_MANIFEST_SHA256,
    MODE_DATA_SHA256,
    MeterV2RuntimeSlotError,
    meter_v2_runtime_slot_profile_fingerprint,
    propose_meter_v2_runtime_digit_modes_v1,
    resolver_connection_allowed,
    runtime_digit_bbox_localization_accepted,
    runtime_slot_policy_train_derived_only,
)


class MeterV2RuntimeSlotLocalizerV1Tests(unittest.TestCase):
    @staticmethod
    def _ink_image() -> Image.Image:
        image = Image.new("L", (256, 192), 255)
        draw = ImageDraw.Draw(image)
        # TRAIN-derived low/high Meter anchor neighborhoods.
        draw.rectangle((42, 58, 56, 135), fill=0)
        draw.rectangle((108, 50, 125, 145), fill=0)
        return image

    def test_frozen_measure_mode_inventory(self) -> None:
        image = self._ink_image()
        expected = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2}
        for measure_number, count in expected.items():
            with self.subTest(measure_number=measure_number):
                proposals = propose_meter_v2_runtime_digit_modes_v1(
                    image, measure_number=measure_number
                )
                self.assertEqual(len(proposals), count)
                self.assertEqual(
                    tuple(item.mode_index for item in proposals), tuple(range(count))
                )

    def test_blank_image_produces_no_candidate_modes(self) -> None:
        blank = Image.new("L", (256, 192), 255)
        for measure_number in range(1, 9):
            self.assertEqual(
                propose_meter_v2_runtime_digit_modes_v1(
                    blank, measure_number=measure_number
                ),
                (),
            )

    def test_output_is_10_of_10_deterministic(self) -> None:
        image = self._ink_image()
        reports = [
            propose_meter_v2_runtime_digit_modes_v1(image, measure_number=6)
            for _ in range(10)
        ]
        self.assertEqual(len(set(reports)), 1)

    def test_boxes_are_valid_and_stacked(self) -> None:
        proposal = propose_meter_v2_runtime_digit_modes_v1(
            self._ink_image(), measure_number=2
        )[0]
        upper = proposal.numerator_bbox
        lower = proposal.denominator_bbox
        self.assertLess((upper.y0 + upper.y1) / 2, (lower.y0 + lower.y1) / 2)
        for box in (upper, lower):
            self.assertGreaterEqual(box.x0, 0)
            self.assertGreaterEqual(box.y0, 0)
            self.assertLessEqual(box.x1, 256)
            self.assertLessEqual(box.y1, 192)
            self.assertLess(box.x0, box.x1)
            self.assertLess(box.y0, box.y1)

    def test_low_support_modes_remain_marked(self) -> None:
        image = self._ink_image()
        m4 = propose_meter_v2_runtime_digit_modes_v1(image, measure_number=4)
        m8 = propose_meter_v2_runtime_digit_modes_v1(image, measure_number=8)
        self.assertFalse(m4[0].low_support)
        self.assertTrue(m4[1].low_support)
        self.assertFalse(m8[0].low_support)
        self.assertTrue(m8[1].low_support)
        self.assertLess(m4[1].train_count, 30)
        self.assertLess(m8[1].train_count, 30)

    def test_malformed_input_fails_closed(self) -> None:
        with self.assertRaises(MeterV2RuntimeSlotError):
            propose_meter_v2_runtime_digit_modes_v1("not-image", measure_number=1)  # type: ignore[arg-type]
        with self.assertRaises(MeterV2RuntimeSlotError):
            propose_meter_v2_runtime_digit_modes_v1(
                Image.new("L", (128, 128), 255), measure_number=1
            )
        for value in (0, 9, True, 1.5):
            with self.subTest(measure_number=value):
                with self.assertRaises(MeterV2RuntimeSlotError):
                    propose_meter_v2_runtime_digit_modes_v1(
                        Image.new("L", (256, 192), 255),
                        measure_number=value,  # type: ignore[arg-type]
                    )

    def test_provenance_profile_and_isolation_are_frozen(self) -> None:
        self.assertEqual(
            M3C2_ANCHOR_MODES_SHA256,
            "f8594e1550027ed2f69670a030ec8a6a7d8247c1b07a5000b111038208b31745",
        )
        self.assertEqual(
            M4A_DATASET_MANIFEST_SHA256,
            "ebda40dae10f0d6490df2c7728dab5cc2cc6f58b5420b198dfbb441a99ecebb9",
        )
        self.assertEqual(
            MODE_DATA_SHA256,
            "19bb1e639225d79a2c085ac1d0fd45455ad4010be1099e11f6ec0ed31477650b",
        )
        self.assertEqual(len(meter_v2_runtime_slot_profile_fingerprint()), 64)
        self.assertTrue(runtime_slot_policy_train_derived_only())
        self.assertFalse(runtime_digit_bbox_localization_accepted())
        self.assertFalse(resolver_connection_allowed())
        source = inspect.getsource(localizer)
        self.assertNotIn("import torch", source)
        self.assertNotIn("import numpy", source)
        self.assertNotIn("runtime_deterministic_resolver", source)
        self.assertNotIn("stage7d11_barline_meter_training", source)


if __name__ == "__main__":
    unittest.main()
