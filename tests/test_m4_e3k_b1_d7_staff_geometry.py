from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from st_omr_training.m4_e3k_b1_d7_staff_geometry import (
    D7_DENSE_THRESHOLD,
    EXPECTED_D7_CHECKPOINT_SHA256,
    EXPECTED_D7_STAFF_STATE_SHA256,
    M4E3KB1GeometryError,
    decode_d7_staff_geometry,
    load_frozen_d7_staff_model,
    predict_d7_staff_geometry,
)
from st_omr_training.stage7d7_specialist_training import (
    FROZEN_D7_CONFIG,
    STAFF_CHANNELS,
    Stage7D7Record,
)


class M4E3KB1D7StaffGeometryTests(unittest.TestCase):
    def _blank_probabilities(self) -> torch.Tensor:
        return torch.full(
            (
                len(STAFF_CHANNELS),
                FROZEN_D7_CONFIG.input_height,
                FROZEN_D7_CONFIG.input_width,
            ),
            0.01,
            dtype=torch.float32,
        )

    def _add_staff(
        self,
        probabilities: torch.Tensor,
        *,
        x0: int,
        x1: int,
        first_y: int,
        spacing: int,
        slope_per_x: float = 0.0,
    ) -> None:
        region_index = STAFF_CHANNELS.index("staff_region")
        line_index = STAFF_CHANNELS.index("staff_lines")
        y0 = max(0, first_y - 2)
        y1 = min(
            FROZEN_D7_CONFIG.input_height - 1,
            first_y + 4 * spacing + 2,
        )
        probabilities[region_index, y0 : y1 + 1, x0 : x1 + 1] = 0.95
        reference_x = (x0 + x1) / 2.0
        for x in range(x0 + 4, x1 - 3):
            for line_index_number in range(5):
                y = round(
                    first_y
                    + line_index_number * spacing
                    + slope_per_x * (x - reference_x)
                )
                if 0 <= y < FROZEN_D7_CONFIG.input_height:
                    probabilities[line_index, y, x] = 0.99

    def test_decodes_two_equal_spaced_staffs_in_original_page_coordinates(self) -> None:
        probabilities = self._blank_probabilities()
        self._add_staff(probabilities, x0=40, x1=470, first_y=12, spacing=3)
        self._add_staff(probabilities, x0=45, x1=465, first_y=58, spacing=4)

        decoded = decode_d7_staff_geometry(
            probabilities,
            original_width=1024,
            original_height=192,
        )

        self.assertEqual(len(decoded), 2)
        self.assertEqual(len(decoded[0].five_staff_lines), 5)
        self.assertEqual(len(decoded[1].five_staff_lines), 5)
        self.assertAlmostEqual(decoded[0].staff_spacing, 6.0, places=6)
        self.assertAlmostEqual(decoded[1].staff_spacing, 8.0, places=6)
        self.assertLess(decoded[0].staff_bbox["y_min"], decoded[1].staff_bbox["y_min"])
        self.assertAlmostEqual(decoded[0].model_staff_slope, 0.0, places=6)

    def test_sloped_staff_is_deskewed_before_five_line_template_search(self) -> None:
        probabilities = self._blank_probabilities()
        self._add_staff(
            probabilities,
            x0=60,
            x1=450,
            first_y=30,
            spacing=4,
            slope_per_x=0.02,
        )
        decoded = decode_d7_staff_geometry(
            probabilities,
            original_width=1024,
            original_height=384,
        )
        self.assertEqual(len(decoded), 1)
        self.assertEqual(len(decoded[0].five_staff_lines), 5)
        self.assertLess(abs(decoded[0].model_staff_slope - 0.02), 0.02)
        self.assertGreater(decoded[0].line_template_score, 0.0)

    def test_small_staff_region_blob_is_rejected_by_frozen_component_gate(self) -> None:
        probabilities = self._blank_probabilities()
        region_index = STAFF_CHANNELS.index("staff_region")
        line_index = STAFF_CHANNELS.index("staff_lines")
        probabilities[region_index, 10:15, 10:20] = 0.99
        probabilities[line_index, 10:15, 10:20] = 0.99
        decoded = decode_d7_staff_geometry(
            probabilities,
            original_width=1024,
            original_height=192,
        )
        self.assertEqual(decoded, ())

    def test_region_without_five_line_support_is_not_invented(self) -> None:
        probabilities = self._blank_probabilities()
        region_index = STAFF_CHANNELS.index("staff_region")
        probabilities[region_index, 20:45, 40:470] = 0.99
        decoded = decode_d7_staff_geometry(
            probabilities,
            original_width=1024,
            original_height=192,
        )
        self.assertEqual(decoded, ())

    def test_decoder_rejects_wrong_shape_and_nonfinite_probabilities(self) -> None:
        with self.assertRaises(M4E3KB1GeometryError):
            decode_d7_staff_geometry(
                torch.zeros((2, 95, 512), dtype=torch.float32),
                original_width=1000,
                original_height=1000,
            )
        probabilities = self._blank_probabilities()
        probabilities[0, 0, 0] = float("nan")
        with self.assertRaises(M4E3KB1GeometryError):
            decode_d7_staff_geometry(
                probabilities,
                original_width=1000,
                original_height=1000,
            )

    def test_checkpoint_loader_rejects_any_nonaccepted_bytes_before_torch_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(b"not-the-accepted-checkpoint")
            with self.assertRaisesRegex(M4E3KB1GeometryError, "SHA-256"):
                load_frozen_d7_staff_model(path)

    def test_predictor_rejects_non_train_record_before_image_access(self) -> None:
        record = Stage7D7Record(
            sample_id="0" * 64,
            family_id="family",
            split="validation",
            png_sha256="1" * 64,
            label_sha256="2" * 64,
            image_path=Path("/definitely/not/read.png"),
            label_path=Path("/definitely/not/read.json"),
        )
        with self.assertRaisesRegex(M4E3KB1GeometryError, "TRAIN"):
            predict_d7_staff_geometry(object(), record, {"image": {"width": 1, "height": 1}})

    def test_frozen_checkpoint_and_threshold_constants(self) -> None:
        self.assertEqual(D7_DENSE_THRESHOLD, 0.50)
        self.assertEqual(
            EXPECTED_D7_CHECKPOINT_SHA256,
            "5f009ca8ba68d38497a7dd25590d4dd98c537f20c5d5525bf66e288afbf417dc",
        )
        self.assertEqual(
            EXPECTED_D7_STAFF_STATE_SHA256,
            "3131548548521229e6acd6fee8cffc66081cb54125645f9eff5a488de7603af8",
        )


if __name__ == "__main__":
    unittest.main()
