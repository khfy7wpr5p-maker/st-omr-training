from __future__ import annotations

from io import BytesIO
import unittest

from PIL import Image
import torch

from st_omr_training.dataset_manifest import DatasetSplit
from st_omr_training.training_data import (
    InputPreprocessConfig,
    TrainingBatch,
    TrainingDataError,
    load_training_samples,
    preprocess_config_fingerprint,
    preprocess_grayscale_png,
)


def png_bytes(mode: str = "L", size: tuple[int, int] = (100, 50)) -> bytes:
    image = Image.new(mode, size, color=0 if mode == "L" else (0, 0, 0))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class PreprocessTests(unittest.TestCase):
    def test_fit_pad_is_deterministic_finite_and_bounded(self) -> None:
        data = png_bytes()
        first = preprocess_grayscale_png(data)
        second = preprocess_grayscale_png(data)
        self.assertEqual(tuple(first.shape), (1, 64, 512))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)
        # Source is not upscaled; white padding must remain present.
        self.assertEqual(float(first[0, 0, 0]), 1.0)

    def test_rgb_png_is_rejected_not_silently_converted(self) -> None:
        with self.assertRaises(TrainingDataError):
            preprocess_grayscale_png(png_bytes(mode="RGB"))

    def test_manifest_dimension_mismatch_is_rejected(self) -> None:
        with self.assertRaises(TrainingDataError):
            preprocess_grayscale_png(
                png_bytes(),
                expected_width=101,
                expected_height=50,
            )

    def test_preprocess_config_is_strict_and_fingerprinted(self) -> None:
        config = InputPreprocessConfig()
        self.assertEqual(preprocess_config_fingerprint(config), preprocess_config_fingerprint(config))
        with self.assertRaises(TrainingDataError):
            InputPreprocessConfig(target_height=True)


class SplitSafetyTests(unittest.TestCase):
    def test_test_split_is_rejected_before_build_or_path_access(self) -> None:
        with self.assertRaises(TrainingDataError):
            load_training_samples(None, "/definitely/not/a/dataset", DatasetSplit.TEST)

    def test_training_batch_rejects_test_split(self) -> None:
        images = torch.zeros((1, 1, 64, 512), dtype=torch.float32)
        decoder = torch.tensor([[1, 3]], dtype=torch.long)
        labels = torch.tensor([[3, 2]], dtype=torch.long)
        with self.assertRaises(TrainingDataError):
            TrainingBatch(
                images=images,
                decoder_input_ids=decoder,
                labels=labels,
                split=DatasetSplit.TEST,
            )

    def test_training_batch_rejects_nan_and_out_of_range_tokens(self) -> None:
        images = torch.zeros((1, 1, 64, 512), dtype=torch.float32)
        images[0, 0, 0, 0] = float("nan")
        decoder = torch.tensor([[1]], dtype=torch.long)
        labels = torch.tensor([[2]], dtype=torch.long)
        with self.assertRaises(TrainingDataError):
            TrainingBatch(images, decoder, labels, DatasetSplit.TRAIN)

        images = torch.zeros((1, 1, 64, 512), dtype=torch.float32)
        with self.assertRaises(TrainingDataError):
            TrainingBatch(
                images,
                torch.tensor([[999]], dtype=torch.long),
                labels,
                DatasetSplit.TRAIN,
            )


if __name__ == "__main__":
    unittest.main()
