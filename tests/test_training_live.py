from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
import unittest

from st_omr_training.dataset_builder import (
    SyntheticDatasetConfig,
    build_synthetic_dataset,
    write_synthetic_dataset,
)
from st_omr_training.dataset_manifest import DatasetSplit
from st_omr_training.training_data import (
    TrainingDataError,
    load_training_samples,
    make_training_batch,
)
from st_omr_training.training_model import TrainerConfig, run_deterministic_cpu_smoke


class RealStage7BSmokePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temp.name) / "dataset"
        cls.build = build_synthetic_dataset(
            SyntheticDatasetConfig(
                family_count=3,
                seed_start=31_000,
                split_seed=17,
                measure_count=1,
                raster_width=512,
                family_profiles=("time-4-4", "note-only", "rest-only"),
                degradation_profiles=("clean",),
            )
        )
        write_synthetic_dataset(cls.build, cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_persisted_stage6_train_and_validation_cross_stage7b_gate(self) -> None:
        train = load_training_samples(self.build, self.root, DatasetSplit.TRAIN, max_samples=1)
        validation = load_training_samples(
            self.build,
            self.root,
            DatasetSplit.VALIDATION,
            max_samples=1,
        )
        self.assertEqual(len(train), 1)
        self.assertEqual(len(validation), 1)
        self.assertNotEqual(train[0].family_id, validation[0].family_id)
        self.assertGreater(len(train[0].target_token_ids), 2)

    def test_real_train_artifact_smoke_replays_exactly(self) -> None:
        train = load_training_samples(self.build, self.root, DatasetSplit.TRAIN, max_samples=1)
        batch = make_training_batch(train)
        config = TrainerConfig(master_seed=12_345, smoke_steps=1)
        first = run_deterministic_cpu_smoke(batch, trainer_config=config)
        second = run_deterministic_cpu_smoke(batch, trainer_config=config)
        self.assertEqual(first, second)

    def test_persisted_image_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / "copy"
            shutil.copytree(self.root, copied)
            train_sample = next(
                sample
                for sample in self.build.manifest.samples
                if sample.split is DatasetSplit.TRAIN
            )
            image_path = copied / "images" / f"{train_sample.png_sha256}.png"
            data = bytearray(image_path.read_bytes())
            data[-1] ^= 0x01
            image_path.write_bytes(bytes(data))
            with self.assertRaises(TrainingDataError):
                load_training_samples(self.build, copied, DatasetSplit.TRAIN, max_samples=1)


if __name__ == "__main__":
    unittest.main()
