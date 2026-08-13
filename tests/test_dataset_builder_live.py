import tempfile
import unittest
from pathlib import Path

from st_omr_training.dataset_builder import (
    DatasetBuildInputError,
    SyntheticDatasetConfig,
    build_metadata_bytes,
    build_synthetic_dataset,
    write_synthetic_dataset,
)
from st_omr_training.dataset_manifest import (
    DatasetSplit,
    canonical_manifest_bytes,
    validate_dataset_manifest,
)


class RealSyntheticDatasetPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SyntheticDatasetConfig(
            dataset_name="st-stage6-live",
            dataset_version="v1",
            family_count=3,
            seed_start=9100,
            split_seed=17,
            measure_count=2,
            raster_width=512,
            family_profiles=("note-only", "rest-only", "chord-only"),
            degradation_profiles=("clean",),
        )
        cls.build_progress = []
        cls.first = build_synthetic_dataset(
            cls.config,
            progress=cls.build_progress.append,
        )
        cls.second = build_synthetic_dataset(cls.config)

    def test_real_stage1_through_stage6_build_passes_independent_manifest_gate(self):
        result = validate_dataset_manifest(self.first.manifest)
        self.assertTrue(result.is_valid, result.issues)
        self.assertEqual(len(self.first.targets), 3)
        self.assertEqual(len(self.first.images), 3)
        self.assertEqual(len(self.first.manifest.samples), 3)
        observed = {
            split: {
                sample.family_id
                for sample in self.first.manifest.samples
                if sample.split is split
            }
            for split in DatasetSplit
        }
        self.assertEqual(len(observed[DatasetSplit.TRAIN]), 1)
        self.assertEqual(len(observed[DatasetSplit.VALIDATION]), 1)
        self.assertEqual(len(observed[DatasetSplit.TEST]), 1)
        self.assertEqual(
            [event["event"] for event in self.build_progress],
            [
                "dataset_family_completed",
                "dataset_family_completed",
                "dataset_family_completed",
                "dataset_validation_completed",
            ],
        )
        self.assertEqual(self.build_progress[-1]["samples_total"], 3)

    def test_same_config_rebuild_is_byte_and_identity_deterministic(self):
        self.assertEqual(self.first.config_fingerprint, self.second.config_fingerprint)
        self.assertEqual(self.first.manifest_sha256, self.second.manifest_sha256)
        self.assertEqual(self.first.build_id, self.second.build_id)
        self.assertEqual(
            canonical_manifest_bytes(self.first.manifest),
            canonical_manifest_bytes(self.second.manifest),
        )
        self.assertEqual(self.first.targets, self.second.targets)
        self.assertEqual(self.first.images, self.second.images)
        self.assertEqual(build_metadata_bytes(self.first), build_metadata_bytes(self.second))

    def test_hash_addressed_writer_is_verified_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as parent:
            root = Path(parent) / "dataset-v1"
            progress = []
            written = write_synthetic_dataset(
                self.first,
                root,
                progress=progress.append,
            )
            self.assertEqual(written, root)
            self.assertEqual(
                (root / "manifest.json").read_bytes(),
                canonical_manifest_bytes(self.first.manifest),
            )
            self.assertEqual(
                (root / "manifest.sha256").read_text(encoding="ascii"),
                f"{self.first.manifest_sha256}  manifest.json\n",
            )
            self.assertEqual((root / "build.json").read_bytes(), build_metadata_bytes(self.first))
            for target in self.first.targets:
                self.assertEqual(
                    (root / "targets" / f"{target.sha256}.musicxml").read_bytes(),
                    target.musicxml,
                )
            for image in self.first.images:
                self.assertEqual(
                    (root / "images" / f"{image.sha256}.png").read_bytes(),
                    image.png,
                )
            self.assertEqual(
                [event["event"] for event in progress],
                [
                    "dataset_target_written",
                    "dataset_image_written",
                    "dataset_persisted",
                ],
            )
            with self.assertRaises(DatasetBuildInputError):
                write_synthetic_dataset(self.first, root)


if __name__ == "__main__":
    unittest.main()
