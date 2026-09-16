from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from st_omr_training.poly_2d_quality_training import Poly2DQualityTrainingConfig
from st_omr_training.poly_evaluation_contract import (
    BenchmarkSampleDescriptor,
    PolyphonicComplexityProfile,
    RobustnessBucket,
)
from st_omr_training.poly_v2_dataset_materialization import persist_native_poly_v2_dataset
from st_omr_training.poly_v2_dataset_reload import load_and_verify_native_poly_v2_dataset
from st_omr_training.poly_v2_experiment_recipe import (
    FIRST_BASELINE_BENCHMARK_ID,
    FIRST_BASELINE_BENCHMARK_VERSION,
    PolyV2ExperimentRecipeError,
    freeze_native_poly_v2_experiment_recipe,
)

from test_poly_v2_dataset_reload import _build


_REPOSITORY_SHA = "a" * 40


def _descriptor(loaded):
    sample = next(
        item for item in loaded.build.manifest.samples if item.split.value == "validation"
    )
    return BenchmarkSampleDescriptor(
        sample_id=sample.sample_id,
        family_id=sample.family_id,
        split="validation",
        complexity=PolyphonicComplexityProfile(
            voice_count=4,
            staff_count=1,
            simultaneous_note_density=1.0,
            chord_density=0.25,
            overlap_density=1.0,
            tie_density=0.0,
            beam_complexity=0.0,
            rhythmic_complexity=0.25,
            tuplet_present=False,
            grace_present=False,
            cross_staff_present=False,
        ),
        robustness_bucket=RobustnessBucket.CLEAN,
    )


class PolyV2ExperimentRecipeTests(unittest.TestCase):
    def _loaded(self, parent: Path):
        build = _build()
        root = persist_native_poly_v2_dataset(build, parent / "dataset")
        return load_and_verify_native_poly_v2_dataset(root)

    def test_freezes_exact_full_population_recipe_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loaded = self._loaded(Path(directory))
            descriptor = _descriptor(loaded)
            first = freeze_native_poly_v2_experiment_recipe(
                loaded=loaded,
                repository_sha=_REPOSITORY_SHA,
                descriptors=(descriptor,),
            )
            second = freeze_native_poly_v2_experiment_recipe(
                loaded=loaded,
                repository_sha=_REPOSITORY_SHA,
                descriptors=(descriptor,),
            )

            self.assertEqual(first.fingerprint(), second.fingerprint())
            self.assertEqual(first.dataset_manifest_sha256, loaded.build.manifest_sha256)
            self.assertEqual(first.dataset_build_id, loaded.build.build_id)
            self.assertEqual(first.preflight_receipt_sha256, loaded.receipt.fingerprint())
            self.assertEqual(first.benchmark_id, FIRST_BASELINE_BENCHMARK_ID)
            self.assertEqual(first.benchmark_version, FIRST_BASELINE_BENCHMARK_VERSION)
            self.assertEqual(first.epochs, 8)
            self.assertEqual(first.max_optimizer_steps, 8192)
            self.assertEqual(first.batch_size, 8)
            self.assertEqual(first.required_optimizer_steps, 8)
            self.assertTrue(first.full_train_population)
            self.assertTrue(first.full_validation_population)
            self.assertFalse(first.test_artifact_bytes_accessed)
            self.assertFalse(first.production_authority)
            self.assertGreaterEqual(first.max_decode_steps, 1)

    def test_descriptor_set_must_cover_exact_validation_population(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loaded = self._loaded(Path(directory))
            with self.assertRaisesRegex(
                PolyV2ExperimentRecipeError,
                "exact VALIDATION population",
            ):
                freeze_native_poly_v2_experiment_recipe(
                    loaded=loaded,
                    repository_sha=_REPOSITORY_SHA,
                    descriptors=(),
                )

    def test_decode_bound_may_not_truncate_validation_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loaded = self._loaded(Path(directory))
            descriptor = _descriptor(loaded)
            validation = next(
                item for item in loaded.build.manifest.samples if item.split.value == "validation"
            )
            minimum = validation.target_token_count - 1
            self.assertGreater(minimum, 1)
            with self.assertRaisesRegex(
                PolyV2ExperimentRecipeError,
                "truncate an admitted VALIDATION target",
            ):
                freeze_native_poly_v2_experiment_recipe(
                    loaded=loaded,
                    repository_sha=_REPOSITORY_SHA,
                    descriptors=(descriptor,),
                    max_decode_steps=minimum - 1,
                )

    def test_optimizer_step_ceiling_must_cover_every_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loaded = self._loaded(Path(directory))
            descriptor = _descriptor(loaded)
            too_small = Poly2DQualityTrainingConfig(
                epochs=2,
                max_optimizer_steps=1,
            )
            with self.assertRaisesRegex(
                PolyV2ExperimentRecipeError,
                "cannot cover every TRAIN batch",
            ):
                freeze_native_poly_v2_experiment_recipe(
                    loaded=loaded,
                    repository_sha=_REPOSITORY_SHA,
                    descriptors=(descriptor,),
                    quality_config=too_small,
                )

    def test_benchmark_descriptor_metadata_changes_recipe_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loaded = self._loaded(Path(directory))
            clean = _descriptor(loaded)
            scan = BenchmarkSampleDescriptor(
                sample_id=clean.sample_id,
                family_id=clean.family_id,
                split=clean.split,
                complexity=clean.complexity,
                robustness_bucket=RobustnessBucket.SCAN,
            )
            first = freeze_native_poly_v2_experiment_recipe(
                loaded=loaded,
                repository_sha=_REPOSITORY_SHA,
                descriptors=(clean,),
            )
            second = freeze_native_poly_v2_experiment_recipe(
                loaded=loaded,
                repository_sha=_REPOSITORY_SHA,
                descriptors=(scan,),
            )
            self.assertNotEqual(
                first.validation_descriptor_manifest_sha256,
                second.validation_descriptor_manifest_sha256,
            )
            self.assertNotEqual(first.benchmark_identity_sha256, second.benchmark_identity_sha256)
            self.assertNotEqual(first.fingerprint(), second.fingerprint())


if __name__ == "__main__":
    unittest.main()
