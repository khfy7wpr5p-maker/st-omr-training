from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from st_omr_training.poly_2d_quality_training import (
    FROZEN_POLY_2D_QUALITY_CONFIG,
    poly_2d_quality_trainer_fingerprint,
)
from st_omr_training.poly_2d_transformer import (
    FROZEN_POLY_2D_CONFIG,
    poly_2d_config_fingerprint,
)
from st_omr_training.poly_v2_baseline_start_gate import (
    NativePolyV2BaselineStartError,
    NativePolyV2BaselineStartPermit,
    authorize_native_poly_v2_baseline_start,
    execute_authorized_native_poly_v2_quality_training,
)
from st_omr_training.poly_v2_dataset_materialization import (
    NativePolyV2DatasetBuild,
    native_poly_v2_materialization_fingerprint,
)
from st_omr_training.poly_v2_dataset_reload import (
    LoadedNativePolyV2Dataset,
    NativePolyV2DatasetPreflightReceipt,
)
from st_omr_training.poly_v2_experiment_recipe import NativePolyV2ExperimentRecipe
from st_omr_training.poly_v2_quality_execution import (
    NativePolyV2QualityExecutionResult,
    poly_v2_quality_execution_profile_fingerprint,
)
from st_omr_training.poly_v2_real_corpus_admission import (
    NativePolyV2RealCorpusAdmissionReceipt,
)


def h(ch: str) -> str:
    return ch * 64


REPOSITORY_SHA = "1" * 40
TRAIN_IDS = (h("a"), h("b"))
VALIDATION_IDS = (h("c"),)
DATASET_MANIFEST_SHA = h("d")
DATASET_BUILD_ID = h("e")
PREFLIGHT_SHA = h("f")
ADMISSION_SHA = h("1")
RECIPE_SHA = h("2")


def make_evidence():
    loaded = Mock(spec=LoadedNativePolyV2Dataset)
    build = Mock(spec=NativePolyV2DatasetBuild)
    build.manifest_sha256 = DATASET_MANIFEST_SHA
    build.build_id = DATASET_BUILD_ID
    preflight = Mock(spec=NativePolyV2DatasetPreflightReceipt)
    preflight.train_sample_ids = TRAIN_IDS
    preflight.validation_sample_ids = VALIDATION_IDS
    preflight.test_artifact_bytes_accessed = False
    preflight.production_authority = False
    preflight.fingerprint.return_value = PREFLIGHT_SHA
    loaded.build = build
    loaded.receipt = preflight

    admission = Mock(spec=NativePolyV2RealCorpusAdmissionReceipt)
    admission.dataset_manifest_sha256 = DATASET_MANIFEST_SHA
    admission.dataset_build_id = DATASET_BUILD_ID
    admission.b8a_preflight_receipt_sha256 = PREFLIGHT_SHA
    admission.train_native_sample_ids = TRAIN_IDS
    admission.validation_native_sample_ids = VALIDATION_IDS
    admission.quality_training_eligible = True
    admission.test_artifact_bytes_accessed = False
    admission.production_authority = False
    admission.commercial_use_authority = False
    admission.fingerprint.return_value = ADMISSION_SHA

    recipe = Mock(spec=NativePolyV2ExperimentRecipe)
    recipe.repository_sha = REPOSITORY_SHA
    recipe.dataset_manifest_sha256 = DATASET_MANIFEST_SHA
    recipe.dataset_build_id = DATASET_BUILD_ID
    recipe.preflight_receipt_sha256 = PREFLIGHT_SHA
    recipe.train_sample_ids = TRAIN_IDS
    recipe.validation_sample_ids = VALIDATION_IDS
    recipe.train_batch_sizes = (2,)
    recipe.validation_batch_sizes = (1,)
    recipe.batch_size = 8
    recipe.epochs = FROZEN_POLY_2D_QUALITY_CONFIG.epochs
    recipe.max_optimizer_steps = FROZEN_POLY_2D_QUALITY_CONFIG.max_optimizer_steps
    recipe.required_optimizer_steps = FROZEN_POLY_2D_QUALITY_CONFIG.epochs
    recipe.model_profile_sha256 = poly_2d_config_fingerprint(FROZEN_POLY_2D_CONFIG)
    recipe.quality_trainer_profile_sha256 = poly_2d_quality_trainer_fingerprint(
        FROZEN_POLY_2D_QUALITY_CONFIG,
        FROZEN_POLY_2D_CONFIG,
    )
    recipe.materialization_fingerprint_sha256 = native_poly_v2_materialization_fingerprint(
        FROZEN_POLY_2D_CONFIG
    )
    recipe.quality_execution_profile_sha256 = poly_v2_quality_execution_profile_fingerprint(
        quality_config=FROZEN_POLY_2D_QUALITY_CONFIG,
        model_config=FROZEN_POLY_2D_CONFIG,
        batch_size=recipe.batch_size,
    )
    recipe.full_train_population = True
    recipe.full_validation_population = True
    recipe.test_artifact_bytes_accessed = False
    recipe.production_authority = False
    recipe.fingerprint.return_value = RECIPE_SHA
    return loaded, admission, recipe


class PolyV2BaselineStartGateTests(unittest.TestCase):
    def test_authorizes_exact_b8a_b8i_b8b_identity_chain(self) -> None:
        loaded, admission, recipe = make_evidence()
        permit = authorize_native_poly_v2_baseline_start(
            loaded=loaded,
            admission=admission,
            recipe=recipe,
            repository_sha=REPOSITORY_SHA,
        )
        self.assertIsInstance(permit, NativePolyV2BaselineStartPermit)
        self.assertEqual(permit.train_sample_ids, TRAIN_IDS)
        self.assertEqual(permit.validation_sample_ids, VALIDATION_IDS)
        self.assertEqual(permit.b8a_preflight_receipt_sha256, PREFLIGHT_SHA)
        self.assertEqual(permit.b8i_admission_receipt_sha256, ADMISSION_SHA)
        self.assertEqual(permit.b8b_recipe_sha256, RECIPE_SHA)
        self.assertFalse(permit.test_artifact_bytes_accessed)
        self.assertFalse(permit.production_authority)
        self.assertFalse(permit.commercial_use_authority)
        self.assertEqual(len(permit.fingerprint()), 64)

    def test_rejects_swapped_or_stale_admission_identity(self) -> None:
        loaded, admission, recipe = make_evidence()
        admission.b8a_preflight_receipt_sha256 = h("0")
        with self.assertRaisesRegex(NativePolyV2BaselineStartError, "B8I B8A receipt"):
            authorize_native_poly_v2_baseline_start(
                loaded=loaded,
                admission=admission,
                recipe=recipe,
                repository_sha=REPOSITORY_SHA,
            )

    def test_rejects_population_drift_between_admission_and_recipe(self) -> None:
        loaded, admission, recipe = make_evidence()
        recipe.train_sample_ids = (TRAIN_IDS[0],)
        with self.assertRaisesRegex(NativePolyV2BaselineStartError, "B8B TRAIN population"):
            authorize_native_poly_v2_baseline_start(
                loaded=loaded,
                admission=admission,
                recipe=recipe,
                repository_sha=REPOSITORY_SHA,
            )

    def test_rejects_any_upstream_test_access_or_authority_escalation(self) -> None:
        cases = (
            ("preflight-test", lambda loaded, admission, recipe: setattr(loaded.receipt, "test_artifact_bytes_accessed", True)),
            ("admission-test", lambda loaded, admission, recipe: setattr(admission, "test_artifact_bytes_accessed", True)),
            ("recipe-test", lambda loaded, admission, recipe: setattr(recipe, "test_artifact_bytes_accessed", True)),
            ("production", lambda loaded, admission, recipe: setattr(admission, "production_authority", True)),
            ("commercial", lambda loaded, admission, recipe: setattr(admission, "commercial_use_authority", True)),
        )
        for label, mutate in cases:
            with self.subTest(label=label):
                loaded, admission, recipe = make_evidence()
                mutate(loaded, admission, recipe)
                with self.assertRaises(NativePolyV2BaselineStartError):
                    authorize_native_poly_v2_baseline_start(
                        loaded=loaded,
                        admission=admission,
                        recipe=recipe,
                        repository_sha=REPOSITORY_SHA,
                    )

    def test_authorized_wrapper_forces_full_population_and_binds_result(self) -> None:
        loaded, admission, recipe = make_evidence()
        permit = authorize_native_poly_v2_baseline_start(
            loaded=loaded,
            admission=admission,
            recipe=recipe,
            repository_sha=REPOSITORY_SHA,
        )
        result = Mock(spec=NativePolyV2QualityExecutionResult)
        result.dataset_manifest_sha256 = DATASET_MANIFEST_SHA
        result.dataset_build_id = DATASET_BUILD_ID
        result.materialization_fingerprint_sha256 = recipe.materialization_fingerprint_sha256
        result.execution_profile_sha256 = recipe.quality_execution_profile_sha256
        result.quality_trainer_profile_sha256 = recipe.quality_trainer_profile_sha256
        result.train_sample_ids = TRAIN_IDS
        result.validation_sample_ids = VALIDATION_IDS
        result.train_batch_sizes = recipe.train_batch_sizes
        result.validation_batch_sizes = recipe.validation_batch_sizes
        result.optimizer_steps = recipe.required_optimizer_steps
        result.test_split_accessed = False
        result.production_authority = False
        result.fingerprint.return_value = h("3")

        with patch(
            "st_omr_training.poly_v2_baseline_start_gate.execute_native_poly_v2_quality_training",
            return_value=result,
        ) as execute:
            authorized = execute_authorized_native_poly_v2_quality_training(
                loaded=loaded,
                admission=admission,
                recipe=recipe,
                permit=permit,
                dataset_root=Path("/tmp/native-v2"),
                output_directory=Path("/tmp/output"),
            )

        execute.assert_called_once()
        call = execute.call_args.kwargs
        self.assertIsNone(call["max_train_samples"])
        self.assertIsNone(call["max_validation_samples"])
        self.assertEqual(call["batch_size"], recipe.batch_size)
        self.assertEqual(call["repository_sha"], REPOSITORY_SHA)
        self.assertEqual(authorized.quality_execution_result, result)
        self.assertEqual(authorized.permit_fingerprint_sha256, permit.fingerprint())
        self.assertEqual(len(authorized.fingerprint()), 64)

    def test_authorized_wrapper_rejects_tampered_permit_before_b6(self) -> None:
        loaded, admission, recipe = make_evidence()
        permit = authorize_native_poly_v2_baseline_start(
            loaded=loaded,
            admission=admission,
            recipe=recipe,
            repository_sha=REPOSITORY_SHA,
        )
        tampered = replace(permit, b8i_admission_receipt_sha256=h("4"))
        with patch(
            "st_omr_training.poly_v2_baseline_start_gate.execute_native_poly_v2_quality_training"
        ) as execute:
            with self.assertRaisesRegex(NativePolyV2BaselineStartError, "permit differs"):
                execute_authorized_native_poly_v2_quality_training(
                    loaded=loaded,
                    admission=admission,
                    recipe=recipe,
                    permit=tampered,
                    dataset_root=Path("/tmp/native-v2"),
                    output_directory=Path("/tmp/output"),
                )
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
