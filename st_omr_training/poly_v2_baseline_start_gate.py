"""Fail-closed start authorization for the first real Native Polyphonic V2 baseline.

TR-POLY-09B8J binds the already-frozen B8A dataset preflight, B8I real-corpus
admission, and B8B experiment recipe before B6 is allowed to start the first
real quality-training run.  The authorized execution wrapper always uses the
full frozen TRAIN/VALIDATION population; prefix/subsample execution is not
exposed through this path.

This module does not discover or admit data, does not open TEST, and grants no
production or commercial-use authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from pathlib import Path
from typing import Final

from .poly_2d_quality_training import (
    FROZEN_POLY_2D_QUALITY_CONFIG,
    Poly2DQualityTrainingConfig,
    poly_2d_quality_trainer_fingerprint,
)
from .poly_2d_transformer import (
    FROZEN_POLY_2D_CONFIG,
    Poly2DTransformerConfig,
    poly_2d_config_fingerprint,
)
from .poly_v2_dataset_materialization import native_poly_v2_materialization_fingerprint
from .poly_v2_dataset_reload import LoadedNativePolyV2Dataset
from .poly_v2_experiment_recipe import NativePolyV2ExperimentRecipe
from .poly_v2_quality_execution import (
    NativePolyV2QualityExecutionResult,
    execute_native_poly_v2_quality_training,
    poly_v2_quality_execution_profile_fingerprint,
)
from .poly_v2_real_corpus_admission import NativePolyV2RealCorpusAdmissionReceipt


POLY_V2_BASELINE_START_GATE_VERSION: Final[str] = "st-omr-native-poly-v2-baseline-start-gate-v1"
_GIT_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class NativePolyV2BaselineStartError(ValueError):
    """Raised when B8A/B8I/B8B evidence cannot authorize the first real run."""


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise NativePolyV2BaselineStartError(
            "B8J evidence is not canonical-JSON serializable"
        ) from exc


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise NativePolyV2BaselineStartError(f"{name} must be lowercase SHA-256 text")
    return value


def _require_git_sha(name: str, value: object) -> str:
    if not isinstance(value, str) or _GIT_SHA40.fullmatch(value) is None:
        raise NativePolyV2BaselineStartError(f"{name} must be lowercase git SHA-40")
    return value


@dataclass(frozen=True, slots=True)
class NativePolyV2BaselineStartPermit:
    repository_sha: str
    dataset_manifest_sha256: str
    dataset_build_id: str
    b8a_preflight_receipt_sha256: str
    b8i_admission_receipt_sha256: str
    b8b_recipe_sha256: str
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    train_batch_sizes: tuple[int, ...]
    validation_batch_sizes: tuple[int, ...]
    batch_size: int
    epochs: int
    max_optimizer_steps: int
    required_optimizer_steps: int
    model_profile_sha256: str
    quality_trainer_profile_sha256: str
    materialization_fingerprint_sha256: str
    quality_execution_profile_sha256: str
    gate_version: str = POLY_V2_BASELINE_START_GATE_VERSION
    full_train_population: bool = True
    full_validation_population: bool = True
    test_artifact_bytes_accessed: bool = False
    production_authority: bool = False
    commercial_use_authority: bool = False

    def __post_init__(self) -> None:
        _require_git_sha("repository_sha", self.repository_sha)
        for name in (
            "dataset_manifest_sha256",
            "dataset_build_id",
            "b8a_preflight_receipt_sha256",
            "b8i_admission_receipt_sha256",
            "b8b_recipe_sha256",
            "model_profile_sha256",
            "quality_trainer_profile_sha256",
            "materialization_fingerprint_sha256",
            "quality_execution_profile_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in ("train_sample_ids", "validation_sample_ids"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise NativePolyV2BaselineStartError(f"{name} must be a non-empty immutable tuple")
            if tuple(sorted(set(values))) != values:
                raise NativePolyV2BaselineStartError(f"{name} must be sorted and unique")
            for value in values:
                _require_sha256(name, value)
        if set(self.train_sample_ids) & set(self.validation_sample_ids):
            raise NativePolyV2BaselineStartError("B8J TRAIN/VALIDATION sample leakage detected")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise NativePolyV2BaselineStartError("B8J batch_size must be a positive integer")
        for name in ("epochs", "max_optimizer_steps", "required_optimizer_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise NativePolyV2BaselineStartError(f"B8J {name} must be a positive integer")
        if self.required_optimizer_steps > self.max_optimizer_steps:
            raise NativePolyV2BaselineStartError("B8J optimizer-step ceiling cannot cover the frozen run")
        for sizes_name, ids_name in (
            ("train_batch_sizes", "train_sample_ids"),
            ("validation_batch_sizes", "validation_sample_ids"),
        ):
            sizes = getattr(self, sizes_name)
            ids = getattr(self, ids_name)
            if not isinstance(sizes, tuple) or not sizes or sum(sizes) != len(ids):
                raise NativePolyV2BaselineStartError(f"{sizes_name} does not cover its full population")
            if any(
                not isinstance(size, int)
                or isinstance(size, bool)
                or not 1 <= size <= self.batch_size
                for size in sizes
            ):
                raise NativePolyV2BaselineStartError(f"{sizes_name} contains an invalid batch size")
        if self.required_optimizer_steps != len(self.train_batch_sizes) * self.epochs:
            raise NativePolyV2BaselineStartError("B8J optimizer-step evidence differs from the frozen batch plan")
        if self.gate_version != POLY_V2_BASELINE_START_GATE_VERSION:
            raise NativePolyV2BaselineStartError("unsupported B8J start-gate version")
        if not self.full_train_population or not self.full_validation_population:
            raise NativePolyV2BaselineStartError("B8J may authorize full-population execution only")
        if self.test_artifact_bytes_accessed or self.production_authority or self.commercial_use_authority:
            raise NativePolyV2BaselineStartError(
                "B8J may not access TEST bytes or grant production/commercial authority"
            )

    def fingerprint(self) -> str:
        payload = asdict(self)
        for name in (
            "train_sample_ids",
            "validation_sample_ids",
            "train_batch_sizes",
            "validation_batch_sizes",
        ):
            payload[name] = list(payload[name])
        return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class AuthorizedNativePolyV2QualityExecution:
    permit_fingerprint_sha256: str
    admission_fingerprint_sha256: str
    recipe_fingerprint_sha256: str
    quality_execution_result: NativePolyV2QualityExecutionResult
    gate_version: str = POLY_V2_BASELINE_START_GATE_VERSION
    test_artifact_bytes_accessed: bool = False
    production_authority: bool = False
    commercial_use_authority: bool = False

    def __post_init__(self) -> None:
        for name in (
            "permit_fingerprint_sha256",
            "admission_fingerprint_sha256",
            "recipe_fingerprint_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if not isinstance(self.quality_execution_result, NativePolyV2QualityExecutionResult):
            raise NativePolyV2BaselineStartError("authorized execution requires a B6 quality result")
        if self.gate_version != POLY_V2_BASELINE_START_GATE_VERSION:
            raise NativePolyV2BaselineStartError("authorized execution gate version mismatch")
        if self.test_artifact_bytes_accessed or self.production_authority or self.commercial_use_authority:
            raise NativePolyV2BaselineStartError(
                "authorized B8J execution may not claim TEST access or production/commercial authority"
            )

    def fingerprint(self) -> str:
        payload = {
            "gate_version": self.gate_version,
            "permit_fingerprint_sha256": self.permit_fingerprint_sha256,
            "admission_fingerprint_sha256": self.admission_fingerprint_sha256,
            "recipe_fingerprint_sha256": self.recipe_fingerprint_sha256,
            "quality_execution_result_sha256": self.quality_execution_result.fingerprint(),
            "test_artifact_bytes_accessed": self.test_artifact_bytes_accessed,
            "production_authority": self.production_authority,
            "commercial_use_authority": self.commercial_use_authority,
        }
        return sha256(_canonical_json_bytes(payload)).hexdigest()


def authorize_native_poly_v2_baseline_start(
    *,
    loaded: LoadedNativePolyV2Dataset,
    admission: NativePolyV2RealCorpusAdmissionReceipt,
    recipe: NativePolyV2ExperimentRecipe,
    repository_sha: str,
) -> NativePolyV2BaselineStartPermit:
    """Authorize the exact first real baseline without touching corpus or TEST bytes."""

    if not isinstance(loaded, LoadedNativePolyV2Dataset):
        raise TypeError("loaded must be LoadedNativePolyV2Dataset")
    if not isinstance(admission, NativePolyV2RealCorpusAdmissionReceipt):
        raise TypeError("admission must be NativePolyV2RealCorpusAdmissionReceipt")
    if not isinstance(recipe, NativePolyV2ExperimentRecipe):
        raise TypeError("recipe must be NativePolyV2ExperimentRecipe")
    _require_git_sha("repository_sha", repository_sha)

    build = loaded.build
    preflight = loaded.receipt
    preflight_sha = preflight.fingerprint()
    admission_sha = admission.fingerprint()
    recipe_sha = recipe.fingerprint()

    for actual, expected, label in (
        (admission.dataset_manifest_sha256, build.manifest_sha256, "B8I dataset manifest"),
        (recipe.dataset_manifest_sha256, build.manifest_sha256, "B8B dataset manifest"),
        (admission.dataset_build_id, build.build_id, "B8I dataset build"),
        (recipe.dataset_build_id, build.build_id, "B8B dataset build"),
        (admission.b8a_preflight_receipt_sha256, preflight_sha, "B8I B8A receipt"),
        (recipe.preflight_receipt_sha256, preflight_sha, "B8B B8A receipt"),
        (recipe.repository_sha, repository_sha, "B8B repository SHA"),
    ):
        if actual != expected:
            raise NativePolyV2BaselineStartError(f"{label} identity mismatch")

    train_ids = tuple(sorted(preflight.train_sample_ids))
    validation_ids = tuple(sorted(preflight.validation_sample_ids))
    if tuple(admission.train_native_sample_ids) != train_ids:
        raise NativePolyV2BaselineStartError("B8I TRAIN population differs from B8A")
    if tuple(admission.validation_native_sample_ids) != validation_ids:
        raise NativePolyV2BaselineStartError("B8I VALIDATION population differs from B8A")
    if tuple(recipe.train_sample_ids) != train_ids:
        raise NativePolyV2BaselineStartError("B8B TRAIN population differs from B8A/B8I")
    if tuple(recipe.validation_sample_ids) != validation_ids:
        raise NativePolyV2BaselineStartError("B8B VALIDATION population differs from B8A/B8I")
    if not admission.quality_training_eligible:
        raise NativePolyV2BaselineStartError("B8I did not grant quality-training eligibility")
    if not recipe.full_train_population or not recipe.full_validation_population:
        raise NativePolyV2BaselineStartError("B8B recipe is not a full-population first-baseline recipe")
    if preflight.test_artifact_bytes_accessed or admission.test_artifact_bytes_accessed or recipe.test_artifact_bytes_accessed:
        raise NativePolyV2BaselineStartError("TEST artifact bytes were accessed before B8J authorization")
    if preflight.production_authority or admission.production_authority or recipe.production_authority:
        raise NativePolyV2BaselineStartError("upstream evidence exceeded the research authority boundary")
    if admission.commercial_use_authority:
        raise NativePolyV2BaselineStartError("B8I admission unexpectedly granted commercial-use authority")

    return NativePolyV2BaselineStartPermit(
        repository_sha=repository_sha,
        dataset_manifest_sha256=build.manifest_sha256,
        dataset_build_id=build.build_id,
        b8a_preflight_receipt_sha256=preflight_sha,
        b8i_admission_receipt_sha256=admission_sha,
        b8b_recipe_sha256=recipe_sha,
        train_sample_ids=train_ids,
        validation_sample_ids=validation_ids,
        train_batch_sizes=recipe.train_batch_sizes,
        validation_batch_sizes=recipe.validation_batch_sizes,
        batch_size=recipe.batch_size,
        epochs=recipe.epochs,
        max_optimizer_steps=recipe.max_optimizer_steps,
        required_optimizer_steps=recipe.required_optimizer_steps,
        model_profile_sha256=recipe.model_profile_sha256,
        quality_trainer_profile_sha256=recipe.quality_trainer_profile_sha256,
        materialization_fingerprint_sha256=recipe.materialization_fingerprint_sha256,
        quality_execution_profile_sha256=recipe.quality_execution_profile_sha256,
    )


def execute_authorized_native_poly_v2_quality_training(
    *,
    loaded: LoadedNativePolyV2Dataset,
    admission: NativePolyV2RealCorpusAdmissionReceipt,
    recipe: NativePolyV2ExperimentRecipe,
    permit: NativePolyV2BaselineStartPermit,
    dataset_root: str | Path,
    output_directory: Path,
    quality_config: Poly2DQualityTrainingConfig = FROZEN_POLY_2D_QUALITY_CONFIG,
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
) -> AuthorizedNativePolyV2QualityExecution:
    """Run B6 only when exact B8A+B8I+B8B evidence authorizes the full baseline."""

    if not isinstance(permit, NativePolyV2BaselineStartPermit):
        raise TypeError("permit must be NativePolyV2BaselineStartPermit")
    expected = authorize_native_poly_v2_baseline_start(
        loaded=loaded,
        admission=admission,
        recipe=recipe,
        repository_sha=permit.repository_sha,
    )
    if permit.fingerprint() != expected.fingerprint():
        raise NativePolyV2BaselineStartError("B8J permit differs from current B8A/B8I/B8B evidence")
    if not isinstance(quality_config, Poly2DQualityTrainingConfig):
        raise TypeError("quality_config must be Poly2DQualityTrainingConfig")
    if not isinstance(model_config, Poly2DTransformerConfig):
        raise TypeError("model_config must be Poly2DTransformerConfig")

    if poly_2d_config_fingerprint(model_config) != recipe.model_profile_sha256:
        raise NativePolyV2BaselineStartError("runtime model config differs from the frozen B8B recipe")
    if poly_2d_quality_trainer_fingerprint(quality_config, model_config) != recipe.quality_trainer_profile_sha256:
        raise NativePolyV2BaselineStartError("runtime quality trainer config differs from the frozen B8B recipe")
    if native_poly_v2_materialization_fingerprint(model_config) != recipe.materialization_fingerprint_sha256:
        raise NativePolyV2BaselineStartError("runtime materialization profile differs from the frozen B8B recipe")
    if poly_v2_quality_execution_profile_fingerprint(
        quality_config=quality_config,
        model_config=model_config,
        batch_size=recipe.batch_size,
    ) != recipe.quality_execution_profile_sha256:
        raise NativePolyV2BaselineStartError("runtime B6 execution profile differs from the frozen B8B recipe")
    if quality_config.epochs != recipe.epochs or quality_config.max_optimizer_steps != recipe.max_optimizer_steps:
        raise NativePolyV2BaselineStartError("runtime epoch/optimizer budget differs from the frozen B8B recipe")

    result = execute_native_poly_v2_quality_training(
        build=loaded.build,
        dataset_root=dataset_root,
        repository_sha=recipe.repository_sha,
        output_directory=output_directory,
        quality_config=quality_config,
        model_config=model_config,
        batch_size=recipe.batch_size,
        max_train_samples=None,
        max_validation_samples=None,
    )
    for actual, expected_value, label in (
        (result.dataset_manifest_sha256, permit.dataset_manifest_sha256, "B6 dataset manifest"),
        (result.dataset_build_id, permit.dataset_build_id, "B6 dataset build"),
        (result.materialization_fingerprint_sha256, permit.materialization_fingerprint_sha256, "B6 materialization"),
        (result.execution_profile_sha256, permit.quality_execution_profile_sha256, "B6 execution profile"),
        (result.quality_trainer_profile_sha256, permit.quality_trainer_profile_sha256, "B6 quality trainer"),
    ):
        if actual != expected_value:
            raise NativePolyV2BaselineStartError(f"{label} identity mismatch after execution")
    if result.train_sample_ids != permit.train_sample_ids or result.validation_sample_ids != permit.validation_sample_ids:
        raise NativePolyV2BaselineStartError("B6 executed a population different from the B8J permit")
    if result.train_batch_sizes != permit.train_batch_sizes or result.validation_batch_sizes != permit.validation_batch_sizes:
        raise NativePolyV2BaselineStartError("B6 batch plan differs from the B8J permit")
    if result.optimizer_steps != permit.required_optimizer_steps:
        raise NativePolyV2BaselineStartError("B6 optimizer-step count differs from the frozen full run")
    if result.test_split_accessed or result.production_authority:
        raise NativePolyV2BaselineStartError("B6 result exceeded the B8J research boundary")

    return AuthorizedNativePolyV2QualityExecution(
        permit_fingerprint_sha256=permit.fingerprint(),
        admission_fingerprint_sha256=admission.fingerprint(),
        recipe_fingerprint_sha256=recipe.fingerprint(),
        quality_execution_result=result,
    )
