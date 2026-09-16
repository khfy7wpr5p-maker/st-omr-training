"""Hash-bound experiment recipe freeze for the first real Native Polyphonic V2 baseline.

TR-POLY-09B8B does not train a model.  It binds an already verified B8A dataset
preflight to the frozen B4/B6/B7 execution surfaces so the first real baseline
cannot silently change dataset population, model/training configuration, batch
policy, descriptor metadata, or decode bounds after evidence is observed.

TEST artifact bytes remain sealed and no production authority is granted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
import re
from typing import Final, Iterable

from .dataset_manifest import DatasetSplit
from .poly_2d_quality_training import (
    FROZEN_POLY_2D_QUALITY_CONFIG,
    Poly2DQualityTrainingConfig,
    poly_2d_quality_trainer_fingerprint,
)
from .poly_2d_training import MAX_POLY_2D_TRAINING_BATCH
from .poly_2d_transformer import (
    FROZEN_POLY_2D_CONFIG,
    Poly2DTransformerConfig,
    poly_2d_config_fingerprint,
)
from .poly_evaluation_contract import BenchmarkSampleDescriptor
from .poly_v2_dataset_materialization import native_poly_v2_materialization_fingerprint
from .poly_v2_dataset_reload import LoadedNativePolyV2Dataset
from .poly_v2_quality_execution import poly_v2_quality_execution_profile_fingerprint
from .poly_v2_validation_execution import (
    PolyV2ValidationExecutionError,
    build_native_poly_v2_validation_benchmark_identity,
    native_poly_v2_validation_split_manifest_sha256,
)


POLY_V2_EXPERIMENT_RECIPE_VERSION: Final[str] = "st-omr-poly-v2-experiment-recipe-v1"
FIRST_BASELINE_BENCHMARK_ID: Final[str] = "st-omr-native-v2-first-quality-baseline"
FIRST_BASELINE_BENCHMARK_VERSION: Final[str] = "v1"
_GIT_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PolyV2ExperimentRecipeError(ValueError):
    """Raised when the B8B first-baseline recipe cannot be frozen exactly."""


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
        raise PolyV2ExperimentRecipeError(
            "B8B recipe evidence is not canonical-JSON serializable"
        ) from exc


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _plain_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PolyV2ExperimentRecipeError(f"{name} must be lowercase SHA-256 text")
    return value


def _batch_sizes(sample_count: int, batch_size: int) -> tuple[int, ...]:
    return tuple(
        min(batch_size, sample_count - index)
        for index in range(0, sample_count, batch_size)
    )


def _descriptor_manifest_sha256(
    descriptors: tuple[BenchmarkSampleDescriptor, ...],
) -> str:
    payload = [
        {
            "sample_id": item.sample_id,
            "family_id": item.family_id,
            "split": item.split,
            "complexity": _jsonable(asdict(item.complexity)),
            "robustness_bucket": item.robustness_bucket.value,
        }
        for item in sorted(descriptors, key=lambda value: value.sample_id)
    ]
    return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class NativePolyV2ExperimentRecipe:
    repository_sha: str
    dataset_manifest_sha256: str
    dataset_build_id: str
    preflight_receipt_sha256: str
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    train_family_ids: tuple[str, ...]
    validation_family_ids: tuple[str, ...]
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
    validation_descriptor_manifest_sha256: str
    validation_split_manifest_sha256: str
    benchmark_identity_sha256: str
    benchmark_id: str
    benchmark_version: str
    max_decode_steps: int
    recipe_version: str = POLY_V2_EXPERIMENT_RECIPE_VERSION
    full_train_population: bool = True
    full_validation_population: bool = True
    test_artifact_bytes_accessed: bool = False
    production_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.repository_sha, str) or _GIT_SHA40.fullmatch(self.repository_sha) is None:
            raise PolyV2ExperimentRecipeError("repository_sha must be lowercase git SHA-40")
        for name in (
            "dataset_manifest_sha256",
            "dataset_build_id",
            "preflight_receipt_sha256",
            "model_profile_sha256",
            "quality_trainer_profile_sha256",
            "materialization_fingerprint_sha256",
            "quality_execution_profile_sha256",
            "validation_descriptor_manifest_sha256",
            "validation_split_manifest_sha256",
            "benchmark_identity_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if not self.train_sample_ids or not self.validation_sample_ids:
            raise PolyV2ExperimentRecipeError("B8B requires full TRAIN and VALIDATION populations")
        if len(self.train_family_ids) != len(self.train_sample_ids) or len(self.validation_family_ids) != len(self.validation_sample_ids):
            raise PolyV2ExperimentRecipeError("B8B family/sample cardinality mismatch")
        if set(self.train_sample_ids) & set(self.validation_sample_ids):
            raise PolyV2ExperimentRecipeError("B8B sample leakage detected")
        if set(self.train_family_ids) & set(self.validation_family_ids):
            raise PolyV2ExperimentRecipeError("B8B family leakage detected")
        if not _plain_positive_int(self.batch_size) or self.batch_size > MAX_POLY_2D_TRAINING_BATCH:
            raise PolyV2ExperimentRecipeError("B8B batch_size exceeds the B6 boundary")
        for name in ("epochs", "max_optimizer_steps", "required_optimizer_steps", "max_decode_steps"):
            if not _plain_positive_int(getattr(self, name)):
                raise PolyV2ExperimentRecipeError(f"B8B {name} must be a positive plain integer")
        if self.train_batch_sizes != _batch_sizes(len(self.train_sample_ids), self.batch_size):
            raise PolyV2ExperimentRecipeError("B8B TRAIN batch plan differs from deterministic full-population batching")
        if self.validation_batch_sizes != _batch_sizes(len(self.validation_sample_ids), self.batch_size):
            raise PolyV2ExperimentRecipeError("B8B VALIDATION batch plan differs from deterministic full-population batching")
        if any(not _plain_positive_int(size) or size > self.batch_size for size in self.train_batch_sizes + self.validation_batch_sizes):
            raise PolyV2ExperimentRecipeError("B8B batch plan contains an invalid batch size")
        if self.required_optimizer_steps != len(self.train_batch_sizes) * self.epochs:
            raise PolyV2ExperimentRecipeError("B8B optimizer-step evidence differs from the frozen TRAIN plan")
        if self.required_optimizer_steps > self.max_optimizer_steps:
            raise PolyV2ExperimentRecipeError("B8B optimizer-step ceiling cannot cover the frozen full TRAIN plan")
        if not isinstance(self.benchmark_id, str) or not self.benchmark_id or not isinstance(self.benchmark_version, str) or not self.benchmark_version:
            raise PolyV2ExperimentRecipeError("B8B benchmark identity text must be non-empty")
        if self.recipe_version != POLY_V2_EXPERIMENT_RECIPE_VERSION:
            raise PolyV2ExperimentRecipeError("B8B recipe version mismatch")
        for name in (
            "full_train_population",
            "full_validation_population",
            "test_artifact_bytes_accessed",
            "production_authority",
        ):
            if not isinstance(getattr(self, name), bool):
                raise PolyV2ExperimentRecipeError(f"B8B {name} must be boolean")
        if not self.full_train_population or not self.full_validation_population:
            raise PolyV2ExperimentRecipeError("first baseline may not use prefix/subsample selection")
        if self.test_artifact_bytes_accessed or self.production_authority:
            raise PolyV2ExperimentRecipeError("B8B may not access TEST bytes or grant production authority")

    def fingerprint(self) -> str:
        payload = asdict(self)
        for name in (
            "train_sample_ids",
            "validation_sample_ids",
            "train_family_ids",
            "validation_family_ids",
            "train_batch_sizes",
            "validation_batch_sizes",
        ):
            payload[name] = list(payload[name])
        return sha256(_canonical_json_bytes(payload)).hexdigest()


def freeze_native_poly_v2_experiment_recipe(
    *,
    loaded: LoadedNativePolyV2Dataset,
    repository_sha: str,
    descriptors: Iterable[BenchmarkSampleDescriptor],
    quality_config: Poly2DQualityTrainingConfig = FROZEN_POLY_2D_QUALITY_CONFIG,
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
    batch_size: int = MAX_POLY_2D_TRAINING_BATCH,
    benchmark_id: str = FIRST_BASELINE_BENCHMARK_ID,
    benchmark_version: str = FIRST_BASELINE_BENCHMARK_VERSION,
    max_decode_steps: int | None = None,
) -> NativePolyV2ExperimentRecipe:
    """Freeze one first-baseline recipe without reading TEST artifact bytes."""

    if not isinstance(loaded, LoadedNativePolyV2Dataset):
        raise TypeError("loaded must be LoadedNativePolyV2Dataset")
    if not isinstance(repository_sha, str) or _GIT_SHA40.fullmatch(repository_sha) is None:
        raise PolyV2ExperimentRecipeError("repository_sha must be lowercase git SHA-40")
    if not isinstance(quality_config, Poly2DQualityTrainingConfig):
        raise TypeError("quality_config must be Poly2DQualityTrainingConfig")
    if not isinstance(model_config, Poly2DTransformerConfig):
        raise TypeError("model_config must be Poly2DTransformerConfig")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= MAX_POLY_2D_TRAINING_BATCH:
        raise PolyV2ExperimentRecipeError("batch_size is outside the B6 boundary")

    build = loaded.build
    receipt = loaded.receipt
    train_samples = tuple(sorted(
        (sample for sample in build.manifest.samples if sample.split is DatasetSplit.TRAIN),
        key=lambda item: item.sample_id,
    ))
    validation_samples = tuple(sorted(
        (sample for sample in build.manifest.samples if sample.split is DatasetSplit.VALIDATION),
        key=lambda item: item.sample_id,
    ))
    train_ids = tuple(item.sample_id for item in train_samples)
    validation_ids = tuple(item.sample_id for item in validation_samples)
    if train_ids != receipt.train_sample_ids or validation_ids != receipt.validation_sample_ids:
        raise PolyV2ExperimentRecipeError("B8A preflight receipt population differs from the loaded build")

    descriptor_values = tuple(descriptors)
    try:
        validation_split_sha = native_poly_v2_validation_split_manifest_sha256(
            build=build,
            descriptors=descriptor_values,
        )
        benchmark = build_native_poly_v2_validation_benchmark_identity(
            build=build,
            descriptors=descriptor_values,
            benchmark_id=benchmark_id,
            benchmark_version=benchmark_version,
        )
    except PolyV2ValidationExecutionError as exc:
        raise PolyV2ExperimentRecipeError("B8B descriptors do not bind the exact VALIDATION population") from exc

    minimum_decode_steps = max(item.target_token_count - 1 for item in validation_samples)
    decode_steps = minimum_decode_steps if max_decode_steps is None else max_decode_steps
    if not isinstance(decode_steps, int) or isinstance(decode_steps, bool):
        raise PolyV2ExperimentRecipeError("max_decode_steps must be a plain integer")
    if decode_steps < minimum_decode_steps:
        raise PolyV2ExperimentRecipeError("max_decode_steps would truncate an admitted VALIDATION target")
    if decode_steps > model_config.max_target_tokens - 1:
        raise PolyV2ExperimentRecipeError("max_decode_steps exceeds the B7/model target boundary")

    train_batch_sizes = _batch_sizes(len(train_samples), batch_size)
    validation_batch_sizes = _batch_sizes(len(validation_samples), batch_size)
    required_steps = len(train_batch_sizes) * quality_config.epochs
    if required_steps > quality_config.max_optimizer_steps:
        raise PolyV2ExperimentRecipeError(
            "frozen max_optimizer_steps cannot cover every TRAIN batch for every epoch"
        )

    return NativePolyV2ExperimentRecipe(
        repository_sha=repository_sha,
        dataset_manifest_sha256=build.manifest_sha256,
        dataset_build_id=build.build_id,
        preflight_receipt_sha256=receipt.fingerprint(),
        train_sample_ids=train_ids,
        validation_sample_ids=validation_ids,
        train_family_ids=tuple(item.family_id for item in train_samples),
        validation_family_ids=tuple(item.family_id for item in validation_samples),
        train_batch_sizes=train_batch_sizes,
        validation_batch_sizes=validation_batch_sizes,
        batch_size=batch_size,
        epochs=quality_config.epochs,
        max_optimizer_steps=quality_config.max_optimizer_steps,
        required_optimizer_steps=required_steps,
        model_profile_sha256=poly_2d_config_fingerprint(model_config),
        quality_trainer_profile_sha256=poly_2d_quality_trainer_fingerprint(
            quality_config, model_config
        ),
        materialization_fingerprint_sha256=native_poly_v2_materialization_fingerprint(
            model_config
        ),
        quality_execution_profile_sha256=poly_v2_quality_execution_profile_fingerprint(
            quality_config=quality_config,
            model_config=model_config,
            batch_size=batch_size,
        ),
        validation_descriptor_manifest_sha256=_descriptor_manifest_sha256(descriptor_values),
        validation_split_manifest_sha256=validation_split_sha,
        benchmark_identity_sha256=benchmark.canonical_sha256(),
        benchmark_id=benchmark.benchmark_id,
        benchmark_version=benchmark.benchmark_version,
        max_decode_steps=decode_steps,
    )
