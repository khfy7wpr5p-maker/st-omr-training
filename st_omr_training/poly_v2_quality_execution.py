"""Native Polyphonic V2 multi-batch quality-training execution.

TR-POLY-09B6 is additive to the frozen TR-POLY-09A smoke execution path. It
materializes admitted TRAIN/VALIDATION artifacts across the selected manifest
population, deterministically partitions them into bounded model batches, runs
the TR-POLY-09B4 quality-training regime, and persists the selected state
through the TR-POLY-09B5 verified quality-checkpoint contract.

TEST artifact bytes are never admitted. This module does not run the B1/B2/B3
VALIDATION benchmark and grants no production authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Final

from .dataset_manifest import DatasetSplit
from .poly_2d_quality_checkpoint import (
    LoadedPoly2DQualityCheckpoint,
    persist_poly_2d_quality_checkpoint,
)
from .poly_2d_quality_training import (
    FROZEN_POLY_2D_QUALITY_CONFIG,
    Poly2DQualityTrainingConfig,
    build_poly_2d_quality_training_provenance,
    poly_2d_quality_trainer_fingerprint,
    run_poly_2d_quality_training,
)
from .poly_2d_training import MAX_POLY_2D_TRAINING_BATCH, Poly2DTrainingBatch
from .poly_2d_transformer import (
    FROZEN_POLY_2D_CONFIG,
    Poly2DTransformerConfig,
    poly_2d_config_fingerprint,
)
from .polyphonic_serialization import parse_canonical_polyphonic_json, validate_roundtrip
from .poly_v2_dataset_materialization import (
    MAX_NATIVE_POLY_V2_SAMPLES,
    NATIVE_POLY_V2_SOURCE_CLASS,
    NATIVE_POLY_V2_TARGET_PROFILE,
    NativePolyV2DatasetBuild,
    NativePolyV2DatasetError,
    NativePolyV2MaterializedSample,
    make_native_poly_2d_training_batch,
    native_poly_v2_materialization_fingerprint,
    profile_polyphonic_score,
    _sha256_bytes,
    _verify_dataset_root,
)
from .training_data import InputPreprocessConfig, TrainingDataError, preprocess_grayscale_png


POLY_V2_QUALITY_EXECUTION_VERSION: Final[str] = "st-omr-poly-v2-quality-execution-v1"
POLY_V2_QUALITY_BATCH_POLICY: Final[str] = "sample-id-sorted-contiguous-batches-v1"
POLY_V2_QUALITY_SELECTION_POLICY: Final[str] = "sample-id-sorted-prefix-or-full-v1"


class PolyV2QualityExecutionError(NativePolyV2DatasetError):
    """Raised when the B6 native quality execution boundary fails closed."""


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
        raise PolyV2QualityExecutionError(
            "B6 execution evidence is not canonical-JSON serializable"
        ) from exc


def _require_positive_int(name: str, value: object, *, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= maximum
    ):
        raise PolyV2QualityExecutionError(
            f"{name} must be an integer in [1,{maximum}]"
        )
    return value


def _selected_manifest_samples(
    build: NativePolyV2DatasetBuild,
    split: DatasetSplit,
    max_samples: int | None,
):
    if split is DatasetSplit.TEST:
        raise PolyV2QualityExecutionError("TEST remains sealed in TR-POLY-09B6")
    if split not in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}:
        raise PolyV2QualityExecutionError("B6 split must be TRAIN or VALIDATION")
    selected = tuple(
        sorted(
            (sample for sample in build.manifest.samples if sample.split is split),
            key=lambda item: item.sample_id,
        )
    )
    if not selected:
        raise PolyV2QualityExecutionError(
            f"native V2 build has no {split.value} samples for B6"
        )
    if max_samples is not None:
        _require_positive_int(
            "max_samples", max_samples, maximum=MAX_NATIVE_POLY_V2_SAMPLES
        )
        selected = selected[:max_samples]
    return selected


def _materialize_one(
    *,
    build: NativePolyV2DatasetBuild,
    root: Path,
    sample,
    model_config: Poly2DTransformerConfig,
) -> NativePolyV2MaterializedSample:
    target_path = root / "targets" / f"{sample.target_sha256}.json"
    image_path = root / "images" / f"{sample.image_sha256}.png"
    for path, label in ((target_path, "target"), (image_path, "image")):
        if path.is_symlink() or not path.is_file():
            raise PolyV2QualityExecutionError(
                f"selected B6 {label} artifact is missing or symlinked"
            )

    target_bytes = target_path.read_bytes()
    if _sha256_bytes(target_bytes) != sample.target_sha256:
        raise PolyV2QualityExecutionError("selected B6 target hash mismatch")
    try:
        score = parse_canonical_polyphonic_json(target_bytes)
        target = validate_roundtrip(score)
    except Exception as exc:
        raise PolyV2QualityExecutionError(
            "selected B6 target failed canonical V2 roundtrip"
        ) from exc
    if score.canonical_sha256() != sample.representation_sha256:
        raise PolyV2QualityExecutionError(
            "selected B6 target representation SHA-256 mismatch"
        )
    if len(target.token_ids) != sample.target_token_count:
        raise PolyV2QualityExecutionError(
            "selected B6 target token count differs from manifest"
        )
    if profile_polyphonic_score(score) != sample.profile:
        raise PolyV2QualityExecutionError(
            "selected B6 target polyphony profile differs from manifest"
        )
    decoder_length = len(target.token_ids) - 1
    if decoder_length < 1 or decoder_length > model_config.max_target_tokens:
        raise PolyV2QualityExecutionError(
            "B6 target exceeds model max_target_tokens; truncation is forbidden"
        )

    image_bytes = image_path.read_bytes()
    if _sha256_bytes(image_bytes) != sample.image_sha256:
        raise PolyV2QualityExecutionError("selected B6 image hash mismatch")
    preprocess = InputPreprocessConfig(
        target_height=model_config.input_height,
        target_width=model_config.input_width,
    )
    try:
        image = preprocess_grayscale_png(
            image_bytes,
            preprocess,
            expected_width=sample.width,
            expected_height=sample.height,
        )
    except TrainingDataError as exc:
        raise PolyV2QualityExecutionError(
            "selected B6 PNG failed deterministic preprocessing"
        ) from exc

    return NativePolyV2MaterializedSample(
        sample_id=sample.sample_id,
        family_id=sample.family_id,
        split=sample.split,
        image_sha256=sample.image_sha256,
        target_sha256=sample.target_sha256,
        representation_sha256=sample.representation_sha256,
        target=target,
        image=image,
        source_width=sample.width,
        source_height=sample.height,
        profile=sample.profile,
    )


@dataclass(frozen=True, slots=True)
class NativePolyV2QualityBatchSet:
    split: DatasetSplit
    batches: tuple[Poly2DTrainingBatch, ...]
    sample_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    batch_sizes: tuple[int, ...]
    dataset_manifest_sha256: str
    materialization_fingerprint_sha256: str
    batch_size_limit: int
    selection_fingerprint_sha256: str
    batch_policy: str = POLY_V2_QUALITY_BATCH_POLICY
    selection_policy: str = POLY_V2_QUALITY_SELECTION_POLICY

    def __post_init__(self) -> None:
        if self.split not in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}:
            raise PolyV2QualityExecutionError("B6 batch set may not contain TEST")
        if not self.batches or any(
            not isinstance(batch, Poly2DTrainingBatch) for batch in self.batches
        ):
            raise PolyV2QualityExecutionError("B6 batch set must contain training batches")
        if any(batch.split is not self.split for batch in self.batches):
            raise PolyV2QualityExecutionError("B6 batch set split mismatch")
        if not self.sample_ids or len(set(self.sample_ids)) != len(self.sample_ids):
            raise PolyV2QualityExecutionError("B6 sample IDs must be non-empty and unique")
        if tuple(sorted(self.sample_ids)) != self.sample_ids:
            raise PolyV2QualityExecutionError("B6 sample IDs must remain sorted")
        if len(self.family_ids) != len(self.sample_ids):
            raise PolyV2QualityExecutionError("B6 family/sample cardinality mismatch")
        if len(self.batch_sizes) != len(self.batches):
            raise PolyV2QualityExecutionError("B6 batch-size evidence mismatch")
        if sum(self.batch_sizes) != len(self.sample_ids):
            raise PolyV2QualityExecutionError("B6 batch sizes do not cover selected samples")
        if any(not 1 <= size <= self.batch_size_limit for size in self.batch_sizes):
            raise PolyV2QualityExecutionError("B6 batch exceeds configured size limit")
        flattened = tuple(sample for batch in self.batches for sample in batch.sample_ids)
        if flattened != self.sample_ids:
            raise PolyV2QualityExecutionError("B6 batch order differs from sample evidence")
        if any(
            batch.dataset_manifest_sha256 != self.dataset_manifest_sha256
            for batch in self.batches
        ):
            raise PolyV2QualityExecutionError("B6 batch dataset identity mismatch")
        for value, name in (
            (self.dataset_manifest_sha256, "dataset_manifest_sha256"),
            (self.materialization_fingerprint_sha256, "materialization_fingerprint_sha256"),
            (self.selection_fingerprint_sha256, "selection_fingerprint_sha256"),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise PolyV2QualityExecutionError(f"{name} must be SHA-256 text")
        if self.batch_policy != POLY_V2_QUALITY_BATCH_POLICY:
            raise PolyV2QualityExecutionError("B6 batch policy mismatch")
        if self.selection_policy != POLY_V2_QUALITY_SELECTION_POLICY:
            raise PolyV2QualityExecutionError("B6 selection policy mismatch")

    def fingerprint(self) -> str:
        payload = {
            "version": POLY_V2_QUALITY_EXECUTION_VERSION,
            "split": self.split.value,
            "sample_ids": list(self.sample_ids),
            "family_ids": list(self.family_ids),
            "batch_sizes": list(self.batch_sizes),
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "materialization_fingerprint_sha256": self.materialization_fingerprint_sha256,
            "batch_size_limit": self.batch_size_limit,
            "selection_fingerprint_sha256": self.selection_fingerprint_sha256,
            "batch_policy": self.batch_policy,
            "selection_policy": self.selection_policy,
        }
        return sha256(_canonical_json_bytes(payload)).hexdigest()


def materialize_native_poly_v2_quality_batches(
    *,
    build: NativePolyV2DatasetBuild,
    dataset_root: str | Path,
    split: DatasetSplit,
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
    batch_size: int = MAX_POLY_2D_TRAINING_BATCH,
    max_samples: int | None = None,
) -> NativePolyV2QualityBatchSet:
    """Materialize a deterministic split population into bounded ordered batches."""

    if split is DatasetSplit.TEST:
        raise PolyV2QualityExecutionError("TEST remains sealed in TR-POLY-09B6")
    if not isinstance(build, NativePolyV2DatasetBuild):
        raise TypeError("build must be NativePolyV2DatasetBuild")
    if not isinstance(dataset_root, (str, Path)):
        raise TypeError("dataset_root must be str or pathlib.Path")
    if not isinstance(model_config, Poly2DTransformerConfig):
        raise TypeError("model_config must be Poly2DTransformerConfig")
    _require_positive_int(
        "batch_size", batch_size, maximum=MAX_POLY_2D_TRAINING_BATCH
    )
    selected = _selected_manifest_samples(build, split, max_samples)
    root = Path(dataset_root)
    _verify_dataset_root(build, root)
    materialized = tuple(
        _materialize_one(
            build=build,
            root=root,
            sample=sample,
            model_config=model_config,
        )
        for sample in selected
    )
    batches = tuple(
        make_native_poly_2d_training_batch(
            materialized[index : index + batch_size],
            dataset_manifest_sha256=build.manifest_sha256,
        )
        for index in range(0, len(materialized), batch_size)
    )
    sample_ids = tuple(sample.sample_id for sample in materialized)
    family_ids = tuple(sample.family_id for sample in materialized)
    batch_sizes = tuple(len(batch.sample_ids) for batch in batches)
    materialization_sha = native_poly_v2_materialization_fingerprint(model_config)
    selection_payload = {
        "version": POLY_V2_QUALITY_EXECUTION_VERSION,
        "selection_policy": POLY_V2_QUALITY_SELECTION_POLICY,
        "split": split.value,
        "dataset_manifest_sha256": build.manifest_sha256,
        "build_id": build.build_id,
        "sample_ids": list(sample_ids),
        "family_ids": list(family_ids),
        "batch_size": batch_size,
        "materialization_fingerprint_sha256": materialization_sha,
    }
    selection_sha = sha256(_canonical_json_bytes(selection_payload)).hexdigest()
    return NativePolyV2QualityBatchSet(
        split=split,
        batches=batches,
        sample_ids=sample_ids,
        family_ids=family_ids,
        batch_sizes=batch_sizes,
        dataset_manifest_sha256=build.manifest_sha256,
        materialization_fingerprint_sha256=materialization_sha,
        batch_size_limit=batch_size,
        selection_fingerprint_sha256=selection_sha,
    )


def poly_v2_quality_execution_profile_fingerprint(
    *,
    quality_config: Poly2DQualityTrainingConfig = FROZEN_POLY_2D_QUALITY_CONFIG,
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
    batch_size: int = MAX_POLY_2D_TRAINING_BATCH,
) -> str:
    if not isinstance(quality_config, Poly2DQualityTrainingConfig):
        raise TypeError("quality_config must be Poly2DQualityTrainingConfig")
    if not isinstance(model_config, Poly2DTransformerConfig):
        raise TypeError("model_config must be Poly2DTransformerConfig")
    _require_positive_int(
        "batch_size", batch_size, maximum=MAX_POLY_2D_TRAINING_BATCH
    )
    payload = {
        "version": POLY_V2_QUALITY_EXECUTION_VERSION,
        "source_class": NATIVE_POLY_V2_SOURCE_CLASS,
        "target_profile": NATIVE_POLY_V2_TARGET_PROFILE,
        "batch_policy": POLY_V2_QUALITY_BATCH_POLICY,
        "selection_policy": POLY_V2_QUALITY_SELECTION_POLICY,
        "batch_size": batch_size,
        "model_profile_sha256": poly_2d_config_fingerprint(model_config),
        "quality_trainer_profile_sha256": poly_2d_quality_trainer_fingerprint(
            quality_config, model_config
        ),
        "materialization_fingerprint_sha256": native_poly_v2_materialization_fingerprint(
            model_config
        ),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class NativePolyV2QualityExecutionResult:
    checkpoint_sha256: str
    checkpoint_metadata_sha256: str
    checkpoint_receipt_sha256: str
    training_result_file_sha256: str
    training_result_fingerprint_sha256: str
    selected_state_sha256: str
    selected_epoch: int
    selected_validation_loss: float
    optimizer_steps: int
    dataset_manifest_sha256: str
    dataset_build_id: str
    materialization_fingerprint_sha256: str
    execution_profile_sha256: str
    quality_trainer_profile_sha256: str
    training_plan_sha256: str
    train_batch_set_sha256: str
    validation_batch_set_sha256: str
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    train_batch_sizes: tuple[int, ...]
    validation_batch_sizes: tuple[int, ...]
    source_class: str = NATIVE_POLY_V2_SOURCE_CLASS
    target_profile: str = NATIVE_POLY_V2_TARGET_PROFILE
    test_split_accessed: bool = False
    benchmark_evidence: bool = False
    production_authority: bool = False

    def __post_init__(self) -> None:
        sha_fields = (
            "checkpoint_sha256",
            "checkpoint_metadata_sha256",
            "checkpoint_receipt_sha256",
            "training_result_file_sha256",
            "training_result_fingerprint_sha256",
            "selected_state_sha256",
            "dataset_manifest_sha256",
            "dataset_build_id",
            "materialization_fingerprint_sha256",
            "execution_profile_sha256",
            "quality_trainer_profile_sha256",
            "training_plan_sha256",
            "train_batch_set_sha256",
            "validation_batch_set_sha256",
        )
        for name in sha_fields:
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise PolyV2QualityExecutionError(f"{name} must be SHA-256 text")
        if not isinstance(self.selected_epoch, int) or isinstance(self.selected_epoch, bool) or self.selected_epoch < 1:
            raise PolyV2QualityExecutionError("selected_epoch must be positive")
        if not isinstance(self.optimizer_steps, int) or isinstance(self.optimizer_steps, bool) or self.optimizer_steps < 1:
            raise PolyV2QualityExecutionError("optimizer_steps must be positive")
        if not isinstance(self.selected_validation_loss, (int, float)) or isinstance(self.selected_validation_loss, bool):
            raise PolyV2QualityExecutionError("selected_validation_loss must be numeric")
        if not self.train_sample_ids or not self.validation_sample_ids:
            raise PolyV2QualityExecutionError("B6 result requires TRAIN and VALIDATION samples")
        if set(self.train_sample_ids) & set(self.validation_sample_ids):
            raise PolyV2QualityExecutionError("B6 result contains sample leakage")
        if sum(self.train_batch_sizes) != len(self.train_sample_ids):
            raise PolyV2QualityExecutionError("B6 TRAIN batch evidence mismatch")
        if sum(self.validation_batch_sizes) != len(self.validation_sample_ids):
            raise PolyV2QualityExecutionError("B6 VALIDATION batch evidence mismatch")
        if self.source_class != NATIVE_POLY_V2_SOURCE_CLASS or self.target_profile != NATIVE_POLY_V2_TARGET_PROFILE:
            raise PolyV2QualityExecutionError("B6 source/target profile mismatch")
        if self.test_split_accessed or self.benchmark_evidence or self.production_authority:
            raise PolyV2QualityExecutionError(
                "B6 may not claim TEST access, benchmark evidence, or production authority"
            )

    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["train_sample_ids"] = list(self.train_sample_ids)
        payload["validation_sample_ids"] = list(self.validation_sample_ids)
        payload["train_batch_sizes"] = list(self.train_batch_sizes)
        payload["validation_batch_sizes"] = list(self.validation_batch_sizes)
        return sha256(_canonical_json_bytes(payload)).hexdigest()


def execute_native_poly_v2_quality_training(
    *,
    build: NativePolyV2DatasetBuild,
    dataset_root: str | Path,
    repository_sha: str,
    output_directory: Path,
    quality_config: Poly2DQualityTrainingConfig = FROZEN_POLY_2D_QUALITY_CONFIG,
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
    batch_size: int = MAX_POLY_2D_TRAINING_BATCH,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
) -> NativePolyV2QualityExecutionResult:
    """Execute native V2 B4 training and B5 persistence without opening TEST."""

    if not isinstance(build, NativePolyV2DatasetBuild):
        raise TypeError("build must be NativePolyV2DatasetBuild")
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    train = materialize_native_poly_v2_quality_batches(
        build=build,
        dataset_root=dataset_root,
        split=DatasetSplit.TRAIN,
        model_config=model_config,
        batch_size=batch_size,
        max_samples=max_train_samples,
    )
    validation = materialize_native_poly_v2_quality_batches(
        build=build,
        dataset_root=dataset_root,
        split=DatasetSplit.VALIDATION,
        model_config=model_config,
        batch_size=batch_size,
        max_samples=max_validation_samples,
    )
    if set(train.family_ids) & set(validation.family_ids):
        raise PolyV2QualityExecutionError(
            "TRAIN/VALIDATION family leakage detected at B6 execution boundary"
        )
    if set(train.sample_ids) & set(validation.sample_ids):
        raise PolyV2QualityExecutionError(
            "TRAIN/VALIDATION sample leakage detected at B6 execution boundary"
        )
    if train.dataset_manifest_sha256 != validation.dataset_manifest_sha256:
        raise PolyV2QualityExecutionError("B6 split manifest identities differ")
    if train.materialization_fingerprint_sha256 != validation.materialization_fingerprint_sha256:
        raise PolyV2QualityExecutionError("B6 split materialization identities differ")

    provenance = build_poly_2d_quality_training_provenance(
        repository_sha=repository_sha,
        dataset_manifest_sha256=build.manifest_sha256,
        preprocess_fingerprint_sha256=train.materialization_fingerprint_sha256,
        quality_config=quality_config,
        model_config=model_config,
    )
    run = run_poly_2d_quality_training(
        train_batches=train.batches,
        validation_batches=validation.batches,
        provenance=provenance,
        quality_config=quality_config,
        model_config=model_config,
    )
    loaded: LoadedPoly2DQualityCheckpoint = persist_poly_2d_quality_checkpoint(
        output_directory,
        run=run,
        provenance=provenance,
        quality_config=quality_config,
        model_config=model_config,
    )
    metadata = loaded.metadata
    result = run.result
    if metadata.dataset_manifest_sha256 != build.manifest_sha256:
        raise PolyV2QualityExecutionError("B6 checkpoint dataset identity mismatch")
    if metadata.preprocess_fingerprint_sha256 != train.materialization_fingerprint_sha256:
        raise PolyV2QualityExecutionError("B6 checkpoint materialization identity mismatch")
    if metadata.selected_state_sha256 != result.selected_state_sha256:
        raise PolyV2QualityExecutionError("B6 checkpoint selected state mismatch")
    if metadata.training_plan_sha256 != result.training_plan_sha256:
        raise PolyV2QualityExecutionError("B6 checkpoint training-plan identity mismatch")
    if metadata.benchmark_evidence or metadata.test_split_accessed or metadata.production_authority:
        raise PolyV2QualityExecutionError("B6 checkpoint exceeded research claim boundary")

    execution_profile = poly_v2_quality_execution_profile_fingerprint(
        quality_config=quality_config,
        model_config=model_config,
        batch_size=batch_size,
    )
    return NativePolyV2QualityExecutionResult(
        checkpoint_sha256=loaded.checkpoint_sha256,
        checkpoint_metadata_sha256=loaded.metadata_sha256,
        checkpoint_receipt_sha256=loaded.receipt_sha256,
        training_result_file_sha256=loaded.training_result_file_sha256,
        training_result_fingerprint_sha256=metadata.training_result_fingerprint_sha256,
        selected_state_sha256=metadata.selected_state_sha256,
        selected_epoch=metadata.selected_epoch,
        selected_validation_loss=metadata.selected_validation_loss,
        optimizer_steps=metadata.optimizer_steps,
        dataset_manifest_sha256=build.manifest_sha256,
        dataset_build_id=build.build_id,
        materialization_fingerprint_sha256=train.materialization_fingerprint_sha256,
        execution_profile_sha256=execution_profile,
        quality_trainer_profile_sha256=metadata.quality_trainer_profile_sha256,
        training_plan_sha256=metadata.training_plan_sha256,
        train_batch_set_sha256=train.fingerprint(),
        validation_batch_set_sha256=validation.fingerprint(),
        train_sample_ids=train.sample_ids,
        validation_sample_ids=validation.sample_ids,
        train_batch_sizes=train.batch_sizes,
        validation_batch_sizes=validation.batch_sizes,
    )
