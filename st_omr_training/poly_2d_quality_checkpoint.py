"""Verified quality-checkpoint artifacts for the Polyphonic V2 2D Transformer.

TR-POLY-09B5 is separate from the frozen TR-POLY-08B <=2-step checkpoint
schema. It persists only the model state selected by TR-POLY-09B4, binds the
exact training recipe/provenance/result, verifies all hashes before model
loading, and exposes a checkpoint-bound B1 greedy-inference wrapper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
from typing import Final, Mapping

import torch

from .model_registry import (
    SEED_MODEL_REGISTRY,
    ModelArtifactBinding,
    ModelKind,
    ModelLifecycle,
    ModelRegistryRecord,
    ResearchAuthority,
    SemanticScope,
    validate_artifact_binding,
)
from .poly_2d_inference import (
    Poly2DInferenceError,
    Poly2DInferenceResult,
    run_poly_2d_greedy_inference,
)
from .poly_2d_quality_training import (
    POLY_2D_QUALITY_PROVENANCE_VERSION,
    POLY_2D_QUALITY_TRAINER_VERSION,
    Poly2DQualityTrainingConfig,
    Poly2DQualityTrainingProvenance,
    Poly2DQualityTrainingResult,
    Poly2DQualityTrainingRun,
    poly_2d_quality_trainer_fingerprint,
)
from .poly_2d_transformer import (
    POLY_2D_TRANSFORMER_VERSION,
    Poly2DTransformerConfig,
    TinyPoly2DTransformer,
    build_tiny_poly_2d_transformer,
    poly_2d_config_fingerprint,
)
from .polyphonic_representation import POLYPHONIC_REPRESENTATION_VERSION
from .polyphonic_serialization import (
    POLYPHONIC_TOKENIZER_VERSION,
    tokenizer_fingerprint,
)
from .training_model import (
    TORCH_PINNED_VERSION,
    TrainingRuntimeError,
    assert_model_finite,
    count_trainable_parameters,
    model_state_sha256,
)


POLY_2D_QUALITY_CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "st-omr-poly-2d-quality-checkpoint-v1"
)
POLY_2D_QUALITY_CHECKPOINT_RECEIPT_VERSION: Final[str] = (
    "st-omr-poly-2d-quality-checkpoint-receipt-v1"
)
POLY_2D_QUALITY_REGISTRY_RECORD_ID: Final[str] = (
    "candidate.poly-2d-transformer.quality-v1"
)
QUALITY_CHECKPOINT_FILENAME: Final[str] = "model.pt"
QUALITY_METADATA_FILENAME: Final[str] = "metadata.json"
QUALITY_TRAINING_RESULT_FILENAME: Final[str] = "training_result.json"
QUALITY_RECEIPT_FILENAME: Final[str] = "receipt.json"
MAX_QUALITY_CHECKPOINT_BYTES: Final[int] = 128 * 1024 * 1024
MAX_QUALITY_METADATA_BYTES: Final[int] = 512 * 1024
MAX_QUALITY_RESULT_BYTES: Final[int] = 4 * 1024 * 1024
MAX_QUALITY_RECEIPT_BYTES: Final[int] = 128 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_FILES: Final[frozenset[str]] = frozenset(
    {
        QUALITY_CHECKPOINT_FILENAME,
        QUALITY_METADATA_FILENAME,
        QUALITY_TRAINING_RESULT_FILENAME,
        QUALITY_RECEIPT_FILENAME,
    }
)


QUALITY_REGISTRY_RECORD: Final[ModelRegistryRecord] = ModelRegistryRecord(
    record_id=POLY_2D_QUALITY_REGISTRY_RECORD_ID,
    model_kind=ModelKind.CANDIDATE_FAMILY,
    lifecycle=ModelLifecycle.TRAINING_IMPLEMENTED,
    authority=ResearchAuthority.EXPERIMENTAL,
    semantic_scope=SemanticScope.POLYPHONIC_V2,
    source_module="st_omr_training.poly_2d_quality_training",
    source_version=POLY_2D_QUALITY_TRAINER_VERSION,
    task_ids=("polyphonic_sequence_omr",),
    tokenizer_version=POLYPHONIC_TOKENIZER_VERSION,
    representation_version=POLYPHONIC_REPRESENTATION_VERSION,
    checkpoint_required_for_evidence=True,
    polyphonic_v2_capable=True,
    production_authority=False,
)


class Poly2DQualityCheckpointError(RuntimeError):
    """Raised when a B5 quality checkpoint fails closed."""


def _quality_registry_records() -> tuple[ModelRegistryRecord, ...]:
    return SEED_MODEL_REGISTRY + (QUALITY_REGISTRY_RECORD,)


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
        raise Poly2DQualityCheckpointError(
            "quality checkpoint evidence is not canonical-JSON serializable"
        ) from exc


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Poly2DQualityCheckpointError(f"{name} must be lowercase SHA-256 hex")
    return value


def _require_git_sha(name: str, value: object) -> str:
    if not isinstance(value, str) or _GIT_SHA40.fullmatch(value) is None:
        raise Poly2DQualityCheckpointError(f"{name} must be lowercase git SHA-40 hex")
    return value


def _sha256_file(path: Path, maximum_bytes: int, name: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise Poly2DQualityCheckpointError(f"{name} must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= maximum_bytes:
        raise Poly2DQualityCheckpointError(f"{name} byte length is outside the B5 boundary")
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_canonical_json(path: Path, maximum_bytes: int, name: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Poly2DQualityCheckpointError(f"{name} must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= maximum_bytes:
        raise Poly2DQualityCheckpointError(f"{name} byte length is outside the B5 boundary")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Poly2DQualityCheckpointError(f"{name} is not valid canonical JSON") from exc
    if not isinstance(payload, dict) or _canonical_json_bytes(payload) != raw:
        raise Poly2DQualityCheckpointError(f"{name} must be canonical JSON object bytes")
    return payload


def _strict_keys(payload: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    actual = frozenset(payload.keys())
    if actual != expected:
        raise Poly2DQualityCheckpointError(
            f"{name} key set mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _quality_result_payload(result: Poly2DQualityTrainingResult) -> dict[str, object]:
    if not isinstance(result, Poly2DQualityTrainingResult):
        raise TypeError("result must be Poly2DQualityTrainingResult")
    payload = asdict(result)
    payload["epoch_evidence"] = [asdict(item) for item in result.epoch_evidence]
    return payload


def _quality_runtime_fingerprint(model_profile_sha256: str) -> str:
    _require_sha256("model_profile_sha256", model_profile_sha256)
    payload = {
        "torch_version": TORCH_PINNED_VERSION,
        "model_profile_sha256": model_profile_sha256,
        "quality_checkpoint_schema": POLY_2D_QUALITY_CHECKPOINT_SCHEMA_VERSION,
        "device": "cpu",
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class Poly2DQualityCheckpointMetadata:
    repository_sha: str
    registry_record_fingerprint_sha256: str
    dataset_manifest_sha256: str
    preprocess_fingerprint_sha256: str
    model_profile_sha256: str
    quality_trainer_profile_sha256: str
    step_primitive_profile_sha256: str
    provenance_sha256: str
    training_plan_sha256: str
    training_result_fingerprint_sha256: str
    tokenizer_fingerprint_sha256: str
    selected_state_sha256: str
    selected_epoch: int
    selected_validation_loss: float
    optimizer_steps: int
    parameter_count: int
    model_config: dict[str, object]
    quality_training_config: dict[str, object]
    provenance: dict[str, object]
    model_version: str = POLY_2D_TRANSFORMER_VERSION
    quality_trainer_version: str = POLY_2D_QUALITY_TRAINER_VERSION
    quality_provenance_version: str = POLY_2D_QUALITY_PROVENANCE_VERSION
    representation_version: str = POLYPHONIC_REPRESENTATION_VERSION
    tokenizer_version: str = POLYPHONIC_TOKENIZER_VERSION
    torch_version: str = TORCH_PINNED_VERSION
    checkpoint_role: str = "quality_training_candidate_research_only"
    benchmark_evidence: bool = False
    test_split_accessed: bool = False
    production_authority: bool = False
    schema_version: str = POLY_2D_QUALITY_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_git_sha("repository_sha", self.repository_sha)
        for name in (
            "registry_record_fingerprint_sha256",
            "dataset_manifest_sha256",
            "preprocess_fingerprint_sha256",
            "model_profile_sha256",
            "quality_trainer_profile_sha256",
            "step_primitive_profile_sha256",
            "provenance_sha256",
            "training_plan_sha256",
            "training_result_fingerprint_sha256",
            "tokenizer_fingerprint_sha256",
            "selected_state_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if not isinstance(self.selected_epoch, int) or isinstance(self.selected_epoch, bool) or self.selected_epoch < 1:
            raise Poly2DQualityCheckpointError("selected_epoch must be positive")
        if not isinstance(self.selected_validation_loss, (int, float)) or isinstance(self.selected_validation_loss, bool):
            raise Poly2DQualityCheckpointError("selected_validation_loss must be numeric")
        if not math.isfinite(float(self.selected_validation_loss)) or float(self.selected_validation_loss) < 0.0:
            raise Poly2DQualityCheckpointError("selected_validation_loss must be finite non-negative")
        for name in ("optimizer_steps", "parameter_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise Poly2DQualityCheckpointError(f"{name} must be positive")
        if self.model_version != POLY_2D_TRANSFORMER_VERSION:
            raise Poly2DQualityCheckpointError("quality checkpoint model version mismatch")
        if self.quality_trainer_version != POLY_2D_QUALITY_TRAINER_VERSION:
            raise Poly2DQualityCheckpointError("quality checkpoint trainer version mismatch")
        if self.quality_provenance_version != POLY_2D_QUALITY_PROVENANCE_VERSION:
            raise Poly2DQualityCheckpointError("quality checkpoint provenance version mismatch")
        if self.representation_version != POLYPHONIC_REPRESENTATION_VERSION:
            raise Poly2DQualityCheckpointError("quality checkpoint representation mismatch")
        if self.tokenizer_version != POLYPHONIC_TOKENIZER_VERSION:
            raise Poly2DQualityCheckpointError("quality checkpoint tokenizer version mismatch")
        if self.torch_version != TORCH_PINNED_VERSION:
            raise Poly2DQualityCheckpointError("quality checkpoint torch version mismatch")
        if self.tokenizer_fingerprint_sha256 != tokenizer_fingerprint():
            raise Poly2DQualityCheckpointError("quality checkpoint tokenizer fingerprint mismatch")
        if self.checkpoint_role != "quality_training_candidate_research_only":
            raise Poly2DQualityCheckpointError("quality checkpoint role mismatch")
        if self.benchmark_evidence or self.test_split_accessed or self.production_authority:
            raise Poly2DQualityCheckpointError(
                "quality checkpoint may not claim benchmark, TEST, or production authority"
            )
        if self.schema_version != POLY_2D_QUALITY_CHECKPOINT_SCHEMA_VERSION:
            raise Poly2DQualityCheckpointError("quality checkpoint schema mismatch")

        try:
            model_config = Poly2DTransformerConfig(**self.model_config)
            quality_config = Poly2DQualityTrainingConfig(**self.quality_training_config)
            provenance = Poly2DQualityTrainingProvenance(**self.provenance)
        except (TypeError, ValueError, Poly2DTrainingError) as exc:  # type: ignore[name-defined]
            raise Poly2DQualityCheckpointError(
                "quality checkpoint embedded config/provenance is invalid"
            ) from exc
        if poly_2d_config_fingerprint(model_config) != self.model_profile_sha256:
            raise Poly2DQualityCheckpointError("quality checkpoint model profile mismatch")
        if poly_2d_quality_trainer_fingerprint(quality_config, model_config) != self.quality_trainer_profile_sha256:
            raise Poly2DQualityCheckpointError("quality checkpoint trainer profile mismatch")
        if provenance.fingerprint() != self.provenance_sha256:
            raise Poly2DQualityCheckpointError("quality checkpoint provenance fingerprint mismatch")
        if provenance.repository_sha != self.repository_sha:
            raise Poly2DQualityCheckpointError("quality checkpoint repository/provenance mismatch")
        if provenance.dataset_manifest_sha256 != self.dataset_manifest_sha256:
            raise Poly2DQualityCheckpointError("quality checkpoint dataset/provenance mismatch")
        if provenance.preprocess_fingerprint_sha256 != self.preprocess_fingerprint_sha256:
            raise Poly2DQualityCheckpointError("quality checkpoint preprocess/provenance mismatch")
        if provenance.model_profile_sha256 != self.model_profile_sha256:
            raise Poly2DQualityCheckpointError("quality checkpoint model/provenance mismatch")
        if provenance.quality_trainer_profile_sha256 != self.quality_trainer_profile_sha256:
            raise Poly2DQualityCheckpointError("quality checkpoint trainer/provenance mismatch")
        if self.registry_record_fingerprint_sha256 != QUALITY_REGISTRY_RECORD.fingerprint():
            raise Poly2DQualityCheckpointError("quality checkpoint registry-record fingerprint mismatch")

    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        return sha256(_canonical_json_bytes(self.canonical_payload())).hexdigest()


_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    Poly2DQualityCheckpointMetadata.__dataclass_fields__.keys()
)
_RESULT_KEYS: Final[frozenset[str]] = frozenset(
    Poly2DQualityTrainingResult.__dataclass_fields__.keys()
)


@dataclass(frozen=True, slots=True)
class Poly2DQualityCheckpointReceipt:
    checkpoint_sha256: str
    metadata_sha256: str
    training_result_file_sha256: str
    metadata_fingerprint_sha256: str
    training_result_fingerprint_sha256: str
    artifact_binding: dict[str, object]
    artifact_binding_sha256: str
    receipt_version: str = POLY_2D_QUALITY_CHECKPOINT_RECEIPT_VERSION
    checkpoint_filename: str = QUALITY_CHECKPOINT_FILENAME
    metadata_filename: str = QUALITY_METADATA_FILENAME
    training_result_filename: str = QUALITY_TRAINING_RESULT_FILENAME

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256",
            "metadata_sha256",
            "training_result_file_sha256",
            "metadata_fingerprint_sha256",
            "training_result_fingerprint_sha256",
            "artifact_binding_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.receipt_version != POLY_2D_QUALITY_CHECKPOINT_RECEIPT_VERSION:
            raise Poly2DQualityCheckpointError("quality checkpoint receipt version mismatch")
        expected_names = (
            (self.checkpoint_filename, QUALITY_CHECKPOINT_FILENAME),
            (self.metadata_filename, QUALITY_METADATA_FILENAME),
            (self.training_result_filename, QUALITY_TRAINING_RESULT_FILENAME),
        )
        if any(actual != expected for actual, expected in expected_names):
            raise Poly2DQualityCheckpointError("quality checkpoint receipt filename mismatch")
        try:
            binding = ModelArtifactBinding(**self.artifact_binding)
            validate_artifact_binding(binding, _quality_registry_records())
        except Exception as exc:
            raise Poly2DQualityCheckpointError("quality artifact binding is invalid") from exc
        if binding.fingerprint() != self.artifact_binding_sha256:
            raise Poly2DQualityCheckpointError("quality artifact binding fingerprint mismatch")
        if binding.checkpoint_sha256 != self.checkpoint_sha256:
            raise Poly2DQualityCheckpointError("artifact binding checkpoint SHA mismatch")

    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)


_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    Poly2DQualityCheckpointReceipt.__dataclass_fields__.keys()
)


@dataclass(frozen=True, slots=True)
class LoadedPoly2DQualityCheckpoint:
    model: TinyPoly2DTransformer
    metadata: Poly2DQualityCheckpointMetadata
    receipt: Poly2DQualityCheckpointReceipt
    checkpoint_sha256: str
    metadata_sha256: str
    training_result_file_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, TinyPoly2DTransformer):
            raise TypeError("model must be TinyPoly2DTransformer")
        if not isinstance(self.metadata, Poly2DQualityCheckpointMetadata):
            raise TypeError("metadata must be Poly2DQualityCheckpointMetadata")
        if not isinstance(self.receipt, Poly2DQualityCheckpointReceipt):
            raise TypeError("receipt must be Poly2DQualityCheckpointReceipt")
        for name in (
            "checkpoint_sha256",
            "metadata_sha256",
            "training_result_file_sha256",
            "receipt_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if model_state_sha256(self.model) != self.metadata.selected_state_sha256:
            raise Poly2DQualityCheckpointError("loaded quality model state mismatch")


def _parse_metadata(payload: dict[str, object]) -> Poly2DQualityCheckpointMetadata:
    _strict_keys(payload, _METADATA_KEYS, "quality metadata")
    try:
        return Poly2DQualityCheckpointMetadata(**payload)
    except (TypeError, ValueError, Poly2DQualityCheckpointError) as exc:
        raise Poly2DQualityCheckpointError("quality metadata payload is invalid") from exc


def _parse_receipt(payload: dict[str, object]) -> Poly2DQualityCheckpointReceipt:
    _strict_keys(payload, _RECEIPT_KEYS, "quality receipt")
    try:
        return Poly2DQualityCheckpointReceipt(**payload)
    except (TypeError, ValueError, Poly2DQualityCheckpointError) as exc:
        raise Poly2DQualityCheckpointError("quality receipt payload is invalid") from exc


def _validate_training_result_payload(
    payload: dict[str, object], metadata: Poly2DQualityCheckpointMetadata
) -> str:
    _strict_keys(payload, _RESULT_KEYS, "quality training result")
    fingerprint = sha256(_canonical_json_bytes(payload)).hexdigest()
    if fingerprint != metadata.training_result_fingerprint_sha256:
        raise Poly2DQualityCheckpointError("quality training-result fingerprint mismatch")
    required_equal = {
        "selected_state_sha256": metadata.selected_state_sha256,
        "selected_epoch": metadata.selected_epoch,
        "selected_validation_loss": metadata.selected_validation_loss,
        "optimizer_steps": metadata.optimizer_steps,
        "training_plan_sha256": metadata.training_plan_sha256,
        "model_profile_sha256": metadata.model_profile_sha256,
        "quality_trainer_profile_sha256": metadata.quality_trainer_profile_sha256,
        "step_primitive_profile_sha256": metadata.step_primitive_profile_sha256,
        "provenance_sha256": metadata.provenance_sha256,
        "dataset_manifest_sha256": metadata.dataset_manifest_sha256,
        "tokenizer_fingerprint_sha256": metadata.tokenizer_fingerprint_sha256,
        "repository_sha": metadata.repository_sha,
        "torch_version": metadata.torch_version,
        "test_split_accessed": False,
        "production_authority": False,
    }
    for key, expected in required_equal.items():
        if payload.get(key) != expected:
            raise Poly2DQualityCheckpointError(
                f"quality training-result field {key} differs from metadata"
            )
    evidence = payload.get("epoch_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise Poly2DQualityCheckpointError("quality training result has no epoch evidence")
    if metadata.selected_epoch > len(evidence):
        raise Poly2DQualityCheckpointError("selected epoch exceeds training-result evidence")
    selected = evidence[metadata.selected_epoch - 1]
    if not isinstance(selected, dict):
        raise Poly2DQualityCheckpointError("selected epoch evidence is malformed")
    if selected.get("model_state_sha256") != metadata.selected_state_sha256:
        raise Poly2DQualityCheckpointError("selected epoch state differs from metadata")
    if selected.get("validation_mean_loss") != metadata.selected_validation_loss:
        raise Poly2DQualityCheckpointError("selected epoch loss differs from metadata")
    return fingerprint


def _validate_directory(directory: Path) -> tuple[Path, Path, Path, Path]:
    if not isinstance(directory, Path):
        raise TypeError("directory must be pathlib.Path")
    if directory.is_symlink() or not directory.is_dir():
        raise Poly2DQualityCheckpointError("quality checkpoint directory must be regular directory")
    names = frozenset(path.name for path in directory.iterdir())
    if names != _REQUIRED_FILES:
        raise Poly2DQualityCheckpointError(
            f"quality checkpoint file set mismatch: missing={sorted(_REQUIRED_FILES - names)}, "
            f"unknown={sorted(names - _REQUIRED_FILES)}"
        )
    return (
        directory / QUALITY_CHECKPOINT_FILENAME,
        directory / QUALITY_METADATA_FILENAME,
        directory / QUALITY_TRAINING_RESULT_FILENAME,
        directory / QUALITY_RECEIPT_FILENAME,
    )


def load_and_verify_poly_2d_quality_checkpoint(
    directory: Path,
) -> LoadedPoly2DQualityCheckpoint:
    checkpoint_path, metadata_path, result_path, receipt_path = _validate_directory(directory)
    checkpoint_sha = _sha256_file(
        checkpoint_path, MAX_QUALITY_CHECKPOINT_BYTES, "quality checkpoint"
    )
    metadata_sha = _sha256_file(
        metadata_path, MAX_QUALITY_METADATA_BYTES, "quality metadata"
    )
    result_file_sha = _sha256_file(
        result_path, MAX_QUALITY_RESULT_BYTES, "quality training result"
    )
    receipt_sha = _sha256_file(
        receipt_path, MAX_QUALITY_RECEIPT_BYTES, "quality receipt"
    )

    receipt = _parse_receipt(
        _read_canonical_json(
            receipt_path, MAX_QUALITY_RECEIPT_BYTES, "quality receipt"
        )
    )
    if receipt.checkpoint_sha256 != checkpoint_sha:
        raise Poly2DQualityCheckpointError("quality checkpoint SHA differs from receipt")
    if receipt.metadata_sha256 != metadata_sha:
        raise Poly2DQualityCheckpointError("quality metadata SHA differs from receipt")
    if receipt.training_result_file_sha256 != result_file_sha:
        raise Poly2DQualityCheckpointError("quality training-result file SHA differs from receipt")

    metadata = _parse_metadata(
        _read_canonical_json(
            metadata_path, MAX_QUALITY_METADATA_BYTES, "quality metadata"
        )
    )
    if metadata.fingerprint() != receipt.metadata_fingerprint_sha256:
        raise Poly2DQualityCheckpointError("quality metadata fingerprint differs from receipt")
    result_payload = _read_canonical_json(
        result_path, MAX_QUALITY_RESULT_BYTES, "quality training result"
    )
    result_fingerprint = _validate_training_result_payload(result_payload, metadata)
    if result_fingerprint != receipt.training_result_fingerprint_sha256:
        raise Poly2DQualityCheckpointError("quality result fingerprint differs from receipt")

    model_config = Poly2DTransformerConfig(**metadata.model_config)
    model = build_tiny_poly_2d_transformer(
        model_config,
        seed=Poly2DQualityTrainingConfig(**metadata.quality_training_config).master_seed,
    )
    try:
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise Poly2DQualityCheckpointError("quality checkpoint deserialization failed") from exc
    if not isinstance(state, dict) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise Poly2DQualityCheckpointError("quality checkpoint must contain tensor state_dict only")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise Poly2DQualityCheckpointError("quality checkpoint state_dict is incompatible") from exc
    try:
        assert_model_finite(model)
    except TrainingRuntimeError as exc:
        raise Poly2DQualityCheckpointError("quality checkpoint contains non-finite model state") from exc
    if model_state_sha256(model) != metadata.selected_state_sha256:
        raise Poly2DQualityCheckpointError("quality checkpoint state SHA differs from metadata")
    if count_trainable_parameters(model) != metadata.parameter_count:
        raise Poly2DQualityCheckpointError("quality checkpoint parameter count mismatch")

    binding = ModelArtifactBinding(**receipt.artifact_binding)
    if binding.model_fingerprint_sha256 != metadata.selected_state_sha256:
        raise Poly2DQualityCheckpointError("quality artifact binding model-state mismatch")
    if binding.training_profile_sha256 != metadata.quality_trainer_profile_sha256:
        raise Poly2DQualityCheckpointError("quality artifact binding trainer mismatch")
    if binding.dataset_manifest_sha256 != metadata.dataset_manifest_sha256:
        raise Poly2DQualityCheckpointError("quality artifact binding dataset mismatch")
    if binding.repository_sha != metadata.repository_sha:
        raise Poly2DQualityCheckpointError("quality artifact binding repository mismatch")
    if binding.runtime_fingerprint_sha256 != _quality_runtime_fingerprint(metadata.model_profile_sha256):
        raise Poly2DQualityCheckpointError("quality artifact binding runtime mismatch")
    if binding.tokenizer_fingerprint_sha256 != metadata.tokenizer_fingerprint_sha256:
        raise Poly2DQualityCheckpointError("quality artifact binding tokenizer mismatch")
    validate_artifact_binding(binding, _quality_registry_records())

    return LoadedPoly2DQualityCheckpoint(
        model=model,
        metadata=metadata,
        receipt=receipt,
        checkpoint_sha256=checkpoint_sha,
        metadata_sha256=metadata_sha,
        training_result_file_sha256=result_file_sha,
        receipt_sha256=receipt_sha,
    )


def persist_poly_2d_quality_checkpoint(
    directory: Path,
    *,
    run: Poly2DQualityTrainingRun,
    provenance: Poly2DQualityTrainingProvenance,
    quality_config: Poly2DQualityTrainingConfig,
    model_config: Poly2DTransformerConfig,
) -> LoadedPoly2DQualityCheckpoint:
    """Persist one selected B4 state as a non-overwriting verified B5 artifact."""

    if not isinstance(directory, Path):
        raise TypeError("directory must be pathlib.Path")
    if directory.exists() or directory.is_symlink():
        raise Poly2DQualityCheckpointError("quality checkpoint target must not already exist")
    if not directory.parent.is_dir() or directory.parent.is_symlink():
        raise Poly2DQualityCheckpointError("quality checkpoint parent must be a regular directory")
    if not isinstance(run, Poly2DQualityTrainingRun):
        raise TypeError("run must be Poly2DQualityTrainingRun")
    if not isinstance(provenance, Poly2DQualityTrainingProvenance):
        raise TypeError("provenance must be Poly2DQualityTrainingProvenance")
    if not isinstance(quality_config, Poly2DQualityTrainingConfig):
        raise TypeError("quality_config must be Poly2DQualityTrainingConfig")
    if not isinstance(model_config, Poly2DTransformerConfig):
        raise TypeError("model_config must be Poly2DTransformerConfig")

    result = run.result
    if model_state_sha256(run.model) != result.selected_state_sha256:
        raise Poly2DQualityCheckpointError("quality run model differs from selected result state")
    if result.provenance_sha256 != provenance.fingerprint():
        raise Poly2DQualityCheckpointError("quality run provenance mismatch")
    if result.repository_sha != provenance.repository_sha:
        raise Poly2DQualityCheckpointError("quality run repository mismatch")
    if result.dataset_manifest_sha256 != provenance.dataset_manifest_sha256:
        raise Poly2DQualityCheckpointError("quality run dataset mismatch")
    model_profile = poly_2d_config_fingerprint(model_config)
    trainer_profile = poly_2d_quality_trainer_fingerprint(quality_config, model_config)
    if result.model_profile_sha256 != model_profile:
        raise Poly2DQualityCheckpointError("quality run model profile mismatch")
    if result.quality_trainer_profile_sha256 != trainer_profile:
        raise Poly2DQualityCheckpointError("quality run trainer profile mismatch")

    result_payload = _quality_result_payload(result)
    result_bytes = _canonical_json_bytes(result_payload)
    result_fingerprint = sha256(result_bytes).hexdigest()
    if result_fingerprint != result.fingerprint():
        raise Poly2DQualityCheckpointError("quality result canonical fingerprint mismatch")

    metadata = Poly2DQualityCheckpointMetadata(
        repository_sha=provenance.repository_sha,
        registry_record_fingerprint_sha256=QUALITY_REGISTRY_RECORD.fingerprint(),
        dataset_manifest_sha256=provenance.dataset_manifest_sha256,
        preprocess_fingerprint_sha256=provenance.preprocess_fingerprint_sha256,
        model_profile_sha256=model_profile,
        quality_trainer_profile_sha256=trainer_profile,
        step_primitive_profile_sha256=result.step_primitive_profile_sha256,
        provenance_sha256=provenance.fingerprint(),
        training_plan_sha256=result.training_plan_sha256,
        training_result_fingerprint_sha256=result_fingerprint,
        tokenizer_fingerprint_sha256=tokenizer_fingerprint(),
        selected_state_sha256=result.selected_state_sha256,
        selected_epoch=result.selected_epoch,
        selected_validation_loss=result.selected_validation_loss,
        optimizer_steps=result.optimizer_steps,
        parameter_count=count_trainable_parameters(run.model),
        model_config=asdict(model_config),
        quality_training_config=asdict(quality_config),
        provenance=asdict(provenance),
    )
    metadata_bytes = _canonical_json_bytes(metadata.canonical_payload())

    temp = directory.parent / (
        f".{directory.name}.tmp-{result_fingerprint[:12]}"
    )
    if temp.exists() or temp.is_symlink():
        raise Poly2DQualityCheckpointError("quality checkpoint temporary path already exists")
    temp.mkdir()
    try:
        checkpoint_path = temp / QUALITY_CHECKPOINT_FILENAME
        metadata_path = temp / QUALITY_METADATA_FILENAME
        result_path = temp / QUALITY_TRAINING_RESULT_FILENAME
        receipt_path = temp / QUALITY_RECEIPT_FILENAME

        torch.save(run.model.state_dict(), checkpoint_path)
        metadata_path.write_bytes(metadata_bytes)
        result_path.write_bytes(result_bytes)

        checkpoint_sha = _sha256_file(
            checkpoint_path, MAX_QUALITY_CHECKPOINT_BYTES, "quality checkpoint"
        )
        metadata_sha = _sha256_file(
            metadata_path, MAX_QUALITY_METADATA_BYTES, "quality metadata"
        )
        result_file_sha = _sha256_file(
            result_path, MAX_QUALITY_RESULT_BYTES, "quality training result"
        )
        binding = ModelArtifactBinding(
            record_id=POLY_2D_QUALITY_REGISTRY_RECORD_ID,
            registry_record_fingerprint_sha256=QUALITY_REGISTRY_RECORD.fingerprint(),
            repository_sha=metadata.repository_sha,
            checkpoint_sha256=checkpoint_sha,
            model_fingerprint_sha256=metadata.selected_state_sha256,
            training_profile_sha256=metadata.quality_trainer_profile_sha256,
            dataset_manifest_sha256=metadata.dataset_manifest_sha256,
            runtime_fingerprint_sha256=_quality_runtime_fingerprint(
                metadata.model_profile_sha256
            ),
            tokenizer_fingerprint_sha256=metadata.tokenizer_fingerprint_sha256,
            tokenizer_version=POLYPHONIC_TOKENIZER_VERSION,
            representation_version=POLYPHONIC_REPRESENTATION_VERSION,
        )
        validate_artifact_binding(binding, _quality_registry_records())
        receipt = Poly2DQualityCheckpointReceipt(
            checkpoint_sha256=checkpoint_sha,
            metadata_sha256=metadata_sha,
            training_result_file_sha256=result_file_sha,
            metadata_fingerprint_sha256=metadata.fingerprint(),
            training_result_fingerprint_sha256=result_fingerprint,
            artifact_binding=binding.canonical_payload(),
            artifact_binding_sha256=binding.fingerprint(),
        )
        receipt_path.write_bytes(_canonical_json_bytes(receipt.canonical_payload()))
        load_and_verify_poly_2d_quality_checkpoint(temp)
        temp.replace(directory)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
        raise
    return load_and_verify_poly_2d_quality_checkpoint(directory)


def run_verified_poly_2d_quality_checkpoint_inference(
    checkpoint_directory: Path,
    image: torch.Tensor,
    *,
    max_decode_steps: int | None = None,
) -> Poly2DInferenceResult:
    """Verify B5 artifact then run the frozen B1 greedy decoder with exact binding."""

    loaded = load_and_verify_poly_2d_quality_checkpoint(checkpoint_directory)
    result = run_poly_2d_greedy_inference(
        loaded.model,
        image,
        max_decode_steps=max_decode_steps,
    )
    metadata = loaded.metadata
    if result.identity.model_state_sha256 != metadata.selected_state_sha256:
        raise Poly2DInferenceError("quality checkpoint state differs from inference model")
    if result.identity.model_profile_sha256 != metadata.model_profile_sha256:
        raise Poly2DInferenceError("quality checkpoint profile differs from inference profile")
    bound_identity = replace(
        result.identity,
        checkpoint_sha256=loaded.checkpoint_sha256,
        checkpoint_metadata_file_sha256=loaded.metadata_sha256,
        checkpoint_receipt_sha256=loaded.receipt_sha256,
        checkpoint_metadata_fingerprint_sha256=metadata.fingerprint(),
        dataset_manifest_sha256=metadata.dataset_manifest_sha256,
        preprocess_fingerprint_sha256=metadata.preprocess_fingerprint_sha256,
        trainer_profile_sha256=metadata.quality_trainer_profile_sha256,
        provenance_sha256=metadata.provenance_sha256,
        registry_record_fingerprint_sha256=metadata.registry_record_fingerprint_sha256,
        repository_sha=metadata.repository_sha,
    )
    return replace(result, identity=bound_identity)
