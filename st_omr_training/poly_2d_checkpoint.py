"""Exact one-shot checkpoint artifact contract for the tiny Polyphonic 2D Transformer.

TR-POLY-08B persists only a bounded research checkpoint produced through the
TR-POLY-08A training surface. The artifact is hash-bound, non-overwriting and
reloaded with ``weights_only=True`` before acceptance. This module does not load
repository datasets, open TEST, run a benchmark, promote a model, or grant
production authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from typing import Final, Mapping

import torch

from .dataset_manifest import DatasetSplit
from .model_registry import ModelLifecycle, ResearchAuthority, registry_by_id
from .poly_2d_training import (
    FROZEN_POLY_2D_TRAINING_CONFIG,
    POLY_2D_PROVENANCE_VERSION,
    POLY_2D_TRAINER_VERSION,
    Poly2DTrainingBatch,
    Poly2DTrainingConfig,
    Poly2DTrainingError,
    Poly2DTrainingProvenance,
    build_poly_2d_optimizer,
    evaluate_poly_2d_validation_loss,
    poly_2d_trainer_fingerprint,
    train_poly_2d_one_step,
)
from .poly_2d_transformer import (
    FROZEN_POLY_2D_CONFIG,
    POLY_2D_TRANSFORMER_VERSION,
    Poly2DTransformerConfig,
    TinyPoly2DTransformer,
    build_tiny_poly_2d_transformer,
    poly_2d_config_fingerprint,
)
from .polyphonic_representation import POLYPHONIC_REPRESENTATION_VERSION
from .polyphonic_serialization import POLYPHONIC_TOKENIZER_VERSION, tokenizer_fingerprint
from .training_model import (
    TORCH_PINNED_VERSION,
    assert_model_finite,
    count_trainable_parameters,
    model_state_sha256,
)


POLY_2D_CHECKPOINT_SCHEMA_VERSION: Final[str] = "st-omr-poly-2d-checkpoint-v1"
POLY_2D_CHECKPOINT_RECEIPT_VERSION: Final[str] = "st-omr-poly-2d-checkpoint-receipt-v1"
POLY_2D_REGISTRY_RECORD_ID: Final[str] = "candidate.poly-2d-transformer.v1"
CHECKPOINT_FILENAME: Final[str] = "model.pt"
METADATA_FILENAME: Final[str] = "metadata.json"
RECEIPT_FILENAME: Final[str] = "receipt.json"
MAX_CHECKPOINT_BYTES: Final[int] = 128 * 1024 * 1024
MAX_METADATA_BYTES: Final[int] = 256 * 1024
MAX_RECEIPT_BYTES: Final[int] = 16 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA40 = re.compile(r"^[0-9a-f]{40}$")


class Poly2DCheckpointError(Poly2DTrainingError):
    """Raised when a checkpoint artifact or reload verification fails closed."""


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
        raise Poly2DCheckpointError("checkpoint metadata is not canonical-JSON serializable") from exc


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Poly2DCheckpointError(f"{name} must be lowercase SHA-256 hex")
    return value


def _require_git_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _GIT_SHA40.fullmatch(value) is None:
        raise Poly2DCheckpointError(f"{name} must be lowercase git SHA-40 hex")
    return value


def _sha256_file(path: Path, maximum_bytes: int, name: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise Poly2DCheckpointError(f"{name} must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= maximum_bytes:
        raise Poly2DCheckpointError(f"{name} byte length is outside the checkpoint boundary")
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_canonical_json(path: Path, maximum_bytes: int, name: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Poly2DCheckpointError(f"{name} must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= maximum_bytes:
        raise Poly2DCheckpointError(f"{name} byte length is outside the checkpoint boundary")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Poly2DCheckpointError(f"{name} is not valid canonical JSON") from exc
    if not isinstance(payload, dict) or _canonical_json_bytes(payload) != raw:
        raise Poly2DCheckpointError(f"{name} must be canonical JSON object bytes")
    return payload


def _strict_keys(payload: Mapping[str, object], expected: frozenset[str], name: str) -> None:
    actual = frozenset(payload.keys())
    if actual != expected:
        raise Poly2DCheckpointError(
            f"{name} key set mismatch: missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


@dataclass(frozen=True, slots=True)
class Poly2DCheckpointMetadata:
    repository_sha: str
    registry_record_fingerprint_sha256: str
    dataset_manifest_sha256: str
    preprocess_fingerprint_sha256: str
    model_profile_sha256: str
    trainer_profile_sha256: str
    provenance_sha256: str
    tokenizer_fingerprint_sha256: str
    final_state_sha256: str
    parameter_count: int
    optimizer_steps: int
    model_config: dict[str, object]
    training_config: dict[str, object]
    provenance: dict[str, object]
    model_version: str = POLY_2D_TRANSFORMER_VERSION
    trainer_version: str = POLY_2D_TRAINER_VERSION
    provenance_version: str = POLY_2D_PROVENANCE_VERSION
    representation_version: str = POLYPHONIC_REPRESENTATION_VERSION
    tokenizer_version: str = POLYPHONIC_TOKENIZER_VERSION
    torch_version: str = TORCH_PINNED_VERSION
    checkpoint_role: str = "bounded_research_artifact_only"
    authoritative_dataset_execution: bool = False
    test_split_accessed: bool = False
    benchmark_evidence: bool = False
    production_authority: bool = False
    schema_version: str = POLY_2D_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_git_sha(self.repository_sha, "repository_sha")
        for name, value in (
            ("registry_record_fingerprint_sha256", self.registry_record_fingerprint_sha256),
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
            ("preprocess_fingerprint_sha256", self.preprocess_fingerprint_sha256),
            ("model_profile_sha256", self.model_profile_sha256),
            ("trainer_profile_sha256", self.trainer_profile_sha256),
            ("provenance_sha256", self.provenance_sha256),
            ("tokenizer_fingerprint_sha256", self.tokenizer_fingerprint_sha256),
            ("final_state_sha256", self.final_state_sha256),
        ):
            _require_sha256(value, name)
        if not isinstance(self.parameter_count, int) or isinstance(self.parameter_count, bool) or self.parameter_count <= 0:
            raise Poly2DCheckpointError("parameter_count must be a positive integer")
        if not isinstance(self.optimizer_steps, int) or isinstance(self.optimizer_steps, bool) or not 1 <= self.optimizer_steps <= 2:
            raise Poly2DCheckpointError("optimizer_steps is outside the bounded training contract")
        if self.model_version != POLY_2D_TRANSFORMER_VERSION:
            raise Poly2DCheckpointError("checkpoint model version mismatch")
        if self.trainer_version != POLY_2D_TRAINER_VERSION:
            raise Poly2DCheckpointError("checkpoint trainer version mismatch")
        if self.provenance_version != POLY_2D_PROVENANCE_VERSION:
            raise Poly2DCheckpointError("checkpoint provenance version mismatch")
        if self.representation_version != POLYPHONIC_REPRESENTATION_VERSION:
            raise Poly2DCheckpointError("checkpoint representation version mismatch")
        if self.tokenizer_version != POLYPHONIC_TOKENIZER_VERSION:
            raise Poly2DCheckpointError("checkpoint tokenizer version mismatch")
        if self.torch_version != TORCH_PINNED_VERSION:
            raise Poly2DCheckpointError("checkpoint torch version mismatch")
        if self.tokenizer_fingerprint_sha256 != tokenizer_fingerprint():
            raise Poly2DCheckpointError("checkpoint tokenizer fingerprint mismatch")
        if self.checkpoint_role != "bounded_research_artifact_only":
            raise Poly2DCheckpointError("checkpoint role is frozen to bounded research evidence")
        for name in (
            "authoritative_dataset_execution",
            "test_split_accessed",
            "benchmark_evidence",
            "production_authority",
        ):
            value = getattr(self, name)
            if not isinstance(value, bool) or value:
                raise Poly2DCheckpointError(f"{name} must remain false in TR-POLY-08B")
        if self.schema_version != POLY_2D_CHECKPOINT_SCHEMA_VERSION:
            raise Poly2DCheckpointError("checkpoint schema version mismatch")

        try:
            model_config = Poly2DTransformerConfig(**self.model_config)
            training_config = Poly2DTrainingConfig(**self.training_config)
            provenance = Poly2DTrainingProvenance(**self.provenance)
        except (TypeError, ValueError, Poly2DTrainingError) as exc:
            raise Poly2DCheckpointError("checkpoint embedded config/provenance is invalid") from exc
        if poly_2d_config_fingerprint(model_config) != self.model_profile_sha256:
            raise Poly2DCheckpointError("checkpoint model profile mismatch")
        if poly_2d_trainer_fingerprint(training_config, model_config) != self.trainer_profile_sha256:
            raise Poly2DCheckpointError("checkpoint trainer profile mismatch")
        if provenance.fingerprint() != self.provenance_sha256:
            raise Poly2DCheckpointError("checkpoint provenance fingerprint mismatch")
        if provenance.repository_sha != self.repository_sha:
            raise Poly2DCheckpointError("checkpoint repository SHA differs from provenance")
        if provenance.dataset_manifest_sha256 != self.dataset_manifest_sha256:
            raise Poly2DCheckpointError("checkpoint dataset identity differs from provenance")
        if provenance.preprocess_fingerprint_sha256 != self.preprocess_fingerprint_sha256:
            raise Poly2DCheckpointError("checkpoint preprocess identity differs from provenance")
        if provenance.model_profile_sha256 != self.model_profile_sha256:
            raise Poly2DCheckpointError("checkpoint model profile differs from provenance")
        if provenance.trainer_profile_sha256 != self.trainer_profile_sha256:
            raise Poly2DCheckpointError("checkpoint trainer profile differs from provenance")

        record = registry_by_id()[POLY_2D_REGISTRY_RECORD_ID]
        if record.lifecycle is not ModelLifecycle.ARCHITECTURE_ONLY or record.authority is not ResearchAuthority.NONE:
            raise Poly2DCheckpointError("TR-POLY-08B requires the registry checkpoint gate to remain closed")
        if record.fingerprint() != self.registry_record_fingerprint_sha256:
            raise Poly2DCheckpointError("checkpoint registry-record fingerprint mismatch")

    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        return sha256(_canonical_json_bytes(self.canonical_payload())).hexdigest()


_METADATA_KEYS: Final[frozenset[str]] = frozenset(Poly2DCheckpointMetadata.__dataclass_fields__.keys())
_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "receipt_version",
        "checkpoint_filename",
        "metadata_filename",
        "checkpoint_sha256",
        "metadata_sha256",
    }
)
_CHECKPOINT_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset({"schema_version", "metadata", "state_dict"})


@dataclass(frozen=True, slots=True)
class Poly2DCheckpointReceipt:
    artifact_directory: Path
    checkpoint_sha256: str
    metadata_sha256: str
    receipt_sha256: str
    state_sha256: str
    metadata_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_directory, Path):
            raise Poly2DCheckpointError("artifact_directory must be pathlib.Path")
        for name, value in (
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("metadata_sha256", self.metadata_sha256),
            ("receipt_sha256", self.receipt_sha256),
            ("state_sha256", self.state_sha256),
            ("metadata_fingerprint_sha256", self.metadata_fingerprint_sha256),
        ):
            _require_sha256(value, name)


@dataclass(slots=True)
class LoadedPoly2DCheckpoint:
    model: TinyPoly2DTransformer
    metadata: Poly2DCheckpointMetadata
    checkpoint_sha256: str
    metadata_sha256: str
    receipt_sha256: str


def _metadata_from_payload(payload: object) -> Poly2DCheckpointMetadata:
    if not isinstance(payload, Mapping):
        raise Poly2DCheckpointError("checkpoint metadata payload must be an object")
    _strict_keys(payload, _METADATA_KEYS, "checkpoint metadata")
    try:
        return Poly2DCheckpointMetadata(**dict(payload))
    except (TypeError, ValueError, Poly2DTrainingError) as exc:
        if isinstance(exc, Poly2DCheckpointError):
            raise
        raise Poly2DCheckpointError("checkpoint metadata payload is invalid") from exc


def _receipt_payload(checkpoint_sha256: str, metadata_sha256: str) -> dict[str, object]:
    return {
        "receipt_version": POLY_2D_CHECKPOINT_RECEIPT_VERSION,
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "metadata_filename": METADATA_FILENAME,
        "checkpoint_sha256": _require_sha256(checkpoint_sha256, "checkpoint_sha256"),
        "metadata_sha256": _require_sha256(metadata_sha256, "metadata_sha256"),
    }


def _validate_artifact_directory(directory: Path) -> tuple[Path, Path, Path]:
    if not isinstance(directory, Path):
        raise TypeError("directory must be pathlib.Path")
    if directory.is_symlink() or not directory.is_dir():
        raise Poly2DCheckpointError("checkpoint artifact directory must be a regular non-symlink directory")
    checkpoint_path = directory / CHECKPOINT_FILENAME
    metadata_path = directory / METADATA_FILENAME
    receipt_path = directory / RECEIPT_FILENAME
    return checkpoint_path, metadata_path, receipt_path


def load_and_verify_poly_2d_checkpoint(directory: Path) -> LoadedPoly2DCheckpoint:
    checkpoint_path, metadata_path, receipt_path = _validate_artifact_directory(directory)
    receipt_payload = _read_canonical_json(receipt_path, MAX_RECEIPT_BYTES, "checkpoint receipt")
    _strict_keys(receipt_payload, _RECEIPT_KEYS, "checkpoint receipt")
    if receipt_payload.get("receipt_version") != POLY_2D_CHECKPOINT_RECEIPT_VERSION:
        raise Poly2DCheckpointError("checkpoint receipt version mismatch")
    if receipt_payload.get("checkpoint_filename") != CHECKPOINT_FILENAME:
        raise Poly2DCheckpointError("checkpoint receipt filename mismatch")
    if receipt_payload.get("metadata_filename") != METADATA_FILENAME:
        raise Poly2DCheckpointError("checkpoint metadata filename mismatch")
    expected_checkpoint_sha = _require_sha256(receipt_payload.get("checkpoint_sha256"), "checkpoint_sha256")
    expected_metadata_sha = _require_sha256(receipt_payload.get("metadata_sha256"), "metadata_sha256")

    actual_checkpoint_sha = _sha256_file(checkpoint_path, MAX_CHECKPOINT_BYTES, "checkpoint")
    actual_metadata_sha = _sha256_file(metadata_path, MAX_METADATA_BYTES, "checkpoint metadata")
    if actual_checkpoint_sha != expected_checkpoint_sha:
        raise Poly2DCheckpointError("checkpoint file SHA-256 mismatch")
    if actual_metadata_sha != expected_metadata_sha:
        raise Poly2DCheckpointError("checkpoint metadata SHA-256 mismatch")

    metadata_payload = _read_canonical_json(metadata_path, MAX_METADATA_BYTES, "checkpoint metadata")
    metadata = _metadata_from_payload(metadata_payload)

    try:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Poly2DCheckpointError("checkpoint could not be loaded with weights_only=True") from exc
    if not isinstance(checkpoint_payload, Mapping):
        raise Poly2DCheckpointError("checkpoint payload must be an object")
    _strict_keys(checkpoint_payload, _CHECKPOINT_PAYLOAD_KEYS, "checkpoint payload")
    if checkpoint_payload.get("schema_version") != POLY_2D_CHECKPOINT_SCHEMA_VERSION:
        raise Poly2DCheckpointError("checkpoint payload schema version mismatch")
    embedded_metadata = checkpoint_payload.get("metadata")
    if embedded_metadata != metadata.canonical_payload():
        raise Poly2DCheckpointError("checkpoint embedded metadata differs from canonical sidecar")
    state_dict = checkpoint_payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise Poly2DCheckpointError("checkpoint state_dict must be a non-empty object")
    if any(not isinstance(name, str) or not isinstance(value, torch.Tensor) for name, value in state_dict.items()):
        raise Poly2DCheckpointError("checkpoint state_dict must contain only string tensor entries")

    try:
        model_config = Poly2DTransformerConfig(**metadata.model_config)
    except (TypeError, ValueError) as exc:
        raise Poly2DCheckpointError("checkpoint model config cannot be reconstructed") from exc
    model = build_tiny_poly_2d_transformer(model_config, seed=0)
    try:
        incompatibility = model.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as exc:
        raise Poly2DCheckpointError("checkpoint state_dict is incompatible with the model config") from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise Poly2DCheckpointError("checkpoint state_dict did not load strictly")
    assert_model_finite(model)
    if count_trainable_parameters(model) != metadata.parameter_count:
        raise Poly2DCheckpointError("checkpoint parameter count mismatch")
    actual_state_sha = model_state_sha256(model)
    if actual_state_sha != metadata.final_state_sha256:
        raise Poly2DCheckpointError("checkpoint reloaded model state SHA-256 mismatch")

    return LoadedPoly2DCheckpoint(
        model=model,
        metadata=metadata,
        checkpoint_sha256=actual_checkpoint_sha,
        metadata_sha256=actual_metadata_sha,
        receipt_sha256=_sha256_file(receipt_path, MAX_RECEIPT_BYTES, "checkpoint receipt"),
    )


def _preflight_output_directory(output_directory: Path) -> tuple[Path, Path]:
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    parent = output_directory.parent
    if parent.is_symlink() or not parent.is_dir():
        raise Poly2DCheckpointError("checkpoint parent must be an existing non-symlink directory")
    temporary = parent / f".{output_directory.name}.tmp"
    if output_directory.exists() or output_directory.is_symlink():
        raise Poly2DCheckpointError("checkpoint artifact directory already exists; overwrite is forbidden")
    if temporary.exists() or temporary.is_symlink():
        raise Poly2DCheckpointError("checkpoint temporary directory already exists")
    return parent, temporary


def _write_checkpoint_artifact(
    *,
    model: TinyPoly2DTransformer,
    metadata: Poly2DCheckpointMetadata,
    output_directory: Path,
) -> Poly2DCheckpointReceipt:
    _parent, temporary = _preflight_output_directory(output_directory)
    assert_model_finite(model)
    if model_state_sha256(model) != metadata.final_state_sha256:
        raise Poly2DCheckpointError("model state differs from checkpoint metadata before persistence")
    if count_trainable_parameters(model) != metadata.parameter_count:
        raise Poly2DCheckpointError("model parameter count differs from checkpoint metadata")

    temporary.mkdir()
    try:
        checkpoint_path = temporary / CHECKPOINT_FILENAME
        metadata_path = temporary / METADATA_FILENAME
        receipt_path = temporary / RECEIPT_FILENAME
        metadata_bytes = _canonical_json_bytes(metadata.canonical_payload())
        if not 1 <= len(metadata_bytes) <= MAX_METADATA_BYTES:
            raise Poly2DCheckpointError("checkpoint metadata exceeds byte boundary")
        metadata_path.write_bytes(metadata_bytes)

        state_dict = {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in model.state_dict().items()
        }
        checkpoint_payload = {
            "schema_version": POLY_2D_CHECKPOINT_SCHEMA_VERSION,
            "metadata": metadata.canonical_payload(),
            "state_dict": state_dict,
        }
        torch.save(checkpoint_payload, checkpoint_path)
        checkpoint_sha = _sha256_file(checkpoint_path, MAX_CHECKPOINT_BYTES, "checkpoint")
        metadata_sha = _sha256_file(metadata_path, MAX_METADATA_BYTES, "checkpoint metadata")
        receipt_bytes = _canonical_json_bytes(_receipt_payload(checkpoint_sha, metadata_sha))
        if not 1 <= len(receipt_bytes) <= MAX_RECEIPT_BYTES:
            raise Poly2DCheckpointError("checkpoint receipt exceeds byte boundary")
        receipt_path.write_bytes(receipt_bytes)

        loaded = load_and_verify_poly_2d_checkpoint(temporary)
        if loaded.metadata.final_state_sha256 != metadata.final_state_sha256:
            raise Poly2DCheckpointError("temporary checkpoint reload changed state identity")
        temporary.rename(output_directory)
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise

    loaded_final = load_and_verify_poly_2d_checkpoint(output_directory)
    return Poly2DCheckpointReceipt(
        artifact_directory=output_directory,
        checkpoint_sha256=loaded_final.checkpoint_sha256,
        metadata_sha256=loaded_final.metadata_sha256,
        receipt_sha256=loaded_final.receipt_sha256,
        state_sha256=loaded_final.metadata.final_state_sha256,
        metadata_fingerprint_sha256=loaded_final.metadata.fingerprint(),
    )


def run_and_persist_bounded_poly_2d_checkpoint(
    *,
    train_batches: tuple[Poly2DTrainingBatch, ...],
    validation_batch: Poly2DTrainingBatch,
    provenance: Poly2DTrainingProvenance,
    output_directory: Path,
    training_config: Poly2DTrainingConfig = FROZEN_POLY_2D_TRAINING_CONFIG,
    model_config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
) -> Poly2DCheckpointReceipt:
    _preflight_output_directory(output_directory)
    if not isinstance(train_batches, tuple) or not train_batches:
        raise Poly2DCheckpointError("train_batches must be a non-empty immutable tuple")
    if any(not isinstance(batch, Poly2DTrainingBatch) or batch.split is not DatasetSplit.TRAIN for batch in train_batches):
        raise Poly2DCheckpointError("checkpoint training may consume only TRAIN batches")
    if not isinstance(validation_batch, Poly2DTrainingBatch) or validation_batch.split is not DatasetSplit.VALIDATION:
        raise Poly2DCheckpointError("checkpoint validation batch must be VALIDATION")
    if not isinstance(provenance, Poly2DTrainingProvenance):
        raise TypeError("provenance must be Poly2DTrainingProvenance")

    expected_model_profile = poly_2d_config_fingerprint(model_config)
    expected_trainer_profile = poly_2d_trainer_fingerprint(training_config, model_config)
    if provenance.model_profile_sha256 != expected_model_profile:
        raise Poly2DCheckpointError("checkpoint provenance model profile mismatch")
    if provenance.trainer_profile_sha256 != expected_trainer_profile:
        raise Poly2DCheckpointError("checkpoint provenance trainer profile mismatch")
    if any(
        batch.dataset_manifest_sha256 != provenance.dataset_manifest_sha256
        for batch in train_batches + (validation_batch,)
    ):
        raise Poly2DCheckpointError("checkpoint batch dataset identity differs from provenance")

    model = build_tiny_poly_2d_transformer(model_config, seed=training_config.master_seed)
    optimizer = build_poly_2d_optimizer(model, training_config)
    for step in range(training_config.smoke_steps):
        train_poly_2d_one_step(
            model,
            train_batches[step % len(train_batches)],
            optimizer,
            training_config,
        )
    _validation_loss = evaluate_poly_2d_validation_loss(model, validation_batch)
    final_state_sha = model_state_sha256(model)
    record = registry_by_id()[POLY_2D_REGISTRY_RECORD_ID]
    metadata = Poly2DCheckpointMetadata(
        repository_sha=provenance.repository_sha,
        registry_record_fingerprint_sha256=record.fingerprint(),
        dataset_manifest_sha256=provenance.dataset_manifest_sha256,
        preprocess_fingerprint_sha256=provenance.preprocess_fingerprint_sha256,
        model_profile_sha256=expected_model_profile,
        trainer_profile_sha256=expected_trainer_profile,
        provenance_sha256=provenance.fingerprint(),
        tokenizer_fingerprint_sha256=tokenizer_fingerprint(),
        final_state_sha256=final_state_sha,
        parameter_count=count_trainable_parameters(model),
        optimizer_steps=training_config.smoke_steps,
        model_config=asdict(model_config),
        training_config=asdict(training_config),
        provenance=provenance.canonical_payload(),
    )
    return _write_checkpoint_artifact(
        model=model,
        metadata=metadata,
        output_directory=output_directory,
    )
