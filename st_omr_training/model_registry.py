"""Deterministic research model registry for ST-OMR.

TR-POLY-07 consolidates model-family identity, lifecycle, target contracts and
checkpoint evidence boundaries without changing any trainer, checkpoint or
runtime path.  Registry presence is descriptive only: it never grants
production authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
import re
from typing import Final


MODEL_REGISTRY_VERSION: Final[str] = "st-omr-model-registry-v1"
MODEL_CARD_SCHEMA_VERSION: Final[str] = "st-omr-model-card-v1"
MODEL_ARTIFACT_BINDING_VERSION: Final[str] = "st-omr-model-artifact-binding-v1"

V1_TOKENIZER_VERSION: Final[str] = "st-omr-semantic-tokenizer-v1"
V2_REPRESENTATION_VERSION: Final[str] = "st-omr-polyphonic-representation-v2"
V2_TOKENIZER_VERSION: Final[str] = "st-omr-polyphonic-tokenizer-v1"
POLY_EVALUATION_CONTRACT_VERSION: Final[str] = "st-omr-poly-evaluation-contract-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_ASCII_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]*$")


class ModelRegistryError(ValueError):
    """Raised when a registry or artifact binding fails closed."""


class ModelKind(str, Enum):
    SEQUENCE_BASELINE = "sequence_baseline"
    VISUAL_SPECIALIST = "visual_specialist"
    LOCAL_REFINER = "local_refiner"
    DETERMINISTIC_FUSION = "deterministic_fusion"
    CANDIDATE_FAMILY = "candidate_family"


class ModelLifecycle(str, Enum):
    FROZEN_REFERENCE = "frozen_reference"
    TRAINING_IMPLEMENTED = "training_implemented"
    ARCHITECTURE_ONLY = "architecture_only"
    PLANNED = "planned"
    DETERMINISTIC = "deterministic"


class ResearchAuthority(str, Enum):
    REFERENCE_ONLY = "reference_only"
    EXPERIMENTAL = "experimental"
    SHADOW_ONLY = "shadow_only"
    NONE = "none"


class SemanticScope(str, Enum):
    V1_SINGLE_VOICE = "v1_single_voice"
    LOCAL_VISUAL_EVIDENCE = "local_visual_evidence"
    DETERMINISTIC_V1_FUSION = "deterministic_v1_fusion"
    POLYPHONIC_V2 = "polyphonic_v2"


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_ascii_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _ASCII_ID.fullmatch(value) is None:
        raise ModelRegistryError(f"{name} must be a canonical lowercase ASCII identifier")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ModelRegistryError(f"{name} must be lowercase SHA-256 hex")
    return value


def _require_git_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _GIT_SHA40.fullmatch(value) is None:
        raise ModelRegistryError(f"{name} must be lowercase git SHA-40 hex")
    return value


@dataclass(frozen=True, slots=True)
class ModelRegistryRecord:
    record_id: str
    model_kind: ModelKind
    lifecycle: ModelLifecycle
    authority: ResearchAuthority
    semantic_scope: SemanticScope
    source_module: str
    source_version: str
    task_ids: tuple[str, ...]
    tokenizer_version: str | None = None
    representation_version: str | None = None
    checkpoint_required_for_evidence: bool = True
    polyphonic_v2_capable: bool = False
    production_authority: bool = False

    def __post_init__(self) -> None:
        _require_ascii_identifier(self.record_id, "record_id")
        if not isinstance(self.model_kind, ModelKind):
            raise ModelRegistryError("model_kind must be ModelKind")
        if not isinstance(self.lifecycle, ModelLifecycle):
            raise ModelRegistryError("lifecycle must be ModelLifecycle")
        if not isinstance(self.authority, ResearchAuthority):
            raise ModelRegistryError("authority must be ResearchAuthority")
        if not isinstance(self.semantic_scope, SemanticScope):
            raise ModelRegistryError("semantic_scope must be SemanticScope")
        if not isinstance(self.source_module, str) or not self.source_module.startswith("st_omr_training."):
            raise ModelRegistryError("source_module must name an st_omr_training module")
        if not isinstance(self.source_version, str) or not self.source_version or not self.source_version.isascii():
            raise ModelRegistryError("source_version must be non-empty ASCII")
        if (
            not isinstance(self.task_ids, tuple)
            or not self.task_ids
            or len(set(self.task_ids)) != len(self.task_ids)
        ):
            raise ModelRegistryError("task_ids must be a non-empty unique tuple")
        for task_id in self.task_ids:
            _require_ascii_identifier(task_id, "task_id")
        for name, value in (
            ("tokenizer_version", self.tokenizer_version),
            ("representation_version", self.representation_version),
        ):
            if value is not None and (not isinstance(value, str) or not value or not value.isascii()):
                raise ModelRegistryError(f"{name} must be None or non-empty ASCII")
        for name, value in (
            ("checkpoint_required_for_evidence", self.checkpoint_required_for_evidence),
            ("polyphonic_v2_capable", self.polyphonic_v2_capable),
            ("production_authority", self.production_authority),
        ):
            if not isinstance(value, bool):
                raise ModelRegistryError(f"{name} must be bool")
        if self.production_authority:
            raise ModelRegistryError("ST-OMR research registry never grants production authority")
        if self.lifecycle in (ModelLifecycle.ARCHITECTURE_ONLY, ModelLifecycle.PLANNED):
            if self.authority is not ResearchAuthority.NONE:
                raise ModelRegistryError("unimplemented records cannot carry research inference authority")
            if not self.checkpoint_required_for_evidence:
                raise ModelRegistryError("unimplemented learned records must require a future checkpoint")
        if self.lifecycle is ModelLifecycle.DETERMINISTIC:
            if self.model_kind is not ModelKind.DETERMINISTIC_FUSION:
                raise ModelRegistryError("deterministic lifecycle is reserved for deterministic fusion")
            if self.authority is not ResearchAuthority.NONE or self.checkpoint_required_for_evidence:
                raise ModelRegistryError("deterministic fusion cannot claim learned authority or checkpoint evidence")
        if self.polyphonic_v2_capable:
            if self.semantic_scope is not SemanticScope.POLYPHONIC_V2:
                raise ModelRegistryError("polyphonic_v2_capable requires POLYPHONIC_V2 semantic scope")
            if self.representation_version != V2_REPRESENTATION_VERSION:
                raise ModelRegistryError("polyphonic V2 records must bind the frozen V2 representation")
            if self.tokenizer_version != V2_TOKENIZER_VERSION:
                raise ModelRegistryError("polyphonic V2 learned records must bind the frozen V2 tokenizer")

    def canonical_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model_kind"] = self.model_kind.value
        payload["lifecycle"] = self.lifecycle.value
        payload["authority"] = self.authority.value
        payload["semantic_scope"] = self.semantic_scope.value
        payload["task_ids"] = list(self.task_ids)
        return payload

    def fingerprint(self) -> str:
        return sha256(_canonical_json_bytes(self.canonical_payload())).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelArtifactBinding:
    """Exact checkpoint/evidence identity for one implemented learned record."""

    record_id: str
    repository_sha: str
    checkpoint_sha256: str
    model_fingerprint_sha256: str
    training_profile_sha256: str
    dataset_manifest_sha256: str
    runtime_fingerprint_sha256: str
    tokenizer_fingerprint_sha256: str | None
    tokenizer_version: str | None
    representation_version: str | None
    binding_version: str = MODEL_ARTIFACT_BINDING_VERSION

    def __post_init__(self) -> None:
        _require_ascii_identifier(self.record_id, "record_id")
        _require_git_sha(self.repository_sha, "repository_sha")
        for name, value in (
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("model_fingerprint_sha256", self.model_fingerprint_sha256),
            ("training_profile_sha256", self.training_profile_sha256),
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
            ("runtime_fingerprint_sha256", self.runtime_fingerprint_sha256),
        ):
            _require_sha256(value, name)
        if self.tokenizer_fingerprint_sha256 is not None:
            _require_sha256(self.tokenizer_fingerprint_sha256, "tokenizer_fingerprint_sha256")
        for name, value in (
            ("tokenizer_version", self.tokenizer_version),
            ("representation_version", self.representation_version),
        ):
            if value is not None and (not isinstance(value, str) or not value or not value.isascii()):
                raise ModelRegistryError(f"{name} must be None or non-empty ASCII")
        if self.binding_version != MODEL_ARTIFACT_BINDING_VERSION:
            raise ModelRegistryError("model artifact binding version mismatch")

    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        return sha256(_canonical_json_bytes(self.canonical_payload())).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelEvidenceBinding:
    """Binds one exact artifact to one exact benchmark result surface."""

    artifact_binding_sha256: str
    benchmark_identity_sha256: str
    metrics_sha256: str
    evaluation_contract_version: str = POLY_EVALUATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(self.artifact_binding_sha256, "artifact_binding_sha256")
        _require_sha256(self.benchmark_identity_sha256, "benchmark_identity_sha256")
        _require_sha256(self.metrics_sha256, "metrics_sha256")
        if self.evaluation_contract_version != POLY_EVALUATION_CONTRACT_VERSION:
            raise ModelRegistryError("evaluation contract version mismatch")

    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        return sha256(_canonical_json_bytes(self.canonical_payload())).hexdigest()


SEED_MODEL_REGISTRY: Final[tuple[ModelRegistryRecord, ...]] = (
    ModelRegistryRecord(
        record_id="baseline.cnn-gru.v1",
        model_kind=ModelKind.SEQUENCE_BASELINE,
        lifecycle=ModelLifecycle.FROZEN_REFERENCE,
        authority=ResearchAuthority.REFERENCE_ONLY,
        semantic_scope=SemanticScope.V1_SINGLE_VOICE,
        source_module="st_omr_training.training_model",
        source_version="st-omr-cnn-gru-baseline-v1",
        task_ids=("sequence_omr",),
        tokenizer_version=V1_TOKENIZER_VERSION,
        checkpoint_required_for_evidence=True,
    ),
    ModelRegistryRecord(
        record_id="specialist.staff.d7.v1",
        model_kind=ModelKind.VISUAL_SPECIALIST,
        lifecycle=ModelLifecycle.TRAINING_IMPLEMENTED,
        authority=ResearchAuthority.EXPERIMENTAL,
        semantic_scope=SemanticScope.LOCAL_VISUAL_EVIDENCE,
        source_module="st_omr_training.stage7d7_specialist_training",
        source_version="stage7d7-staff-dense-segmentation-v1",
        task_ids=("staff_geometry",),
        checkpoint_required_for_evidence=True,
    ),
    ModelRegistryRecord(
        record_id="specialist.structure.d7.v1",
        model_kind=ModelKind.VISUAL_SPECIALIST,
        lifecycle=ModelLifecycle.TRAINING_IMPLEMENTED,
        authority=ResearchAuthority.EXPERIMENTAL,
        semantic_scope=SemanticScope.LOCAL_VISUAL_EVIDENCE,
        source_module="st_omr_training.stage7d7_specialist_training",
        source_version="stage7d7-structure-dense-segmentation-v1",
        task_ids=("structure",),
        checkpoint_required_for_evidence=True,
    ),
    ModelRegistryRecord(
        record_id="refiner.barline.d11.v1",
        model_kind=ModelKind.LOCAL_REFINER,
        lifecycle=ModelLifecycle.TRAINING_IMPLEMENTED,
        authority=ResearchAuthority.SHADOW_ONLY,
        semantic_scope=SemanticScope.LOCAL_VISUAL_EVIDENCE,
        source_module="st_omr_training.stage7d11_barline_meter_training",
        source_version="stage7d11-barline-refiner-v1",
        task_ids=("barline",),
        checkpoint_required_for_evidence=True,
    ),
    ModelRegistryRecord(
        record_id="refiner.meter.d11.v1",
        model_kind=ModelKind.LOCAL_REFINER,
        lifecycle=ModelLifecycle.TRAINING_IMPLEMENTED,
        authority=ResearchAuthority.SHADOW_ONLY,
        semantic_scope=SemanticScope.LOCAL_VISUAL_EVIDENCE,
        source_module="st_omr_training.stage7d11_barline_meter_training",
        source_version="stage7d11-meter-refiner-v1",
        task_ids=("meter",),
        checkpoint_required_for_evidence=True,
    ),
    ModelRegistryRecord(
        record_id="specialist.notehead.declared.v1",
        model_kind=ModelKind.VISUAL_SPECIALIST,
        lifecycle=ModelLifecycle.ARCHITECTURE_ONLY,
        authority=ResearchAuthority.NONE,
        semantic_scope=SemanticScope.LOCAL_VISUAL_EVIDENCE,
        source_module="st_omr_training.stage7d4_specialist_architecture",
        source_version="stage7d4-specialist-omr-architecture-v1",
        task_ids=("notehead",),
    ),
    ModelRegistryRecord(
        record_id="specialist.rest.declared.v1",
        model_kind=ModelKind.VISUAL_SPECIALIST,
        lifecycle=ModelLifecycle.ARCHITECTURE_ONLY,
        authority=ResearchAuthority.NONE,
        semantic_scope=SemanticScope.LOCAL_VISUAL_EVIDENCE,
        source_module="st_omr_training.stage7d4_specialist_architecture",
        source_version="stage7d4-specialist-omr-architecture-v1",
        task_ids=("rest",),
    ),
    ModelRegistryRecord(
        record_id="specialist.accidental.declared.v1",
        model_kind=ModelKind.VISUAL_SPECIALIST,
        lifecycle=ModelLifecycle.ARCHITECTURE_ONLY,
        authority=ResearchAuthority.NONE,
        semantic_scope=SemanticScope.LOCAL_VISUAL_EVIDENCE,
        source_module="st_omr_training.stage7d4_specialist_architecture",
        source_version="stage7d4-specialist-omr-architecture-v1",
        task_ids=("accidental",),
    ),
    ModelRegistryRecord(
        record_id="specialist.rhythm.declared.v1",
        model_kind=ModelKind.VISUAL_SPECIALIST,
        lifecycle=ModelLifecycle.ARCHITECTURE_ONLY,
        authority=ResearchAuthority.NONE,
        semantic_scope=SemanticScope.LOCAL_VISUAL_EVIDENCE,
        source_module="st_omr_training.stage7d4_specialist_architecture",
        source_version="stage7d4-specialist-omr-architecture-v1",
        task_ids=("rhythm",),
    ),
    ModelRegistryRecord(
        record_id="specialist.staff-position.declared.v1",
        model_kind=ModelKind.VISUAL_SPECIALIST,
        lifecycle=ModelLifecycle.ARCHITECTURE_ONLY,
        authority=ResearchAuthority.NONE,
        semantic_scope=SemanticScope.LOCAL_VISUAL_EVIDENCE,
        source_module="st_omr_training.stage7d4_specialist_architecture",
        source_version="stage7d4-specialist-omr-architecture-v1",
        task_ids=("staff_position",),
    ),
    ModelRegistryRecord(
        record_id="specialist.chord-grouping.declared.v1",
        model_kind=ModelKind.VISUAL_SPECIALIST,
        lifecycle=ModelLifecycle.ARCHITECTURE_ONLY,
        authority=ResearchAuthority.NONE,
        semantic_scope=SemanticScope.LOCAL_VISUAL_EVIDENCE,
        source_module="st_omr_training.stage7d4_specialist_architecture",
        source_version="stage7d4-specialist-omr-architecture-v1",
        task_ids=("chord_grouping",),
    ),
    ModelRegistryRecord(
        record_id="fusion.context-validator.v1",
        model_kind=ModelKind.DETERMINISTIC_FUSION,
        lifecycle=ModelLifecycle.DETERMINISTIC,
        authority=ResearchAuthority.NONE,
        semantic_scope=SemanticScope.DETERMINISTIC_V1_FUSION,
        source_module="st_omr_training.stage7d4_specialist_architecture",
        source_version="stage7d4-specialist-omr-architecture-v1",
        task_ids=("context_validation",),
        checkpoint_required_for_evidence=False,
    ),
    ModelRegistryRecord(
        record_id="candidate.poly-2d-transformer.v1",
        model_kind=ModelKind.CANDIDATE_FAMILY,
        lifecycle=ModelLifecycle.PLANNED,
        authority=ResearchAuthority.NONE,
        semantic_scope=SemanticScope.POLYPHONIC_V2,
        source_module="st_omr_training.model_registry",
        source_version="tr-poly-08-planned-2d-transformer-v1",
        task_ids=("polyphonic_sequence_omr",),
        tokenizer_version=V2_TOKENIZER_VERSION,
        representation_version=V2_REPRESENTATION_VERSION,
        checkpoint_required_for_evidence=True,
        polyphonic_v2_capable=True,
    ),
    ModelRegistryRecord(
        record_id="candidate.relation-graph.v1",
        model_kind=ModelKind.CANDIDATE_FAMILY,
        lifecycle=ModelLifecycle.PLANNED,
        authority=ResearchAuthority.NONE,
        semantic_scope=SemanticScope.POLYPHONIC_V2,
        source_module="st_omr_training.model_registry",
        source_version="tr-poly-11-planned-relation-graph-v1",
        task_ids=("polyphonic_relation_graph",),
        tokenizer_version=V2_TOKENIZER_VERSION,
        representation_version=V2_REPRESENTATION_VERSION,
        checkpoint_required_for_evidence=True,
        polyphonic_v2_capable=True,
    ),
)


def validate_registry(records: object = SEED_MODEL_REGISTRY) -> tuple[ModelRegistryRecord, ...]:
    if not isinstance(records, tuple) or not records:
        raise ModelRegistryError("registry must be a non-empty tuple")
    seen_ids: set[str] = set()
    normalized: list[ModelRegistryRecord] = []
    for record in records:
        if not isinstance(record, ModelRegistryRecord):
            raise ModelRegistryError("registry entries must be ModelRegistryRecord")
        if record.record_id in seen_ids:
            raise ModelRegistryError("duplicate registry record_id")
        seen_ids.add(record.record_id)
        normalized.append(record)
    return tuple(normalized)


def registry_fingerprint(records: object = SEED_MODEL_REGISTRY) -> str:
    validated = validate_registry(records)
    payload = {
        "registry_version": MODEL_REGISTRY_VERSION,
        "records": [record.canonical_payload() for record in sorted(validated, key=lambda item: item.record_id)],
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def registry_by_id(records: object = SEED_MODEL_REGISTRY) -> dict[str, ModelRegistryRecord]:
    return {record.record_id: record for record in validate_registry(records)}


def validate_artifact_binding(
    binding: ModelArtifactBinding,
    records: object = SEED_MODEL_REGISTRY,
) -> ModelRegistryRecord:
    if not isinstance(binding, ModelArtifactBinding):
        raise ModelRegistryError("binding must be ModelArtifactBinding")
    record = registry_by_id(records).get(binding.record_id)
    if record is None:
        raise ModelRegistryError("artifact binding references unknown registry record")
    if record.lifecycle in (ModelLifecycle.ARCHITECTURE_ONLY, ModelLifecycle.PLANNED, ModelLifecycle.DETERMINISTIC):
        raise ModelRegistryError("record lifecycle does not admit checkpoint evidence")
    if not record.checkpoint_required_for_evidence:
        raise ModelRegistryError("record does not use checkpoint evidence")
    if binding.tokenizer_version != record.tokenizer_version:
        raise ModelRegistryError("artifact tokenizer version does not match registry record")
    if binding.representation_version != record.representation_version:
        raise ModelRegistryError("artifact representation version does not match registry record")
    if record.tokenizer_version is None and binding.tokenizer_fingerprint_sha256 is not None:
        raise ModelRegistryError("tokenizer-free record cannot carry tokenizer fingerprint")
    if record.tokenizer_version is not None and binding.tokenizer_fingerprint_sha256 is None:
        raise ModelRegistryError("tokenized record requires tokenizer fingerprint")
    return record


def model_card_payload(record: ModelRegistryRecord) -> dict[str, object]:
    if not isinstance(record, ModelRegistryRecord):
        raise ModelRegistryError("record must be ModelRegistryRecord")
    return {
        "schema_version": MODEL_CARD_SCHEMA_VERSION,
        "registry_version": MODEL_REGISTRY_VERSION,
        "registry_record_fingerprint": record.fingerprint(),
        "record": record.canonical_payload(),
        "claim_boundary": {
            "production_authority": False,
            "checkpoint_required_for_learned_evidence": record.checkpoint_required_for_evidence,
            "registry_presence_is_performance_evidence": False,
            "registry_presence_is_promotion_authority": False,
        },
    }


def model_card_sha256(record: ModelRegistryRecord) -> str:
    return sha256(_canonical_json_bytes(model_card_payload(record))).hexdigest()
