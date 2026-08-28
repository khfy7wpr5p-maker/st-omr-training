from __future__ import annotations

from dataclasses import replace
import unittest

from st_omr_training.model_registry import (
    MODEL_ARTIFACT_BINDING_VERSION,
    MODEL_CARD_SCHEMA_VERSION,
    MODEL_REGISTRY_VERSION,
    POLY_EVALUATION_CONTRACT_VERSION,
    SEED_MODEL_REGISTRY,
    V1_TOKENIZER_VERSION,
    V2_REPRESENTATION_VERSION,
    V2_TOKENIZER_VERSION,
    ModelArtifactBinding,
    ModelEvidenceBinding,
    ModelKind,
    ModelLifecycle,
    ModelRegistryError,
    ModelRegistryRecord,
    ResearchAuthority,
    SemanticScope,
    model_card_payload,
    model_card_sha256,
    registry_by_id,
    registry_fingerprint,
    validate_artifact_binding,
    validate_registry,
)
from st_omr_training.poly_2d_transformer import POLY_2D_TRANSFORMER_VERSION
from st_omr_training.poly_evaluation_contract import (
    POLY_EVALUATION_CONTRACT_VERSION as ACTUAL_EVALUATION_CONTRACT_VERSION,
)
from st_omr_training.polyphonic_representation import POLYPHONIC_REPRESENTATION_VERSION
from st_omr_training.polyphonic_serialization import POLYPHONIC_TOKENIZER_VERSION
from st_omr_training.stage7d11_barline_meter_training import BARLINE_MODEL_VERSION, METER_MODEL_VERSION
from st_omr_training.stage7d7_specialist_training import STAFF_MODEL_VERSION, STRUCTURE_MODEL_VERSION
from st_omr_training.training_model import BASELINE_MODEL_VERSION
from st_omr_training.training_tokens import TOKENIZER_VERSION

_SHA = "1" * 64
_SHA2 = "2" * 64
_SHA3 = "3" * 64
_SHA4 = "4" * 64
_SHA5 = "5" * 64
_GIT = "a" * 40


def _binding(record_id: str, *, tokenizer_version=None, representation_version=None, tokenizer_fp=None, record_fp=None):
    record = registry_by_id().get(record_id)
    if record_fp is None:
        record_fp = record.fingerprint() if record is not None else _SHA5
    return ModelArtifactBinding(
        record_id=record_id,
        registry_record_fingerprint_sha256=record_fp,
        repository_sha=_GIT,
        checkpoint_sha256=_SHA,
        model_fingerprint_sha256=_SHA2,
        training_profile_sha256=_SHA3,
        dataset_manifest_sha256=_SHA4,
        runtime_fingerprint_sha256=_SHA5,
        tokenizer_fingerprint_sha256=tokenizer_fp,
        tokenizer_version=tokenizer_version,
        representation_version=representation_version,
    )


class ModelRegistryTests(unittest.TestCase):
    def test_versions_match_frozen_source_contracts(self) -> None:
        self.assertEqual(MODEL_REGISTRY_VERSION, "st-omr-model-registry-v1")
        self.assertEqual(MODEL_CARD_SCHEMA_VERSION, "st-omr-model-card-v1")
        self.assertEqual(MODEL_ARTIFACT_BINDING_VERSION, "st-omr-model-artifact-binding-v1")
        self.assertEqual(V1_TOKENIZER_VERSION, TOKENIZER_VERSION)
        self.assertEqual(V2_REPRESENTATION_VERSION, POLYPHONIC_REPRESENTATION_VERSION)
        self.assertEqual(V2_TOKENIZER_VERSION, POLYPHONIC_TOKENIZER_VERSION)
        self.assertEqual(POLY_EVALUATION_CONTRACT_VERSION, ACTUAL_EVALUATION_CONTRACT_VERSION)

    def test_seed_registry_consolidates_existing_model_versions(self) -> None:
        records = registry_by_id()
        self.assertEqual(records["baseline.cnn-gru.v1"].source_version, BASELINE_MODEL_VERSION)
        self.assertEqual(records["specialist.staff.d7.v1"].source_version, STAFF_MODEL_VERSION)
        self.assertEqual(records["specialist.structure.d7.v1"].source_version, STRUCTURE_MODEL_VERSION)
        self.assertEqual(records["refiner.barline.d11.v1"].source_version, BARLINE_MODEL_VERSION)
        self.assertEqual(records["refiner.meter.d11.v1"].source_version, METER_MODEL_VERSION)
        self.assertEqual(records["candidate.poly-2d-transformer.v1"].source_version, POLY_2D_TRANSFORMER_VERSION)

    def test_registry_is_unique_deterministic_and_order_independent(self) -> None:
        records = validate_registry()
        self.assertEqual(len(records), len({record.record_id for record in records}))
        self.assertEqual(registry_fingerprint(records), registry_fingerprint(tuple(reversed(records))))
        self.assertEqual(len(registry_fingerprint(records)), 64)

    def test_registry_never_grants_production_authority(self) -> None:
        for record in SEED_MODEL_REGISTRY:
            self.assertFalse(record.production_authority)
        with self.assertRaisesRegex(ModelRegistryError, "never grants production"):
            replace(SEED_MODEL_REGISTRY[0], production_authority=True)

    def test_declared_and_planned_records_have_no_inference_authority(self) -> None:
        for record in SEED_MODEL_REGISTRY:
            if record.lifecycle in (ModelLifecycle.ARCHITECTURE_ONLY, ModelLifecycle.PLANNED):
                self.assertIs(record.authority, ResearchAuthority.NONE)
                self.assertTrue(record.checkpoint_required_for_evidence)
        with self.assertRaisesRegex(ModelRegistryError, "cannot carry research inference authority"):
            ModelRegistryRecord(
                record_id="candidate.bad.v1",
                model_kind=ModelKind.CANDIDATE_FAMILY,
                lifecycle=ModelLifecycle.PLANNED,
                authority=ResearchAuthority.SHADOW_ONLY,
                semantic_scope=SemanticScope.POLYPHONIC_V2,
                source_module="st_omr_training.model_registry",
                source_version="candidate-bad-v1",
                task_ids=("polyphonic_sequence_omr",),
                tokenizer_version=V2_TOKENIZER_VERSION,
                representation_version=V2_REPRESENTATION_VERSION,
                polyphonic_v2_capable=True,
            )

    def test_polyphonic_2d_prototype_is_registered_without_checkpoint_authority(self) -> None:
        candidate = registry_by_id()["candidate.poly-2d-transformer.v1"]
        self.assertTrue(candidate.polyphonic_v2_capable)
        self.assertIs(candidate.semantic_scope, SemanticScope.POLYPHONIC_V2)
        self.assertEqual(candidate.representation_version, V2_REPRESENTATION_VERSION)
        self.assertEqual(candidate.tokenizer_version, V2_TOKENIZER_VERSION)
        self.assertEqual(candidate.source_module, "st_omr_training.poly_2d_transformer")
        self.assertEqual(candidate.source_version, POLY_2D_TRANSFORMER_VERSION)
        self.assertIs(candidate.lifecycle, ModelLifecycle.ARCHITECTURE_ONLY)
        self.assertIs(candidate.authority, ResearchAuthority.NONE)
        self.assertFalse(candidate.production_authority)

    def test_baseline_artifact_requires_exact_tokenizer_binding(self) -> None:
        binding = _binding("baseline.cnn-gru.v1", tokenizer_version=V1_TOKENIZER_VERSION, tokenizer_fp=_SHA)
        self.assertEqual(validate_artifact_binding(binding).record_id, "baseline.cnn-gru.v1")
        with self.assertRaisesRegex(ModelRegistryError, "tokenizer version"):
            validate_artifact_binding(replace(binding, tokenizer_version=V2_TOKENIZER_VERSION))
        with self.assertRaisesRegex(ModelRegistryError, "requires tokenizer fingerprint"):
            validate_artifact_binding(replace(binding, tokenizer_fingerprint_sha256=None))

    def test_polyphonic_2d_prototype_cannot_register_checkpoint_evidence_yet(self) -> None:
        binding = _binding(
            "candidate.poly-2d-transformer.v1",
            tokenizer_version=V2_TOKENIZER_VERSION,
            representation_version=V2_REPRESENTATION_VERSION,
            tokenizer_fp=_SHA,
        )
        with self.assertRaisesRegex(ModelRegistryError, "does not admit checkpoint evidence"):
            validate_artifact_binding(binding)

    def test_artifact_is_bound_to_exact_registry_record_fingerprint(self) -> None:
        binding = _binding("specialist.staff.d7.v1")
        original = registry_by_id()["specialist.staff.d7.v1"]
        drifted = replace(original, source_version=original.source_version + "-drift")
        drifted_registry = tuple(drifted if r.record_id == original.record_id else r for r in SEED_MODEL_REGISTRY)
        with self.assertRaisesRegex(ModelRegistryError, "registry-record fingerprint mismatch"):
            validate_artifact_binding(binding, drifted_registry)

    def test_tokenizer_free_visual_artifact_rejects_spurious_tokenizer_identity(self) -> None:
        binding = _binding("specialist.staff.d7.v1")
        self.assertEqual(validate_artifact_binding(binding).record_id, "specialist.staff.d7.v1")
        with self.assertRaisesRegex(ModelRegistryError, "tokenizer-free"):
            validate_artifact_binding(replace(binding, tokenizer_fingerprint_sha256=_SHA))

    def test_unimplemented_and_deterministic_records_cannot_masquerade_as_checkpoint_evidence(self) -> None:
        for record_id in (
            "specialist.notehead.declared.v1",
            "candidate.poly-2d-transformer.v1",
            "candidate.relation-graph.v1",
            "fusion.context-validator.v1",
        ):
            with self.assertRaisesRegex(ModelRegistryError, "does not admit checkpoint evidence"):
                validate_artifact_binding(_binding(record_id))

    def test_unknown_record_and_malformed_hashes_fail_closed(self) -> None:
        with self.assertRaisesRegex(ModelRegistryError, "unknown registry record"):
            validate_artifact_binding(_binding("unknown.model.v1"))
        with self.assertRaisesRegex(ModelRegistryError, "checkpoint_sha256"):
            replace(_binding("specialist.staff.d7.v1"), checkpoint_sha256="ABC")
        with self.assertRaisesRegex(ModelRegistryError, "repository_sha"):
            replace(_binding("specialist.staff.d7.v1"), repository_sha="b" * 39)
        with self.assertRaisesRegex(ModelRegistryError, "registry_record_fingerprint_sha256"):
            replace(_binding("specialist.staff.d7.v1"), registry_record_fingerprint_sha256="BAD")

    def test_model_cards_are_deterministic_and_do_not_overclaim(self) -> None:
        record = registry_by_id()["candidate.poly-2d-transformer.v1"]
        card = model_card_payload(record)
        self.assertEqual(card["schema_version"], MODEL_CARD_SCHEMA_VERSION)
        self.assertFalse(card["claim_boundary"]["production_authority"])
        self.assertFalse(card["claim_boundary"]["registry_presence_is_performance_evidence"])
        self.assertFalse(card["claim_boundary"]["registry_presence_is_promotion_authority"])
        self.assertEqual(model_card_sha256(record), model_card_sha256(record))
        self.assertEqual(len(model_card_sha256(record)), 64)

    def test_evidence_binding_requires_exact_poly_evaluation_contract(self) -> None:
        evidence = ModelEvidenceBinding(artifact_binding_sha256=_SHA, benchmark_identity_sha256=_SHA2, metrics_sha256=_SHA3)
        self.assertEqual(len(evidence.fingerprint()), 64)
        with self.assertRaisesRegex(ModelRegistryError, "evaluation contract version"):
            replace(evidence, evaluation_contract_version="wrong-contract")

    def test_duplicate_registry_id_fails_closed(self) -> None:
        with self.assertRaisesRegex(ModelRegistryError, "duplicate registry"):
            validate_registry((SEED_MODEL_REGISTRY[0], SEED_MODEL_REGISTRY[0]))

    def test_deterministic_fusion_is_explicitly_not_a_model_checkpoint(self) -> None:
        fusion = registry_by_id()["fusion.context-validator.v1"]
        self.assertIs(fusion.model_kind, ModelKind.DETERMINISTIC_FUSION)
        self.assertIs(fusion.lifecycle, ModelLifecycle.DETERMINISTIC)
        self.assertFalse(fusion.checkpoint_required_for_evidence)
        self.assertIs(fusion.authority, ResearchAuthority.NONE)


if __name__ == "__main__":
    unittest.main()
