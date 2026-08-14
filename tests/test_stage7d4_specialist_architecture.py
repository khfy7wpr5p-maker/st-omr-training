from __future__ import annotations

from dataclasses import replace
import unittest

import st_omr_training.stage7d4_specialist_architecture as contract


class Stage7D4SpecialistArchitectureTests(unittest.TestCase):
    def test_frozen_v1_specialist_surface_is_exact(self) -> None:
        self.assertEqual(
            tuple(task.task_id for task in contract.V1_SPECIALIST_TASKS),
            (
                "staff_geometry",
                "structure",
                "notehead",
                "rest",
                "accidental",
                "rhythm",
                "staff_position",
                "chord_grouping",
                "context_validation",
            ),
        )
        self.assertEqual(
            tuple(item.set_name for item in contract.V1_SPECIALIST_DATASETS),
            (
                "StaffSet",
                "StructureSet",
                "NoteHeadSet",
                "RhythmSet",
                "RestSet",
                "AccidentalSet",
                "PitchSet",
                "ChordSet",
                "ContextSet",
            ),
        )
        contract.validate_stage7d4_architecture()

    def test_v1_music_scope_and_deferred_surface_do_not_expand(self) -> None:
        self.assertEqual(contract.V1_MUSIC_POLICY["parts"], 1)
        self.assertEqual(contract.V1_MUSIC_POLICY["staves"], 1)
        self.assertEqual(contract.V1_MUSIC_POLICY["voices"], 1)
        self.assertEqual(contract.V1_MUSIC_POLICY["clef"], "G2")
        self.assertEqual(contract.V1_MUSIC_POLICY["key_fifths"], 0)
        self.assertEqual(contract.V1_MUSIC_POLICY["meters"], ("2/4", "3/4", "4/4"))
        self.assertEqual(contract.V1_MUSIC_POLICY["chord_sizes"], (2, 3, 4))
        for deferred in (
            "multiple_voices",
            "grand_staff",
            "tuplets",
            "ties",
            "slurs",
            "dotted_values",
            "nonzero_key_signatures",
        ):
            self.assertIn(deferred, contract.DEFERRED_SPECIALIST_SURFACE)

    def test_ground_truth_has_no_learned_or_ai_source(self) -> None:
        self.assertNotIn("ai", contract.ALLOWED_GT_SOURCES)
        self.assertNotIn("model_prediction", contract.ALLOWED_GT_SOURCES)
        self.assertFalse(contract.TRAINING_DATA_POLICY["ai_generated_ground_truth"])
        for dataset in contract.V1_SPECIALIST_DATASETS:
            for source in dataset.synthetic_ground_truth + dataset.real_ground_truth:
                self.assertIn(source, contract.ALLOWED_GT_SOURCES)

    def test_geometry_sets_require_renderer_lineage_and_real_human_annotation(self) -> None:
        geometry_sets = [item for item in contract.V1_SPECIALIST_DATASETS if item.geometry_required]
        self.assertGreater(len(geometry_sets), 0)
        for dataset in geometry_sets:
            self.assertIn(contract.GT_RENDERER_GEOMETRY, dataset.synthetic_ground_truth)
            self.assertIn(contract.GT_DETERMINISTIC_TRANSFORM, dataset.synthetic_ground_truth)
            self.assertIn(contract.GT_HUMAN_VERIFIED_ANNOTATION, dataset.real_ground_truth)

        with self.assertRaises(ValueError):
            contract.SpecialistDatasetContract(
                set_name="BadSet",
                task_id="bad",
                labels=("bbox",),
                synthetic_ground_truth=(contract.GT_CANONICAL_MUSIC,),
                real_ground_truth=(contract.GT_ADMITTED_REAL_MUSICXML,),
                geometry_required=True,
            )

    def test_pitch_specialist_predicts_staff_position_not_absolute_pitch(self) -> None:
        pitch_task = next(task for task in contract.V1_SPECIALIST_TASKS if task.task_id == "staff_position")
        self.assertIn("staff_position", pitch_task.outputs)
        for forbidden in ("pitch", "pitch_name", "midi", "octave", "step"):
            self.assertNotIn(forbidden, pitch_task.outputs)
        self.assertEqual(
            contract.FUSION_POLICY["pitch_rule"],
            "G2_clef_plus_staff_position_plus_measure_accidental_state",
        )
        self.assertFalse(contract.FUSION_POLICY["direct_absolute_pitch_prediction_is_authoritative"])

    def test_context_fusion_is_deterministic_and_fail_closed(self) -> None:
        context = next(task for task in contract.V1_SPECIALIST_TASKS if task.task_id == "context_validation")
        self.assertFalse(context.trainable)
        self.assertFalse(contract.FUSION_POLICY["learned_fusion"])
        self.assertTrue(contract.FUSION_POLICY["measure_duration_must_fill_exactly"])
        self.assertTrue(contract.FUSION_POLICY["chord_members_share_onset"])
        self.assertTrue(contract.FUSION_POLICY["chord_members_share_duration"])
        self.assertEqual(
            contract.FUSION_POLICY["unsupported_or_ambiguous_result"],
            "veto_or_low_confidence_candidate",
        )

    def test_dependency_graph_is_acyclic_and_context_depends_on_all_visual_specialists(self) -> None:
        contract.validate_stage7d4_architecture()
        context = next(task for task in contract.V1_SPECIALIST_TASKS if task.task_id == "context_validation")
        self.assertEqual(
            set(context.depends_on),
            {
                "staff_geometry",
                "structure",
                "notehead",
                "rest",
                "accidental",
                "rhythm",
                "staff_position",
                "chord_grouping",
            },
        )

    def test_test_split_and_teacher_data_remain_sealed(self) -> None:
        self.assertFalse(contract.SPLIT_POLICY["test_model_development_access"])
        self.assertFalse(contract.SPLIT_POLICY["test_specialist_dataset_derivation_during_development"])
        self.assertEqual(contract.SPLIT_POLICY["sealed_test_stage"], 9)
        self.assertTrue(contract.SPLIT_POLICY["family_exclusive"])
        self.assertFalse(contract.TRAINING_DATA_POLICY["scoremosaic_upload_is_automatic_training_data"])
        self.assertFalse(contract.TRAINING_DATA_POLICY["teacher_correction_is_automatic_training_data"])
        self.assertTrue(contract.TRAINING_DATA_POLICY["teacher_correction_requires_explicit_admission"])
        self.assertFalse(contract.TRAINING_DATA_POLICY["online_learning"])
        self.assertFalse(contract.TRAINING_DATA_POLICY["automatic_learning"])

    def test_contract_payload_and_fingerprint_are_canonical_and_deterministic(self) -> None:
        first_payload = contract.stage7d4_architecture_payload()
        second_payload = contract.stage7d4_architecture_payload()
        self.assertEqual(first_payload, second_payload)
        first = contract.stage7d4_architecture_fingerprint()
        second = contract.stage7d4_architecture_fingerprint()
        self.assertEqual(first, second)
        self.assertTrue(contract.is_sha256_hex(first))
        self.assertEqual(first_payload["schema"], contract.STAGE7D4_CONTRACT_SCHEMA)
        self.assertEqual(first_payload["version"], contract.STAGE7D4_ARCHITECTURE_VERSION)

    def test_task_contract_rejects_duplicate_edges_and_empty_io(self) -> None:
        base = contract.V1_SPECIALIST_TASKS[1]
        with self.assertRaises(ValueError):
            replace(base, depends_on=("staff_geometry", "staff_geometry"))
        with self.assertRaises(ValueError):
            replace(base, inputs=())
        with self.assertRaises(ValueError):
            replace(base, outputs=())


if __name__ == "__main__":
    unittest.main()
