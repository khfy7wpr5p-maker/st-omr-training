from __future__ import annotations

import unittest

from st_omr_training.polyphonic_representation import (
    ClefAssignment,
    EventKind,
    ExactRational,
    KeySignature,
    NoteAtom,
    NoteType,
    PitchSpelling,
    PolyEvent,
    PolyMeasure,
    PolyPart,
    PolyScore,
    TimeSignature,
)
from st_omr_training.scoregraph_v2 import (
    BlockReason,
    CapabilityOutcome,
    GraphAnchor,
    GraphAnchorKind,
    RelationKind,
    ScoreGraphCapability,
    ScoreGraphRelation,
    ScoreGraphV2,
    SourceNavigationEvent,
    SourceNavigationKind,
)


def two_note_score() -> PolyScore:
    first = PolyEvent(
        event_id="event-1",
        kind=EventKind.NOTE,
        onset=ExactRational(0, 1),
        duration=ExactRational(1, 4),
        voice=1,
        staff=1,
        note_type=NoteType.QUARTER,
        noteheads=(NoteAtom("atom-1", PitchSpelling("C", 0, 4)),),
    )
    second = PolyEvent(
        event_id="event-2",
        kind=EventKind.NOTE,
        onset=ExactRational(1, 4),
        duration=ExactRational(1, 4),
        voice=2,
        staff=1,
        note_type=NoteType.QUARTER,
        noteheads=(NoteAtom("atom-2", PitchSpelling("E", 0, 4)),),
    )
    measure = PolyMeasure(
        measure_index=1,
        source_number="1",
        time_signature=TimeSignature((4,), 4),
        key_signature=KeySignature(0),
        clefs=(ClefAssignment(1, "G", 2),),
        events=(first, second),
    )
    return PolyScore((PolyPart("P1", 1, (measure,)),))


class ScoreGraphV2ContractTests(unittest.TestCase):
    def test_wrap_preserves_frozen_poly_score_fingerprint_and_derives_nodes(self) -> None:
        core = two_note_score()
        graph = ScoreGraphV2(core=core)

        self.assertEqual(graph.core_sha256, core.canonical_sha256())
        self.assertEqual(
            tuple((node.part_id, node.staff) for node in graph.staff_nodes),
            (("P1", 1),),
        )
        self.assertEqual(
            tuple((node.part_id, node.voice) for node in graph.voice_nodes),
            (("P1", 1), ("P1", 2)),
        )

    def test_backup_and_forward_are_explicit_source_navigation_evidence(self) -> None:
        graph = ScoreGraphV2(
            core=two_note_score(),
            source_navigation=(
                SourceNavigationEvent(
                    kind=SourceNavigationKind.BACKUP,
                    part_id="P1",
                    measure_index=1,
                    sequence_index=3,
                    duration=ExactRational(1, 4),
                ),
                SourceNavigationEvent(
                    kind=SourceNavigationKind.FORWARD,
                    part_id="P1",
                    measure_index=1,
                    sequence_index=4,
                    duration=ExactRational(1, 8),
                ),
            ),
        )

        self.assertEqual(
            tuple(item.kind for item in graph.source_navigation),
            (SourceNavigationKind.BACKUP, SourceNavigationKind.FORWARD),
        )

    def test_slur_relation_is_first_class_without_mutating_poly_score(self) -> None:
        core = two_note_score()
        relation = ScoreGraphRelation(
            relation_id="slur-1",
            kind=RelationKind.SLUR,
            source=GraphAnchor(GraphAnchorKind.EVENT, "event-1"),
            target=GraphAnchor(GraphAnchorKind.EVENT, "event-2"),
            number=1,
        )
        graph = ScoreGraphV2(core=core, relations=(relation,))

        self.assertEqual(graph.relations, (relation,))
        self.assertEqual(graph.core_sha256, core.canonical_sha256())

    def test_review_required_is_valid_for_unsupported_semantics(self) -> None:
        finding = ScoreGraphCapability(
            code="musicxml.unsupported_slur_shape",
            outcome=CapabilityOutcome.REVIEW_REQUIRED,
            path="$.parts[0].measures[0]",
        )
        graph = ScoreGraphV2(core=two_note_score(), capabilities=(finding,))
        self.assertEqual(graph.capabilities, (finding,))

    def test_blocked_requires_a_safety_unparseable_or_boundedness_reason(self) -> None:
        with self.assertRaises(ValueError):
            ScoreGraphCapability(
                code="musicxml.unsupported_slur_shape",
                outcome=CapabilityOutcome.BLOCKED,
                path="$",
            )

        finding = ScoreGraphCapability(
            code="input.too_large",
            outcome=CapabilityOutcome.BLOCKED,
            path="$",
            block_reason=BlockReason.BOUNDEDNESS,
        )
        self.assertEqual(finding.block_reason, BlockReason.BOUNDEDNESS)

    def test_relation_and_navigation_references_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            ScoreGraphV2(
                core=two_note_score(),
                source_navigation=(
                    SourceNavigationEvent(
                        kind=SourceNavigationKind.BACKUP,
                        part_id="P1",
                        measure_index=2,
                        sequence_index=1,
                        duration=ExactRational(1, 4),
                    ),
                ),
            )

        with self.assertRaises(ValueError):
            ScoreGraphV2(
                core=two_note_score(),
                relations=(
                    ScoreGraphRelation(
                        relation_id="slur-bad",
                        kind=RelationKind.SLUR,
                        source=GraphAnchor(GraphAnchorKind.EVENT, "event-missing"),
                        target=GraphAnchor(GraphAnchorKind.EVENT, "event-2"),
                        number=1,
                    ),
                ),
            )

    def test_envelope_fingerprint_is_deterministic_and_separate_from_core(self) -> None:
        finding = ScoreGraphCapability(
            code="musicxml.direction_deferred",
            outcome=CapabilityOutcome.REVIEW_REQUIRED,
            path="$.parts[0]",
        )
        left = ScoreGraphV2(core=two_note_score(), capabilities=(finding,))
        right = ScoreGraphV2(core=two_note_score(), capabilities=(finding,))

        self.assertEqual(left.canonical_sha256(), right.canonical_sha256())
        self.assertNotEqual(left.canonical_sha256(), left.core_sha256)


if __name__ == "__main__":
    unittest.main()
