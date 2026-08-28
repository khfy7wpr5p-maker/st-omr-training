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
    PolyphonicRepresentationError,
    TimeSignature,
)


def _event(event_id: str, atom_id: str, *, voice: int, staff: int, onset=(0, 1)) -> PolyEvent:
    return PolyEvent(
        event_id=event_id,
        kind=EventKind.NOTE,
        onset=ExactRational(*onset),
        duration=ExactRational(1, 4),
        voice=voice,
        staff=staff,
        note_type=NoteType.QUARTER,
        noteheads=(NoteAtom(atom_id, PitchSpelling("C", 0, 4)),),
    )


class PolyphonicRepresentationReviewRegressions(unittest.TestCase):
    def test_same_voice_equal_onset_must_be_one_chord_event(self) -> None:
        first = _event("e1", "a1", voice=1, staff=1)
        second = _event("e2", "a2", voice=1, staff=2)
        with self.assertRaises(PolyphonicRepresentationError):
            PolyMeasure(
                measure_index=1,
                source_number="1",
                time_signature=TimeSignature((4,), 4),
                key_signature=KeySignature(0),
                clefs=(ClefAssignment(1, "G", 2), ClefAssignment(2, "F", 4)),
                events=(first, second),
            )

    def test_same_voice_nonoverlapping_cross_staff_events_remain_valid(self) -> None:
        first = _event("e1", "a1", voice=1, staff=1, onset=(0, 1))
        second = _event("e2", "a2", voice=1, staff=2, onset=(1, 4))
        measure = PolyMeasure(
            measure_index=1,
            source_number="1",
            time_signature=TimeSignature((4,), 4),
            key_signature=KeySignature(0),
            clefs=(ClefAssignment(1, "G", 2), ClefAssignment(2, "F", 4)),
            events=(first, second),
        )
        part = PolyPart("P1", 2, (measure,))
        self.assertEqual(tuple(event.staff for event in part.measures[0].events), (1, 2))

    def test_every_staff_requires_explicit_clef_assignment(self) -> None:
        event = _event("e1", "a1", voice=1, staff=1)
        measure = PolyMeasure(
            measure_index=1,
            source_number="1",
            time_signature=TimeSignature((4,), 4),
            key_signature=KeySignature(0),
            clefs=(ClefAssignment(1, "G", 2),),
            events=(event,),
        )
        with self.assertRaises(PolyphonicRepresentationError):
            PolyPart("P1", 2, (measure,))


if __name__ == "__main__":
    unittest.main()
