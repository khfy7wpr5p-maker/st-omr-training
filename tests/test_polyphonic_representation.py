from __future__ import annotations

import json
import unittest

from st_omr_training.polyphonic_representation import (
    Barline,
    BarlineLocation,
    BarlineStyle,
    BeamMark,
    BeamState,
    ClefAssignment,
    DisplayAccidentalV2,
    EventKind,
    ExactRational,
    GraceSpec,
    KeySignature,
    NoteAtom,
    NoteType,
    POLYPHONIC_REPRESENTATION_VERSION,
    POLYPHONIC_TIME_UNIT,
    PitchSpelling,
    PolyEvent,
    PolyMeasure,
    PolyPart,
    PolyScore,
    PolyphonicRepresentationError,
    REQUIRED_POLYPHONIC_SEMANTICS,
    StemDirection,
    TieState,
    TimeSignature,
    TupletBoundary,
    TupletMark,
)


def _pitch(step: str, octave: int, *, alter: int = 0, accidental=DisplayAccidentalV2.NONE):
    return PitchSpelling(step, alter, octave, accidental)


def _note(
    event_id: str,
    atom_id: str,
    *,
    onset=(0, 1),
    duration=(1, 4),
    voice=1,
    staff=1,
    staff_override=None,
    pitch=None,
):
    return PolyEvent(
        event_id=event_id,
        kind=EventKind.NOTE,
        onset=ExactRational(*onset),
        duration=ExactRational(*duration),
        voice=voice,
        staff=staff,
        note_type=NoteType.QUARTER,
        noteheads=(
            NoteAtom(
                atom_id=atom_id,
                pitch=pitch or _pitch("C", 4),
                staff_override=staff_override,
            ),
        ),
    )


def _measure(events, *, staff_count=2):
    return PolyMeasure(
        measure_index=1,
        source_number="1",
        time_signature=TimeSignature((4,), 4),
        key_signature=KeySignature(0, "major"),
        clefs=tuple(
            ClefAssignment(staff=index, sign="G" if index == 1 else "F", line=2 if index == 1 else 4)
            for index in range(1, staff_count + 1)
        ),
        events=tuple(events),
        barlines=(Barline(BarlineLocation.RIGHT, BarlineStyle.REGULAR),),
    )


def _score(events, *, staff_count=2):
    return PolyScore((PolyPart("P1", staff_count, (_measure(events, staff_count=staff_count),)),))


class PolyphonicRepresentationTests(unittest.TestCase):
    def test_version_and_time_unit_are_frozen(self) -> None:
        self.assertEqual(POLYPHONIC_REPRESENTATION_VERSION, "st-omr-polyphonic-representation-v2")
        self.assertEqual(POLYPHONIC_TIME_UNIT, "whole-note-fraction")
        self.assertIn("voice", REQUIRED_POLYPHONIC_SEMANTICS)
        self.assertIn("onset", REQUIRED_POLYPHONIC_SEMANTICS)
        self.assertIn("cross_staff", REQUIRED_POLYPHONIC_SEMANTICS)

    def test_exact_rational_normalizes_without_float_loss(self) -> None:
        value = ExactRational(2, 8)
        self.assertEqual((value.numerator, value.denominator), (1, 4))
        self.assertEqual(value.fraction.numerator, 1)
        self.assertEqual(value.fraction.denominator, 4)
        with self.assertRaises(PolyphonicRepresentationError):
            ExactRational(-1, 4)

    def test_two_independent_voices_can_share_onset(self) -> None:
        first = _note("e1", "a1", voice=1, pitch=_pitch("C", 4))
        second = _note("e2", "a2", voice=2, pitch=_pitch("G", 4))
        score = _score((first, second), staff_count=1)
        events = score.parts[0].measures[0].events
        self.assertEqual(events[0].onset, events[1].onset)
        self.assertEqual({event.voice for event in events}, {1, 2})
        self.assertTrue(all(event.kind is EventKind.NOTE for event in events))

    def test_same_voice_can_move_between_staves(self) -> None:
        first = _note("e1", "a1", onset=(0, 1), voice=1, staff=1)
        second = _note("e2", "a2", onset=(1, 4), voice=1, staff=2)
        score = _score((first, second))
        self.assertEqual(tuple(event.staff for event in score.parts[0].measures[0].events), (1, 2))

    def test_cross_staff_chord_keeps_notehead_staff_override(self) -> None:
        chord = PolyEvent(
            event_id="chord-1",
            kind=EventKind.CHORD,
            onset=ExactRational(0, 1),
            duration=ExactRational(1, 4),
            voice=1,
            staff=1,
            note_type=NoteType.QUARTER,
            noteheads=(
                NoteAtom("atom-low", _pitch("B", 3), staff_override=2),
                NoteAtom("atom-high", _pitch("G", 4)),
            ),
            stem=StemDirection.UP,
        )
        score = _score((chord,))
        event = score.parts[0].measures[0].events[0]
        self.assertTrue(event.is_cross_staff)
        self.assertEqual(event.noteheads[0].staff_override, 2)

    def test_chord_grouping_is_distinct_from_simultaneous_voices(self) -> None:
        chord = PolyEvent(
            event_id="chord",
            kind=EventKind.CHORD,
            onset=ExactRational(0, 1),
            duration=ExactRational(1, 2),
            voice=1,
            staff=1,
            note_type=NoteType.HALF,
            noteheads=(
                NoteAtom("c1", _pitch("C", 4)),
                NoteAtom("c2", _pitch("E", 4)),
                NoteAtom("c3", _pitch("G", 4)),
            ),
        )
        independent = _note("voice-2", "v2", duration=(1, 2), voice=2, pitch=_pitch("A", 4))
        score = _score((chord, independent), staff_count=1)
        self.assertEqual(score.parts[0].measures[0].events[0].kind, EventKind.CHORD)
        self.assertEqual(score.parts[0].measures[0].events[1].kind, EventKind.NOTE)

    def test_notehead_keeps_individual_accidental_and_tie_intent(self) -> None:
        chord = PolyEvent(
            event_id="e1",
            kind=EventKind.CHORD,
            onset=ExactRational(0, 1),
            duration=ExactRational(1, 4),
            voice=1,
            staff=1,
            note_type=NoteType.QUARTER,
            noteheads=(
                NoteAtom(
                    "a1",
                    _pitch("F", 4, alter=1, accidental=DisplayAccidentalV2.SHARP),
                    ties=(TieState.START,),
                ),
                NoteAtom(
                    "a2",
                    _pitch("A", 4, alter=-1, accidental=DisplayAccidentalV2.FLAT),
                    ties=(TieState.STOP,),
                ),
            ),
        )
        score = _score((chord,), staff_count=1)
        notes = score.parts[0].measures[0].events[0].noteheads
        self.assertEqual(notes[0].pitch.display_accidental, DisplayAccidentalV2.SHARP)
        self.assertEqual(notes[0].ties, (TieState.START,))
        self.assertEqual(notes[1].ties, (TieState.STOP,))

    def test_beam_tuplet_grace_metadata_is_retained(self) -> None:
        grace = PolyEvent(
            event_id="grace-1",
            kind=EventKind.NOTE,
            onset=ExactRational(0, 1),
            duration=ExactRational(0, 1),
            voice=1,
            staff=1,
            note_type=NoteType.EIGHTH,
            noteheads=(NoteAtom("ga", _pitch("D", 5)),),
            beams=(BeamMark(1, BeamState.BEGIN),),
            tuplets=(TupletMark(1, 3, 2, TupletBoundary.START),),
            grace=GraceSpec(slash=True),
        )
        score = _score((grace,), staff_count=1)
        event = score.parts[0].measures[0].events[0]
        self.assertTrue(event.grace.slash)
        self.assertEqual(event.beams[0].state, BeamState.BEGIN)
        self.assertEqual(event.tuplets[0].actual_notes, 3)

    def test_rest_is_explicit_and_contains_no_noteheads(self) -> None:
        rest = PolyEvent(
            event_id="r1",
            kind=EventKind.REST,
            onset=ExactRational(0, 1),
            duration=ExactRational(1, 4),
            voice=1,
            staff=1,
            note_type=NoteType.QUARTER,
        )
        score = _score((rest,), staff_count=1)
        self.assertEqual(score.parts[0].measures[0].events[0].kind, EventKind.REST)

    def test_invalid_staff_and_voice_fail_closed(self) -> None:
        with self.assertRaises(PolyphonicRepresentationError):
            _note("e1", "a1", voice=0)
        event = _note("e2", "a2", staff=2)
        with self.assertRaises(PolyphonicRepresentationError):
            _score((event,), staff_count=1)

    def test_global_duplicate_event_and_atom_ids_fail_closed(self) -> None:
        first_measure = _measure((_note("same", "same-atom"),), staff_count=1)
        second_measure = PolyMeasure(
            measure_index=2,
            source_number="2",
            time_signature=TimeSignature((4,), 4),
            key_signature=KeySignature(0),
            clefs=(ClefAssignment(1, "G", 2),),
            events=(_note("same", "another-atom"),),
        )
        with self.assertRaises(PolyphonicRepresentationError):
            PolyScore((PolyPart("P1", 1, (first_measure, second_measure)),))

        first = _note("e1", "same-atom", voice=1)
        second = _note("e2", "same-atom", voice=2)
        with self.assertRaises(PolyphonicRepresentationError):
            _score((first, second), staff_count=1)

    def test_noncanonical_event_order_fails_closed(self) -> None:
        later = _note("later", "a1", onset=(1, 4))
        earlier = _note("earlier", "a2", onset=(0, 1))
        with self.assertRaises(PolyphonicRepresentationError):
            _measure((later, earlier), staff_count=1)

    def test_event_cannot_overrun_measure_capacity(self) -> None:
        event = _note("e1", "a1", onset=(3, 4), duration=(1, 2))
        with self.assertRaises(PolyphonicRepresentationError):
            _score((event,), staff_count=1)

    def test_additive_meter_capacity_is_exact(self) -> None:
        signature = TimeSignature((3, 2), 8)
        self.assertEqual(signature.capacity.numerator, 5)
        self.assertEqual(signature.capacity.denominator, 8)

    def test_canonical_json_and_sha_are_deterministic(self) -> None:
        score = _score((_note("e1", "a1"),), staff_count=1)
        first = score.canonical_json()
        second = score.canonical_json()
        self.assertEqual(first, second)
        self.assertEqual(score.canonical_sha256(), score.canonical_sha256())
        decoded = json.loads(first)
        self.assertEqual(decoded["representation_version"], POLYPHONIC_REPRESENTATION_VERSION)
        self.assertEqual(decoded["parts"][0]["measures"][0]["events"][0]["kind"], "note")

    def test_representation_version_mismatch_is_rejected(self) -> None:
        part = PolyPart("P1", 1, (_measure((_note("e1", "a1"),), staff_count=1),))
        with self.assertRaises(PolyphonicRepresentationError):
            PolyScore((part,), representation_version="future-v3")


if __name__ == "__main__":
    unittest.main()
