import unittest
from dataclasses import FrozenInstanceError
from fractions import Fraction

from st_omr_training import (
    ChordEvent,
    DisplayAccidental,
    NoteEvent,
    NotationIntent,
    Pitch,
    RationalDuration,
    RestEvent,
)


Q = RationalDuration(1, 4)


class RationalDurationTests(unittest.TestCase):
    def test_reduces_to_canonical_fraction(self):
        self.assertEqual(RationalDuration(2, 8), Q)
        self.assertEqual(hash(RationalDuration(2, 8)), hash(Q))

    def test_normalizes_negative_denominator(self):
        self.assertEqual(RationalDuration(-1, -4), Q)

    def test_zero_duration_rejected(self):
        with self.assertRaises(ValueError):
            RationalDuration(0, 4)

    def test_negative_duration_rejected(self):
        with self.assertRaises(ValueError):
            RationalDuration(-1, 4)

    def test_zero_denominator_rejected(self):
        with self.assertRaises(ValueError):
            RationalDuration(1, 0)

    def test_float_and_bool_rejected(self):
        with self.assertRaises(TypeError):
            RationalDuration(1.0, 4)
        with self.assertRaises(TypeError):
            RationalDuration(True, 4)

    def test_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            Q.numerator = 2


class PitchTests(unittest.TestCase):
    def test_valid_pitch_is_canonicalized(self):
        self.assertEqual(Pitch("c", 1, 4), Pitch("C", 1, 4))

    def test_invalid_step_rejected(self):
        with self.assertRaises(ValueError):
            Pitch("H", 0, 4)

    def test_invalid_alter_rejected(self):
        with self.assertRaises(ValueError):
            Pitch("C", 2, 4)

    def test_invalid_octave_rejected(self):
        for value in (-1, 10, 4.0, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Pitch("C", 0, value)

    def test_hash_is_deterministic_for_equal_pitch(self):
        self.assertEqual(hash(Pitch("C", 0, 4)), hash(Pitch("c", 0, 4)))


class NotationIntentTests(unittest.TestCase):
    def test_default_has_no_visible_accidental(self):
        self.assertEqual(NotationIntent().display_accidental, DisplayAccidental.NONE)

    def test_pitch_and_display_intent_are_separate(self):
        pitch = Pitch("F", 1, 4)
        hidden = NoteEvent(0, Q, pitch)
        shown = NoteEvent(0, Q, pitch, NotationIntent(DisplayAccidental.SHARP))
        self.assertNotEqual(hidden, shown)

    def test_raw_string_rejected(self):
        with self.assertRaises(TypeError):
            NotationIntent("sharp")


class EventTests(unittest.TestCase):
    def test_note_onset_is_exact_fraction(self):
        event = NoteEvent(Fraction(2, 8), Q, Pitch("C", 0, 4))
        self.assertEqual(event.onset, Fraction(1, 4))

    def test_zero_onset_allowed(self):
        self.assertEqual(NoteEvent(0, Q, Pitch("C", 0, 4)).onset, Fraction(0, 1))

    def test_negative_onset_rejected(self):
        with self.assertRaises(ValueError):
            NoteEvent(Fraction(-1, 8), Q, Pitch("C", 0, 4))

    def test_float_onset_rejected(self):
        with self.assertRaises(TypeError):
            RestEvent(0.5, Q)

    def test_invalid_voice_and_staff_rejected(self):
        with self.assertRaises(ValueError):
            RestEvent(0, Q, voice=0)
        with self.assertRaises(ValueError):
            RestEvent(0, Q, staff=-1)

    def test_event_immutable(self):
        event = RestEvent(0, Q)
        with self.assertRaises(FrozenInstanceError):
            event.voice = 2


class ChordEventTests(unittest.TestCase):
    def setUp(self):
        self.c = Pitch("C", 0, 4)
        self.e = Pitch("E", 0, 4)
        self.g = Pitch("G", 0, 4)
        self.b = Pitch("B", 0, 4)
        self.d = Pitch("D", 0, 5)

    def test_two_through_four_note_chords_allowed(self):
        for notes in (
            (self.c, self.e),
            (self.c, self.e, self.g),
            (self.c, self.e, self.g, self.b),
        ):
            with self.subTest(size=len(notes)):
                chord = ChordEvent(0, Q, notes)
                self.assertEqual(chord.notes, notes)

    def test_one_note_chord_rejected(self):
        with self.assertRaises(ValueError):
            ChordEvent(0, Q, (self.c,))

    def test_five_note_chord_rejected(self):
        with self.assertRaises(ValueError):
            ChordEvent(0, Q, (self.c, self.e, self.g, self.b, self.d))

    def test_empty_chord_rejected(self):
        with self.assertRaises(ValueError):
            ChordEvent(0, Q, ())

    def test_duplicate_pitch_rejected(self):
        with self.assertRaises(ValueError):
            ChordEvent(0, Q, (self.c, self.e, self.c))

    def test_mutable_note_container_rejected(self):
        with self.assertRaises(TypeError):
            ChordEvent(0, Q, [self.c, self.e])

    def test_chord_hash_and_equality_are_stable(self):
        first = ChordEvent(0, Q, (self.c, self.e, self.g))
        second = ChordEvent(
            Fraction(0, 8), RationalDuration(2, 8), (self.c, self.e, self.g)
        )
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))


if __name__ == "__main__":
    unittest.main()
