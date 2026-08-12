import unittest
from fractions import Fraction

from st_omr_training import (
    ChordEvent,
    DisplayAccidental,
    NoteEvent,
    NotationIntent,
    Pitch,
    RationalDuration,
    RestEvent,
    ValidationIssue,
    ValidationResult,
    validate_chord_event,
    validate_note_event,
    validate_rest_event,
    validate_v1_event,
)


Q = RationalDuration(1, 4)


def corrupt(obj, name, value):
    object.__setattr__(obj, name, value)
    return obj


def codes(result):
    return [issue.code for issue in result.issues]


class ValidationResultTests(unittest.TestCase):
    def test_valid_result(self):
        result = ValidationResult()
        self.assertTrue(result.is_valid)
        self.assertEqual(result.issues, ())

    def test_invalid_result(self):
        issue = ValidationIssue("x", "$", "bad")
        result = ValidationResult((issue,))
        self.assertFalse(result.is_valid)


class NoteValidatorTests(unittest.TestCase):
    def test_valid_note(self):
        note = NoteEvent(0, Q, Pitch("C", 0, 4))
        self.assertTrue(validate_note_event(note).is_valid)

    def test_wrong_object_type_is_reported(self):
        result = validate_note_event(RestEvent(0, Q))
        self.assertEqual(codes(result), ["event.type"])

    def test_v1_voice_and_staff_are_independently_enforced(self):
        note = NoteEvent(0, Q, Pitch("C", 0, 4), voice=2, staff=2)
        result = validate_note_event(note)
        self.assertIn("v1.voice", codes(result))
        self.assertIn("v1.staff", codes(result))

    def test_negative_onset_is_detected_after_corruption(self):
        note = NoteEvent(0, Q, Pitch("C", 0, 4))
        corrupt(note, "onset", Fraction(-1, 8))
        self.assertIn("onset.negative", codes(validate_note_event(note)))

    def test_noncanonical_onset_type_is_detected(self):
        note = NoteEvent(0, Q, Pitch("C", 0, 4))
        corrupt(note, "onset", 0)
        self.assertIn("onset.type", codes(validate_note_event(note)))

    def test_corrupt_nonpositive_duration_is_detected(self):
        duration = RationalDuration(1, 4)
        corrupt(duration, "numerator", 0)
        note = NoteEvent(0, Q, Pitch("C", 0, 4))
        corrupt(note, "duration", duration)
        self.assertIn("duration.non_positive", codes(validate_note_event(note)))

    def test_corrupt_unreduced_duration_is_detected(self):
        duration = RationalDuration(1, 4)
        corrupt(duration, "numerator", 2)
        corrupt(duration, "denominator", 8)
        note = NoteEvent(0, Q, Pitch("C", 0, 4))
        corrupt(note, "duration", duration)
        self.assertIn("duration.not_canonical", codes(validate_note_event(note)))

    def test_corrupt_pitch_fields_are_detected(self):
        pitch = Pitch("C", 0, 4)
        corrupt(pitch, "step", "H")
        corrupt(pitch, "alter", 2)
        corrupt(pitch, "octave", 10)
        note = NoteEvent(0, Q, Pitch("C", 0, 4))
        corrupt(note, "pitch", pitch)
        result_codes = codes(validate_note_event(note))
        self.assertIn("pitch.step", result_codes)
        self.assertIn("pitch.alter", result_codes)
        self.assertIn("pitch.octave", result_codes)

    def test_sharp_display_requires_sharp_pitch(self):
        note = NoteEvent(
            0,
            Q,
            Pitch("F", 0, 4),
            NotationIntent(DisplayAccidental.SHARP),
        )
        self.assertIn(
            "notation_intent.accidental_mismatch",
            codes(validate_note_event(note)),
        )

    def test_flat_display_requires_flat_pitch(self):
        note = NoteEvent(
            0,
            Q,
            Pitch("B", 0, 4),
            NotationIntent(DisplayAccidental.FLAT),
        )
        self.assertIn(
            "notation_intent.accidental_mismatch",
            codes(validate_note_event(note)),
        )

    def test_natural_display_requires_unaltered_pitch(self):
        note = NoteEvent(
            0,
            Q,
            Pitch("F", 1, 4),
            NotationIntent(DisplayAccidental.NATURAL),
        )
        self.assertIn(
            "notation_intent.accidental_mismatch",
            codes(validate_note_event(note)),
        )

    def test_matching_display_accidentals_are_valid(self):
        cases = (
            (Pitch("F", 1, 4), DisplayAccidental.SHARP),
            (Pitch("B", -1, 4), DisplayAccidental.FLAT),
            (Pitch("F", 0, 4), DisplayAccidental.NATURAL),
        )
        for pitch, accidental in cases:
            with self.subTest(accidental=accidental):
                note = NoteEvent(0, Q, pitch, NotationIntent(accidental))
                self.assertTrue(validate_note_event(note).is_valid)

    def test_none_display_does_not_force_alter(self):
        note = NoteEvent(0, Q, Pitch("F", 1, 4))
        self.assertTrue(validate_note_event(note).is_valid)


class RestValidatorTests(unittest.TestCase):
    def test_valid_rest(self):
        self.assertTrue(validate_rest_event(RestEvent(0, Q)).is_valid)

    def test_rest_v1_voice_policy(self):
        rest = RestEvent(0, Q, voice=2)
        self.assertIn("v1.voice", codes(validate_rest_event(rest)))


class ChordValidatorTests(unittest.TestCase):
    def note(self, pitch, *, onset=0, duration=Q, voice=1, staff=1):
        return NoteEvent(onset, duration, pitch, voice=voice, staff=staff)

    def chord(self):
        return ChordEvent(
            0,
            Q,
            (
                self.note(Pitch("C", 0, 4)),
                self.note(Pitch("E", 0, 4)),
                self.note(Pitch("G", 0, 4)),
            ),
        )

    def test_valid_chord(self):
        self.assertTrue(validate_chord_event(self.chord()).is_valid)

    def test_corrupt_size_is_detected(self):
        chord = self.chord()
        corrupt(chord, "notes", (chord.notes[0],))
        self.assertIn("chord.size", codes(validate_chord_event(chord)))

    def test_corrupt_member_type_is_detected(self):
        chord = self.chord()
        corrupt(chord, "notes", (chord.notes[0], "not-a-note"))
        self.assertIn("chord.member_type", codes(validate_chord_event(chord)))

    def test_duplicate_pitch_is_detected_after_corruption(self):
        chord = self.chord()
        duplicate = self.note(Pitch("C", 0, 4))
        corrupt(chord, "notes", (chord.notes[0], duplicate))
        self.assertIn("chord.duplicate_pitch", codes(validate_chord_event(chord)))

    def test_member_onset_mismatch_is_detected(self):
        chord = self.chord()
        corrupt(chord.notes[1], "onset", Fraction(1, 8))
        self.assertIn("chord.member_onset", codes(validate_chord_event(chord)))

    def test_member_duration_mismatch_is_detected(self):
        chord = self.chord()
        corrupt(chord.notes[1], "duration", RationalDuration(1, 8))
        self.assertIn("chord.member_duration", codes(validate_chord_event(chord)))

    def test_member_voice_and_staff_mismatch_are_detected(self):
        chord = self.chord()
        corrupt(chord.notes[1], "voice", 2)
        corrupt(chord.notes[2], "staff", 2)
        result_codes = codes(validate_chord_event(chord))
        self.assertIn("chord.member_voice", result_codes)
        self.assertIn("chord.member_staff", result_codes)
        self.assertIn("v1.voice", result_codes)
        self.assertIn("v1.staff", result_codes)

    def test_chord_level_v1_policy_is_enforced(self):
        notes = (
            self.note(Pitch("C", 0, 4), voice=2, staff=2),
            self.note(Pitch("E", 0, 4), voice=2, staff=2),
        )
        chord = ChordEvent(0, Q, notes, voice=2, staff=2)
        result_codes = codes(validate_chord_event(chord))
        self.assertIn("v1.voice", result_codes)
        self.assertIn("v1.staff", result_codes)

    def test_member_accidental_coherence_is_checked(self):
        chord = self.chord()
        bad = NoteEvent(
            0,
            Q,
            Pitch("F", 0, 4),
            NotationIntent(DisplayAccidental.SHARP),
        )
        corrupt(chord, "notes", (chord.notes[0], bad))
        self.assertIn(
            "notation_intent.accidental_mismatch",
            codes(validate_chord_event(chord)),
        )


class GenericValidatorTests(unittest.TestCase):
    def test_dispatches_all_supported_event_types(self):
        note = NoteEvent(0, Q, Pitch("C", 0, 4))
        rest = RestEvent(0, Q)
        chord = ChordEvent(
            0,
            Q,
            (
                NoteEvent(0, Q, Pitch("C", 0, 4)),
                NoteEvent(0, Q, Pitch("E", 0, 4)),
            ),
        )
        for event in (note, rest, chord):
            with self.subTest(event=type(event).__name__):
                self.assertTrue(validate_v1_event(event).is_valid)

    def test_unsupported_type_is_rejected(self):
        self.assertEqual(
            codes(validate_v1_event(object())),
            ["event.unsupported_type"],
        )

    def test_issue_order_is_deterministic(self):
        note = NoteEvent(0, Q, Pitch("C", 0, 4), voice=2, staff=2)
        first = validate_v1_event(note)
        second = validate_v1_event(note)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
