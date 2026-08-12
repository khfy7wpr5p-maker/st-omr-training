import unittest
from fractions import Fraction

from st_omr_training.core import ChordEvent, NoteEvent, Pitch, RationalDuration, RestEvent
from st_omr_training.structure import Measure, Part, Score, TimeSignature, Voice
from st_omr_training.structure_validator import (
    validate_measure,
    validate_part,
    validate_score,
    validate_time_signature,
    validate_voice,
)

Q = RationalDuration(1, 4)
E = RationalDuration(1, 8)
H = RationalDuration(1, 2)
W = RationalDuration(1, 1)


def codes(result):
    return [issue.code for issue in result.issues]


def corrupt(obj, name, value):
    object.__setattr__(obj, name, value)
    return obj


class Fixtures:
    def note(self, onset, duration=Q, pitch=None, voice=1, staff=1):
        return NoteEvent(onset, duration, pitch or Pitch("C", 0, 4), voice=voice, staff=staff)

    def valid_measure(self, signature=(4, 4)):
        ts = TimeSignature(*signature)
        if signature == (4, 4):
            events = (self.note(0, Q), self.note(Fraction(1, 4), Q), self.note(Fraction(1, 2), H))
        elif signature == (3, 4):
            events = (self.note(0, Q), self.note(Fraction(1, 4), H))
        else:
            events = (self.note(0, Q), self.note(Fraction(1, 4), Q))
        return Measure(1, ts, (Voice(1, events),))

    def valid_score(self, measure=None):
        m = measure or self.valid_measure()
        part = Part("P1", (m,))
        return Score("score-1", "1", "fixture", 42, (("source_type", "procedural"),), (part,))


class TimeSignatureValidatorTests(unittest.TestCase):
    def test_valid_signatures(self):
        for sig in ((2, 4), (3, 4), (4, 4)):
            self.assertTrue(validate_time_signature(TimeSignature(*sig)).is_valid)

    def test_corrupt_unsupported_signature_detected(self):
        ts = TimeSignature(4, 4)
        corrupt(ts, "numerator", 6)
        corrupt(ts, "denominator", 8)
        self.assertIn("time_signature.unsupported", codes(validate_time_signature(ts)))


class VoiceValidatorTests(unittest.TestCase, Fixtures):
    def test_valid_voice(self):
        self.assertTrue(validate_voice(Voice(1, (self.note(0),))).is_valid)

    def test_voice_id_two_rejected_by_v1_policy(self):
        voice = Voice(2, (self.note(0, voice=2),))
        self.assertIn("v1.voice_id", codes(validate_voice(voice)))

    def test_event_voice_must_match_container(self):
        voice = Voice(1, (self.note(0, voice=2),))
        self.assertIn("voice.event_voice", codes(validate_voice(voice)))

    def test_out_of_order_events_rejected(self):
        voice = Voice(1, (self.note(Fraction(1, 4)), self.note(0)))
        self.assertIn("voice.event_order", codes(validate_voice(voice)))


class MeasureValidatorTests(unittest.TestCase, Fixtures):
    def test_exact_4_4_fill_passes(self):
        self.assertTrue(validate_measure(self.valid_measure((4, 4))).is_valid)

    def test_exact_3_4_fill_passes(self):
        self.assertTrue(validate_measure(self.valid_measure((3, 4))).is_valid)

    def test_exact_2_4_fill_passes(self):
        self.assertTrue(validate_measure(self.valid_measure((2, 4))).is_valid)

    def test_three_quarters_in_4_4_is_underflow(self):
        events = (self.note(0), self.note(Fraction(1, 4)), self.note(Fraction(1, 2)))
        m = Measure(1, TimeSignature(4, 4), (Voice(1, events),))
        self.assertIn("measure.underflow", codes(validate_measure(m)))

    def test_whole_note_in_3_4_is_overflow(self):
        m = Measure(1, TimeSignature(3, 4), (Voice(1, (self.note(0, W),)),))
        self.assertIn("measure.overflow", codes(validate_measure(m)))

    def test_gap_requires_explicit_rest(self):
        events = (self.note(0, Q), self.note(Fraction(1, 2), H))
        m = Measure(1, TimeSignature(4, 4), (Voice(1, events),))
        result = codes(validate_measure(m))
        self.assertIn("measure.gap", result)
        self.assertIn("measure.underflow", result)

    def test_overlap_rejected(self):
        events = (self.note(0, H), self.note(Fraction(1, 4), H), self.note(Fraction(3, 4), Q))
        m = Measure(1, TimeSignature(4, 4), (Voice(1, events),))
        self.assertIn("measure.overlap", codes(validate_measure(m)))

    def test_explicit_rest_can_fill_gap(self):
        events = (
            self.note(0, Q),
            RestEvent(Fraction(1, 4), Q),
            self.note(Fraction(1, 2), H),
        )
        m = Measure(1, TimeSignature(4, 4), (Voice(1, events),))
        self.assertTrue(validate_measure(m).is_valid)

    def test_dotted_duration_is_not_v1(self):
        dotted = RationalDuration(3, 8)
        events = (
            self.note(0, dotted),
            self.note(Fraction(3, 8), Q),
            RestEvent(Fraction(5, 8), E),
            self.note(Fraction(3, 4), Q),
        )
        m = Measure(1, TimeSignature(4, 4), (Voice(1, events),))
        self.assertIn("measure.event_duration_unsupported", codes(validate_measure(m)))

    def test_whole_rest_is_deferred(self):
        m = Measure(1, TimeSignature(4, 4), (Voice(1, (RestEvent(0, W),)),))
        self.assertIn("measure.event_duration_unsupported", codes(validate_measure(m)))

    def test_key_signature_must_be_zero(self):
        m = self.valid_measure()
        corrupt(m, "key_signature", 1)
        self.assertIn("v1.key_signature", codes(validate_measure(m)))

    def test_clef_must_be_treble(self):
        m = self.valid_measure()
        corrupt(m, "clef", "bass")
        self.assertIn("v1.clef", codes(validate_measure(m)))

    def test_exactly_one_voice(self):
        m = self.valid_measure()
        corrupt(m, "voices", (m.voices[0], m.voices[0]))
        self.assertIn("v1.voice_count", codes(validate_measure(m)))

    def test_expected_duration_must_match_signature(self):
        m = self.valid_measure((3, 4))
        corrupt(m, "expected_duration", Fraction(1, 1))
        self.assertIn("measure.expected_duration_mismatch", codes(validate_measure(m)))

    def test_chord_consumes_one_shared_timeline_duration(self):
        notes = (
            self.note(0, H, Pitch("C", 0, 4)),
            self.note(0, H, Pitch("E", 0, 4)),
            self.note(0, H, Pitch("G", 0, 4)),
        )
        chord = ChordEvent(0, H, notes)
        m = Measure(1, TimeSignature(2, 4), (Voice(1, (chord,)),))
        self.assertTrue(validate_measure(m).is_valid)


class PartAndScoreValidatorTests(unittest.TestCase, Fixtures):
    def test_valid_part_and_score(self):
        score = self.valid_score()
        self.assertTrue(validate_part(score.parts[0]).is_valid)
        self.assertTrue(validate_score(score).is_valid)

    def test_staff_count_two_rejected(self):
        score = self.valid_score()
        part = score.parts[0]
        corrupt(part, "staff_count", 2)
        self.assertIn("v1.staff_count", codes(validate_part(part)))

    def test_measure_numbers_must_be_sequential(self):
        first = self.valid_measure((2, 4))
        second = self.valid_measure((2, 4))
        corrupt(second, "number", 3)
        part = Part("P1", (first, second))
        self.assertIn("part.measure_number_sequence", codes(validate_part(part)))

    def test_score_requires_one_part(self):
        score = self.valid_score()
        corrupt(score, "parts", (score.parts[0], score.parts[0]))
        self.assertIn("v1.part_count", codes(validate_score(score)))

    def test_empty_part_rejected(self):
        part = Part("P1", ())
        self.assertIn("part.empty", codes(validate_part(part)))

    def test_corrupt_score_metadata_detected(self):
        score = self.valid_score()
        corrupt(score, "score_id", "")
        corrupt(score, "seed", True)
        result = codes(validate_score(score))
        self.assertIn("score.score_id", result)
        self.assertIn("score.seed", result)

    def test_issue_order_is_deterministic(self):
        m = self.valid_measure()
        corrupt(m, "key_signature", 1)
        first = validate_measure(m)
        second = validate_measure(m)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
