import unittest
from dataclasses import FrozenInstanceError
from fractions import Fraction

from st_omr_training.core import NoteEvent, Pitch, RationalDuration, RestEvent
from st_omr_training.structure import Clef, Measure, Part, Score, TimeSignature, Voice


Q = RationalDuration(1, 4)
H = RationalDuration(1, 2)


class TimeSignatureTests(unittest.TestCase):
    def test_supported_capacities_are_exact(self):
        self.assertEqual(TimeSignature(2, 4).capacity, Fraction(1, 2))
        self.assertEqual(TimeSignature(3, 4).capacity, Fraction(3, 4))
        self.assertEqual(TimeSignature(4, 4).capacity, Fraction(1, 1))

    def test_unsupported_signature_rejected(self):
        with self.assertRaises(ValueError):
            TimeSignature(6, 8)

    def test_bool_rejected(self):
        with self.assertRaises(TypeError):
            TimeSignature(True, 4)


class StructureTests(unittest.TestCase):
    def note(self, onset=0, duration=Q):
        return NoteEvent(onset, duration, Pitch("C", 0, 4))

    def test_voice_is_immutable(self):
        voice = Voice(1, (self.note(),))
        with self.assertRaises(FrozenInstanceError):
            voice.voice_id = 2

    def test_voice_rejects_mutable_event_container(self):
        with self.assertRaises(TypeError):
            Voice(1, [self.note()])

    def test_measure_defaults_match_contract(self):
        voice = Voice(1, (self.note(0, H),))
        measure = Measure(1, TimeSignature(2, 4), (voice,))
        self.assertEqual(measure.key_signature, 0)
        self.assertIs(measure.clef, Clef.TREBLE)
        self.assertEqual(measure.expected_duration, Fraction(1, 2))

    def test_measure_normalizes_explicit_expected_duration(self):
        voice = Voice(1, (self.note(0, H),))
        measure = Measure(1, TimeSignature(2, 4), (voice,), expected_duration=Fraction(2, 4))
        self.assertEqual(measure.expected_duration, Fraction(1, 2))

    def test_part_keeps_explicit_staff_structure(self):
        measure = Measure(1, TimeSignature(2, 4), (Voice(1, (self.note(0, H),)),))
        part = Part("P1", (measure,))
        self.assertEqual(part.staff_count, 1)
        self.assertEqual(part.instrument_class, "generic_treble_staff")

    def test_score_keeps_reproduction_metadata(self):
        measure = Measure(1, TimeSignature(2, 4), (Voice(1, (self.note(0, H),)),))
        part = Part("P1", (measure,))
        score = Score(
            "score-1",
            "1",
            "stage-1c-fixture",
            7,
            (("source_type", "procedural"),),
            (part,),
        )
        self.assertEqual(score.seed, 7)
        self.assertEqual(score.provenance, (("source_type", "procedural"),))

    def test_score_rejects_mutable_provenance(self):
        measure = Measure(1, TimeSignature(2, 4), (Voice(1, (self.note(0, H),)),))
        part = Part("P1", (measure,))
        with self.assertRaises(TypeError):
            Score("s", "1", "g", 1, {"source": "x"}, (part,))

    def test_rest_can_fill_timeline_as_explicit_silence(self):
        voice = Voice(1, (RestEvent(0, Q), self.note(Fraction(1, 4), Q)))
        measure = Measure(1, TimeSignature(2, 4), (voice,))
        self.assertEqual(measure.expected_duration, Fraction(1, 2))


if __name__ == "__main__":
    unittest.main()
