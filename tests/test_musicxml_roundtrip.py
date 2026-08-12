import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import xml.etree.ElementTree as ET

from st_omr_training.core import (
    ChordEvent,
    DisplayAccidental,
    NoteEvent,
    NotationIntent,
    Pitch,
    RationalDuration,
    RestEvent,
)
from st_omr_training.generator import GeneratorConfig, generate_score
from st_omr_training.musicxml_roundtrip import (
    SemanticEventProjection,
    SemanticMeasureProjection,
    SemanticPartProjection,
    SemanticPitchProjection,
    SemanticScoreProjection,
    SemanticVoiceProjection,
    SupportedV1RoundTripError,
    compare_semantic_projections,
    parse_supported_v1_musicxml_projection,
    project_score_semantics,
    verify_supported_v1_round_trip,
)
from st_omr_training.musicxml_writer import write_musicxml
from st_omr_training.structure import Measure, Part, Score, TimeSignature, Voice


E = RationalDuration(1, 8)
Q = RationalDuration(1, 4)
H = RationalDuration(1, 2)
GOLDEN_DIR = Path(__file__).with_name("golden")


def note(onset, duration, step, alter=0, octave=4, accidental=DisplayAccidental.NONE):
    return NoteEvent(
        onset,
        duration,
        Pitch(step, alter, octave),
        NotationIntent(accidental),
    )


def rich_score(score_id="roundtrip-rich", seed=7, provenance=(("source_type", "targeted"),)):
    first = Measure(
        1,
        TimeSignature(2, 4),
        (
            Voice(
                1,
                (
                    note(0, Q, "C", 1, 4, DisplayAccidental.SHARP),
                    RestEvent(Fraction(1, 4), Q),
                ),
            ),
        ),
    )
    chord_notes = (
        note(0, H, "D", -1, 4, DisplayAccidental.FLAT),
        note(0, H, "F", 0, 4, DisplayAccidental.NATURAL),
        note(0, H, "A", 0, 4),
    )
    second = Measure(
        2,
        TimeSignature(3, 4),
        (
            Voice(
                1,
                (
                    ChordEvent(0, H, chord_notes),
                    note(Fraction(1, 2), Q, "B", 0, 4),
                ),
            ),
        ),
    )
    return Score(
        score_id,
        "st-canonical-1",
        "roundtrip-test",
        seed,
        provenance,
        (Part("P1", (first, second)),),
    )


def codes(result):
    return [issue.code for issue in result.issues]


class CanonicalProjectionTests(unittest.TestCase):
    def test_projection_contains_only_frozen_semantics(self):
        projection = project_score_semantics(rich_score())
        self.assertEqual(len(projection.parts), 1)
        part = projection.parts[0]
        self.assertEqual(part.part_id, "P1")
        self.assertEqual(part.staff_count, 1)
        self.assertEqual([m.number for m in part.measures], [1, 2])
        self.assertEqual([m.time_signature for m in part.measures], [(2, 4), (3, 4)])
        self.assertEqual([m.key_signature for m in part.measures], [0, 0])
        self.assertEqual([m.clef for m in part.measures], ["treble", "treble"])

    def test_generator_only_metadata_is_not_part_of_projection(self):
        left = project_score_semantics(rich_score("one", 1, (("a", "b"),)))
        right = project_score_semantics(rich_score("two", 999, (("x", "y"),)))
        self.assertEqual(left, right)

    def test_projection_preserves_note_rest_chord_and_member_order(self):
        projection = project_score_semantics(rich_score())
        first_events = projection.parts[0].measures[0].voices[0].events
        second_events = projection.parts[0].measures[1].voices[0].events
        self.assertEqual([event.event_type for event in first_events], ["note", "rest"])
        self.assertEqual([event.event_type for event in second_events], ["chord", "note"])
        chord = second_events[0]
        self.assertEqual([pitch.step for pitch in chord.pitches], ["D", "F", "A"])
        self.assertEqual(
            [pitch.display_accidental for pitch in chord.pitches],
            [DisplayAccidental.FLAT, DisplayAccidental.NATURAL, DisplayAccidental.NONE],
        )

    def test_projection_preserves_exact_fractional_onsets_and_durations(self):
        projection = project_score_semantics(rich_score())
        first_events = projection.parts[0].measures[0].voices[0].events
        self.assertEqual(first_events[0].onset, Fraction(0, 1))
        self.assertEqual(first_events[0].duration, Fraction(1, 4))
        self.assertEqual(first_events[1].onset, Fraction(1, 4))
        self.assertEqual(first_events[1].duration, Fraction(1, 4))

    def test_non_score_is_rejected(self):
        with self.assertRaises(SupportedV1RoundTripError) as caught:
            project_score_semantics(object())
        self.assertIn("score.type", codes(caught.exception.validation))


class LimitedParserTests(unittest.TestCase):
    def test_writer_output_projects_identically(self):
        score = rich_score()
        expected = project_score_semantics(score)
        actual = parse_supported_v1_musicxml_projection(write_musicxml(score))
        self.assertEqual(actual, expected)

    def test_time_signature_change_is_carried_into_effective_measure_semantics(self):
        projection = parse_supported_v1_musicxml_projection(write_musicxml(rich_score()))
        measures = projection.parts[0].measures
        self.assertEqual(measures[0].time_signature, (2, 4))
        self.assertEqual(measures[1].time_signature, (3, 4))

    def test_chord_continuations_do_not_advance_onset(self):
        projection = parse_supported_v1_musicxml_projection(write_musicxml(rich_score()))
        events = projection.parts[0].measures[1].voices[0].events
        self.assertEqual(events[0].event_type, "chord")
        self.assertEqual(events[0].onset, Fraction(0, 1))
        self.assertEqual(events[0].duration, Fraction(1, 2))
        self.assertEqual(events[1].onset, Fraction(1, 2))

    def test_all_stage2b_golden_fixtures_are_accepted_by_limited_parser(self):
        for path in sorted(GOLDEN_DIR.glob("*.musicxml")):
            with self.subTest(path=path.name):
                projection = parse_supported_v1_musicxml_projection(path.read_bytes())
                self.assertEqual(projection.parts[0].part_id, "P1")

    def test_wrong_namespace_is_rejected_not_normalized(self):
        data = write_musicxml(rich_score()).replace(
            b"<score-partwise ", b'<score-partwise xmlns="urn:not-v1" ', 1
        )
        with self.assertRaises(SupportedV1RoundTripError) as caught:
            parse_supported_v1_musicxml_projection(data)
        self.assertIn("musicxml.root", codes(caught.exception.validation))

    def test_unsupported_direction_is_rejected_not_ignored(self):
        root = ET.fromstring(write_musicxml(rich_score()))
        root.find("./part/measure").append(ET.Element("direction"))
        data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        with self.assertRaises(SupportedV1RoundTripError) as caught:
            parse_supported_v1_musicxml_projection(data)
        self.assertIn("musicxml.unsupported_element", codes(caught.exception.validation))

    def test_doctype_is_rejected_before_projection(self):
        data = b'<?xml version="1.0"?><!DOCTYPE score-partwise><score-partwise version="4.0"/>'
        with self.assertRaises(SupportedV1RoundTripError) as caught:
            parse_supported_v1_musicxml_projection(data)
        self.assertIn("musicxml.doctype_forbidden", codes(caught.exception.validation))

    def test_non_bytes_input_is_rejected(self):
        with self.assertRaises(SupportedV1RoundTripError) as caught:
            parse_supported_v1_musicxml_projection("<score-partwise/>")
        self.assertIn("musicxml.input_type", codes(caught.exception.validation))


class ComparatorTests(unittest.TestCase):
    def base_projection(self):
        pitch = SemanticPitchProjection("C", 0, 4, DisplayAccidental.NONE)
        event = SemanticEventProjection("note", Fraction(0), Fraction(1, 4), 1, (pitch,))
        voice = SemanticVoiceProjection(1, (event,))
        measure = SemanticMeasureProjection(1, (2, 4), 0, "treble", (voice,))
        part = SemanticPartProjection("P1", 1, (measure,))
        return SemanticScoreProjection((part,))

    def test_equal_projections_pass(self):
        projection = self.base_projection()
        self.assertTrue(compare_semantic_projections(projection, projection).is_valid)

    def test_wrong_projection_type_fails_closed(self):
        self.assertEqual(
            codes(compare_semantic_projections(self.base_projection(), object())),
            ["roundtrip.projection_type"],
        )

    def test_structure_difference_is_reported(self):
        base = self.base_projection()
        changed_part = replace(base.parts[0], part_id="P2", staff_count=2)
        changed = replace(base, parts=(changed_part,))
        result = codes(compare_semantic_projections(base, changed))
        self.assertIn("roundtrip.part_id", result)
        self.assertIn("roundtrip.staff_count", result)

    def test_measure_and_voice_difference_is_reported(self):
        base = self.base_projection()
        measure = base.parts[0].measures[0]
        voice = replace(measure.voices[0], voice_id=2)
        changed_measure = replace(measure, number=2, time_signature=(3, 4), key_signature=1, clef="bass", voices=(voice,))
        changed_part = replace(base.parts[0], measures=(changed_measure,))
        result = codes(compare_semantic_projections(base, replace(base, parts=(changed_part,))))
        for code in (
            "roundtrip.measure_number",
            "roundtrip.time_signature",
            "roundtrip.key_signature",
            "roundtrip.clef",
            "roundtrip.voice_id",
        ):
            self.assertIn(code, result)

    def test_event_timing_type_staff_and_pitch_difference_is_reported(self):
        base = self.base_projection()
        measure = base.parts[0].measures[0]
        event = measure.voices[0].events[0]
        pitch = replace(
            event.pitches[0],
            step="D",
            alter=1,
            octave=5,
            display_accidental=DisplayAccidental.SHARP,
        )
        changed_event = replace(
            event,
            event_type="chord",
            onset=Fraction(1, 8),
            duration=Fraction(1, 8),
            staff=2,
            pitches=(pitch, pitch),
        )
        voice = replace(measure.voices[0], events=(changed_event,))
        changed_measure = replace(measure, voices=(voice,))
        changed_part = replace(base.parts[0], measures=(changed_measure,))
        result = codes(compare_semantic_projections(base, replace(base, parts=(changed_part,))))
        for code in (
            "roundtrip.event_type",
            "roundtrip.onset",
            "roundtrip.duration",
            "roundtrip.staff",
            "roundtrip.pitch_count",
            "roundtrip.pitch_step",
            "roundtrip.pitch_alter",
            "roundtrip.pitch_octave",
            "roundtrip.display_accidental",
        ):
            self.assertIn(code, result)


class EndToEndRoundTripTests(unittest.TestCase):
    def test_rich_score_round_trip_passes(self):
        self.assertTrue(verify_supported_v1_round_trip(rich_score()).is_valid)

    def test_generated_scores_round_trip(self):
        config = GeneratorConfig(measure_count=6)
        for seed in range(100):
            with self.subTest(seed=seed):
                self.assertTrue(verify_supported_v1_round_trip(generate_score(config, seed)).is_valid)

    def test_invalid_canonical_input_fails_closed(self):
        result = verify_supported_v1_round_trip(object())
        self.assertEqual(codes(result)[:2], ["roundtrip.canonical_invalid", "score.type"])


if __name__ == "__main__":
    unittest.main()
