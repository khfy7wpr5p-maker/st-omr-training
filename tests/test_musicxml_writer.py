import unittest
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
from st_omr_training.musicxml_writer import (
    MUSICXML_PART_ID,
    MUSICXML_PART_NAME,
    MUSICXML_VERSION,
    MusicXMLWriteError,
    compute_musicxml_divisions,
    musicxml_sha256,
    write_musicxml,
)
from st_omr_training.structure import Measure, Part, Score, TimeSignature, Voice


E = RationalDuration(1, 8)
Q = RationalDuration(1, 4)
H = RationalDuration(1, 2)
W = RationalDuration(1, 1)
GOLDEN_DIR = Path(__file__).with_name("golden")


def note(onset, duration, step="C", alter=0, octave=4, accidental=DisplayAccidental.NONE):
    return NoteEvent(
        onset,
        duration,
        Pitch(step, alter, octave),
        NotationIntent(accidental),
    )


def chord(onset, duration, pitches):
    members = tuple(
        note(onset, duration, step, alter, octave, accidental)
        for step, alter, octave, accidental in pitches
    )
    return ChordEvent(onset, duration, members)


def make_measure(number, signature, events):
    return Measure(
        number,
        TimeSignature(*signature),
        (Voice(1, tuple(events)),),
    )


def make_score(*measures, part_id="P1"):
    return Score(
        "musicxml-writer-fixture",
        "st-canonical-1",
        "test-fixture",
        7,
        (("source_type", "targeted"),),
        (Part(part_id, tuple(measures)),),
    )


def golden_bytes(name):
    return (GOLDEN_DIR / name).read_bytes()


class MusicXMLDivisionsTests(unittest.TestCase):
    def test_quarter_half_whole_only_uses_one_division_per_quarter(self):
        score = make_score(
            make_measure(
                1,
                (4, 4),
                (
                    note(0, Q),
                    note(Fraction(1, 4), Q, "D"),
                    note(Fraction(1, 2), H, "E"),
                ),
            )
        )
        self.assertEqual(compute_musicxml_divisions(score), 1)

    def test_eighth_notes_require_two_divisions_per_quarter(self):
        score = make_score(
            make_measure(
                1,
                (2, 4),
                tuple(
                    note(Fraction(index, 8), E, step)
                    for index, step in enumerate(("C", "D", "E", "F"))
                ),
            )
        )
        self.assertEqual(compute_musicxml_divisions(score), 2)
        root = ET.fromstring(write_musicxml(score))
        self.assertEqual(root.findtext("./part/measure/attributes/divisions"), "2")
        self.assertEqual(
            [item.text for item in root.findall("./part/measure/note/duration")],
            ["1", "1", "1", "1"],
        )


class MusicXMLShapeTests(unittest.TestCase):
    def test_root_and_part_identity_are_frozen(self):
        score = make_score(make_measure(1, (2, 4), (note(0, H),)))
        root = ET.fromstring(write_musicxml(score))
        self.assertEqual(root.tag, "score-partwise")
        self.assertEqual(root.attrib, {"version": MUSICXML_VERSION})
        score_part = root.find("./part-list/score-part")
        part = root.find("./part")
        self.assertEqual(score_part.attrib["id"], MUSICXML_PART_ID)
        self.assertEqual(score_part.findtext("part-name"), MUSICXML_PART_NAME)
        self.assertEqual(part.attrib["id"], MUSICXML_PART_ID)
        self.assertNotIn("}", root.tag)

    def test_first_measure_attributes_follow_v1_contract(self):
        score = make_score(make_measure(1, (3, 4), (note(0, H), note(Fraction(1, 2), Q))))
        root = ET.fromstring(write_musicxml(score))
        attributes = root.find("./part/measure/attributes")
        self.assertEqual(
            [child.tag for child in attributes],
            ["divisions", "key", "time", "clef"],
        )
        self.assertEqual(attributes.findtext("key/fifths"), "0")
        self.assertEqual(attributes.findtext("time/beats"), "3")
        self.assertEqual(attributes.findtext("time/beat-type"), "4")
        self.assertEqual(attributes.findtext("clef/sign"), "G")
        self.assertEqual(attributes.findtext("clef/line"), "2")

    def test_time_change_emits_only_time_attributes_after_first_measure(self):
        score = make_score(
            make_measure(1, (2, 4), (note(0, H),)),
            make_measure(2, (3, 4), (note(0, H, "D"), note(Fraction(1, 2), Q, "E"))),
        )
        root = ET.fromstring(write_musicxml(score))
        second_attributes = root.find("./part/measure[@number='2']/attributes")
        self.assertEqual([child.tag for child in second_attributes], ["time"])
        self.assertIsNone(second_attributes.find("divisions"))
        self.assertIsNone(second_attributes.find("key"))
        self.assertIsNone(second_attributes.find("clef"))

    def test_unchanged_time_signature_does_not_emit_redundant_attributes(self):
        score = make_score(
            make_measure(1, (2, 4), (note(0, H),)),
            make_measure(2, (2, 4), (note(0, H, "D"),)),
        )
        root = ET.fromstring(write_musicxml(score))
        self.assertIsNone(root.find("./part/measure[@number='2']/attributes"))


class MusicXMLEventMappingTests(unittest.TestCase):
    def test_note_child_order_and_pitch_mapping(self):
        event = note(0, H, "F", 1, 5, DisplayAccidental.SHARP)
        root = ET.fromstring(write_musicxml(make_score(make_measure(1, (2, 4), (event,)))))
        xml_note = root.find("./part/measure/note")
        self.assertEqual(
            [child.tag for child in xml_note],
            ["pitch", "duration", "voice", "type", "accidental", "staff"],
        )
        self.assertEqual(xml_note.findtext("pitch/step"), "F")
        self.assertEqual(xml_note.findtext("pitch/alter"), "1")
        self.assertEqual(xml_note.findtext("pitch/octave"), "5")
        self.assertEqual(xml_note.findtext("type"), "half")
        self.assertEqual(xml_note.findtext("accidental"), "sharp")

    def test_unaltered_pitch_omits_alter_but_can_emit_natural(self):
        event = note(0, H, "C", 0, 4, DisplayAccidental.NATURAL)
        root = ET.fromstring(write_musicxml(make_score(make_measure(1, (2, 4), (event,)))))
        xml_note = root.find("./part/measure/note")
        self.assertIsNone(xml_note.find("pitch/alter"))
        self.assertEqual(xml_note.findtext("accidental"), "natural")

    def test_none_accidental_intent_is_omitted(self):
        event = note(0, H, "F", 1, 4, DisplayAccidental.NONE)
        root = ET.fromstring(write_musicxml(make_score(make_measure(1, (2, 4), (event,)))))
        xml_note = root.find("./part/measure/note")
        self.assertEqual(xml_note.findtext("pitch/alter"), "1")
        self.assertIsNone(xml_note.find("accidental"))

    def test_rest_maps_to_rest_note_without_pitch(self):
        event = RestEvent(0, H)
        root = ET.fromstring(write_musicxml(make_score(make_measure(1, (2, 4), (event,)))))
        xml_note = root.find("./part/measure/note")
        self.assertIsNotNone(xml_note.find("rest"))
        self.assertIsNone(xml_note.find("pitch"))
        self.assertEqual(xml_note.findtext("duration"), "2")
        self.assertEqual(xml_note.findtext("type"), "half")

    def test_chord_members_preserve_order_and_use_chord_marker_after_first(self):
        event = chord(
            0,
            H,
            (
                ("C", 0, 4, DisplayAccidental.NONE),
                ("E", 0, 4, DisplayAccidental.NONE),
                ("G", 0, 4, DisplayAccidental.NONE),
            ),
        )
        root = ET.fromstring(write_musicxml(make_score(make_measure(1, (2, 4), (event,)))))
        notes = root.findall("./part/measure/note")
        self.assertEqual([item.findtext("pitch/step") for item in notes], ["C", "E", "G"])
        self.assertIsNone(notes[0].find("chord"))
        self.assertIsNotNone(notes[1].find("chord"))
        self.assertIsNotNone(notes[2].find("chord"))
        self.assertEqual([item.findtext("duration") for item in notes], ["2", "2", "2"])
        self.assertEqual([child.tag for child in notes[1]][0], "chord")

    def test_writer_does_not_emit_backup_or_forward_for_v1(self):
        score = make_score(
            make_measure(
                1,
                (4, 4),
                (
                    note(0, Q),
                    RestEvent(Fraction(1, 4), Q),
                    note(Fraction(1, 2), H, "G"),
                ),
            )
        )
        root = ET.fromstring(write_musicxml(score))
        self.assertEqual(root.findall(".//backup"), [])
        self.assertEqual(root.findall(".//forward"), [])


class MusicXMLGoldenTests(unittest.TestCase):
    def test_basic_2_4_golden(self):
        score = make_score(
            make_measure(1, (2, 4), (note(0, Q, "C"), note(Fraction(1, 4), Q, "D")))
        )
        self.assertEqual(write_musicxml(score), golden_bytes("basic_2_4.musicxml"))

    def test_rest_3_4_golden(self):
        score = make_score(
            make_measure(1, (3, 4), (RestEvent(0, H), note(Fraction(1, 2), Q, "E")))
        )
        self.assertEqual(write_musicxml(score), golden_bytes("rest_3_4.musicxml"))

    def test_basic_4_4_golden(self):
        score = make_score(make_measure(1, (4, 4), (note(0, W, "G"),)))
        self.assertEqual(write_musicxml(score), golden_bytes("basic_4_4.musicxml"))

    def test_chords_2_3_4_golden(self):
        c2 = chord(
            0,
            Q,
            (("C", 0, 4, DisplayAccidental.NONE), ("E", 0, 4, DisplayAccidental.NONE)),
        )
        c3 = chord(
            Fraction(1, 4),
            Q,
            (
                ("D", 0, 4, DisplayAccidental.NONE),
                ("F", 0, 4, DisplayAccidental.NONE),
                ("A", 0, 4, DisplayAccidental.NONE),
            ),
        )
        c4 = chord(
            Fraction(1, 2),
            H,
            (
                ("C", 0, 4, DisplayAccidental.NONE),
                ("E", 0, 4, DisplayAccidental.NONE),
                ("G", 0, 4, DisplayAccidental.NONE),
                ("B", 0, 4, DisplayAccidental.NONE),
            ),
        )
        score = make_score(make_measure(1, (4, 4), (c2, c3, c4)))
        self.assertEqual(write_musicxml(score), golden_bytes("chords_2_3_4.musicxml"))

    def test_accidentals_golden(self):
        score = make_score(
            make_measure(
                1,
                (4, 4),
                (
                    note(0, Q, "C", 1, 4, DisplayAccidental.SHARP),
                    note(Fraction(1, 4), Q, "C", 0, 4, DisplayAccidental.NATURAL),
                    note(Fraction(1, 2), Q, "D", -1, 4, DisplayAccidental.FLAT),
                    note(Fraction(3, 4), Q, "D", 0, 4, DisplayAccidental.NATURAL),
                ),
            )
        )
        self.assertEqual(write_musicxml(score), golden_bytes("accidentals.musicxml"))

    def test_time_change_golden(self):
        score = make_score(
            make_measure(1, (2, 4), (note(0, H, "C"),)),
            make_measure(2, (3, 4), (note(0, H, "D"), note(Fraction(1, 2), Q, "E"))),
        )
        self.assertEqual(write_musicxml(score), golden_bytes("time_change.musicxml"))


class MusicXMLSafetyAndDeterminismTests(unittest.TestCase):
    def test_writer_rejects_score_that_fails_independent_validation(self):
        score = make_score(make_measure(1, (2, 4), (note(0, H),)))
        object.__setattr__(score.parts[0].measures[0], "key_signature", 1)
        with self.assertRaises(MusicXMLWriteError):
            write_musicxml(score)

    def test_writer_rejects_noncanonical_part_id(self):
        score = make_score(make_measure(1, (2, 4), (note(0, H),)), part_id="OTHER")
        with self.assertRaises(MusicXMLWriteError):
            write_musicxml(score)

    def test_writer_rejects_non_score_object(self):
        with self.assertRaises(MusicXMLWriteError):
            write_musicxml(object())

    def test_same_score_produces_identical_bytes_and_digest(self):
        score = generate_score(
            GeneratorConfig(measure_count=12),
            20260812,
        )
        first = write_musicxml(score)
        second = write_musicxml(score)
        self.assertEqual(first, second)
        self.assertEqual(musicxml_sha256(first), musicxml_sha256(second))
        self.assertEqual(len(musicxml_sha256(first)), 64)

    def test_digest_requires_bytes(self):
        with self.assertRaises(TypeError):
            musicxml_sha256("not-bytes")

    def test_serialized_generated_scores_are_well_formed(self):
        for seed in range(50):
            with self.subTest(seed=seed):
                data = write_musicxml(generate_score(GeneratorConfig(measure_count=4), seed))
                root = ET.fromstring(data)
                self.assertEqual(root.tag, "score-partwise")
                self.assertEqual(root.attrib.get("version"), "4.0")


if __name__ == "__main__":
    unittest.main()
