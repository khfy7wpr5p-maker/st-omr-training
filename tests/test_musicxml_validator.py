import hashlib
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

from st_omr_training.core import NoteEvent, Pitch, RationalDuration
from st_omr_training.generator import GeneratorConfig, generate_score
from st_omr_training.musicxml_validator import (
    MAX_MUSICXML_BYTES,
    MUSICXML_SCHEMA_SHA256,
    validate_musicxml,
    validate_musicxml_semantics,
    validate_musicxml_xsd,
    verify_musicxml_schema_assets,
)
from st_omr_training.musicxml_writer import write_musicxml
from st_omr_training.structure import Measure, Part, Score, TimeSignature, Voice


Q = RationalDuration(1, 4)
GOLDEN_DIR = Path(__file__).with_name("golden")


def codes(result):
    return [issue.code for issue in result.issues]


def simple_score():
    events = (
        NoteEvent(0, Q, Pitch("C", 0, 4)),
        NoteEvent(Q.fraction, Q, Pitch("D", 0, 4)),
    )
    measure = Measure(1, TimeSignature(2, 4), (Voice(1, events),))
    return Score(
        "validator-fixture",
        "st-canonical-1",
        "validator-test",
        1,
        (("source_type", "targeted"),),
        (Part("P1", (measure,)),),
    )


def valid_xml():
    return write_musicxml(simple_score())


def xml_mutation(callback):
    root = ET.fromstring(valid_xml())
    callback(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


class InputSafetyTests(unittest.TestCase):
    def test_bytes_only(self):
        self.assertEqual(codes(validate_musicxml_semantics("<x/>")), ["musicxml.input_type"])
        self.assertEqual(codes(validate_musicxml_xsd("<x/>")), ["musicxml.input_type"])

    def test_empty_rejected(self):
        self.assertEqual(codes(validate_musicxml_semantics(b"")), ["musicxml.empty"])

    def test_oversized_rejected(self):
        data = b"x" * (MAX_MUSICXML_BYTES + 1)
        self.assertEqual(codes(validate_musicxml_semantics(data)), ["musicxml.too_large"])

    def test_doctype_and_external_entity_surface_are_rejected_before_parse(self):
        data = b'<?xml version="1.0"?><!DOCTYPE score-partwise [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><score-partwise version="4.0">&xxe;</score-partwise>'
        self.assertEqual(codes(validate_musicxml_semantics(data)), ["musicxml.doctype_forbidden"])
        self.assertEqual(codes(validate_musicxml_xsd(data)), ["musicxml.doctype_forbidden"])
        self.assertEqual(codes(validate_musicxml(data)), ["musicxml.doctype_forbidden"])

    def test_malformed_xml_rejected(self):
        self.assertEqual(codes(validate_musicxml_semantics(b"<score-partwise>")), ["musicxml.malformed"])


class SemanticShapeTests(unittest.TestCase):
    def test_writer_output_passes_semantic_validation(self):
        self.assertTrue(validate_musicxml_semantics(valid_xml()).is_valid)

    def test_stage2b_golden_fixtures_pass_semantics(self):
        for path in sorted(GOLDEN_DIR.glob("*.musicxml")):
            with self.subTest(path=path.name):
                self.assertTrue(validate_musicxml_semantics(path.read_bytes()).is_valid)

    def test_generated_writer_outputs_pass_semantics(self):
        for seed in range(100):
            with self.subTest(seed=seed):
                data = write_musicxml(generate_score(GeneratorConfig(measure_count=5), seed))
                self.assertTrue(validate_musicxml_semantics(data).is_valid)

    def test_wrong_root_rejected(self):
        data = xml_mutation(lambda root: setattr(root, "tag", "score-timewise"))
        self.assertIn("musicxml.root", codes(validate_musicxml_semantics(data)))

    def test_default_namespace_rejected(self):
        data = valid_xml().replace(b"<score-partwise ", b'<score-partwise xmlns="urn:not-v1" ', 1)
        self.assertIn("musicxml.root", codes(validate_musicxml_semantics(data)))

    def test_version_must_be_exact(self):
        data = xml_mutation(lambda root: root.set("version", "3.1"))
        self.assertIn("musicxml.version", codes(validate_musicxml_semantics(data)))

    def test_unsupported_top_level_element_rejected(self):
        def mutate(root):
            root.insert(1, ET.Element("identification"))
        data = xml_mutation(mutate)
        self.assertIn("musicxml.root_shape", codes(validate_musicxml_semantics(data)))

    def test_part_identity_and_name_are_frozen(self):
        def mutate(root):
            root.find("./part-list/score-part").set("id", "P2")
            root.find("./part").set("id", "P2")
            root.find("./part-list/score-part/part-name").text = "Other"
        result = codes(validate_musicxml_semantics(xml_mutation(mutate)))
        self.assertIn("musicxml.score_part_id", result)
        self.assertIn("musicxml.part_id", result)
        self.assertIn("musicxml.part_name", result)

    def test_measure_numbers_must_be_sequential_and_canonical(self):
        data = xml_mutation(lambda root: root.find("./part/measure").set("number", "02"))
        self.assertIn("musicxml.measure_number", codes(validate_musicxml_semantics(data)))

    def test_first_measure_requires_full_attributes(self):
        def mutate(root):
            measure = root.find("./part/measure")
            measure.remove(measure.find("attributes"))
        self.assertIn("musicxml.first_attributes_missing", codes(validate_musicxml_semantics(xml_mutation(mutate))))

    def test_divisions_must_be_positive_canonical_integer(self):
        def mutate(root):
            root.find("./part/measure/attributes/divisions").text = "00"
        self.assertIn("musicxml.divisions", codes(validate_musicxml_semantics(xml_mutation(mutate))))

    def test_key_signature_is_zero(self):
        def mutate(root):
            root.find("./part/measure/attributes/key/fifths").text = "1"
        self.assertIn("musicxml.key_signature", codes(validate_musicxml_semantics(xml_mutation(mutate))))

    def test_time_signature_is_v1_only(self):
        def mutate(root):
            root.find("./part/measure/attributes/time/beats").text = "6"
            root.find("./part/measure/attributes/time/beat-type").text = "8"
        self.assertIn("musicxml.time_signature", codes(validate_musicxml_semantics(xml_mutation(mutate))))

    def test_clef_is_treble_g2(self):
        def mutate(root):
            root.find("./part/measure/attributes/clef/sign").text = "F"
        self.assertIn("musicxml.clef", codes(validate_musicxml_semantics(xml_mutation(mutate))))

    def test_unsupported_measure_element_rejected(self):
        def mutate(root):
            root.find("./part/measure").append(ET.Element("direction"))
        self.assertIn("musicxml.unsupported_element", codes(validate_musicxml_semantics(xml_mutation(mutate))))


class SemanticEventTests(unittest.TestCase):
    def test_voice_and_staff_must_be_one(self):
        def mutate(root):
            note = root.find("./part/measure/note")
            note.find("voice").text = "2"
            note.find("staff").text = "2"
        result = codes(validate_musicxml_semantics(xml_mutation(mutate)))
        self.assertIn("musicxml.voice", result)
        self.assertIn("musicxml.staff", result)

    def test_duration_must_match_type_and_divisions(self):
        def mutate(root):
            root.find("./part/measure/note/duration").text = "2"
        self.assertIn("musicxml.duration_type_mismatch", codes(validate_musicxml_semantics(xml_mutation(mutate))))

    def test_pitch_step_is_restricted(self):
        def mutate(root):
            root.find("./part/measure/note/pitch/step").text = "H"
        self.assertIn("musicxml.pitch_step", codes(validate_musicxml_semantics(xml_mutation(mutate))))

    def test_explicit_zero_alter_is_noncanonical(self):
        def mutate(root):
            pitch = root.find("./part/measure/note/pitch")
            pitch.insert(1, ET.Element("alter"))
            pitch[1].text = "0"
        self.assertIn("musicxml.pitch_alter", codes(validate_musicxml_semantics(xml_mutation(mutate))))

    def test_accidental_must_match_pitch_alter(self):
        def mutate(root):
            note = root.find("./part/measure/note")
            accidental = ET.Element("accidental")
            accidental.text = "sharp"
            note.insert(len(note) - 1, accidental)
        self.assertIn("musicxml.accidental_mismatch", codes(validate_musicxml_semantics(xml_mutation(mutate))))

    def test_chord_continuation_requires_base_note(self):
        def mutate(root):
            note = root.find("./part/measure/note")
            note.insert(0, ET.Element("chord"))
        self.assertIn("musicxml.chord_without_base", codes(validate_musicxml_semantics(xml_mutation(mutate))))

    def test_chord_duplicate_pitch_is_rejected(self):
        def mutate(root):
            measure = root.find("./part/measure")
            first = measure.findall("note")[0]
            duplicate = deepcopy(first)
            duplicate.insert(0, ET.Element("chord"))
            measure.insert(list(measure).index(first) + 1, duplicate)
        self.assertIn("musicxml.chord_duplicate_pitch", codes(validate_musicxml_semantics(xml_mutation(mutate))))

    def test_chord_larger_than_four_rejected(self):
        def mutate(root):
            measure = root.find("./part/measure")
            first = measure.findall("note")[0]
            insert_at = list(measure).index(first) + 1
            for index in range(4):
                extra = deepcopy(first)
                extra.insert(0, ET.Element("chord"))
                extra.find("pitch/step").text = "CDEFG"[index + 1]
                measure.insert(insert_at + index, extra)
        self.assertIn("musicxml.chord_size", codes(validate_musicxml_semantics(xml_mutation(mutate))))

    def test_measure_underflow_detected(self):
        def mutate(root):
            measure = root.find("./part/measure")
            measure.remove(measure.findall("note")[-1])
        self.assertIn("musicxml.measure_underflow", codes(validate_musicxml_semantics(xml_mutation(mutate))))

    def test_measure_overflow_detected(self):
        def mutate(root):
            measure = root.find("./part/measure")
            extra = deepcopy(measure.findall("note")[-1])
            measure.append(extra)
        result = codes(validate_musicxml_semantics(xml_mutation(mutate)))
        self.assertTrue("musicxml.measure_overflow" in result or "musicxml.measure_duration_overflow" in result)

    def test_issue_order_is_deterministic(self):
        def mutate(root):
            note = root.find("./part/measure/note")
            note.find("voice").text = "2"
            note.find("staff").text = "2"
        data = xml_mutation(mutate)
        self.assertEqual(validate_musicxml_semantics(data), validate_musicxml_semantics(data))


class SchemaIntegrityTests(unittest.TestCase):
    def make_schema_dir(self, root: Path, main_schema: bytes | None = None):
        assets = {
            "musicxml.xsd": main_schema or b'<?xml version="1.0"?><xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"><xs:element name="score-partwise" type="xs:anyType"/></xs:schema>',
            "xlink.xsd": b"xlink",
            "xml.xsd": b"xml",
            "catalog.xml": b"catalog",
        }
        for name, payload in assets.items():
            (root / name).write_bytes(payload)
        return assets

    def hashes(self, assets):
        return {name: hashlib.sha256(payload).hexdigest() for name, payload in assets.items()}

    def test_integrity_checker_accepts_exact_pinned_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = self.make_schema_dir(root)
            with patch.dict(MUSICXML_SCHEMA_SHA256, self.hashes(assets), clear=True):
                self.assertTrue(verify_musicxml_schema_assets(root).is_valid)

    def test_integrity_checker_fails_closed_on_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = self.make_schema_dir(root)
            expected = self.hashes(assets)
            (root / "xlink.xsd").write_bytes(b"tampered")
            with patch.dict(MUSICXML_SCHEMA_SHA256, expected, clear=True):
                self.assertIn("musicxml.schema_hash_mismatch", codes(verify_musicxml_schema_assets(root)))

    def test_integrity_checker_rejects_unpinned_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = self.make_schema_dir(root)
            expected = self.hashes(assets)
            expected["musicxml.xsd"] = "__PENDING__"
            with patch.dict(MUSICXML_SCHEMA_SHA256, expected, clear=True):
                self.assertIn("musicxml.schema_hash_unpinned", codes(verify_musicxml_schema_assets(root)))

    def test_xsd_adapter_validates_offline_local_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = self.make_schema_dir(root)
            with patch.dict(MUSICXML_SCHEMA_SHA256, self.hashes(assets), clear=True):
                valid = b'<?xml version="1.0"?><score-partwise version="4.0"/>'
                invalid = b'<?xml version="1.0"?><other/>'
                self.assertTrue(validate_musicxml_xsd(valid, schema_dir=root).is_valid)
                self.assertIn("musicxml.xsd_invalid", codes(validate_musicxml_xsd(invalid, schema_dir=root)))

    def test_xsd_adapter_refuses_unknown_external_import(self):
        schema = b'''<?xml version="1.0"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:import namespace="urn:evil" schemaLocation="https://example.invalid/evil.xsd"/>
  <xs:element name="score-partwise" type="xs:anyType"/>
</xs:schema>'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = self.make_schema_dir(root, schema)
            with patch.dict(MUSICXML_SCHEMA_SHA256, self.hashes(assets), clear=True):
                result = validate_musicxml_xsd(b"<score-partwise/>", schema_dir=root)
                self.assertIn("musicxml.xsd_parse_error", codes(result))


if __name__ == "__main__":
    unittest.main()
