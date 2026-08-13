from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from st_omr_training.musicxml_validator import validate_musicxml
from st_omr_training.stage8_primus_adapter import (
    PrimusV1AdapterError,
    adapt_primus_v1_to_musicxml,
    primus_v1_adapter_policy_fingerprint,
)


def _mei(*, meter: str = "4", unit: str = "4", measures: tuple[str, ...]) -> bytes:
    body = "".join(
        f'<measure n="{index}"><staff n="1"><layer n="1">{events}</layer></staff></measure>'
        for index, events in enumerate(measures, start=1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="4.0.0">
  <music><body><mdiv><score>
    <scoreDef key.sig="0" meter.count="{meter}" meter.unit="{unit}">
      <staffGrp><staffDef n="1" lines="5" clef.shape="G" clef.line="2"/></staffGrp>
    </scoreDef>
    <section>{body}</section>
  </score></mdiv></body></music>
</mei>'''.encode("utf-8")


def _qnote(step: str, octave: int, *, accidental: str | None = None, ges: str | None = None) -> str:
    attrs = [f'dur="4"', f'oct="{octave}"', f'pname="{step.lower()}"']
    if accidental is not None:
        attrs.append(f'accid="{accidental}"')
    if ges is not None:
        attrs.append(f'accid.ges="{ges}"')
    return f"<note {' '.join(attrs)}/>"


class Stage8PrimusV1AdapterTests(unittest.TestCase):
    def test_supported_pair_is_deterministic_and_roundtrip_valid(self) -> None:
        mei = _mei(
            measures=(
                "".join(
                    (
                        _qnote("C", 4, ges="n"),
                        _qnote("C", 4, accidental="s", ges="s"),
                        _qnote("C", 4),
                        _qnote("C", 4, accidental="n", ges="n"),
                    )
                ),
            )
        )
        semantic = (
            b"clef-G2 keySignature-CM timeSignature-4/4 "
            b"note-C4_quarter note-C#4_quarter note-C#4_quarter note-C4_quarter barline"
        )
        first_xml, first_evidence = adapt_primus_v1_to_musicxml(mei_bytes=mei, semantic_bytes=semantic)
        second_xml, second_evidence = adapt_primus_v1_to_musicxml(mei_bytes=mei, semantic_bytes=semantic)
        self.assertEqual(first_xml, second_xml)
        self.assertEqual(first_evidence, second_evidence)
        self.assertTrue(validate_musicxml(first_xml).is_valid)
        self.assertEqual(first_evidence.measure_count, 1)
        self.assertEqual(first_evidence.note_count, 4)
        self.assertEqual(first_evidence.rest_count, 0)
        self.assertEqual(first_evidence.policy_fingerprint, primus_v1_adapter_policy_fingerprint())
        root = ET.fromstring(first_xml)
        accidentals = [node.text for node in root.findall("./part/measure/note/accidental")]
        self.assertEqual(accidentals, ["sharp", "natural"])

    def test_supported_rest_measure_is_accepted_and_overflow_fails(self) -> None:
        mei_ok = _mei(meter="2", measures=('<rest dur="4"/>' + _qnote("D", 4),))
        semantic_ok = b"clef-G2 keySignature-CM timeSignature-2/4 rest-quarter note-D4_quarter"
        musicxml, evidence = adapt_primus_v1_to_musicxml(mei_bytes=mei_ok, semantic_bytes=semantic_ok)
        self.assertTrue(validate_musicxml(musicxml).is_valid)
        self.assertEqual((evidence.note_count, evidence.rest_count), (1, 1))

        semantic_overflow = (
            b"clef-G2 keySignature-CM timeSignature-2/4 "
            b"rest-quarter note-D4_quarter rest-quarter"
        )
        with self.assertRaises(PrimusV1AdapterError):
            adapt_primus_v1_to_musicxml(mei_bytes=mei_ok, semantic_bytes=semantic_overflow)

    def test_two_measures_and_trailing_barline_preserve_measure_identity(self) -> None:
        mei = _mei(
            meter="2",
            measures=(
                _qnote("C", 4) + _qnote("D", 4),
                _qnote("E", 4) + _qnote("F", 4),
            ),
        )
        semantic = (
            b"clef-G2 keySignature-CM timeSignature-2/4 "
            b"note-C4_quarter note-D4_quarter barline "
            b"note-E4_quarter note-F4_quarter barline"
        )
        musicxml, evidence = adapt_primus_v1_to_musicxml(mei_bytes=mei, semantic_bytes=semantic)
        self.assertEqual(evidence.measure_count, 2)
        root = ET.fromstring(musicxml)
        self.assertEqual([m.attrib["number"] for m in root.findall("./part/measure")], ["1", "2"])

    def test_mei_and_semantic_event_mismatch_fails_closed(self) -> None:
        semantic = (
            b"clef-G2 keySignature-CM timeSignature-4/4 "
            b"note-C4_quarter note-D4_quarter note-E4_quarter note-F4_quarter"
        )
        wrong_pitch = _mei(
            measures=(_qnote("C", 4) + _qnote("D", 4) + _qnote("E", 4) + _qnote("G", 4),)
        )
        with self.assertRaises(PrimusV1AdapterError):
            adapt_primus_v1_to_musicxml(mei_bytes=wrong_pitch, semantic_bytes=semantic)

        wrong_duration = _mei(
            measures=('<note dur="2" oct="4" pname="c"/>' + _qnote("D", 4) + _qnote("E", 4),)
        )
        with self.assertRaises(PrimusV1AdapterError):
            adapt_primus_v1_to_musicxml(mei_bytes=wrong_duration, semantic_bytes=semantic)

    def test_header_mismatch_is_rejected(self) -> None:
        mei = _mei(meter="3", measures=(_qnote("C", 4) + _qnote("D", 4) + _qnote("E", 4),))
        semantic = (
            b"clef-G2 keySignature-CM timeSignature-4/4 "
            b"note-C4_quarter note-D4_quarter note-E4_quarter note-F4_quarter"
        )
        with self.assertRaises(PrimusV1AdapterError):
            adapt_primus_v1_to_musicxml(mei_bytes=mei, semantic_bytes=semantic)
        with self.assertRaises(PrimusV1AdapterError):
            adapt_primus_v1_to_musicxml(
                mei_bytes=_mei(measures=(_qnote("C", 4) * 4,)),
                semantic_bytes=semantic.replace(b"clef-G2", b"clef-C1"),
            )

    def test_deferred_or_unknown_semantic_tokens_are_rejected(self) -> None:
        mei = _mei(measures=(_qnote("C", 4) * 4,))
        for token in (b"note-C4_quarter.", b"note-C4_sixteenth", b"multirest-4", b"fermata"):
            with self.subTest(token=token):
                semantic = b"clef-G2 keySignature-CM timeSignature-4/4 " + token
                with self.assertRaises(PrimusV1AdapterError):
                    adapt_primus_v1_to_musicxml(mei_bytes=mei, semantic_bytes=semantic)

    def test_deferred_mei_structure_and_empty_measures_are_rejected(self) -> None:
        semantic = (
            b"clef-G2 keySignature-CM timeSignature-4/4 "
            b"note-C4_quarter note-D4_quarter note-E4_quarter note-F4_quarter"
        )
        mei_beam = _mei(
            measures=(
                '<beam><note dur="4" oct="4" pname="c"/></beam>'
                + _qnote("D", 4) + _qnote("E", 4) + _qnote("F", 4),
            )
        )
        with self.assertRaises(PrimusV1AdapterError):
            adapt_primus_v1_to_musicxml(mei_bytes=mei_beam, semantic_bytes=semantic)
        with self.assertRaises(PrimusV1AdapterError):
            adapt_primus_v1_to_musicxml(mei_bytes=_mei(measures=("",)), semantic_bytes=semantic)

    def test_semantic_measure_must_be_exact_not_padded_or_truncated(self) -> None:
        mei = _mei(measures=(_qnote("C", 4) * 4,))
        underfilled = (
            b"clef-G2 keySignature-CM timeSignature-4/4 "
            b"note-C4_quarter note-C4_quarter note-C4_quarter"
        )
        with self.assertRaises(PrimusV1AdapterError):
            adapt_primus_v1_to_musicxml(mei_bytes=mei, semantic_bytes=underfilled)
        empty_bar = (
            b"clef-G2 keySignature-CM timeSignature-4/4 "
            b"barline note-C4_quarter note-C4_quarter note-C4_quarter note-C4_quarter"
        )
        with self.assertRaises(PrimusV1AdapterError):
            adapt_primus_v1_to_musicxml(mei_bytes=mei, semantic_bytes=empty_bar)


if __name__ == "__main__":
    unittest.main()
