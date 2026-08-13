from __future__ import annotations

import unittest

from st_omr_training.stage8_auxiliary_admission_gate import (
    Stage8AuxiliaryAdmissionGateError,
    adapt_guarded_primus_v1_to_musicxml,
    auxiliary_admission_gate_policy_fingerprint,
)
from st_omr_training.stage8_auxiliary_triage_gate import (
    inspect_guarded_primus_auxiliary_package,
)
from st_omr_training.stage8_pilot_preparation import Stage8PilotPreparationError


def _note(*, accid: str | None = None, ges: str | None = None) -> str:
    attrs = ['dur="4"', 'oct="4"', 'pname="c"']
    if accid is not None:
        attrs.append(f'accid="{accid}"')
    if ges is not None:
        attrs.append(f'accid.ges="{ges}"')
    return f"<note {' '.join(attrs)}/>"


def _mei(events: str) -> bytes:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="4.0.0">
  <music><body><mdiv><score>
    <scoreDef key.sig="0" meter.count="4" meter.unit="4">
      <staffGrp><staffDef n="1" lines="5" clef.shape="G" clef.line="2"/></staffGrp>
    </scoreDef>
    <section><measure n="1"><staff n="1"><layer n="1">{events}</layer></staff></measure></section>
  </score></mdiv></body></music>
</mei>'''.encode("utf-8")


class Stage8AuxiliaryAdmissionGuardTests(unittest.TestCase):
    def test_gestural_natural_without_visible_glyph_is_allowed_when_state_unchanged(self) -> None:
        mei = _mei(_note(ges="n") + _note() + _note() + _note())
        semantic = (
            b"clef-G2 keySignature-CM timeSignature-4/4 "
            b"note-C4_quarter note-C4_quarter note-C4_quarter note-C4_quarter"
        )
        musicxml, evidence = adapt_guarded_primus_v1_to_musicxml(
            mei_bytes=mei,
            semantic_bytes=semantic,
        )
        self.assertTrue(musicxml.startswith(b"<?xml"))
        self.assertEqual(
            evidence.gate_policy_fingerprint,
            auxiliary_admission_gate_policy_fingerprint(),
        )

    def test_visible_sharp_repeat_and_visible_natural_transition_are_corroborated(self) -> None:
        mei = _mei(
            _note(accid="s", ges="s")
            + _note()
            + _note(accid="n", ges="n")
            + _note()
        )
        semantic = (
            b"clef-G2 keySignature-CM timeSignature-4/4 "
            b"note-C#4_quarter note-C#4_quarter note-C4_quarter note-C4_quarter"
        )
        musicxml, evidence = adapt_guarded_primus_v1_to_musicxml(
            mei_bytes=mei,
            semantic_bytes=semantic,
        )
        self.assertTrue(musicxml)
        self.assertEqual(evidence.note_count, 4)

    def test_sounding_transition_without_visible_glyph_fails_closed(self) -> None:
        mei = _mei(_note(ges="s") + _note() + _note() + _note())
        semantic = (
            b"clef-G2 keySignature-CM timeSignature-4/4 "
            b"note-C#4_quarter note-C#4_quarter note-C#4_quarter note-C#4_quarter"
        )
        with self.assertRaises(Stage8AuxiliaryAdmissionGateError):
            adapt_guarded_primus_v1_to_musicxml(
                mei_bytes=mei,
                semantic_bytes=semantic,
            )

    def test_redundant_visible_natural_fails_closed(self) -> None:
        mei = _mei(_note(accid="n", ges="n") + _note() + _note() + _note())
        semantic = (
            b"clef-G2 keySignature-CM timeSignature-4/4 "
            b"note-C4_quarter note-C4_quarter note-C4_quarter note-C4_quarter"
        )
        with self.assertRaises(Stage8AuxiliaryAdmissionGateError):
            adapt_guarded_primus_v1_to_musicxml(
                mei_bytes=mei,
                semantic_bytes=semantic,
            )

    def test_visible_and_gestural_disagreement_fails_closed(self) -> None:
        mei = _mei(_note(accid="s", ges="f") + _note() + _note() + _note())
        semantic = (
            b"clef-G2 keySignature-CM timeSignature-4/4 "
            b"note-C#4_quarter note-C#4_quarter note-C#4_quarter note-C#4_quarter"
        )
        with self.assertRaises(Stage8AuxiliaryAdmissionGateError):
            adapt_guarded_primus_v1_to_musicxml(
                mei_bytes=mei,
                semantic_bytes=semantic,
            )

    def test_declarative_xml_surface_fails_before_adapter_and_triage(self) -> None:
        safe = _mei(_note() + _note() + _note() + _note())
        declaration = b"<!" + b"DOC" + b"TYPE mei>"
        guarded = safe.replace(b"<mei xmlns=", declaration + b"<mei xmlns=", 1)
        semantic = (
            b"clef-G2 keySignature-CM timeSignature-4/4 "
            b"note-C4_quarter note-C4_quarter note-C4_quarter note-C4_quarter"
        )
        with self.assertRaises(Stage8AuxiliaryAdmissionGateError):
            adapt_guarded_primus_v1_to_musicxml(
                mei_bytes=guarded,
                semantic_bytes=semantic,
            )
        with self.assertRaises(Stage8PilotPreparationError):
            inspect_guarded_primus_auxiliary_package(
                mei_bytes=guarded,
                semantic_bytes=semantic,
                agnostic_bytes=b"clef.G-L2 note.quarter-L3",
            )

    def test_guarded_triage_preserves_successful_v1_triage(self) -> None:
        inspection = inspect_guarded_primus_auxiliary_package(
            mei_bytes=_mei(_note() + _note() + _note() + _note()),
            semantic_bytes=(
                b"clef-G2 keySignature-CM timeSignature-4/4 "
                b"note-C4_quarter note-C4_quarter note-C4_quarter note-C4_quarter"
            ),
            agnostic_bytes=b"clef.G-L2 note.quarter-L3 barline",
        )
        self.assertTrue(inspection.v1_eligible)


if __name__ == "__main__":
    unittest.main()
