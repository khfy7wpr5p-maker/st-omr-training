from __future__ import annotations

import unittest

from st_omr_training.renderer import RendererConfig, _load_verovio_runtime
from st_omr_training.system_geometry_evidence_extractor_v1 import (
    SYSTEM_GEOMETRY_EVIDENCE_EXTRACTOR_VERSION,
    SystemGeometryEvidenceExtractorError,
    extract_system_geometry_evidence_v1,
    system_geometry_evidence_extractor_runtime_connection_allowed,
)
from st_omr_training.system_geometry_evidence_v1 import StaffSystemRelation


def _grand_staff_musicxml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions><key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>2</staves>
        <part-symbol top-staff="1" bottom-staff="2">brace</part-symbol>
        <clef number="1"><sign>G</sign><line>2</line></clef>
        <clef number="2"><sign>F</sign><line>4</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type><staff>1</staff></note>
      <backup><duration>4</duration></backup>
      <note><pitch><step>C</step><octave>3</octave></pitch><duration>4</duration><voice>2</voice><type>whole</type><staff>2</staff></note>
      <barline location="right"><bar-style>light-heavy</bar-style></barline>
    </measure>
  </part>
</score-partwise>"""


def _two_system_musicxml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Fixture</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions><key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type><staff>1</staff></note>
      <barline location="right"><bar-style>regular</bar-style></barline>
    </measure>
    <measure number="2">
      <print new-system="yes"/>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type><staff>1</staff></note>
      <barline location="right"><bar-style>light-heavy</bar-style></barline>
    </measure>
  </part>
</score-partwise>"""


def _render(xml_text: str, *, encoded_breaks: bool = False) -> str:
    verovio, package_version = _load_verovio_runtime()
    toolkit = verovio.toolkit()
    if not str(toolkit.getVersion()).startswith(package_version):
        raise AssertionError("pinned Verovio runtime mismatch")
    if toolkit.setInputFrom("xml") is False:
        raise AssertionError("fixture input mode rejected")
    config = RendererConfig(breaks="encoded" if encoded_breaks else "auto")
    options = dict(config.verovio_options())
    options.update({"svgBoundingBoxes": True, "svgContentBoundingBoxes": True})
    if toolkit.setOptions(options) is False:
        raise AssertionError("fixture options rejected")
    if toolkit.loadData(xml_text) is False:
        raise AssertionError("fixture MusicXML rejected")
    if toolkit.getPageCount() != 1:
        raise AssertionError("fixture must remain on one page")
    return toolkit.renderToSVG(1, True)


class SystemGeometryEvidenceExtractorV1Tests(unittest.TestCase):
    def test_grand_staff_extracts_plural_membership_and_raw_observations(self) -> None:
        report = extract_system_geometry_evidence_v1(
            page_id="grand-staff", svg=_render(_grand_staff_musicxml())
        )
        self.assertEqual(len(report.systems), 1)
        system = report.systems[0]
        self.assertEqual(len(system.staff_instance_ids), 2)
        self.assertEqual(len(system.measures), 1)
        self.assertEqual(len(system.measures[0].staff_svg_ids), 2)
        self.assertGreaterEqual(len(system.measures[0].barline_svg_ids), 1)
        self.assertIn("grpSym", system.grouping_tokens)
        relations = report.evidence_page.staff_pair_relations()
        self.assertEqual(len(relations), 1)
        self.assertIs(relations[0].relation, StaffSystemRelation.SAME_SYSTEM)

    def test_encoded_break_extracts_negative_different_system_pair(self) -> None:
        report = extract_system_geometry_evidence_v1(
            page_id="two-systems",
            svg=_render(_two_system_musicxml(), encoded_breaks=True),
        )
        self.assertEqual(len(report.systems), 2)
        self.assertEqual([len(x.staff_instance_ids) for x in report.systems], [1, 1])
        relations = report.evidence_page.staff_pair_relations()
        self.assertEqual(len(relations), 1)
        self.assertIs(relations[0].relation, StaffSystemRelation.DIFFERENT_SYSTEM)

    def test_extractor_fingerprint_is_deterministic_10_of_10(self) -> None:
        svg = _render(_grand_staff_musicxml())
        fingerprints = {
            extract_system_geometry_evidence_v1(page_id="grand-staff", svg=svg).fingerprint()
            for _ in range(10)
        }
        self.assertEqual(len(fingerprints), 1)
        self.assertEqual(len(next(iter(fingerprints))), 64)

    def test_malformed_or_topology_free_svg_fails_closed(self) -> None:
        with self.assertRaises(SystemGeometryEvidenceExtractorError):
            extract_system_geometry_evidence_v1(page_id="bad", svg="<svg>")
        with self.assertRaisesRegex(SystemGeometryEvidenceExtractorError, "no visible"):
            extract_system_geometry_evidence_v1(
                page_id="empty", svg='<svg xmlns="http://www.w3.org/2000/svg"/>'
            )

    def test_contract_version_and_runtime_boundary_are_explicit(self) -> None:
        self.assertEqual(
            SYSTEM_GEOMETRY_EVIDENCE_EXTRACTOR_VERSION,
            "system-geometry-evidence-extractor-v1",
        )
        self.assertFalse(system_geometry_evidence_extractor_runtime_connection_allowed())


if __name__ == "__main__":
    unittest.main()
