from __future__ import annotations

from hashlib import sha256
import unittest
import xml.etree.ElementTree as ET

from st_omr_training.renderer import RendererConfig, _load_verovio_runtime
from st_omr_training.system_geometry_evidence_v1 import (
    StaffSystemRelation,
    SystemGeometryEvidenceError,
    SystemGeometryEvidencePageV1,
    SystemTopologyEvidenceV1,
    system_geometry_evidence_runtime_connection_allowed,
)


def _grand_staff_musicxml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>Piano</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>2</staves>
        <part-symbol top-staff="1" bottom-staff="2">brace</part-symbol>
        <clef number="1"><sign>G</sign><line>2</line></clef>
        <clef number="2"><sign>F</sign><line>4</line></clef>
      </attributes>
      <note>
        <pitch><step>C</step><octave>5</octave></pitch>
        <duration>4</duration><voice>1</voice><type>whole</type><staff>1</staff>
      </note>
      <backup><duration>4</duration></backup>
      <note>
        <pitch><step>C</step><octave>3</octave></pitch>
        <duration>4</duration><voice>2</voice><type>whole</type><staff>2</staff>
      </note>
      <barline location="right"><bar-style>light-heavy</bar-style></barline>
    </measure>
  </part>
</score-partwise>
"""


def _two_system_single_staff_musicxml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>Fixture</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note>
        <pitch><step>C</step><octave>5</octave></pitch>
        <duration>4</duration><voice>1</voice><type>whole</type><staff>1</staff>
      </note>
      <barline location="right"><bar-style>regular</bar-style></barline>
    </measure>
    <measure number="2">
      <print new-system="yes"/>
      <note>
        <pitch><step>D</step><octave>5</octave></pitch>
        <duration>4</duration><voice>1</voice><type>whole</type><staff>1</staff>
      </note>
      <barline location="right"><bar-style>light-heavy</bar-style></barline>
    </measure>
  </part>
</score-partwise>
"""


def _tokens(element: ET.Element) -> set[str]:
    return set(element.attrib.get("class", "").split())


def _visible_groups(root: ET.Element, class_name: str) -> list[ET.Element]:
    found: list[ET.Element] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "g":
            continue
        tokens = _tokens(element)
        if class_name not in tokens:
            continue
        if "bounding-box" in tokens or "content-bounding-box" in tokens:
            continue
        found.append(element)
    return found


def _render_fixture(xml_text: str, *, breaks: str = "auto") -> str:
    verovio, package_version = _load_verovio_runtime()
    toolkit = verovio.toolkit()
    if not str(toolkit.getVersion()).startswith(package_version):
        raise AssertionError("pinned Verovio runtime mismatch")
    if toolkit.setInputFrom("xml") is False:
        raise AssertionError("Verovio rejected fixture input mode")
    options = dict(RendererConfig(breaks=breaks).verovio_options())
    options.update({"svgBoundingBoxes": True, "svgContentBoundingBoxes": True})
    if toolkit.setOptions(options) is False:
        raise AssertionError("Verovio rejected fixture options")
    if toolkit.loadData(xml_text) is False:
        raise AssertionError("Verovio rejected fixture MusicXML")
    if toolkit.getPageCount() != 1:
        raise AssertionError("fixture must remain on one page")
    return toolkit.renderToSVG(1, True)


def _extract_fixture_page(page_id: str, svg: str) -> SystemGeometryEvidencePageV1:
    root = ET.fromstring(svg)
    systems = _visible_groups(root, "system")
    evidence_systems: list[SystemTopologyEvidenceV1] = []

    for system_index, system in enumerate(systems, start=1):
        system_id = system.attrib.get("id") or f"fixture-system-{system_index}"
        measures = _visible_groups(system, "measure")
        measure_ids = tuple(
            measure.attrib.get("id") or f"{system_id}-measure-{index}"
            for index, measure in enumerate(measures, start=1)
        )

        staff_ids: list[str] = []
        for measure_index, measure in enumerate(measures, start=1):
            staffs = _visible_groups(measure, "staff")
            for staff_index, staff in enumerate(staffs, start=1):
                staff_ids.append(
                    staff.attrib.get("id")
                    or f"{system_id}-m{measure_index}-staff-{staff_index}"
                )

        class_tokens = sorted({token for item in system.iter() for token in _tokens(item)})
        grouping_tokens = tuple(
            token
            for token in class_tokens
            if any(
                piece in token.lower()
                for piece in ("brace", "bracket", "group", "grpsym")
            )
        )
        barlines = [
            item
            for item in system.iter()
            if item.tag.rsplit("}", 1)[-1] == "g"
            and ({"barLine", "barLineAttr"} & _tokens(item))
            and "bounding-box" not in _tokens(item)
            and "content-bounding-box" not in _tokens(item)
        ]

        evidence_systems.append(
            SystemTopologyEvidenceV1(
                system_id=system_id,
                staff_instance_ids=tuple(staff_ids),
                measure_ids=measure_ids,
                grouping_tokens=grouping_tokens,
                barline_group_count=len(barlines),
            )
        )

    return SystemGeometryEvidencePageV1(
        page_id=page_id,
        source_svg_sha256=sha256(svg.encode("utf-8")).hexdigest(),
        systems=tuple(evidence_systems),
    )


class SystemGeometryEvidenceV1Tests(unittest.TestCase):
    def test_grand_staff_fixture_produces_positive_same_system_relation(self) -> None:
        page = _extract_fixture_page("grand-staff", _render_fixture(_grand_staff_musicxml()))
        self.assertEqual(len(page.systems), 1)
        self.assertEqual(len(page.systems[0].staff_instance_ids), 2)
        self.assertIn("grpSym", page.systems[0].grouping_tokens)
        self.assertGreaterEqual(page.systems[0].barline_group_count, 1)

        relations = page.staff_pair_relations()
        self.assertEqual(len(relations), 1)
        self.assertIs(relations[0].relation, StaffSystemRelation.SAME_SYSTEM)
        self.assertEqual(relations[0].system_a_id, relations[0].system_b_id)

    def test_forced_system_break_produces_negative_different_system_relation(self) -> None:
        page = _extract_fixture_page(
            "two-systems",
            _render_fixture(_two_system_single_staff_musicxml(), breaks="encoded"),
        )
        self.assertEqual(len(page.systems), 2)
        self.assertEqual([len(system.staff_instance_ids) for system in page.systems], [1, 1])

        relations = page.staff_pair_relations()
        self.assertEqual(len(relations), 1)
        self.assertIs(relations[0].relation, StaffSystemRelation.DIFFERENT_SYSTEM)
        self.assertNotEqual(relations[0].system_a_id, relations[0].system_b_id)

    def test_fingerprint_is_canonical_and_repeats_10_of_10(self) -> None:
        page = _extract_fixture_page("grand-staff", _render_fixture(_grand_staff_musicxml()))
        fingerprints = {page.fingerprint() for _ in range(10)}
        self.assertEqual(len(fingerprints), 1)
        fingerprint = next(iter(fingerprints))
        self.assertEqual(len(fingerprint), 64)

    def test_staff_cannot_be_owned_by_two_systems(self) -> None:
        source_sha = "0" * 64
        with self.assertRaisesRegex(SystemGeometryEvidenceError, "more than one system"):
            SystemGeometryEvidencePageV1(
                page_id="bad",
                source_svg_sha256=source_sha,
                systems=(
                    SystemTopologyEvidenceV1("s1", ("staff-x",), ("m1",)),
                    SystemTopologyEvidenceV1("s2", ("staff-x",), ("m2",)),
                ),
            )

    def test_fixture_pilot_has_no_runtime_authorization(self) -> None:
        self.assertFalse(system_geometry_evidence_runtime_connection_allowed())


if __name__ == "__main__":
    unittest.main()
