from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET

from st_omr_training.renderer import RendererConfig, _load_verovio_runtime


def _grand_staff_musicxml() -> str:
    return """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<score-partwise version=\"4.0\">
  <part-list>
    <score-part id=\"P1\"><part-name>Piano</part-name></score-part>
  </part-list>
  <part id=\"P1\">
    <measure number=\"1\">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>2</staves>
        <part-symbol top-staff=\"1\" bottom-staff=\"2\">brace</part-symbol>
        <clef number=\"1\"><sign>G</sign><line>2</line></clef>
        <clef number=\"2\"><sign>F</sign><line>4</line></clef>
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
      <barline location=\"right\"><bar-style>light-heavy</bar-style></barline>
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


class SystemGeometryMultistaffFeasibilityV1Tests(unittest.TestCase):
    def test_pinned_verovio_exposes_two_staff_groups_inside_one_system(self) -> None:
        verovio, package_version = _load_verovio_runtime()
        toolkit = verovio.toolkit()
        self.assertTrue(str(toolkit.getVersion()).startswith(package_version))
        self.assertIsNot(toolkit.setInputFrom("xml"), False)

        options = dict(RendererConfig().verovio_options())
        options.update({"svgBoundingBoxes": True, "svgContentBoundingBoxes": True})
        self.assertIsNot(toolkit.setOptions(options), False)
        self.assertIsNot(toolkit.loadData(_grand_staff_musicxml()), False)
        self.assertEqual(toolkit.getPageCount(), 1)

        svg = toolkit.renderToSVG(1, True)
        root = ET.fromstring(svg)
        systems = _visible_groups(root, "system")
        self.assertEqual(len(systems), 1)

        system = systems[0]
        measures = _visible_groups(system, "measure")
        self.assertEqual(len(measures), 1)
        staffs = _visible_groups(measures[0], "staff")
        self.assertEqual(len(staffs), 2)

        barlines = [
            e
            for e in measures[0].iter()
            if e.tag.rsplit("}", 1)[-1] == "g"
            and ({"barLine", "barLineAttr"} & _tokens(e))
            and "bounding-box" not in _tokens(e)
            and "content-bounding-box" not in _tokens(e)
        ]
        class_tokens = sorted({token for e in system.iter() for token in _tokens(e)})
        grouping_tokens = [
            token
            for token in class_tokens
            if any(piece in token.lower() for piece in ("brace", "bracket", "group", "grpsym"))
        ]

        report = {
            "version": "system-geometry-multistaff-feasibility-v1",
            "verovio_package_version": package_version,
            "system_count": len(systems),
            "measure_count_first_system": len(measures),
            "staff_count_first_measure": len(staffs),
            "barline_group_count_first_measure": len(barlines),
            "grouping_class_tokens": grouping_tokens,
            "claim_boundary": "PINNED_VEROVIO_SVG_TOPOLOGY_ONLY_NO_ST_V1_ADMISSION",
        }
        print("SYSTEM_GEOMETRY_MULTISTAFF_FEASIBILITY=" + json.dumps(report, sort_keys=True))

        self.assertGreaterEqual(len(barlines), 1)


if __name__ == "__main__":
    unittest.main()
