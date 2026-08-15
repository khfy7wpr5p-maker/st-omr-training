from __future__ import annotations

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from st_omr_training import _stage7d5_geometry_v1 as d5
from st_omr_training.stage7d5_geometry import render_musicxml_geometry_svg


GOLDEN = Path(__file__).parent / "golden"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _payload(group: ET.Element):
    rows = []
    for element in group.iter():
        if element is group:
            continue
        tokens = sorted(d5._class_tokens(element)) if _local(element.tag) == "g" else []
        if "bounding-box" in tokens or "content-bounding-box" in tokens:
            continue
        attrs = {
            key.rsplit("}", 1)[-1]: value
            for key, value in element.attrib.items()
            if key.rsplit("}", 1)[-1] in {"id", "class", "href", "d", "x", "y"}
        }
        rows.append((_local(element.tag), attrs))
    return rows


class Stage7D12RendererShapeDiagnostic(unittest.TestCase):
    def test_report_accid_payload_shape(self) -> None:
        report = {}
        for name in ("basic_2_4.musicxml", "accidentals.musicxml"):
            musicxml = (GOLDEN / name).read_bytes()
            render = render_musicxml_geometry_svg(musicxml)
            page = render.pages[0]
            root = ET.fromstring(page.svg)
            coordinate_root, _ = d5._coordinate_root(root)
            notes = [
                element
                for element in coordinate_root.iter()
                if d5._is_visible_object_group(element, "note")
            ]
            rows = []
            for note in notes:
                accids = [
                    element
                    for element in note.iter()
                    if element is not note
                    and d5._is_visible_object_group(element, "accid")
                ]
                noteheads = [
                    element
                    for element in note.iter()
                    if element is not note
                    and d5._is_visible_object_group(element, "notehead")
                ]
                rows.append(
                    {
                        "note_id": note.attrib.get("id", ""),
                        "notehead_ids": [x.attrib.get("id", "") for x in noteheads],
                        "accids": [
                            {
                                "id": accid.attrib.get("id", ""),
                                "payload": _payload(accid),
                            }
                            for accid in accids
                        ],
                    }
                )
            report[name] = rows
        self.fail("D12_ACCID_SHAPE=" + repr(report))


if __name__ == "__main__":
    unittest.main()
