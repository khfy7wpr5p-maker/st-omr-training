from __future__ import annotations

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from st_omr_training import _stage7d5_geometry_v1 as d5
from st_omr_training.stage7d5_geometry import render_musicxml_geometry_svg


GOLDEN = Path(__file__).parent / "golden"
XLINK = "{http://www.w3.org/1999/xlink}href"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _href(element: ET.Element) -> str:
    return element.attrib.get("href", element.attrib.get(XLINK, ""))


def _compact_attrs(element: ET.Element) -> dict[str, str]:
    keep = {"id", "class", "href", "x", "y", "transform", "viewBox", "d"}
    result = {}
    for key, value in element.attrib.items():
        local = key.rsplit("}", 1)[-1]
        if local in keep:
            result[local] = value
    if _href(element):
        result["href"] = _href(element)
    return result


class Stage7D12RendererShapeDiagnostic(unittest.TestCase):
    def test_report_notehead_glyph_geometry(self) -> None:
        report = {}
        for name in ("basic_2_4.musicxml", "chords_2_3_4.musicxml"):
            musicxml = (GOLDEN / name).read_bytes()
            render = render_musicxml_geometry_svg(musicxml)
            root = ET.fromstring(render.pages[0].svg)
            coordinate_root, _ = d5._coordinate_root(root)
            id_map = {
                element.attrib["id"]: element
                for element in root.iter()
                if element.attrib.get("id")
            }
            notes = [
                element
                for element in coordinate_root.iter()
                if d5._is_visible_object_group(element, "note")
            ]
            rows = []
            for note in notes[:3]:
                heads = [
                    element
                    for element in note.iter()
                    if element is not note
                    and d5._is_visible_object_group(element, "notehead")
                ]
                head_rows = []
                for head in heads:
                    descendants = []
                    refs = []
                    for element in head.iter():
                        if element is head:
                            continue
                        descendants.append((_local(element.tag), _compact_attrs(element)))
                        href = _href(element)
                        if href.startswith("#") and href[1:] in id_map:
                            ref = id_map[href[1:]]
                            refs.append(
                                {
                                    "ref_tag": _local(ref.tag),
                                    "ref_attrs": _compact_attrs(ref),
                                    "ref_children": [
                                        (_local(child.tag), _compact_attrs(child))
                                        for child in list(ref)[:4]
                                    ],
                                }
                            )
                    head_rows.append(
                        {
                            "head_attrs": _compact_attrs(head),
                            "descendants": descendants,
                            "refs": refs,
                        }
                    )
                rows.append(
                    {
                        "note_id": note.attrib.get("id", ""),
                        "notehead": head_rows,
                    }
                )
            report[name] = rows
        self.fail("D12_NOTEHEAD_SHAPE=" + repr(report))


if __name__ == "__main__":
    unittest.main()
