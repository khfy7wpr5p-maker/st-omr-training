from __future__ import annotations

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from st_omr_training import _stage7d5_geometry_v1 as d5
from st_omr_training.stage7d5_geometry import render_musicxml_geometry_svg


GOLDEN = Path(__file__).parent / "golden"
GOLDEN_NAMES = (
    "accidentals.musicxml",
    "basic_2_4.musicxml",
    "basic_4_4.musicxml",
    "chords_2_3_4.musicxml",
    "rest_3_4.musicxml",
    "time_change.musicxml",
)
XLINK = "{http://www.w3.org/1999/xlink}href"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _href(element: ET.Element) -> str:
    return element.attrib.get("href", element.attrib.get(XLINK, ""))


def _compact(element: ET.Element) -> dict[str, object]:
    return {
        "tag": _local(element.tag),
        "id": element.attrib.get("id", ""),
        "class": sorted(d5._class_tokens(element)),
        "href": _href(element),
        "transform": element.attrib.get("transform", ""),
    }


class Stage7D12RendererShapeDiagnostic(unittest.TestCase):
    def test_inventory_symbol_renderer_shape(self) -> None:
        notehead_inventory = {}
        notehead_examples = {}
        accidental_rows = {}
        for name in GOLDEN_NAMES:
            musicxml = (GOLDEN / name).read_bytes()
            render = render_musicxml_geometry_svg(musicxml)
            file_rows = []
            for page in render.pages:
                root = ET.fromstring(page.svg)
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
                for note in notes:
                    heads = [
                        element
                        for element in note.iter()
                        if element is not note
                        and d5._is_visible_object_group(element, "notehead")
                    ]
                    for head in heads:
                        uses = [x for x in head.iter() if _local(x.tag) == "use"]
                        for use in uses:
                            href = _href(use)
                            ref = id_map.get(href[1:]) if href.startswith("#") else None
                            if ref is None:
                                continue
                            paths = [
                                child
                                for child in ref.iter()
                                if _local(child.tag) == "path"
                            ]
                            definition = tuple(
                                (path.attrib.get("transform", ""), path.attrib.get("d", ""))
                                for path in paths
                            )
                            notehead_inventory[href] = definition
                            notehead_examples.setdefault(
                                href,
                                {
                                    "source": name,
                                    "note_id": note.attrib.get("id", ""),
                                    "use_transform": use.attrib.get("transform", ""),
                                },
                            )

                    accids = [
                        element
                        for element in note.iter()
                        if element is not note
                        and d5._is_visible_object_group(element, "accid")
                    ]
                    if accids:
                        accid_report = []
                        for accid in accids:
                            descendants = []
                            for element in accid.iter():
                                if element is accid:
                                    continue
                                tokens = d5._class_tokens(element)
                                if "bounding-box" in tokens or "content-bounding-box" in tokens:
                                    continue
                                descendants.append(_compact(element))
                            accid_report.append(
                                {
                                    "group": _compact(accid),
                                    "descendants": descendants,
                                }
                            )
                        file_rows.append(
                            {
                                "note_id": note.attrib.get("id", ""),
                                "accids": accid_report,
                            }
                        )
            accidental_rows[name] = file_rows
        self.fail(
            "D12_SYMBOL_SHAPE="
            + repr(
                {
                    "notehead_inventory": notehead_inventory,
                    "notehead_examples": notehead_examples,
                    "accidental_rows": accidental_rows,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
