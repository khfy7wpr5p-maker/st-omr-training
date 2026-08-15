from __future__ import annotations

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from st_omr_training import _stage7d5_geometry_v1 as d5
from st_omr_training.stage7d5_geometry import render_musicxml_geometry_svg


GOLDEN = Path(__file__).parent / "golden"


class Stage7D12RendererShapeDiagnostic(unittest.TestCase):
    def test_report_note_descendant_shape(self) -> None:
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
                descendants = []
                for element in note.iter():
                    if element is note or element.tag.rsplit("}", 1)[-1] != "g":
                        continue
                    tokens = sorted(d5._class_tokens(element))
                    if "bounding-box" in tokens or "content-bounding-box" in tokens:
                        continue
                    descendants.append(
                        {
                            "id": element.attrib.get("id", ""),
                            "class": tokens,
                        }
                    )
                rows.append(
                    {
                        "note_id": note.attrib.get("id", ""),
                        "descendants": descendants,
                    }
                )
            report[name] = rows
        self.fail("D12_RENDERER_SHAPE=" + repr(report))


if __name__ == "__main__":
    unittest.main()
