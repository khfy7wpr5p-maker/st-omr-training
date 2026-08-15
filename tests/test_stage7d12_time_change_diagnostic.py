from __future__ import annotations

from pathlib import Path
import unittest

from st_omr_training import stage7d12_symbol_geometry as d12
from st_omr_training.stage7d5_geometry import render_musicxml_geometry_svg


GOLDEN = Path(__file__).parent / "golden" / "time_change.musicxml"


def _raw_bbox_rect(group, coordinate_root, parent_map):
    object_id = group.attrib.get("id", "")
    rows = []
    for prefix, class_name in (
        ("bbox-", "bounding-box"),
        ("cbbox-", "content-bounding-box"),
    ):
        expected_id = prefix + object_id
        matches = [
            element
            for element in group.iter()
            if d12._local(element) == "g"
            and element.attrib.get("id") == expected_id
            and class_name in d12._d5._class_tokens(element)
        ]
        if len(matches) != 1:
            continue
        rects = [
            child for child in matches[0] if d12._local(child) == "rect"
        ]
        if len(rects) != 1:
            continue
        rect = rects[0]
        rows.append(
            {
                "kind": class_name,
                "raw": {
                    key: rect.attrib.get(key, "")
                    for key in ("x", "y", "width", "height")
                },
                "mapped": repr(
                    d12._d5._box_from_rect(rect, coordinate_root, parent_map)
                ),
            }
        )
    return rows


class Stage7D12TimeChangeDiagnostic(unittest.TestCase):
    def test_report_time_change_bbox_quantization(self) -> None:
        musicxml = GOLDEN.read_bytes()
        render = render_musicxml_geometry_svg(musicxml)
        canonical = d12._canonical_measures(musicxml)
        renderer = d12._renderer_measures(render)
        report = []
        for canonical_measure, renderer_measure in zip(canonical, renderer, strict=True):
            atoms = d12._renderer_atoms(renderer_measure)
            rows = []
            for canonical_atom, renderer_group in zip(
                canonical_measure.atoms, atoms, strict=True
            ):
                note_box = d12._bbox_for_visible_group(
                    renderer_group,
                    renderer_measure.coordinate_root,
                    renderer_measure.parent_map,
                )
                if canonical_atom.kind == "rest":
                    symbol_box = note_box
                    kind = "rest"
                else:
                    heads = d12._visible_descendants(renderer_group, "notehead")
                    symbol_box = d12._notehead_bbox(
                        heads[0],
                        expected_glyph_code=canonical_atom.renderer_glyph_code,
                        measure=renderer_measure,
                    )
                    kind = "notehead"
                rows.append(
                    {
                        "event": canonical_atom.canonical_event_id,
                        "kind": kind,
                        "renderer_id": renderer_group.attrib.get("id", ""),
                        "symbol_bbox": repr(symbol_box),
                        "note_bbox": repr(note_box),
                        "symbol_minus_note": {
                            "left": symbol_box.x_min - note_box.x_min,
                            "top": symbol_box.y_min - note_box.y_min,
                            "right": symbol_box.x_max - note_box.x_max,
                            "bottom": symbol_box.y_max - note_box.y_max,
                        },
                        "symbol_minus_measure": {
                            "left": symbol_box.x_min - renderer_measure.measure_bbox.x_min,
                            "top": symbol_box.y_min - renderer_measure.measure_bbox.y_min,
                            "right": symbol_box.x_max - renderer_measure.measure_bbox.x_max,
                            "bottom": symbol_box.y_max - renderer_measure.measure_bbox.y_max,
                        },
                        "note_raw_boxes": _raw_bbox_rect(
                            renderer_group,
                            renderer_measure.coordinate_root,
                            renderer_measure.parent_map,
                        ),
                    }
                )
            report.append(
                {
                    "measure": canonical_measure.number,
                    "renderer_measure_id": renderer_measure.measure_group.attrib.get(
                        "id", ""
                    ),
                    "measure_bbox": repr(renderer_measure.measure_bbox),
                    "measure_raw_boxes": _raw_bbox_rect(
                        renderer_measure.measure_group,
                        renderer_measure.coordinate_root,
                        renderer_measure.parent_map,
                    ),
                    "symbols": rows,
                }
            )
        self.fail("D12_BBOX_QUANTIZATION=" + repr(report))


if __name__ == "__main__":
    unittest.main()
