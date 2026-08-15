from __future__ import annotations

from pathlib import Path
import unittest

from st_omr_training import stage7d12_symbol_geometry as d12
from st_omr_training.stage7d5_geometry import render_musicxml_geometry_svg


GOLDEN = Path(__file__).parent / "golden" / "time_change.musicxml"


class Stage7D12TimeChangeDiagnostic(unittest.TestCase):
    def test_report_time_change_symbol_containment(self) -> None:
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
                if canonical_atom.kind == "rest":
                    box = d12._bbox_for_visible_group(
                        renderer_group,
                        renderer_measure.coordinate_root,
                        renderer_measure.parent_map,
                    )
                    kind = "rest"
                else:
                    heads = d12._visible_descendants(renderer_group, "notehead")
                    box = d12._notehead_bbox(
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
                        "symbol_bbox": (
                            box.x_min,
                            box.y_min,
                            box.x_max,
                            box.y_max,
                        ),
                        "inside_measure": d12._box_inside_box(
                            box, renderer_measure.measure_bbox
                        ),
                    }
                )
            report.append(
                {
                    "measure": canonical_measure.number,
                    "renderer_measure_id": renderer_measure.measure_group.attrib.get(
                        "id", ""
                    ),
                    "measure_bbox": (
                        renderer_measure.measure_bbox.x_min,
                        renderer_measure.measure_bbox.y_min,
                        renderer_measure.measure_bbox.x_max,
                        renderer_measure.measure_bbox.y_max,
                    ),
                    "symbols": rows,
                }
            )
        self.fail("D12_TIME_CHANGE_CONTAINMENT=" + repr(report))


if __name__ == "__main__":
    unittest.main()
