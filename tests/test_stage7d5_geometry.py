from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from PIL import Image

from st_omr_training.musicxml_writer import musicxml_sha256
from st_omr_training.renderer import (
    RENDERER_ADAPTER_VERSION,
    RENDERER_NAME,
    RendererConfig,
    renderer_config_fingerprint,
)
from st_omr_training.stage7d5_geometry import (
    D5_D4_LABEL_CORRECTION,
    D5_STRUCTURE_LABELS,
    GeometryRenderedPage,
    GeometryRenderResult,
    Stage7D5GeometryError,
    extract_staff_structure_geometry,
    geometry_instrumentation_fingerprint,
    geometry_report_payload,
    map_page_geometry_to_final_png,
)


GOLDEN = Path(__file__).parent / "golden"


def _svg_one_measure(*, line_count: int = 5, include_measure_bbox: bool = True) -> bytes:
    line_paths = "\n".join(
        f'<path d="M 100 {100 + index * 20} L 900 {100 + index * 20}" style="stroke-width: 2;"/>'
        for index in range(line_count)
    )
    measure_bbox = (
        '<g class="measure content-bounding-box" id="cbbox-measure-svg-1">'
        '<rect x="90" y="80" width="820" height="130" fill="transparent" stroke-width="0"/>'
        "</g>"
        if include_measure_bbox
        else ""
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 400">
      <g class="system" id="system-svg-1">
        <g class="system content-bounding-box" id="cbbox-system-svg-1">
          <rect x="80" y="70" width="840" height="150" fill="transparent" stroke-width="0"/>
        </g>
        <g class="measure" id="measure-svg-1">
          {measure_bbox}
          <g class="staff" id="staff-svg-1">
            {line_paths}
            <g class="layer" id="layer-svg-1"/>
          </g>
          <g class="clef" id="clef-svg-1">
            <g class="clef bounding-box" id="bbox-clef-svg-1">
              <rect x="110" y="90" width="30" height="90" fill="transparent" stroke-width="0"/>
            </g>
          </g>
          <g class="meterSig" id="meter-svg-1">
            <g class="meterSig bounding-box" id="bbox-meter-svg-1">
              <rect x="150" y="100" width="40" height="80" fill="transparent" stroke-width="0"/>
            </g>
          </g>
          <g class="barLineAttr" id="barline-svg-1">
            <path d="M 900 100 L 900 180" style="stroke-width: 3;"/>
          </g>
        </g>
      </g>
    </svg>"""
    return svg.encode("utf-8")


def _render_result(musicxml: bytes, svg: bytes) -> GeometryRenderResult:
    config = RendererConfig()
    return GeometryRenderResult(
        source_musicxml_sha256=musicxml_sha256(musicxml),
        renderer_name=RENDERER_NAME,
        renderer_package_version="6.2.1",
        renderer_runtime_version="6.2.1",
        renderer_adapter_version=RENDERER_ADAPTER_VERSION,
        base_renderer_config_fingerprint=renderer_config_fingerprint(config),
        geometry_instrumentation_fingerprint=geometry_instrumentation_fingerprint(config),
        pages=(GeometryRenderedPage(1, svg, sha256(svg).hexdigest()),),
    )


def _fake_degraded(page, *, rotation_mdeg: int):
    clean_size = (1000, 400)
    if rotation_mdeg:
        final_size = Image.new("L", clean_size, 255).rotate(
            rotation_mdeg / 1000.0,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=255,
        ).size
    else:
        final_size = clean_size
    return SimpleNamespace(
        page_number=1,
        source_musicxml_sha256=page.source_musicxml_sha256,
        renderer_config_fingerprint=page.base_renderer_config_fingerprint,
        degradation_config_fingerprint="a" * 64,
        config=SimpleNamespace(rotation_mdeg=rotation_mdeg),
        clean_width=clean_size[0],
        clean_height=clean_size[1],
        width=final_size[0],
        height=final_size[1],
    )


class Stage7D5ContractCorrectionTests(unittest.TestCase):
    def test_rotated_final_png_uses_barline_segment_not_scalar_x(self) -> None:
        self.assertNotIn("barline_x", D5_STRUCTURE_LABELS)
        self.assertIn("barline_segment", D5_STRUCTURE_LABELS)
        self.assertEqual(D5_D4_LABEL_CORRECTION["superseded_label"], "barline_x")
        self.assertEqual(D5_D4_LABEL_CORRECTION["replacement_label"], "barline_segment")


class Stage7D5ExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.musicxml = (GOLDEN / "basic_4_4.musicxml").read_bytes()

    def test_extracts_staff_structure_and_canonical_meter(self) -> None:
        svg = _svg_one_measure()
        pages = extract_staff_structure_geometry(_render_result(self.musicxml, svg), self.musicxml)
        self.assertEqual(len(pages), 1)
        page = pages[0]
        self.assertEqual(page.coordinate_space, "pinned_verovio_svg")
        self.assertEqual(len(page.systems), 1)
        self.assertEqual(len(page.staff_instances), 1)
        self.assertEqual(len(page.measures), 1)

        staff = page.staff_instances[0]
        self.assertEqual(len(staff.five_staff_lines), 5)
        self.assertEqual(staff.staff_spacing, 20.0)
        self.assertEqual(staff.five_staff_lines[0].start.x, 100.0)
        self.assertEqual(staff.five_staff_lines[-1].start.y, 180.0)

        measure = page.measures[0]
        self.assertEqual(measure.measure_number, 1)
        self.assertEqual(measure.meter_class, "4/4")
        self.assertEqual(measure.barline_segment.start.x, 900.0)
        self.assertIsNotNone(measure.clef_g2_bbox)
        self.assertIsNotNone(measure.meter_bbox)

    def test_missing_required_renderer_bbox_fails_closed(self) -> None:
        svg = _svg_one_measure(include_measure_bbox=False)
        with self.assertRaises(Stage7D5GeometryError):
            extract_staff_structure_geometry(_render_result(self.musicxml, svg), self.musicxml)

    def test_nonfive_line_staff_fails_closed(self) -> None:
        svg = _svg_one_measure(line_count=4)
        with self.assertRaises(Stage7D5GeometryError):
            extract_staff_structure_geometry(_render_result(self.musicxml, svg), self.musicxml)

    def test_geometry_svg_hash_tamper_fails_closed(self) -> None:
        svg = _svg_one_measure()
        result = _render_result(self.musicxml, svg)
        tampered = GeometryRenderResult(
            source_musicxml_sha256=result.source_musicxml_sha256,
            renderer_name=result.renderer_name,
            renderer_package_version=result.renderer_package_version,
            renderer_runtime_version=result.renderer_runtime_version,
            renderer_adapter_version=result.renderer_adapter_version,
            base_renderer_config_fingerprint=result.base_renderer_config_fingerprint,
            geometry_instrumentation_fingerprint=result.geometry_instrumentation_fingerprint,
            pages=(GeometryRenderedPage(1, svg + b" ", result.pages[0].sha256),),
        )
        with self.assertRaises(Stage7D5GeometryError):
            extract_staff_structure_geometry(tampered, self.musicxml)

    def test_canonical_measure_count_mismatch_fails_closed(self) -> None:
        two_measure_musicxml = (GOLDEN / "time_change.musicxml").read_bytes()
        svg = _svg_one_measure()
        with self.assertRaisesRegex(Stage7D5GeometryError, "measure count mismatch"):
            extract_staff_structure_geometry(
                _render_result(two_measure_musicxml, svg), two_measure_musicxml
            )


class Stage7D5TransformTests(unittest.TestCase):
    def setUp(self) -> None:
        musicxml = (GOLDEN / "basic_4_4.musicxml").read_bytes()
        svg = _svg_one_measure()
        self.page = extract_staff_structure_geometry(
            _render_result(musicxml, svg), musicxml
        )[0]

    def test_clean_mapping_is_exact_uniform_viewbox_scale(self) -> None:
        mapped = map_page_geometry_to_final_png(
            self.page, _fake_degraded(self.page, rotation_mdeg=0)
        )
        self.assertEqual(mapped.coordinate_space, "final_png_pixels")
        self.assertEqual(mapped.staff_instances[0].staff_spacing, 20.0)
        self.assertEqual(mapped.measures[0].barline_segment.start.x, 900.0)
        self.assertIsNotNone(mapped.geometry_transform_fingerprint)
        self.assertEqual(len(mapped.geometry_transform_fingerprint), 64)

    def test_rotation_replays_pillow_expand_geometry_and_slants_barline(self) -> None:
        degraded = _fake_degraded(self.page, rotation_mdeg=1000)
        mapped = map_page_geometry_to_final_png(self.page, degraded)
        line = mapped.measures[0].barline_segment
        self.assertNotAlmostEqual(line.start.x, line.end.x)
        self.assertAlmostEqual(line.length, 80.0, places=6)
        for staff_line in mapped.staff_instances[0].five_staff_lines:
            for point in (staff_line.start, staff_line.end):
                self.assertGreaterEqual(point.x, 0.0)
                self.assertGreaterEqual(point.y, 0.0)
                self.assertLessEqual(point.x, degraded.width)
                self.assertLessEqual(point.y, degraded.height)

    def test_provenance_mismatch_fails_closed(self) -> None:
        degraded = _fake_degraded(self.page, rotation_mdeg=0)
        degraded.source_musicxml_sha256 = "b" * 64
        with self.assertRaisesRegex(Stage7D5GeometryError, "MusicXML provenance"):
            map_page_geometry_to_final_png(self.page, degraded)

    def test_report_payload_is_json_serializable_and_records_correction(self) -> None:
        mapped = map_page_geometry_to_final_png(
            self.page, _fake_degraded(self.page, rotation_mdeg=0)
        )
        payload = geometry_report_payload((mapped,))
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.assertIn('"barline_segment"', encoded)
        self.assertIn('"final_png_pixels"', encoded)


if __name__ == "__main__":
    unittest.main()
