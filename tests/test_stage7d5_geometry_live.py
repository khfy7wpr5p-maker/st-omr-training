from __future__ import annotations

from pathlib import Path
import unittest

from st_omr_training.degradation import (
    DegradationConfig,
    DegradationSource,
    degrade_page,
    sample_degradation_config,
)
from st_omr_training.renderer import render_musicxml_svg
from st_omr_training.stage7d5_geometry import (
    extract_staff_structure_geometry,
    map_page_geometry_to_final_png,
    render_musicxml_geometry_svg,
)


GOLDEN = Path(__file__).parent / "golden"
GOLDEN_NAMES = (
    "accidentals.musicxml",
    "basic_2_4.musicxml",
    "basic_4_4.musicxml",
    "chords_2_3_4.musicxml",
    "rest_3_4.musicxml",
    "time_change.musicxml",
)


def _source(*, family_id: str, render_result, page) -> DegradationSource:
    return DegradationSource(
        family_id=family_id,
        page_number=page.page_number,
        source_musicxml_sha256=render_result.source_musicxml_sha256,
        renderer_config_fingerprint=render_result.config_fingerprint,
        svg=page.svg,
        svg_sha256=page.sha256,
    )


def _geometry_source(*, family_id: str, geometry_result, page) -> DegradationSource:
    return DegradationSource(
        family_id=family_id,
        page_number=page.page_number,
        source_musicxml_sha256=geometry_result.source_musicxml_sha256,
        renderer_config_fingerprint=geometry_result.base_renderer_config_fingerprint,
        svg=page.svg,
        svg_sha256=page.sha256,
    )


class Stage7D5LiveGeometryTests(unittest.TestCase):
    def test_all_goldens_preserve_visible_raster_and_extract_staff_structure(self) -> None:
        for name in GOLDEN_NAMES:
            with self.subTest(name=name):
                musicxml = (GOLDEN / name).read_bytes()
                base = render_musicxml_svg(musicxml)
                geometry = render_musicxml_geometry_svg(musicxml)
                self.assertEqual(len(base.pages), len(geometry.pages))
                self.assertEqual(
                    base.config_fingerprint,
                    geometry.base_renderer_config_fingerprint,
                )
                self.assertEqual(
                    base.source_musicxml_sha256,
                    geometry.source_musicxml_sha256,
                )

                for base_page, geometry_page in zip(base.pages, geometry.pages, strict=True):
                    config = DegradationConfig(seed=0, raster_width=1400)
                    base_clean = degrade_page(
                        _source(
                            family_id=f"d5-base-{name}",
                            render_result=base,
                            page=base_page,
                        ),
                        config,
                    )
                    geometry_clean = degrade_page(
                        _geometry_source(
                            family_id=f"d5-geometry-{name}",
                            geometry_result=geometry,
                            page=geometry_page,
                        ),
                        config,
                    )
                    self.assertEqual(
                        (base_clean.clean_width, base_clean.clean_height),
                        (geometry_clean.clean_width, geometry_clean.clean_height),
                    )
                    self.assertEqual(
                        base_clean.clean_raster_sha256,
                        geometry_clean.clean_raster_sha256,
                        "bbox instrumentation changed visible raster pixels",
                    )

                pages = extract_staff_structure_geometry(geometry, musicxml)
                self.assertEqual(len(pages), len(geometry.pages))
                self.assertGreaterEqual(sum(len(page.systems) for page in pages), 1)
                self.assertEqual(
                    sum(len(page.systems) for page in pages),
                    sum(len(page.staff_instances) for page in pages),
                )
                for page in pages:
                    for staff in page.staff_instances:
                        self.assertEqual(len(staff.five_staff_lines), 5)
                        self.assertGreater(staff.staff_spacing, 0.0)
                    for measure in page.measures:
                        self.assertIn(measure.meter_class, {"2/4", "3/4", "4/4"})
                        self.assertGreater(measure.barline_segment.length, 0.0)

    def test_clean_light_medium_geometry_maps_inside_real_derivative(self) -> None:
        musicxml = (GOLDEN / "time_change.musicxml").read_bytes()
        base = render_musicxml_svg(musicxml)
        geometry = render_musicxml_geometry_svg(musicxml)
        svg_pages = extract_staff_structure_geometry(geometry, musicxml)
        self.assertEqual(len(base.pages), len(svg_pages))

        for profile, seed in (("clean", 4101), ("light", 4102), ("medium", 4103)):
            with self.subTest(profile=profile):
                config = sample_degradation_config(seed, profile, raster_width=1400)
                for base_page, svg_geometry in zip(base.pages, svg_pages, strict=True):
                    degraded = degrade_page(
                        _source(
                            family_id=f"d5-map-{profile}",
                            render_result=base,
                            page=base_page,
                        ),
                        config,
                    )
                    mapped = map_page_geometry_to_final_png(svg_geometry, degraded)
                    self.assertEqual(mapped.coordinate_space, "final_png_pixels")
                    self.assertIsNotNone(mapped.geometry_transform_fingerprint)
                    for staff in mapped.staff_instances:
                        for line in staff.five_staff_lines:
                            for point in (line.start, line.end):
                                self.assertGreaterEqual(point.x, -1e-6)
                                self.assertGreaterEqual(point.y, -1e-6)
                                self.assertLessEqual(point.x, degraded.width + 1e-6)
                                self.assertLessEqual(point.y, degraded.height + 1e-6)
                    for measure in mapped.measures:
                        for point in (
                            measure.barline_segment.start,
                            measure.barline_segment.end,
                        ):
                            self.assertGreaterEqual(point.x, -1e-6)
                            self.assertGreaterEqual(point.y, -1e-6)
                            self.assertLessEqual(point.x, degraded.width + 1e-6)
                            self.assertLessEqual(point.y, degraded.height + 1e-6)


if __name__ == "__main__":
    unittest.main()
