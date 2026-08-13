import unittest
from pathlib import Path

from st_omr_training.degradation import (
    CAIROSVG_PINNED_VERSION,
    PILLOW_PINNED_VERSION,
    degrade_render_result_page,
    sample_degradation_config,
)
from st_omr_training.generator import GeneratorConfig, generate_score
from st_omr_training.musicxml_writer import write_musicxml
from st_omr_training.renderer import render_musicxml_svg


GOLDEN_DIR = Path(__file__).with_name("golden")


class RealStage4PipelineTests(unittest.TestCase):
    def test_all_stage2_goldens_cross_real_renderer_and_clean_raster_boundary(self):
        for path in sorted(GOLDEN_DIR.glob("*.musicxml")):
            with self.subTest(path=path.name):
                rendered = render_musicxml_svg(path.read_bytes())
                for page in rendered.pages:
                    result = degrade_render_result_page(
                        rendered,
                        family_id=f"golden-{path.stem}",
                        page_number=page.page_number,
                        config=sample_degradation_config(0, "clean", raster_width=512),
                    )
                    self.assertEqual(result.cairosvg_version, CAIROSVG_PINNED_VERSION)
                    self.assertEqual(result.pillow_version, PILLOW_PINNED_VERSION)
                    self.assertEqual(result.source_svg_sha256, page.sha256)
                    self.assertEqual(result.mode, "L")
                    self.assertTrue(result.png.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_real_pipeline_degradation_is_byte_deterministic(self):
        data = write_musicxml(generate_score(GeneratorConfig(measure_count=6), 20260813))
        rendered = render_musicxml_svg(data)
        config = sample_degradation_config(4242, "medium", raster_width=512)
        first = degrade_render_result_page(
            rendered, family_id="generated-determinism", page_number=1, config=config
        )
        second = degrade_render_result_page(
            rendered, family_id="generated-determinism", page_number=1, config=config
        )
        self.assertEqual(first.png, second.png)
        self.assertEqual(first.png_sha256, second.png_sha256)
        self.assertEqual(first.derivative_id, second.derivative_id)

    def test_generated_scores_cross_renderer_and_medium_degradation(self):
        hashes = set()
        for seed in range(12):
            with self.subTest(seed=seed):
                data = write_musicxml(generate_score(GeneratorConfig(measure_count=5), seed))
                rendered = render_musicxml_svg(data)
                result = degrade_render_result_page(
                    rendered,
                    family_id=f"generated-{seed}",
                    page_number=1,
                    config=sample_degradation_config(seed, "medium", raster_width=512),
                )
                self.assertEqual(result.source_musicxml_sha256, rendered.source_musicxml_sha256)
                self.assertEqual(result.renderer_config_fingerprint, rendered.config_fingerprint)
                self.assertGreater(result.width, 0)
                self.assertGreater(result.height, 0)
                hashes.add(result.png_sha256)
        self.assertEqual(len(hashes), 12)


if __name__ == "__main__":
    unittest.main()
