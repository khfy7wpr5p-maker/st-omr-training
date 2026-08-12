import unittest
from pathlib import Path

from st_omr_training.generator import GeneratorConfig, generate_score
from st_omr_training.musicxml_writer import write_musicxml
from st_omr_training.renderer import VEROVIO_PINNED_VERSION, render_musicxml_svg


GOLDEN_DIR = Path(__file__).with_name("golden")


class RealVerovioRendererTests(unittest.TestCase):
    def test_all_stage2_golden_musicxml_render_with_exact_pinned_runtime(self):
        for path in sorted(GOLDEN_DIR.glob("*.musicxml")):
            with self.subTest(path=path.name):
                result = render_musicxml_svg(path.read_bytes())
                self.assertEqual(result.renderer_package_version, VEROVIO_PINNED_VERSION)
                self.assertTrue(result.renderer_runtime_version.startswith(VEROVIO_PINNED_VERSION))
                self.assertGreaterEqual(len(result.pages), 1)
                self.assertTrue(all(page.svg.startswith(b"<?xml") or b"<svg" in page.svg for page in result.pages))

    def test_real_renderer_is_byte_deterministic_for_same_input(self):
        data = write_musicxml(generate_score(GeneratorConfig(measure_count=12), 20260813))
        first = render_musicxml_svg(data)
        second = render_musicxml_svg(data)
        self.assertEqual(first.config_fingerprint, second.config_fingerprint)
        self.assertEqual(
            tuple(page.sha256 for page in first.pages),
            tuple(page.sha256 for page in second.pages),
        )
        self.assertEqual(
            tuple(page.svg for page in first.pages),
            tuple(page.svg for page in second.pages),
        )

    def test_generated_scores_render_with_real_runtime(self):
        for seed in range(50):
            with self.subTest(seed=seed):
                data = write_musicxml(generate_score(GeneratorConfig(measure_count=8), seed))
                result = render_musicxml_svg(data)
                self.assertGreaterEqual(len(result.pages), 1)
                self.assertTrue(all(len(page.sha256) == 64 for page in result.pages))


if __name__ == "__main__":
    unittest.main()
