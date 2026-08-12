import unittest
from copy import deepcopy
from unittest.mock import patch

from st_omr_training.generator import GeneratorConfig, generate_score
from st_omr_training.musicxml_writer import write_musicxml
from st_omr_training.renderer import (
    MAX_RENDER_PAGES,
    RENDERER_ADAPTER_VERSION,
    VEROVIO_PINNED_VERSION,
    RenderExecutionError,
    RenderInputError,
    RendererConfig,
    RendererUnavailableError,
    render_musicxml_svg,
    renderer_config_fingerprint,
)


class FakeToolkit:
    def __init__(self):
        self.input_from = None
        self.options = None
        self.loaded = None
        self.page_count = 2
        self.runtime_version = VEROVIO_PINNED_VERSION
        self.reject_input_mode = False
        self.reject_options = False
        self.reject_load = False
        self.svg_by_page = {
            1: '<svg xmlns="http://www.w3.org/2000/svg"><defs><g id="x"/></defs><use href="#x"/></svg>',
            2: '<svg xmlns="http://www.w3.org/2000/svg"><g id="y"/></svg>',
        }

    def getVersion(self):
        return self.runtime_version

    def setInputFrom(self, value):
        self.input_from = value
        return not self.reject_input_mode

    def setOptions(self, options):
        self.options = deepcopy(options)
        return not self.reject_options

    def loadData(self, data):
        self.loaded = data
        return not self.reject_load

    def getPageCount(self):
        return self.page_count

    def renderToSVG(self, page_number, xml_declaration):
        return self.svg_by_page[page_number]


class FakeVerovio:
    def __init__(self, toolkit=None):
        self.instance = toolkit or FakeToolkit()

    def toolkit(self):
        return self.instance


def valid_xml(seed=1, measures=2):
    return write_musicxml(generate_score(GeneratorConfig(measure_count=measures), seed))


class RendererConfigTests(unittest.TestCase):
    def test_defaults_are_frozen_v1_values(self):
        config = RendererConfig()
        self.assertEqual(config.page_height, 2970)
        self.assertEqual(config.page_width, 2100)
        self.assertEqual(config.scale, 100)
        self.assertEqual(config.breaks, "auto")
        self.assertEqual(config.font, "Leipzig")

    def test_options_make_determinism_controls_explicit(self):
        options = RendererConfig().verovio_options()
        self.assertEqual(options["font"], "Leipzig")
        self.assertEqual(options["fontFallback"], "Leipzig")
        self.assertEqual(options["smuflTextFont"], "embedded")
        self.assertIs(options["xmlIdChecksum"], True)
        self.assertIs(options["svgFormatRaw"], True)
        self.assertIs(options["svgViewBox"], True)
        self.assertIs(options["svgRemoveXlink"], True)

    def test_invalid_bool_integer_rejected(self):
        with self.assertRaises(TypeError):
            RendererConfig(scale=True)

    def test_invalid_ranges_rejected(self):
        with self.assertRaises(ValueError):
            RendererConfig(page_width=99)
        with self.assertRaises(ValueError):
            RendererConfig(page_margin_left=501)
        with self.assertRaises(ValueError):
            RendererConfig(scale=1001)

    def test_invalid_break_mode_rejected(self):
        with self.assertRaises(ValueError):
            RendererConfig(breaks="random")

    def test_v1_font_is_pinned(self):
        with self.assertRaises(ValueError):
            RendererConfig(font="Bravura")

    def test_config_fingerprint_is_stable_and_sensitive(self):
        base = RendererConfig()
        self.assertEqual(renderer_config_fingerprint(base), renderer_config_fingerprint(base))
        self.assertNotEqual(
            renderer_config_fingerprint(base),
            renderer_config_fingerprint(RendererConfig(page_width=2200)),
        )
        self.assertEqual(len(renderer_config_fingerprint(base)), 64)

    def test_fingerprint_requires_config(self):
        with self.assertRaises(TypeError):
            renderer_config_fingerprint(object())


class RendererBoundaryTests(unittest.TestCase):
    def test_invalid_musicxml_rejected_before_renderer_import(self):
        with patch("st_omr_training.renderer._load_verovio_runtime") as loader:
            with self.assertRaises(RenderInputError) as raised:
                render_musicxml_svg(b"<not-musicxml/>")
            loader.assert_not_called()
            self.assertFalse(raised.exception.validation.is_valid)

    def test_non_config_rejected(self):
        with self.assertRaises(TypeError):
            with patch(
                "st_omr_training.renderer._load_verovio_runtime",
                return_value=(FakeVerovio(), VEROVIO_PINNED_VERSION),
            ):
                render_musicxml_svg(valid_xml(), object())

    def test_runtime_version_mismatch_fails_closed(self):
        toolkit = FakeToolkit()
        toolkit.runtime_version = "6.2.0"
        with patch(
            "st_omr_training.renderer._load_verovio_runtime",
            return_value=(FakeVerovio(toolkit), VEROVIO_PINNED_VERSION),
        ):
            with self.assertRaises(RendererUnavailableError):
                render_musicxml_svg(valid_xml())

    def test_input_mode_rejection_fails_closed(self):
        toolkit = FakeToolkit()
        toolkit.reject_input_mode = True
        with patch(
            "st_omr_training.renderer._load_verovio_runtime",
            return_value=(FakeVerovio(toolkit), VEROVIO_PINNED_VERSION),
        ):
            with self.assertRaises(RenderExecutionError):
                render_musicxml_svg(valid_xml())

    def test_option_rejection_fails_closed(self):
        toolkit = FakeToolkit()
        toolkit.reject_options = True
        with patch(
            "st_omr_training.renderer._load_verovio_runtime",
            return_value=(FakeVerovio(toolkit), VEROVIO_PINNED_VERSION),
        ):
            with self.assertRaises(RenderExecutionError):
                render_musicxml_svg(valid_xml())

    def test_load_rejection_fails_closed(self):
        toolkit = FakeToolkit()
        toolkit.reject_load = True
        with patch(
            "st_omr_training.renderer._load_verovio_runtime",
            return_value=(FakeVerovio(toolkit), VEROVIO_PINNED_VERSION),
        ):
            with self.assertRaises(RenderExecutionError):
                render_musicxml_svg(valid_xml())

    def test_page_count_zero_rejected(self):
        toolkit = FakeToolkit()
        toolkit.page_count = 0
        with patch(
            "st_omr_training.renderer._load_verovio_runtime",
            return_value=(FakeVerovio(toolkit), VEROVIO_PINNED_VERSION),
        ):
            with self.assertRaises(RenderExecutionError):
                render_musicxml_svg(valid_xml())

    def test_page_count_limit_enforced(self):
        toolkit = FakeToolkit()
        toolkit.page_count = MAX_RENDER_PAGES + 1
        with patch(
            "st_omr_training.renderer._load_verovio_runtime",
            return_value=(FakeVerovio(toolkit), VEROVIO_PINNED_VERSION),
        ):
            with self.assertRaises(RenderExecutionError):
                render_musicxml_svg(valid_xml())

    def test_non_integer_page_count_rejected(self):
        toolkit = FakeToolkit()
        toolkit.page_count = True
        with patch(
            "st_omr_training.renderer._load_verovio_runtime",
            return_value=(FakeVerovio(toolkit), VEROVIO_PINNED_VERSION),
        ):
            with self.assertRaises(RenderExecutionError):
                render_musicxml_svg(valid_xml())


class RendererOutputTests(unittest.TestCase):
    def render_with(self, toolkit=None, xml=None, config=None):
        toolkit = toolkit or FakeToolkit()
        with patch(
            "st_omr_training.renderer._load_verovio_runtime",
            return_value=(FakeVerovio(toolkit), VEROVIO_PINNED_VERSION),
        ):
            return render_musicxml_svg(xml or valid_xml(), config)

    def test_success_records_renderer_and_source_provenance(self):
        data = valid_xml()
        result = self.render_with(xml=data)
        self.assertEqual(result.renderer_name, "verovio")
        self.assertEqual(result.renderer_package_version, VEROVIO_PINNED_VERSION)
        self.assertEqual(result.renderer_runtime_version, VEROVIO_PINNED_VERSION)
        self.assertEqual(result.adapter_version, RENDERER_ADAPTER_VERSION)
        self.assertEqual(len(result.source_musicxml_sha256), 64)
        self.assertEqual(len(result.config_fingerprint), 64)
        self.assertEqual([p.page_number for p in result.pages], [1, 2])
        self.assertTrue(all(len(p.sha256) == 64 for p in result.pages))

    def test_explicit_musicxml_mode_and_options_are_applied(self):
        toolkit = FakeToolkit()
        config = RendererConfig(page_width=2200, scale=90)
        self.render_with(toolkit=toolkit, config=config)
        self.assertEqual(toolkit.input_from, "xml")
        self.assertEqual(toolkit.options["pageWidth"], 2200)
        self.assertEqual(toolkit.options["scale"], 90)
        self.assertEqual(toolkit.options["xmlIdChecksum"], True)

    def test_same_input_runtime_and_config_produce_same_result(self):
        data = valid_xml(seed=7, measures=4)
        first = self.render_with(xml=data)
        second = self.render_with(xml=data)
        self.assertEqual(first, second)

    def test_svg_script_is_rejected(self):
        toolkit = FakeToolkit()
        toolkit.page_count = 1
        toolkit.svg_by_page[1] = '<svg xmlns="http://www.w3.org/2000/svg"><script>bad()</script></svg>'
        with self.assertRaises(RenderExecutionError):
            self.render_with(toolkit=toolkit)

    def test_external_svg_reference_is_rejected(self):
        toolkit = FakeToolkit()
        toolkit.page_count = 1
        toolkit.svg_by_page[1] = '<svg xmlns="http://www.w3.org/2000/svg"><use href="https://example.invalid/x"/></svg>'
        with self.assertRaises(RenderExecutionError):
            self.render_with(toolkit=toolkit)

    def test_internal_svg_reference_is_allowed(self):
        toolkit = FakeToolkit()
        toolkit.page_count = 1
        toolkit.svg_by_page[1] = '<svg xmlns="http://www.w3.org/2000/svg"><g id="a"/><use href="#a"/></svg>'
        result = self.render_with(toolkit=toolkit)
        self.assertEqual(len(result.pages), 1)

    def test_non_svg_root_rejected(self):
        toolkit = FakeToolkit()
        toolkit.page_count = 1
        toolkit.svg_by_page[1] = '<html></html>'
        with self.assertRaises(RenderExecutionError):
            self.render_with(toolkit=toolkit)

    def test_malformed_svg_rejected(self):
        toolkit = FakeToolkit()
        toolkit.page_count = 1
        toolkit.svg_by_page[1] = '<svg>'
        with self.assertRaises(RenderExecutionError):
            self.render_with(toolkit=toolkit)

    def test_generated_musicxml_inputs_cross_renderer_boundary(self):
        for seed in range(30):
            with self.subTest(seed=seed):
                result = self.render_with(xml=valid_xml(seed=seed, measures=5))
                self.assertEqual(len(result.pages), 2)


if __name__ == "__main__":
    unittest.main()
