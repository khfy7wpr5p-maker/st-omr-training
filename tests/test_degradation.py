from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from st_omr_training.degradation import (
    CAIROSVG_PINNED_VERSION,
    DEGRADATION_VERSION,
    PILLOW_PINNED_VERSION,
    DegradationConfig,
    DegradationExecutionError,
    DegradationInputError,
    DegradationRuntimeError,
    DegradationSource,
    degrade_page,
    degradation_config_fingerprint,
    degrade_render_result_page,
    sample_degradation_config,
    source_from_render_result,
)


def sample_svg() -> bytes:
    return b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2100 2970">
    <rect width="2100" height="2970" fill="white"/>
    <g stroke="black" stroke-width="10">
      <line x1="200" y1="500" x2="1900" y2="500"/>
      <line x1="200" y1="550" x2="1900" y2="550"/>
      <line x1="200" y1="600" x2="1900" y2="600"/>
      <line x1="200" y1="650" x2="1900" y2="650"/>
      <line x1="200" y1="700" x2="1900" y2="700"/>
    </g>
    <ellipse cx="1000" cy="600" rx="45" ry="32" fill="black"/>
    <path d="M1040 600 V 350" stroke="black" stroke-width="10"/>
    </svg>'''


def source(svg: bytes | None = None) -> DegradationSource:
    data = sample_svg() if svg is None else svg
    return DegradationSource(
        family_id="score-001",
        page_number=1,
        source_musicxml_sha256="a" * 64,
        renderer_config_fingerprint="b" * 64,
        svg=data,
        svg_sha256=sha256(data).hexdigest(),
    )


class ConfigTests(unittest.TestCase):
    def test_defaults_are_identity_profile(self):
        config = DegradationConfig()
        self.assertEqual(config.rotation_mdeg, 0)
        self.assertEqual(config.jpeg_quality, 0)

    def test_bool_rejected_as_integer(self):
        with self.assertRaises(TypeError):
            DegradationConfig(seed=True)

    def test_bounds_fail_closed(self):
        invalid = [
            {"raster_width": 511},
            {"rotation_mdeg": 3001},
            {"blur_milli": 2001},
            {"noise_level": 21},
            {"brightness_milli": 799},
            {"contrast_milli": 1251},
            {"jpeg_quality": 64},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DegradationConfig(**kwargs)

    def test_fingerprint_is_deterministic_and_parameter_sensitive(self):
        a = DegradationConfig(seed=7, raster_width=700)
        b = DegradationConfig(seed=7, raster_width=700)
        c = DegradationConfig(seed=8, raster_width=700)
        self.assertEqual(degradation_config_fingerprint(a), degradation_config_fingerprint(b))
        self.assertNotEqual(degradation_config_fingerprint(a), degradation_config_fingerprint(c))

    def test_seeded_profile_sampling_is_replayable(self):
        self.assertEqual(sample_degradation_config(123, "medium", raster_width=700),
                         sample_degradation_config(123, "medium", raster_width=700))
        self.assertNotEqual(sample_degradation_config(123, "medium", raster_width=700),
                            sample_degradation_config(124, "medium", raster_width=700))

    def test_profiles_stay_within_contract(self):
        for profile in ("clean", "light", "medium"):
            for seed in range(50):
                config = sample_degradation_config(seed, profile, raster_width=700)
                self.assertIsInstance(config, DegradationConfig)


class SourceTests(unittest.TestCase):
    def test_hash_mismatch_rejected(self):
        data = sample_svg()
        with self.assertRaises(DegradationInputError):
            DegradationSource("score-1", 1, "a"*64, "b"*64, data, "0"*64)

    def test_invalid_family_id_rejected(self):
        data = sample_svg()
        with self.assertRaises(DegradationInputError):
            DegradationSource("../score", 1, "a"*64, "b"*64, data, sha256(data).hexdigest())

    def test_doctype_rejected(self):
        data = b'<!DOCTYPE svg><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M0 0L1 1"/></svg>'
        with self.assertRaises(DegradationInputError):
            source(data)

    def test_external_href_rejected(self):
        data = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><use href="https://example.invalid/a.svg#x"/></svg>'
        with self.assertRaises(DegradationInputError):
            source(data)

    def test_external_css_url_rejected(self):
        data = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><style>.x{fill:url(https://example.invalid/x)}</style><path class="x" d="M0 0L1 1"/></svg>'
        with self.assertRaises(DegradationInputError):
            source(data)

    def test_internal_fragment_reference_allowed(self):
        data = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><defs><path id="x" d="M0 0L1 1"/></defs><use href="#x"/></svg>'
        self.assertEqual(source(data).page_number, 1)

    def test_missing_or_extreme_viewbox_rejected(self):
        values = [
            b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L1 1"/></svg>',
            b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 10000"><path d="M0 0L1 1"/></svg>',
            b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 nan 10"><path d="M0 0L1 1"/></svg>',
        ]
        for data in values:
            with self.subTest(data=data), self.assertRaises(DegradationInputError):
                source(data)


class RenderResultBridgeTests(unittest.TestCase):
    def test_bridge_copies_and_revalidates_stage3_lineage(self):
        data = sample_svg()
        page = SimpleNamespace(page_number=1, svg=data, sha256=sha256(data).hexdigest())
        render_result = SimpleNamespace(
            source_musicxml_sha256="a" * 64,
            config_fingerprint="b" * 64,
            pages=(page,),
        )
        src = source_from_render_result(render_result, family_id="score-bridge")
        self.assertEqual(src.svg_sha256, page.sha256)
        result = degrade_render_result_page(
            render_result, family_id="score-bridge", config=DegradationConfig(raster_width=700)
        )
        self.assertEqual(result.family_id, "score-bridge")

    def test_bridge_rejects_mutable_page_collection(self):
        data = sample_svg()
        page = SimpleNamespace(page_number=1, svg=data, sha256=sha256(data).hexdigest())
        render_result = SimpleNamespace(
            source_musicxml_sha256="a" * 64, config_fingerprint="b" * 64, pages=[page]
        )
        with self.assertRaises(DegradationInputError):
            source_from_render_result(render_result, family_id="score-bridge")

    def test_bridge_rejects_tampered_page_hash(self):
        data = sample_svg()
        page = SimpleNamespace(page_number=1, svg=data, sha256="0" * 64)
        render_result = SimpleNamespace(
            source_musicxml_sha256="a" * 64, config_fingerprint="b" * 64, pages=(page,)
        )
        with self.assertRaises(DegradationInputError):
            source_from_render_result(render_result, family_id="score-bridge")


class RuntimeTests(unittest.TestCase):
    def test_exact_runtime_versions_are_exposed_in_result(self):
        result = degrade_page(source(), DegradationConfig(raster_width=700))
        self.assertEqual(result.cairosvg_version, CAIROSVG_PINNED_VERSION)
        self.assertEqual(result.pillow_version, PILLOW_PINNED_VERSION)
        self.assertEqual(result.degradation_version, DEGRADATION_VERSION)

    def test_runtime_version_drift_fails_closed(self):
        def fake_version(name):
            return "0.0.0" if name == "CairoSVG" else PILLOW_PINNED_VERSION
        with patch("st_omr_training.degradation.metadata.version", side_effect=fake_version):
            with self.assertRaises(DegradationRuntimeError):
                degrade_page(source(), DegradationConfig(raster_width=700))


class DegradationTests(unittest.TestCase):
    def test_clean_raster_is_grayscale_png_with_lineage(self):
        src = source()
        result = degrade_page(src, DegradationConfig(seed=9, raster_width=700))
        self.assertEqual(result.family_id, src.family_id)
        self.assertEqual(result.page_number, 1)
        self.assertEqual(result.source_svg_sha256, src.svg_sha256)
        self.assertEqual(result.clean_raster_sha256, result.png_sha256)
        self.assertEqual(result.mode, "L")
        self.assertRegex(result.derivative_id, r"^[0-9a-f]{64}$")
        with Image.open(BytesIO(result.png)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.mode, "L")
            self.assertEqual(image.width, 700)

    def test_identity_is_byte_deterministic(self):
        config = DegradationConfig(seed=11, raster_width=700)
        a = degrade_page(source(), config)
        b = degrade_page(source(), config)
        self.assertEqual(a.png, b.png)
        self.assertEqual(a.png_sha256, b.png_sha256)
        self.assertEqual(a.derivative_id, b.derivative_id)

    def test_medium_profile_is_byte_deterministic(self):
        config = sample_degradation_config(42, "medium", raster_width=700)
        a = degrade_page(source(), config)
        b = degrade_page(source(), config)
        self.assertEqual(a.png, b.png)
        self.assertEqual(a.derivative_id, b.derivative_id)
        self.assertNotEqual(a.clean_raster_sha256, a.png_sha256)

    def test_rotation_expands_instead_of_cropping(self):
        clean = degrade_page(source(), DegradationConfig(raster_width=700))
        rotated = degrade_page(source(), DegradationConfig(raster_width=700, rotation_mdeg=3000))
        self.assertGreater(rotated.width * rotated.height, clean.width * clean.height)

    def test_noise_seed_changes_derivative(self):
        a = degrade_page(source(), DegradationConfig(seed=1, raster_width=700, noise_level=5))
        b = degrade_page(source(), DegradationConfig(seed=2, raster_width=700, noise_level=5))
        self.assertNotEqual(a.png_sha256, b.png_sha256)
        self.assertNotEqual(a.derivative_id, b.derivative_id)

    def test_blank_svg_fails_closed_after_rasterization(self):
        data = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2100 2970"><rect width="2100" height="2970" fill="white"/></svg>'
        with self.assertRaises(DegradationExecutionError):
            degrade_page(source(data), DegradationConfig(raster_width=700))

    def test_multiple_seeded_medium_derivatives_are_valid_and_distinct(self):
        hashes = set()
        src = source()
        for seed in range(10):
            result = degrade_page(src, sample_degradation_config(seed, "medium", raster_width=512))
            hashes.add(result.png_sha256)
        self.assertEqual(len(hashes), 10)


if __name__ == "__main__":
    unittest.main()
