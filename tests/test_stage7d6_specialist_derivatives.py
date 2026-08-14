from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import unittest

from st_omr_training.degradation import (
    DegradationConfig,
    DegradationSource,
    degrade_page,
    degradation_config_fingerprint,
)
from st_omr_training.renderer import render_musicxml_svg
from st_omr_training.stage7d5_geometry import (
    extract_staff_structure_geometry,
    map_page_geometry_to_final_png,
    render_musicxml_geometry_svg,
)
from st_omr_training.stage7d6_specialist_derivatives import (
    EXPECTED_DEVELOPMENT_FAMILY_COUNT,
    EXPECTED_DEVELOPMENT_SAMPLE_COUNT,
    STAGE7D6_LABEL_SCHEMA,
    _canonical_json,
    _config_from_row,
    _development_rows,
    _label_payload,
    _manifest_page_view,
    _source_row_index,
    stage7d6_profile_fingerprint,
)


GOLDEN = Path(__file__).parent / "golden"


class _TestSplitTrap(Mapping[str, object]):
    """A TEST row that explodes if D6 asks for anything except split."""

    def __getitem__(self, key: str) -> object:
        if key == "split":
            return "test"
        raise AssertionError(f"TEST field accessed: {key}")

    def __iter__(self):
        # Mapping.get() uses __getitem__; iteration must never be needed before skip.
        raise AssertionError("TEST row iterated before split skip")

    def __len__(self) -> int:
        raise AssertionError("TEST row length inspected before split skip")

    def get(self, key: str, default=None):
        if key == "split":
            return "test"
        raise AssertionError(f"TEST field accessed: {key}")


def _sha(text: str) -> str:
    return sha256(text.encode("ascii")).hexdigest()


def _development_meta_row(family: str, split: str, derivative: int) -> dict[str, object]:
    return {
        "sample_id": _sha(f"sample:{family}:{derivative}"),
        "family_id": family,
        "split": split,
        "page_number": 1,
        "source_musicxml_sha256": _sha(f"xml:{family}"),
        "source_svg_sha256": _sha(f"svg:{family}"),
        "renderer_config_fingerprint": _sha("renderer"),
        "degradation_config_fingerprint": _sha(f"degrade:{family}:{derivative}"),
        "png_sha256": _sha(f"png:{family}:{derivative}"),
        "width": 1400,
        "height": 1980,
    }


class Stage7D6SplitSafetyTests(unittest.TestCase):
    def test_test_row_is_skipped_before_any_other_field_access(self) -> None:
        self.assertEqual(_development_rows([_TestSplitTrap()]), ())

    def test_exact_frozen_development_surface_is_family_exclusive(self) -> None:
        rows: list[Mapping[str, object]] = []
        for index in range(410):
            family = f"train-{index:04d}"
            rows.extend(_development_meta_row(family, "train", d) for d in range(3))
        for index in range(51):
            family = f"validation-{index:04d}"
            rows.extend(_development_meta_row(family, "validation", d) for d in range(3))
        self.assertEqual(len(rows), EXPECTED_DEVELOPMENT_SAMPLE_COUNT)
        index = _source_row_index(tuple(rows))
        self.assertEqual(len(index), EXPECTED_DEVELOPMENT_SAMPLE_COUNT)
        self.assertEqual(
            len({row["family_id"] for row in index.values()}),
            EXPECTED_DEVELOPMENT_FAMILY_COUNT,
        )

    def test_profile_fingerprint_is_stable_sha256(self) -> None:
        first = stage7d6_profile_fingerprint()
        second = stage7d6_profile_fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        int(first, 16)


class Stage7D6LiveLabelTests(unittest.TestCase):
    def test_manifest_lineage_maps_exactly_to_final_png_geometry(self) -> None:
        musicxml = (GOLDEN / "time_change.musicxml").read_bytes()
        base = render_musicxml_svg(musicxml)
        geometry_render = render_musicxml_geometry_svg(musicxml)
        geometry_pages = extract_staff_structure_geometry(geometry_render, musicxml)
        self.assertEqual(len(base.pages), len(geometry_pages))

        config = DegradationConfig(
            seed=7006,
            raster_width=1400,
            rotation_mdeg=1700,
            blur_milli=300,
            noise_level=2,
            brightness_milli=980,
            contrast_milli=1030,
        )
        base_page = base.pages[0]
        source = DegradationSource(
            family_id="d6-live-family",
            page_number=base_page.page_number,
            source_musicxml_sha256=base.source_musicxml_sha256,
            renderer_config_fingerprint=base.config_fingerprint,
            svg=base_page.svg,
            svg_sha256=base_page.sha256,
        )
        degraded = degrade_page(source, config)
        row = {
            "sample_id": _sha("d6-live-sample"),
            "family_id": "d6-live-family",
            "split": "train",
            "page_number": degraded.page_number,
            "source_musicxml_sha256": degraded.source_musicxml_sha256,
            "renderer_config_fingerprint": degraded.renderer_config_fingerprint,
            "source_svg_sha256": degraded.source_svg_sha256,
            "clean_raster_sha256": degraded.clean_raster_sha256,
            "degradation_config_fingerprint": degraded.degradation_config_fingerprint,
            "degradation_config": asdict(config),
            "derivative_id": degraded.derivative_id,
            "png_sha256": degraded.png_sha256,
            "degradation_version": degraded.degradation_version,
            "cairosvg_version": degraded.cairosvg_version,
            "pillow_version": degraded.pillow_version,
            "cairo_runtime_version": degraded.cairo_runtime_version,
            "python_version": degraded.python_version,
            "platform_system": degraded.platform_system,
            "platform_machine": degraded.platform_machine,
            "clean_width": degraded.clean_width,
            "clean_height": degraded.clean_height,
            "width": degraded.width,
            "height": degraded.height,
            "mode": degraded.mode,
            "image_format": "png",
        }
        parsed_config = _config_from_row(row)
        view = _manifest_page_view(row, parsed_config)
        mapped_from_manifest = map_page_geometry_to_final_png(geometry_pages[0], view)
        mapped_from_artifact = map_page_geometry_to_final_png(geometry_pages[0], degraded)
        self.assertEqual(asdict(mapped_from_manifest), asdict(mapped_from_artifact))

        label = _label_payload(row, mapped_from_manifest)
        self.assertEqual(label["schema_version"], STAGE7D6_LABEL_SCHEMA)
        self.assertEqual(label["image"]["png_sha256"], degraded.png_sha256)
        self.assertEqual(label["geometry"]["coordinate_space"], "final_png_pixels")
        self.assertEqual(
            label["lineage"]["degradation_config_fingerprint"],
            degradation_config_fingerprint(config),
        )
        encoded = _canonical_json(label)
        self.assertEqual(encoded, _canonical_json(label))
        self.assertGreater(len(encoded), 0)


if __name__ == "__main__":
    unittest.main()
