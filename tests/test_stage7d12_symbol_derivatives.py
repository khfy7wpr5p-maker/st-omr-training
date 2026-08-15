from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from st_omr_training.degradation import DegradationConfig, degradation_config_fingerprint
from st_omr_training.stage7d12_symbol_derivatives import (
    EXPECTED_D6_ARTIFACT_BINDING_SHA256,
    EXPECTED_D6_DERIVATIVE_BUILD_ID,
    EXPECTED_D6_MANIFEST_SHA256,
    Stage7D12DerivativeError,
    _prepare_output_root,
    development_rows,
    map_symbol_page_to_final_png,
    stage7d12_derivative_profile_fingerprint,
)
from st_omr_training.stage7d12_symbol_geometry import extract_symbol_geometry
from st_omr_training.stage7d5_geometry import render_musicxml_geometry_svg


GOLDEN = Path(__file__).parent / "golden"


class _HostileTestRow(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        if key == "split":
            return "test"
        raise AssertionError(f"D12 builder touched sealed TEST field: {key}")

    def __iter__(self):
        yield "split"

    def __len__(self) -> int:
        return 1

    def get(self, key: str, default=None):
        if key == "split":
            return "test"
        raise AssertionError(f"D12 builder touched sealed TEST field: {key}")


class Stage7D12SymbolDerivativeTests(unittest.TestCase):
    def test_development_rows_skip_test_before_other_field_access(self) -> None:
        rows = development_rows(
            [
                _HostileTestRow(),
                {"split": "train", "sample_id": "not-read-here"},
                {"split": "validation", "sample_id": "not-read-here"},
            ]
        )
        self.assertEqual([row.get("split") for row in rows], ["train", "validation"])

    def test_profile_binds_exact_accepted_d6_identity(self) -> None:
        self.assertEqual(
            EXPECTED_D6_DERIVATIVE_BUILD_ID,
            "0faafe229f3497b1147cf0f0ac0ce4b7efe6fa31f360a6a33a3b82c986c8c519",
        )
        self.assertEqual(
            EXPECTED_D6_MANIFEST_SHA256,
            "e8e415eb6ba9d91a1a880709c3f31d559aa20bf5149734f45b5f84ced16afee9",
        )
        self.assertEqual(
            EXPECTED_D6_ARTIFACT_BINDING_SHA256,
            "3b7558f0f927ad47a61ed5afb5faa8584dca8647cf8683d4043686eb7b077ea1",
        )
        first = stage7d12_derivative_profile_fingerprint()
        second = stage7d12_derivative_profile_fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_symbol_mapping_replays_final_png_transform_without_learning(self) -> None:
        musicxml = (GOLDEN / "accidentals.musicxml").read_bytes()
        render = render_musicxml_geometry_svg(musicxml)
        pages = extract_symbol_geometry(render, musicxml)
        self.assertEqual(len(pages), 1)
        page = pages[0]

        config = DegradationConfig(raster_width=1400)
        x0, y0, vb_width, vb_height = page.view_box
        self.assertGreater(vb_width, 0)
        self.assertGreater(vb_height, 0)
        clean_width = config.raster_width
        clean_height = round(vb_height * clean_width / vb_width)
        view = SimpleNamespace(
            page_number=page.page_number,
            source_musicxml_sha256=page.source_musicxml_sha256,
            renderer_config_fingerprint=page.base_renderer_config_fingerprint,
            degradation_config_fingerprint=degradation_config_fingerprint(config),
            config=config,
            clean_width=clean_width,
            clean_height=clean_height,
            width=clean_width,
            height=clean_height,
        )

        measures, fingerprint = map_symbol_page_to_final_png(page, view)
        self.assertEqual(len(fingerprint), 64)
        noteheads = [row for measure in measures for row in measure["noteheads"]]
        accidentals = [row for measure in measures for row in measure["accidentals"]]
        self.assertEqual(len(noteheads), 4)
        self.assertEqual(len(accidentals), 4)
        self.assertEqual(
            [row["accidental_class"] for row in accidentals],
            ["sharp", "natural", "flat", "natural"],
        )
        self.assertEqual(
            [row["canonical_event_id"] for row in noteheads],
            [row["canonical_event_id"] for row in accidentals],
        )
        for measure in measures:
            for row in measure["noteheads"]:
                box = row["notehead_bbox"]
                center = row["notehead_center"]
                self.assertLess(box["x_min"], box["x_max"])
                self.assertLess(box["y_min"], box["y_max"])
                self.assertLessEqual(box["x_min"], center["x"])
                self.assertLessEqual(center["x"], box["x_max"])
                self.assertLessEqual(box["y_min"], center["y"])
                self.assertLessEqual(center["y"], box["y_max"])

    def test_output_root_must_be_fresh_and_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "existing"
            existing.mkdir()
            with self.assertRaises(Stage7D12DerivativeError):
                _prepare_output_root(existing)

            fresh = Path(temporary) / "fresh"
            _prepare_output_root(fresh)
            self.assertTrue((fresh / "labels").is_dir())

        repository_child = Path(__file__).resolve().parents[1] / "forbidden-d12-output"
        with self.assertRaises(Stage7D12DerivativeError):
            _prepare_output_root(repository_child)


if __name__ == "__main__":
    unittest.main()
