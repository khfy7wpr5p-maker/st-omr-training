from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import tools.meter_real_domain_background_runner_v1 as background_runner
from st_omr_training.stage7d7_specialist_training import Stage7D7Record
from st_omr_training.stage7d10_local_roi_derivatives import (
    Stage7D10DerivativeError,
    load_authoritative_stage7d10_records,
    materialize_stage7d10_derivatives,
    verify_stage7d10_derivatives,
)
from test_stage7d10_local_roi_derivatives import _source


def _record(sample: int, family: int, split: str) -> Stage7D7Record:
    return Stage7D7Record(
        sample_id=f"sample-{split}-{sample}",
        family_id=f"family-{split}-{family}",
        split=split,
        png_sha256="1" * 64,
        label_sha256="2" * 64,
        image_path=Path("/not/read/image.png"),
        label_path=Path("/not/read/label.json"),
    )


def _exact_surface() -> tuple[Stage7D7Record, ...]:
    records: list[Stage7D7Record] = []
    for family in range(410):
        for local in range(3):
            records.append(_record(family * 3 + local, family, "train"))
    for family in range(51):
        for local in range(3):
            records.append(_record(family * 3 + local, family, "validation"))
    return tuple(records)


class Stage7D10AuthoritativeSurfaceTests(unittest.TestCase):
    def test_incomplete_verified_surface_cannot_be_authoritative(self) -> None:
        tiny = (_record(0, 0, "train"),)
        with patch(
            "st_omr_training.stage7d10_local_roi_derivatives.load_verified_stage7d7_records",
            return_value=tiny,
        ):
            with self.assertRaises(Stage7D10DerivativeError):
                load_authoritative_stage7d10_records("corpus", "d6")

    def test_exact_1230_153_family_exclusive_surface_passes_cardinality_gate(self) -> None:
        records = _exact_surface()
        self.assertEqual(len(records), 1383)
        with patch(
            "st_omr_training.stage7d10_local_roi_derivatives.load_verified_stage7d7_records",
            return_value=records,
        ):
            loaded = load_authoritative_stage7d10_records("corpus", "d6")
        self.assertEqual(len(loaded), 1383)

    def test_family_split_leakage_fails_before_materialization(self) -> None:
        with TemporaryDirectory() as temp:
            base = Path(temp)
            repo = base / "repo"
            repo.mkdir()
            with self.assertRaises(Stage7D10DerivativeError):
                materialize_stage7d10_derivatives(
                    [
                        _source(split="train", sample_id="a", family_id="shared"),
                        _source(split="validation", sample_id="b", family_id="shared"),
                    ],
                    output_root=base / "out",
                    repository_root=repo,
                )
            self.assertFalse((base / "out").exists())


class Stage7D10IndependentVerifierTests(unittest.TestCase):
    def _build_fixture(self, base: Path) -> Path:
        repo = base / "repo"
        repo.mkdir()
        output = base / "out"
        materialize_stage7d10_derivatives(
            [
                _source(split="train", sample_id="train-1", family_id="family-a"),
                _source(
                    split="validation",
                    sample_id="val-1",
                    family_id="family-b",
                    visible_meter=False,
                ),
            ],
            output_root=output,
            repository_root=repo,
        )
        return output

    def test_independent_verifier_reopens_complete_fixture(self) -> None:
        with TemporaryDirectory() as temp:
            output = self._build_fixture(Path(temp))
            receipt = verify_stage7d10_derivatives(
                output,
                expected_authoritative_surface=False,
            )
            self.assertEqual(receipt.source_sample_count, 2)
            self.assertEqual(receipt.test_records, 0)
            self.assertEqual(receipt.optimizer_steps, 0)

    def test_background_cache_preserves_independent_verifier_contract(self) -> None:
        with TemporaryDirectory() as temp:
            base = Path(temp)
            source = self._build_fixture(base)
            cache = base / "cache"
            manifest = json.loads((source / "manifest.json").read_text("ascii"))
            old_expected = background_runner.EXPECTED_D10_RECORDS
            background_runner.EXPECTED_D10_RECORDS = len(manifest["records"])
            try:
                background_runner.materialize_d10_cache(
                    source_root=source,
                    cache_root=cache,
                    expected_manifest_sha256=(source / "manifest.sha256").read_text("ascii").split()[0],
                    status=Mock(),
                )
            finally:
                background_runner.EXPECTED_D10_RECORDS = old_expected
            receipt = verify_stage7d10_derivatives(
                cache,
                expected_authoritative_surface=False,
            )
            self.assertEqual(receipt.source_sample_count, 2)
            self.assertEqual(
                {path.name for path in cache.iterdir()},
                {"images", "labels", "manifest.json", "manifest.sha256", "receipt.json", "COMPLETE"},
            )

    def test_tiny_fixture_cannot_be_mislabeled_authoritative(self) -> None:
        with TemporaryDirectory() as temp:
            output = self._build_fixture(Path(temp))
            with self.assertRaises(Stage7D10DerivativeError):
                verify_stage7d10_derivatives(
                    output,
                    expected_authoritative_surface=True,
                )

    def test_persisted_image_tamper_is_detected(self) -> None:
        with TemporaryDirectory() as temp:
            output = self._build_fixture(Path(temp))
            image_path = next((output / "images").iterdir())
            raw = bytearray(image_path.read_bytes())
            raw[-1] ^= 1
            image_path.write_bytes(bytes(raw))
            with self.assertRaises(Stage7D10DerivativeError):
                verify_stage7d10_derivatives(
                    output,
                    expected_authoritative_surface=False,
                )

    def test_persisted_label_tamper_is_detected(self) -> None:
        with TemporaryDirectory() as temp:
            output = self._build_fixture(Path(temp))
            label_path = next((output / "labels").iterdir())
            raw = label_path.read_bytes()
            label_path.write_bytes(raw + b" ")
            with self.assertRaises(Stage7D10DerivativeError):
                verify_stage7d10_derivatives(
                    output,
                    expected_authoritative_surface=False,
                )

    def test_unexpected_artifact_file_is_detected(self) -> None:
        with TemporaryDirectory() as temp:
            output = self._build_fixture(Path(temp))
            (output / "images" / "unexpected.png").write_bytes(b"x")
            with self.assertRaises(Stage7D10DerivativeError):
                verify_stage7d10_derivatives(
                    output,
                    expected_authoritative_surface=False,
                )

    def test_complete_marker_tamper_is_detected(self) -> None:
        with TemporaryDirectory() as temp:
            output = self._build_fixture(Path(temp))
            (output / "COMPLETE").write_text("0" * 64 + "\n", encoding="ascii")
            with self.assertRaises(Stage7D10DerivativeError):
                verify_stage7d10_derivatives(
                    output,
                    expected_authoritative_surface=False,
                )


if __name__ == "__main__":
    unittest.main()
