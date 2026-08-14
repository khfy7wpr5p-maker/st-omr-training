from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from st_omr_training.stage7d9_structure_refinement_contract import (
    BARLINE_ROI,
    METER_ROI,
    stage7d9_contract_fingerprint,
)
from st_omr_training.stage7d10_local_roi_derivatives import (
    D10SourceRecord,
    STAGE7D10_LABEL_SCHEMA,
    Stage7D10DerivativeError,
    derive_source_record,
    development_rows,
    materialize_stage7d10_derivatives,
)


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _png(*, mode: str = "L", width: int = 400, height: int = 180) -> bytes:
    image = Image.new(mode, (width, height), 255 if mode == "L" else (255, 255, 255))
    if mode == "L":
        for y in (70, 75, 80, 85, 90):
            for x in range(20, width - 20):
                image.putpixel((x, y), 0)
        for y in range(60, 105):
            image.putpixel((200, y), 0)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _label(*, visible_meter: bool = True, active_meter: str = "3/4") -> dict[str, object]:
    return {
        "schema_version": "stage7d6-staff-structure-label-v1",
        "geometry": {
            "staff_instances": [
                {
                    "staff_instance_id": "staff-1",
                    "system_id": "system-1",
                    "staff_instance_bbox": {
                        "x_min": 20.0,
                        "y_min": 70.0,
                        "x_max": 380.0,
                        "y_max": 90.0,
                    },
                    "staff_spacing": 5.0,
                    "five_staff_lines": [
                        {"start": {"x": 20.0, "y": float(y)}, "end": {"x": 380.0, "y": float(y)}}
                        for y in (70, 75, 80, 85, 90)
                    ],
                }
            ],
            "measures": [
                {
                    "measure_id": "measure-1",
                    "measure_number": 1,
                    "system_id": "system-1",
                    "measure_bbox": {
                        "x_min": 40.0,
                        "y_min": 55.0,
                        "x_max": 200.0,
                        "y_max": 105.0,
                    },
                    "barline_segment": {
                        "start": {"x": 200.0, "y": 60.0},
                        "end": {"x": 200.0, "y": 100.0},
                    },
                    "clef_g2_bbox": None,
                    "meter_bbox": (
                        {
                            "x_min": 48.0,
                            "y_min": 66.0,
                            "x_max": 60.0,
                            "y_max": 94.0,
                        }
                        if visible_meter
                        else None
                    ),
                    "meter_class": active_meter,
                }
            ],
        },
    }


def _source(
    *,
    split: str = "train",
    sample_id: str = "sample-1",
    family_id: str = "family-1",
    visible_meter: bool = True,
    active_meter: str = "3/4",
    image_mode: str = "L",
) -> D10SourceRecord:
    image = _png(mode=image_mode)
    label = _label(visible_meter=visible_meter, active_meter=active_meter)
    label_raw = _canonical(label)
    return D10SourceRecord(
        split=split,
        sample_id=sample_id,
        family_id=family_id,
        image_sha256=sha256(image).hexdigest(),
        label_sha256=sha256(label_raw).hexdigest(),
        image_bytes=image,
        label=label,
    )


class _HostileTestRow(dict):
    def get(self, key: str, default=None):
        if key != "split":
            raise AssertionError(f"TEST field accessed: {key}")
        return "test"


class Stage7D10SplitSafetyTests(unittest.TestCase):
    def test_test_row_is_skipped_before_any_other_field_access(self) -> None:
        rows = development_rows(
            [
                _HostileTestRow(),
                {"split": "train", "sample_id": "a"},
                {"split": "validation", "sample_id": "b"},
            ]
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["split"] for row in rows], ["train", "validation"])

    def test_unknown_split_fails_closed(self) -> None:
        with self.assertRaises(Stage7D10DerivativeError):
            development_rows([{"split": "dev"}])


class Stage7D10SourceIntegrityTests(unittest.TestCase):
    def test_source_hashes_are_required_exactly(self) -> None:
        source = _source()
        with self.assertRaises(ValueError):
            D10SourceRecord(
                split=source.split,
                sample_id=source.sample_id,
                family_id=source.family_id,
                image_sha256="0" * 64,
                label_sha256=source.label_sha256,
                image_bytes=source.image_bytes,
                label=source.label,
            )
        with self.assertRaises(ValueError):
            D10SourceRecord(
                split=source.split,
                sample_id=source.sample_id,
                family_id=source.family_id,
                image_sha256=source.image_sha256,
                label_sha256="0" * 64,
                image_bytes=source.image_bytes,
                label=source.label,
            )

    def test_non_grayscale_source_is_rejected(self) -> None:
        source = _source(image_mode="RGB")
        with self.assertRaises(Stage7D10DerivativeError):
            derive_source_record(source)


class Stage7D10GeometryTests(unittest.TestCase):
    def test_one_measure_emits_barline_and_meter_artifacts(self) -> None:
        artifacts = derive_source_record(_source())
        self.assertEqual(len(artifacts), 2)
        by_kind = {artifact.kind: artifact for artifact in artifacts}
        self.assertEqual(set(by_kind), {"barline", "meter"})

        with Image.open(BytesIO(by_kind["barline"].image_bytes)) as image:
            self.assertEqual(image.mode, "L")
            self.assertEqual(image.size, (BARLINE_ROI.output_width, BARLINE_ROI.output_height))
        with Image.open(BytesIO(by_kind["meter"].image_bytes)) as image:
            self.assertEqual(image.mode, "L")
            self.assertEqual(image.size, (METER_ROI.output_width, METER_ROI.output_height))

        self.assertEqual(by_kind["barline"].label["schema_version"], STAGE7D10_LABEL_SCHEMA)
        self.assertEqual(
            by_kind["barline"].label["d9_contract_fingerprint"],
            stage7d9_contract_fingerprint(),
        )
        target = by_kind["barline"].label["target"]
        self.assertIn("barline_segment", target)

        meter_target = by_kind["meter"].label["target"]
        self.assertEqual(meter_target["meter_class"], "3/4")
        self.assertIsNotNone(meter_target["meter_bbox"])

    def test_active_meter_without_visible_glyph_becomes_none(self) -> None:
        artifacts = derive_source_record(
            _source(visible_meter=False, active_meter="4/4")
        )
        meter = next(item for item in artifacts if item.kind == "meter")
        target = meter.label["target"]
        self.assertEqual(target["meter_class"], "none")
        self.assertIsNone(target["meter_bbox"])

    def test_visible_meter_requires_supported_class(self) -> None:
        with self.assertRaises(Stage7D10DerivativeError):
            derive_source_record(_source(visible_meter=True, active_meter="6/8"))

    def test_derivation_is_byte_deterministic(self) -> None:
        first = derive_source_record(_source())
        second = derive_source_record(_source())
        self.assertEqual(
            [(x.record_id, x.image_sha256, x.label_sha256, x.image_bytes) for x in first],
            [(x.record_id, x.image_sha256, x.label_sha256, x.image_bytes) for x in second],
        )

    def test_duplicate_measure_number_fails_closed(self) -> None:
        source = _source()
        label = json.loads(_canonical(source.label).decode("ascii"))
        label["geometry"]["measures"].append(label["geometry"]["measures"][0])
        raw = _canonical(label)
        bad = D10SourceRecord(
            split="train",
            sample_id="duplicate-measure",
            family_id="family-1",
            image_sha256=source.image_sha256,
            label_sha256=sha256(raw).hexdigest(),
            image_bytes=source.image_bytes,
            label=label,
        )
        with self.assertRaises(Stage7D10DerivativeError):
            derive_source_record(bad)


class Stage7D10PersistenceTests(unittest.TestCase):
    def test_materialization_is_hash_bound_and_keeps_test_zero(self) -> None:
        with TemporaryDirectory() as temp:
            base = Path(temp)
            repo = base / "repo"
            repo.mkdir()
            output = base / "out"
            receipt = materialize_stage7d10_derivatives(
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
            self.assertEqual(receipt.test_records, 0)
            self.assertEqual(receipt.optimizer_steps, 0)
            self.assertEqual(receipt.source_sample_count, 2)
            self.assertEqual(receipt.source_family_count, 2)
            self.assertEqual(receipt.roi_record_count, 4)
            self.assertEqual(receipt.kind_counts, {"barline": 2, "meter": 2})
            self.assertEqual(receipt.split_counts, {"train": 2, "validation": 2})
            self.assertEqual(receipt.meter_class_counts["3/4"], 1)
            self.assertEqual(receipt.meter_class_counts["none"], 1)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "receipt.json").is_file())
            self.assertTrue((output / "COMPLETE").is_file())
            manifest_raw = (output / "manifest.json").read_bytes()
            self.assertEqual(sha256(manifest_raw).hexdigest(), receipt.manifest_sha256)

    def test_output_root_must_be_fresh_and_outside_repo(self) -> None:
        with TemporaryDirectory() as temp:
            base = Path(temp)
            repo = base / "repo"
            repo.mkdir()
            with self.assertRaises(Stage7D10DerivativeError):
                materialize_stage7d10_derivatives(
                    [_source()],
                    output_root=repo / "artifacts",
                    repository_root=repo,
                )
            existing = base / "existing"
            existing.mkdir()
            with self.assertRaises(Stage7D10DerivativeError):
                materialize_stage7d10_derivatives(
                    [_source()],
                    output_root=existing,
                    repository_root=repo,
                )

    def test_duplicate_source_sample_id_fails_closed(self) -> None:
        with TemporaryDirectory() as temp:
            base = Path(temp)
            repo = base / "repo"
            repo.mkdir()
            with self.assertRaises(Stage7D10DerivativeError):
                materialize_stage7d10_derivatives(
                    [_source(), _source()],
                    output_root=base / "out",
                    repository_root=repo,
                )


class Stage7D10ArchitectureTests(unittest.TestCase):
    def test_module_has_no_training_or_checkpoint_path(self) -> None:
        import st_omr_training.stage7d10_local_roi_derivatives as module

        source = inspect.getsource(module)
        for forbidden in (
            "torch.optim",
            ".backward(",
            ".step(",
            "DataLoader(",
            "torch.load(",
            "optimizer =",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("test_records=0", source)
        self.assertIn("optimizer_steps=0", source)


if __name__ == "__main__":
    unittest.main()
