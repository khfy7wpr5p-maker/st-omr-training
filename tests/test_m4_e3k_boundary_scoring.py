from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from st_omr_training.m4_e3k_boundary_scoring import (
    M4E3KScoringError,
    profile_fingerprint,
    score_e3k_a_split,
)
from st_omr_training.stage7d6_specialist_derivatives import (
    STAGE7D6_LABEL_SCHEMA,
    STAGE7D6_VERSION,
)
from st_omr_training.stage7d7_specialist_training import Stage7D7Record


STAFF_LINES = tuple(
    {
        "start": {"x": 20.0, "y": float(y)},
        "end": {"x": 380.0, "y": float(y)},
    }
    for y in (40, 50, 60, 70, 80)
)


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _png_bytes(*, include_barlines: bool = True) -> bytes:
    image = Image.new("L", (400, 120), 255)
    draw = ImageDraw.Draw(image)
    for y in (40, 50, 60, 70, 80):
        draw.line((20, y, 380, y), fill=0, width=1)
    if include_barlines:
        draw.line((100, 40, 100, 80), fill=0, width=2)
        draw.line((350, 40, 350, 80), fill=0, width=2)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _label(*, sample_id: str, split: str) -> dict[str, object]:
    return {
        "schema_version": STAGE7D6_LABEL_SCHEMA,
        "stage7d6_version": STAGE7D6_VERSION,
        "sample_id": sample_id,
        "split": split,
        "image": {"width": 400, "height": 120},
        "geometry": {
            "systems": [
                {
                    "system_id": "sys-1",
                    "system_bbox": {"x_min": 15.0, "y_min": 30.0, "x_max": 385.0, "y_max": 90.0},
                    "staff_instance_id": "staff-1",
                    "measure_numbers": [1, 2],
                }
            ],
            "staff_instances": [
                {
                    "staff_instance_id": "staff-1",
                    "system_id": "sys-1",
                    "five_staff_lines": list(STAFF_LINES),
                    "staff_instance_bbox": {"x_min": 20.0, "y_min": 40.0, "x_max": 380.0, "y_max": 80.0},
                    "staff_spacing": 10.0,
                }
            ],
            "measures": [
                {
                    "measure_id": "m1",
                    "measure_number": 1,
                    "system_id": "sys-1",
                    "barline_segment": {
                        "start": {"x": 100.0, "y": 38.0},
                        "end": {"x": 100.0, "y": 82.0},
                    },
                },
                {
                    "measure_id": "m2",
                    "measure_number": 2,
                    "system_id": "sys-1",
                    "barline_segment": {
                        "start": {"x": 350.0, "y": 38.0},
                        "end": {"x": 350.0, "y": 82.0},
                    },
                },
            ],
        },
    }


def _record(root: Path, *, split: str = "train", include_barlines: bool = True) -> Stage7D7Record:
    sample_id = "a" * 64
    image_raw = _png_bytes(include_barlines=include_barlines)
    label_raw = _canonical(_label(sample_id=sample_id, split=split))
    image_path = root / "image.png"
    label_path = root / "label.json"
    image_path.write_bytes(image_raw)
    label_path.write_bytes(label_raw)
    return Stage7D7Record(
        sample_id=sample_id,
        family_id="family-1",
        split=split,
        png_sha256=sha256(image_raw).hexdigest(),
        label_sha256=sha256(label_raw).hexdigest(),
        image_path=image_path,
        label_path=label_path,
    )


class M4E3KBoundaryScoringTests(unittest.TestCase):
    def test_profile_fingerprint_is_deterministic(self) -> None:
        first = profile_fingerprint()
        second = profile_fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_train_feasibility_scores_only_interior_boundary(self) -> None:
        with TemporaryDirectory() as temp:
            record = _record(Path(temp), split="train")
            with patch(
                "st_omr_training.m4_e3k_boundary_scoring.load_verified_stage7d7_records",
                return_value=(record,),
            ):
                report = score_e3k_a_split(temp, temp, split="train")
        self.assertEqual(report["surface"]["records"], 1)
        self.assertEqual(report["surface"]["systems"], 1)
        self.assertEqual(report["surface"]["topology_relevant_interior_boundaries"], 1)
        self.assertEqual(report["metrics"]["boundary_recall_by_tolerance_staff_spaces"]["1.0"], 1.0)
        self.assertTrue(report["gate"]["pass"])
        self.assertTrue(report["gate"]["authorizes_e3k_b"])
        self.assertFalse(report["gate"]["authorizes_d11_validator"])
        self.assertFalse(report["safety"]["deployment_readiness_claimed"])
        self.assertEqual(report["safety"]["optimizer_steps"], 0)
        self.assertFalse(report["safety"]["test_opened"])

    def test_validation_a_is_still_upper_bound_and_does_not_authorize_d11(self) -> None:
        with TemporaryDirectory() as temp:
            record = _record(Path(temp), split="validation")
            with patch(
                "st_omr_training.m4_e3k_boundary_scoring.load_verified_stage7d7_records",
                return_value=(record,),
            ):
                report = score_e3k_a_split(temp, temp, split="validation")
        self.assertTrue(report["gate"]["pass"])
        self.assertTrue(report["gate"]["authorizes_e3k_b"])
        self.assertFalse(report["gate"]["authorizes_d11_validator"])
        self.assertEqual(report["safety"]["staff_geometry_source"], "authoritative_D6_ground_truth_upper_bound_only")

    def test_no_proposal_failure_does_not_hide_infinite_nearest_error(self) -> None:
        with TemporaryDirectory() as temp:
            record = _record(Path(temp), split="train", include_barlines=False)
            with patch(
                "st_omr_training.m4_e3k_boundary_scoring.load_verified_stage7d7_records",
                return_value=(record,),
            ):
                report = score_e3k_a_split(temp, temp, split="train")
        self.assertFalse(report["gate"]["pass"])
        self.assertEqual(report["metrics"]["boundary_recall_by_tolerance_staff_spaces"]["1.0"], 0.0)
        self.assertIsNone(report["metrics"]["nearest_boundary_error_p50_staff_spaces"])
        self.assertIsNone(report["metrics"]["nearest_boundary_error_p95_staff_spaces"])
        self.assertEqual(report["metrics"]["reason_counts"]["NO_PROPOSAL"], 1)

    def test_test_split_is_rejected_before_loader(self) -> None:
        with patch(
            "st_omr_training.m4_e3k_boundary_scoring.load_verified_stage7d7_records"
        ) as loader:
            with self.assertRaises(M4E3KScoringError):
                score_e3k_a_split("unused", "unused", split="test")
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
