from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import inspect
import json
import math
import unittest

from PIL import Image

from st_omr_training.m4_e3i_meter_roi_adapter import (
    D11_PROPOSAL_THRESHOLD,
    FROZEN_D11_CHECKPOINT_SHA256,
    FROZEN_D7_CHECKPOINT_SHA256,
    FROZEN_TRAIN_ANCHOR_POLICY,
    MAX_CANDIDATES_PER_SYSTEM,
    SPECIALIST_THRESHOLDS,
    FrozenTrainAnchorPolicy,
    M4E3IAdapterError,
    MeasureStartCandidate,
    recover_canonical_meter_roi_candidates,
    recover_measure_start_candidates,
    render_canonical_meter_roi,
)
from st_omr_training.stage7d10_local_roi_derivatives import (
    D10SourceRecord,
    derive_source_record,
)


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _staff_lines(lefts: tuple[float, ...] = (20.0, 20.0, 20.0, 20.0, 20.0)):
    assert len(lefts) == 5
    return tuple(
        {
            "start": {"x": left, "y": float(y)},
            "end": {"x": 380.0, "y": float(y)},
        }
        for left, y in zip(lefts, (70, 75, 80, 85, 90))
    )


def _staff_bbox() -> dict[str, float]:
    return {"x_min": 20.0, "y_min": 70.0, "x_max": 380.0, "y_max": 90.0}


def _png() -> bytes:
    image = Image.new("L", (400, 180), 255)
    for y in (70, 75, 80, 85, 90):
        for x in range(20, 380):
            image.putpixel((x, y), 0)
    # Meter-like dark strokes near canonical measure start.
    for x in range(48, 61):
        for y in range(66, 95):
            if x in (48, 49, 59, 60) or y in (66, 67, 93, 94):
                image.putpixel((x, y), 0)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _d10_source() -> D10SourceRecord:
    image = _png()
    label = {
        "schema_version": "stage7d6-staff-structure-label-v1",
        "geometry": {
            "staff_instances": [
                {
                    "staff_instance_id": "staff-1",
                    "system_id": "system-1",
                    "staff_instance_bbox": _staff_bbox(),
                    "staff_spacing": 5.0,
                    "five_staff_lines": list(_staff_lines()),
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
                    "meter_bbox": {
                        "x_min": 48.0,
                        "y_min": 66.0,
                        "x_max": 60.0,
                        "y_max": 94.0,
                    },
                    "meter_class": "3/4",
                }
            ],
        },
    }
    raw = _canonical(label)
    return D10SourceRecord(
        split="train",
        sample_id="e3i-parity",
        family_id="family-e3i",
        image_sha256=sha256(image).hexdigest(),
        label_sha256=sha256(raw).hexdigest(),
        image_bytes=image,
        label=label,
    )


class M4E3IPolicyTests(unittest.TestCase):
    def test_frozen_dependencies_and_thresholds_are_unchanged(self) -> None:
        self.assertEqual(D11_PROPOSAL_THRESHOLD, 0.90)
        self.assertEqual(SPECIALIST_THRESHOLDS, (("2", 0.48), ("3", 0.60), ("4", 0.47)))
        self.assertEqual(
            FROZEN_D11_CHECKPOINT_SHA256,
            "cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3",
        )
        self.assertEqual(
            FROZEN_D7_CHECKPOINT_SHA256,
            "5f009ca8ba68d38497a7dd25590d4dd98c537f20c5d5525bf66e288afbf417dc",
        )
        self.assertEqual(MAX_CANDIDATES_PER_SYSTEM, 2)

    def test_frozen_anchor_policy_is_train_only_and_hash_bound(self) -> None:
        self.assertEqual(FROZEN_TRAIN_ANCHOR_POLICY.derivation_split, "train")
        self.assertEqual(FROZEN_TRAIN_ANCHOR_POLICY.max_candidates, 2)
        self.assertEqual(
            FROZEN_TRAIN_ANCHOR_POLICY.source_result_sha256,
            "db9536b983c7aabee30243696fb88e8ea74016b4600a70b93ad630562b8b86ec",
        )
        self.assertTrue(
            math.isclose(
                FROZEN_TRAIN_ANCHOR_POLICY.first_measure_offset_staff_spaces,
                -0.06619667590040451,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )

    def test_validation_or_test_derived_policy_fails_closed(self) -> None:
        for forbidden_split in ("validation", "test"):
            with self.assertRaises(M4E3IAdapterError):
                FrozenTrainAnchorPolicy(
                    source_stage="forbidden",
                    source_result_sha256="0" * 64,
                    derivation_split=forbidden_split,
                    first_measure_offset_staff_spaces=0.0,
                    candidate_methods=(
                        "median_five_line_left",
                        "coverage_min_five_line_left",
                    ),
                    max_candidates=2,
                )


class M4E3ICandidateTests(unittest.TestCase):
    def test_identical_staff_lefts_deduplicate_to_one_candidate(self) -> None:
        candidates = recover_measure_start_candidates(_staff_lines())
        self.assertEqual(len(candidates), 1)
        expected = 20.0 + 5.0 * FROZEN_TRAIN_ANCHOR_POLICY.first_measure_offset_staff_spaces
        self.assertTrue(math.isclose(candidates[0].anchor_x, expected, rel_tol=0.0, abs_tol=1e-12))
        self.assertEqual(candidates[0].method, "median_five_line_left")
        self.assertEqual(candidates[0].staff_spacing, 5.0)

    def test_late_majority_line_fragments_get_bounded_coverage_fallback(self) -> None:
        candidates = recover_measure_start_candidates(
            _staff_lines((20.0, 200.0, 200.0, 200.0, 200.0))
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            [candidate.method for candidate in candidates],
            ["median_five_line_left", "coverage_min_five_line_left"],
        )
        self.assertGreater(candidates[0].anchor_x, candidates[1].anchor_x)
        self.assertLessEqual(len(candidates), MAX_CANDIDATES_PER_SYSTEM)

    def test_wrong_line_count_and_degenerate_spacing_fail_closed(self) -> None:
        with self.assertRaises(M4E3IAdapterError):
            recover_measure_start_candidates(_staff_lines()[:4])
        degenerate = list(_staff_lines())
        degenerate[1] = {
            "start": {"x": 20.0, "y": 70.0},
            "end": {"x": 380.0, "y": 70.0},
        }
        with self.assertRaises(M4E3IAdapterError):
            recover_measure_start_candidates(tuple(degenerate))


class M4E3ICanonicalParityTests(unittest.TestCase):
    def test_manual_anchor_is_byte_identical_to_canonical_d10_meter_roi(self) -> None:
        source = _d10_source()
        canonical = next(
            artifact for artifact in derive_source_record(source) if artifact.kind == "meter"
        )
        candidate = MeasureStartCandidate(
            anchor_x=40.0,
            method="median_five_line_left",
            staff_left_x=40.0,
            staff_spacing=5.0,
        )
        recovered = render_canonical_meter_roi(
            source.image_bytes,
            staff_bbox=_staff_bbox(),
            candidate=candidate,
        )
        self.assertEqual(recovered.image_bytes, canonical.image_bytes)
        self.assertEqual(recovered.image_sha256, canonical.image_sha256)
        self.assertEqual(asdict(recovered.transform), canonical.label["roi_transform"])

    def test_end_to_end_recovery_never_emits_more_than_two_canonical_rois(self) -> None:
        rois = recover_canonical_meter_roi_candidates(
            _png(),
            staff_bbox=_staff_bbox(),
            five_staff_lines=_staff_lines((20.0, 200.0, 200.0, 200.0, 200.0)),
        )
        self.assertEqual(len(rois), 2)
        self.assertTrue(all(len(item.image_bytes) > 0 for item in rois))
        self.assertTrue(all(len(item.image_sha256) == 64 for item in rois))

    def test_non_grayscale_input_fails_closed_via_d10(self) -> None:
        image = Image.new("RGB", (400, 180), (255, 255, 255))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        candidate = recover_measure_start_candidates(_staff_lines())[0]
        with self.assertRaises(M4E3IAdapterError):
            render_canonical_meter_roi(
                buffer.getvalue(),
                staff_bbox=_staff_bbox(),
                candidate=candidate,
            )


class M4E3IArchitectureTests(unittest.TestCase):
    def test_adapter_contains_no_model_or_optimizer_execution_path(self) -> None:
        import st_omr_training.m4_e3i_meter_roi_adapter as module

        source = inspect.getsource(module)
        for forbidden in (
            "import torch",
            "torch.optim",
            ".backward(",
            "optimizer.step(",
            "load_state_dict(",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
