from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from st_omr_training.meter_v5_1r1_specialist_input_audit import (
    CROP_SIZE,
    EXPECTED_CHECKPOINT_ROLE,
    EXPECTED_CHECKPOINT_SHA,
    CropScoreV1,
    MeterV5_1R1AuditError,
    RuntimeDigitSpecialistV1,
    arbitrate_digit_probabilities_v1,
    audit_pilot_with_scorer_v1,
    crop_digit_to_64_v1,
    derive_digit_slots_from_full_meter_bbox_v1,
    make_frozen_checkpoint_scorer_v1,
    specialist_input_audit_profile_fingerprint_v1,
)
from st_omr_training.runtime_meter_real_checkpoint_audit_v1 import AuditedCheckpointStateV1


def _sha_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class MeterV5_1R1SpecialistInputAuditTests(unittest.TestCase):
    def test_slot_split_is_single_integer_midpoint_without_overlap(self) -> None:
        upper, lower = derive_digit_slots_from_full_meter_bbox_v1(
            x=5, y=7, w=9, h=11, image_width=40, image_height=40
        )
        self.assertEqual(upper.as_list(), [5, 7, 14, 12])
        self.assertEqual(lower.as_list(), [5, 12, 14, 18])

    def test_slot_split_fails_closed_outside_image(self) -> None:
        with self.assertRaises(MeterV5_1R1AuditError):
            derive_digit_slots_from_full_meter_bbox_v1(
                x=15, y=5, w=10, h=10, image_width=20, image_height=20
            )

    def test_crop_is_exact_gray64_and_does_not_upscale_content(self) -> None:
        image = Image.new("L", (30, 30), 255)
        for x in range(5, 15):
            for y in range(5, 15):
                image.putpixel((x, y), 50)
        upper, _ = derive_digit_slots_from_full_meter_bbox_v1(
            x=5, y=5, w=10, h=20, image_width=30, image_height=30
        )
        crop = crop_digit_to_64_v1(image, upper)
        self.assertEqual(crop.mode, "L")
        self.assertEqual(crop.size, (CROP_SIZE, CROP_SIZE))
        self.assertEqual(min(crop.getdata()), 50)
        self.assertEqual(crop.getpixel((0, 0)), 255)

    def test_conservative_arbitration_does_not_round_up(self) -> None:
        result = arbitrate_digit_probabilities_v1({2: 0.479999, 3: 0.599999, 4: 0.469999})
        self.assertEqual(result["state"], "NO_HIT")
        self.assertEqual(result["passing_digits"], [])
        conflict = arbitrate_digit_probabilities_v1({2: 0.80, 3: 0.20, 4: 0.80})
        self.assertEqual(conflict["state"], "CONFLICT")
        self.assertEqual(conflict["passing_digits"], [2, 4])

    def test_profile_fingerprint_is_stable_sha(self) -> None:
        first = specialist_input_audit_profile_fingerprint_v1()
        second = specialist_input_audit_profile_fingerprint_v1()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def _build_surface(self, root: Path) -> tuple[Path, Path]:
        dataset = root / "METER_V2_1500_PACKAGE_AB_CLEAN"
        annotations_dir = dataset / "annotations"
        annotations_dir.mkdir(parents=True)

        selection_rows: list[dict[str, object]] = []
        annotation_rows: list[dict[str, object]] = []
        index = 0
        intensity_by_digit = {2: 50, 3: 100, 4: 150}
        for meter, numerator_digit in (("2/4", 2), ("3/4", 3), ("4/4", 4)):
            meter_dir = meter.replace("/", "_")
            for local in range(10):
                sample_id = f"{meter_dir}-{local:02d}"
                family_id = f"family-{sample_id}"
                folder = f"sample-{sample_id}"
                rel = Path("train") / meter_dir / folder / "image.png"
                path = dataset / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                image = Image.new("L", (32, 32), 255)
                # Approved full-meter bbox is x=8,y=6,w=10,h=20. Upper/lower
                # halves encode the expected visual digits for the synthetic scorer.
                for x in range(8, 18):
                    for y in range(6, 16):
                        image.putpixel((x, y), intensity_by_digit[numerator_digit])
                    for y in range(16, 26):
                        image.putpixel((x, y), intensity_by_digit[4])
                image.save(path, format="PNG")
                image_sha = _sha_file(path)
                selection_rows.append(
                    {
                        "index": index,
                        "sample_id": sample_id,
                        "family_id": family_id,
                        "meter": meter,
                        "split": "train",
                        "folder": folder,
                        "image_relpath": rel.as_posix(),
                        "image_sha256": image_sha,
                        "image_width": 32,
                        "image_height": 32,
                        "selection_rank": index,
                    }
                )
                annotation_rows.append(
                    {
                        "sample_id": sample_id,
                        "meter": meter,
                        "split": "train",
                        "x": 8,
                        "y": 6,
                        "w": 10,
                        "h": 20,
                        "status": "PASS",
                        "image_sha256": image_sha,
                        "image_width": 32,
                        "image_height": 32,
                        "updated_utc": "2026-08-22T00:00:00Z",
                    }
                )
                index += 1

        selection_path = annotations_dir / "bbox_pilot_30_selection.csv"
        annotation_path = annotations_dir / "bbox_pilot_30.csv"
        _write_csv(selection_path, list(selection_rows[0]), selection_rows)
        _write_csv(annotation_path, list(annotation_rows[0]), annotation_rows)

        evidence = {
            "schema": "st-omr-meter-v5-1-bbox-pilot-result-evidence-v1",
            "stage": "METER V5-1",
            "pilot_result": "PASS",
            "dataset": {
                "name": "METER_V2_1500_PACKAGE_AB_CLEAN",
                "fingerprint_sha256": sha256(b"dataset").hexdigest(),
            },
            "annotation_audit": {
                "annotation_count": 30,
                "pass_count": 30,
                "review_count": 0,
                "mechanical_gate": "PASS",
                "original_pilot_image_binding_preserved": True,
                "annotation_contract_freeze_ready": True,
            },
            "artifacts": {
                "selection_csv": {"sha256": _sha_file(selection_path)},
                "annotation_csv": {"sha256": _sha_file(annotation_path)},
            },
            "safety": {
                "annotation_scope": "train_pilot_30_only",
                "final_holdout_locked": True,
                "training_authorized": False,
                "model_opened": False,
            },
        }
        evidence_path = root / "pilot_evidence.json"
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        return dataset, evidence_path

    @staticmethod
    def _perfect_scorer(crop: Image.Image) -> CropScoreV1:
        minimum = min(crop.getdata())
        if minimum == 50:
            digit = 2
        elif minimum == 100:
            digit = 3
        elif minimum == 150:
            digit = 4
        else:
            raise AssertionError(f"unexpected synthetic crop intensity {minimum}")
        probabilities = {2: 0.01, 3: 0.01, 4: 0.01}
        probabilities[digit] = 0.99
        return CropScoreV1(probabilities=probabilities, replay_stable=True)

    def test_exact_30_sample_surface_can_pass_strict_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            dataset, evidence = self._build_surface(Path(temp))
            result = audit_pilot_with_scorer_v1(
                dataset_root=dataset,
                pilot_evidence_path=evidence,
                score_crop=self._perfect_scorer,
            )
            self.assertEqual(result["decision"], "PASS_SCALE_ANNOTATION")
            self.assertEqual(result["aggregate"]["correct_samples"], 30)
            self.assertEqual(result["aggregate"]["correct_slots"], 60)
            self.assertEqual(result["surface"]["validation_opened"], False)
            self.assertEqual(result["surface"]["final_holdout_opened"], False)

    def test_one_conflict_holds_entire_input_contract(self) -> None:
        calls = {"count": 0}

        def scorer(crop: Image.Image) -> CropScoreV1:
            calls["count"] += 1
            if calls["count"] == 1:
                return CropScoreV1(probabilities={2: 0.99, 3: 0.01, 4: 0.99}, replay_stable=True)
            return self._perfect_scorer(crop)

        with tempfile.TemporaryDirectory() as temp:
            dataset, evidence = self._build_surface(Path(temp))
            result = audit_pilot_with_scorer_v1(
                dataset_root=dataset,
                pilot_evidence_path=evidence,
                score_crop=scorer,
            )
            self.assertEqual(result["decision"], "HOLD_INPUT_CONTRACT")
            self.assertEqual(result["aggregate"]["slot_outcomes"]["CONFLICT"], 1)

    def test_changed_annotation_bytes_fail_parent_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            dataset, evidence = self._build_surface(Path(temp))
            annotation = dataset / "annotations" / "bbox_pilot_30.csv"
            annotation.write_text(annotation.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(MeterV5_1R1AuditError):
                audit_pilot_with_scorer_v1(
                    dataset_root=dataset,
                    pilot_evidence_path=evidence,
                    score_crop=self._perfect_scorer,
                )

    def test_checkpoint_runtime_mirror_is_deterministic_on_synthetic_states(self) -> None:
        audited: dict[int, AuditedCheckpointStateV1] = {}
        for digit in (2, 3, 4):
            model = RuntimeDigitSpecialistV1()
            zero_state = {name: torch.zeros_like(tensor) for name, tensor in model.state_dict().items()}
            audited[digit] = AuditedCheckpointStateV1(
                role=EXPECTED_CHECKPOINT_ROLE[digit],
                checkpoint_sha256=EXPECTED_CHECKPOINT_SHA[digit],
                byte_length=1,
                model_state=zero_state,
            )
        scorer = make_frozen_checkpoint_scorer_v1(audited)
        crop = Image.new("L", (64, 64), 255)
        score = scorer(crop)
        self.assertTrue(score.replay_stable)
        self.assertEqual(score.probabilities, {2: 0.5, 3: 0.5, 4: 0.5})
        arbitration = arbitrate_digit_probabilities_v1(score.probabilities)
        self.assertEqual(arbitration["state"], "CONFLICT")
        self.assertEqual(arbitration["passing_digits"], [2, 4])


if __name__ == "__main__":
    unittest.main()
