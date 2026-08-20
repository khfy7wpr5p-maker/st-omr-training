"""Validate TRAIN-derived Meter V2 runtime digit slots on D10 VALIDATION only.

This is the first true runtime-slot shadow evaluation: positive digit boxes are
produced from Meter ROI pixels + measure_number by the frozen TRAIN-derived
localizer, never from validation ground truth and never from D11 bbox geometry.

No training, threshold tuning, TEST access, Drive writes, Resolver wiring, or
production promotion. Output is written only under /content.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image
import torch
from torch import nn

from st_omr_training.meter_v2_deterministic_composer_v1 import (
    ACCEPTED, AMBIGUOUS, REJECTED, compose_meter_v2,
)
from st_omr_training.meter_v2_digit_crop_adapter_v1 import crop_meter_digit_to_64_v1
from st_omr_training.meter_v2_joint_digit_arbitration_v1 import (
    MeterDigitSlotProbabilities, digit_observation_from_probabilities_v1,
)
from st_omr_training.meter_v2_presence_shadow_v1 import (
    M3B_PRESENCE_CACHE_SHA256, presence_from_m3b_score_v1,
)
from st_omr_training.meter_v2_runtime_slot_localizer_v1 import (
    meter_v2_runtime_slot_profile_fingerprint,
    propose_meter_v2_runtime_digit_modes_v1,
)

EXPECTED_TORCH = "2.13.0+cpu"
EXPECTED_D10_MANIFEST_SHA = "6927e1bcc5251257a983a306e2f1875c9515f97c6724a8fe9f24382c6ff30db4"
EXPECTED_CHECKPOINTS = {
    2: "92b985d989e4338e3ae39b0a984879f4188be32c0d281390839117e1e9a715fa",
    3: "5ee45faf2efe0e2c83dbad716736d7ae16ad7251730431d368c10c4574836485",
    4: "dcd582b60b39e65798aa77aacea3cc797cd7513b7925151f0573be4aec6af43f",
}
DRIVE = Path("/content/gdrive_r2/MyDrive")
D10 = DRIVE / "ST-OMR-D10" / "stage7d10-authoritative-562c8fcfabf1b41573f1ef591d88ae65335ce16a"
M3B = DRIVE / "ST-OMR-METER-SPECIALISTS" / "m3b-v4-target213-resume-v1"
CHECKPOINT_PATHS = {
    2: DRIVE / "ST-OMR-METER-SPECIALISTS" / "m4c5-2ai-train-hard-none-v1" / "checkpoints" / "step-0768.pt",
    3: DRIVE / "ST-OMR-METER-SPECIALISTS" / "m4c2-none75-negative-sampling-v1" / "checkpoints" / "3-AI" / "step-0768.pt",
    4: DRIVE / "ST-OMR-METER-SPECIALISTS" / "m4c2-none75-negative-sampling-v1" / "checkpoints" / "4-AI" / "step-0768.pt",
}
OUT = Path("/content/meter-v2-runtime-slot-validation-v1")


def file_sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class TinyBinaryCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(torch.flatten(self.features(x), 1)).squeeze(1)


def load_specialist(digit: int) -> TinyBinaryCNN:
    path = CHECKPOINT_PATHS[digit]
    if not path.is_file():
        raise RuntimeError(f"missing {digit}-AI checkpoint: {path}")
    actual_sha = file_sha(path)
    if actual_sha != EXPECTED_CHECKPOINTS[digit]:
        raise RuntimeError(f"{digit}-AI checkpoint SHA mismatch: {actual_sha}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("step") != 768:
        raise RuntimeError(f"{digit}-AI checkpoint step is not 768")
    model = TinyBinaryCNN().cpu()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def score_box(models, image: Image.Image, slot_id: str, box):
    crop = crop_meter_digit_to_64_v1(image, (box.x0, box.y0, box.x1, box.y1))
    tensor = (
        torch.frombuffer(bytearray(crop.tobytes()), dtype=torch.uint8)
        .clone().reshape(1, 1, 64, 64).to(dtype=torch.float32) / 255.0
    )
    with torch.no_grad():
        scores = {d: float(torch.sigmoid(models[d](tensor))[0].item()) for d in (2, 3, 4)}
    slot = MeterDigitSlotProbabilities(
        slot_id=slot_id, bbox=box,
        score_2=scores[2], score_3=scores[3], score_4=scores[4],
    )
    return scores, digit_observation_from_probabilities_v1(slot)


@dataclass(frozen=True)
class AggregateResult:
    status: str
    meter_class: str | None
    reasons: tuple[str, ...]
    accepted_mode_index: int | None = None
    accepted_low_support: bool = False


def aggregate_modes(mode_results):
    accepted = [(proposal, result) for proposal, result in mode_results if result.status == ACCEPTED]
    if len(accepted) == 1:
        proposal, result = accepted[0]
        return AggregateResult(ACCEPTED, result.meter_class, (), proposal.mode_index, proposal.low_support)
    if len(accepted) > 1:
        return AggregateResult(AMBIGUOUS, None, ("METER_RUNTIME_MULTIPLE_ACCEPTED_MODES",))
    if not mode_results:
        return AggregateResult(AMBIGUOUS, None, ("METER_RUNTIME_NO_SLOT_MODE",))
    if any(result.status == AMBIGUOUS for _, result in mode_results):
        return AggregateResult(AMBIGUOUS, None, ("METER_RUNTIME_MODE_AMBIGUOUS",))
    return AggregateResult(REJECTED, None, ("METER_RUNTIME_ALL_MODES_REJECTED",))


def class_metrics(rows, label):
    tp = sum(1 for expected, predicted in rows if expected == label and predicted == label)
    fp = sum(1 for expected, predicted in rows if expected != label and predicted == label)
    fn = sum(1 for expected, predicted in rows if expected == label and predicted != label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(f"PyTorch mismatch: expected {EXPECTED_TORCH}, got {torch.__version__}")
    if not DRIVE.is_dir():
        raise RuntimeError("Google Drive is not mounted at /content/gdrive_r2")

    d10_manifest_path = D10 / "manifest.json"
    m3b_cache_path = M3B / "validation-proposal-cache.json"
    if file_sha(d10_manifest_path) != EXPECTED_D10_MANIFEST_SHA:
        raise RuntimeError("D10 manifest SHA mismatch")
    if file_sha(m3b_cache_path) != M3B_PRESENCE_CACHE_SHA256:
        raise RuntimeError("M3B cache SHA mismatch")

    d10_manifest = load_json(d10_manifest_path)
    m3b_cache = load_json(m3b_cache_path)
    if d10_manifest.get("test_records") != 0:
        raise RuntimeError("D10 TEST is not sealed")
    if m3b_cache.get("test_opened") is not False:
        raise RuntimeError("M3B cache TEST safety flag is not false")
    if any(row.get("split") == "test" for row in d10_manifest["records"]):
        raise RuntimeError("TEST record present in D10 manifest")

    meter_records = {
        row["record_id"]: row for row in d10_manifest["records"]
        if row.get("kind") == "meter" and row.get("split") == "validation"
    }
    if len(meter_records) != 1224:
        raise RuntimeError(f"expected 1224 validation Meter records, got {len(meter_records)}")
    cache_rows = {row["record_id"]: row for row in m3b_cache["records"]}
    if set(cache_rows) != set(meter_records):
        raise RuntimeError("M3B cache record IDs do not exactly match D10 validation Meter IDs")

    models = {digit: load_specialist(digit) for digit in (2, 3, 4)}
    statuses = Counter(); expected_counts = Counter(); predicted_counts = Counter()
    candidate_mode_counts = Counter(); slot_outcomes = Counter()
    low_support_accepted = 0; exact = 0; eval_rows = []; mismatches = []

    for index, record_id in enumerate(sorted(meter_records), start=1):
        record = meter_records[record_id]; cache = cache_rows[record_id]
        expected = str(cache["true_class"]); expected_counts[expected] += 1
        presence = presence_from_m3b_score_v1(float(cache["presence_score"]))

        if presence.status != ACCEPTED or not presence.present:
            base = compose_meter_v2(presence, ())
            final = AggregateResult(base.status, base.meter_class, base.reasons)
            candidate_mode_counts[0] += 1
        else:
            image_path = D10 / record["image_path"]
            if not image_path.is_file():
                raise RuntimeError(f"missing validation image: {image_path}")
            with Image.open(image_path) as opened:
                image = opened.convert("L")
            proposals = propose_meter_v2_runtime_digit_modes_v1(
                image, measure_number=int(record["measure_number"])
            )
            candidate_mode_counts[len(proposals)] += 1
            mode_results = []
            for proposal in proposals:
                observations = []
                for role, box in (("num", proposal.numerator_bbox), ("den", proposal.denominator_bbox)):
                    _, observation = score_box(models, image, f"m{proposal.mode_index}-{role}", box)
                    if observation is None:
                        slot_outcomes["no_digit"] += 1
                    else:
                        slot_outcomes[observation.status] += 1
                        observations.append(observation)
                mode_results.append((proposal, compose_meter_v2(presence, tuple(observations))))
            final = aggregate_modes(mode_results)

        statuses[final.status] += 1
        predicted = final.meter_class if final.status == ACCEPTED else final.status.upper()
        predicted_counts[predicted] += 1
        eval_rows.append((expected, final.meter_class if final.status == ACCEPTED else None))
        if final.status == ACCEPTED and final.meter_class == expected:
            exact += 1
            if final.accepted_low_support:
                low_support_accepted += 1
        elif len(mismatches) < 100:
            mismatches.append({
                "record_id": record_id, "measure_number": record["measure_number"],
                "expected": expected, "presence_score": cache["presence_score"],
                "status": final.status, "predicted_meter": final.meter_class,
                "reasons": list(final.reasons),
                "accepted_mode_index": final.accepted_mode_index,
                "accepted_low_support": final.accepted_low_support,
            })
        if index % 128 == 0 or index == 1224:
            print(f"VALIDATION {index}/1224", flush=True)

    per_class = {label: class_metrics(eval_rows, label) for label in ("none", "2/4", "3/4", "4/4")}
    report = {
        "stage": "meter-v2-runtime-slot-validation-v1",
        "state": "COMPLETE_DIAGNOSTIC",
        "claim_boundary": "TRAIN_DERIVED_RUNTIME_SLOTS_VALIDATION_SHADOW_ONLY",
        "validation_records": 1224, "test_records": 0,
        "exact_meter_match": exact, "exact_meter_accuracy": exact / 1224,
        "status_counts": dict(statuses), "expected_class_counts": dict(expected_counts),
        "predicted_counts": dict(predicted_counts), "per_class": per_class,
        "candidate_mode_counts": dict(candidate_mode_counts), "slot_outcomes": dict(slot_outcomes),
        "low_support_accepted_correct": low_support_accepted,
        "mismatch_examples_first_100": mismatches,
        "runtime_slot_profile_fingerprint": meter_v2_runtime_slot_profile_fingerprint(),
        "source_sha256": {
            "d10_manifest": EXPECTED_D10_MANIFEST_SHA, "m3b_cache": M3B_PRESENCE_CACHE_SHA256,
            "2-AI": EXPECTED_CHECKPOINTS[2], "3-AI": EXPECTED_CHECKPOINTS[3], "4-AI": EXPECTED_CHECKPOINTS[4],
        },
        "safety": {
            "training_started": False, "optimizer_steps_added": 0, "threshold_tuning": False,
            "test_opened": False, "drive_write": False, "resolver_wiring": False,
            "production_promotion": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("=" * 72)
    print("METER V2 RUNTIME-SLOT VALIDATION COMPLETE")
    print("EXACT MATCH :", f"{exact}/1224 = {exact/1224:.6f}")
    print("STATUSES    :", dict(statuses))
    print("CANDIDATES  :", dict(candidate_mode_counts))
    print("SLOT OUTCOME:", dict(slot_outcomes))
    print("LOW SUPPORT :", low_support_accepted)
    print("TEST        : CLOSED")
    print("TRAINING    : NONE")
    print("RESOLVER    : NOT WIRED")
    print("REPORT      :", path)


if __name__ == "__main__":
    main()
