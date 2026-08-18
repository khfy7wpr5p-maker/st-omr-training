"""Run the Meter V2 GT-slot upper-bound on D10 VALIDATION only.

This is intentionally NOT a runtime-localization result. Positive meters use the
frozen M4A/M2 ground-truth digit bboxes; ``none`` false-presence diagnostics use
the frozen M3-C2 negative anchor slots. The purpose is to test the already-frozen
2-AI/3-AI/4-AI models jointly before spending effort on a runtime digit localizer.

Writes only to /content/meter-v2-gt-slot-upper-bound-v1 by default.
Never reads TEST records and never trains.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn

from st_omr_training.meter_v2_deterministic_composer_v1 import compose_meter_v2, MeterBox
from st_omr_training.meter_v2_digit_crop_adapter_v1 import crop_meter_digit_to_64_v1
from st_omr_training.meter_v2_joint_digit_arbitration_v1 import (
    MeterDigitSlotProbabilities,
    digit_observation_from_probabilities_v1,
)
from st_omr_training.meter_v2_presence_shadow_v1 import (
    M3B_PRESENCE_CACHE_SHA256,
    presence_from_m3b_score_v1,
)

EXPECTED_TORCH = "2.13.0+cpu"
EXPECTED_D10_MANIFEST_SHA = "6927e1bcc5251257a983a306e2f1875c9515f97c6724a8fe9f24382c6ff30db4"
EXPECTED_M4A_MANIFEST_SHA = "ebda40dae10f0d6490df2c7728dab5cc2cc6f58b5420b198dfbb441a99ecebb9"
EXPECTED_CHECKPOINTS = {
    2: "92b985d989e4338e3ae39b0a984879f4188be32c0d281390839117e1e9a715fa",
    3: "5ee45faf2efe0e2c83dbad716736d7ae16ad7251730431d368c10c4574836485",
    4: "dcd582b60b39e65798aa77aacea3cc797cd7513b7925151f0573be4aec6af43f",
}

DRIVE = Path("/content/gdrive_r2/MyDrive")
D10 = DRIVE / "ST-OMR-D10" / "stage7d10-authoritative-562c8fcfabf1b41573f1ef591d88ae65335ce16a"
M4A = DRIVE / "ST-OMR-METER-SPECIALISTS" / "m4a-234-digit-specialist-dataset-freeze-v2"
M3B = DRIVE / "ST-OMR-METER-SPECIALISTS" / "m3b-v4-target213-resume-v1"
CHECKPOINT_PATHS = {
    2: DRIVE / "ST-OMR-METER-SPECIALISTS" / "m4c5-2ai-train-hard-none-v1" / "checkpoints" / "step-0768.pt",
    3: DRIVE / "ST-OMR-METER-SPECIALISTS" / "m4c2-none75-negative-sampling-v1" / "checkpoints" / "3-AI" / "step-0768.pt",
    4: DRIVE / "ST-OMR-METER-SPECIALISTS" / "m4c2-none75-negative-sampling-v1" / "checkpoints" / "4-AI" / "step-0768.pt",
}
OUT = Path("/content/meter-v2-gt-slot-upper-bound-v1")


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
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
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


def score_slot(models, image: Image.Image, row):
    array = crop_meter_digit_to_64_v1(image, row["bbox"])
    tensor = torch.from_numpy(np.asarray(array).copy()).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    scores = {}
    with torch.no_grad():
        for digit in (2, 3, 4):
            scores[digit] = float(torch.sigmoid(models[digit](tensor))[0].item())
    bbox = tuple(float(x) for x in row["bbox"])
    slot = MeterDigitSlotProbabilities(
        slot_id=str(row["crop_id"]),
        bbox=MeterBox(*bbox),
        score_2=scores[2],
        score_3=scores[3],
        score_4=scores[4],
    )
    return slot, digit_observation_from_probabilities_v1(slot)


def main() -> None:
    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(f"PyTorch mismatch: expected {EXPECTED_TORCH}, got {torch.__version__}")
    if not DRIVE.is_dir():
        raise RuntimeError("Google Drive is not mounted at /content/gdrive_r2")

    d10_manifest_path = D10 / "manifest.json"
    m4a_manifest_path = M4A / "dataset-manifest.json"
    m3b_cache_path = M3B / "validation-proposal-cache.json"
    for path in (d10_manifest_path, m4a_manifest_path, m3b_cache_path):
        if not path.is_file():
            raise RuntimeError(f"required evidence missing: {path}")

    if file_sha(d10_manifest_path) != EXPECTED_D10_MANIFEST_SHA:
        raise RuntimeError("D10 manifest SHA mismatch")
    if file_sha(m4a_manifest_path) != EXPECTED_M4A_MANIFEST_SHA:
        raise RuntimeError("M4A manifest SHA mismatch")
    if file_sha(m3b_cache_path) != M3B_PRESENCE_CACHE_SHA256:
        raise RuntimeError("M3B cache SHA mismatch")

    d10_manifest = load_json(d10_manifest_path)
    m4a_manifest = load_json(m4a_manifest_path)
    m3b_cache = load_json(m3b_cache_path)

    if d10_manifest.get("test_records") != 0:
        raise RuntimeError("D10 TEST is not sealed")
    if m4a_manifest.get("safety", {}).get("test_opened") is not False:
        raise RuntimeError("M4A TEST safety flag is not false")
    if m3b_cache.get("test_opened") is not False:
        raise RuntimeError("M3B cache TEST safety flag is not false")

    meter_records = {
        row["record_id"]: row
        for row in d10_manifest["records"]
        if row.get("kind") == "meter" and row.get("split") == "validation"
    }
    if len(meter_records) != 1224:
        raise RuntimeError(f"expected 1224 validation Meter records, got {len(meter_records)}")
    if any(row.get("split") == "test" for row in d10_manifest["records"]):
        raise RuntimeError("TEST record present in D10 manifest")

    m4a_rows = defaultdict(list)
    for row in m4a_manifest["records"]:
        if row.get("split") == "test":
            raise RuntimeError("TEST record present in M4A manifest")
        if row.get("split") == "validation":
            m4a_rows[row["source_record_id"]].append(row)

    cache_rows = {row["record_id"]: row for row in m3b_cache["records"]}
    if set(cache_rows) != set(meter_records):
        raise RuntimeError("M3B cache record IDs do not exactly match D10 validation Meter IDs")

    models = {digit: load_specialist(digit) for digit in (2, 3, 4)}

    exact = 0
    statuses = Counter()
    expected_counts = Counter()
    predicted_counts = Counter()
    slot_outcomes = Counter()
    mismatch_examples = []
    processed = 0

    for record_id in sorted(meter_records):
        record = meter_records[record_id]
        cache = cache_rows[record_id]
        expected = str(cache["true_class"])
        expected_counts[expected] += 1
        presence = presence_from_m3b_score_v1(float(cache["presence_score"]))

        observations = []
        # Presence is a gate. Only a visual-present proposal invokes digit specialists.
        if presence.status == "accepted" and presence.present:
            image_path = D10 / record["image_path"]
            if not image_path.is_file():
                raise RuntimeError(f"missing validation image: {image_path}")
            with Image.open(image_path) as opened:
                image = opened.convert("L")
                rows = m4a_rows.get(record_id, [])
                if expected != "none":
                    rows = [row for row in rows if row.get("source_type") == "M2_TRUE_DIGIT_ZONE"]
                    if len(rows) != 2:
                        raise RuntimeError(f"positive record {record_id} does not have exactly two GT digit rows")
                else:
                    rows = [row for row in rows if row.get("source_type") == "M3C2_DETERMINISTIC_NONE_SLOT"]
                    if not rows:
                        raise RuntimeError(f"none record {record_id} has no frozen negative anchor slots")

                for row in sorted(rows, key=lambda item: str(item["crop_id"])):
                    _, observation = score_slot(models, image, row)
                    if observation is None:
                        slot_outcomes["no_digit"] += 1
                    else:
                        slot_outcomes[observation.status] += 1
                        observations.append(observation)

        result = compose_meter_v2(presence, tuple(observations))
        statuses[result.status] += 1
        predicted = result.meter_class if result.status == "accepted" else result.status.upper()
        predicted_counts[predicted] += 1
        if result.status == "accepted" and result.meter_class == expected:
            exact += 1
        elif len(mismatch_examples) < 100:
            mismatch_examples.append(
                {
                    "record_id": record_id,
                    "expected": expected,
                    "presence_score": cache["presence_score"],
                    "result_status": result.status,
                    "predicted_meter": result.meter_class,
                    "reasons": list(result.reasons),
                    "observations": [
                        {
                            "id": obs.observation_id,
                            "status": obs.status,
                            "digit": obs.digit,
                            "confidence_milli": obs.confidence_milli,
                            "reasons": list(obs.reasons),
                        }
                        for obs in observations
                    ],
                }
            )

        processed += 1
        if processed % 128 == 0 or processed == 1224:
            print(f"VALIDATION {processed}/1224", flush=True)

    report = {
        "stage": "meter-v2-gt-slot-upper-bound-v1",
        "state": "COMPLETE_DIAGNOSTIC",
        "claim_boundary": "GT_SLOT_UPPER_BOUND_NOT_RUNTIME_LOCALIZATION",
        "validation_records": 1224,
        "test_records": 0,
        "exact_meter_match": exact,
        "exact_meter_accuracy": exact / 1224,
        "status_counts": dict(statuses),
        "expected_class_counts": dict(expected_counts),
        "predicted_counts": dict(predicted_counts),
        "slot_outcomes": dict(slot_outcomes),
        "mismatch_examples_first_100": mismatch_examples,
        "source_sha256": {
            "d10_manifest": EXPECTED_D10_MANIFEST_SHA,
            "m4a_manifest": EXPECTED_M4A_MANIFEST_SHA,
            "m3b_cache": M3B_PRESENCE_CACHE_SHA256,
            "2-AI": EXPECTED_CHECKPOINTS[2],
            "3-AI": EXPECTED_CHECKPOINTS[3],
            "4-AI": EXPECTED_CHECKPOINTS[4],
        },
        "safety": {
            "training_started": False,
            "optimizer_steps_added": 0,
            "test_opened": False,
            "drive_write": False,
            "resolver_wiring": False,
            "production_promotion": False,
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("=" * 72)
    print("METER V2 GT-SLOT UPPER BOUND COMPLETE")
    print("EXACT MATCH :", f"{exact}/1224 = {exact/1224:.6f}")
    print("STATUSES    :", dict(statuses))
    print("SLOT OUTCOME:", dict(slot_outcomes))
    print("TEST        : CLOSED")
    print("RUNTIME SLOT: NOT CLAIMED")
    print("REPORT      :", report_path)


if __name__ == "__main__":
    main()
