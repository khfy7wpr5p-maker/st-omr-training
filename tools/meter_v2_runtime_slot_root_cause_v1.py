"""Post-hoc root-cause audit for Meter V2 runtime-slot VALIDATION failures.

This script is diagnostic only. It reruns the already-frozen shadow pipeline on the
1,224 D10 VALIDATION Meter records and uses M4A VALIDATION GT digit boxes only
AFTER runtime proposals have been produced, solely to classify failure causes.

It never feeds validation GT geometry into runtime predictions, never tunes
thresholds or geometry from VALIDATION, never reads TEST, never trains, never
writes Drive, and never wires the runtime Resolver.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median

from PIL import Image
import torch
from torch import nn

from st_omr_training.meter_v2_deterministic_composer_v1 import (
    ACCEPTED,
    AMBIGUOUS,
    REJECTED,
    MeterBox,
    compose_meter_v2,
)
from st_omr_training.meter_v2_digit_crop_adapter_v1 import crop_meter_digit_to_64_v1
from st_omr_training.meter_v2_joint_digit_arbitration_v1 import (
    MeterDigitSlotProbabilities,
    digit_observation_from_probabilities_v1,
)
from st_omr_training.meter_v2_presence_shadow_v1 import (
    M3B_PRESENCE_CACHE_SHA256,
    presence_from_m3b_score_v1,
)
from st_omr_training.meter_v2_runtime_slot_localizer_v1 import (
    propose_meter_v2_runtime_digit_modes_v1,
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
OUT = Path("/content/meter-v2-runtime-slot-root-cause-v1")


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
    actual = file_sha(path)
    if actual != EXPECTED_CHECKPOINTS[digit]:
        raise RuntimeError(f"{digit}-AI checkpoint SHA mismatch: {actual}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("step") != 768:
        raise RuntimeError(f"{digit}-AI checkpoint step is not 768")
    model = TinyBinaryCNN().cpu()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def score_box(models, image: Image.Image, slot_id: str, box: MeterBox):
    crop = crop_meter_digit_to_64_v1(image, (box.x0, box.y0, box.x1, box.y1))
    tensor = (
        torch.frombuffer(bytearray(crop.tobytes()), dtype=torch.uint8)
        .clone().reshape(1, 1, 64, 64).to(dtype=torch.float32) / 255.0
    )
    with torch.no_grad():
        scores = {d: float(torch.sigmoid(models[d](tensor))[0].item()) for d in (2, 3, 4)}
    slot = MeterDigitSlotProbabilities(
        slot_id=slot_id,
        bbox=box,
        score_2=scores[2],
        score_3=scores[3],
        score_4=scores[4],
    )
    return scores, digit_observation_from_probabilities_v1(slot)


@dataclass(frozen=True)
class AggregateResult:
    status: str
    meter_class: str | None
    reasons: tuple[str, ...]


def aggregate_modes(mode_results):
    accepted = [result for _proposal, result in mode_results if result.status == ACCEPTED]
    if len(accepted) == 1:
        return AggregateResult(ACCEPTED, accepted[0].meter_class, ())
    if len(accepted) > 1:
        return AggregateResult(AMBIGUOUS, None, ("METER_RUNTIME_MULTIPLE_ACCEPTED_MODES",))
    if not mode_results:
        return AggregateResult(AMBIGUOUS, None, ("METER_RUNTIME_NO_SLOT_MODE",))
    if any(result.status == AMBIGUOUS for _proposal, result in mode_results):
        return AggregateResult(AMBIGUOUS, None, ("METER_RUNTIME_MODE_AMBIGUOUS",))
    return AggregateResult(REJECTED, None, ("METER_RUNTIME_ALL_MODES_REJECTED",))


def box_from_row(row) -> MeterBox:
    x0, y0, x1, y1 = (float(v) for v in row["bbox"])
    return MeterBox(x0, y0, x1, y1)


def box_iou(a: MeterBox, b: MeterBox) -> float:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, a.x1 - a.x0) * max(0.0, a.y1 - a.y0)
    area_b = max(0.0, b.x1 - b.x0) * max(0.0, b.y1 - b.y0)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def center_delta(a: MeterBox, b: MeterBox) -> tuple[float, float]:
    acx = (a.x0 + a.x1) / 2
    acy = (a.y0 + a.y1) / 2
    bcx = (b.x0 + b.x1) / 2
    bcy = (b.y0 + b.y1) / 2
    return acx - bcx, acy - bcy


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * fraction))
    return float(ordered[idx])


def obs_digit(observation):
    if observation is None or observation.status != ACCEPTED:
        return None
    return int(observation.digit)


def score_gt_pair(models, image, rows, presence):
    observations = []
    role_info = {}
    for row in sorted(rows, key=lambda r: 0 if r["role"] == "numerator" else 1):
        box = box_from_row(row)
        scores, obs = score_box(models, image, "gt-" + str(row["role"]), box)
        role_info[str(row["role"])] = {
            "expected_digit": int(row["digit_label"]),
            "observed_digit": obs_digit(obs),
            "observation_status": "no_digit" if obs is None else obs.status,
            "scores": scores,
        }
        if obs is not None:
            observations.append(obs)
    result = compose_meter_v2(presence, tuple(observations))
    return result, role_info


def main() -> None:
    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(f"PyTorch mismatch: expected {EXPECTED_TORCH}, got {torch.__version__}")
    if not DRIVE.is_dir():
        raise RuntimeError("Google Drive is not mounted at /content/gdrive_r2")

    d10_path = D10 / "manifest.json"
    m4a_path = M4A / "dataset-manifest.json"
    m3b_path = M3B / "validation-proposal-cache.json"
    if file_sha(d10_path) != EXPECTED_D10_MANIFEST_SHA:
        raise RuntimeError("D10 manifest SHA mismatch")
    if file_sha(m4a_path) != EXPECTED_M4A_MANIFEST_SHA:
        raise RuntimeError("M4A manifest SHA mismatch")
    if file_sha(m3b_path) != M3B_PRESENCE_CACHE_SHA256:
        raise RuntimeError("M3B cache SHA mismatch")

    d10 = load_json(d10_path)
    m4a = load_json(m4a_path)
    m3b = load_json(m3b_path)
    if d10.get("test_records") != 0:
        raise RuntimeError("D10 TEST is not sealed")
    if m4a.get("safety", {}).get("test_opened") is not False:
        raise RuntimeError("M4A TEST safety flag is not false")
    if m3b.get("test_opened") is not False:
        raise RuntimeError("M3B cache TEST safety flag is not false")
    if any(row.get("split") == "test" for row in d10["records"]):
        raise RuntimeError("TEST record present in D10 manifest")
    if any(row.get("split") == "test" for row in m4a["records"]):
        raise RuntimeError("TEST record present in M4A manifest")

    records = {
        row["record_id"]: row
        for row in d10["records"]
        if row.get("kind") == "meter" and row.get("split") == "validation"
    }
    if len(records) != 1224:
        raise RuntimeError(f"expected 1224 validation Meter records, got {len(records)}")
    cache = {row["record_id"]: row for row in m3b["records"]}
    if set(cache) != set(records):
        raise RuntimeError("M3B cache does not exactly match D10 VALIDATION")

    gt_rows = defaultdict(list)
    for row in m4a["records"]:
        if row.get("split") == "validation":
            gt_rows[row["source_record_id"]].append(row)

    models = {d: load_specialist(d) for d in (2, 3, 4)}

    root = Counter()
    runtime_status = Counter()
    runtime_exact = 0
    per_measure = defaultdict(lambda: Counter(total=0, correct=0, fail=0))
    role_fail = Counter()
    slot_obs = Counter()
    mismatch_rows = []
    iou_correct = []
    iou_fail = []
    dx_abs_correct = []
    dx_abs_fail = []
    accepted_wrong = 0

    for idx, record_id in enumerate(sorted(records), start=1):
        record = records[record_id]
        c = cache[record_id]
        expected = str(c["true_class"])
        presence = presence_from_m3b_score_v1(float(c["presence_score"]))
        measure_number = int(record["measure_number"])
        per_measure[measure_number]["total"] += 1

        if presence.status != ACCEPTED or not presence.present:
            base = compose_meter_v2(presence, ())
            final = AggregateResult(base.status, base.meter_class, base.reasons)
            mode_details = []
        else:
            image_path = D10 / record["image_path"]
            if not image_path.is_file():
                raise RuntimeError(f"missing validation image: {image_path}")
            with Image.open(image_path) as opened:
                image = opened.convert("L")

            proposals = propose_meter_v2_runtime_digit_modes_v1(
                image, measure_number=measure_number
            )
            mode_results = []
            mode_details = []
            positive_gt = {}
            if expected != "none":
                posrows = [
                    r for r in gt_rows[record_id]
                    if r.get("source_type") == "M2_TRUE_DIGIT_ZONE"
                ]
                positive_gt = {str(r["role"]): r for r in posrows}

            for proposal in proposals:
                observations = []
                detail = {
                    "mode_index": proposal.mode_index,
                    "low_support": proposal.low_support,
                    "roles": {},
                }
                for role, box in (
                    ("numerator", proposal.numerator_bbox),
                    ("denominator", proposal.denominator_bbox),
                ):
                    scores, obs = score_box(
                        models, image, f"rt-m{proposal.mode_index}-{role}", box
                    )
                    status = "no_digit" if obs is None else obs.status
                    slot_obs[status] += 1
                    if obs is not None:
                        observations.append(obs)
                    rd = {
                        "status": status,
                        "digit": obs_digit(obs),
                        "scores": scores,
                        "bbox": [box.x0, box.y0, box.x1, box.y1],
                    }
                    if role in positive_gt:
                        gtbox = box_from_row(positive_gt[role])
                        dx, dy = center_delta(box, gtbox)
                        rd.update(
                            {
                                "expected_digit": int(positive_gt[role]["digit_label"]),
                                "iou_to_gt": box_iou(box, gtbox),
                                "center_dx_to_gt": dx,
                                "center_dy_to_gt": dy,
                            }
                        )
                    detail["roles"][role] = rd
                result = compose_meter_v2(presence, tuple(observations))
                detail["result_status"] = result.status
                detail["meter_class"] = result.meter_class
                mode_results.append((proposal, result))
                mode_details.append(detail)
            final = aggregate_modes(mode_results)

        runtime_status[final.status] += 1
        is_correct = final.status == ACCEPTED and final.meter_class == expected
        if is_correct:
            runtime_exact += 1
            per_measure[measure_number]["correct"] += 1
        else:
            per_measure[measure_number]["fail"] += 1

        category = "CORRECT"
        diag = {}

        if not is_correct:
            if expected == "none":
                if presence.status == ACCEPTED and presence.present:
                    category = "PRESENCE_FALSE_POSITIVE"
                else:
                    category = "NONE_OTHER"
            else:
                if presence.status != ACCEPTED or not presence.present:
                    category = "PRESENCE_FALSE_NEGATIVE"
                else:
                    posrows = [
                        r for r in gt_rows[record_id]
                        if r.get("source_type") == "M2_TRUE_DIGIT_ZONE"
                    ]
                    if len(posrows) != 2:
                        raise RuntimeError(
                            f"positive record {record_id} lacks exactly 2 GT digit rows"
                        )
                    gt_result, gt_role = score_gt_pair(models, image, posrows, presence)
                    gt_ok = gt_result.status == ACCEPTED and gt_result.meter_class == expected

                    if not gt_ok:
                        category = "GT_SLOT_MODEL_OR_ARBITRATION_LIMIT"
                        diag["gt_role"] = gt_role
                    elif not mode_details:
                        category = "RUNTIME_NO_SLOT_MODE"
                    else:
                        accepted_modes = [
                            d for d in mode_details if d["result_status"] == ACCEPTED
                        ]
                        if len(accepted_modes) > 1:
                            category = "RUNTIME_MULTIPLE_ACCEPTED_MODES"
                        elif (
                            len(accepted_modes) == 1
                            and accepted_modes[0]["meter_class"] != expected
                        ):
                            category = "RUNTIME_WRONG_METER"
                            accepted_wrong += 1
                        else:
                            for d in mode_details:
                                vals = [
                                    float(d["roles"][role].get("iou_to_gt", 0.0))
                                    for role in ("numerator", "denominator")
                                ]
                                d["mean_iou_to_gt"] = sum(vals) / 2.0
                            best = max(
                                mode_details, key=lambda d: d["mean_iou_to_gt"]
                            )
                            num = best["roles"]["numerator"]
                            den = best["roles"]["denominator"]
                            num_ok = (
                                num.get("digit") == num.get("expected_digit")
                                and num.get("status") == ACCEPTED
                            )
                            den_ok = (
                                den.get("digit") == den.get("expected_digit")
                                and den.get("status") == ACCEPTED
                            )
                            if not num_ok and den_ok:
                                category = "RUNTIME_SLOT_MISS_NUMERATOR"
                                role_fail["numerator_only"] += 1
                            elif num_ok and not den_ok:
                                category = "RUNTIME_SLOT_MISS_DENOMINATOR"
                                role_fail["denominator_only"] += 1
                            elif not num_ok and not den_ok:
                                category = "RUNTIME_SLOT_MISS_BOTH"
                                role_fail["both"] += 1
                            else:
                                category = "RUNTIME_MODE_COMPOSITION_AMBIGUITY"
                            diag["best_mode"] = best
                            diag["gt_role"] = gt_role
        elif expected != "none" and mode_details:
            for d in mode_details:
                vals = [
                    float(d["roles"][role].get("iou_to_gt", 0.0))
                    for role in ("numerator", "denominator")
                ]
                d["mean_iou_to_gt"] = sum(vals) / 2.0
            best = max(mode_details, key=lambda d: d["mean_iou_to_gt"])
            iou_correct.append(best["mean_iou_to_gt"])
            dx_abs_correct.extend(
                abs(float(best["roles"][r].get("center_dx_to_gt", 0.0)))
                for r in ("numerator", "denominator")
            )

        root[category] += 1

        if not is_correct and expected != "none" and mode_details:
            for d in mode_details:
                vals = [
                    float(d["roles"][role].get("iou_to_gt", 0.0))
                    for role in ("numerator", "denominator")
                ]
                d["mean_iou_to_gt"] = sum(vals) / 2.0
            best = max(mode_details, key=lambda d: d["mean_iou_to_gt"])
            iou_fail.append(best["mean_iou_to_gt"])
            dx_abs_fail.extend(
                abs(float(best["roles"][r].get("center_dx_to_gt", 0.0)))
                for r in ("numerator", "denominator")
            )

        if not is_correct and len(mismatch_rows) < 200:
            mismatch_rows.append(
                {
                    "record_id": record_id,
                    "measure_number": measure_number,
                    "expected": expected,
                    "presence_score": c["presence_score"],
                    "runtime_status": final.status,
                    "runtime_meter": final.meter_class,
                    "root_cause": category,
                    "diagnostic": diag,
                }
            )

        if idx % 128 == 0 or idx == 1224:
            print(f"DIAGNOSTIC {idx}/1224", flush=True)

    if runtime_exact != 1100:
        raise RuntimeError(
            f"runtime exact-match integrity mismatch: expected 1100, got {runtime_exact}"
        )

    report = {
        "stage": "meter-v2-runtime-slot-root-cause-v1",
        "state": "COMPLETE_DIAGNOSTIC",
        "claim_boundary": "POST_HOC_VALIDATION_DIAGNOSTIC_NO_TUNING",
        "validation_records": 1224,
        "test_records": 0,
        "runtime_exact_match": runtime_exact,
        "runtime_exact_accuracy": runtime_exact / 1224,
        "runtime_status_counts": dict(runtime_status),
        "root_cause_counts": dict(root),
        "role_miss_counts": dict(role_fail),
        "accepted_wrong_count": accepted_wrong,
        "per_measure": {
            str(k): dict(v) for k, v in sorted(per_measure.items())
        },
        "slot_observation_counts": dict(slot_obs),
        "geometry_summary": {
            "correct_positive_best_mode_mean_iou_median": (
                median(iou_correct) if iou_correct else None
            ),
            "correct_positive_best_mode_mean_iou_p10": quantile(iou_correct, 0.10),
            "failing_positive_best_mode_mean_iou_median": (
                median(iou_fail) if iou_fail else None
            ),
            "failing_positive_best_mode_mean_iou_p90": quantile(iou_fail, 0.90),
            "correct_positive_abs_center_dx_median_px": (
                median(dx_abs_correct) if dx_abs_correct else None
            ),
            "correct_positive_abs_center_dx_p95_px": quantile(dx_abs_correct, 0.95),
            "failing_positive_abs_center_dx_median_px": (
                median(dx_abs_fail) if dx_abs_fail else None
            ),
            "failing_positive_abs_center_dx_p95_px": quantile(dx_abs_fail, 0.95),
        },
        "mismatch_examples_first_200": mismatch_rows,
        "safety": {
            "training_started": False,
            "optimizer_steps_added": 0,
            "threshold_tuning": False,
            "validation_used_for_parameter_selection": False,
            "test_opened": False,
            "drive_write": False,
            "resolver_wiring": False,
            "production_promotion": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 72)
    print("METER V2 RUNTIME-SLOT ROOT-CAUSE COMPLETE")
    print("RUNTIME EXACT :", f"{runtime_exact}/1224 = {runtime_exact/1224:.6f}")
    print("ROOT CAUSES   :", dict(root))
    print("ROLE MISSES   :", dict(role_fail))
    print(
        "PER MEASURE   :",
        {str(k): dict(v) for k, v in sorted(per_measure.items())},
    )
    print("GEOMETRY      :", report["geometry_summary"])
    print("TEST          : CLOSED")
    print("TUNING        : NONE")
    print("TRAINING      : NONE")
    print("RESOLVER      : NOT WIRED")
    print("REPORT        :", path)


if __name__ == "__main__":
    main()
