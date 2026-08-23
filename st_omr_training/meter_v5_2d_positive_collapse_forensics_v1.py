"""Meter V5-2D read-only forensic audit for V5-2B positive collapse.

This module deliberately performs no training, no threshold tuning and no new
spatial derivation. It diagnoses label/loss/batch/state/logit behavior using the
already-approved V5 adaptation TRAIN slots and historical M4A TRAIN replay.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Final, Mapping

from PIL import Image

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2c_historical_retention_v1 as ret_legacy
from . import meter_v5_2c_historical_retention_v2 as ret_v2


FORENSIC_SCHEMA: Final[str] = "st-omr-meter-v5-2d-positive-collapse-forensics-v1"
FORENSIC_REPORT_NAME: Final[str] = "v5_2d_positive_collapse_forensics_v1.json"

EXPECTED_M4A_TRAIN_COUNTS: Final[dict[str, int]] = {
    "2": 1527,
    "3": 1587,
    "4": 6396,
    "NONE": 17454,
}
EXPECTED_M4A_TRAIN_TOTAL: Final[int] = 26964
V5_TRAIN_SLOT_TOTAL: Final[int] = 540
V5_POSITIVE_PER_SPECIALIST: Final[int] = 90
V5_NEGATIVE_PER_SPECIALIST: Final[int] = 450
EXPECTED_POS_WEIGHT: Final[float] = 5.0

ProgressCallback = Callable[[int, int, str], None]


def _fail(message: str) -> None:
    raise v52b.MeterV5_2BError(message)


def _stats(values: list[float]) -> dict[str, object]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        _fail("non-finite value reached forensic statistics")

    def q(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p05": q(0.05),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p95": q(0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _probability_stats_from_logits(logits: list[float]) -> dict[str, object]:
    probabilities = [1.0 / (1.0 + math.exp(-value)) for value in logits]
    return _stats(probabilities)


def _verify_retention_hold(root: Path) -> dict[str, object]:
    path = root / v51.ANNOTATIONS_DIR / ret_v2.RETENTION_REPORT_NAME
    report = v52b._read_json(path)
    if report.get("schema") != ret_v2.RETENTION_SCHEMA:
        _fail("V5-2D requires the exact V5-2C retention V2 report")
    if report.get("historical_pixel_path_reproduced") is not True:
        _fail("historical pixel path is not proven before V5-2D")
    if report.get("gate") != "HOLD":
        _fail("V5-2D is only valid for the observed retention HOLD")
    if report.get("validation_bbox_stage_authorized") is not False:
        _fail("validation-BBox stage must remain unauthorized")
    if report.get("v5_validation_opened") is not False:
        _fail("V5 validation must remain closed")
    if report.get("final_holdout_locked") is not True:
        _fail("FINAL_HOLDOUT must remain locked")
    expected_candidates = {
        "2": ret_v2.DIGIT2_CANDIDATE_SHA256,
        "3": ret_v2.DIGIT3_CANDIDATE_SHA256,
    }
    if report.get("candidate_sha256") != expected_candidates:
        _fail("retention HOLD candidate identities changed")
    return report


def _verify_v5_train_contract(
    root: Path,
) -> tuple[Path, list[dict[str, str]], dict[str, object]]:
    manifest_path, rows, slot_audit = v52b.verify_slot_manifest_v1(root)
    train_rows = [row for row in rows if row.get("data_role") == "adaptation_train"]
    if len(train_rows) != V5_TRAIN_SLOT_TOTAL:
        _fail("V5 adaptation TRAIN slot count changed")
    for row in train_rows:
        if row.get("slot_role") not in {"numerator", "denominator"}:
            _fail("unexpected V5 slot role")
        meter = row.get("meter")
        if meter not in {"2/4", "3/4", "4/4"}:
            _fail("unexpected V5 meter label")
        for digit in ("2", "3"):
            observed = int(row[f"label_digit{digit}"])
            expected = int(row["slot_role"] == "numerator" and meter == f"{digit}/4")
            if observed != expected:
                _fail(
                    f"V5 label polarity/manifest mismatch for {digit}-AI: "
                    f"{row.get('sample_id')} {row.get('slot_role')}"
                )
            if row["slot_role"] == "denominator" and observed != 0:
                _fail("denominator must remain negative for 2-AI/3-AI")
    for digit in ("2", "3"):
        positives = sum(int(row[f"label_digit{digit}"]) for row in train_rows)
        negatives = len(train_rows) - positives
        if (positives, negatives) != (
            V5_POSITIVE_PER_SPECIALIST,
            V5_NEGATIVE_PER_SPECIALIST,
        ):
            _fail(f"{digit}-AI V5 train balance changed")
    return manifest_path, train_rows, slot_audit


def _batch_construction_audit(train_rows: list[dict[str, str]], *, digit: str) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    config = v52b.FROZEN_TRAIN_CONFIG
    labels = [int(row[f"label_digit{digit}"]) for row in train_rows]
    batch_positive_counts: list[int] = []
    batch_negative_counts: list[int] = []
    epoch_summaries: list[dict[str, object]] = []
    seed = config.master_seed + int(digit)
    for epoch in range(config.epochs):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch)
        order = torch.randperm(len(train_rows), generator=generator).tolist()
        if sorted(order) != list(range(len(train_rows))):
            _fail("deterministic epoch order is not a permutation of all V5 rows")
        epoch_positive = 0
        epoch_negative = 0
        epoch_batches = 0
        for start in range(0, len(order), config.batch_size):
            indexes = order[start:start + config.batch_size]
            positives = sum(labels[index] for index in indexes)
            negatives = len(indexes) - positives
            batch_positive_counts.append(positives)
            batch_negative_counts.append(negatives)
            epoch_positive += positives
            epoch_negative += negatives
            epoch_batches += 1
        if (epoch_positive, epoch_negative) != (
            V5_POSITIVE_PER_SPECIALIST,
            V5_NEGATIVE_PER_SPECIALIST,
        ):
            _fail("one deterministic epoch does not contain the full 90/450 balance")
        epoch_summaries.append({
            "epoch": epoch,
            "positive": epoch_positive,
            "negative": epoch_negative,
            "batch_count": epoch_batches,
        })
    return {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "all_rows_seen_once_per_epoch": True,
        "epoch_summaries": epoch_summaries,
        "batch_positive": _stats([float(value) for value in batch_positive_counts]),
        "batch_negative": _stats([float(value) for value in batch_negative_counts]),
        "zero_positive_batches": sum(value == 0 for value in batch_positive_counts),
        "zero_negative_batches": sum(value == 0 for value in batch_negative_counts),
    }


def _load_frozen_model(path: Path, *, digit: str):
    return ret_legacy._frozen_model(path, digit=digit)


def _parameter_drift(frozen_model, candidate_model) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    frozen_state = frozen_model.state_dict()
    candidate_state = candidate_model.state_dict()
    if set(frozen_state) != set(candidate_state):
        _fail("candidate/frozen parameter keys differ")
    tensors: dict[str, dict[str, object]] = {}
    for name in sorted(frozen_state):
        frozen = frozen_state[name].detach().cpu().to(dtype=torch.float64).reshape(-1)
        candidate = candidate_state[name].detach().cpu().to(dtype=torch.float64).reshape(-1)
        delta = candidate - frozen
        frozen_norm = float(torch.linalg.vector_norm(frozen).item())
        candidate_norm = float(torch.linalg.vector_norm(candidate).item())
        delta_norm = float(torch.linalg.vector_norm(delta).item())
        denominator = max(frozen_norm, 1e-12)
        dot = float(torch.dot(frozen, candidate).item())
        cosine_denom = frozen_norm * candidate_norm
        cosine = dot / cosine_denom if cosine_denom > 0.0 else None
        tensors[name] = {
            "frozen_l2": frozen_norm,
            "candidate_l2": candidate_norm,
            "delta_l2": delta_norm,
            "relative_delta_l2": delta_norm / denominator,
            "cosine_similarity": cosine,
            "frozen_mean": float(frozen.mean().item()),
            "candidate_mean": float(candidate.mean().item()),
            "mean_shift": float((candidate.mean() - frozen.mean()).item()),
        }
    return {
        "tensors": tensors,
        "head_weight": tensors["head.weight"],
        "head_bias": tensors["head.bias"],
        "feature_tensor_names": [name for name in tensors if name.startswith("features.")],
    }


def _score_models_on_v5_train(
    root: Path,
    train_rows: list[dict[str, str]],
    *,
    frozen_models: Mapping[str, object],
    candidate_models: Mapping[str, object],
) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    ann_dir = root / v51.ANNOTATIONS_DIR
    images = torch.stack(
        [v52b._tensor_from_crop(ann_dir / row["crop_relpath"]) for row in train_rows],
        dim=0,
    )
    report: dict[str, object] = {}
    with torch.no_grad():
        for digit in ("2", "3"):
            labels = torch.tensor(
                [int(row[f"label_digit{digit}"]) for row in train_rows],
                dtype=torch.float32,
            )
            threshold = v52b.FROZEN_THRESHOLDS[digit]
            source_logits = frozen_models[digit](images).cpu()
            candidate_logits = candidate_models[digit](images).cpu()
            if not bool(torch.isfinite(source_logits).all().item()):
                _fail("non-finite frozen logits on V5 TRAIN")
            if not bool(torch.isfinite(candidate_logits).all().item()):
                _fail("non-finite candidate logits on V5 TRAIN")
            pos_weight = torch.tensor(EXPECTED_POS_WEIGHT, dtype=torch.float32)
            source_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                source_logits,
                labels,
                pos_weight=pos_weight,
                reduction="none",
            )
            candidate_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                candidate_logits,
                labels,
                pos_weight=pos_weight,
                reduction="none",
            )
            source_prob = torch.sigmoid(source_logits)
            candidate_prob = torch.sigmoid(candidate_logits)
            groups: dict[str, list[int]] = defaultdict(list)
            for index, row in enumerate(train_rows):
                groups[str(row["expected_digit"])].append(index)
            group_report: dict[str, object] = {}
            for group, indexes in sorted(groups.items()):
                source_group_logits = [float(source_logits[index].item()) for index in indexes]
                candidate_group_logits = [float(candidate_logits[index].item()) for index in indexes]
                group_report[group] = {
                    "count": len(indexes),
                    "source_logits": _stats(source_group_logits),
                    "candidate_logits": _stats(candidate_group_logits),
                    "source_probabilities": _probability_stats_from_logits(source_group_logits),
                    "candidate_probabilities": _probability_stats_from_logits(candidate_group_logits),
                    "source_positive_rate": sum(float(source_prob[index].item()) >= threshold for index in indexes) / len(indexes),
                    "candidate_positive_rate": sum(float(candidate_prob[index].item()) >= threshold for index in indexes) / len(indexes),
                }
            positive_mask = labels >= 0.5
            negative_mask = ~positive_mask
            report[digit] = {
                "shared_input_tensor_for_frozen_and_candidate": True,
                "threshold": threshold,
                "positive_weight": EXPECTED_POS_WEIGHT,
                "source_metrics": v52b._binary_counts(source_prob, labels, threshold),
                "candidate_metrics": v52b._binary_counts(candidate_prob, labels, threshold),
                "weighted_loss": {
                    "source_positive_sum": float(source_losses[positive_mask].sum().item()),
                    "source_negative_sum": float(source_losses[negative_mask].sum().item()),
                    "source_positive_mean": float(source_losses[positive_mask].mean().item()),
                    "source_negative_mean": float(source_losses[negative_mask].mean().item()),
                    "candidate_positive_sum": float(candidate_losses[positive_mask].sum().item()),
                    "candidate_negative_sum": float(candidate_losses[negative_mask].sum().item()),
                    "candidate_positive_mean": float(candidate_losses[positive_mask].mean().item()),
                    "candidate_negative_mean": float(candidate_losses[negative_mask].mean().item()),
                },
                "by_expected_digit": group_report,
            }
    return report


def _historical_train_records(
    *,
    m4a_root: Path,
    d10_root: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    m4a_manifest = m4a_root / "dataset-manifest.json"
    d10_manifest = d10_root / "manifest.json"
    if v52b._sha_file(m4a_manifest) != ret_v2.M4A_MANIFEST_SHA256:
        _fail("M4A manifest SHA mismatch")
    if v52b._sha_file(d10_manifest) != ret_v2.D10_MANIFEST_SHA256:
        _fail("D10 manifest SHA mismatch")
    m4a = v52b._read_json(m4a_manifest)
    d10 = v52b._read_json(d10_manifest)
    records = m4a.get("records")
    d10_records = d10.get("records")
    if not isinstance(records, list) or not isinstance(d10_records, list):
        _fail("historical manifests missing records")
    train = [row for row in records if isinstance(row, dict) and row.get("split") == "train"]
    counts = Counter(str(row.get("digit_label")) for row in train)
    if len(train) != EXPECTED_M4A_TRAIN_TOTAL or dict(counts) != EXPECTED_M4A_TRAIN_COUNTS:
        _fail(f"M4A TRAIN identity changed: total={len(train)} counts={dict(counts)}")
    d10_meter = {
        str(row.get("record_id")): row
        for row in d10_records
        if isinstance(row, dict) and row.get("kind") == "meter"
    }
    if len(d10_meter) != 11064:
        _fail("D10 meter record count changed")
    for row in train:
        source_id = str(row.get("source_record_id"))
        if source_id not in d10_meter:
            _fail(f"M4A TRAIN references missing D10 meter record: {source_id}")
    return train, d10_meter


def _historical_train_score_profiles(
    *,
    train: list[dict[str, object]],
    d10_meter: Mapping[str, Mapping[str, object]],
    d10_root: Path,
    frozen_models: Mapping[str, object],
    candidate_models: Mapping[str, object],
    progress: ProgressCallback | None,
) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    logits_by_model_label: dict[str, dict[str, list[float]]] = {
        key: {label: [] for label in ("2", "3", "4", "NONE")}
        for key in ("frozen-2", "candidate-2", "frozen-3", "candidate-3")
    }
    positive_counts: dict[str, Counter[str]] = {
        key: Counter() for key in logits_by_model_label
    }
    total_counts = Counter(str(row["digit_label"]) for row in train)
    current_path: Path | None = None
    current_image: Image.Image | None = None
    batch_tensors = []
    batch_labels: list[str] = []

    def flush() -> None:
        nonlocal batch_tensors, batch_labels
        if not batch_tensors:
            return
        images = torch.stack(batch_tensors, dim=0).to(dtype=torch.float32).unsqueeze(1) / 255.0
        with torch.no_grad():
            outputs = {
                "frozen-2": frozen_models["2"](images).cpu(),
                "candidate-2": candidate_models["2"](images).cpu(),
                "frozen-3": frozen_models["3"](images).cpu(),
                "candidate-3": candidate_models["3"](images).cpu(),
            }
        for key, logits in outputs.items():
            if not bool(torch.isfinite(logits).all().item()):
                _fail(f"non-finite historical TRAIN logits: {key}")
            digit = key[-1]
            threshold = v52b.FROZEN_THRESHOLDS[digit]
            probabilities = torch.sigmoid(logits)
            for index, label in enumerate(batch_labels):
                value = float(logits[index].item())
                logits_by_model_label[key][label].append(value)
                if float(probabilities[index].item()) >= threshold:
                    positive_counts[key][label] += 1
        batch_tensors = []
        batch_labels = []

    try:
        for index, row in enumerate(train, start=1):
            source_id = str(row["source_record_id"])
            d10_row = d10_meter[source_id]
            image_relpath = d10_row.get("image_path")
            if not isinstance(image_relpath, str) or not image_relpath:
                _fail(f"D10 meter image path missing: {source_id}")
            image_path = d10_root / image_relpath
            if image_path != current_path:
                if current_image is not None:
                    current_image.close()
                if image_path.is_symlink() or not image_path.is_file():
                    _fail(f"D10 source image missing/non-regular: {image_path}")
                current_image = Image.open(image_path).convert("L")
                current_path = image_path
            assert current_image is not None
            canvas = ret_legacy._historical_canvas_from_bbox(current_image, row.get("bbox"))
            tensor = torch.tensor(list(canvas.getdata()), dtype=torch.uint8).reshape(64, 64)
            batch_tensors.append(tensor)
            batch_labels.append(str(row["digit_label"]))
            if len(batch_tensors) == 256:
                flush()
            if progress is not None and (index == 1 or index % 500 == 0 or index == len(train)):
                progress(index, len(train), "historical-m4a-train-logit-forensics")
        flush()
    finally:
        if current_image is not None:
            current_image.close()

    result: dict[str, object] = {}
    for key in logits_by_model_label:
        digit = key[-1]
        by_label: dict[str, object] = {}
        for label in ("2", "3", "4", "NONE"):
            values = logits_by_model_label[key][label]
            count = total_counts[label]
            if len(values) != count:
                _fail(f"historical TRAIN profile count mismatch: {key} {label}")
            by_label[label] = {
                "count": count,
                "logits": _stats(values),
                "probabilities": _probability_stats_from_logits(values),
                "positive_count": positive_counts[key][label],
                "positive_rate": positive_counts[key][label] / count,
            }
        negative_labels = [label for label in ("2", "3", "4", "NONE") if label != digit]
        negative_total = sum(total_counts[label] for label in negative_labels)
        negative_positive = sum(positive_counts[key][label] for label in negative_labels)
        result[key] = {
            "digit": digit,
            "threshold": v52b.FROZEN_THRESHOLDS[digit],
            "shared_historical_input_tensor_with_pair": True,
            "by_true_label": by_label,
            "negative_positive_rate": negative_positive / negative_total,
            "positive_true_label_rate": positive_counts[key][digit] / total_counts[digit],
        }
    return result


def run_positive_collapse_forensics_v1(
    v5_data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    digit2_candidate: str | Path,
    digit3_candidate: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    root = Path(v5_data_root)
    m4a = Path(m4a_root)
    d10 = Path(d10_root)
    frozen_paths = {"2": Path(digit2_frozen), "3": Path(digit3_frozen)}
    candidate_paths = {"2": Path(digit2_candidate), "3": Path(digit3_candidate)}

    retention_report = _verify_retention_hold(root)
    manifest_path, train_rows, slot_audit = _verify_v5_train_contract(root)
    manifest_sha = v52b._sha_file(manifest_path)

    expected_frozen_sha = {"2": v52b.DIGIT2_SHA256, "3": v52b.DIGIT3_SHA256}
    expected_candidate_sha = {
        "2": ret_v2.DIGIT2_CANDIDATE_SHA256,
        "3": ret_v2.DIGIT3_CANDIDATE_SHA256,
    }
    frozen_models: dict[str, object] = {}
    candidate_models: dict[str, object] = {}
    for digit in ("2", "3"):
        if v52b._sha_file(frozen_paths[digit]) != expected_frozen_sha[digit]:
            _fail(f"frozen {digit}-AI SHA changed")
        if v52b._sha_file(candidate_paths[digit]) != expected_candidate_sha[digit]:
            _fail(f"candidate {digit}-AI SHA changed")
        frozen_models[digit] = _load_frozen_model(frozen_paths[digit], digit=digit)
        candidate_models[digit] = v52b._load_candidate_model(
            candidate_paths[digit],
            digit=digit,
            manifest_sha256=manifest_sha,
        )

    label_and_batch: dict[str, object] = {}
    drift: dict[str, object] = {}
    for digit in ("2", "3"):
        label_and_batch[digit] = {
            "positive": V5_POSITIVE_PER_SPECIALIST,
            "negative": V5_NEGATIVE_PER_SPECIALIST,
            "positive_weight": EXPECTED_POS_WEIGHT,
            "effective_total_positive_weight": V5_POSITIVE_PER_SPECIALIST * EXPECTED_POS_WEIGHT,
            "effective_total_negative_weight": float(V5_NEGATIVE_PER_SPECIALIST),
            "label_polarity_manifest_consistent": True,
            "denominator_positive_count": 0,
            "batch_construction": _batch_construction_audit(train_rows, digit=digit),
        }
        drift[digit] = _parameter_drift(frozen_models[digit], candidate_models[digit])

    v5_profiles = _score_models_on_v5_train(
        root,
        train_rows,
        frozen_models=frozen_models,
        candidate_models=candidate_models,
    )

    historical_train, d10_meter = _historical_train_records(
        m4a_root=m4a,
        d10_root=d10,
    )
    historical_profiles = _historical_train_score_profiles(
        train=historical_train,
        d10_meter=d10_meter,
        d10_root=d10,
        frozen_models=frozen_models,
        candidate_models=candidate_models,
        progress=progress,
    )

    collapse_flags: dict[str, object] = {}
    for digit in ("2", "3"):
        frozen_negative_rate = float(historical_profiles[f"frozen-{digit}"]["negative_positive_rate"])
        candidate_negative_rate = float(historical_profiles[f"candidate-{digit}"]["negative_positive_rate"])
        collapse_flags[digit] = {
            "frozen_historical_train_negative_positive_rate": frozen_negative_rate,
            "candidate_historical_train_negative_positive_rate": candidate_negative_rate,
            "candidate_positive_collapse_flag": candidate_negative_rate >= 0.90,
            "candidate_minus_frozen_negative_positive_rate": candidate_negative_rate - frozen_negative_rate,
        }

    report: dict[str, object] = {
        "schema": FORENSIC_SCHEMA,
        "retention_v2_report_sha256": v52b._sha_file(
            root / v51.ANNOTATIONS_DIR / ret_v2.RETENTION_REPORT_NAME
        ),
        "retention_v2_gate": retention_report["gate"],
        "slot_manifest_sha256": manifest_sha,
        "slot_audit_sha256": v52b._sha_file(root / v51.ANNOTATIONS_DIR / v52b.SLOT_AUDIT_NAME),
        "m4a_manifest_sha256": ret_v2.M4A_MANIFEST_SHA256,
        "d10_manifest_sha256": ret_v2.D10_MANIFEST_SHA256,
        "candidate_sha256": expected_candidate_sha,
        "frozen_checkpoint_sha256": expected_frozen_sha,
        "v5_adaptation_train_slot_count": len(train_rows),
        "m4a_train_record_count": len(historical_train),
        "m4a_train_label_counts": dict(EXPECTED_M4A_TRAIN_COUNTS),
        "label_and_batch_audit": label_and_batch,
        "parameter_drift": drift,
        "v5_train_score_and_loss_profiles": v5_profiles,
        "historical_train_score_profiles": historical_profiles,
        "collapse_flags": collapse_flags,
        "candidate_specific_preprocessing_branch": False,
        "frozen_and_candidate_receive_same_tensor_per_comparison": True,
        "root_cause_conclusion": "NOT_ASSIGNED_AUTOMATICALLY",
        "training_repair_authorized": False,
        "new_bbox_work_authorized": False,
        "reserve_v5_train_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "optimizer_steps": 0,
        "threshold_tuning": False,
        "resolver_wiring_authorized": False,
        "production_promotion_authorized": False,
        "frozen_control_specialist": "4-AI",
    }
    v51._atomic_write_json(
        root / v51.ANNOTATIONS_DIR / FORENSIC_REPORT_NAME,
        report,
    )
    return report


def training_allowed_by_this_module() -> bool:
    return False


def validation_opened_by_this_module() -> bool:
    return False


def final_holdout_locked() -> bool:
    return True


def production_promotion_allowed() -> bool:
    return False
