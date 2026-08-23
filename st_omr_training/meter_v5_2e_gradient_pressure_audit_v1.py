"""Meter V5-2E no-gradient pressure audit.

This module is diagnostic only.  It analytically evaluates dL/dz for the
already-frozen BCE-with-logits objective on the existing V5 adaptation TRAIN
surface and the exact historical M4A TRAIN replay.  It performs no backward
pass, optimizer step, checkpoint write, threshold tuning, or new spatial
inference.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Final, Mapping

from PIL import Image

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2c_historical_retention_v1 as ret_legacy
from . import meter_v5_2c_historical_retention_v2 as ret_v2
from . import meter_v5_2d_positive_collapse_forensics_v1 as v52d


PRESSURE_SCHEMA: Final[str] = "st-omr-meter-v5-2e-gradient-pressure-audit-v1"
PRESSURE_REPORT_NAME: Final[str] = "v5_2e_gradient_pressure_audit_v1.json"
COUNTERFACTUAL_POS_WEIGHTS: Final[tuple[float, ...]] = (1.0, 5.0)
DOMINANCE_RATIO_FLOOR: Final[float] = 100.0

ProgressCallback = Callable[[int, int, str], None]


def _fail(message: str) -> None:
    raise v52b.MeterV5_2BError(message)


def _verify_v52d_evidence(root: Path) -> tuple[Path, dict[str, object]]:
    path = root / v51.ANNOTATIONS_DIR / v52d.FORENSIC_REPORT_NAME
    report = v52b._read_json(path)
    expected_scalars = {
        "schema": v52d.FORENSIC_SCHEMA,
        "retention_v2_gate": "HOLD",
        "v5_adaptation_train_slot_count": v52d.V5_TRAIN_SLOT_TOTAL,
        "m4a_train_record_count": v52d.EXPECTED_M4A_TRAIN_TOTAL,
        "m4a_train_label_counts": dict(v52d.EXPECTED_M4A_TRAIN_COUNTS),
        "candidate_specific_preprocessing_branch": False,
        "frozen_and_candidate_receive_same_tensor_per_comparison": True,
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
    for key, expected in expected_scalars.items():
        if report.get(key) != expected:
            _fail(f"V5-2D evidence changed before V5-2E: {key}")

    expected_candidates = {
        "2": ret_v2.DIGIT2_CANDIDATE_SHA256,
        "3": ret_v2.DIGIT3_CANDIDATE_SHA256,
    }
    expected_frozen = {"2": v52b.DIGIT2_SHA256, "3": v52b.DIGIT3_SHA256}
    if report.get("candidate_sha256") != expected_candidates:
        _fail("V5-2D candidate identities changed")
    if report.get("frozen_checkpoint_sha256") != expected_frozen:
        _fail("V5-2D frozen checkpoint identities changed")

    label_audit = report.get("label_and_batch_audit")
    v5_profiles = report.get("v5_train_score_and_loss_profiles")
    historical_profiles = report.get("historical_train_score_profiles")
    if not isinstance(label_audit, Mapping) or not isinstance(v5_profiles, Mapping):
        _fail("V5-2D label/loss evidence missing")
    if not isinstance(historical_profiles, Mapping):
        _fail("V5-2D historical evidence missing")

    for digit in ("2", "3"):
        audit = label_audit.get(digit)
        if not isinstance(audit, Mapping):
            _fail(f"V5-2D {digit}-AI label audit missing")
        required = {
            "positive": 90,
            "negative": 450,
            "positive_weight": 5.0,
            "effective_total_positive_weight": 450.0,
            "effective_total_negative_weight": 450.0,
            "label_polarity_manifest_consistent": True,
            "denominator_positive_count": 0,
        }
        for key, expected in required.items():
            if audit.get(key) != expected:
                _fail(f"V5-2D {digit}-AI label evidence changed: {key}")
        batch = audit.get("batch_construction")
        if not isinstance(batch, Mapping):
            _fail(f"V5-2D {digit}-AI batch evidence missing")
        if batch.get("all_rows_seen_once_per_epoch") is not True:
            _fail(f"V5-2D {digit}-AI epoch permutation evidence changed")
        if batch.get("zero_positive_batches") != 0 or batch.get("zero_negative_batches") != 0:
            _fail(f"V5-2D {digit}-AI batch class coverage changed")

        profile = v5_profiles.get(digit)
        if not isinstance(profile, Mapping):
            _fail(f"V5-2D {digit}-AI V5 profile missing")
        source_metrics = profile.get("source_metrics")
        if not isinstance(source_metrics, Mapping):
            _fail(f"V5-2D {digit}-AI frozen V5 metrics missing")
        if tuple(source_metrics.get(key) for key in ("tp", "fp", "fn", "tn")) != (0, 0, 90, 450):
            _fail(f"V5-2D {digit}-AI frozen V5 behavior changed")

    return path, report


def _pressure_profile_from_logits(
    logits,
    labels,
    *,
    positive_weight: float,
) -> dict[str, object]:
    """Compute analytical BCE-with-logits dL/dz without autograd."""
    torch, _nn = v52b._import_torch()
    logits = logits.detach().cpu().to(dtype=torch.float64).reshape(-1)
    labels = labels.detach().cpu().to(dtype=torch.float64).reshape(-1)
    if logits.numel() != labels.numel() or logits.numel() == 0:
        _fail("pressure profile requires equally-sized non-empty logits/labels")
    if not bool(torch.isfinite(logits).all().item()):
        _fail("non-finite logits reached pressure audit")
    if not bool(((labels == 0.0) | (labels == 1.0)).all().item()):
        _fail("pressure audit labels must be binary")
    if positive_weight <= 0.0:
        _fail("positive weight must be > 0")

    probabilities = torch.sigmoid(logits)
    positive_mask = labels >= 0.5
    negative_mask = ~positive_mask
    signed = torch.where(
        positive_mask,
        float(positive_weight) * (probabilities - 1.0),
        probabilities,
    )
    absolute = torch.abs(signed)

    def stats(tensor) -> dict[str, object]:
        return v52d._stats([float(value) for value in tensor.tolist()])

    positive_total = float(absolute[positive_mask].sum().item())
    negative_total = float(absolute[negative_mask].sum().item())
    if negative_total == 0.0:
        ratio = None
        ratio_state = "NEGATIVE_PRESSURE_ZERO"
    else:
        ratio = positive_total / negative_total
        ratio_state = "FINITE"

    return {
        "count": int(logits.numel()),
        "positive_count": int(positive_mask.sum().item()),
        "negative_count": int(negative_mask.sum().item()),
        "positive_weight": float(positive_weight),
        "logits": stats(logits),
        "probabilities": stats(probabilities),
        "signed_dldz": stats(signed),
        "absolute_dldz": stats(absolute),
        "positive_signed_dldz": stats(signed[positive_mask]),
        "negative_signed_dldz": stats(signed[negative_mask]),
        "positive_abs_dldz": stats(absolute[positive_mask]),
        "negative_abs_dldz": stats(absolute[negative_mask]),
        "absolute_pressure_sum": float(absolute.sum().item()),
        "absolute_pressure_mean": float(absolute.mean().item()),
        "positive_pressure_total": positive_total,
        "negative_pressure_total": negative_total,
        "positive_to_negative_pressure_ratio": ratio,
        "pressure_ratio_state": ratio_state,
    }


def _load_models(
    root: Path,
    *,
    digit2_frozen: Path,
    digit3_frozen: Path,
    digit2_candidate: Path,
    digit3_candidate: Path,
):
    manifest_path, train_rows, slot_audit = v52d._verify_v5_train_contract(root)
    manifest_sha = v52b._sha_file(manifest_path)
    frozen_paths = {"2": digit2_frozen, "3": digit3_frozen}
    candidate_paths = {"2": digit2_candidate, "3": digit3_candidate}
    frozen_models: dict[str, object] = {}
    candidate_models: dict[str, object] = {}
    for digit in ("2", "3"):
        if v52b._sha_file(frozen_paths[digit]) != {"2": v52b.DIGIT2_SHA256, "3": v52b.DIGIT3_SHA256}[digit]:
            _fail(f"frozen {digit}-AI SHA changed")
        if v52b._sha_file(candidate_paths[digit]) != {"2": ret_v2.DIGIT2_CANDIDATE_SHA256, "3": ret_v2.DIGIT3_CANDIDATE_SHA256}[digit]:
            _fail(f"candidate {digit}-AI SHA changed")
        frozen_models[digit] = v52d._load_frozen_model(frozen_paths[digit], digit=digit)
        candidate_models[digit] = v52b._load_candidate_model(
            candidate_paths[digit], digit=digit, manifest_sha256=manifest_sha
        )
    return manifest_path, train_rows, slot_audit, frozen_models, candidate_models


def _v5_pressure_profiles(
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
    result: dict[str, object] = {}
    with torch.no_grad():
        for digit in ("2", "3"):
            labels = torch.tensor(
                [int(row[f"label_digit{digit}"]) for row in train_rows],
                dtype=torch.float32,
            )
            frozen_logits = frozen_models[digit](images).cpu()
            candidate_logits = candidate_models[digit](images).cpu()
            result[digit] = {
                "shared_input_tensor_for_frozen_and_candidate": True,
                "frozen": {
                    f"pos_weight_{weight:g}": _pressure_profile_from_logits(
                        frozen_logits, labels, positive_weight=weight
                    )
                    for weight in COUNTERFACTUAL_POS_WEIGHTS
                },
                "candidate": {
                    f"pos_weight_{weight:g}": _pressure_profile_from_logits(
                        candidate_logits, labels, positive_weight=weight
                    )
                    for weight in COUNTERFACTUAL_POS_WEIGHTS
                },
            }
    return result


def _historical_pressure_profiles(
    *,
    train: list[dict[str, object]],
    d10_meter: Mapping[str, Mapping[str, object]],
    d10_root: Path,
    frozen_models: Mapping[str, object],
    candidate_models: Mapping[str, object],
    progress: ProgressCallback | None,
) -> dict[str, object]:
    """Replay exact historical M4A TRAIN pixels and collect pressure statistics."""
    torch, _nn = v52b._import_torch()
    logits: dict[str, list[float]] = {
        key: [] for key in ("frozen-2", "candidate-2", "frozen-3", "candidate-3")
    }
    labels_by_digit: dict[str, list[int]] = {"2": [], "3": []}
    true_labels: list[str] = []
    current_path: Path | None = None
    current_image: Image.Image | None = None
    batch_tensors = []
    batch_true: list[str] = []

    def flush() -> None:
        nonlocal batch_tensors, batch_true
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
        for key, values in outputs.items():
            if not bool(torch.isfinite(values).all().item()):
                _fail(f"non-finite historical TRAIN logits: {key}")
            logits[key].extend(float(value) for value in values.tolist())
        for label in batch_true:
            true_labels.append(label)
            labels_by_digit["2"].append(int(label == "2"))
            labels_by_digit["3"].append(int(label == "3"))
        batch_tensors = []
        batch_true = []

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
            batch_true.append(str(row["digit_label"]))
            if len(batch_tensors) == 256:
                flush()
            if progress is not None and (index == 1 or index % 500 == 0 or index == len(train)):
                progress(index, len(train), "historical-m4a-train-pressure-audit")
        flush()
    finally:
        if current_image is not None:
            current_image.close()

    if len(true_labels) != v52d.EXPECTED_M4A_TRAIN_TOTAL:
        _fail("historical pressure replay count mismatch")
    if dict(Counter(true_labels)) != v52d.EXPECTED_M4A_TRAIN_COUNTS:
        _fail("historical pressure replay label counts changed")

    result: dict[str, object] = {}
    for digit in ("2", "3"):
        labels = torch.tensor(labels_by_digit[digit], dtype=torch.float32)
        result[digit] = {
            "frozen": {},
            "candidate": {},
            "shared_historical_pixels_for_pair": True,
        }
        for model_role in ("frozen", "candidate"):
            values = torch.tensor(logits[f"{model_role}-{digit}"], dtype=torch.float64)
            result[digit][model_role] = {
                f"pos_weight_{weight:g}": _pressure_profile_from_logits(
                    values, labels, positive_weight=weight
                )
                for weight in COUNTERFACTUAL_POS_WEIGHTS
            }
    return result


def _dominance(profile: Mapping[str, object]) -> bool:
    ratio = profile.get("positive_to_negative_pressure_ratio")
    if ratio is None:
        return bool(profile.get("positive_pressure_total", 0.0) > 0.0)
    return float(ratio) >= DOMINANCE_RATIO_FLOOR


def run_gradient_pressure_audit_v1(
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

    v52d_path, v52d_report = _verify_v52d_evidence(root)
    manifest_path, train_rows, slot_audit, frozen_models, candidate_models = _load_models(
        root,
        digit2_frozen=Path(digit2_frozen),
        digit3_frozen=Path(digit3_frozen),
        digit2_candidate=Path(digit2_candidate),
        digit3_candidate=Path(digit3_candidate),
    )

    v5_profiles = _v5_pressure_profiles(
        root,
        train_rows,
        frozen_models=frozen_models,
        candidate_models=candidate_models,
    )
    historical_train, d10_meter = v52d._historical_train_records(
        m4a_root=m4a,
        d10_root=d10,
    )
    historical_profiles = _historical_pressure_profiles(
        train=historical_train,
        d10_meter=d10_meter,
        d10_root=d10,
        frozen_models=frozen_models,
        candidate_models=candidate_models,
        progress=progress,
    )

    evidence: dict[str, object] = {}
    for digit in ("2", "3"):
        frozen_v5_w5 = v5_profiles[digit]["frozen"]["pos_weight_5"]
        frozen_v5_w1 = v5_profiles[digit]["frozen"]["pos_weight_1"]
        candidate_hist_w5 = historical_profiles[digit]["candidate"]["pos_weight_5"]
        evidence[digit] = {
            "frozen_v5_positive_pressure_dominates_100x_at_w5": _dominance(frozen_v5_w5),
            "frozen_v5_positive_pressure_dominates_100x_at_w1": _dominance(frozen_v5_w1),
            "candidate_historical_negative_pressure_total_at_w5": candidate_hist_w5["negative_pressure_total"],
            "historical_negative_pressure_absent_from_original_v5_only_objective": True,
        }

    root_cause_supported = all(
        evidence[digit]["frozen_v5_positive_pressure_dominates_100x_at_w5"]
        and evidence[digit]["historical_negative_pressure_absent_from_original_v5_only_objective"]
        for digit in ("2", "3")
    )
    pos_weight_unique_cause_supported = all(
        not evidence[digit]["frozen_v5_positive_pressure_dominates_100x_at_w1"]
        for digit in ("2", "3")
    )

    report: dict[str, object] = {
        "schema": PRESSURE_SCHEMA,
        "v5_2d_report_sha256": v52b._sha_file(v52d_path),
        "slot_manifest_sha256": v52b._sha_file(manifest_path),
        "slot_audit_sha256": v52b._sha_file(root / v51.ANNOTATIONS_DIR / v52b.SLOT_AUDIT_NAME),
        "m4a_manifest_sha256": ret_v2.M4A_MANIFEST_SHA256,
        "d10_manifest_sha256": ret_v2.D10_MANIFEST_SHA256,
        "v5_adaptation_train_slot_count": len(train_rows),
        "m4a_train_record_count": len(historical_train),
        "counterfactual_positive_weights": list(COUNTERFACTUAL_POS_WEIGHTS),
        "analytical_derivative_contract": {
            "positive": "w * (sigmoid(z) - 1)",
            "negative": "sigmoid(z)",
            "autograd_used": False,
        },
        "v5_adaptation_train_pressure": v5_profiles,
        "historical_m4a_train_pressure": historical_profiles,
        "evidence_flags": evidence,
        "root_cause_class": (
            "V5_ONLY_DOMAIN_ADAPTATION_WITH_UNCONSTRAINED_SOURCE_FORGETTING"
            if root_cause_supported
            else "UNRESOLVED"
        ),
        "dominant_mechanism": "UNRESOLVED",
        "pos_weight_5_unique_root_cause_supported": pos_weight_unique_cause_supported,
        "repair_training_authorized": False,
        "replay_ratio_selected": False,
        "training": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_write": False,
        "threshold_tuning": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "reserve_v5_train_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "frozen_control_specialist": "4-AI",
        "resolver_wiring_authorized": False,
        "production_promotion_authorized": False,
    }
    v51._atomic_write_json(
        root / v51.ANNOTATIONS_DIR / PRESSURE_REPORT_NAME,
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
