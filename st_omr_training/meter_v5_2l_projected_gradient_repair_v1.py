"""Meter V5-2L projected-gradient repair for 2-AI and 3-AI.

This module executes one fixed A-GEM-style repair configuration after the
V5-2K parameter-gradient audit demonstrated strong V5/source conflict.

The historical M4A TRAIN mean gradient is recomputed once at the beginning of
every epoch. Each V5 minibatch gradient is projected onto the source-safe
half-space when it conflicts with that epoch reference gradient, then a direct
SGD step is applied. There is no optimizer state, momentum, weight decay,
gradient renormalization, clipping, sweep, or automatic second configuration.

Historical retention is evaluated before the first-30 V5 diagnostic. V5 VAL,
FINAL_HOLDOUT, reserve TRAIN, 4-AI mutation, threshold tuning, Resolver wiring,
and production promotion remain closed.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Callable, Final, Mapping

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2c_historical_retention_v1 as ret_legacy
from . import meter_v5_2c_historical_retention_v2 as ret_v2
from . import meter_v5_2k_parameter_gradient_balance_audit_v1 as v52k


SCHEMA: Final[str] = "st-omr-meter-v5-2l-projected-gradient-repair-v1"
APPROVAL_TOKEN: Final[str] = "V5_2L_PROJECTED_GRADIENT_REPAIR_APPROVED"
TRAINING_REPORT_NAME: Final[str] = "v5_2l_projected_gradient_training_report.json"
RETENTION_REPORT_NAME: Final[str] = "v5_2l_historical_retention.json"
DIAGNOSTIC_REPORT_NAME: Final[str] = "v5_2l_first30_diagnostic.json"
FINAL_REPORT_NAME: Final[str] = "v5_2l_projected_gradient_final_report.json"
CANDIDATE_DIR_NAME: Final[str] = "v5_2l_projected_gradient_candidates"

POS_WEIGHT: Final[float] = 1.0
LEARNING_RATE: Final[float] = 1e-4
V5_BATCH_SIZE: Final[int] = 64
EPOCHS: Final[int] = 12
MASTER_SEED: Final[int] = 52_023
EXPECTED_V5_COUNT: Final[int] = 540
EXPECTED_V5_BATCHES_PER_EPOCH: Final[int] = 9
EXPECTED_UPDATES_PER_SPECIALIST: Final[int] = EPOCHS * EXPECTED_V5_BATCHES_PER_EPOCH
PROJECTION_TOLERANCE: Final[float] = 1e-9

ProgressCallback = Callable[[int, int, str], None]


class MeterV5_2LError(RuntimeError):
    """Raised whenever the fixed V5-2L contract deviates or fails closed."""


def _fail(message: str) -> None:
    raise MeterV5_2LError(message)


def safety_boundary() -> dict[str, object]:
    return {
        "single_fixed_projected_repair_authorized": True,
        "automatic_second_configuration": False,
        "positive_weight": POS_WEIGHT,
        "learning_rate": LEARNING_RATE,
        "v5_batch_size": V5_BATCH_SIZE,
        "epochs": EPOCHS,
        "updates_per_specialist": EXPECTED_UPDATES_PER_SPECIALIST,
        "source_reference_recomputed_each_epoch": True,
        "direct_sgd_no_momentum": True,
        "weight_decay": 0.0,
        "gradient_clipping": False,
        "gradient_renormalization": False,
        "threshold_tuning": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "reserve_v5_train_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "resolver_wiring": False,
        "production_promotion": False,
    }


def gate_order() -> tuple[str, str]:
    return ("historical_retention", "v5_first30_diagnostic")


def _verify_v52k_evidence(root: Path) -> tuple[Path, dict[str, object]]:
    path = root / v51.ANNOTATIONS_DIR / v52k.REPORT_NAME
    report = v52b._read_json(path)
    expected = {
        "schema": v52k.SCHEMA,
        "objective_contract": "mean(V5_BCE_w1)+lambda_source*mean(HISTORICAL_BCE_w1)",
        "positive_weight": 1.0,
        "v5_adaptation_train_slot_count": 540,
        "m4a_train_record_count": 26_964,
        "domain_weight_selected": False,
        "repair_training_authorized": False,
        "training": False,
        "autograd_grad_used": True,
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
        "digit4_frozen": True,
        "resolver_wiring": False,
        "production_promotion": False,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            _fail(f"V5-2K carried-forward evidence changed: {key}")

    expected_sha = {"2": v52b.DIGIT2_SHA256, "3": v52b.DIGIT3_SHA256}
    if report.get("frozen_checkpoint_sha256") != expected_sha:
        _fail("V5-2K frozen checkpoint identity changed")

    per = report.get("per_specialist")
    if not isinstance(per, Mapping):
        _fail("V5-2K per-specialist evidence missing")
    for digit in ("2", "3"):
        item = per.get(digit)
        if not isinstance(item, Mapping):
            _fail(f"V5-2K {digit}-AI evidence missing")
        groups = item.get("groups")
        if not isinstance(groups, Mapping) or not isinstance(groups.get("all"), Mapping):
            _fail(f"V5-2K {digit}-AI all-parameter evidence missing")
        all_metrics = groups["all"]
        if all_metrics.get("gradient_conflict") is not True:
            _fail(f"V5-2L requires measured gradient conflict for {digit}-AI")
        cosine = float(all_metrics.get("cosine_similarity"))
        if digit == "2" and cosine > -0.98:
            _fail("2-AI V5/source gradient conflict is weaker than the approved V5-2K evidence")
        if digit == "3" and cosine > -0.90:
            _fail("3-AI V5/source gradient conflict is weaker than the approved V5-2K evidence")
        lam = float(all_metrics.get("minimum_norm_lambda_source"))
        if not math.isfinite(lam) or lam <= 0.0:
            _fail(f"V5-2K {digit}-AI projection coefficient evidence invalid")
    return path, report


def _projection_coefficient_from_scalars(dot: float, source_sq: float) -> float:
    if not math.isfinite(dot) or not math.isfinite(source_sq) or source_sq <= 0.0:
        raise ValueError("projection requires finite dot and positive source norm squared")
    return max(0.0, -dot / source_sq) if dot < 0.0 else 0.0


def _projection_summary(
    v5_grad: Mapping[str, object],
    source_grad: Mapping[str, object],
) -> dict[str, float | bool]:
    torch, _nn = v52b._import_torch()
    if set(v5_grad) != set(source_grad) or not v5_grad:
        _fail("V5/source gradient keys differ during projection")
    v5_sq = 0.0
    source_sq = 0.0
    dot = 0.0
    for name in sorted(v5_grad):
        a = v5_grad[name].detach().cpu().to(dtype=torch.float64).reshape(-1)
        b = source_grad[name].detach().cpu().to(dtype=torch.float64).reshape(-1)
        if a.shape != b.shape:
            _fail(f"projection gradient shape mismatch: {name}")
        v5_sq += float(torch.dot(a, a).item())
        source_sq += float(torch.dot(b, b).item())
        dot += float(torch.dot(a, b).item())
    if v5_sq <= 0.0 or source_sq <= 0.0:
        _fail("projection encountered zero gradient norm")
    coeff = _projection_coefficient_from_scalars(dot, source_sq)
    projected_source_dot = dot + coeff * source_sq
    scale = max(1.0, abs(dot), abs(coeff * source_sq))
    if dot < 0.0 and abs(projected_source_dot) > PROJECTION_TOLERANCE * scale:
        _fail("conflicting gradient projection did not reach the source-safe boundary")
    if dot >= 0.0 and projected_source_dot < -PROJECTION_TOLERANCE * scale:
        _fail("non-conflicting gradient became source-unsafe")
    projected_sq = max(0.0, v5_sq + 2.0 * coeff * dot + coeff * coeff * source_sq)
    cosine = dot / math.sqrt(v5_sq * source_sq)
    return {
        "v5_gradient_l2": math.sqrt(v5_sq),
        "source_gradient_l2": math.sqrt(source_sq),
        "dot_product_before_projection": dot,
        "cosine_before_projection": min(1.0, max(-1.0, cosine)),
        "conflict": dot < 0.0,
        "projection_coefficient": coeff,
        "dot_with_source_after_projection": projected_source_dot,
        "projected_gradient_l2": math.sqrt(projected_sq),
        "projected_over_v5_gradient_l2": math.sqrt(projected_sq) / math.sqrt(v5_sq),
    }


def _v5_batch_gradient(model, images, labels) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    model.train()
    named = v52k._named_params(model)
    names = [name for name, _parameter in named]
    params = [parameter for _name, parameter in named]
    logits = model(images)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
    if not bool(torch.isfinite(loss).item()):
        _fail("V5 minibatch produced non-finite loss")
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
    if any(parameter.grad is not None for parameter in params):
        _fail("torch.autograd.grad unexpectedly populated .grad during V5-2L")
    return {name: grad.detach().cpu().clone() for name, grad in zip(names, grads)}


def _apply_projected_step(
    model,
    *,
    v5_grad: Mapping[str, object],
    source_grad: Mapping[str, object],
) -> dict[str, float | bool]:
    torch, _nn = v52b._import_torch()
    summary = _projection_summary(v5_grad, source_grad)
    coeff = float(summary["projection_coefficient"])
    named = dict(v52k._named_params(model))
    if set(named) != set(v5_grad):
        _fail("model parameter keys changed before projected step")
    with torch.no_grad():
        for name in sorted(named):
            parameter = named[name]
            target = v5_grad[name].detach().cpu().to(dtype=torch.float64)
            source = source_grad[name].detach().cpu().to(dtype=torch.float64)
            update = target + coeff * source
            if not bool(torch.isfinite(update).all().item()):
                _fail("projected update contains non-finite values")
            parameter.add_(update.to(dtype=parameter.dtype), alpha=-LEARNING_RATE)
    return summary


def _candidate_path(target_dir: Path, digit: str) -> Path:
    return target_dir / f"digit{digit}_v5_2l_projected_candidate.pt"


def _load_candidate(
    path: Path,
    *,
    digit: str,
    slot_manifest_sha256: str,
    v52k_report_sha256: str,
):
    torch, _nn = v52b._import_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise MeterV5_2LError(f"cannot load V5-2L candidate {digit}-AI") from exc
    if not isinstance(payload, Mapping):
        _fail("V5-2L checkpoint payload must be a mapping")
    metadata = payload.get("metadata")
    state = payload.get("model_state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
        _fail("V5-2L checkpoint missing state/metadata")
    source_sha = v52b.DIGIT2_SHA256 if digit == "2" else v52b.DIGIT3_SHA256
    expected = {
        "schema": SCHEMA,
        "role": f"digit-{digit}-v5-2l-projected-candidate",
        "source_checkpoint_sha256": source_sha,
        "slot_manifest_sha256": slot_manifest_sha256,
        "v5_2k_report_sha256": v52k_report_sha256,
        "positive_weight": POS_WEIGHT,
        "learning_rate": LEARNING_RATE,
        "v5_batch_size": V5_BATCH_SIZE,
        "epochs": EPOCHS,
        "updates": EXPECTED_UPDATES_PER_SPECIALIST,
        "update_method": "direct-projected-sgd",
        "weight_decay": 0.0,
        "momentum": 0.0,
        "threshold": v52b.FROZEN_THRESHOLDS[digit],
        "threshold_tuned": False,
        "diagnostic_seed_gradient_updates": 0,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            _fail(f"V5-2L candidate {digit}-AI metadata changed: {key}")
    model = v52b._build_digit_model().cpu()
    model.load_state_dict(dict(state), strict=True)
    if metadata.get("state_fingerprint") != v52b._state_fingerprint(model):
        _fail(f"V5-2L candidate {digit}-AI state fingerprint mismatch")
    model.eval()
    return model


def _metrics(model, images, labels, *, threshold: float) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(images)).cpu()
    if not bool(torch.isfinite(probabilities).all().item()):
        _fail("candidate probabilities are non-finite")
    return v52b._binary_counts(probabilities, labels.cpu(), threshold)


def train_projected_repair_v1(
    data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    confirmation: str,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    if confirmation != APPROVAL_TOKEN:
        _fail("exact V5-2L approval token missing")
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    v52k_path, _v52k_report = _verify_v52k_evidence(root)
    manifest_path, v5_rows = v52k._verify_v5_train(root)
    if len(v5_rows) != EXPECTED_V5_COUNT:
        _fail("V5 adaptation TRAIN count changed")

    historical_rows, d10_meter = v52k.forensic._historical_train_records(
        m4a_root=Path(m4a_root), d10_root=Path(d10_root)
    )
    counts = Counter(str(row.get("digit_label")) for row in historical_rows)
    if dict(counts) != v52k.forensic.EXPECTED_M4A_TRAIN_COUNTS:
        _fail("historical TRAIN identity changed")

    frozen_paths = {"2": Path(digit2_frozen), "3": Path(digit3_frozen)}
    expected_sha = {"2": v52b.DIGIT2_SHA256, "3": v52b.DIGIT3_SHA256}
    models: dict[str, object] = {}
    for digit in ("2", "3"):
        if v52b._sha_file(frozen_paths[digit]) != expected_sha[digit]:
            _fail(f"frozen {digit}-AI checkpoint SHA changed")
        models[digit] = ret_legacy._frozen_model(frozen_paths[digit], digit=digit)

    target_dir = ann_dir / CANDIDATE_DIR_NAME
    training_path = ann_dir / TRAINING_REPORT_NAME
    if target_dir.exists() or training_path.exists():
        _fail("refusing to overwrite existing V5-2L training evidence")

    torch, _nn = v52b._import_torch()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    v5_images = torch.stack(
        [v52b._tensor_from_crop(ann_dir / row["crop_relpath"]) for row in v5_rows],
        dim=0,
    )
    v5_labels = {
        digit: torch.tensor([float(row[f"label_digit{digit}"]) for row in v5_rows], dtype=torch.float32)
        for digit in ("2", "3")
    }

    epoch_records: dict[str, list[dict[str, object]]] = {"2": [], "3": []}
    updates = {"2": 0, "3": 0}
    conflict_updates = {"2": 0, "3": 0}

    for epoch in range(EPOCHS):
        source_grads, source_losses = v52k._historical_gradients(
            historical_rows=historical_rows,
            d10_meter=d10_meter,
            d10_root=Path(d10_root),
            models=models,
            progress=(
                (lambda processed, total, phase, e=epoch: progress(
                    e * total + processed,
                    EPOCHS * total,
                    f"v5-2l-source-gradient-epoch-{e + 1}",
                )) if progress is not None else None
            ),
        )
        for digit in ("2", "3"):
            model = models[digit]
            seed = MASTER_SEED + int(digit) + epoch
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            order = torch.randperm(EXPECTED_V5_COUNT, generator=generator)
            if sorted(order.tolist()) != list(range(EXPECTED_V5_COUNT)):
                _fail("deterministic V5 epoch order is not a full permutation")

            batch_records: list[dict[str, object]] = []
            batch_count = 0
            for start in range(0, EXPECTED_V5_COUNT, V5_BATCH_SIZE):
                idx = order[start:start + V5_BATCH_SIZE]
                grad = _v5_batch_gradient(model, v5_images[idx], v5_labels[digit][idx])
                summary = _apply_projected_step(
                    model,
                    v5_grad=grad,
                    source_grad=source_grads[digit],
                )
                batch_count += 1
                updates[digit] += 1
                if bool(summary["conflict"]):
                    conflict_updates[digit] += 1
                batch_records.append({"batch": batch_count, **summary})
                if progress is not None:
                    progress(
                        updates[digit],
                        EXPECTED_UPDATES_PER_SPECIALIST,
                        f"v5-2l-training-{digit}-AI",
                    )
            if batch_count != EXPECTED_V5_BATCHES_PER_EPOCH:
                _fail(f"{digit}-AI V5 batch count changed: {batch_count}")
            epoch_records[digit].append({
                "epoch": epoch + 1,
                "historical_source_mean_bce_at_epoch_start": source_losses[digit],
                "batch_count": batch_count,
                "conflict_batch_count": sum(bool(row["conflict"]) for row in batch_records),
                "projection_coefficient_min": min(float(row["projection_coefficient"]) for row in batch_records),
                "projection_coefficient_max": max(float(row["projection_coefficient"]) for row in batch_records),
                "projected_over_v5_gradient_l2_min": min(float(row["projected_over_v5_gradient_l2"]) for row in batch_records),
                "projected_over_v5_gradient_l2_max": max(float(row["projected_over_v5_gradient_l2"]) for row in batch_records),
                "batches": batch_records,
            })

    if updates != {"2": EXPECTED_UPDATES_PER_SPECIALIST, "3": EXPECTED_UPDATES_PER_SPECIALIST}:
        _fail(f"V5-2L update count changed: {updates}")

    target_dir.mkdir(parents=True, exist_ok=False)
    slot_sha = v52b._sha_file(manifest_path)
    v52k_sha = v52b._sha_file(v52k_path)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "approval_token_verified": True,
        "slot_manifest_sha256": slot_sha,
        "v5_2k_report_sha256": v52k_sha,
        "source_checkpoint_sha256": dict(expected_sha),
        "recipe": {
            "positive_weight": POS_WEIGHT,
            "learning_rate": LEARNING_RATE,
            "v5_batch_size": V5_BATCH_SIZE,
            "epochs": EPOCHS,
            "updates_per_specialist": EXPECTED_UPDATES_PER_SPECIALIST,
            "source_reference": "full-M4A-TRAIN-mean-gradient-recomputed-once-per-epoch",
            "projection": "A-GEM-style-full-parameter-source-safe-half-space",
            "update_method": "direct-projected-sgd",
            "momentum": 0.0,
            "weight_decay": 0.0,
            "gradient_clipping": False,
            "gradient_renormalization": False,
            "checkpoint_selection": "fixed-final-epoch-no-sweep",
        },
        "v5_train_slot_count": EXPECTED_V5_COUNT,
        "historical_source_record_count": len(historical_rows),
        "diagnostic_seed_gradient_updates": 0,
        "updates_per_specialist": dict(updates),
        "conflict_updates_per_specialist": dict(conflict_updates),
        "epoch_records": epoch_records,
        "threshold_tuning": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "candidates": {},
    }

    for digit in ("2", "3"):
        model = models[digit]
        final_v5_metrics = _metrics(
            model,
            v5_images,
            v5_labels[digit],
            threshold=v52b.FROZEN_THRESHOLDS[digit],
        )
        state_fp = v52b._state_fingerprint(model)
        candidate_path = _candidate_path(target_dir, digit)
        payload = {
            "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "metadata": {
                "schema": SCHEMA,
                "role": f"digit-{digit}-v5-2l-projected-candidate",
                "source_checkpoint_sha256": expected_sha[digit],
                "slot_manifest_sha256": slot_sha,
                "v5_2k_report_sha256": v52k_sha,
                "positive_weight": POS_WEIGHT,
                "learning_rate": LEARNING_RATE,
                "v5_batch_size": V5_BATCH_SIZE,
                "epochs": EPOCHS,
                "updates": EXPECTED_UPDATES_PER_SPECIALIST,
                "update_method": "direct-projected-sgd",
                "weight_decay": 0.0,
                "momentum": 0.0,
                "state_fingerprint": state_fp,
                "threshold": v52b.FROZEN_THRESHOLDS[digit],
                "threshold_tuned": False,
                "diagnostic_seed_gradient_updates": 0,
                "v5_validation_opened": False,
                "final_holdout_locked": True,
            },
        }
        torch.save(payload, candidate_path)
        report["candidates"][digit] = {
            "candidate_path": str(candidate_path),
            "candidate_sha256": v52b._sha_file(candidate_path),
            "state_fingerprint": state_fp,
            "v5_adaptation_train_metrics_at_frozen_threshold": final_v5_metrics,
            "updates": updates[digit],
            "conflict_updates": conflict_updates[digit],
        }

    v51._atomic_write_json(training_path, report)
    return report


def run_historical_retention_gate_v1(
    data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    digit4_frozen: str | Path,
    digit2_candidate: str | Path,
    digit3_candidate: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    output = ann_dir / RETENTION_REPORT_NAME
    if output.exists():
        _fail("refusing to overwrite existing V5-2L retention evidence")
    training = v52b._read_json(ann_dir / TRAINING_REPORT_NAME)
    if training.get("schema") != SCHEMA:
        _fail("V5-2L training report missing/wrong schema")
    slot_sha = str(training.get("slot_manifest_sha256"))
    v52k_sha = str(training.get("v5_2k_report_sha256"))

    frozen_paths = {"2": Path(digit2_frozen), "3": Path(digit3_frozen), "4": Path(digit4_frozen)}
    candidate_paths = {"2": Path(digit2_candidate), "3": Path(digit3_candidate)}
    expected_frozen = {"2": v52b.DIGIT2_SHA256, "3": v52b.DIGIT3_SHA256, "4": v52b.DIGIT4_SHA256}
    for digit in ("2", "3", "4"):
        if v52b._sha_file(frozen_paths[digit]) != expected_frozen[digit]:
            _fail(f"retention frozen {digit}-AI SHA changed")
    for digit in ("2", "3"):
        expected_candidate = training.get("candidates", {}).get(digit, {}).get("candidate_sha256")
        if expected_candidate != v52b._sha_file(candidate_paths[digit]):
            _fail(f"retention {digit}-AI candidate differs from training report")

    validation, d10_meter = ret_legacy._load_manifests(m4a_root=Path(m4a_root), d10_root=Path(d10_root))
    images, labels = ret_legacy._prepare_inputs(
        validation=validation,
        d10_meter=d10_meter,
        d10_root=Path(d10_root),
        progress=progress,
    )
    frozen_metrics: dict[str, dict[str, object]] = {}
    for digit in ("2", "3", "4"):
        probs = ret_legacy._probabilities(
            ret_legacy._frozen_model(frozen_paths[digit], digit=digit),
            images,
            progress=progress,
            phase=f"v5-2l-frozen-{digit}-AI-retention-self-check",
        )
        metrics = ret_legacy._binary_counts(
            probs,
            ret_legacy._truth_tensor(labels, digit),
            v52b.FROZEN_THRESHOLDS[digit],
        )
        frozen_metrics[digit] = metrics
        expected = ret_v2.EXPECTED_FROZEN_COUNTS[digit]
        if any(metrics[key] != expected[key] for key in ("tp", "fp", "fn", "tn")):
            _fail(f"historical validation reproduction failed for {digit}-AI: {metrics}")

    candidate_metrics: dict[str, dict[str, object]] = {}
    for digit in ("2", "3"):
        model = _load_candidate(
            candidate_paths[digit],
            digit=digit,
            slot_manifest_sha256=slot_sha,
            v52k_report_sha256=v52k_sha,
        )
        probs = ret_legacy._probabilities(
            model,
            images,
            progress=progress,
            phase=f"v5-2l-candidate-{digit}-AI-retention",
        )
        candidate_metrics[digit] = ret_legacy._binary_counts(
            probs,
            ret_legacy._truth_tensor(labels, digit),
            v52b.FROZEN_THRESHOLDS[digit],
        )

    gate = ret_legacy.evaluate_retention_gate_v1(
        frozen_metrics=frozen_metrics,
        candidate_metrics=candidate_metrics,
    )
    report = {
        "schema": SCHEMA,
        "gate_kind": "historical-retention-first",
        "historical_pixel_path_reproduced": True,
        "validation_record_count": len(validation),
        "frozen_metrics": frozen_metrics,
        "candidate_metrics": candidate_metrics,
        "gate": gate["gate"],
        "reasons": gate["reasons"],
        "per_digit_retention": gate["per_digit"],
        "thresholds": dict(v52b.FROZEN_THRESHOLDS),
        "threshold_tuning": False,
        "v5_diagnostic_authorized": gate["gate"] == "PASS",
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "production_promotion": False,
    }
    v51._atomic_write_json(output, report)
    return report


def run_v5_diagnostic_gate_v1(
    data_root: str | Path,
    *,
    digit2_candidate: str | Path,
    digit3_candidate: str | Path,
    digit4_frozen: str | Path,
) -> dict[str, object]:
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    output = ann_dir / DIAGNOSTIC_REPORT_NAME
    if output.exists():
        _fail("refusing to overwrite existing V5-2L diagnostic evidence")
    retention = v52b._read_json(ann_dir / RETENTION_REPORT_NAME)
    if retention.get("schema") != SCHEMA or retention.get("gate") != "PASS":
        _fail("V5-2L diagnostic cannot run before historical retention PASS")
    training = v52b._read_json(ann_dir / TRAINING_REPORT_NAME)
    slot_sha = str(training.get("slot_manifest_sha256"))
    v52k_sha = str(training.get("v5_2k_report_sha256"))
    candidate_paths = {"2": Path(digit2_candidate), "3": Path(digit3_candidate)}
    for digit in ("2", "3"):
        expected = training.get("candidates", {}).get(digit, {}).get("candidate_sha256")
        if expected != v52b._sha_file(candidate_paths[digit]):
            _fail(f"diagnostic {digit}-AI candidate differs from training report")
    model2 = _load_candidate(candidate_paths["2"], digit="2", slot_manifest_sha256=slot_sha, v52k_report_sha256=v52k_sha)
    model3 = _load_candidate(candidate_paths["3"], digit="3", slot_manifest_sha256=slot_sha, v52k_report_sha256=v52k_sha)
    if v52b._sha_file(Path(digit4_frozen)) != v52b.DIGIT4_SHA256:
        _fail("4-AI frozen checkpoint SHA changed")
    model4 = ret_legacy._frozen_model(Path(digit4_frozen), digit="4")

    _manifest_path, rows, _slot_audit = v52b.verify_slot_manifest_v1(root)
    diagnostic = [row for row in rows if row.get("data_role") == "diagnostic_seed"]
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in diagnostic:
        grouped.setdefault(row["sample_id"], {})[row["slot_role"]] = row
    if len(grouped) != 30 or any(set(slots) != {"numerator", "denominator"} for slots in grouped.values()):
        _fail("first-30 diagnostic identity changed")

    per_meter_pass: Counter[str] = Counter()
    denominator_exact4 = 0
    samples: list[dict[str, object]] = []
    for sample_id in sorted(grouped, key=lambda sid: int(grouped[sid]["numerator"]["sample_index"])):
        slots = grouped[sample_id]
        meter = slots["numerator"]["meter"]
        expected_num = meter.split("/")[0]
        slot_results: dict[str, dict[str, object]] = {}
        for role in ("numerator", "denominator"):
            crop_path = ann_dir / slots[role]["crop_relpath"]
            probabilities = {
                "2": v52b._probability(model2, crop_path),
                "3": v52b._probability(model3, crop_path),
                "4": v52b._probability(model4, crop_path),
            }
            hits = [digit for digit in ("2", "3", "4") if probabilities[digit] >= v52b.FROZEN_THRESHOLDS[digit]]
            slot_results[role] = {"probabilities": probabilities, "hits": hits}
        numerator_ok = slot_results["numerator"]["hits"] == [expected_num]
        denominator_ok = slot_results["denominator"]["hits"] == ["4"]
        meter_pass = bool(numerator_ok and denominator_ok)
        if meter_pass:
            per_meter_pass[meter] += 1
        if denominator_ok:
            denominator_exact4 += 1
        samples.append({
            "sample_id": sample_id,
            "meter": meter,
            "numerator": slot_results["numerator"],
            "denominator": slot_results["denominator"],
            "meter_pass": meter_pass,
        })

    required = {"2/4": 8, "3/4": 8, "4/4": 9}
    reasons = [
        f"{meter}_PASS_BELOW_{minimum}_OF_10"
        for meter, minimum in required.items()
        if per_meter_pass[meter] < minimum
    ]
    if denominator_exact4 < 26:
        reasons.append("DENOMINATOR_EXACT4_BELOW_26_OF_30")
    report = {
        "schema": SCHEMA,
        "gate_kind": "v5-first30-after-retention-pass",
        "diagnostic_seed_count": 30,
        "diagnostic_seed_gradient_updates": 0,
        "per_meter_pass": {meter: per_meter_pass[meter] for meter in ("2/4", "3/4", "4/4")},
        "denominator_exact4": denominator_exact4,
        "gate": "PASS" if not reasons else "HOLD",
        "reasons": reasons,
        "thresholds": dict(v52b.FROZEN_THRESHOLDS),
        "threshold_tuning": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "production_promotion": False,
        "samples": samples,
    }
    v51._atomic_write_json(output, report)
    return report


def run_projected_gradient_repair_v1(
    data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    digit4_frozen: str | Path,
    confirmation: str,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    for name in (TRAINING_REPORT_NAME, RETENTION_REPORT_NAME, DIAGNOSTIC_REPORT_NAME, FINAL_REPORT_NAME):
        if (ann_dir / name).exists():
            _fail(f"V5-2L evidence already exists; refusing rerun: {ann_dir / name}")
    if (ann_dir / CANDIDATE_DIR_NAME).exists():
        _fail("V5-2L candidate directory already exists; refusing rerun")

    training = train_projected_repair_v1(
        root,
        m4a_root=m4a_root,
        d10_root=d10_root,
        digit2_frozen=digit2_frozen,
        digit3_frozen=digit3_frozen,
        confirmation=confirmation,
        progress=progress,
    )
    candidates = training["candidates"]
    digit2_candidate = str(candidates["2"]["candidate_path"])
    digit3_candidate = str(candidates["3"]["candidate_path"])
    retention = run_historical_retention_gate_v1(
        root,
        m4a_root=m4a_root,
        d10_root=d10_root,
        digit2_frozen=digit2_frozen,
        digit3_frozen=digit3_frozen,
        digit4_frozen=digit4_frozen,
        digit2_candidate=digit2_candidate,
        digit3_candidate=digit3_candidate,
        progress=progress,
    )
    diagnostic: dict[str, object] | None = None
    if retention["gate"] == "PASS":
        diagnostic = run_v5_diagnostic_gate_v1(
            root,
            digit2_candidate=digit2_candidate,
            digit3_candidate=digit3_candidate,
            digit4_frozen=digit4_frozen,
        )
    overall = retention["gate"] == "PASS" and diagnostic is not None and diagnostic["gate"] == "PASS"
    final = {
        "schema": SCHEMA,
        "training_completed": True,
        "candidate_sha256": {
            "2": candidates["2"]["candidate_sha256"],
            "3": candidates["3"]["candidate_sha256"],
        },
        "updates_per_specialist": dict(training["updates_per_specialist"]),
        "historical_retention_gate": retention["gate"],
        "historical_retention_reasons": retention["reasons"],
        "v5_diagnostic_executed": diagnostic is not None,
        "v5_diagnostic_gate": diagnostic["gate"] if diagnostic is not None else "NOT_RUN",
        "v5_diagnostic_reasons": diagnostic["reasons"] if diagnostic is not None else ["HISTORICAL_RETENTION_NOT_PASS"],
        "overall_gate": "PASS" if overall else "HOLD",
        "automatic_second_configuration": False,
        "threshold_tuning": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "reserve_v5_train_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "resolver_wiring": False,
        "production_promotion": False,
    }
    v51._atomic_write_json(ann_dir / FINAL_REPORT_NAME, final)
    return final


def automatic_second_configuration_allowed() -> bool:
    return False


def validation_opened_by_this_module() -> bool:
    return False


def final_holdout_locked() -> bool:
    return True


def production_promotion_allowed() -> bool:
    return False
