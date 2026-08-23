"""Meter V5-2P frozen-backbone fixed-bias balanced-domain head repair.

V5-2N/V5-2O established that the frozen 64D representations retain strong V5
class information while the V5 domain is badly displaced under the unchanged
shared heads.  This module executes one bounded repair: freeze every feature
parameter and the scalar head bias, optimize only the 64 head weights for
2-AI/3-AI with a deterministic full-batch equal-domain BCE objective, then run
corrected historical retention before the immutable first-30 V5 diagnostic.

No threshold, crop, BBox, spatial rule, reserve TRAIN, V5 validation,
FINAL_HOLDOUT, 4-AI, Resolver, or production authority is opened here.
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
from . import meter_v5_2m_retention_contract_v3 as ret_v3
from . import meter_v5_2n_frozen_feature_transfer_audit_v1 as v52n
from . import meter_v5_2o_frozen_head_axis_audit_v1 as v52o


SCHEMA: Final[str] = "st-omr-meter-v5-2p-fixed-bias-head-repair-v1"
APPROVAL_TOKEN: Final[str] = "V5_2P_FIXED_BIAS_HEAD_REPAIR_APPROVED"
TRAINING_REPORT_NAME: Final[str] = "v5_2p_fixed_bias_head_training_report.json"
RETENTION_REPORT_NAME: Final[str] = "v5_2p_historical_retention_v3.json"
DIAGNOSTIC_REPORT_NAME: Final[str] = "v5_2p_first30_diagnostic.json"
FINAL_REPORT_NAME: Final[str] = "v5_2p_fixed_bias_head_final_report.json"
CANDIDATE_DIR_NAME: Final[str] = "v5_2p_fixed_bias_head_candidates"

EXPECTED_V5_COUNT: Final[int] = 540
EXPECTED_HISTORICAL_COUNT: Final[int] = 26_964
EXPECTED_FEATURE_DIM: Final[int] = 64
POS_WEIGHT: Final[float] = 1.0
V5_DOMAIN_WEIGHT: Final[float] = 0.5
HISTORICAL_DOMAIN_WEIGHT: Final[float] = 0.5
LBFGS_LR: Final[float] = 1.0
LBFGS_MAX_ITER: Final[int] = 100
LBFGS_MAX_EVAL: Final[int] = 125
LBFGS_HISTORY_SIZE: Final[int] = 20
LBFGS_TOLERANCE_GRAD: Final[float] = 1e-9
LBFGS_TOLERANCE_CHANGE: Final[float] = 1e-12
LBFGS_LINE_SEARCH: Final[str] = "strong_wolfe"

ProgressCallback = Callable[[int, int, str], None]


class MeterV5_2PError(RuntimeError):
    """Raised whenever the V5-2P contract deviates and must fail closed."""


def _fail(message: str) -> None:
    raise MeterV5_2PError(message)


def safety_boundary() -> dict[str, object]:
    return {
        "single_fixed_repair_authorized": True,
        "automatic_second_configuration": False,
        "trainable_surface": "head.weight-only-64-parameters",
        "frozen_backbone": True,
        "frozen_head_bias": True,
        "runtime_threshold_tuning": False,
        "alternative_threshold_evaluated": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "old_d11_glyph_window_reused": False,
        "reserve_v5_train_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "resolver_wiring": False,
        "runtime_domain_routing": False,
        "production_promotion": False,
    }


def gate_order() -> tuple[str, str]:
    return ("historical_retention_v3", "v5_first30_diagnostic")


def objective_contract() -> dict[str, object]:
    return {
        "formula": "0.5*mean(V5_BCE_w1)+0.5*mean(HISTORICAL_BCE_w1)",
        "v5_domain_weight": V5_DOMAIN_WEIGHT,
        "historical_domain_weight": HISTORICAL_DOMAIN_WEIGHT,
        "positive_weight": POS_WEIGHT,
        "class_reweighting": False,
        "replay_ratio": None,
        "full_batch": True,
        "head_bias_trainable": False,
        "backbone_trainable": False,
    }


def solver_contract() -> dict[str, object]:
    return {
        "optimizer": "LBFGS",
        "dtype": "float64-head-optimization-copy-back-float32",
        "lr": LBFGS_LR,
        "max_iter": LBFGS_MAX_ITER,
        "max_eval": LBFGS_MAX_EVAL,
        "history_size": LBFGS_HISTORY_SIZE,
        "tolerance_grad": LBFGS_TOLERANCE_GRAD,
        "tolerance_change": LBFGS_TOLERANCE_CHANGE,
        "line_search_fn": LBFGS_LINE_SEARCH,
        "initialization": "exact-frozen-head-weight",
        "checkpoint_selection": "single-final-solver-state-no-sweep",
        "weight_decay": 0.0,
        "momentum": 0.0,
    }


def _verify_prerequisite_evidence(root: Path) -> dict[str, str]:
    """Bind V5-2P to the exact read-only conclusions that justify head-only repair."""
    ann = root / v51.ANNOTATIONS_DIR
    n_path = ann / v52n.REPORT_NAME
    o_path = ann / v52o.REPORT_NAME
    n = v52b._read_json(n_path)
    o = v52b._read_json(o_path)

    if n.get("schema") != v52n.SCHEMA or o.get("schema") != v52o.SCHEMA:
        _fail("V5-2N/V5-2O evidence schema mismatch")
    expected_sha = {"2": v52b.DIGIT2_SHA256, "3": v52b.DIGIT3_SHA256}
    for payload, name in ((n, "V5-2N"), (o, "V5-2O")):
        if payload.get("frozen_checkpoint_sha256") != expected_sha:
            _fail(f"{name} frozen checkpoint binding changed")
        if payload.get("v5_adaptation_train_slot_count") != EXPECTED_V5_COUNT:
            _fail(f"{name} V5 TRAIN surface changed")
        if payload.get("m4a_train_record_count") != EXPECTED_HISTORICAL_COUNT:
            _fail(f"{name} historical TRAIN surface changed")
        if payload.get("v5_validation_opened") is not False:
            _fail(f"{name} unexpectedly opened V5 validation")
        if payload.get("final_holdout_locked") is not True:
            _fail(f"{name} FINAL_HOLDOUT lock changed")
        if payload.get("digit4_frozen") is not True:
            _fail(f"{name} 4-AI freeze changed")

    if o.get("classifier_fit_performed") is not False:
        _fail("V5-2O unexpectedly fitted a classifier")
    if o.get("alternative_threshold_evaluated") is not False:
        _fail("V5-2O unexpectedly evaluated an alternative threshold")
    if o.get("bias_parameter_selected") is not False:
        _fail("V5-2O unexpectedly selected a bias")
    if o.get("repair_training_authorized") is not False:
        _fail("V5-2O unexpectedly authorized repair training")

    per = o.get("per_specialist")
    if not isinstance(per, Mapping):
        _fail("V5-2O per-specialist evidence missing")
    for digit in ("2", "3"):
        d = per.get(digit)
        if not isinstance(d, Mapping):
            _fail(f"V5-2O {digit}-AI evidence missing")
        v5 = d.get("v5_adaptation_train")
        cross = d.get("source_to_v5")
        if not isinstance(v5, Mapping) or not isinstance(cross, Mapping):
            _fail(f"V5-2O {digit}-AI geometry evidence missing")
        auc = float(v5.get("rank_auc"))
        if not math.isfinite(auc):
            _fail(f"V5-2O {digit}-AI AUC non-finite")
        if d.get("bias_or_threshold_selected") is not False:
            _fail(f"V5-2O {digit}-AI selected bias/threshold")
        if d.get("classifier_fit_performed") is not False:
            _fail(f"V5-2O {digit}-AI fitted classifier")
        if v5.get("all_positive_below_frozen_boundary") is not True:
            _fail(f"V5-2O {digit}-AI positive placement changed")
        if v5.get("all_negative_below_frozen_boundary") is not True:
            _fail(f"V5-2O {digit}-AI negative placement changed")
        if cross.get("class_gap_direction_preserved_along_head") is not True:
            _fail(f"V5-2O {digit}-AI head-axis class direction changed")

        if digit == "2":
            if auc < 0.999999999:
                _fail("2-AI V5 head-axis ranking no longer effectively perfect")
            if d.get("same_frozen_head_direction_strictly_separates_v5_train") is not True:
                _fail("2-AI V5 strict head-axis separation evidence changed")
            if float(v5.get("strict_separation_gap_logit")) <= 0.0:
                _fail("2-AI V5 strict gap is no longer positive")
        else:
            if auc < 0.999:
                _fail("3-AI V5 head-axis ranking is weaker than approved evidence")
            if float(v5.get("strict_separation_gap_logit")) > 0.0:
                _fail("3-AI strict-gap regime changed; V5-2P preregistration no longer applies")

    return {
        "v5_2n_report_sha256": v52b._sha_file(n_path),
        "v5_2o_report_sha256": v52b._sha_file(o_path),
    }


def _balanced_domain_bce_v1(*, v5_logits, v5_targets, historical_logits, historical_targets):
    """Exact equal-domain BCE objective; class frequencies remain empirical."""
    torch, _nn = v52b._import_torch()
    if v5_logits.numel() != v5_targets.numel() or historical_logits.numel() != historical_targets.numel():
        _fail("objective logit/target cardinality mismatch")
    if v5_logits.numel() == 0 or historical_logits.numel() == 0:
        _fail("objective requires both domains")
    v5_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        v5_logits, v5_targets, reduction="mean"
    )
    historical_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        historical_logits, historical_targets, reduction="mean"
    )
    total = V5_DOMAIN_WEIGHT * v5_loss + HISTORICAL_DOMAIN_WEIGHT * historical_loss
    if not bool(torch.isfinite(total).item()):
        _fail("V5-2P objective became non-finite")
    return total, v5_loss, historical_loss


def _frozen_state_snapshot(model) -> dict[str, object]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _verify_only_head_weight_changed(model, frozen_state: Mapping[str, object]) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    current = model.state_dict()
    if set(current) != set(frozen_state):
        _fail("candidate state keys changed")
    changed: list[str] = []
    for name in sorted(current):
        before = frozen_state[name].detach().cpu()
        after = current[name].detach().cpu()
        if before.shape != after.shape or before.dtype != after.dtype:
            _fail(f"candidate tensor contract changed: {name}")
        if not torch.equal(before, after):
            changed.append(name)
    illegal = [name for name in changed if name != "head.weight"]
    if illegal:
        _fail(f"frozen tensor changed during V5-2P: {illegal}")
    if not torch.equal(
        frozen_state["head.bias"].detach().cpu(), current["head.bias"].detach().cpu()
    ):
        _fail("head.bias changed during V5-2P")
    return {
        "changed_state_keys": changed,
        "only_head_weight_changed": not illegal,
        "backbone_bit_identical": all(
            torch.equal(frozen_state[name].detach().cpu(), current[name].detach().cpu())
            for name in current if name.startswith("features.")
        ),
        "head_bias_bit_identical": True,
    }


def _fit_fixed_bias_head(
    model,
    *,
    v5_features,
    v5_targets,
    historical_features,
    historical_targets,
) -> dict[str, object]:
    """Optimize only a float64 copy of head.weight, then copy it back once."""
    torch, _nn = v52b._import_torch()
    model.eval()
    if model.head.bias is None:
        _fail("digit specialist head bias missing")
    if model.head.weight.numel() != EXPECTED_FEATURE_DIM:
        _fail("digit specialist head weight dimension changed")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
        if parameter.grad is not None:
            _fail("unexpected pre-existing model gradient")

    x_v5 = v5_features.detach().cpu().to(dtype=torch.float64)
    y_v5 = v5_targets.detach().cpu().to(dtype=torch.float64).reshape(-1)
    x_hist = historical_features.detach().cpu().to(dtype=torch.float64)
    y_hist = historical_targets.detach().cpu().to(dtype=torch.float64).reshape(-1)
    if x_v5.shape != (EXPECTED_V5_COUNT, EXPECTED_FEATURE_DIM):
        _fail(f"V5 frozen feature shape changed: {tuple(x_v5.shape)}")
    if x_hist.shape != (EXPECTED_HISTORICAL_COUNT, EXPECTED_FEATURE_DIM):
        _fail(f"historical frozen feature shape changed: {tuple(x_hist.shape)}")
    if int(y_v5.sum().item()) != 90:
        _fail("V5 specialist positive count changed")
    if not bool(torch.isfinite(x_v5).all().item()) or not bool(torch.isfinite(x_hist).all().item()):
        _fail("non-finite frozen features")

    frozen_weight = model.head.weight.detach().cpu().reshape(-1).to(dtype=torch.float64)
    frozen_bias = float(model.head.bias.detach().cpu().reshape(-1)[0].item())
    weight = torch.nn.Parameter(frozen_weight.clone())

    def logits(x):
        return x @ weight + frozen_bias

    with torch.no_grad():
        initial_total, initial_v5, initial_hist = _balanced_domain_bce_v1(
            v5_logits=logits(x_v5),
            v5_targets=y_v5,
            historical_logits=logits(x_hist),
            historical_targets=y_hist,
        )
        initial_weight_norm = float(torch.linalg.vector_norm(weight).item())

    optimizer = torch.optim.LBFGS(
        [weight],
        lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
        max_eval=LBFGS_MAX_EVAL,
        tolerance_grad=LBFGS_TOLERANCE_GRAD,
        tolerance_change=LBFGS_TOLERANCE_CHANGE,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn=LBFGS_LINE_SEARCH,
    )
    closure_evaluations = 0

    def closure():
        nonlocal closure_evaluations
        optimizer.zero_grad(set_to_none=True)
        total, _v5, _hist = _balanced_domain_bce_v1(
            v5_logits=logits(x_v5),
            v5_targets=y_v5,
            historical_logits=logits(x_hist),
            historical_targets=y_hist,
        )
        total.backward()
        closure_evaluations += 1
        if weight.grad is None or not bool(torch.isfinite(weight.grad).all().item()):
            _fail("LBFGS produced missing/non-finite head-weight gradient")
        return total

    optimizer.step(closure)

    with torch.no_grad():
        final_total, final_v5, final_hist = _balanced_domain_bce_v1(
            v5_logits=logits(x_v5),
            v5_targets=y_v5,
            historical_logits=logits(x_hist),
            historical_targets=y_hist,
        )
        if not bool(torch.isfinite(weight).all().item()):
            _fail("LBFGS produced non-finite head weight")
        if float(final_total.item()) > float(initial_total.item()) + 1e-10:
            _fail("fixed final LBFGS state increased the preregistered objective")
        final_weight_norm = float(torch.linalg.vector_norm(weight).item())
        delta_norm = float(torch.linalg.vector_norm(weight - frozen_weight).item())
        model.head.weight.copy_(weight.to(dtype=model.head.weight.dtype).reshape_as(model.head.weight))

    return {
        "initial_total_loss": float(initial_total.item()),
        "initial_v5_mean_bce": float(initial_v5.item()),
        "initial_historical_mean_bce": float(initial_hist.item()),
        "final_total_loss": float(final_total.item()),
        "final_v5_mean_bce": float(final_v5.item()),
        "final_historical_mean_bce": float(final_hist.item()),
        "closure_evaluations": closure_evaluations,
        "initial_head_weight_l2": initial_weight_norm,
        "final_head_weight_l2": final_weight_norm,
        "head_weight_delta_l2": delta_norm,
        "head_bias": frozen_bias,
    }


def _feature_metrics(model, features, targets, *, threshold: float) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    model.eval()
    with torch.no_grad():
        logits = model.head(features.to(dtype=model.head.weight.dtype)).squeeze(1)
        probs = torch.sigmoid(logits).cpu()
    if not bool(torch.isfinite(probs).all().item()):
        _fail("candidate feature inference produced non-finite probabilities")
    return v52b._binary_counts(probs, targets.cpu(), threshold)


def _candidate_path(target_dir: Path, digit: str) -> Path:
    return target_dir / f"digit{digit}_v5_2p_fixed_bias_head_candidate.pt"


def _save_candidate(
    *,
    model,
    path: Path,
    digit: str,
    source_sha: str,
    manifest_sha: str,
    evidence_sha: Mapping[str, str],
    invariants: Mapping[str, object],
) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    state_fp = v52b._state_fingerprint(model)
    payload = {
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "metadata": {
            "schema": SCHEMA,
            "role": f"digit-{digit}-v5-2p-fixed-bias-head-candidate",
            "source_checkpoint_sha256": source_sha,
            "slot_manifest_sha256": manifest_sha,
            **dict(evidence_sha),
            "trainable_surface": "head.weight-only-64-parameters",
            "head_bias_frozen": True,
            "backbone_frozen": True,
            "state_fingerprint": state_fp,
            "threshold": v52b.FROZEN_THRESHOLDS[digit],
            "threshold_tuned": False,
            "diagnostic_seed_gradient_updates": 0,
            "v5_validation_opened": False,
            "final_holdout_locked": True,
            "invariants": dict(invariants),
            "objective_contract": objective_contract(),
            "solver_contract": solver_contract(),
        },
    }
    torch.save(payload, path)
    return {
        "candidate_path": str(path),
        "candidate_sha256": v52b._sha_file(path),
        "state_fingerprint": state_fp,
    }


def _load_candidate(path: Path, *, digit: str, training_report: Mapping[str, object]):
    torch, _nn = v52b._import_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise MeterV5_2PError(f"cannot load V5-2P {digit}-AI candidate") from exc
    if not isinstance(payload, Mapping):
        _fail("candidate payload must be mapping")
    metadata = payload.get("metadata")
    state = payload.get("model_state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
        _fail("candidate state/metadata missing")
    expected_source = v52b.DIGIT2_SHA256 if digit == "2" else v52b.DIGIT3_SHA256
    expected = {
        "schema": SCHEMA,
        "role": f"digit-{digit}-v5-2p-fixed-bias-head-candidate",
        "source_checkpoint_sha256": expected_source,
        "slot_manifest_sha256": training_report.get("slot_manifest_sha256"),
        "trainable_surface": "head.weight-only-64-parameters",
        "head_bias_frozen": True,
        "backbone_frozen": True,
        "threshold": v52b.FROZEN_THRESHOLDS[digit],
        "threshold_tuned": False,
        "diagnostic_seed_gradient_updates": 0,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "objective_contract": objective_contract(),
        "solver_contract": solver_contract(),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            _fail(f"candidate metadata changed for {digit}-AI: {key}")
    model = v52b._build_digit_model().cpu()
    model.load_state_dict(dict(state), strict=True)
    if metadata.get("state_fingerprint") != v52b._state_fingerprint(model):
        _fail(f"candidate {digit}-AI state fingerprint mismatch")
    model.eval()
    return model


def train_fixed_bias_head_repair_v1(
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
        _fail("exact V5-2P approval token missing")
    root = Path(data_root)
    ann = root / v51.ANNOTATIONS_DIR
    evidence_sha = _verify_prerequisite_evidence(root)

    models = v52n._frozen_models(
        digit2_frozen=Path(digit2_frozen),
        digit3_frozen=Path(digit3_frozen),
    )
    manifest_path, _rows, v5_features, v5_targets, _frozen_v5_metrics = v52n._v5_surface(root, models)
    historical_features, historical_targets = v52n._historical_surface(
        m4a_root=Path(m4a_root),
        d10_root=Path(d10_root),
        models=models,
        progress=progress,
    )

    target_dir = ann / CANDIDATE_DIR_NAME
    training_path = ann / TRAINING_REPORT_NAME
    if target_dir.exists() or training_path.exists():
        _fail("refusing to overwrite V5-2P training evidence")

    torch, _nn = v52b._import_torch()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    target_dir.mkdir(parents=True, exist_ok=False)
    manifest_sha = v52b._sha_file(manifest_path)

    report: dict[str, object] = {
        "schema": SCHEMA,
        "approval_token_verified": True,
        "slot_manifest_sha256": manifest_sha,
        **evidence_sha,
        "source_checkpoint_sha256": {
            "2": v52b.DIGIT2_SHA256,
            "3": v52b.DIGIT3_SHA256,
        },
        "v5_train_slot_count": EXPECTED_V5_COUNT,
        "historical_train_record_count": EXPECTED_HISTORICAL_COUNT,
        "feature_dim": EXPECTED_FEATURE_DIM,
        "objective_contract": objective_contract(),
        "solver_contract": solver_contract(),
        "diagnostic_seed_gradient_updates": 0,
        "candidates": {},
        **safety_boundary(),
    }

    for digit in ("2", "3"):
        model = models[digit]
        frozen_state = _frozen_state_snapshot(model)
        fit = _fit_fixed_bias_head(
            model,
            v5_features=v5_features[digit],
            v5_targets=v5_targets[digit],
            historical_features=historical_features[digit],
            historical_targets=historical_targets[digit],
        )
        invariants = _verify_only_head_weight_changed(model, frozen_state)
        v5_metrics = _feature_metrics(
            model,
            v5_features[digit],
            v5_targets[digit],
            threshold=v52b.FROZEN_THRESHOLDS[digit],
        )
        historical_metrics = _feature_metrics(
            model,
            historical_features[digit],
            historical_targets[digit],
            threshold=v52b.FROZEN_THRESHOLDS[digit],
        )
        path = _candidate_path(target_dir, digit)
        saved = _save_candidate(
            model=model,
            path=path,
            digit=digit,
            source_sha=(v52b.DIGIT2_SHA256 if digit == "2" else v52b.DIGIT3_SHA256),
            manifest_sha=manifest_sha,
            evidence_sha=evidence_sha,
            invariants=invariants,
        )
        report["candidates"][digit] = {
            **saved,
            "fit": fit,
            "state_invariants": invariants,
            "v5_adaptation_train_metrics_at_frozen_threshold": v5_metrics,
            "historical_train_metrics_at_frozen_threshold": historical_metrics,
        }

    v51._atomic_write_json(training_path, report)
    return report


def run_historical_retention_gate_v3(
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
    ann = root / v51.ANNOTATIONS_DIR
    output = ann / RETENTION_REPORT_NAME
    if output.exists():
        _fail("refusing to overwrite V5-2P retention evidence")
    training = v52b._read_json(ann / TRAINING_REPORT_NAME)
    if training.get("schema") != SCHEMA:
        _fail("V5-2P training report missing/wrong schema")

    frozen_paths = {
        "2": Path(digit2_frozen),
        "3": Path(digit3_frozen),
        "4": Path(digit4_frozen),
    }
    candidate_paths = {"2": Path(digit2_candidate), "3": Path(digit3_candidate)}
    expected_frozen_sha = {
        "2": v52b.DIGIT2_SHA256,
        "3": v52b.DIGIT3_SHA256,
        "4": v52b.DIGIT4_SHA256,
    }
    for digit in ("2", "3", "4"):
        if v52b._sha_file(frozen_paths[digit]) != expected_frozen_sha[digit]:
            _fail(f"retention frozen {digit}-AI SHA changed")
    for digit in ("2", "3"):
        expected_candidate = training.get("candidates", {}).get(digit, {}).get("candidate_sha256")
        if expected_candidate != v52b._sha_file(candidate_paths[digit]):
            _fail(f"retention {digit}-AI candidate differs from training report")

    validation, d10_meter = ret_legacy._load_manifests(
        m4a_root=Path(m4a_root), d10_root=Path(d10_root)
    )
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
            phase=f"v5-2p-frozen-{digit}-AI-retention-self-check",
        )
        metrics = ret_legacy._binary_counts(
            probs,
            ret_legacy._truth_tensor(labels, digit),
            v52b.FROZEN_THRESHOLDS[digit],
        )
        expected = ret_v2.EXPECTED_FROZEN_COUNTS[digit]
        if any(metrics[key] != expected[key] for key in ("tp", "fp", "fn", "tn")):
            _fail(f"historical frozen oracle reproduction failed for {digit}-AI: {metrics}")
        frozen_metrics[digit] = metrics

    candidate_metrics: dict[str, dict[str, object]] = {}
    for digit in ("2", "3"):
        model = _load_candidate(candidate_paths[digit], digit=digit, training_report=training)
        probs = ret_legacy._probabilities(
            model,
            images,
            progress=progress,
            phase=f"v5-2p-candidate-{digit}-AI-retention",
        )
        candidate_metrics[digit] = ret_legacy._binary_counts(
            probs,
            ret_legacy._truth_tensor(labels, digit),
            v52b.FROZEN_THRESHOLDS[digit],
        )

    gate = ret_v3.evaluate_retention_gate_v3(
        frozen_metrics=frozen_metrics,
        candidate_metrics=candidate_metrics,
    )
    report = {
        "schema": SCHEMA,
        "gate_kind": "historical-retention-v3-first",
        "retention_contract_schema": ret_v3.SCHEMA,
        "historical_pixel_path_reproduced": True,
        "validation_record_count": len(validation),
        "frozen_metrics": frozen_metrics,
        "candidate_metrics": candidate_metrics,
        "gate": gate["gate"],
        "reasons": gate["reasons"],
        "per_digit_retention": gate["per_digit"],
        "absolute_precision_floor_used": False,
        "absolute_recall_floor_used": False,
        "thresholds": dict(v52b.FROZEN_THRESHOLDS),
        "threshold_tuned": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
    }
    v51._atomic_write_json(output, report)
    return report


def _probability(model, crop_path: Path) -> float:
    torch, _nn = v52b._import_torch()
    model.eval()
    with torch.no_grad():
        value = torch.sigmoid(model(v52b._tensor_from_crop(crop_path).unsqueeze(0))).item()
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        _fail("candidate inference produced invalid probability")
    return float(value)


def run_first30_diagnostic_v1(
    data_root: str | Path,
    *,
    digit2_candidate: str | Path,
    digit3_candidate: str | Path,
    digit4_frozen: str | Path,
) -> dict[str, object]:
    root = Path(data_root)
    ann = root / v51.ANNOTATIONS_DIR
    output = ann / DIAGNOSTIC_REPORT_NAME
    if output.exists():
        _fail("refusing to overwrite V5-2P first30 evidence")
    retention = v52b._read_json(ann / RETENTION_REPORT_NAME)
    if retention.get("gate") != "PASS":
        _fail("V5 first30 diagnostic cannot run before historical retention PASS")
    training = v52b._read_json(ann / TRAINING_REPORT_NAME)

    manifest_path, rows, _audit = v52b.verify_slot_manifest_v1(root)
    if training.get("slot_manifest_sha256") != v52b._sha_file(manifest_path):
        _fail("slot manifest changed after V5-2P training")

    candidates = {"2": Path(digit2_candidate), "3": Path(digit3_candidate)}
    for digit in ("2", "3"):
        if training.get("candidates", {}).get(digit, {}).get("candidate_sha256") != v52b._sha_file(candidates[digit]):
            _fail(f"first30 {digit}-AI candidate differs from training report")
    if v52b._sha_file(Path(digit4_frozen)) != v52b.DIGIT4_SHA256:
        _fail("first30 4-AI frozen checkpoint SHA changed")

    model2 = _load_candidate(candidates["2"], digit="2", training_report=training)
    model3 = _load_candidate(candidates["3"], digit="3", training_report=training)
    model4 = ret_legacy._frozen_model(Path(digit4_frozen), digit="4")

    diagnostic = [row for row in rows if row.get("data_role") == "diagnostic_seed"]
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in diagnostic:
        grouped.setdefault(row["sample_id"], {})[row["slot_role"]] = row
    if len(grouped) != 30 or any(set(slots) != {"numerator", "denominator"} for slots in grouped.values()):
        _fail("diagnostic manifest must contain exactly two slots for each of 30 seeds")

    samples: list[dict[str, object]] = []
    per_meter_pass: Counter[str] = Counter()
    denominator_exact4 = 0
    for sample_id in sorted(grouped, key=lambda sid: int(grouped[sid]["numerator"]["sample_index"])):
        slots = grouped[sample_id]
        meter = slots["numerator"]["meter"]
        expected_num = meter.split("/")[0]
        slot_results: dict[str, dict[str, object]] = {}
        for role in ("numerator", "denominator"):
            crop_path = ann / slots[role]["crop_relpath"]
            probabilities = {
                "2": _probability(model2, crop_path),
                "3": _probability(model3, crop_path),
                "4": _probability(model4, crop_path),
            }
            hits = [
                digit for digit in ("2", "3", "4")
                if probabilities[digit] >= v52b.FROZEN_THRESHOLDS[digit]
            ]
            slot_results[role] = {"probabilities": probabilities, "hits": hits}
        numerator_ok = slot_results["numerator"]["hits"] == [expected_num]
        denominator_ok = slot_results["denominator"]["hits"] == ["4"]
        meter_pass = bool(numerator_ok and denominator_ok)
        if denominator_ok:
            denominator_exact4 += 1
        if meter_pass:
            per_meter_pass[meter] += 1
        samples.append({
            "sample_id": sample_id,
            "meter": meter,
            "numerator": slot_results["numerator"],
            "denominator": slot_results["denominator"],
            "numerator_correct": numerator_ok,
            "denominator_correct": denominator_ok,
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
        "gate_kind": "immutable-first30-after-retention",
        "slot_manifest_sha256": v52b._sha_file(manifest_path),
        "digit2_candidate_sha256": v52b._sha_file(candidates["2"]),
        "digit3_candidate_sha256": v52b._sha_file(candidates["3"]),
        "digit4_frozen_sha256": v52b.DIGIT4_SHA256,
        "thresholds": dict(v52b.FROZEN_THRESHOLDS),
        "threshold_tuned": False,
        "diagnostic_seed_count": 30,
        "diagnostic_seed_gradient_updates": 0,
        "per_meter_pass": {meter: per_meter_pass[meter] for meter in ("2/4", "3/4", "4/4")},
        "denominator_exact4": denominator_exact4,
        "gate": "PASS" if not reasons else "HOLD",
        "reasons": reasons,
        "validation_opened": False,
        "validation_opening_authorized_for_separate_review": not reasons,
        "final_holdout_locked": True,
        "resolver_wiring_authorized": False,
        "production_promotion_authorized": False,
        "samples": samples,
    }
    v51._atomic_write_json(output, report)
    return report


def run_fixed_bias_head_repair_v1(
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
    ann = root / v51.ANNOTATIONS_DIR
    final_path = ann / FINAL_REPORT_NAME
    if final_path.exists():
        _fail("refusing to overwrite V5-2P final evidence")

    training = train_fixed_bias_head_repair_v1(
        root,
        m4a_root=m4a_root,
        d10_root=d10_root,
        digit2_frozen=digit2_frozen,
        digit3_frozen=digit3_frozen,
        confirmation=confirmation,
        progress=progress,
    )
    candidate2 = training["candidates"]["2"]["candidate_path"]
    candidate3 = training["candidates"]["3"]["candidate_path"]

    retention = run_historical_retention_gate_v3(
        root,
        m4a_root=m4a_root,
        d10_root=d10_root,
        digit2_frozen=digit2_frozen,
        digit3_frozen=digit3_frozen,
        digit4_frozen=digit4_frozen,
        digit2_candidate=candidate2,
        digit3_candidate=candidate3,
        progress=progress,
    )

    diagnostic: dict[str, object] | None = None
    if retention["gate"] == "PASS":
        diagnostic = run_first30_diagnostic_v1(
            root,
            digit2_candidate=candidate2,
            digit3_candidate=candidate3,
            digit4_frozen=digit4_frozen,
        )

    overall = "PASS" if retention["gate"] == "PASS" and diagnostic is not None and diagnostic["gate"] == "PASS" else "HOLD"
    final = {
        "schema": SCHEMA,
        "overall_gate": overall,
        "historical_retention_gate": retention["gate"],
        "v5_diagnostic_gate": diagnostic["gate"] if diagnostic is not None else "NOT_RUN",
        "candidate_sha256": {
            "2": training["candidates"]["2"]["candidate_sha256"],
            "3": training["candidates"]["3"]["candidate_sha256"],
        },
        "gate_order": list(gate_order()),
        "automatic_second_configuration": False,
        "validation_opened": False,
        "validation_opening_requires_separate_review": overall == "PASS",
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "resolver_wiring_authorized": False,
        "production_promotion_authorized": False,
        "training_report": str(ann / TRAINING_REPORT_NAME),
        "retention_report": str(ann / RETENTION_REPORT_NAME),
        "diagnostic_report": str(ann / DIAGNOSTIC_REPORT_NAME) if diagnostic is not None else None,
        **safety_boundary(),
    }
    v51._atomic_write_json(final_path, final)
    return final


def production_promotion_allowed() -> bool:
    return False


def validation_opened_by_this_module() -> bool:
    return False


def final_holdout_locked() -> bool:
    return True
