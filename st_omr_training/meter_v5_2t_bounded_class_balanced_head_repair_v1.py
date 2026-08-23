"""Single-configuration V5-2T implementation of the V5-2S contract.

The implementation fits only a float64 copy of the 64D head weight. Four
TRAIN-only domain/class groups receive equal BCE coefficients and a frozen-head
proximal term confines any accepted final state to the preregistered 15-degree
trust region. Backbone, bias, thresholds, 4-AI, and closed evidence surfaces
remain unchanged.

This module stops after candidate and numerical-integrity evidence creation.
Historical retention and all V5 diagnostic/validation surfaces are separate
later gates.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Callable, Final, Mapping

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2n_frozen_feature_transfer_audit_v1 as v52n
from . import meter_v5_2p_fixed_bias_head_repair_v1 as v52p
from . import meter_v5_2p_numerical_evidence_guard_v1 as v52p_guard
from . import meter_v5_2s_bounded_class_balanced_head_contract_v1 as v52s


SCHEMA: Final[str] = "st-omr-meter-v5-2t-bounded-class-balanced-head-repair-v1"
APPROVAL_TOKEN: Final[str] = "V5_2T_SINGLE_BOUNDED_CLASS_BALANCED_RUN_APPROVED"
TRAINING_REPORT_NAME: Final[str] = "v5_2t_bounded_class_balanced_training_report.json"
CANDIDATE_DIR_NAME: Final[str] = "v5_2t_bounded_class_balanced_candidates"
TEMP_CANDIDATE_DIR_NAME: Final[str] = ".v5_2t_bounded_class_balanced_candidates.tmp"
EXPECTED_V5_COUNT: Final[int] = 540
EXPECTED_HISTORICAL_COUNT: Final[int] = 26_964
EXPECTED_V5_POSITIVE_COUNT: Final[int] = 90
EXPECTED_HISTORICAL_POSITIVE_COUNT: Final[dict[str, int]] = {"2": 1_527, "3": 1_587}
EXPECTED_FEATURE_DIM: Final[int] = 64
LOSS_TOLERANCE: Final[float] = 1e-10
ProgressCallback = Callable[[int, int, str], None]


class MeterV5_2TError(RuntimeError):
    """Raised when V5-2T departs from its exact preregistered contract."""


def _fail(message: str) -> None:
    raise MeterV5_2TError(message)


def safety_boundary() -> dict[str, object]:
    return {
        "single_fixed_training_entry": True,
        "automatic_second_configuration": False,
        "hyperparameter_sweep": False,
        "trainable_surface": "head.weight-only-64-parameters",
        "frozen_backbone": True,
        "frozen_head_bias": True,
        "runtime_threshold_tuning": False,
        "alternative_threshold_evaluated": False,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "reserve_v5_train_opened": False,
        "historical_validation_opened": False,
        "historical_retention_executed_by_this_module": False,
        "first30_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "production_promotion": False,
    }


def implementation_contract() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "prerequisite": v52s.prerequisite_evidence_contract(),
        "objective": v52s.objective_contract(),
        "solver": {
            **v52s.solver_contract(),
            "execution_authorized": True,
        },
        "gate_order": list(v52s.gate_order()),
        "actual_data_execution_requires_exact_sha_colab_harness": True,
        "colab_harness_present_in_this_stage": False,
        **safety_boundary(),
    }


def _validate_feature_surface(
    *,
    features,
    targets,
    name: str,
    expected_count: int | None,
    expected_positive_count: int | None,
) -> tuple[object, object]:
    torch, _nn = v52b._import_torch()
    x = features.detach().cpu().to(dtype=torch.float64)
    y = targets.detach().cpu().to(dtype=torch.float64).reshape(-1)
    if x.ndim != 2 or x.shape[1] != EXPECTED_FEATURE_DIM:
        _fail(f"{name} feature shape changed: {tuple(x.shape)}")
    if x.shape[0] != y.numel() or x.shape[0] == 0:
        _fail(f"{name} feature/target cardinality changed")
    if expected_count is not None and x.shape[0] != expected_count:
        _fail(f"{name} record count changed: {x.shape[0]}")
    if not bool(torch.isfinite(x).all().item()) or not bool(torch.isfinite(y).all().item()):
        _fail(f"{name} contains non-finite values")
    if not bool(torch.all((y == 0.0) | (y == 1.0)).item()):
        _fail(f"{name} targets are not binary")
    positive_count = int(y.sum().item())
    if positive_count <= 0 or positive_count >= y.numel():
        _fail(f"{name} must contain both classes")
    if expected_positive_count is not None and positive_count != expected_positive_count:
        _fail(f"{name} positive count changed: {positive_count}")
    return x, y


def _four_group_bce_torch_v1(*, v5_logits, v5_targets, historical_logits, historical_targets):
    torch, _nn = v52b._import_torch()
    groups = {
        "v5_positive": (v5_logits, v5_targets, 1.0),
        "v5_negative": (v5_logits, v5_targets, 0.0),
        "historical_positive": (historical_logits, historical_targets, 1.0),
        "historical_negative": (historical_logits, historical_targets, 0.0),
    }
    losses: dict[str, object] = {}
    counts: dict[str, int] = {}
    total = None
    for name in v52s.GROUPS:
        logits, targets, label = groups[name]
        mask = targets == label
        count = int(mask.sum().item())
        if count <= 0:
            _fail(f"{name} objective group is empty")
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[mask], targets[mask], reduction="mean"
        )
        if not bool(torch.isfinite(loss).item()):
            _fail(f"{name} BCE became non-finite")
        losses[name] = loss
        counts[name] = count
        total = v52s.GROUP_WEIGHT * loss if total is None else total + v52s.GROUP_WEIGHT * loss
    if total is None or not bool(torch.isfinite(total).item()):
        _fail("V5-2T balanced BCE became non-finite")
    return total, losses, counts


def _objective_torch_v1(
    *,
    weight,
    frozen_weight,
    frozen_bias: float,
    x_v5,
    y_v5,
    x_historical,
    y_historical,
    proximal_lambda: float,
):
    torch, _nn = v52b._import_torch()
    balanced, losses, counts = _four_group_bce_torch_v1(
        v5_logits=x_v5 @ weight + frozen_bias,
        v5_targets=y_v5,
        historical_logits=x_historical @ weight + frozen_bias,
        historical_targets=y_historical,
    )
    penalty = 0.5 * float(proximal_lambda) * torch.sum((weight - frozen_weight) ** 2)
    total = balanced + penalty
    if not bool(torch.isfinite(total).item()) or not bool(torch.isfinite(penalty).item()):
        _fail("V5-2T bounded objective became non-finite")
    return total, balanced, penalty, losses, counts


def _fit_bounded_head_v1(
    model,
    *,
    v5_features,
    v5_targets,
    historical_features,
    historical_targets,
    enforce_preregistered_counts: bool,
    expected_historical_positive_count: int | None = None,
) -> dict[str, object]:
    """Execute one deterministic full-batch solve on already-frozen features."""
    torch, _nn = v52b._import_torch()
    model.eval()
    if model.head.bias is None or model.head.weight.numel() != EXPECTED_FEATURE_DIM:
        _fail("digit specialist head contract changed")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        if parameter.grad is not None:
            _fail("unexpected pre-existing model gradient")

    x_v5, y_v5 = _validate_feature_surface(
        features=v5_features,
        targets=v5_targets,
        name="V5 TRAIN",
        expected_count=EXPECTED_V5_COUNT if enforce_preregistered_counts else None,
        expected_positive_count=(
            EXPECTED_V5_POSITIVE_COUNT if enforce_preregistered_counts else None
        ),
    )
    x_hist, y_hist = _validate_feature_surface(
        features=historical_features,
        targets=historical_targets,
        name="historical TRAIN",
        expected_count=(EXPECTED_HISTORICAL_COUNT if enforce_preregistered_counts else None),
        expected_positive_count=(
            expected_historical_positive_count if enforce_preregistered_counts else None
        ),
    )

    frozen_weight = model.head.weight.detach().cpu().reshape(-1).to(dtype=torch.float64)
    frozen_bias = float(model.head.bias.detach().cpu().reshape(-1)[0].item())
    weight = torch.nn.Parameter(frozen_weight.clone())

    with torch.no_grad():
        initial_balanced, initial_groups, group_counts = _four_group_bce_torch_v1(
            v5_logits=x_v5 @ frozen_weight + frozen_bias,
            v5_targets=y_v5,
            historical_logits=x_hist @ frozen_weight + frozen_bias,
            historical_targets=y_hist,
        )
    proximal = v52s.derive_proximal_contract_v1(
        frozen_weight=frozen_weight,
        initial_balanced_bce=float(initial_balanced.item()),
    )
    proximal_lambda = proximal["proximal_lambda"]

    optimizer = torch.optim.LBFGS(
        [weight],
        lr=v52p.LBFGS_LR,
        max_iter=v52p.LBFGS_MAX_ITER,
        max_eval=v52p.LBFGS_MAX_EVAL,
        tolerance_grad=v52p.LBFGS_TOLERANCE_GRAD,
        tolerance_change=v52p.LBFGS_TOLERANCE_CHANGE,
        history_size=v52p.LBFGS_HISTORY_SIZE,
        line_search_fn=v52p.LBFGS_LINE_SEARCH,
    )
    closure_evaluations = 0

    def closure():
        nonlocal closure_evaluations
        optimizer.zero_grad(set_to_none=True)
        total, _balanced, _penalty, _groups, _counts = _objective_torch_v1(
            weight=weight,
            frozen_weight=frozen_weight,
            frozen_bias=frozen_bias,
            x_v5=x_v5,
            y_v5=y_v5,
            x_historical=x_hist,
            y_historical=y_hist,
            proximal_lambda=proximal_lambda,
        )
        total.backward()
        closure_evaluations += 1
        if weight.grad is None or not bool(torch.isfinite(weight.grad).all().item()):
            _fail("LBFGS produced missing/non-finite head-weight gradient")
        return total

    optimizer.step(closure)
    state = optimizer.state.get(weight, {})

    final_probe = torch.nn.Parameter(weight.detach().clone())
    final_total, final_balanced, final_penalty, final_groups, final_counts = (
        _objective_torch_v1(
            weight=final_probe,
            frozen_weight=frozen_weight,
            frozen_bias=frozen_bias,
            x_v5=x_v5,
            y_v5=y_v5,
            x_historical=x_hist,
            y_historical=y_hist,
            proximal_lambda=proximal_lambda,
        )
    )
    final_total.backward()
    if final_probe.grad is None or not bool(torch.isfinite(final_probe.grad).all().item()):
        _fail("final V5-2T objective gradient is missing/non-finite")

    initial_total_value = float(initial_balanced.item())
    final_total_value = float(final_total.detach().item())
    final_balanced_value = float(final_balanced.detach().item())
    final_penalty_value = float(final_penalty.detach().item())
    if final_total_value > initial_total_value + LOSS_TOLERANCE:
        _fail("V5-2T final objective increased")

    geometry = v52s.geometry_evidence_v1(
        frozen_weight=frozen_weight,
        candidate_weight=weight.detach(),
    )
    if geometry["gate"] != "PASS":
        _fail("V5-2T solver escaped the preregistered geometry bound")

    final_gradient_l2 = float(torch.linalg.vector_norm(final_probe.grad).item())
    final_gradient_inf = float(torch.max(torch.abs(final_probe.grad)).item())
    termination = v52p_guard._termination_evidence_v1(
        n_iter=int(state.get("n_iter", 0)),
        func_evals=int(state.get("func_evals", 0)),
        closure_evaluations=closure_evaluations,
        final_gradient_inf_norm=final_gradient_inf,
        final_gradient_l2_norm=final_gradient_l2,
    )

    with torch.no_grad():
        expected_copy32 = weight.detach().to(dtype=model.head.weight.dtype)
        model.head.weight.copy_(expected_copy32.reshape_as(model.head.weight))
    copied_weight32 = model.head.weight.detach().cpu().reshape(-1)
    if not torch.equal(copied_weight32, expected_copy32.to(dtype=torch.float32)):
        _fail("V5-2T float32 copy-back is not bit-exact")
    copied_weight64 = copied_weight32.to(dtype=torch.float64)
    with torch.no_grad():
        copy_total, copy_balanced, copy_penalty, _copy_groups, copy_counts = (
            _objective_torch_v1(
                weight=copied_weight64,
                frozen_weight=frozen_weight,
                frozen_bias=frozen_bias,
                x_v5=x_v5,
                y_v5=y_v5,
                x_historical=x_hist,
                y_historical=y_hist,
                proximal_lambda=proximal_lambda,
            )
        )
    copy_total_value = float(copy_total.item())
    if copy_total_value > initial_total_value + LOSS_TOLERANCE:
        _fail("V5-2T float32 copy-back objective increased")
    copy_geometry = v52s.geometry_evidence_v1(
        frozen_weight=frozen_weight,
        candidate_weight=copied_weight64,
    )
    if copy_geometry["gate"] != "PASS":
        _fail("V5-2T float32 copy-back escaped the geometry bound")
    if final_counts != group_counts or copy_counts != group_counts:
        _fail("V5-2T objective group counts changed during solve")

    return {
        "trainable_parameter_count": int(weight.numel()),
        "initial_total_objective": initial_total_value,
        "initial_balanced_bce": initial_total_value,
        "initial_group_mean_bce": {
            name: float(initial_groups[name].item()) for name in v52s.GROUPS
        },
        "final_total_objective_float64": final_total_value,
        "final_balanced_bce_float64": final_balanced_value,
        "final_proximal_penalty_float64": final_penalty_value,
        "final_group_mean_bce_float64": {
            name: float(final_groups[name].detach().item()) for name in v52s.GROUPS
        },
        "float32_copy_back_total_objective": copy_total_value,
        "float32_copy_back_balanced_bce": float(copy_balanced.item()),
        "float32_copy_back_proximal_penalty": float(copy_penalty.item()),
        "float32_copy_back_bit_exact": True,
        "finite_non_increasing_objective": True,
        "group_counts": group_counts,
        "group_coefficients": {name: v52s.GROUP_WEIGHT for name in v52s.GROUPS},
        "proximal_contract": proximal,
        "geometry_float64": geometry,
        "geometry_float32_copy_back": copy_geometry,
        "lbfgs_termination": termination,
        "optimizer_state_keys": sorted(str(key) for key in state),
        "closure_evaluations": closure_evaluations,
        "head_bias": frozen_bias,
    }


def _candidate_path(directory: Path, digit: str) -> Path:
    return directory / f"digit{digit}_v5_2t_bounded_class_balanced_candidate.pt"


def _state_fingerprint_without_numpy_v1(model) -> str:
    """Hash exact tensor state without relying on optional NumPy."""
    torch, _nn = v52b._import_torch()
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(cpu.shape)).encode("ascii"))
        digest.update(b"\0")
        raw = bytes(cpu.view(torch.uint8).reshape(-1).tolist())
        digest.update(raw)
        digest.update(b"\0")
    return digest.hexdigest()


def _save_candidate(
    *,
    model,
    path: Path,
    digit: str,
    source_sha: str,
    manifest_sha: str,
    fit: Mapping[str, object],
    invariants: Mapping[str, object],
) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    state_fingerprint = _state_fingerprint_without_numpy_v1(model)
    payload = {
        "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "metadata": {
            "schema": SCHEMA,
            "role": f"digit-{digit}-v5-2t-bounded-class-balanced-candidate",
            "source_checkpoint_sha256": source_sha,
            "slot_manifest_sha256": manifest_sha,
            "trainable_surface": "head.weight-only-64-parameters",
            "head_bias_frozen": True,
            "backbone_frozen": True,
            "threshold": v52b.FROZEN_THRESHOLDS[digit],
            "threshold_tuned": False,
            "state_fingerprint": state_fingerprint,
            "prerequisite": v52s.prerequisite_evidence_contract(),
            "objective": v52s.objective_contract(),
            "solver": implementation_contract()["solver"],
            "fit_geometry": fit["geometry_float32_copy_back"],
            "invariants": dict(invariants),
            "historical_retention_executed": False,
            "first30_opened": False,
            "v5_validation_opened": False,
            "final_holdout_locked": True,
        },
    }
    torch.save(payload, path)
    return {
        "candidate_path": str(path),
        "candidate_sha256": v52b._sha_file(path),
        "state_fingerprint": state_fingerprint,
    }


def _load_candidate(
    path: Path,
    *,
    digit: str,
    source_sha: str,
    manifest_sha: str,
):
    torch, _nn = v52b._import_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise MeterV5_2TError(f"cannot reload V5-2T {digit}-AI candidate") from exc
    if not isinstance(payload, Mapping):
        _fail(f"{digit}-AI candidate payload must be a mapping")
    metadata = payload.get("metadata")
    state = payload.get("model_state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
        _fail(f"{digit}-AI candidate metadata/state missing")
    expected = {
        "schema": SCHEMA,
        "role": f"digit-{digit}-v5-2t-bounded-class-balanced-candidate",
        "source_checkpoint_sha256": source_sha,
        "slot_manifest_sha256": manifest_sha,
        "trainable_surface": "head.weight-only-64-parameters",
        "head_bias_frozen": True,
        "backbone_frozen": True,
        "threshold": v52b.FROZEN_THRESHOLDS[digit],
        "threshold_tuned": False,
        "prerequisite": v52s.prerequisite_evidence_contract(),
        "objective": v52s.objective_contract(),
        "solver": implementation_contract()["solver"],
        "historical_retention_executed": False,
        "first30_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            _fail(f"{digit}-AI candidate metadata changed: {key}")
    model = v52b._build_digit_model().cpu()
    model.load_state_dict(dict(state), strict=True)
    if metadata.get("state_fingerprint") != _state_fingerprint_without_numpy_v1(model):
        _fail(f"{digit}-AI candidate state fingerprint mismatch")
    geometry = metadata.get("fit_geometry")
    if not isinstance(geometry, Mapping) or geometry.get("gate") != "PASS":
        _fail(f"{digit}-AI candidate geometry evidence missing/HOLD")
    model.eval()
    return model


def train_bounded_class_balanced_head_repair_v1(
    data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    v5_2r_report: str | Path,
    v5_2r_execution_envelope: str | Path,
    confirmation: str,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run the one preregistered solve and stop before historical retention."""
    if confirmation != APPROVAL_TOKEN:
        _fail("exact V5-2T approval token missing")
    root = Path(data_root)
    ann = root / v51.ANNOTATIONS_DIR
    training_path = ann / TRAINING_REPORT_NAME
    candidate_dir = ann / CANDIDATE_DIR_NAME
    temporary_dir = ann / TEMP_CANDIDATE_DIR_NAME
    if training_path.exists() or candidate_dir.exists() or temporary_dir.exists():
        _fail("refusing to overwrite/rerun V5-2T evidence")

    evidence = v52s.verify_exact_v5_2r_evidence(
        report_path=v5_2r_report,
        envelope_path=v5_2r_execution_envelope,
    )
    models = v52n._frozen_models(
        digit2_frozen=Path(digit2_frozen),
        digit3_frozen=Path(digit3_frozen),
    )
    manifest_path, _rows, v5_features, v5_targets, _metrics = v52n._v5_surface(
        root, models
    )
    historical_features, historical_targets = v52n._historical_surface(
        m4a_root=Path(m4a_root),
        d10_root=Path(d10_root),
        models=models,
        progress=progress,
    )

    torch, _nn = v52b._import_torch()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    per_specialist: dict[str, dict[str, object]] = {}
    frozen_states: dict[str, Mapping[str, object]] = {}
    for digit in ("2", "3"):
        model = models[digit]
        frozen_state = v52p._frozen_state_snapshot(model)
        frozen_states[digit] = frozen_state
        fit = _fit_bounded_head_v1(
            model,
            v5_features=v5_features[digit],
            v5_targets=v5_targets[digit],
            historical_features=historical_features[digit],
            historical_targets=historical_targets[digit],
            enforce_preregistered_counts=True,
            expected_historical_positive_count=EXPECTED_HISTORICAL_POSITIVE_COUNT[digit],
        )
        try:
            invariants = v52p._verify_only_head_weight_changed(model, frozen_state)
        except v52p.MeterV5_2PError as exc:
            raise MeterV5_2TError(f"{digit}-AI frozen-state integrity failed") from exc
        if invariants["only_head_weight_changed"] is not True:
            _fail(f"{digit}-AI changed an illegal state tensor")
        per_specialist[digit] = {
            "fit": fit,
            "state_invariants": invariants,
            "threshold": v52b.FROZEN_THRESHOLDS[digit],
            "threshold_unchanged": True,
            "v5_train_metrics_at_frozen_threshold": v52p._feature_metrics(
                model,
                v5_features[digit],
                v5_targets[digit],
                threshold=v52b.FROZEN_THRESHOLDS[digit],
            ),
            "historical_train_metrics_at_frozen_threshold": v52p._feature_metrics(
                model,
                historical_features[digit],
                historical_targets[digit],
                threshold=v52b.FROZEN_THRESHOLDS[digit],
            ),
        }

    temporary_dir.mkdir(parents=True, exist_ok=False)
    for digit in ("2", "3"):
        source_sha = v52b.DIGIT2_SHA256 if digit == "2" else v52b.DIGIT3_SHA256
        saved = _save_candidate(
            model=models[digit],
            path=_candidate_path(temporary_dir, digit),
            digit=digit,
            source_sha=source_sha,
            manifest_sha=v52b._sha_file(manifest_path),
            fit=per_specialist[digit]["fit"],
            invariants=per_specialist[digit]["state_invariants"],
        )
        per_specialist[digit]["candidate"] = saved
    temporary_dir.replace(candidate_dir)
    for digit in ("2", "3"):
        final_path = _candidate_path(candidate_dir, digit)
        per_specialist[digit]["candidate"]["candidate_path"] = str(final_path)
        per_specialist[digit]["candidate"]["candidate_sha256"] = v52b._sha_file(final_path)
        source_sha = v52b.DIGIT2_SHA256 if digit == "2" else v52b.DIGIT3_SHA256
        reloaded = _load_candidate(
            final_path,
            digit=digit,
            source_sha=source_sha,
            manifest_sha=v52b._sha_file(manifest_path),
        )
        try:
            reload_invariants = v52p._verify_only_head_weight_changed(
                reloaded, frozen_states[digit]
            )
        except v52p.MeterV5_2PError as exc:
            raise MeterV5_2TError(
                f"{digit}-AI reloaded candidate state integrity failed"
            ) from exc
        if reload_invariants != per_specialist[digit]["state_invariants"]:
            _fail(f"{digit}-AI reloaded candidate invariants changed")
        per_specialist[digit]["candidate"]["reload_verified"] = True

    report: dict[str, object] = {
        "schema": SCHEMA,
        "approval_token_verified": True,
        "exact_v5_2r_evidence": evidence,
        "slot_manifest_sha256": v52b._sha_file(manifest_path),
        "source_checkpoint_sha256": {
            "2": v52b.DIGIT2_SHA256,
            "3": v52b.DIGIT3_SHA256,
        },
        "v5_train_slot_count": EXPECTED_V5_COUNT,
        "historical_train_record_count": EXPECTED_HISTORICAL_COUNT,
        "feature_dim": EXPECTED_FEATURE_DIM,
        "implementation_contract": implementation_contract(),
        "per_specialist": per_specialist,
        "numerical_integrity_gate": {"gate": "PASS", "reasons": []},
        "historical_preservation_claimed": False,
        **safety_boundary(),
    }
    v51._atomic_write_json(training_path, report)
    return report


def historical_retention_executed_by_this_module() -> bool:
    return False


def validation_opened_by_this_module() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False
