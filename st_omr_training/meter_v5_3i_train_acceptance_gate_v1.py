"""Meter V5-3I read-only TRAIN acceptance gate.

V5-3I consumes only the completed V5-3G/V5-3H execution receipt, the two
separate rescue artifacts, V5 adaptation TRAIN, and historical TRAIN. It
performs no fitting and opens no protected validation surface.

Acceptance is intentionally stronger on V5 TRAIN than on historical TRAIN:
- each 2-AI/3-AI combined specialist must reach exact V5 TRAIN F1 == 1.0;
- no example that the frozen specialist already classified correctly may
  become wrong on either V5 TRAIN or historical TRAIN;
- original frozen specialist state must remain bit-identical;
- execution evidence must prove that only the separate rescue namespace was
  trainable.

Historical validation, immutable First-30, V5 reserve/validation and
FINAL_HOLDOUT remain closed regardless of PASS/HOLD.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Final, Mapping

from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2n_frozen_feature_transfer_audit_v1 as v52n
from . import meter_v5_3e_rescue_training_preregistration_v1 as v53e
from . import meter_v5_3f_rescue_training_execution_harness_v1 as v53f
from . import meter_v5_3g_authoritative_rescue_training_v1 as v53g


SCHEMA: Final[str] = "st-omr-meter-v5-3i-train-acceptance-gate-v1"
REPORT_NAME: Final[str] = "v5_3i_train_acceptance_gate_v1.json"

V53G_HEAD_SHA: Final[str] = "b36a9d2f5daade2c3568cac8cbc736ca75ca435f"
V53H_WRAPPER_HEAD_SHA: Final[str] = "aa426442efdef97e3323906096087dabffa1171b"
V53H_ENVELOPE_SCHEMA: Final[str] = (
    "st-omr-meter-v5-3h-authoritative-rescue-execution-envelope-v1"
)
V53H_ENVELOPE_NAME: Final[str] = (
    "v5_3g_execution_envelope_b36a9d2f5daade2c3568cac8cbc736ca75ca435f.json"
)

EXPECTED_V53G_REPORT_SHA256: Final[str] = (
    "682c2d405287051fef18b803e2597777cb7fc55c6ba0814ea3b2d4df0fa35b9d"
)
EXPECTED_V53H_ENVELOPE_SHA256: Final[str] = (
    "f41b0fddb9d139018e0ddd16c9765d9415031e6308efd67e16aef3a05d205bf7"
)
EXPECTED_RESCUE_ARTIFACT_SHA256: Final[dict[str, str]] = {
    "2": "a27cef8d4ff89565cfe4a15e0e429a21e60daa2656324ed0380fde8674a022e6",
    "3": "b8a4f379c33d3aa0df77b54821996a799251abb0e7cbd8de9764b09c5efd3d65",
}

ProgressCallback = Callable[[int, int, str], None]


class MeterV5_3IError(RuntimeError):
    """Raised when V5-3I receipt/integrity boundaries fail closed."""


def _fail(message: str) -> None:
    raise MeterV5_3IError(message)


def safety_boundary() -> dict[str, object]:
    return {
        "training": False,
        "autograd_grad_used": False,
        "backward": False,
        "optimizer_steps": 0,
        "v5_adaptation_train_read": True,
        "historical_train_read": True,
        "rescue_artifact_read": True,
        "frozen_checkpoint_read": True,
        "checkpoint_write": False,
        "rescue_artifact_write": False,
        "frozen_model_mutation_allowed": False,
        "digit4_loaded": False,
        "digit4_frozen": True,
        "threshold_tuning": False,
        "hyperparameter_sweep": False,
        "automatic_second_configuration": False,
        "historical_validation_opened": False,
        "first30_opened": False,
        "v5_reserve_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "resolver_wiring": False,
        "runtime_authority_changed": False,
        "production_promotion": False,
        "retraining_authorized_on_hold": False,
        "gate_report_write_only": True,
    }


def acceptance_contract() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "prerequisite_v5_3g_head": V53G_HEAD_SHA,
        "prerequisite_v5_3h_wrapper_head": V53H_WRAPPER_HEAD_SHA,
        "bound_v5_3g_report_sha256": EXPECTED_V53G_REPORT_SHA256,
        "bound_v5_3h_envelope_sha256": EXPECTED_V53H_ENVELOPE_SHA256,
        "bound_rescue_artifact_sha256": dict(EXPECTED_RESCUE_ARTIFACT_SHA256),
        "rescue_threshold": v53e.RESCUE_THRESHOLD,
        "frozen_thresholds": {
            "2": v52b.FROZEN_THRESHOLDS["2"],
            "3": v52b.FROZEN_THRESHOLDS["3"],
        },
        "v5_train_required_f1": {"2": 1.0, "3": 1.0},
        "v5_train_required_false_positive_count": {"2": 0, "3": 0},
        "v5_train_required_false_negative_count": {"2": 0, "3": 0},
        "frozen_correct_regression_count_max": {
            "v5_train": {"2": 0, "3": 0},
            "historical_train": {"2": 0, "3": 0},
        },
        "frozen_state_bit_identical_required": True,
        "only_rescue_parameters_changed_required": True,
        "hold_does_not_authorize_retraining": True,
        **safety_boundary(),
    }


def _sha_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        _fail(f"expected regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_bound(path: Path, *, expected_sha256: str, label: str) -> dict[str, object]:
    actual = _sha_file(path)
    if actual != expected_sha256:
        _fail(f"{label} SHA changed: {actual}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeterV5_3IError(f"invalid {label} JSON") from exc
    if not isinstance(payload, dict):
        _fail(f"{label} JSON must be an object")
    return payload


def _require_gate(value: object, *, label: str) -> None:
    if not isinstance(value, Mapping) or value.get("gate") != "PASS":
        _fail(f"{label} is not PASS")
    if value.get("reasons") != []:
        _fail(f"{label} reasons changed")


def _validate_execution_receipt(
    *,
    report: Mapping[str, object],
    envelope: Mapping[str, object],
) -> None:
    expected_report_scalars = {
        "schema": v53g.SCHEMA,
        "single_authoritative_execution_completed": True,
        "candidate_configuration_count": 1,
        "train_performance_gate_executed": False,
        "historical_validation_retention_executed": False,
        "first30_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "runtime_authority_changed": False,
        "production_promotion": False,
        "v5_3f_head_sha": v53g.V53F_HEAD_SHA,
    }
    for key, expected in expected_report_scalars.items():
        if report.get(key) != expected:
            _fail(f"V5-3G report field changed: {key}")
    _require_gate(report.get("numerical_integrity_gate"), label="numerical integrity")
    _require_gate(report.get("frozen_state_isolation_gate"), label="frozen-state isolation")

    if report.get("source_checkpoint_sha256") != {
        "2": v52b.DIGIT2_SHA256,
        "3": v52b.DIGIT3_SHA256,
    }:
        _fail("V5-3G source checkpoint binding changed")

    expected_envelope_scalars = {
        "schema": V53H_ENVELOPE_SCHEMA,
        "repository": "khfy7wpr5p-maker/st-omr-training",
        "expected_head": V53G_HEAD_SHA,
        "actual_head": V53G_HEAD_SHA,
        "ci_run_id": 32769348282,
        "single_authoritative_execution_completed": True,
        "candidate_configuration_count": 1,
        "numerical_integrity_gate": "PASS",
        "frozen_state_isolation_gate": "PASS",
        "train_performance_gate_executed": False,
        "historical_validation_opened": False,
        "first30_opened": False,
        "v5_reserve_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "threshold_tuning": False,
        "hyperparameter_sweep": False,
        "automatic_second_configuration": False,
        "runtime_authority_changed": False,
        "production_promotion": False,
        "isolated_runtime": True,
        "python_no_user_site": True,
        "venv_bootstrap": "stdlib-venv-without-pip+host-pip--python",
        "report_sha256": EXPECTED_V53G_REPORT_SHA256,
    }
    for key, expected in expected_envelope_scalars.items():
        if envelope.get(key) != expected:
            _fail(f"V5-3H envelope field changed: {key}")
    if envelope.get("artifact_sha256") != EXPECTED_RESCUE_ARTIFACT_SHA256:
        _fail("V5-3H rescue artifact binding changed")

    per_specialist = report.get("per_specialist")
    if not isinstance(per_specialist, Mapping) or set(per_specialist) != {"2", "3"}:
        _fail("V5-3G per-specialist receipt changed")
    envelope_groups = envelope.get("group_fingerprints")
    if not isinstance(envelope_groups, Mapping):
        _fail("V5-3H group fingerprints missing")
    for digit in ("2", "3"):
        item = per_specialist.get(digit)
        if not isinstance(item, Mapping):
            _fail(f"{digit}-AI V5-3G receipt missing")
        if item.get("frozen_state_bit_identical") is not True:
            _fail(f"{digit}-AI frozen state was not bit-identical")
        if item.get("frozen_state_before") != item.get("frozen_state_after"):
            _fail(f"{digit}-AI frozen before/after receipt differs")
        execution = item.get("execution")
        materialization = item.get("materialization")
        artifact = item.get("artifact")
        if not all(isinstance(value, Mapping) for value in (execution, materialization, artifact)):
            _fail(f"{digit}-AI receipt shape changed")
        if execution.get("optimizer_steps") != v53e.FIXED_OPTIMIZER_STEPS:
            _fail(f"{digit}-AI optimizer-step receipt changed")
        if execution.get("authoritative_dataset_execution") is not True:
            _fail(f"{digit}-AI authoritative execution receipt missing")
        if execution.get("checkpoint_write") is not False:
            _fail(f"{digit}-AI execution wrote a checkpoint")
        if execution.get("protected_evaluation_opened") is not False:
            _fail(f"{digit}-AI execution opened protected evaluation")
        if materialization.get("group_counts") != v53e.EXPECTED_TRAIN_GROUP_COUNTS[digit]:
            _fail(f"{digit}-AI group-count receipt changed")
        if envelope_groups.get(digit) != materialization.get("group_fingerprints"):
            _fail(f"{digit}-AI report/envelope group fingerprints differ")
        if artifact.get("artifact_sha256") != EXPECTED_RESCUE_ARTIFACT_SHA256[digit]:
            _fail(f"{digit}-AI artifact SHA receipt changed")
        if artifact.get("reload_verified") is not True:
            _fail(f"{digit}-AI artifact was not reload-verified")


def _load_rescue_artifact(
    path: Path,
    *,
    digit: str,
    report_item: Mapping[str, object],
):
    if digit not in ("2", "3"):
        _fail("V5-3I supports only 2-AI and 3-AI")
    if _sha_file(path) != EXPECTED_RESCUE_ARTIFACT_SHA256[digit]:
        _fail(f"{digit}-AI rescue artifact bytes changed")

    torch, _nn = v52b._import_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise MeterV5_3IError(f"cannot read {digit}-AI rescue artifact") from exc
    if not isinstance(payload, Mapping):
        _fail(f"{digit}-AI rescue artifact payload changed")
    metadata = payload.get("metadata")
    state = payload.get("rescue_state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
        _fail(f"{digit}-AI rescue artifact metadata/state missing")

    materialization = report_item.get("materialization")
    execution = report_item.get("execution")
    artifact_receipt = report_item.get("artifact")
    if not all(isinstance(value, Mapping) for value in (materialization, execution, artifact_receipt)):
        _fail(f"{digit}-AI report receipt shape changed")

    expected_metadata = {
        "schema": v53g.RESCUE_ARTIFACT_SCHEMA,
        "role": f"digit-{digit}-v5-3g-rescue",
        "v5_3f_head_sha": v53g.V53F_HEAD_SHA,
        "recipe_id": v53e.RECIPE_ID,
        "source_checkpoint_sha256": (
            v52b.DIGIT2_SHA256 if digit == "2" else v52b.DIGIT3_SHA256
        ),
        "slot_manifest_sha256": report_item.get("_slot_manifest_sha256"),
        "rescue_threshold": v53e.RESCUE_THRESHOLD,
        "frozen_threshold": v52b.FROZEN_THRESHOLDS[digit],
        "parameter_count": v53e.PARAMETERS_PER_RESCUE,
        "trainable_surface": "new-rescue-parameters-only",
        "digit4_frozen": True,
        "group_counts": materialization.get("group_counts"),
        "group_fingerprints": materialization.get("group_fingerprints"),
        "optimizer_steps": v53e.FIXED_OPTIMIZER_STEPS,
        "initial_state_fingerprint": execution.get("initial_state_fingerprint"),
        "final_state_fingerprint": execution.get("final_state_fingerprint"),
        "state_fingerprint": artifact_receipt.get("state_fingerprint"),
        "historical_validation_opened": False,
        "first30_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "runtime_authority_changed": False,
        "production_promotion": False,
    }
    if dict(metadata) != expected_metadata:
        _fail(f"{digit}-AI rescue artifact metadata changed")

    model = v53f._build_rescue_model_v1()
    model.load_state_dict(dict(state), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        if parameter.grad is not None:
            _fail(f"{digit}-AI rescue artifact has unexpected gradient")
    fingerprint = v53f._state_fingerprint(model)
    if fingerprint != artifact_receipt.get("state_fingerprint"):
        _fail(f"{digit}-AI rescue state fingerprint changed")
    return model


def _probabilities_from_rescue(model, features, *, digit: str):
    torch, _nn = v52b._import_torch()
    x = features.detach().cpu().to(dtype=torch.float32)
    if x.ndim != 2 or x.shape[1] != v53e.FEATURE_DIM:
        _fail(f"{digit}-AI rescue feature shape changed")
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x)).cpu()
    if probabilities.ndim != 1 or probabilities.numel() != x.shape[0]:
        _fail(f"{digit}-AI rescue probability shape changed")
    if not bool(torch.isfinite(probabilities).all().item()):
        _fail(f"{digit}-AI rescue probabilities became non-finite")
    return probabilities


def _binary_metrics_from_predictions(predictions, targets) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    p = predictions.detach().cpu().to(dtype=torch.bool).reshape(-1)
    y = targets.detach().cpu().to(dtype=torch.float32).reshape(-1)
    if p.numel() != y.numel() or p.numel() <= 0:
        _fail("binary metric cardinality changed")
    if not bool(((y == 0.0) | (y == 1.0)).all().item()):
        _fail("binary metric targets are not binary")
    truth = y == 1.0
    tp = int((p & truth).sum().item())
    fp = int((p & ~truth).sum().item())
    fn = int((~p & truth).sum().item())
    tn = int((~p & ~truth).sum().item())
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "count": int(p.numel()),
        "positive_count": int(truth.sum().item()),
        "negative_count": int((~truth).sum().item()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / int(p.numel()),
    }


def _combined_prediction_evidence(
    *,
    digit: str,
    frozen_model,
    rescue_model,
    features,
    targets,
) -> dict[str, object]:
    torch, _nn = v52b._import_torch()
    frozen_probability = v53g._frozen_probabilities_from_features(
        frozen_model, features, digit=digit
    )
    rescue_probability = _probabilities_from_rescue(rescue_model, features, digit=digit)
    frozen_threshold = v52b.FROZEN_THRESHOLDS[digit]
    rescue_threshold = v53e.RESCUE_THRESHOLD

    frozen_prediction = frozen_probability >= frozen_threshold
    rescue_eligible = frozen_probability < frozen_threshold
    rescue_prediction = rescue_probability >= rescue_threshold
    combined_prediction = torch.where(
        rescue_eligible,
        rescue_prediction,
        frozen_prediction,
    )

    truth = targets.detach().cpu().to(dtype=torch.float32).reshape(-1) == 1.0
    if truth.numel() != combined_prediction.numel():
        _fail(f"{digit}-AI target cardinality changed")
    frozen_correct = frozen_prediction == truth
    combined_correct = combined_prediction == truth
    regressions = frozen_correct & ~combined_correct
    corrections = ~frozen_correct & combined_correct

    return {
        "frozen_threshold": frozen_threshold,
        "rescue_threshold": rescue_threshold,
        "rescue_eligible_count": int(rescue_eligible.sum().item()),
        "frozen_correct_count": int(frozen_correct.sum().item()),
        "frozen_incorrect_count": int((~frozen_correct).sum().item()),
        "frozen_correct_regression_count": int(regressions.sum().item()),
        "frozen_incorrect_correction_count": int(corrections.sum().item()),
        "combined_metrics": _binary_metrics_from_predictions(combined_prediction, targets),
    }


def run_train_acceptance_gate_v1(
    v5_data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    v53g_report: str | Path,
    v53h_envelope: str | Path,
    rescue_artifact_dir: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Evaluate the preregistered TRAIN acceptance gate without fitting."""
    root = Path(v5_data_root)
    report = _read_json_bound(
        Path(v53g_report),
        expected_sha256=EXPECTED_V53G_REPORT_SHA256,
        label="V5-3G report",
    )
    envelope = _read_json_bound(
        Path(v53h_envelope),
        expected_sha256=EXPECTED_V53H_ENVELOPE_SHA256,
        label="V5-3H envelope",
    )
    artifact_dir = Path(rescue_artifact_dir)
    _validate_execution_receipt(report=report, envelope=envelope)

    frozen_models = v52n._frozen_models(
        digit2_frozen=Path(digit2_frozen),
        digit3_frozen=Path(digit3_frozen),
    )
    frozen_state_before = {
        digit: v53f._state_fingerprint(frozen_models[digit]) for digit in ("2", "3")
    }

    manifest_path, _rows, v5_features, v5_targets, frozen_v5_metrics = v52n._v5_surface(
        root, frozen_models
    )
    actual_slot_manifest_sha = v52b._sha_file(manifest_path)
    if actual_slot_manifest_sha != report.get("slot_manifest_sha256"):
        _fail("V5 slot manifest no longer matches V5-3G receipt")

    historical_features, historical_targets = v52n._historical_surface(
        m4a_root=Path(m4a_root),
        d10_root=Path(d10_root),
        models=frozen_models,
        progress=progress,
    )

    per_specialist_report = report.get("per_specialist")
    if not isinstance(per_specialist_report, Mapping):
        _fail("V5-3G per-specialist receipt missing")

    rescue_models: dict[str, object] = {}
    rescue_state_before: dict[str, str] = {}
    for digit in ("2", "3"):
        item = per_specialist_report.get(digit)
        if not isinstance(item, Mapping):
            _fail(f"{digit}-AI V5-3G receipt missing")
        item_with_slot = dict(item)
        item_with_slot["_slot_manifest_sha256"] = actual_slot_manifest_sha
        rescue_models[digit] = _load_rescue_artifact(
            artifact_dir / f"digit_{digit}_rescue.pt",
            digit=digit,
            report_item=item_with_slot,
        )
        rescue_state_before[digit] = v53f._state_fingerprint(rescue_models[digit])

    per_specialist: dict[str, object] = {}
    acceptance_reasons: list[str] = []

    for digit in ("2", "3"):
        item = per_specialist_report.get(digit)
        if not isinstance(item, Mapping):
            _fail(f"{digit}-AI V5-3G receipt missing")
        groups, group_evidence = v53g._materialize_frozen_negative_groups_v1(
            digit=digit,
            model=frozen_models[digit],
            v5_features=v5_features[digit],
            v5_targets=v5_targets[digit],
            historical_features=historical_features[digit],
            historical_targets=historical_targets[digit],
            enforce_preregistered_counts=True,
        )
        del groups

        materialization = item.get("materialization")
        if not isinstance(materialization, Mapping):
            _fail(f"{digit}-AI materialization receipt missing")
        if group_evidence.get("group_counts") != materialization.get("group_counts"):
            _fail(f"{digit}-AI rematerialized group counts changed")
        if group_evidence.get("group_fingerprints") != materialization.get("group_fingerprints"):
            _fail(f"{digit}-AI rematerialized group fingerprints changed")
        envelope_groups = envelope.get("group_fingerprints")
        if not isinstance(envelope_groups, Mapping):
            _fail("V5-3H group fingerprints missing")
        if group_evidence.get("group_fingerprints") != envelope_groups.get(digit):
            _fail(f"{digit}-AI rematerialization does not match envelope")

        v5_evidence = _combined_prediction_evidence(
            digit=digit,
            frozen_model=frozen_models[digit],
            rescue_model=rescue_models[digit],
            features=v5_features[digit],
            targets=v5_targets[digit],
        )
        historical_evidence = _combined_prediction_evidence(
            digit=digit,
            frozen_model=frozen_models[digit],
            rescue_model=rescue_models[digit],
            features=historical_features[digit],
            targets=historical_targets[digit],
        )

        v5_metrics = v5_evidence.get("combined_metrics")
        if not isinstance(v5_metrics, Mapping):
            _fail(f"{digit}-AI V5 metric evidence missing")
        if v5_metrics.get("f1") != 1.0:
            acceptance_reasons.append(f"{digit}-AI V5 TRAIN F1 != 1.0")
        if v5_metrics.get("fp") != 0:
            acceptance_reasons.append(f"{digit}-AI V5 TRAIN false positives != 0")
        if v5_metrics.get("fn") != 0:
            acceptance_reasons.append(f"{digit}-AI V5 TRAIN false negatives != 0")
        if v5_evidence["frozen_correct_regression_count"] != 0:
            acceptance_reasons.append(
                f"{digit}-AI V5 TRAIN frozen-correct regression count != 0"
            )
        if historical_evidence["frozen_correct_regression_count"] != 0:
            acceptance_reasons.append(
                f"{digit}-AI historical TRAIN frozen-correct regression count != 0"
            )

        per_specialist[digit] = {
            "frozen_v5_metrics_before_rescue": frozen_v5_metrics[digit],
            "v5_train": v5_evidence,
            "historical_train": historical_evidence,
            "group_identity_reverified": True,
            "rescue_artifact_sha256": EXPECTED_RESCUE_ARTIFACT_SHA256[digit],
        }

    frozen_state_after = {
        digit: v53f._state_fingerprint(frozen_models[digit]) for digit in ("2", "3")
    }
    rescue_state_after = {
        digit: v53f._state_fingerprint(rescue_models[digit]) for digit in ("2", "3")
    }
    frozen_state_bit_identical = frozen_state_before == frozen_state_after
    rescue_state_bit_identical_during_gate = rescue_state_before == rescue_state_after
    if not frozen_state_bit_identical:
        _fail("V5-3I mutated a frozen specialist")
    if not rescue_state_bit_identical_during_gate:
        _fail("V5-3I mutated a rescue artifact in memory")

    only_rescue_parameters_changed = all(
        isinstance(per_specialist_report.get(digit), Mapping)
        and per_specialist_report[digit].get("frozen_state_bit_identical") is True
        and isinstance(per_specialist_report[digit].get("artifact"), Mapping)
        for digit in ("2", "3")
    )
    for digit in ("2", "3"):
        item = per_specialist_report.get(digit)
        if not isinstance(item, Mapping):
            _fail(f"{digit}-AI V5-3G receipt missing")
        artifact = item.get("artifact")
        if not isinstance(artifact, Mapping):
            _fail(f"{digit}-AI artifact receipt missing")
        if artifact.get("state_fingerprint") != rescue_state_before[digit]:
            only_rescue_parameters_changed = False
    implementation_contract = report.get("implementation_contract")
    if not isinstance(implementation_contract, Mapping):
        _fail("V5-3G implementation contract missing")
    if implementation_contract.get("trainable_surface") != "new-rescue-parameters-only":
        only_rescue_parameters_changed = False
    if implementation_contract.get("frozen_model_mutation_allowed") is not False:
        only_rescue_parameters_changed = False
    if not only_rescue_parameters_changed:
        acceptance_reasons.append("only-rescue-parameters-changed evidence failed")

    decision = "PASS" if not acceptance_reasons else "HOLD"
    return {
        "schema": SCHEMA,
        "decision": decision,
        "acceptance_reasons": acceptance_reasons,
        "acceptance_contract": acceptance_contract(),
        "bound_evidence": {
            "v5_3g_head_sha": V53G_HEAD_SHA,
            "v5_3h_wrapper_head_sha": V53H_WRAPPER_HEAD_SHA,
            "v5_3g_report_sha256": EXPECTED_V53G_REPORT_SHA256,
            "v5_3h_envelope_sha256": EXPECTED_V53H_ENVELOPE_SHA256,
            "rescue_artifact_sha256": dict(EXPECTED_RESCUE_ARTIFACT_SHA256),
            "slot_manifest_sha256": actual_slot_manifest_sha,
        },
        "per_specialist": per_specialist,
        "frozen_state_before": frozen_state_before,
        "frozen_state_after": frozen_state_after,
        "frozen_state_bit_identical": frozen_state_bit_identical,
        "rescue_state_bit_identical_during_gate": rescue_state_bit_identical_during_gate,
        "only_rescue_parameters_changed": only_rescue_parameters_changed,
        "historical_validation_retention_executed": False,
        "first30_opened": False,
        "v5_reserve_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "retraining_authorized": False,
        "next_gate_if_pass": "separately_staged_historical_validation_retention",
        **safety_boundary(),
    }


def future_gate_order() -> tuple[str, ...]:
    return (
        "v5_3i_train_acceptance",
        "separately_staged_historical_validation_retention",
        "immutable_v5_first30_diagnostic",
        "separately_authorized_v5_validation",
        "separately_authorized_final_holdout",
    )


def retraining_allowed_after_hold() -> bool:
    return False


def historical_validation_access_allowed() -> bool:
    return False


def first30_access_allowed() -> bool:
    return False


def v5_validation_access_allowed() -> bool:
    return False


def final_holdout_access_allowed() -> bool:
    return False
