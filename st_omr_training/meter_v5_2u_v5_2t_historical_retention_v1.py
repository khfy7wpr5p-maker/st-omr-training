"""Read-only Historical Retention V3 gate for the exact V5-2T candidates.

This stage binds the completed exact-SHA V5-2T training report, execution
envelope, and candidate checkpoint hashes.  It evaluates the candidates and
the exact frozen 2/3/4 specialists on the existing historical validation pixel
surface at unchanged thresholds.  It never trains, tunes, or opens First-30,
V5 validation, or FINAL_HOLDOUT.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Final, Mapping

from . import meter_v5_1_bbox_pilot as v51
from . import meter_v5_2b_specialist_adaptation as v52b
from . import meter_v5_2c_historical_retention_v1 as ret_legacy
from . import meter_v5_2c_historical_retention_v2 as ret_v2
from . import meter_v5_2m_retention_contract_v3 as ret_v3
from . import meter_v5_2t_bounded_class_balanced_head_repair_v1 as v52t


SCHEMA: Final[str] = "st-omr-meter-v5-2u-v5-2t-historical-retention-v1"
REPORT_NAME: Final[str] = "v5_2u_v5_2t_historical_retention_v3.json"
V52T_IMPLEMENTATION_HEAD: Final[str] = (
    "8d98c1f6ad66ee896d28c02fb7ff1afafab23be9"
)
V52T_TRAINING_REPORT_SHA256: Final[str] = (
    "18851e86d9e2aa7d0d55ccbafdb2983c96c9276913419b928512ea76e3a2bc57"
)
V52T_EXECUTION_ENVELOPE_SHA256: Final[str] = (
    "1a043631118612a85e6d2a78baaa7f26aeb0742254684b96d8b3a4d25c031382"
)
V52T_CANDIDATE_SHA256: Final[dict[str, str]] = {
    "2": "13fb7dd0af1faa8a762433df2b27d6c82553fca398b84b060cb8d573d2d228de",
    "3": "8ec37448af27f57bdd7840eeeee8a61cee5d21e6c21156ccfb4c1510d3542514",
}
ProgressCallback = Callable[[int, int, str], None]


class MeterV5_2UError(RuntimeError):
    """Raised when exact evidence or retention execution departs from contract."""


def _fail(message: str) -> None:
    raise MeterV5_2UError(message)


def safety_boundary() -> dict[str, object]:
    return {
        "training": False,
        "autograd_grad_used": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_write": False,
        "candidate_checkpoint_mutation": False,
        "runtime_threshold_tuning": False,
        "alternative_threshold_evaluated": False,
        "historical_validation_opened": True,
        "historical_retention_executed": True,
        "first30_opened": False,
        "v5_validation_opened": False,
        "final_holdout_locked": True,
        "digit4_frozen": True,
        "new_bbox": False,
        "new_crop_geometry": False,
        "new_spatial_heuristic": False,
        "production_promotion": False,
    }


def _validate_training_payload_v1(payload: Mapping[str, object]) -> None:
    if payload.get("schema") != v52t.SCHEMA:
        _fail("V5-2T training report schema mismatch")
    if payload.get("numerical_integrity_gate") != {"gate": "PASS", "reasons": []}:
        _fail("V5-2T numerical integrity is not exact PASS")
    if payload.get("historical_preservation_claimed") is not False:
        _fail("V5-2T report unexpectedly claims historical preservation")
    if payload.get("historical_validation_opened") is not False:
        _fail("V5-2T training unexpectedly opened historical validation")
    if payload.get("first30_opened") is not False:
        _fail("V5-2T training unexpectedly opened First-30")
    if payload.get("v5_validation_opened") is not False:
        _fail("V5-2T training unexpectedly opened V5 validation")
    if payload.get("final_holdout_locked") is not True:
        _fail("V5-2T FINAL_HOLDOUT lock changed")
    if payload.get("digit4_frozen") is not True:
        _fail("V5-2T 4-AI freeze changed")
    per = payload.get("per_specialist")
    if not isinstance(per, Mapping):
        _fail("V5-2T per-specialist evidence missing")
    for digit in ("2", "3"):
        item = per.get(digit)
        if not isinstance(item, Mapping):
            _fail(f"V5-2T {digit}-AI evidence missing")
        candidate = item.get("candidate")
        invariants = item.get("state_invariants")
        fit = item.get("fit")
        if not isinstance(candidate, Mapping) or not isinstance(invariants, Mapping):
            _fail(f"V5-2T {digit}-AI candidate/invariants missing")
        if not isinstance(fit, Mapping):
            _fail(f"V5-2T {digit}-AI fit evidence missing")
        if candidate.get("candidate_sha256") != V52T_CANDIDATE_SHA256[digit]:
            _fail(f"V5-2T {digit}-AI candidate SHA binding changed")
        if candidate.get("reload_verified") is not True:
            _fail(f"V5-2T {digit}-AI candidate reload was not verified")
        if invariants.get("changed_state_keys") != ["head.weight"]:
            _fail(f"V5-2T {digit}-AI changed-state surface is invalid")
        for key in (
            "only_head_weight_changed",
            "backbone_bit_identical",
            "head_bias_bit_identical",
        ):
            if invariants.get(key) is not True:
                _fail(f"V5-2T {digit}-AI invariant failed: {key}")
        geometry = fit.get("geometry_float32_copy_back")
        termination = fit.get("lbfgs_termination")
        if not isinstance(geometry, Mapping) or geometry.get("gate") != "PASS":
            _fail(f"V5-2T {digit}-AI geometry is not PASS")
        if fit.get("finite_non_increasing_objective") is not True:
            _fail(f"V5-2T {digit}-AI numerical objective evidence failed")
        if not isinstance(termination, Mapping) or termination.get("final_gradient_finite") is not True:
            _fail(f"V5-2T {digit}-AI final gradient is not finite")


def _read_exact_execution_evidence_v1(
    *, training_report: Path, execution_envelope: Path
) -> tuple[dict[str, object], dict[str, object]]:
    if not training_report.is_file() or not execution_envelope.is_file():
        _fail("exact V5-2T report/envelope missing")
    report_bytes = training_report.read_bytes()
    envelope_bytes = execution_envelope.read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != V52T_TRAINING_REPORT_SHA256:
        _fail("V5-2T training report SHA256 mismatch")
    if hashlib.sha256(envelope_bytes).hexdigest() != V52T_EXECUTION_ENVELOPE_SHA256:
        _fail("V5-2T execution envelope SHA256 mismatch")
    report = v52b._read_json(training_report)
    envelope = v52b._read_json(execution_envelope)
    _validate_training_payload_v1(report)
    if envelope.get("expected_head") != V52T_IMPLEMENTATION_HEAD:
        _fail("V5-2T execution envelope HEAD mismatch")
    if envelope.get("training_report_sha256") != V52T_TRAINING_REPORT_SHA256:
        _fail("V5-2T envelope report binding mismatch")
    if envelope.get("candidate_checkpoint_sha256") != V52T_CANDIDATE_SHA256:
        _fail("V5-2T envelope candidate binding mismatch")
    if envelope.get("historical_preservation_claimed") is not False:
        _fail("V5-2T envelope unexpectedly claims preservation")
    return report, envelope


def run_historical_retention_v1(
    data_root: str | Path,
    *,
    m4a_root: str | Path,
    d10_root: str | Path,
    digit2_frozen: str | Path,
    digit3_frozen: str | Path,
    digit4_frozen: str | Path,
    digit2_candidate: str | Path,
    digit3_candidate: str | Path,
    training_report: str | Path,
    execution_envelope: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Execute only the preregistered historical retention comparison."""
    root = Path(data_root)
    output = root / v51.ANNOTATIONS_DIR / REPORT_NAME
    if output.exists():
        _fail("refusing to overwrite/rerun V5-2U retention evidence")
    training, _envelope = _read_exact_execution_evidence_v1(
        training_report=Path(training_report),
        execution_envelope=Path(execution_envelope),
    )
    frozen_paths = {
        "2": Path(digit2_frozen),
        "3": Path(digit3_frozen),
        "4": Path(digit4_frozen),
    }
    candidate_paths = {"2": Path(digit2_candidate), "3": Path(digit3_candidate)}
    expected_frozen = {
        "2": v52b.DIGIT2_SHA256,
        "3": v52b.DIGIT3_SHA256,
        "4": v52b.DIGIT4_SHA256,
    }
    for digit in ("2", "3", "4"):
        if v52b._sha_file(frozen_paths[digit]) != expected_frozen[digit]:
            _fail(f"frozen {digit}-AI checkpoint SHA changed")
    for digit in ("2", "3"):
        if v52b._sha_file(candidate_paths[digit]) != V52T_CANDIDATE_SHA256[digit]:
            _fail(f"V5-2T {digit}-AI candidate file SHA changed")

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
        probabilities = ret_legacy._probabilities(
            ret_legacy._frozen_model(frozen_paths[digit], digit=digit),
            images,
            progress=progress,
            phase=f"v5-2u-frozen-{digit}-AI-retention-self-check",
        )
        metrics = ret_legacy._binary_counts(
            probabilities,
            ret_legacy._truth_tensor(labels, digit),
            v52b.FROZEN_THRESHOLDS[digit],
        )
        expected = ret_v2.EXPECTED_FROZEN_COUNTS[digit]
        if any(metrics[key] != expected[key] for key in ("tp", "fp", "fn", "tn")):
            _fail(f"historical frozen oracle failed for {digit}-AI: {metrics}")
        frozen_metrics[digit] = metrics

    candidate_metrics: dict[str, dict[str, object]] = {}
    manifest_sha = str(training.get("slot_manifest_sha256"))
    for digit in ("2", "3"):
        source_sha = v52b.DIGIT2_SHA256 if digit == "2" else v52b.DIGIT3_SHA256
        model = v52t._load_candidate(
            candidate_paths[digit],
            digit=digit,
            source_sha=source_sha,
            manifest_sha=manifest_sha,
        )
        probabilities = ret_legacy._probabilities(
            model,
            images,
            progress=progress,
            phase=f"v5-2u-candidate-{digit}-AI-retention",
        )
        candidate_metrics[digit] = ret_legacy._binary_counts(
            probabilities,
            ret_legacy._truth_tensor(labels, digit),
            v52b.FROZEN_THRESHOLDS[digit],
        )

    gate = ret_v3.evaluate_retention_gate_v3(
        frozen_metrics=frozen_metrics,
        candidate_metrics=candidate_metrics,
    )
    report: dict[str, object] = {
        "schema": SCHEMA,
        "gate_kind": "historical-retention-v3-after-v5-2t",
        "retention_contract_schema": ret_v3.SCHEMA,
        "exact_v5_2t_binding": {
            "implementation_head": V52T_IMPLEMENTATION_HEAD,
            "training_report_sha256": V52T_TRAINING_REPORT_SHA256,
            "execution_envelope_sha256": V52T_EXECUTION_ENVELOPE_SHA256,
            "candidate_checkpoint_sha256": dict(V52T_CANDIDATE_SHA256),
        },
        "historical_pixel_path_reproduced": True,
        "validation_record_count": len(validation),
        "frozen_metrics": frozen_metrics,
        "candidate_metrics": candidate_metrics,
        "gate": gate["gate"],
        "reasons": gate["reasons"],
        "per_digit_retention": gate["per_digit"],
        "thresholds": dict(v52b.FROZEN_THRESHOLDS),
        "threshold_tuned": False,
        **safety_boundary(),
    }
    v51._atomic_write_json(output, report)
    return report


def first30_authorized(retention_report: Mapping[str, object]) -> bool:
    return retention_report.get("schema") == SCHEMA and retention_report.get("gate") == "PASS"


def validation_opened_by_this_module() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False
