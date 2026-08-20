"""Deterministic retention schedule and shadow runner for Meter V3.

V3 deliberately reuses the exact V2 data/model/loss/gate implementation and
changes one training mechanism only: the optimizer learning rate is reduced at
the fixed midpoint of the 20-epoch run.  The schedule is applied through the
public PyTorch global optimizer pre-hook API and is removed in ``finally`` so
it cannot leak outside this bounded invocation.

The underlying V2 metrics remain intact.  V3 writes a separate canonical
receipt that binds those metrics to the frozen schedule.  TEST, runtime,
Resolver, D11 replacement, and production promotion remain closed.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
from pathlib import Path
from typing import Final


METER_REAL_DOMAIN_RETENTION_V3: Final[str] = "meter-real-domain-retention-v3"
RETENTION_RECEIPT_SCHEMA_V3: Final[str] = "st-omr-meter-real-domain-retention-receipt-v3"
TOTAL_EPOCHS_V3: Final[int] = 20
EXPECTED_BATCHES_PER_EPOCH_V3: Final[int] = 30
TOTAL_OPTIMIZER_STEPS_V3: Final[int] = TOTAL_EPOCHS_V3 * EXPECTED_BATCHES_PER_EPOCH_V3
MIDPOINT_DECAY_EPOCH_V3: Final[int] = 11
EARLY_LEARNING_RATE_MICROS_V3: Final[int] = 1000
LATE_LEARNING_RATE_MICROS_V3: Final[int] = 250


class MeterRealDomainRetentionV3Error(RuntimeError):
    """Raised when the frozen V3 schedule or provenance boundary is violated."""


def _fail(message: str) -> None:
    raise MeterRealDomainRetentionV3Error(message)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise MeterRealDomainRetentionV3Error("V3 receipt is not canonical JSON serializable") from exc


def learning_rate_micros_for_epoch_v3(epoch: int) -> int:
    """Return the frozen V3 learning rate for a one-based epoch index."""
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        _fail("epoch must be an integer")
    if not 1 <= epoch <= TOTAL_EPOCHS_V3:
        _fail("epoch is outside the frozen 20-epoch V3 run")
    if epoch < MIDPOINT_DECAY_EPOCH_V3:
        return EARLY_LEARNING_RATE_MICROS_V3
    return LATE_LEARNING_RATE_MICROS_V3


def learning_rate_micros_for_optimizer_step_v3(step: int) -> int:
    """Map the one-based optimizer step to the frozen epoch-indexed schedule."""
    if not isinstance(step, int) or isinstance(step, bool):
        _fail("optimizer step must be an integer")
    if not 1 <= step <= TOTAL_OPTIMIZER_STEPS_V3:
        _fail("optimizer step is outside the frozen V3 run")
    epoch = ((step - 1) // EXPECTED_BATCHES_PER_EPOCH_V3) + 1
    return learning_rate_micros_for_epoch_v3(epoch)


def apply_learning_rate_v3(optimizer: object, epoch: int) -> int:
    """Apply the frozen schedule to every optimizer param group."""
    micros = learning_rate_micros_for_epoch_v3(epoch)
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or not groups:
        _fail("optimizer must expose non-empty param_groups")
    rate = micros / 1_000_000.0
    for group in groups:
        if not isinstance(group, dict) or "lr" not in group:
            _fail("optimizer param group is malformed")
        group["lr"] = rate
    return micros


def schedule_fingerprint_payload_v3() -> dict[str, int | str]:
    """Return canonical primitive schedule data for provenance binding."""
    return {
        "version": METER_REAL_DOMAIN_RETENTION_V3,
        "total_epochs": TOTAL_EPOCHS_V3,
        "batches_per_epoch": EXPECTED_BATCHES_PER_EPOCH_V3,
        "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS_V3,
        "midpoint_decay_epoch": MIDPOINT_DECAY_EPOCH_V3,
        "early_learning_rate_micros": EARLY_LEARNING_RATE_MICROS_V3,
        "late_learning_rate_micros": LATE_LEARNING_RATE_MICROS_V3,
    }


def _completed_adamw_steps(optimizer: object) -> int:
    state = getattr(optimizer, "state", None)
    if not isinstance(state, Mapping):
        _fail("AdamW optimizer state is malformed")
    steps: list[int] = []
    for item in state.values():
        if not isinstance(item, Mapping) or "step" not in item:
            continue
        value = item["step"]
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail("AdamW step state is not numeric")
        integer = int(value)
        if float(value) != float(integer) or integer < 0:
            _fail("AdamW step state is not a non-negative integer")
        steps.append(integer)
    if not steps:
        return 0
    if len(set(steps)) != 1:
        _fail("AdamW parameter step counters diverged")
    return steps[0]


def _write_retention_receipt_v3(root: Path, metrics: Mapping[str, object]) -> tuple[str, str]:
    metric_paths = sorted(root.glob("metrics-*.json"))
    if len(metric_paths) != 1:
        _fail("V3 requires exactly one underlying V2 metrics file")
    metric_path = metric_paths[0]
    if metric_path.is_symlink() or not metric_path.is_file():
        _fail("V3 underlying metrics must be a regular non-symlink file")
    metric_raw = metric_path.read_bytes()
    metric_sha = sha256(metric_raw).hexdigest()
    receipt = {
        "schema_version": RETENTION_RECEIPT_SCHEMA_V3,
        "retention_version": METER_REAL_DOMAIN_RETENTION_V3,
        "underlying_adaptation_version": metrics.get("adaptation_version"),
        "underlying_metrics_filename": metric_path.name,
        "underlying_metrics_sha256": metric_sha,
        "underlying_run_id": metrics.get("run_id"),
        "repository_sha": metrics.get("repository_sha"),
        "profile_fingerprint": metrics.get("profile_fingerprint"),
        "schedule": schedule_fingerprint_payload_v3(),
        "optimizer_steps": metrics.get("optimizer_steps"),
        "status": metrics.get("status"),
        "test_opened": metrics.get("test_opened"),
        "runtime_connected": metrics.get("runtime_connected"),
        "resolver_connected": metrics.get("resolver_connected"),
        "production_promotion_authorized": metrics.get("production_promotion_authorized"),
    }
    raw = _canonical_json(receipt)
    receipt_sha = sha256(raw).hexdigest()
    final = root / f"retention-v3-receipt-{receipt_sha}.json"
    existing = sorted(root.glob("retention-v3-receipt-*.json"))
    if existing and existing != [final]:
        _fail("conflicting V3 retention receipt already exists")
    if final.exists():
        if final.is_symlink() or not final.is_file() or final.read_bytes() != raw:
            _fail("existing V3 retention receipt does not match canonical bytes")
    else:
        temporary = root / "retention-v3-receipt.tmp"
        temporary.write_bytes(raw)
        temporary.replace(final)
    return final.name, receipt_sha


def run_meter_real_domain_retention_v3(
    *,
    teacher_bundle_root: str | Path,
    d10_root: str | Path,
    base_checkpoint_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    expected_d10_manifest_sha256: str,
    expected_d10_artifact_binding_sha256: str,
    progress=None,
    resume: bool = False,
) -> dict[str, object]:
    """Run V2 unchanged except for the frozen V3 midpoint LR schedule."""
    if not isinstance(resume, bool):
        raise TypeError("resume must be bool")
    try:
        import torch
        from torch.optim.optimizer import register_optimizer_step_pre_hook
        from .meter_real_domain_adaptation_v2 import (
            FROZEN_ADAPTATION_CONFIG_V2,
            run_meter_real_domain_adaptation_v2,
        )
    except ModuleNotFoundError as exc:
        raise MeterRealDomainRetentionV3Error(
            "torch and the pinned V2 runtime are required for Meter retention V3"
        ) from exc

    config = FROZEN_ADAPTATION_CONFIG_V2
    if config.epochs != TOTAL_EPOCHS_V3 or config.learning_rate_micros != EARLY_LEARNING_RATE_MICROS_V3:
        _fail("V2 base configuration changed beneath the V3 schedule")

    expected_optimizer_type = torch.optim.AdamW
    observed = {"last_step": 0, "last_rate_micros": 0, "training_shape_verified": False}

    def optimizer_pre_hook(optimizer, args, kwargs):
        if type(optimizer) is not expected_optimizer_type:
            _fail("unexpected optimizer stepped inside the bounded V3 invocation")
        completed = _completed_adamw_steps(optimizer)
        next_step = completed + 1
        micros = learning_rate_micros_for_optimizer_step_v3(next_step)
        epoch = ((next_step - 1) // EXPECTED_BATCHES_PER_EPOCH_V3) + 1
        apply_learning_rate_v3(optimizer, epoch)
        observed["last_step"] = next_step
        observed["last_rate_micros"] = micros
        return None

    def v3_progress(event: str, payload: Mapping[str, object]) -> None:
        translated = dict(payload)
        if event == "phase_started" and payload.get("phase") == "training_and_validation":
            if payload.get("epochs_total") != TOTAL_EPOCHS_V3:
                _fail("V3 training epoch cardinality changed")
            if payload.get("batches_per_epoch") != EXPECTED_BATCHES_PER_EPOCH_V3:
                _fail("V3 batches-per-epoch cardinality changed")
            observed["training_shape_verified"] = True
        if event == "training_batch" and not observed["training_shape_verified"]:
            _fail("V3 optimizer activity began before training shape verification")
        epoch = payload.get("epoch")
        if isinstance(epoch, int) and not isinstance(epoch, bool) and 1 <= epoch <= TOTAL_EPOCHS_V3:
            translated["retention_v3_learning_rate_micros"] = learning_rate_micros_for_epoch_v3(epoch)
        translated["retention_v3_schedule"] = schedule_fingerprint_payload_v3()
        if progress is not None:
            progress(event, translated)

    handle = register_optimizer_step_pre_hook(optimizer_pre_hook)
    try:
        metrics = run_meter_real_domain_adaptation_v2(
            teacher_bundle_root=teacher_bundle_root,
            d10_root=d10_root,
            base_checkpoint_path=base_checkpoint_path,
            output_root=output_root,
            repository_root=repository_root,
            expected_d10_manifest_sha256=expected_d10_manifest_sha256,
            expected_d10_artifact_binding_sha256=expected_d10_artifact_binding_sha256,
            config=config,
            progress=v3_progress,
            resume=resume,
        )
    finally:
        handle.remove()

    if not observed["training_shape_verified"]:
        _fail("V3 training shape was never verified")
    if metrics.get("optimizer_steps") != TOTAL_OPTIMIZER_STEPS_V3:
        _fail("V3 optimizer step count changed")
    configuration = metrics.get("configuration")
    if not isinstance(configuration, Mapping):
        _fail("V3 underlying metrics configuration is malformed")
    if configuration.get("epochs") != TOTAL_EPOCHS_V3 or configuration.get("learning_rate_micros") != EARLY_LEARNING_RATE_MICROS_V3:
        _fail("V3 underlying V2 configuration changed")
    if metrics.get("test_opened") is not False:
        _fail("V3 TEST boundary was violated")
    if metrics.get("runtime_connected") is not False or metrics.get("resolver_connected") is not False:
        _fail("V3 runtime/Resolver boundary was violated")
    if metrics.get("production_promotion_authorized") is not False:
        _fail("V3 production boundary was violated")
    history = metrics.get("history")
    if not isinstance(history, list) or len(history) != TOTAL_EPOCHS_V3:
        _fail("V3 requires the complete 20-epoch history")
    if observed["last_step"] not in {0, TOTAL_OPTIMIZER_STEPS_V3}:
        _fail("V3 ended with a partial optimizer-step observation")

    receipt_filename, receipt_sha = _write_retention_receipt_v3(Path(output_root), metrics)
    result = dict(metrics)
    result["retention_version"] = METER_REAL_DOMAIN_RETENTION_V3
    result["retention_schedule"] = schedule_fingerprint_payload_v3()
    result["retention_receipt_filename"] = receipt_filename
    result["retention_receipt_sha256"] = receipt_sha
    return result
