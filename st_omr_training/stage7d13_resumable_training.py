"""Epoch-atomic resumable Stage 7-D13 specialist training.

The authoritative D13 math/profile remains defined by ``stage7d13_training``.
This wrapper adds crash-safe epoch checkpoints outside the final run directory so
Colab/runtime loss can resume from the last fully validated epoch without
replaying any persisted optimizer step. Partial epochs are intentionally not
committed and are deterministically replayed from the previous epoch snapshot.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
from typing import Final, Mapping, Sequence

import torch

from .stage7c_execution import verify_stage7c_runtime
from .stage7d13_symbol_models import (
    acceptance_passed,
    build_symbol_model,
    detector_loss,
    encode_detector_targets,
    model_profile_fingerprint,
)
from .stage7d13_symbol_training_contract import FROZEN_D13_CONFIG, SPECIALIST_CLASSES
from .stage7d13_training import (
    STAGE7D13_TRAINING_VERSION,
    D13Record,
    SpecialistTrainingResult,
    Stage7D13TrainingError,
    Stage7D13TrainingReceipt,
    _batches,
    _canonical_json,
    _clone_state,
    _evaluate,
    _load_records,
    _repo_identity,
    _seed,
    _stack,
    training_profile_fingerprint,
)
from .stage7d13_verified_surface import (
    D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
    D13_DERIVATIVE_BUILD_ID,
    D13_DERIVATIVE_MANIFEST_SHA256,
    D13_EXPECTED_OPTIMIZER_STEPS,
    D13_EXPECTED_OPTIMIZER_STEPS_TOTAL,
    D13_RECORD_SPLIT_COUNTS,
)
from .training_model import (
    assert_finite_tensor,
    assert_model_finite,
    assert_optimizer_finite,
    count_trainable_parameters,
    model_state_sha256,
    set_deterministic_cpu,
)


STAGE7D13_RESUME_VERSION: Final[str] = "stage7d13-epoch-resume-v1"
_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_RESUME_EPOCHS: Final[range] = range(0, FROZEN_D13_CONFIG.epochs + 1)


def _fail(message: str) -> None:
    raise Stage7D13TrainingError(message)


def _sha64(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX for c in value):
        _fail(f"{name} must be lowercase SHA-256")
    return value


def _finite(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def _assert_external_dir(path: Path, repository_root: Path, name: str) -> None:
    resolved = path.resolve()
    repo = repository_root.resolve()
    if resolved == repo or repo in resolved.parents:
        _fail(f"{name} must remain outside repository")
    if path.is_symlink():
        _fail(f"{name} must not be a symlink")


def _ensure_resume_root(path: Path, repository_root: Path) -> None:
    _assert_external_dir(path, repository_root, "D13 resume root")
    if path.exists():
        if not path.is_dir():
            _fail("D13 resume root must be a directory")
    else:
        path.mkdir(parents=True)


def _atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256(path.read_bytes()).hexdigest()


def _snapshot_paths(root: Path, specialist: str, epoch: int) -> tuple[Path, Path]:
    if specialist not in SPECIALIST_CLASSES:
        _fail("unknown D13 specialist for resume")
    if epoch not in _RESUME_EPOCHS:
        _fail("D13 resume epoch outside frozen range")
    directory = root / specialist
    stem = f"epoch-{epoch:02d}"
    return directory / f"{stem}.pt", directory / f"{stem}.json"


def _snapshot_metadata(
    *, specialist: str, epoch: int, checkpoint_sha256: str,
    repository_sha: str, profile: str, optimizer_steps: int,
    history_length: int,
) -> dict[str, object]:
    return {
        "version": STAGE7D13_RESUME_VERSION,
        "specialist": specialist,
        "completed_epoch": epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "repository_sha": repository_sha,
        "training_profile_fingerprint": profile,
        "derivative_build_id": D13_DERIVATIVE_BUILD_ID,
        "derivative_manifest_sha256": D13_DERIVATIVE_MANIFEST_SHA256,
        "derivative_artifact_binding_sha256": D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
        "optimizer_steps": optimizer_steps,
        "history_length": history_length,
    }


def _save_snapshot(
    *, root: Path, specialist: str, epoch: int, repository_sha: str,
    profile: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    parameter_count: int, initial_validation_loss: float, best_loss: float,
    best_epoch: int, best_state: dict[str, torch.Tensor],
    history: Sequence[dict[str, float | int]], optimizer_steps: int,
    heartbeat=print,
) -> None:
    expected_steps = (
        D13_EXPECTED_OPTIMIZER_STEPS[specialist] // FROZEN_D13_CONFIG.epochs
    ) * epoch
    if optimizer_steps != expected_steps:
        _fail(f"{specialist} resume optimizer-step boundary mismatch")
    if len(history) != epoch:
        _fail(f"{specialist} resume history length mismatch")
    if not 0 <= best_epoch <= epoch:
        _fail(f"{specialist} resume best epoch is invalid")

    checkpoint_path, metadata_path = _snapshot_paths(root, specialist, epoch)
    checkpoint_payload = {
        "version": STAGE7D13_RESUME_VERSION,
        "specialist": specialist,
        "completed_epoch": epoch,
        "repository_sha": repository_sha,
        "training_profile_fingerprint": profile,
        "derivative_build_id": D13_DERIVATIVE_BUILD_ID,
        "derivative_manifest_sha256": D13_DERIVATIVE_MANIFEST_SHA256,
        "derivative_artifact_binding_sha256": D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
        "optimizer_steps": optimizer_steps,
        "parameter_count": parameter_count,
        "model_fingerprint": model_profile_fingerprint(specialist),
        "initial_validation_loss": initial_validation_loss,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "history": list(history),
        "model_state": _clone_state(model),
        "best_state": {
            name: value.detach().cpu().clone() for name, value in best_state.items()
        },
        "optimizer_state": optimizer.state_dict(),
    }
    checkpoint_sha = _atomic_torch_save(checkpoint_path, checkpoint_payload)
    metadata = _snapshot_metadata(
        specialist=specialist,
        epoch=epoch,
        checkpoint_sha256=checkpoint_sha,
        repository_sha=repository_sha,
        profile=profile,
        optimizer_steps=optimizer_steps,
        history_length=len(history),
    )
    _atomic_bytes(metadata_path, _canonical_json(metadata))
    if heartbeat is not None:
        heartbeat(
            f"{specialist.upper()} RESUME SAVED | EPOCH {epoch}/{FROZEN_D13_CONFIG.epochs} | "
            f"STEP {optimizer_steps}/{D13_EXPECTED_OPTIMIZER_STEPS[specialist]}"
        )


def _read_metadata(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        _fail("D13 resume metadata must be regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D13TrainingError("D13 resume metadata is invalid JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != raw:
        _fail("D13 resume metadata must be canonical JSON")
    return value


def _load_snapshot(
    *, root: Path, specialist: str, repository_sha: str, profile: str,
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, heartbeat=print,
) -> tuple[
    int, int, float, float, int, dict[str, torch.Tensor],
    list[dict[str, float | int]]
] | None:
    directory = root / specialist
    if not directory.exists():
        return None
    if directory.is_symlink() or not directory.is_dir():
        _fail(f"{specialist} resume directory must be regular directory")

    for epoch in reversed(list(_RESUME_EPOCHS)):
        checkpoint_path, metadata_path = _snapshot_paths(root, specialist, epoch)
        if not metadata_path.exists():
            continue
        metadata = _read_metadata(metadata_path)
        expected_steps = (
            D13_EXPECTED_OPTIMIZER_STEPS[specialist] // FROZEN_D13_CONFIG.epochs
        ) * epoch
        expected_metadata = _snapshot_metadata(
            specialist=specialist,
            epoch=epoch,
            checkpoint_sha256=str(metadata.get("checkpoint_sha256", "")),
            repository_sha=repository_sha,
            profile=profile,
            optimizer_steps=expected_steps,
            history_length=epoch,
        )
        if metadata != expected_metadata:
            _fail(f"{specialist} resume metadata drift")
        checkpoint_sha = _sha64(
            metadata.get("checkpoint_sha256"), "resume checkpoint SHA"
        )
        if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
            _fail(f"{specialist} committed resume checkpoint is missing")
        if sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_sha:
            _fail(f"{specialist} resume checkpoint SHA mismatch")
        try:
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise Stage7D13TrainingError(
                f"{specialist} resume checkpoint safe load failed"
            ) from exc
        if not isinstance(payload, Mapping):
            _fail(f"{specialist} resume checkpoint must be mapping")
        scalar_expected = {
            "version": STAGE7D13_RESUME_VERSION,
            "specialist": specialist,
            "completed_epoch": epoch,
            "repository_sha": repository_sha,
            "training_profile_fingerprint": profile,
            "derivative_build_id": D13_DERIVATIVE_BUILD_ID,
            "derivative_manifest_sha256": D13_DERIVATIVE_MANIFEST_SHA256,
            "derivative_artifact_binding_sha256": D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
            "optimizer_steps": expected_steps,
            "model_fingerprint": model_profile_fingerprint(specialist),
        }
        for name, expected in scalar_expected.items():
            if payload.get(name) != expected:
                _fail(f"{specialist} resume {name} mismatch")
        if payload.get("parameter_count") != count_trainable_parameters(model):
            _fail(f"{specialist} resume parameter count mismatch")
        history_raw = payload.get("history")
        if not isinstance(history_raw, list) or len(history_raw) != epoch:
            _fail(f"{specialist} resume history malformed")
        history: list[dict[str, float | int]] = []
        for expected_epoch, row in enumerate(history_raw, start=1):
            if not isinstance(row, Mapping) or row.get("epoch") != expected_epoch:
                _fail(f"{specialist} resume history epoch mismatch")
            history.append(dict(row))
        initial_loss = _finite(
            payload.get("initial_validation_loss"), "resume initial validation loss"
        )
        best_loss = _finite(payload.get("best_loss"), "resume best loss")
        best_epoch = payload.get("best_epoch")
        if (
            not isinstance(best_epoch, int)
            or isinstance(best_epoch, bool)
            or not 0 <= best_epoch <= epoch
        ):
            _fail(f"{specialist} resume best epoch invalid")
        model_state = payload.get("model_state")
        best_state_raw = payload.get("best_state")
        optimizer_state = payload.get("optimizer_state")
        if (
            not isinstance(model_state, Mapping)
            or not isinstance(best_state_raw, Mapping)
            or not isinstance(optimizer_state, Mapping)
        ):
            _fail(f"{specialist} resume state payload malformed")
        try:
            model.load_state_dict(dict(model_state), strict=True)
            optimizer.load_state_dict(dict(optimizer_state))
        except Exception as exc:
            raise Stage7D13TrainingError(
                f"{specialist} resume state load failed"
            ) from exc
        assert_model_finite(model)
        assert_optimizer_finite(optimizer)
        best_state = {
            str(name): value.detach().cpu().clone()
            for name, value in best_state_raw.items()
        }
        if heartbeat is not None:
            heartbeat(
                f"{specialist.upper()} RESUME LOADED | EPOCH {epoch}/{FROZEN_D13_CONFIG.epochs} | "
                f"STEP {expected_steps}/{D13_EXPECTED_OPTIMIZER_STEPS[specialist]}"
            )
        return (
            epoch, expected_steps, initial_loss, best_loss,
            best_epoch, best_state, history,
        )
    return None


def _train_specialist_resumable(
    specialist: str,
    train_records: Sequence[D13Record],
    validation_records: Sequence[D13Record],
    *, resume_root: Path, repository_sha: str, profile: str, heartbeat=print,
) -> tuple[SpecialistTrainingResult, dict[str, torch.Tensor]]:
    set_deterministic_cpu(_seed(specialist, 0))
    model = build_symbol_model(specialist, seed=_seed(specialist, 0))
    parameter_count = count_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FROZEN_D13_CONFIG.learning_rate_micros / 1_000_000,
        weight_decay=FROZEN_D13_CONFIG.weight_decay_micros / 1_000_000,
        foreach=False,
        fused=False,
    )

    resumed = _load_snapshot(
        root=resume_root,
        specialist=specialist,
        repository_sha=repository_sha,
        profile=profile,
        model=model,
        optimizer=optimizer,
        heartbeat=heartbeat,
    )
    if resumed is None:
        initial_loss, _initial_metrics = _evaluate(
            model, validation_records, specialist
        )
        best_loss = initial_loss
        best_epoch = 0
        best_state = _clone_state(model)
        history: list[dict[str, float | int]] = []
        steps = 0
        completed_epoch = 0
        _save_snapshot(
            root=resume_root,
            specialist=specialist,
            epoch=0,
            repository_sha=repository_sha,
            profile=profile,
            model=model,
            optimizer=optimizer,
            parameter_count=parameter_count,
            initial_validation_loss=initial_loss,
            best_loss=best_loss,
            best_epoch=best_epoch,
            best_state=best_state,
            history=history,
            optimizer_steps=steps,
            heartbeat=heartbeat,
        )
    else:
        (
            completed_epoch, steps, initial_loss, best_loss,
            best_epoch, best_state, history,
        ) = resumed

    for epoch in range(completed_epoch + 1, FROZEN_D13_CONFIG.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        batches = _batches(train_records, specialist, epoch)
        for batch_index, batch in enumerate(batches, start=1):
            images, rows = _stack(batch, specialist)
            targets = encode_detector_targets(specialist, rows)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = detector_loss(specialist, outputs, targets)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    assert_finite_tensor(
                        f"D13 gradient {specialist}.{name}", parameter.grad
                    )
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=FROZEN_D13_CONFIG.grad_clip_milli / 1000.0,
                error_if_nonfinite=True,
            )
            if not math.isfinite(float(norm)):
                _fail("D13 gradient norm is non-finite")
            optimizer.step()
            assert_model_finite(model)
            assert_optimizer_finite(optimizer)
            value = float(loss.detach().item())
            if not math.isfinite(value):
                _fail("D13 training loss is non-finite")
            loss_sum += value * len(batch)
            seen += len(batch)
            steps += 1
            if (
                heartbeat is not None
                and batch_index % FROZEN_D13_CONFIG.heartbeat_batches == 0
            ):
                heartbeat(
                    f"{specialist.upper()} EPOCH {epoch}/{FROZEN_D13_CONFIG.epochs} "
                    f"BATCH {batch_index}/{len(batches)} "
                    f"STEP {steps}/{D13_EXPECTED_OPTIMIZER_STEPS[specialist]}"
                )
        validation_loss, metrics = _evaluate(
            model, validation_records, specialist
        )
        train_loss = loss_sum / seen
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "center_f1_4px": metrics.class_aware_center_f1_4px,
                "bbox_f1_iou50": metrics.class_aware_bbox_f1_iou50,
                "macro_class_f1": metrics.macro_class_f1,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _clone_state(model)
        _save_snapshot(
            root=resume_root,
            specialist=specialist,
            epoch=epoch,
            repository_sha=repository_sha,
            profile=profile,
            model=model,
            optimizer=optimizer,
            parameter_count=parameter_count,
            initial_validation_loss=initial_loss,
            best_loss=best_loss,
            best_epoch=best_epoch,
            best_state=best_state,
            history=history,
            optimizer_steps=steps,
            heartbeat=heartbeat,
        )
        if heartbeat is not None:
            heartbeat(
                f"{specialist.upper()} EPOCH {epoch}/{FROZEN_D13_CONFIG.epochs} | "
                f"VAL LOSS={validation_loss:.8f} | "
                f"CENTER-F1={metrics.class_aware_center_f1_4px:.6f} | "
                f"BBOX-F1={metrics.class_aware_bbox_f1_iou50:.6f} | "
                f"MACRO-F1={metrics.macro_class_f1:.6f}"
            )

    if steps != D13_EXPECTED_OPTIMIZER_STEPS[specialist]:
        _fail(f"{specialist} optimizer-step count mismatch")
    model.load_state_dict(best_state, strict=True)
    assert_model_finite(model)
    final_loss, final_metrics = _evaluate(model, validation_records, specialist)
    if not math.isclose(final_loss, best_loss, rel_tol=0.0, abs_tol=1e-12):
        _fail(
            f"{specialist} restored state does not reproduce best validation loss"
        )
    result = SpecialistTrainingResult(
        specialist=specialist,
        parameter_count=parameter_count,
        model_fingerprint=model_profile_fingerprint(specialist),
        final_state_sha256=model_state_sha256(model),
        optimizer_steps=steps,
        best_epoch=best_epoch,
        initial_validation_loss=initial_loss,
        final_validation_loss=final_loss,
        metrics=final_metrics,
        accepted=acceptance_passed(specialist, final_metrics),
        history=tuple(history),
    )
    return result, best_state


def _prepare_staging(
    staging: Path, repository: Path, *, run_id: str, profile: str
) -> None:
    _assert_external_dir(staging, repository, "D13 finalization staging root")
    marker_payload = {
        "version": STAGE7D13_RESUME_VERSION,
        "purpose": "finalization-staging",
        "run_id": run_id,
        "training_profile_fingerprint": profile,
    }
    marker_raw = _canonical_json(marker_payload)
    marker = staging / "STAGING.json"
    if staging.exists():
        if (
            staging.is_symlink()
            or not staging.is_dir()
            or not marker.is_file()
            or marker.is_symlink()
        ):
            _fail(
                "D13 stale finalization staging root is not safely recognizable"
            )
        if marker.read_bytes() != marker_raw:
            _fail("D13 stale finalization staging identity mismatch")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    _atomic_bytes(marker, marker_raw)


def run_stage7d13_resumable_training(
    *, derivative_root: str | Path, output_root: str | Path,
    repository_root: str | Path, expected_repository_sha: str,
    resume_root: str | Path | None = None, heartbeat=print,
) -> Stage7D13TrainingReceipt:
    """Train all D13 specialists with epoch-atomic resume checkpoints."""
    derivative = Path(derivative_root)
    output = Path(output_root)
    repository = Path(repository_root)
    head_before, origin_before = _repo_identity(repository)
    if head_before != expected_repository_sha:
        _fail("D13 training repository HEAD differs from authorized exact head")
    verify_stage7c_runtime()
    if output.exists() or output.is_symlink():
        _fail("D13 final training output must be absent before resumable training")
    records = _load_records(derivative)
    train_records = tuple(row for row in records if row.split == "train")
    validation_records = tuple(row for row in records if row.split == "validation")
    if (
        len(train_records) != D13_RECORD_SPLIT_COUNTS["train"]
        or len(validation_records) != D13_RECORD_SPLIT_COUNTS["validation"]
    ):
        _fail("D13 training record cardinality drift")

    profile = training_profile_fingerprint()
    run_id = sha256(
        _canonical_json(
            {
                "version": STAGE7D13_TRAINING_VERSION,
                "repository_sha": head_before,
                "training_profile_fingerprint": profile,
                "derivative_manifest_sha256": D13_DERIVATIVE_MANIFEST_SHA256,
            }
        )
    ).hexdigest()
    resume = (
        Path(resume_root)
        if resume_root is not None
        else output.with_name(output.name + ".resume")
    )
    _ensure_resume_root(resume, repository)
    if heartbeat is not None:
        heartbeat(f"D13 RESUME ROOT | {resume}")

    results: dict[str, SpecialistTrainingResult] = {}
    checkpoint: dict[str, dict[str, torch.Tensor]] = {}
    for specialist in SPECIALIST_CLASSES:
        result, state = _train_specialist_resumable(
            specialist,
            train_records,
            validation_records,
            resume_root=resume,
            repository_sha=head_before,
            profile=profile,
            heartbeat=heartbeat,
        )
        results[specialist] = result
        checkpoint[specialist] = state

    steps = {name: result.optimizer_steps for name, result in results.items()}
    if (
        steps != D13_EXPECTED_OPTIMIZER_STEPS
        or sum(steps.values()) != D13_EXPECTED_OPTIMIZER_STEPS_TOTAL
    ):
        _fail("D13 total optimizer-step evidence mismatch")

    head_after, origin_after = _repo_identity(repository)
    verify_stage7c_runtime()
    if (head_after, origin_after) != (head_before, origin_before):
        _fail("repository identity changed during D13 training")

    specialists_payload = {
        name: {
            "parameter_count": result.parameter_count,
            "model_fingerprint": result.model_fingerprint,
            "final_state_sha256": result.final_state_sha256,
            "optimizer_steps": result.optimizer_steps,
            "best_epoch": result.best_epoch,
            "initial_validation_loss": result.initial_validation_loss,
            "final_validation_loss": result.final_validation_loss,
            "metrics": asdict(result.metrics),
            "accepted": result.accepted,
            "history": list(result.history),
        }
        for name, result in results.items()
    }
    metrics_payload = {
        "version": STAGE7D13_TRAINING_VERSION,
        "run_id": run_id,
        "training_profile_fingerprint": profile,
        "specialists": specialists_payload,
        "acceptance": all(result.accepted for result in results.values()),
    }
    metrics_raw = _canonical_json(metrics_payload)
    metrics_sha = sha256(metrics_raw).hexdigest()

    staging = output.with_name(output.name + ".finalizing")
    _prepare_staging(staging, repository, run_id=run_id, profile=profile)
    checkpoint_path = staging / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = sha256(checkpoint_path.read_bytes()).hexdigest()
    _atomic_bytes(staging / "metrics.json", metrics_raw)

    run_payload = {
        "version": STAGE7D13_TRAINING_VERSION,
        "run_id": run_id,
        "repository_sha": head_before,
        "repository_origin": origin_before,
        "training_profile_fingerprint": profile,
        "derivative": {
            "build_id": D13_DERIVATIVE_BUILD_ID,
            "manifest_sha256": D13_DERIVATIVE_MANIFEST_SHA256,
            "artifact_binding_sha256": D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
            "train_records": len(train_records),
            "validation_records": len(validation_records),
            "test_records": 0,
        },
        "optimizer_steps": steps,
        "optimizer_steps_total": sum(steps.values()),
        "checkpoint_sha256": checkpoint_sha,
        "metrics_sha256": metrics_sha,
        "test_opened": False,
        "complete_marker_written": False,
    }
    run_raw = _canonical_json(run_payload)
    run_sha = sha256(run_raw).hexdigest()
    _atomic_bytes(staging / "run.json", run_raw)
    (staging / "STAGING.json").unlink()
    os.replace(staging, output)

    if heartbeat is not None:
        heartbeat(
            "D13 FINAL TRAINING SURFACE PERSISTED | "
            "resume snapshots retained outside final run"
        )

    return Stage7D13TrainingReceipt(
        version=STAGE7D13_TRAINING_VERSION,
        run_id=run_id,
        repository_sha=head_before,
        repository_origin=origin_before,
        training_profile_fingerprint=profile,
        derivative_build_id=D13_DERIVATIVE_BUILD_ID,
        derivative_manifest_sha256=D13_DERIVATIVE_MANIFEST_SHA256,
        derivative_artifact_binding_sha256=D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
        train_records=len(train_records),
        validation_records=len(validation_records),
        test_opened=False,
        optimizer_steps=steps,
        optimizer_steps_total=sum(steps.values()),
        checkpoint_sha256=checkpoint_sha,
        metrics_sha256=metrics_sha,
        run_sha256=run_sha,
        specialists=specialists_payload,
        acceptance=all(result.accepted for result in results.values()),
        complete_marker_written=False,
    )
