"""Durable shadow execution for Meter V3-A1.

This run keeps the exact D11 model and all bbox outputs frozen. Only the
classification adapter from meter_real_domain_adaptation_v3_a1 may update.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import math
from pathlib import Path
import random
from typing import Final

from .meter_real_domain_adaptation_v1 import (
    PRESENCE_D11_SHA256,
    _canonical_json,
    _evaluate_synthetic,
    _evaluate_teacher,
    _evaluation_from_payload_v1,
    _hex64,
    _load_teacher_records,
    _prepare_output_root,
    _read_regular,
    _sha,
    _stack_teacher,
    _teacher_inference_fingerprint,
    _teacher_target,
    deterministic_replay_ids_v1,
)
from .meter_real_domain_adaptation_v2 import (
    AdaptationGateDecisionV2,
    _augment_real,
    _balanced_real_records,
    _better_candidate,
    _clone_state,
)
from .meter_real_domain_adaptation_v3_a1 import (
    FROZEN_ADAPTATION_CONFIG_V3_A1,
    METER_REAL_DOMAIN_ADAPTATION_V3_A1,
    MeterRealDomainAdaptationConfigV3A1,
    MeterRealDomainAdaptationV3A1Error,
    adaptation_acceptance_v3_a1,
    build_meter_classification_adapter_v3_a1,
    meter_real_domain_adaptation_fingerprint_v3_a1,
    train_batch_v3_a1,
)
from .meter_teacher_gold_admission_v1 import METER_CLASSES, verify_meter_teacher_gold_bundle_v1


METRICS_SCHEMA_V3_A1: Final[str] = "st-omr-meter-real-domain-adaptation-metrics-v3-a1"
VERIFICATION_SCHEMA_V3_A1: Final[str] = "st-omr-meter-real-domain-adaptation-verification-v3-a1"
CHECKPOINT_ROLE_V3_A1: Final[str] = "meter-real-domain-shadow-candidate-v3-a1"
RESUME_ROLE_V3_A1: Final[str] = "meter-real-domain-adaptation-resume-v3-a1"


def _fail(message: str) -> None:
    raise MeterRealDomainAdaptationV3A1Error(message)


def run_meter_real_domain_adaptation_v3_a1(
    *,
    teacher_bundle_root: str | Path,
    d10_root: str | Path,
    base_checkpoint_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    expected_d10_manifest_sha256: str,
    expected_d10_artifact_binding_sha256: str,
    config: MeterRealDomainAdaptationConfigV3A1 = FROZEN_ADAPTATION_CONFIG_V3_A1,
    progress=None,
    resume: bool = False,
) -> dict[str, object]:
    if not isinstance(config, MeterRealDomainAdaptationConfigV3A1):
        raise TypeError("config must be MeterRealDomainAdaptationConfigV3A1")
    if config != FROZEN_ADAPTATION_CONFIG_V3_A1:
        _fail("Meter V3-A1 requires the frozen configuration")
    if not isinstance(resume, bool):
        raise TypeError("resume must be bool")

    try:
        import torch
        from .runtime_meter_real_checkpoint_audit_v1 import audit_presence_d11_checkpoint_v1
        from .stage7c_execution import verify_authoritative_repository
        from .stage7d11_barline_meter_training import (
            FROZEN_D11_CONFIG,
            _load_d11_label,
            _stack_meter,
            build_meter_refiner,
            load_verified_stage7d11_records,
        )
        from .training_model import assert_model_finite, model_state_sha256, set_deterministic_cpu
    except ModuleNotFoundError as exc:
        raise MeterRealDomainAdaptationV3A1Error(
            "torch and the pinned training runtime are required for Meter V3-A1"
        ) from exc

    teacher_root = Path(teacher_bundle_root)
    if progress is not None:
        progress("phase_started", {"phase": "teacher_gold_verify", "phase_index": 1, "phase_total": 7})
    teacher_receipt = verify_meter_teacher_gold_bundle_v1(teacher_root)
    teacher_records = _load_teacher_records(teacher_root)
    real_train = tuple(record for record in teacher_records if record.split == "train")
    real_validation = tuple(record for record in teacher_records if record.split == "validation")
    balanced_real = _balanced_real_records(real_train, config.real_balanced_repeat_factor)

    d10_manifest = _hex64("expected D10 manifest SHA", expected_d10_manifest_sha256)
    d10_binding = _hex64("expected D10 artifact binding SHA", expected_d10_artifact_binding_sha256)
    if progress is not None:
        progress(
            "phase_started",
            {"phase": "d10_full_integrity_verify", "phase_index": 2, "phase_total": 7, "records_total": 22_128},
        )
    d10_records = load_verified_stage7d11_records(
        d10_root,
        expected_manifest_sha256=d10_manifest,
        expected_artifact_binding_sha256=d10_binding,
    )
    synthetic_train_all = tuple(
        record for record in d10_records if record.kind == "meter" and record.split == "train"
    )
    synthetic_validation = tuple(
        record for record in d10_records if record.kind == "meter" and record.split == "validation"
    )
    if len(synthetic_train_all) != 9_840 or len(synthetic_validation) != 1_224:
        _fail("accepted D10 Meter surface cardinality changed")

    class_to_ids: defaultdict[str, list[str]] = defaultdict(list)
    record_by_id = {record.record_id: record for record in synthetic_train_all}
    for record in synthetic_train_all:
        target = _load_d11_label(record).get("target")
        if not isinstance(target, Mapping) or target.get("meter_class") not in METER_CLASSES:
            _fail("D10 replay target class is invalid")
        class_to_ids[str(target["meter_class"])].append(record.record_id)
    replay_ids = deterministic_replay_ids_v1(
        class_to_ids,
        per_class=config.synthetic_replay_per_class,
        seed=config.master_seed,
    )
    synthetic_replay = tuple(record_by_id[record_id] for record_id in replay_ids)

    base_path = Path(base_checkpoint_path)
    if progress is not None:
        progress("phase_started", {"phase": "d11_checkpoint_audit", "phase_index": 3, "phase_total": 7})
    if _sha(_read_regular(base_path, maximum=64 * 1024 * 1024, name="base D11 checkpoint")) != PRESENCE_D11_SHA256:
        _fail("base D11 checkpoint SHA-256 mismatch")
    audited = audit_presence_d11_checkpoint_v1(base_path)
    if audited.checkpoint_sha256 != PRESENCE_D11_SHA256 or audited.role != "presence-d11-bridge":
        _fail("base checkpoint audit did not return the exact D11 Meter state")

    repository_sha, repository_origin = verify_authoritative_repository(repository_root)
    profile = meter_real_domain_adaptation_fingerprint_v3_a1(
        teacher_manifest_sha256=teacher_receipt.manifest_sha256,
        d10_manifest_sha256=d10_manifest,
        d10_artifact_binding_sha256=d10_binding,
        config=config,
    )
    root = Path(output_root)
    _prepare_output_root(root, Path(repository_root), resume=resume)

    set_deterministic_cpu(config.master_seed)
    base_model = build_meter_refiner(FROZEN_D11_CONFIG)
    base_model.load_state_dict(dict(audited.model_state), strict=True)
    for parameter in base_model.parameters():
        parameter.requires_grad = False
    base_state_sha = model_state_sha256(base_model)
    model = build_meter_classification_adapter_v3_a1(base_model, config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.requires_grad for parameter in model.base.parameters()):
        _fail("V3-A1 must train only the classification adapter")
    if hasattr(model, "bbox_delta_head"):
        _fail("V3-A1 must not expose a trainable bbox head")
    assert_model_finite(model)

    if progress is not None:
        progress("phase_started", {"phase": "baseline_real_validation", "phase_index": 4, "phase_total": 7})
    baseline_real = _evaluate_teacher(base_model, real_validation, config)
    if progress is not None:
        progress(
            "phase_started",
            {"phase": "baseline_synthetic_validation", "phase_index": 5, "phase_total": 7, "records_total": 1_224},
        )
    baseline_synthetic = _evaluate_synthetic(base_model, synthetic_validation, config)

    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate_micros / 1_000_000.0,
        weight_decay=config.weight_decay_micros / 1_000_000.0,
    )
    training_items = [("real", record) for record in balanced_real]
    training_items += [("synthetic", record) for record in synthetic_replay]
    batches_per_epoch = math.ceil(len(training_items) / config.batch_size)

    history: list[dict[str, object]] = []
    best_state = None
    best_decision = AdaptationGateDecisionV2(False, ("NO_EPOCH_EVALUATED",))
    best_real = baseline_real
    best_synthetic = baseline_synthetic
    best_epoch = 0
    optimizer_steps = 0
    completed_epoch = 0

    resume_path = root / "resume.pt"
    if resume_path.exists():
        if resume_path.is_symlink() or not resume_path.is_file():
            _fail("V3-A1 resume state must be a regular file")
        try:
            snapshot = torch.load(resume_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise MeterRealDomainAdaptationV3A1Error("V3-A1 resume state cannot be loaded safely") from exc
        if not isinstance(snapshot, Mapping):
            _fail("V3-A1 resume state must be a mapping")
        checks = {
            "role": RESUME_ROLE_V3_A1,
            "adaptation_version": METER_REAL_DOMAIN_ADAPTATION_V3_A1,
            "repository_sha": repository_sha,
            "profile_fingerprint": profile,
            "teacher_manifest_sha256": teacher_receipt.manifest_sha256,
            "d10_manifest_sha256": d10_manifest,
            "base_checkpoint_sha256": PRESENCE_D11_SHA256,
            "base_meter_state_sha256": base_state_sha,
            "baseline_real": asdict(baseline_real),
            "baseline_synthetic": asdict(baseline_synthetic),
        }
        for name, expected in checks.items():
            if snapshot.get(name) != expected:
                _fail(f"V3-A1 resume state {name} mismatch")
        try:
            completed_epoch = int(snapshot["completed_epoch"])
            best_epoch = int(snapshot["best_epoch"])
            optimizer_steps = int(snapshot["optimizer_steps"])
            if not 1 <= completed_epoch <= config.epochs:
                _fail("V3-A1 resume epoch is outside the frozen run")
            model.load_state_dict(snapshot["current_model_state"], strict=True)
            optimizer.load_state_dict(snapshot["optimizer_state_dict"])
            loaded_best = snapshot.get("best_model_state")
            best_state = None if loaded_best is None else dict(loaded_best)
            decision = snapshot["best_decision"]
            if not isinstance(decision, Mapping):
                _fail("V3-A1 resume best decision is malformed")
            reasons = decision.get("reasons")
            if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes, bytearray)):
                _fail("V3-A1 resume reasons are malformed")
            best_decision = AdaptationGateDecisionV2(
                bool(decision.get("accepted")), tuple(str(value) for value in reasons)
            )
            best_real = _evaluation_from_payload_v1(snapshot["best_real"], name="V3-A1 resume best real")
            best_synthetic = _evaluation_from_payload_v1(
                snapshot["best_synthetic"], name="V3-A1 resume best synthetic"
            )
            history = list(snapshot["history"])
        except MeterRealDomainAdaptationV3A1Error:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise MeterRealDomainAdaptationV3A1Error("V3-A1 resume state is malformed") from exc
        if progress is not None:
            progress(
                "resume_loaded",
                {"completed_epoch": completed_epoch, "epochs_total": config.epochs, "optimizer_steps": optimizer_steps},
            )

    if len(history) != completed_epoch or optimizer_steps != completed_epoch * batches_per_epoch:
        _fail("V3-A1 resume history/optimizer counters are inconsistent")

    if progress is not None:
        progress(
            "phase_started",
            {
                "phase": "training_and_validation",
                "phase_index": 6,
                "phase_total": 7,
                "completed_epoch": completed_epoch,
                "epochs_total": config.epochs,
                "batches_per_epoch": batches_per_epoch,
            },
        )

    for epoch in range(completed_epoch + 1, config.epochs + 1):
        order = list(range(len(training_items)))
        random.Random(config.master_seed + epoch * 1_000_003).shuffle(order)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(order), config.batch_size):
            items = [training_items[index] for index in order[start : start + config.batch_size]]
            teacher_batch = [record for kind, record in items if kind == "real"]
            synthetic_batch = [record for kind, record in items if kind == "synthetic"]
            tensors = []
            if teacher_batch:
                real_tensors = list(_stack_teacher(teacher_batch))
                real_tensors[0] = _augment_real(
                    real_tensors[0],
                    seed=config.master_seed + epoch * 10_000_019 + batches * 1_009,
                    maximum_shift=config.augmentation_shift_px,
                )
                tensors.append(tuple(real_tensors))
            if synthetic_batch:
                tensors.append(_stack_meter(synthetic_batch))
            images = torch.cat([value[0] for value in tensors], dim=0)
            classes = torch.cat([value[1] for value in tensors], dim=0)
            positive = torch.cat([value[3] for value in tensors], dim=0)
            total_loss += train_batch_v3_a1(
                model,
                images,
                classes,
                positive,
                real_count=len(teacher_batch),
                optimizer=optimizer,
                config=config,
            )
            optimizer_steps += 1
            batches += 1
            if progress is not None:
                progress(
                    "training_batch",
                    {
                        "epoch": epoch,
                        "epochs_total": config.epochs,
                        "batch": batches,
                        "batches_total": batches_per_epoch,
                        "optimizer_steps": optimizer_steps,
                    },
                )

        real_metrics = _evaluate_teacher(model, real_validation, config)
        synthetic_metrics = _evaluate_synthetic(model, synthetic_validation, config)
        if synthetic_metrics.positive_localization_f1_2px != baseline_synthetic.positive_localization_f1_2px:
            _fail("V3-A1 changed synthetic localization despite exact frozen bbox output")
        decision = adaptation_acceptance_v3_a1(
            candidate_real=real_metrics,
            baseline_synthetic=baseline_synthetic,
            candidate_synthetic=synthetic_metrics,
            config=config,
        )
        event = {
            "epoch": epoch,
            "train_loss": total_loss / batches,
            "real_validation": asdict(real_metrics),
            "synthetic_validation": asdict(synthetic_metrics),
            "gate": asdict(decision),
        }
        history.append(event)
        if decision.accepted and (
            best_state is None
            or _better_candidate(real_metrics, synthetic_metrics, best_real, best_synthetic)
        ):
            best_state = _clone_state(model)
            best_decision = decision
            best_real = real_metrics
            best_synthetic = synthetic_metrics
            best_epoch = epoch
        elif best_state is None and (
            best_epoch == 0
            or _better_candidate(real_metrics, synthetic_metrics, best_real, best_synthetic)
        ):
            best_decision = decision
            best_real = real_metrics
            best_synthetic = synthetic_metrics
            best_epoch = epoch

        temporary_resume = root / "resume.tmp.pt"
        torch.save(
            {
                "role": RESUME_ROLE_V3_A1,
                "adaptation_version": METER_REAL_DOMAIN_ADAPTATION_V3_A1,
                "repository_sha": repository_sha,
                "profile_fingerprint": profile,
                "teacher_manifest_sha256": teacher_receipt.manifest_sha256,
                "d10_manifest_sha256": d10_manifest,
                "base_checkpoint_sha256": PRESENCE_D11_SHA256,
                "base_meter_state_sha256": base_state_sha,
                "baseline_real": asdict(baseline_real),
                "baseline_synthetic": asdict(baseline_synthetic),
                "completed_epoch": epoch,
                "current_model_state": _clone_state(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_model_state": best_state,
                "best_decision": asdict(best_decision),
                "best_real": asdict(best_real),
                "best_synthetic": asdict(best_synthetic),
                "best_epoch": best_epoch,
                "optimizer_steps": optimizer_steps,
                "history": history,
            },
            temporary_resume,
        )
        temporary_resume.replace(resume_path)
        if progress is not None:
            progress("epoch_checkpointed", {"epoch": epoch, "epochs_total": config.epochs, "resume_path": str(resume_path)})
            progress("epoch_complete", event)

    if progress is not None:
        progress("phase_started", {"phase": "final_verification", "phase_index": 7, "phase_total": 7})
    ending_sha, ending_origin = verify_authoritative_repository(repository_root)
    if (ending_sha, ending_origin) != (repository_sha, repository_origin):
        _fail("repository identity changed during Meter V3-A1")
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    assert_model_finite(model)
    if model_state_sha256(model.base) != base_state_sha:
        _fail("V3-A1 mutated the fully frozen D11 model")

    candidate_state_sha = model_state_sha256(model) if best_state is not None else None
    replay_fingerprints: tuple[str, ...] = ()
    if best_state is not None:
        replay_fingerprints = tuple(
            _teacher_inference_fingerprint(model, real_validation) for _ in range(10)
        )
        if len(set(replay_fingerprints)) != 1:
            _fail("V3-A1 candidate inference is not deterministic 10/10")

    status = "SHADOW_CANDIDATE_ACCEPTED" if best_state is not None else "HOLD_NO_ACCEPTED_CANDIDATE"
    run_id = _sha(
        _canonical_json(
            {
                "version": METER_REAL_DOMAIN_ADAPTATION_V3_A1,
                "repository_sha": repository_sha,
                "profile_fingerprint": profile,
                "teacher_manifest_sha256": teacher_receipt.manifest_sha256,
                "d10_manifest_sha256": d10_manifest,
            }
        )
    )

    checkpoint_path = None
    checkpoint_sha = None
    checkpoint_reload_verified = False
    if best_state is not None:
        temporary = root / "checkpoint.tmp.pt"
        torch.save(
            {
                "role": CHECKPOINT_ROLE_V3_A1,
                "model_state_dict": best_state,
                "base_checkpoint_sha256": PRESENCE_D11_SHA256,
                "base_meter_state_sha256": base_state_sha,
                "candidate_meter_state_sha256": candidate_state_sha,
                "profile_fingerprint": profile,
                "teacher_manifest_sha256": teacher_receipt.manifest_sha256,
                "d10_manifest_sha256": d10_manifest,
                "best_epoch": best_epoch,
                "runtime_connected": False,
                "production_promotion_authorized": False,
            },
            temporary,
        )
        checkpoint_sha = _sha(temporary.read_bytes())
        checkpoint_path = root / f"checkpoint-{checkpoint_sha}.pt"
        temporary.rename(checkpoint_path)
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping) or payload.get("role") != CHECKPOINT_ROLE_V3_A1:
                _fail("V3-A1 checkpoint role changed during reload")
            reload_base = build_meter_refiner(FROZEN_D11_CONFIG)
            reload_base.load_state_dict(dict(audited.model_state), strict=True)
            reloaded = build_meter_classification_adapter_v3_a1(reload_base, config)
            reloaded.load_state_dict(payload["model_state_dict"], strict=True)
            assert_model_finite(reloaded)
        except MeterRealDomainAdaptationV3A1Error:
            raise
        except Exception as exc:
            raise MeterRealDomainAdaptationV3A1Error("V3-A1 checkpoint cannot be strictly reloaded") from exc
        if model_state_sha256(reloaded) != candidate_state_sha or model_state_sha256(reloaded.base) != base_state_sha:
            _fail("V3-A1 checkpoint state hash mismatch after reload")
        checkpoint_reload_verified = True

    metrics = {
        "schema_version": METRICS_SCHEMA_V3_A1,
        "adaptation_version": METER_REAL_DOMAIN_ADAPTATION_V3_A1,
        "status": status,
        "run_id": run_id,
        "repository_sha": repository_sha,
        "repository_origin": repository_origin,
        "profile_fingerprint": profile,
        "configuration": asdict(config),
        "base_checkpoint_sha256": PRESENCE_D11_SHA256,
        "base_meter_state_sha256": base_state_sha,
        "teacher_gold": {
            "manifest_sha256": teacher_receipt.manifest_sha256,
            "artifact_binding_sha256": teacher_receipt.artifact_binding_sha256,
            "train_records": 54,
            "validation_records": 18,
        },
        "synthetic_replay": {
            "d10_manifest_sha256": d10_manifest,
            "d10_artifact_binding_sha256": d10_binding,
            "train_records": len(synthetic_replay),
            "train_records_per_class": config.synthetic_replay_per_class,
            "validation_records": len(synthetic_validation),
        },
        "baseline": {
            "real_validation": asdict(baseline_real),
            "synthetic_validation": asdict(baseline_synthetic),
        },
        "best": {
            "epoch": best_epoch,
            "real_validation": asdict(best_real),
            "synthetic_validation": asdict(best_synthetic),
            "gate": asdict(best_decision),
            "candidate_meter_state_sha256": candidate_state_sha,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_filename": checkpoint_path.name if checkpoint_path is not None else None,
        },
        "history": history,
        "optimizer_steps": optimizer_steps,
        "d11_fully_frozen": True,
        "bbox_frozen_exact": True,
        "source_retention": "d10-logit-distillation-plus-adapter-residual-zero",
        "candidate_replay_10_of_10": best_state is not None and len(set(replay_fingerprints)) == 1,
        "candidate_replay_fingerprint": replay_fingerprints[0] if replay_fingerprints else None,
        "checkpoint_reload_verified": checkpoint_reload_verified,
        "test_records": 0,
        "test_opened": False,
        "runtime_connected": False,
        "resolver_connected": False,
        "production_promotion_authorized": False,
    }
    metrics_raw = _canonical_json(metrics)
    metrics_sha = _sha(metrics_raw)
    metrics_path = root / f"metrics-{metrics_sha}.json"
    metrics_path.write_bytes(metrics_raw)

    verification = {
        "schema_version": VERIFICATION_SCHEMA_V3_A1,
        "adaptation_version": METER_REAL_DOMAIN_ADAPTATION_V3_A1,
        "status": status,
        "run_id": run_id,
        "repository_sha": repository_sha,
        "profile_fingerprint": profile,
        "metrics_sha256": metrics_sha,
        "checkpoint_sha256": checkpoint_sha,
        "base_checkpoint_sha256": PRESENCE_D11_SHA256,
        "teacher_manifest_sha256": teacher_receipt.manifest_sha256,
        "d10_manifest_sha256": d10_manifest,
        "d10_artifact_binding_sha256": d10_binding,
        "optimizer_steps": optimizer_steps,
        "d11_fully_frozen": True,
        "bbox_frozen_exact": True,
        "checkpoint_is_shadow_only": True,
        "candidate_replay_10_of_10": best_state is not None and len(set(replay_fingerprints)) == 1,
        "checkpoint_reload_verified": checkpoint_reload_verified,
        "test_records": 0,
        "test_opened": False,
        "runtime_connected": False,
        "resolver_connected": False,
        "production_promotion_authorized": False,
        "repository_stable_during_run": True,
    }
    verification_raw = _canonical_json(verification)
    verification_sha = _sha(verification_raw)
    verification_path = root / f"verification-{verification_sha}.json"
    verification_path.write_bytes(verification_raw)

    lines = [
        f"{verification_sha}  {verification_path.name}",
        f"{metrics_sha}  {metrics_path.name}",
    ]
    if checkpoint_path is not None and checkpoint_sha is not None:
        lines.append(f"{checkpoint_sha}  {checkpoint_path.name}")
    (root / "RUN_COMPLETE").write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    if progress is not None:
        progress(
            "run_complete",
            {"status": status, "best_epoch": best_epoch, "epochs_total": config.epochs, "run_root": str(root)},
        )
    return metrics
