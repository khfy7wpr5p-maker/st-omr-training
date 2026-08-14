"""Authoritative Stage 7-D2 training on the accepted Synthetic Curriculum v1.

D1 is re-run first as an integrity-only gate. After D1 returns, this module
creates model-development references for TRAIN and VALIDATION only. No TEST
artifact path or TEST artifact byte is exposed to the training/evaluation code.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Final

import torch

from .dataset_manifest import DatasetSplit
from .stage7c_execution import verify_authoritative_repository, verify_stage7c_runtime
from .stage7d2_profile import (
    EXPECTED_D1_ARTIFACT_BINDING_SHA256,
    EXPECTED_D1_IMAGE_BYTES_TOTAL,
    EXPECTED_D1_TARGET_BYTES_TOTAL,
    STAGE7D2_EVIDENCE_SCHEMA,
    STAGE7D2_FROZEN_MODEL_CONFIG,
    STAGE7D2_FROZEN_PREPROCESS_CONFIG,
    STAGE7D2_FROZEN_RUN_CONFIG,
    STAGE7D2_FROZEN_RUN_FINGERPRINT,
    STAGE7D2_FROZEN_TRAINER_CONFIG,
    STAGE7D2_RUN_VERSION,
    STAGE7D2_VERIFICATION_SCHEMA,
)
from .synthetic_curriculum_acceptance import (
    EXPECTED_BUILD_ID,
    EXPECTED_CONFIG_FINGERPRINT,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SOURCE_COMMIT,
    EXPECTED_TRANSPORT_SHA256,
)
from .synthetic_curriculum_corpus_gate import (
    EXPECTED_ARCHIVE_NAME,
    EXPECTED_ARCHIVE_SIZE_BYTES,
    SyntheticCurriculumCorpusReceipt,
    verify_stage7d_corpus,
)
from .training_data import (
    TrainingSampleRef,
    make_training_batch,
    preprocess_config_fingerprint,
    preprocess_grayscale_png,
)
from .training_model import (
    assert_model_finite,
    build_baseline_model,
    count_trainable_parameters,
    model_config_fingerprint,
    model_state_sha256,
    train_one_smoke_step,
    trainer_config_fingerprint,
)
from .training_run import (
    BaselineRunResult,
    ProgressCallback,
    _batch_groups,
    _dependency_versions,
    _evaluate_predictions,
    _mean_validation_loss,
    _report_progress,
    _sha256_file,
    _write_checkpoint,
)
from .training_tokens import PAD_TOKEN_ID, TokenizationError, tokenize_musicxml, tokenizer_fingerprint


MAX_MANIFEST_BYTES: Final[int] = 32 * 1024 * 1024
MAX_BUILD_BYTES: Final[int] = 64 * 1024
_HEX = frozenset("0123456789abcdef")


class Stage7D2ExecutionError(RuntimeError):
    """Raised when Stage 7-D2 cannot prove its frozen execution contract."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_sha(name: str, value: object, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in _HEX for character in value)
    ):
        raise Stage7D2ExecutionError(f"{name} is not canonical lowercase SHA hex")
    return value


def _read_bounded(path: Path, maximum: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise Stage7D2ExecutionError(f"{label} must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= maximum:
        raise Stage7D2ExecutionError(f"{label} size is outside the D2 bound")
    return path.read_bytes()


def _load_canonical_json(path: Path, maximum: int, label: str) -> tuple[dict[str, object], bytes]:
    raw = _read_bounded(path, maximum, label)
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D2ExecutionError(f"{label} is not valid ASCII JSON") from exc
    if not isinstance(payload, dict) or _canonical_json_bytes(payload) != raw:
        raise Stage7D2ExecutionError(f"{label} is not a canonical JSON object")
    return payload, raw


def _verify_d1_receipt(receipt: SyntheticCurriculumCorpusReceipt) -> None:
    expected = {
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "build_id": EXPECTED_BUILD_ID,
        "config_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "transport_sha256": EXPECTED_TRANSPORT_SHA256,
        "transport_archive": EXPECTED_ARCHIVE_NAME,
        "archive_size_bytes": EXPECTED_ARCHIVE_SIZE_BYTES,
        "sample_count": 1536,
        "target_count": 512,
        "image_count": 1536,
        "target_bytes_total": EXPECTED_D1_TARGET_BYTES_TOTAL,
        "image_bytes_total": EXPECTED_D1_IMAGE_BYTES_TOTAL,
        "artifact_binding_sha256": EXPECTED_D1_ARTIFACT_BINDING_SHA256,
    }
    for name, value in expected.items():
        if getattr(receipt, name) != value:
            raise Stage7D2ExecutionError(f"D1 receipt {name} differs from the accepted corpus")
    if receipt.sample_split_counts != {"test": 153, "train": 1230, "validation": 153}:
        raise Stage7D2ExecutionError("D1 receipt sample split counts mismatch")
    if receipt.family_split_counts != {"test": 51, "train": 410, "validation": 51}:
        raise Stage7D2ExecutionError("D1 receipt family split counts mismatch")


def _load_development_refs(
    corpus_root: Path,
) -> tuple[tuple[TrainingSampleRef, ...], tuple[TrainingSampleRef, ...]]:
    """Create references only for train/validation after D1 has accepted storage bytes."""

    manifest, raw_manifest = _load_canonical_json(
        corpus_root / "manifest.json", MAX_MANIFEST_BYTES, "manifest.json"
    )
    if sha256(raw_manifest).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise Stage7D2ExecutionError("manifest changed after D1 acceptance")

    build, _raw_build = _load_canonical_json(
        corpus_root / "build.json", MAX_BUILD_BYTES, "build.json"
    )
    build_expected = {
        "build_id": EXPECTED_BUILD_ID,
        "config_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "sample_count": 1536,
        "target_count": 512,
        "image_count": 1536,
    }
    for key, value in build_expected.items():
        if build.get(key) != value:
            raise Stage7D2ExecutionError(f"build.json {key} changed after D1 acceptance")

    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != 1536:
        raise Stage7D2ExecutionError("manifest sample list changed after D1 acceptance")

    target_cache: dict[str, tuple[int, ...]] = {}
    refs: dict[DatasetSplit, list[TrainingSampleRef]] = {
        DatasetSplit.TRAIN: [],
        DatasetSplit.VALIDATION: [],
    }
    families: dict[DatasetSplit, set[str]] = {
        DatasetSplit.TRAIN: set(),
        DatasetSplit.VALIDATION: set(),
    }

    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise Stage7D2ExecutionError(f"manifest sample[{index}] is not an object")
        split_text = sample.get("split")

        # Critical boundary: TEST is skipped before any artifact path is derived or read.
        if split_text == DatasetSplit.TEST.value:
            continue
        try:
            split = DatasetSplit(split_text)
        except (TypeError, ValueError) as exc:
            raise Stage7D2ExecutionError(f"manifest sample[{index}] has invalid split") from exc
        if split not in refs:
            raise Stage7D2ExecutionError("only train/validation may cross the D2 loader boundary")

        sample_id = _require_sha("sample_id", sample.get("sample_id"), 64)
        image_sha = _require_sha("png_sha256", sample.get("png_sha256"), 64)
        target_sha = _require_sha(
            "source_musicxml_sha256", sample.get("source_musicxml_sha256"), 64
        )
        family_id = sample.get("family_id")
        width = sample.get("width")
        height = sample.get("height")
        if not isinstance(family_id, str) or not family_id:
            raise Stage7D2ExecutionError("development family_id is invalid")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise Stage7D2ExecutionError("development image width is invalid")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise Stage7D2ExecutionError("development image height is invalid")

        target_path = corpus_root / "targets" / f"{target_sha}.musicxml"
        image_path = corpus_root / "images" / f"{image_sha}.png"
        if target_path.is_symlink() or not target_path.is_file():
            raise Stage7D2ExecutionError("development target artifact is missing or symlinked")
        if image_path.is_symlink() or not image_path.is_file():
            raise Stage7D2ExecutionError("development image artifact is missing or symlinked")

        token_ids = target_cache.get(target_sha)
        if token_ids is None:
            target_bytes = target_path.read_bytes()
            if sha256(target_bytes).hexdigest() != target_sha:
                raise Stage7D2ExecutionError("development MusicXML changed after D1 acceptance")
            try:
                token_ids = tokenize_musicxml(target_bytes).token_ids
            except TokenizationError as exc:
                raise Stage7D2ExecutionError("development MusicXML failed tokenization") from exc
            if len(token_ids) > STAGE7D2_FROZEN_RUN_CONFIG.max_decode_tokens:
                raise Stage7D2ExecutionError("development target exceeds the frozen decode ceiling")
            target_cache[target_sha] = token_ids

        image_bytes = image_path.read_bytes()
        if sha256(image_bytes).hexdigest() != image_sha:
            raise Stage7D2ExecutionError("development PNG changed after D1 acceptance")
        preprocess_grayscale_png(
            image_bytes,
            STAGE7D2_FROZEN_PREPROCESS_CONFIG,
            expected_width=width,
            expected_height=height,
        )

        refs[split].append(
            TrainingSampleRef(
                sample_id=sample_id,
                family_id=family_id,
                split=split,
                image_path=image_path,
                image_sha256=image_sha,
                target_path=target_path,
                target_sha256=target_sha,
                target_token_ids=token_ids,
                source_width=width,
                source_height=height,
            )
        )
        families[split].add(family_id)

    train = tuple(sorted(refs[DatasetSplit.TRAIN], key=lambda item: item.sample_id))
    validation = tuple(sorted(refs[DatasetSplit.VALIDATION], key=lambda item: item.sample_id))
    config = STAGE7D2_FROZEN_RUN_CONFIG
    if len(train) != config.train_samples or len(validation) != config.validation_samples:
        raise Stage7D2ExecutionError("D2 development sample counts are not exact")
    if len(families[DatasetSplit.TRAIN]) != config.train_families:
        raise Stage7D2ExecutionError("D2 train family count is not exact")
    if len(families[DatasetSplit.VALIDATION]) != config.validation_families:
        raise Stage7D2ExecutionError("D2 validation family count is not exact")
    if any(sample.split is DatasetSplit.TEST for sample in train + validation):
        raise Stage7D2ExecutionError("sealed TEST sample crossed the D2 boundary")
    return train, validation


@dataclass(frozen=True, slots=True)
class VerifiedStage7D2RunResult:
    result: BaselineRunResult
    verification_path: Path
    verification_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.result, BaselineRunResult):
            raise Stage7D2ExecutionError("result must be BaselineRunResult")
        if not isinstance(self.verification_path, Path) or not self.verification_path.is_file():
            raise Stage7D2ExecutionError("verification_path must reference a file")
        _require_sha("verification_sha256", self.verification_sha256, 64)
        if sha256(self.verification_path.read_bytes()).hexdigest() != self.verification_sha256:
            raise Stage7D2ExecutionError("verification file hash mismatch")


def _run_training(
    train_samples: tuple[TrainingSampleRef, ...],
    validation_samples: tuple[TrainingSampleRef, ...],
    run_root: Path,
    *,
    repository_sha: str,
    d1_receipt: SyntheticCurriculumCorpusReceipt,
    progress: ProgressCallback | None,
) -> BaselineRunResult:
    config = STAGE7D2_FROZEN_RUN_CONFIG
    model_config = STAGE7D2_FROZEN_MODEL_CONFIG
    trainer_config = STAGE7D2_FROZEN_TRAINER_CONFIG
    preprocess_config = STAGE7D2_FROZEN_PREPROCESS_CONFIG

    identity = {
        "run_version": STAGE7D2_RUN_VERSION,
        "repository_sha": repository_sha,
        "dataset_build_id": EXPECTED_BUILD_ID,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "artifact_binding_sha256": d1_receipt.artifact_binding_sha256,
        "run_fingerprint": STAGE7D2_FROZEN_RUN_FINGERPRINT,
    }
    run_id = sha256(_canonical_json_bytes(identity)).hexdigest()
    run_root.mkdir(parents=True, exist_ok=True)
    run_directory = run_root / run_id
    if run_directory.exists():
        raise Stage7D2ExecutionError("D2 run directory already exists; silent resume is forbidden")
    run_directory.mkdir()
    incomplete = run_directory / "INCOMPLETE"
    incomplete.write_bytes(_canonical_json_bytes(identity))

    train_groups = _batch_groups(train_samples, config.batch_size)
    training_steps_total = len(train_groups) * config.epochs
    _report_progress(
        progress,
        "training_started",
        stage="7-D2",
        epochs_total=config.epochs,
        train_samples_total=len(train_samples),
        validation_samples_total=len(validation_samples),
        training_steps_total=training_steps_total,
    )

    model = build_baseline_model(model_config, seed=trainer_config.master_seed)
    parameter_count = count_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer_config.learning_rate_micros / 1_000_000,
        weight_decay=trainer_config.weight_decay_micros / 1_000_000,
        foreach=False,
        fused=False,
    )

    untrained_validation_loss = _mean_validation_loss(
        model,
        validation_samples,
        batch_size=config.batch_size,
        preprocess_config=preprocess_config,
        progress=progress,
        phase="untrained",
        epochs_total=config.epochs,
    )
    best_validation_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_checkpoint: Path | None = None
    best_checkpoint_sha = ""
    training_steps = 0
    epoch_records: list[dict[str, object]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss_sum = 0.0
        train_step_count = 0
        _report_progress(progress, "epoch_started", epoch=epoch, epochs_total=config.epochs)
        for epoch_step, group in enumerate(train_groups, start=1):
            batch = make_training_batch(group, preprocess_config)
            value = train_one_smoke_step(model, batch, optimizer, trainer_config)
            train_loss_sum += value
            train_step_count += 1
            training_steps += 1
            if epoch_step % 16 == 0 or epoch_step == len(train_groups):
                _report_progress(
                    progress,
                    "training_step_completed",
                    epoch=epoch,
                    epochs_total=config.epochs,
                    epoch_steps_completed=epoch_step,
                    epoch_steps_total=len(train_groups),
                    training_steps=training_steps,
                    training_steps_total=training_steps_total,
                )
        if train_step_count <= 0:
            raise Stage7D2ExecutionError("D2 epoch executed no optimizer steps")
        mean_train_loss = train_loss_sum / train_step_count
        current_validation_loss = _mean_validation_loss(
            model,
            validation_samples,
            batch_size=config.batch_size,
            preprocess_config=preprocess_config,
            progress=progress,
            phase="epoch",
            epoch=epoch,
            epochs_total=config.epochs,
        )
        selected_best = current_validation_loss < best_validation_loss
        if selected_best:
            previous = best_checkpoint
            best_validation_loss = current_validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            best_checkpoint, best_checkpoint_sha = _write_checkpoint(
                model,
                run_directory,
                epoch=epoch,
                model_fingerprint=model_config_fingerprint(model_config),
            )
            if previous is not None and previous != best_checkpoint:
                previous.unlink()
        epoch_records.append(
            {
                "epoch": epoch,
                "mean_train_loss": mean_train_loss,
                "validation_loss": current_validation_loss,
                "selected_best": selected_best,
            }
        )
        _report_progress(
            progress,
            "epoch_completed",
            epoch=epoch,
            epochs_total=config.epochs,
            mean_train_loss=float(mean_train_loss),
            validation_loss=float(current_validation_loss),
            selected_best=selected_best,
            training_steps=training_steps,
        )

    if best_state is None or best_checkpoint is None:
        raise Stage7D2ExecutionError("D2 did not select a checkpoint")
    if not best_validation_loss < untrained_validation_loss:
        raise Stage7D2ExecutionError("D2 validation loss did not improve on the untrained model")
    model.load_state_dict(best_state, strict=True)
    assert_model_finite(model)
    if _sha256_file(best_checkpoint) != best_checkpoint_sha:
        raise Stage7D2ExecutionError("selected D2 checkpoint changed after hashing")

    prediction_metrics = _evaluate_predictions(
        model,
        validation_samples,
        preprocess_config=preprocess_config,
        max_decode_tokens=config.max_decode_tokens,
        measure_count=config.decode_measure_count,
        progress=progress,
    )
    if prediction_metrics.valid_semantic_predictions < 1:
        raise Stage7D2ExecutionError("D2 produced no semantically valid validation prediction")

    evidence = {
        "schema_version": STAGE7D2_EVIDENCE_SCHEMA,
        "run_version": STAGE7D2_RUN_VERSION,
        "run_id": run_id,
        "repository_sha": repository_sha,
        "dataset": {
            "source_commit": EXPECTED_SOURCE_COMMIT,
            "build_id": EXPECTED_BUILD_ID,
            "config_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "transport_sha256": EXPECTED_TRANSPORT_SHA256,
            "transport_archive": EXPECTED_ARCHIVE_NAME,
            "artifact_binding_sha256": d1_receipt.artifact_binding_sha256,
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "test_samples_exposed_to_development": 0,
        },
        "fingerprints": {
            "run": STAGE7D2_FROZEN_RUN_FINGERPRINT,
            "tokenizer": tokenizer_fingerprint(),
            "preprocess": preprocess_config_fingerprint(preprocess_config),
            "model": model_config_fingerprint(model_config),
            "trainer": trainer_config_fingerprint(trainer_config),
        },
        "configuration": {
            "run": asdict(config),
            "model": asdict(model_config),
            "trainer": asdict(trainer_config),
            "preprocess": asdict(preprocess_config),
        },
        "runtime": {
            "device": "cpu",
            "dependencies": _dependency_versions(),
        },
        "training": {
            "parameter_count": parameter_count,
            "data_ordering": "sample_id-ascending-fixed-each-epoch",
            "epochs_completed": config.epochs,
            "training_steps": training_steps,
            "untrained_validation_loss": untrained_validation_loss,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "epoch_records": epoch_records,
        },
        "prediction_metrics": asdict(prediction_metrics),
        "checkpoint": {
            "sha256": best_checkpoint_sha,
            "state_sha256": model_state_sha256(model),
            "filename": best_checkpoint.name,
        },
        "sealed_test_split_opened_for_model_development": False,
    }
    metrics_bytes = _canonical_json_bytes(evidence)
    metrics_sha = sha256(metrics_bytes).hexdigest()
    metrics_path = run_directory / f"metrics-{metrics_sha}.json"
    metrics_path.write_bytes(metrics_bytes)
    if _sha256_file(metrics_path) != metrics_sha:
        raise Stage7D2ExecutionError("D2 metrics changed after hashing")
    (run_directory / "COMPLETE").write_text(
        f"{metrics_sha}  {metrics_path.name}\n", encoding="ascii"
    )
    incomplete.unlink()

    return BaselineRunResult(
        run_id=run_id,
        run_directory=run_directory,
        repository_sha=repository_sha,
        dataset_build_id=EXPECTED_BUILD_ID,
        manifest_sha256=EXPECTED_MANIFEST_SHA256,
        untrained_validation_loss=float(untrained_validation_loss),
        best_validation_loss=float(best_validation_loss),
        best_epoch=best_epoch,
        training_steps=training_steps,
        checkpoint_sha256=best_checkpoint_sha,
        metrics_sha256=metrics_sha,
        prediction_metrics=prediction_metrics,
    )


def _verify_completed_run(
    result: BaselineRunResult,
    *,
    repository_sha: str,
    repository_origin: str,
    runtime_versions: dict[str, str],
    d1_receipt: SyntheticCurriculumCorpusReceipt,
) -> VerifiedStage7D2RunResult:
    metrics_path = result.run_directory / f"metrics-{result.metrics_sha256}.json"
    checkpoint_path = result.run_directory / f"checkpoint-{result.checkpoint_sha256}.pt"
    complete_path = result.run_directory / "COMPLETE"
    if not metrics_path.is_file() or not checkpoint_path.is_file() or not complete_path.is_file():
        raise Stage7D2ExecutionError("completed D2 run is missing evidence artifacts")
    metrics_bytes = metrics_path.read_bytes()
    if sha256(metrics_bytes).hexdigest() != result.metrics_sha256:
        raise Stage7D2ExecutionError("D2 metrics artifact hash mismatch")
    if sha256(checkpoint_path.read_bytes()).hexdigest() != result.checkpoint_sha256:
        raise Stage7D2ExecutionError("D2 checkpoint artifact hash mismatch")
    expected_complete = f"{result.metrics_sha256}  {metrics_path.name}\n".encode("ascii")
    if complete_path.read_bytes() != expected_complete:
        raise Stage7D2ExecutionError("D2 COMPLETE marker mismatch")

    try:
        evidence = json.loads(metrics_bytes.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D2ExecutionError("D2 metrics is not valid ASCII JSON") from exc
    if not isinstance(evidence, dict) or _canonical_json_bytes(evidence) != metrics_bytes:
        raise Stage7D2ExecutionError("D2 metrics is not canonical JSON")
    if evidence.get("schema_version") != STAGE7D2_EVIDENCE_SCHEMA:
        raise Stage7D2ExecutionError("D2 metrics schema mismatch")
    if evidence.get("repository_sha") != repository_sha:
        raise Stage7D2ExecutionError("D2 repository provenance mismatch")
    if evidence.get("sealed_test_split_opened_for_model_development") is not False:
        raise Stage7D2ExecutionError("D2 evidence does not prove TEST stayed sealed")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Stage7D2ExecutionError("D2 checkpoint cannot be safely reloaded") from exc
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "epoch", "model_fingerprint", "model_state_dict"
    }:
        raise Stage7D2ExecutionError("D2 checkpoint structure mismatch")
    if checkpoint.get("epoch") != result.best_epoch:
        raise Stage7D2ExecutionError("D2 checkpoint epoch mismatch")
    if checkpoint.get("model_fingerprint") != model_config_fingerprint(
        STAGE7D2_FROZEN_MODEL_CONFIG
    ):
        raise Stage7D2ExecutionError("D2 checkpoint model fingerprint mismatch")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise Stage7D2ExecutionError("D2 checkpoint model state is invalid")
    model = build_baseline_model(STAGE7D2_FROZEN_MODEL_CONFIG, seed=0)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise Stage7D2ExecutionError("D2 checkpoint state cannot be loaded strictly") from exc
    assert_model_finite(model)
    state_sha = model_state_sha256(model)
    checkpoint_evidence = evidence.get("checkpoint")
    if not isinstance(checkpoint_evidence, dict):
        raise Stage7D2ExecutionError("D2 checkpoint evidence missing")
    if checkpoint_evidence.get("state_sha256") != state_sha:
        raise Stage7D2ExecutionError("D2 checkpoint state hash mismatch")

    verification = {
        "schema_version": STAGE7D2_VERIFICATION_SCHEMA,
        "repository_origin": repository_origin,
        "repository_sha": repository_sha,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "dataset_build_id": EXPECTED_BUILD_ID,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "artifact_binding_sha256": d1_receipt.artifact_binding_sha256,
        "run_profile_fingerprint": STAGE7D2_FROZEN_RUN_FINGERPRINT,
        "run_id": result.run_id,
        "metrics_sha256": result.metrics_sha256,
        "checkpoint_sha256": result.checkpoint_sha256,
        "checkpoint_state_sha256": state_sha,
        "runtime": runtime_versions,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "prediction_metrics": asdict(result.prediction_metrics),
        "train_samples": STAGE7D2_FROZEN_RUN_CONFIG.train_samples,
        "validation_samples": STAGE7D2_FROZEN_RUN_CONFIG.validation_samples,
        "test_samples_exposed_to_model_development": 0,
        "d1_reverified_before_training": True,
        "source_tree_clean_before_and_after": True,
    }
    verification_bytes = _canonical_json_bytes(verification)
    verification_sha = sha256(verification_bytes).hexdigest()
    verification_path = result.run_directory / f"verification-{verification_sha}.json"
    verification_path.write_bytes(verification_bytes)
    return VerifiedStage7D2RunResult(result, verification_path, verification_sha)


def run_verified_stage7d2_training(
    corpus_root: str | Path,
    transport_archive: str | Path,
    run_root: str | Path,
    repository_root: str | Path,
    *,
    progress: ProgressCallback | None = None,
) -> VerifiedStage7D2RunResult:
    """Execute the frozen D2 train/validation run after re-verifying D1 bytes."""

    if not isinstance(corpus_root, (str, Path)):
        raise TypeError("corpus_root must be str or pathlib.Path")
    if not isinstance(transport_archive, (str, Path)):
        raise TypeError("transport_archive must be str or pathlib.Path")
    if not isinstance(run_root, (str, Path)):
        raise TypeError("run_root must be str or pathlib.Path")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")

    repository_sha, repository_origin = verify_authoritative_repository(repository_root)
    runtime_versions = verify_stage7c_runtime()

    _report_progress(progress, "d1_reverification_started")
    receipt = verify_stage7d_corpus(corpus_root, transport_archive)
    _verify_d1_receipt(receipt)
    _report_progress(
        progress,
        "d1_reverification_completed",
        artifact_binding_sha256=receipt.artifact_binding_sha256,
    )

    train_samples, validation_samples = _load_development_refs(Path(corpus_root))
    result = _run_training(
        train_samples,
        validation_samples,
        Path(run_root),
        repository_sha=repository_sha,
        d1_receipt=receipt,
        progress=progress,
    )

    ending_sha, ending_origin = verify_authoritative_repository(repository_root)
    ending_runtime = verify_stage7c_runtime()
    if ending_sha != repository_sha or ending_origin != repository_origin:
        raise Stage7D2ExecutionError("repository identity changed during D2 execution")
    if ending_runtime != runtime_versions:
        raise Stage7D2ExecutionError("runtime identity changed during D2 execution")

    verified = _verify_completed_run(
        result,
        repository_sha=repository_sha,
        repository_origin=repository_origin,
        runtime_versions=runtime_versions,
        d1_receipt=receipt,
    )
    _report_progress(
        progress,
        "stage7d2_completed",
        run_id=result.run_id,
        metrics_sha256=result.metrics_sha256,
        checkpoint_sha256=result.checkpoint_sha256,
        verification_sha256=verified.verification_sha256,
    )
    return verified


def _stderr_progress(event: dict[str, object]) -> None:
    sys.stderr.write(_canonical_json_bytes(event).decode("ascii") + "\n")
    sys.stderr.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen Stage 7-D2 synthetic training")
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repository-root", default=".")
    args = parser.parse_args(argv)

    verified = run_verified_stage7d2_training(
        args.corpus_root,
        args.archive,
        args.run_root,
        args.repository_root,
        progress=_stderr_progress,
    )
    result = verified.result
    summary = {
        "run_id": result.run_id,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "checkpoint_sha256": result.checkpoint_sha256,
        "metrics_sha256": result.metrics_sha256,
        "verification_sha256": verified.verification_sha256,
        "prediction_metrics": asdict(result.prediction_metrics),
    }
    sys.stdout.write(_canonical_json_bytes(summary).decode("ascii") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
