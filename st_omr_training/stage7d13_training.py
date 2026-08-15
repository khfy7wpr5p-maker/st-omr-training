"""Deterministic Stage 7-D13 specialist training runner.

The runner is bound to the frozen authoritative D13 derivative surface and trains
NoteHeadSet, RestSet and AccidentalSet sequentially with independent model and
optimizer state.  It never touches TEST and never writes COMPLETE; persisted-run
verification is a separate D13 gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Final, Mapping, Sequence

from PIL import Image
import torch

from .stage7c_execution import verify_authoritative_repository, verify_stage7c_runtime
from .stage7d13_measure_derivatives import (
    STAGE7D13_LABEL_SCHEMA,
    STAGE7D13_DERIVATIVE_VERSION,
)
from .stage7d13_symbol_models import (
    SpecialistMetrics,
    acceptance_passed,
    build_symbol_model,
    compute_specialist_metrics,
    decode_detections,
    detector_loss,
    encode_detector_targets,
    ground_truth_rows,
    model_profile_fingerprint,
)
from .stage7d13_symbol_training_contract import (
    FROZEN_D13_CONFIG,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    SPECIALIST_CLASSES,
    stage7d13_contract_fingerprint,
)
from .stage7d13_verified_surface import (
    D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
    D13_DERIVATIVE_BUILD_ID,
    D13_DERIVATIVE_MANIFEST_SHA256,
    D13_EXPECTED_OPTIMIZER_STEPS,
    D13_EXPECTED_OPTIMIZER_STEPS_TOTAL,
    D13_RECORD_COUNT,
    D13_RECORD_SPLIT_COUNTS,
    D13_TEST_SPECIALIST_RECORDS,
)
from .training_model import (
    assert_finite_tensor,
    assert_model_finite,
    assert_optimizer_finite,
    count_trainable_parameters,
    model_state_sha256,
    set_deterministic_cpu,
)


STAGE7D13_TRAINING_VERSION: Final[str] = "stage7d13-symbol-authoritative-training-v1"
_MAX_JSON_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_IMAGE_BYTES: Final[int] = 32 * 1024 * 1024
_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_EXPECTED_DERIVATIVE_TOP: Final[frozenset[str]] = frozenset(
    {"manifest.json", "manifest.sha256", "build.json", "images", "labels"}
)


class Stage7D13TrainingError(RuntimeError):
    """Raised when D13 training input, numeric state or persistence fails closed."""


def _fail(message: str) -> None:
    raise Stage7D13TrainingError(message)


def _canonical_json(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Stage7D13TrainingError("D13 training payload is not canonical JSON") from exc


def _sha64(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        _fail(f"{name} must be lowercase SHA-256")
    return value


def _read_file(path: Path, maximum: int, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        _fail(f"{name} must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= maximum:
        _fail(f"{name} byte length is outside bound")
    return path.read_bytes()


def _read_json(path: Path, maximum: int, name: str) -> tuple[dict[str, object], bytes]:
    raw = _read_file(path, maximum, name)
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D13TrainingError(f"{name} is not valid ASCII JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != raw:
        _fail(f"{name} must be canonical JSON object bytes")
    return value, raw


def _prepare_output_root(path: Path, repository_root: Path) -> None:
    resolved = path.resolve()
    repo = repository_root.resolve()
    if resolved == repo or repo in resolved.parents:
        _fail("D13 training output must remain outside repository")
    if path.exists() or path.is_symlink():
        _fail("D13 training output root must be fresh")
    path.mkdir(parents=True)


def _repo_identity(repository_root: Path) -> tuple[str, str]:
    identity = verify_authoritative_repository(repository_root)
    if not isinstance(identity, tuple) or len(identity) != 2:
        _fail("repository verifier returned unexpected identity")
    head, origin = identity
    if not isinstance(head, str) or len(head) != 40 or any(c not in _HEX for c in head):
        _fail("repository HEAD must be canonical lowercase SHA")
    if not isinstance(origin, str) or not origin:
        _fail("repository origin is missing")
    return head, origin


@dataclass(frozen=True, slots=True)
class D13Record:
    record_id: str
    split: str
    image_sha256: str
    label_sha256: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True, slots=True)
class SpecialistTrainingResult:
    specialist: str
    parameter_count: int
    model_fingerprint: str
    final_state_sha256: str
    optimizer_steps: int
    best_epoch: int
    initial_validation_loss: float
    final_validation_loss: float
    metrics: SpecialistMetrics
    accepted: bool
    history: tuple[dict[str, float | int], ...]


@dataclass(frozen=True, slots=True)
class Stage7D13TrainingReceipt:
    version: str
    run_id: str
    repository_sha: str
    repository_origin: str
    training_profile_fingerprint: str
    derivative_build_id: str
    derivative_manifest_sha256: str
    derivative_artifact_binding_sha256: str
    train_records: int
    validation_records: int
    test_opened: bool
    optimizer_steps: dict[str, int]
    optimizer_steps_total: int
    checkpoint_sha256: str
    metrics_sha256: str
    run_sha256: str
    specialists: dict[str, dict[str, object]]
    acceptance: bool
    complete_marker_written: bool


def _load_records(root: Path) -> tuple[D13Record, ...]:
    if root.is_symlink() or not root.is_dir():
        _fail("D13 derivative root must be regular directory")
    if {p.name for p in root.iterdir()} != _EXPECTED_DERIVATIVE_TOP:
        _fail("D13 derivative top-level surface differs from frozen bundle")
    manifest, manifest_raw = _read_json(root / "manifest.json", _MAX_JSON_BYTES, "D13 manifest")
    if sha256(manifest_raw).hexdigest() != D13_DERIVATIVE_MANIFEST_SHA256:
        _fail("D13 derivative manifest SHA mismatch")
    if manifest.get("derivative_build_id") != D13_DERIVATIVE_BUILD_ID:
        _fail("D13 derivative build id mismatch")
    build, _build_raw = _read_json(root / "build.json", 4 * 1024 * 1024, "D13 build")
    expected_build = {
        "derivative_build_id": D13_DERIVATIVE_BUILD_ID,
        "manifest_sha256": D13_DERIVATIVE_MANIFEST_SHA256,
        "artifact_binding_sha256": D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
        "record_count": D13_RECORD_COUNT,
        "record_split_counts": D13_RECORD_SPLIT_COUNTS,
        "test_specialist_records": D13_TEST_SPECIALIST_RECORDS,
        "optimizer_steps": 0,
        "complete_marker_written": False,
    }
    for name, expected in expected_build.items():
        if build.get(name) != expected:
            _fail(f"D13 derivative build {name} mismatch")
    rows = manifest.get("records")
    if not isinstance(rows, list) or len(rows) != D13_RECORD_COUNT:
        _fail("D13 derivative manifest record count mismatch")
    records: list[D13Record] = []
    counts = {"train": 0, "validation": 0}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            _fail("D13 derivative record must be object")
        split = row.get("split")
        if split not in counts:
            _fail("D13 training encountered forbidden split")
        record_id = _sha64(row.get("record_id"), "record_id")
        image_sha = _sha64(row.get("image_sha256"), "image_sha256")
        label_sha = _sha64(row.get("label_sha256"), "label_sha256")
        if record_id in seen:
            _fail("duplicate D13 training record id")
        seen.add(record_id)
        counts[str(split)] += 1
        records.append(
            D13Record(
                record_id=record_id,
                split=str(split),
                image_sha256=image_sha,
                label_sha256=label_sha,
                image_path=root / "images" / f"{image_sha}.png",
                label_path=root / "labels" / f"{label_sha}.json",
            )
        )
    if counts != D13_RECORD_SPLIT_COUNTS:
        _fail("D13 training split counts mismatch")
    return tuple(sorted(records, key=lambda item: item.record_id))


def _load_example(record: D13Record, specialist: str) -> tuple[torch.Tensor, list[Mapping[str, object]]]:
    image_raw = _read_file(record.image_path, _MAX_IMAGE_BYTES, "D13 measure PNG")
    if sha256(image_raw).hexdigest() != record.image_sha256:
        _fail("D13 measure PNG SHA mismatch")
    try:
        with Image.open(BytesIO(image_raw)) as opened:
            opened.load()
            if opened.format != "PNG" or opened.mode != "L" or opened.size != (INPUT_WIDTH, INPUT_HEIGHT):
                _fail("D13 measure PNG metadata mismatch")
            raw_pixels = bytearray(opened.tobytes())
    except (OSError, ValueError) as exc:
        raise Stage7D13TrainingError("D13 measure PNG cannot be decoded") from exc
    image = torch.frombuffer(raw_pixels, dtype=torch.uint8).clone().reshape(INPUT_HEIGHT, INPUT_WIDTH)
    tensor = 1.0 - image.to(dtype=torch.float32) / 255.0
    label, label_raw = _read_json(record.label_path, 4 * 1024 * 1024, "D13 measure label")
    if sha256(label_raw).hexdigest() != record.label_sha256:
        _fail("D13 measure label SHA mismatch")
    if label.get("schema_version") != STAGE7D13_LABEL_SCHEMA or label.get("stage7d13_derivative_version") != STAGE7D13_DERIVATIVE_VERSION:
        _fail("D13 measure label schema/version mismatch")
    if label.get("record_id") != record.record_id or label.get("split") != record.split:
        _fail("D13 measure label identity mismatch")
    targets = label.get("targets")
    if not isinstance(targets, Mapping):
        _fail("D13 measure targets missing")
    rows = targets.get(specialist)
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        _fail("D13 specialist target list malformed")
    return tensor.unsqueeze(0), list(rows)


def _seed(specialist: str, epoch: int) -> int:
    raw = f"{FROZEN_D13_CONFIG.master_seed}:{specialist}:{epoch}".encode("ascii")
    return int.from_bytes(sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def _batches(records: Sequence[D13Record], specialist: str, epoch: int) -> list[list[D13Record]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed(specialist, epoch))
    order = torch.randperm(len(records), generator=generator).tolist()
    size = FROZEN_D13_CONFIG.batch_size
    return [[records[index] for index in order[start : start + size]] for start in range(0, len(order), size)]


def _stack(batch: Sequence[D13Record], specialist: str) -> tuple[torch.Tensor, list[list[Mapping[str, object]]]]:
    images: list[torch.Tensor] = []
    target_rows: list[list[Mapping[str, object]]] = []
    for record in batch:
        image, rows = _load_example(record, specialist)
        images.append(image)
        target_rows.append(rows)
    return torch.stack(images), target_rows


def _evaluate(model: torch.nn.Module, records: Sequence[D13Record], specialist: str) -> tuple[float, SpecialistMetrics]:
    state_before = model_state_sha256(model)
    model.eval()
    total_loss = 0.0
    total_records = 0
    examples = []
    batch_size = FROZEN_D13_CONFIG.batch_size
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            images, rows = _stack(batch, specialist)
            targets = encode_detector_targets(specialist, rows)
            outputs = model(images)
            loss = detector_loss(specialist, outputs, targets)
            value = float(loss.item())
            if not math.isfinite(value):
                _fail("D13 validation loss is non-finite")
            total_loss += value * len(batch)
            total_records += len(batch)
            decoded = decode_detections(specialist, outputs)
            for predictions, target_rows in zip(decoded, rows, strict=True):
                examples.append((predictions, ground_truth_rows(specialist, target_rows)))
    if state_before != model_state_sha256(model):
        _fail("D13 validation mutated model state")
    return total_loss / total_records, compute_specialist_metrics(specialist, examples)


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def training_profile_fingerprint() -> str:
    payload = {
        "version": STAGE7D13_TRAINING_VERSION,
        "contract": stage7d13_contract_fingerprint(),
        "derivative": {
            "build_id": D13_DERIVATIVE_BUILD_ID,
            "manifest_sha256": D13_DERIVATIVE_MANIFEST_SHA256,
            "artifact_binding_sha256": D13_DERIVATIVE_ARTIFACT_BINDING_SHA256,
        },
        "expected_steps": D13_EXPECTED_OPTIMIZER_STEPS,
        "model_fingerprints": {
            specialist: model_profile_fingerprint(specialist)
            for specialist in SPECIALIST_CLASSES
        },
        "config": asdict(FROZEN_D13_CONFIG),
    }
    return sha256(_canonical_json(payload)).hexdigest()


def _train_specialist(
    specialist: str,
    train_records: Sequence[D13Record],
    validation_records: Sequence[D13Record],
    *,
    heartbeat=print,
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
    initial_loss, _initial_metrics = _evaluate(model, validation_records, specialist)
    best_loss = initial_loss
    best_epoch = 0
    best_state = _clone_state(model)
    history: list[dict[str, float | int]] = []
    steps = 0

    for epoch in range(1, FROZEN_D13_CONFIG.epochs + 1):
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
                    assert_finite_tensor(f"D13 gradient {specialist}.{name}", parameter.grad)
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
            if heartbeat is not None and batch_index % FROZEN_D13_CONFIG.heartbeat_batches == 0:
                heartbeat(
                    f"{specialist.upper()} EPOCH {epoch}/{FROZEN_D13_CONFIG.epochs} "
                    f"BATCH {batch_index}/{len(batches)} STEP {steps}/{D13_EXPECTED_OPTIMIZER_STEPS[specialist]}"
                )
        validation_loss, metrics = _evaluate(model, validation_records, specialist)
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
        if heartbeat is not None:
            heartbeat(
                f"{specialist.upper()} EPOCH {epoch}/{FROZEN_D13_CONFIG.epochs} | "
                f"VAL LOSS={validation_loss:.8f} | CENTER-F1={metrics.class_aware_center_f1_4px:.6f} | "
                f"BBOX-F1={metrics.class_aware_bbox_f1_iou50:.6f} | MACRO-F1={metrics.macro_class_f1:.6f}"
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _clone_state(model)

    if steps != D13_EXPECTED_OPTIMIZER_STEPS[specialist]:
        _fail(f"{specialist} optimizer-step count mismatch")
    model.load_state_dict(best_state, strict=True)
    assert_model_finite(model)
    final_loss, final_metrics = _evaluate(model, validation_records, specialist)
    if not math.isclose(final_loss, best_loss, rel_tol=0.0, abs_tol=1e-12):
        _fail(f"{specialist} restored state does not reproduce best validation loss")
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


def run_stage7d13_training(
    *,
    derivative_root: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    expected_repository_sha: str,
    heartbeat=print,
) -> Stage7D13TrainingReceipt:
    """Train all three D13 specialists on the exact frozen development surface."""
    derivative = Path(derivative_root)
    output = Path(output_root)
    repository = Path(repository_root)
    head_before, origin_before = _repo_identity(repository)
    if head_before != expected_repository_sha:
        _fail("D13 training repository HEAD differs from authorized exact head")
    verify_stage7c_runtime()
    records = _load_records(derivative)
    train_records = tuple(row for row in records if row.split == "train")
    validation_records = tuple(row for row in records if row.split == "validation")
    if len(train_records) != D13_RECORD_SPLIT_COUNTS["train"] or len(validation_records) != D13_RECORD_SPLIT_COUNTS["validation"]:
        _fail("D13 training record cardinality drift")
    _prepare_output_root(output, repository)

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

    results: dict[str, SpecialistTrainingResult] = {}
    checkpoint: dict[str, dict[str, torch.Tensor]] = {}
    for specialist in SPECIALIST_CLASSES:
        result, state = _train_specialist(
            specialist,
            train_records,
            validation_records,
            heartbeat=heartbeat,
        )
        results[specialist] = result
        checkpoint[specialist] = state

    steps = {name: result.optimizer_steps for name, result in results.items()}
    if steps != D13_EXPECTED_OPTIMIZER_STEPS or sum(steps.values()) != D13_EXPECTED_OPTIMIZER_STEPS_TOTAL:
        _fail("D13 total optimizer-step evidence mismatch")

    head_after, origin_after = _repo_identity(repository)
    verify_stage7c_runtime()
    if (head_after, origin_after) != (head_before, origin_before):
        _fail("repository identity changed during D13 training")

    checkpoint_path = output / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = sha256(checkpoint_path.read_bytes()).hexdigest()

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
    metrics_path = output / "metrics.json"
    metrics_path.write_bytes(metrics_raw)
    metrics_sha = sha256(metrics_raw).hexdigest()

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
    run_path = output / "run.json"
    run_path.write_bytes(run_raw)
    run_sha = sha256(run_raw).hexdigest()

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
