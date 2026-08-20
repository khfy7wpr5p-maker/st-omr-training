"""Shadow-only real-domain adaptation for the frozen D11 Meter refiner.

The run starts from the exact audited D11 checkpoint, consumes an explicitly
admitted 72-record teacher-gold bundle, mixes only a deterministic balanced
sample of accepted D10 TRAIN Meter records, and evaluates against both the
held-out real pilot validation families and the unchanged D10 VALIDATION
surface.  TEST remains sealed.  A candidate is emitted only when real-domain
gates pass without materially regressing the synthetic baseline.

This module never mutates the runtime checkpoint constant, connects a Resolver,
or authorizes production promotion.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import random
from typing import Final

from PIL import Image, UnidentifiedImageError

from .meter_teacher_gold_admission_v1 import (
    LABEL_SCHEMA as TEACHER_LABEL_SCHEMA,
    METER_CLASSES,
    METER_TEACHER_GOLD_ADMISSION_V1,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    verify_meter_teacher_gold_bundle_v1,
)
METER_REAL_DOMAIN_ADAPTATION_V1: Final[str] = "meter-real-domain-adaptation-v1"
PRESENCE_D11_SHA256: Final[str] = "cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3"
METRICS_SCHEMA: Final[str] = "st-omr-meter-real-domain-adaptation-metrics-v1"
VERIFICATION_SCHEMA: Final[str] = "st-omr-meter-real-domain-adaptation-verification-v1"
CHECKPOINT_ROLE: Final[str] = "meter-real-domain-shadow-candidate-v1"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_LABEL_BYTES = 256 * 1024
_MAX_IMAGE_BYTES = 2 * 1024 * 1024


class MeterRealDomainAdaptationError(RuntimeError):
    """Raised when adaptation provenance, data, numeric state, or gates fail closed."""


def _fail(message: str) -> None:
    raise MeterRealDomainAdaptationError(message)


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
        raise MeterRealDomainAdaptationError("adaptation payload is not canonical JSON serializable") from exc


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _hex64(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        _fail(f"{name} must be canonical lowercase SHA-256")
    return value


def _read_regular(path: Path, *, maximum: int, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        _fail(f"{name} must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= maximum:
        _fail(f"{name} byte length is outside the adaptation boundary")
    return path.read_bytes()


def _read_canonical_json(path: Path, *, maximum: int, name: str) -> tuple[dict[str, object], bytes]:
    raw = _read_regular(path, maximum=maximum, name=name)
    try:
        payload = json.loads(raw.decode("ascii"), parse_constant=lambda token: _fail(f"non-finite {name}: {token}"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MeterRealDomainAdaptationError(f"{name} is not valid ASCII JSON") from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        _fail(f"{name} must be a canonical JSON object")
    return payload, raw


@dataclass(frozen=True, slots=True)
class MeterRealDomainAdaptationConfigV1:
    batch_size: int = 16
    epochs: int = 8
    learning_rate_micros: int = 100
    weight_decay_micros: int = 100
    grad_clip_milli: int = 1000
    master_seed: int = 811_021
    real_repeat_factor: int = 4
    synthetic_replay_per_class: int = 64
    trainable_surface: str = "projection-classifier-bbox-encoder-frozen"
    optimizer: str = "adamw"
    objective: str = "balanced-ce-plus-positive-smooth-l1-v1"
    checkpoint_selection: str = "real-gates-then-real-macro-f1-then-real-loss"
    real_min_macro_f1_milli: int = 800
    real_min_accuracy_milli: int = 833
    real_min_none_recall_milli: int = 888
    # 2/3 on the deliberately tiny three-example-per-positive-class pilot.
    real_min_positive_class_recall_milli: int = 666
    real_min_macro_gain_milli: int = 200
    synthetic_max_macro_f1_drop_milli: int = 20
    synthetic_max_localization_drop_milli: int = 30

    def __post_init__(self) -> None:
        bounds = {
            "batch_size": (self.batch_size, 1, 64),
            "epochs": (self.epochs, 1, 32),
            "learning_rate_micros": (self.learning_rate_micros, 1, 10_000),
            "weight_decay_micros": (self.weight_decay_micros, 0, 100_000),
            "grad_clip_milli": (self.grad_clip_milli, 1, 100_000),
            "master_seed": (self.master_seed, 0, 2**63 - 1),
            "real_repeat_factor": (self.real_repeat_factor, 1, 32),
            "synthetic_replay_per_class": (self.synthetic_replay_per_class, 1, 1024),
        }
        for name, (value, low, high) in bounds.items():
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ValueError(f"{name} is outside Meter adaptation v1 bounds")
        for name in (
            "real_min_macro_f1_milli",
            "real_min_accuracy_milli",
            "real_min_none_recall_milli",
            "real_min_positive_class_recall_milli",
            "real_min_macro_gain_milli",
            "synthetic_max_macro_f1_drop_milli",
            "synthetic_max_localization_drop_milli",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 1000:
                raise ValueError(f"{name} must be integer milli-units in [0,1000]")
        frozen = {
            "trainable_surface": "projection-classifier-bbox-encoder-frozen",
            "optimizer": "adamw",
            "objective": "balanced-ce-plus-positive-smooth-l1-v1",
            "checkpoint_selection": "real-gates-then-real-macro-f1-then-real-loss",
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} is frozen to {expected!r}")


FROZEN_ADAPTATION_CONFIG_V1: Final[MeterRealDomainAdaptationConfigV1] = MeterRealDomainAdaptationConfigV1()


@dataclass(frozen=True, slots=True)
class MeterEvaluationV1:
    loss: float
    macro_f1: float
    accuracy: float
    positive_localization_f1_2px: float
    class_counts: dict[str, int]
    per_class_recall: dict[str, float]
    confusion: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self) -> None:
        for name in ("loss", "macro_f1", "accuracy", "positive_localization_f1_2px"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0 <= self.macro_f1 <= 1 or not 0 <= self.accuracy <= 1 or not 0 <= self.positive_localization_f1_2px <= 1:
            raise ValueError("Meter evaluation rates must be in [0,1]")
        if set(self.class_counts) != set(METER_CLASSES) or set(self.per_class_recall) != set(METER_CLASSES):
            raise ValueError("Meter evaluation must cover all four classes")
        if len(self.confusion) != 4 or any(len(row) != 4 for row in self.confusion):
            raise ValueError("Meter confusion matrix must be 4x4")


@dataclass(frozen=True, slots=True)
class AdaptationGateDecisionV1:
    accepted: bool
    reasons: tuple[str, ...]


def adaptation_acceptance_v1(
    *,
    baseline_real: MeterEvaluationV1,
    candidate_real: MeterEvaluationV1,
    baseline_synthetic: MeterEvaluationV1,
    candidate_synthetic: MeterEvaluationV1,
    config: MeterRealDomainAdaptationConfigV1 = FROZEN_ADAPTATION_CONFIG_V1,
) -> AdaptationGateDecisionV1:
    """Apply fixed real-improvement and synthetic-regression gates."""
    reasons: list[str] = []
    threshold = lambda milli: milli / 1000.0
    if candidate_real.macro_f1 < threshold(config.real_min_macro_f1_milli):
        reasons.append("REAL_MACRO_F1_BELOW_MINIMUM")
    if candidate_real.accuracy < threshold(config.real_min_accuracy_milli):
        reasons.append("REAL_ACCURACY_BELOW_MINIMUM")
    if candidate_real.per_class_recall["none"] < threshold(config.real_min_none_recall_milli):
        reasons.append("REAL_NONE_RECALL_BELOW_MINIMUM")
    for label in METER_CLASSES[1:]:
        if candidate_real.per_class_recall[label] < threshold(config.real_min_positive_class_recall_milli):
            reasons.append(f"REAL_{label.replace('/', '_')}_RECALL_BELOW_MINIMUM")
    if candidate_real.macro_f1 - baseline_real.macro_f1 < threshold(config.real_min_macro_gain_milli):
        reasons.append("REAL_MACRO_F1_GAIN_TOO_SMALL")
    if baseline_synthetic.macro_f1 - candidate_synthetic.macro_f1 > threshold(config.synthetic_max_macro_f1_drop_milli):
        reasons.append("SYNTHETIC_MACRO_F1_REGRESSION")
    if (
        baseline_synthetic.positive_localization_f1_2px - candidate_synthetic.positive_localization_f1_2px
        > threshold(config.synthetic_max_localization_drop_milli)
    ):
        reasons.append("SYNTHETIC_LOCALIZATION_REGRESSION")
    return AdaptationGateDecisionV1(accepted=not reasons, reasons=tuple(reasons))


def meter_real_domain_adaptation_fingerprint_v1(
    *,
    teacher_manifest_sha256: str,
    d10_manifest_sha256: str,
    d10_artifact_binding_sha256: str,
    config: MeterRealDomainAdaptationConfigV1 = FROZEN_ADAPTATION_CONFIG_V1,
) -> str:
    return _sha(
        _canonical_json(
            {
                "version": METER_REAL_DOMAIN_ADAPTATION_V1,
                "base_checkpoint_sha256": PRESENCE_D11_SHA256,
                "teacher_admission_version": METER_TEACHER_GOLD_ADMISSION_V1,
                "teacher_manifest_sha256": _hex64("teacher_manifest_sha256", teacher_manifest_sha256),
                "d10_manifest_sha256": _hex64("d10_manifest_sha256", d10_manifest_sha256),
                "d10_artifact_binding_sha256": _hex64(
                    "d10_artifact_binding_sha256", d10_artifact_binding_sha256
                ),
                "config": asdict(config),
                "classes": METER_CLASSES,
                "test_policy": "sealed-test-never-enumerated-or-opened",
                "promotion_policy": "shadow-candidate-only-explicit-later-gate",
            }
        )
    )


@dataclass(frozen=True, slots=True)
class TeacherMeterRecordV1:
    record_id: str
    split: str
    family_id: str
    image_path: Path
    image_sha256: str
    label_path: Path
    label_sha256: str


def _load_teacher_records(bundle_root: Path) -> tuple[TeacherMeterRecordV1, ...]:
    receipt = verify_meter_teacher_gold_bundle_v1(bundle_root)
    manifest, raw = _read_canonical_json(
        bundle_root / "manifest.json", maximum=_MAX_MANIFEST_BYTES, name="teacher-gold manifest"
    )
    if _sha(raw) != receipt.manifest_sha256:
        _fail("teacher-gold receipt/manifest SHA mismatch")
    records: list[TeacherMeterRecordV1] = []
    rows = manifest.get("records")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        _fail("teacher-gold manifest records must be a sequence")
    for row in rows:
        if not isinstance(row, Mapping):
            _fail("teacher-gold manifest record must be an object")
        split = row.get("split")
        if split == "test":
            _fail("sealed TEST record reached real-domain adaptation")
        if split not in {"train", "validation"}:
            _fail("teacher-gold adaptation split must be train or validation")
        record_id = _hex64("teacher record_id", row.get("record_id"))
        image_path = bundle_root / str(row.get("image_path"))
        label_path = bundle_root / str(row.get("label_path"))
        if bundle_root.resolve() not in image_path.resolve().parents or bundle_root.resolve() not in label_path.resolve().parents:
            _fail("teacher-gold artifact path escapes bundle root")
        records.append(
            TeacherMeterRecordV1(
                record_id=record_id,
                split=str(split),
                family_id=str(row.get("family_id")),
                image_path=image_path,
                image_sha256=_hex64("teacher image_sha256", row.get("image_sha256")),
                label_path=label_path,
                label_sha256=_hex64("teacher label_sha256", row.get("label_sha256")),
            )
        )
    if Counter(record.split for record in records) != {"train": 54, "validation": 18}:
        _fail("teacher-gold adaptation record counts changed")
    return tuple(sorted(records, key=lambda record: (record.split, record.family_id, record.record_id)))


def _teacher_target(record: TeacherMeterRecordV1) -> Mapping[str, object]:
    label, raw = _read_canonical_json(record.label_path, maximum=_MAX_LABEL_BYTES, name="teacher-gold label")
    if _sha(raw) != record.label_sha256:
        _fail("teacher-gold label SHA mismatch")
    if label.get("schema_version") != TEACHER_LABEL_SCHEMA or label.get("record_id") != record.record_id:
        _fail("teacher-gold label schema/identity mismatch")
    target = label.get("target")
    if not isinstance(target, Mapping) or target.get("meter_class") not in METER_CLASSES:
        _fail("teacher-gold target class is invalid")
    return target


def _teacher_image(record: TeacherMeterRecordV1):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise MeterRealDomainAdaptationError("torch is required only for the adaptation execution stage") from exc
    raw = _read_regular(record.image_path, maximum=_MAX_IMAGE_BYTES, name="teacher-gold ROI image")
    if _sha(raw) != record.image_sha256:
        _fail("teacher-gold image SHA mismatch")
    try:
        with Image.open(BytesIO(raw)) as opened:
            opened.load()
            if opened.format != "PNG" or opened.mode != "L" or opened.size != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
                _fail("teacher-gold ROI must be exact gray8 256x192 PNG")
            pixels = bytearray(opened.tobytes())
    except MeterRealDomainAdaptationError:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise MeterRealDomainAdaptationError("teacher-gold ROI cannot be decoded") from exc
    tensor = torch.frombuffer(pixels, dtype=torch.uint8).clone().reshape(OUTPUT_HEIGHT, OUTPUT_WIDTH)
    return (1.0 - tensor.to(dtype=torch.float32) / 255.0).unsqueeze(0)


def deterministic_replay_ids_v1(
    class_to_record_ids: Mapping[str, Sequence[str]],
    *,
    per_class: int,
    seed: int,
) -> tuple[str, ...]:
    """Pure deterministic balanced sampler used by the D10 replay boundary."""
    if not isinstance(per_class, int) or isinstance(per_class, bool) or per_class <= 0:
        raise ValueError("per_class must be a positive integer")
    selected: list[str] = []
    for label in METER_CLASSES:
        values = class_to_record_ids.get(label)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            _fail(f"D10 replay class {label} is missing")
        unique = set(values)
        if len(unique) != len(values) or len(values) < per_class:
            _fail(f"D10 replay class {label} lacks unique records")
        ranked = sorted(
            values,
            key=lambda record_id: _sha(
                _canonical_json(
                    {
                        "version": METER_REAL_DOMAIN_ADAPTATION_V1,
                        "seed": seed,
                        "class": label,
                        "record_id": record_id,
                    }
                )
            ),
        )
        selected.extend(ranked[:per_class])
    return tuple(selected)


def balanced_class_weight_values_v1(class_counts: Mapping[str, int]) -> tuple[float, float, float, float]:
    """Return normalized inverse-frequency weights in frozen Meter class order."""
    if set(class_counts) != set(METER_CLASSES):
        _fail("adaptation class counts must cover exactly the four Meter classes")
    counts: list[int] = []
    for label in METER_CLASSES:
        value = class_counts[label]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            _fail(f"adaptation class count for {label} must be a positive integer")
        counts.append(value)
    total = float(sum(counts))
    raw = [total / value for value in counts]
    mean = sum(raw) / len(raw)
    return tuple(min(4.0, max(0.25, value / mean)) for value in raw)  # type: ignore[return-value]


def _stack_teacher(records: Sequence[TeacherMeterRecordV1]):
    try:
        import torch
        from .stage7d11_barline_meter_training import meter_target
    except ModuleNotFoundError as exc:
        raise MeterRealDomainAdaptationError("torch is required only for the adaptation execution stage") from exc
    images, classes, boxes, positive = [], [], [], []
    for record in records:
        class_index, box, is_positive = meter_target(_teacher_target(record))
        images.append(_teacher_image(record))
        classes.append(class_index)
        boxes.append(box)
        positive.append(is_positive)
    return (
        torch.stack(images),
        torch.tensor(classes, dtype=torch.long),
        torch.stack(boxes),
        torch.tensor(positive, dtype=torch.bool),
    )


def _metrics_from_outputs(*, loss: float, confusion, locations: Sequence[float]) -> MeterEvaluationV1:
    class_counts: dict[str, int] = {}
    recalls: dict[str, float] = {}
    f1_values: list[float] = []
    for index, label in enumerate(METER_CLASSES):
        true_count = int(confusion[index, :].sum().item())
        true_positive = int(confusion[index, index].item())
        false_positive = int(confusion[:, index].sum().item()) - true_positive
        false_negative = true_count - true_positive
        precision = 0.0 if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
        recall = 0.0 if true_count == 0 else true_positive / true_count
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        class_counts[label] = true_count
        recalls[label] = recall
        f1_values.append(f1)
    total = int(confusion.sum().item())
    correct = sum(int(confusion[index, index].item()) for index in range(4))
    return MeterEvaluationV1(
        loss=loss,
        macro_f1=sum(f1_values) / 4,
        accuracy=0.0 if total == 0 else correct / total,
        positive_localization_f1_2px=0.0 if not locations else sum(locations) / len(locations),
        class_counts=class_counts,
        per_class_recall=recalls,
        confusion=tuple(tuple(int(value) for value in row.tolist()) for row in confusion),
    )


def _evaluate_teacher(model, records: Sequence[TeacherMeterRecordV1], config: MeterRealDomainAdaptationConfigV1) -> MeterEvaluationV1:
    import torch
    from .stage7d11_barline_meter_training import _bbox_mask, _tolerant_f1, meter_loss

    weights = torch.ones(4, dtype=torch.float32)
    model.eval()
    total_loss, batches = 0.0, 0
    confusion = torch.zeros((4, 4), dtype=torch.int64)
    locations: list[float] = []
    with torch.no_grad():
        for start in range(0, len(records), config.batch_size):
            images, classes, boxes, positive = _stack_teacher(records[start : start + config.batch_size])
            logits, predicted_boxes = model(images)
            # Reuse the D11 objective while preserving the frozen bbox coefficient.
            from .stage7d11_barline_meter_training import FROZEN_D11_CONFIG

            total_loss += float(
                meter_loss(logits, predicted_boxes, classes, boxes, positive, weights, FROZEN_D11_CONFIG).item()
            )
            batches += 1
            for true_class, predicted_class in zip(classes.tolist(), logits.argmax(1).tolist()):
                confusion[true_class, predicted_class] += 1
            for index in torch.nonzero(positive, as_tuple=False).flatten().tolist():
                locations.append(
                    _tolerant_f1(
                        _bbox_mask(predicted_boxes[index], OUTPUT_HEIGHT, OUTPUT_WIDTH),
                        _bbox_mask(boxes[index], OUTPUT_HEIGHT, OUTPUT_WIDTH),
                        2,
                    )
                )
    if batches == 0 or not locations:
        _fail("real teacher validation produced incomplete metrics")
    return _metrics_from_outputs(loss=total_loss / batches, confusion=confusion, locations=locations)


def _teacher_inference_fingerprint(model, records: Sequence[TeacherMeterRecordV1]) -> str:
    import torch

    model.eval()
    with torch.no_grad():
        images, _classes, _boxes, _positive = _stack_teacher(records)
        logits, predicted_boxes = model(images)
        probabilities = torch.softmax(logits, dim=1)
    return _sha(
        _canonical_json(
            {
                "record_ids": [record.record_id for record in records],
                "class_probabilities": probabilities.tolist(),
                "predicted_boxes": predicted_boxes.tolist(),
            }
        )
    )


def _evaluation_from_d11_metrics(loss: float, d11_metrics, records, model, config) -> MeterEvaluationV1:
    """Re-evaluate confusion to add accuracy/recalls to the frozen D11 metrics."""
    import torch
    from .stage7d11_barline_meter_training import _stack_meter

    confusion = torch.zeros((4, 4), dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), config.batch_size):
            images, classes, _boxes, _positive = _stack_meter(records[start : start + config.batch_size])
            logits, _predicted = model(images)
            for true_class, predicted_class in zip(classes.tolist(), logits.argmax(1).tolist()):
                confusion[true_class, predicted_class] += 1
    base = _metrics_from_outputs(loss=loss, confusion=confusion, locations=(d11_metrics.positive_localization_f1_2px,))
    return MeterEvaluationV1(
        loss=base.loss,
        macro_f1=d11_metrics.macro_f1,
        accuracy=base.accuracy,
        positive_localization_f1_2px=d11_metrics.positive_localization_f1_2px,
        class_counts=base.class_counts,
        per_class_recall=base.per_class_recall,
        confusion=base.confusion,
    )


def _evaluate_synthetic(model, records, config: MeterRealDomainAdaptationConfigV1) -> MeterEvaluationV1:
    import torch
    from .stage7d11_barline_meter_training import FROZEN_D11_CONFIG, _evaluate_meter

    weights = torch.ones(4, dtype=torch.float32)
    d11_config = FROZEN_D11_CONFIG
    loss, metrics = _evaluate_meter(model, records, weights, d11_config)
    return _evaluation_from_d11_metrics(loss, metrics, records, model, config)


def _fresh_output_root(output_root: Path, repository_root: Path) -> None:
    output = output_root.resolve()
    repository = repository_root.resolve()
    if output == repository or repository in output.parents:
        _fail("adaptation outputs/checkpoints must remain outside Git")
    if output_root.exists() or output_root.is_symlink():
        _fail("adaptation output root must be fresh")
    output_root.mkdir(parents=True)


def _prepare_output_root(output_root: Path, repository_root: Path, *, resume: bool) -> None:
    if not output_root.exists() and not output_root.is_symlink():
        _fresh_output_root(output_root, repository_root)
        return
    output = output_root.resolve()
    repository = repository_root.resolve()
    if output == repository or repository in output.parents:
        _fail("adaptation outputs/checkpoints must remain outside Git")
    if output_root.is_symlink() or not output_root.is_dir():
        _fail("adaptation output root must be a regular directory")
    if not resume:
        _fail("adaptation output root must be fresh unless resume is explicit")
    if (output_root / "RUN_COMPLETE").exists():
        _fail("completed adaptation output cannot be resumed")


def _evaluation_from_payload_v1(payload: object, *, name: str) -> MeterEvaluationV1:
    if not isinstance(payload, Mapping):
        _fail(f"{name} must be an evaluation object")
    try:
        confusion = tuple(tuple(int(value) for value in row) for row in payload["confusion"])
        return MeterEvaluationV1(
            loss=float(payload["loss"]),
            macro_f1=float(payload["macro_f1"]),
            accuracy=float(payload["accuracy"]),
            positive_localization_f1_2px=float(payload["positive_localization_f1_2px"]),
            class_counts={str(key): int(value) for key, value in dict(payload["class_counts"]).items()},
            per_class_recall={
                str(key): float(value) for key, value in dict(payload["per_class_recall"]).items()
            },
            confusion=confusion,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MeterRealDomainAdaptationError(f"{name} is malformed") from exc


def run_meter_real_domain_adaptation_v1(
    *,
    teacher_bundle_root: str | Path,
    d10_root: str | Path,
    base_checkpoint_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
    expected_d10_manifest_sha256: str,
    expected_d10_artifact_binding_sha256: str,
    config: MeterRealDomainAdaptationConfigV1 = FROZEN_ADAPTATION_CONFIG_V1,
    progress=None,
    resume: bool = False,
) -> dict[str, object]:
    """Run one deterministic, shadow-only adaptation and return its metrics payload."""
    if not isinstance(config, MeterRealDomainAdaptationConfigV1):
        raise TypeError("config must be MeterRealDomainAdaptationConfigV1")
    if config != FROZEN_ADAPTATION_CONFIG_V1:
        _fail("Meter real-domain adaptation v1 requires the frozen configuration")
    if not isinstance(resume, bool):
        raise TypeError("resume must be bool")
    try:
        import torch
        from .runtime_meter_real_checkpoint_audit_v1 import audit_presence_d11_checkpoint_v1
        from .stage7c_execution import verify_authoritative_repository
        from .stage7d11_barline_meter_training import (
            FROZEN_D11_CONFIG,
            _clone_state,
            _load_d11_label,
            _stack_meter,
            _train_meter_batch,
            build_meter_refiner,
            load_verified_stage7d11_records,
        )
        from .training_model import assert_model_finite, model_state_sha256, set_deterministic_cpu
    except ModuleNotFoundError as exc:
        raise MeterRealDomainAdaptationError("torch and the pinned training runtime are required for adaptation") from exc

    teacher_root = Path(teacher_bundle_root)
    if progress is not None:
        progress("phase_started", {"phase": "teacher_gold_verify", "phase_index": 1, "phase_total": 7})
    teacher_receipt = verify_meter_teacher_gold_bundle_v1(teacher_root)
    teacher_records = _load_teacher_records(teacher_root)
    real_train = tuple(record for record in teacher_records if record.split == "train")
    real_validation = tuple(record for record in teacher_records if record.split == "validation")
    if Counter(_teacher_target(record).get("meter_class") for record in real_train) != {
        "none": 27,
        "2/4": 9,
        "3/4": 9,
        "4/4": 9,
    }:
        _fail("real TRAIN class balance changed")

    d10_manifest = _hex64("expected D10 manifest SHA", expected_d10_manifest_sha256)
    d10_binding = _hex64("expected D10 artifact binding SHA", expected_d10_artifact_binding_sha256)
    if progress is not None:
        progress(
            "phase_started",
            {
                "phase": "d10_full_integrity_verify",
                "phase_index": 2,
                "phase_total": 7,
                "records_total": 22_128,
            },
        )
    d10_records = load_verified_stage7d11_records(
        d10_root,
        expected_manifest_sha256=d10_manifest,
        expected_artifact_binding_sha256=d10_binding,
    )
    synthetic_train_all = tuple(record for record in d10_records if record.kind == "meter" and record.split == "train")
    synthetic_validation = tuple(
        record for record in d10_records if record.kind == "meter" and record.split == "validation"
    )
    if len(synthetic_train_all) != 9840 or len(synthetic_validation) != 1224:
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
    profile = meter_real_domain_adaptation_fingerprint_v1(
        teacher_manifest_sha256=teacher_receipt.manifest_sha256,
        d10_manifest_sha256=d10_manifest,
        d10_artifact_binding_sha256=d10_binding,
        config=config,
    )
    root = Path(output_root)
    _prepare_output_root(root, Path(repository_root), resume=resume)

    set_deterministic_cpu(config.master_seed)
    model = build_meter_refiner(FROZEN_D11_CONFIG)
    model.load_state_dict(dict(audited.model_state), strict=True)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        _fail("D11 encoder must remain frozen during pilot adaptation")
    encoder_state_sha_before = model_state_sha256(model.encoder)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        _fail("adaptation trainable head surface is empty")
    base_state_sha = model_state_sha256(model)
    if progress is not None:
        progress("phase_started", {"phase": "baseline_real_validation", "phase_index": 4, "phase_total": 7})
    baseline_real = _evaluate_teacher(model, real_validation, config)
    if progress is not None:
        progress(
            "phase_started",
            {"phase": "baseline_synthetic_validation", "phase_index": 5, "phase_total": 7, "records_total": 1_224},
        )
    baseline_synthetic = _evaluate_synthetic(model, synthetic_validation, config)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate_micros / 1_000_000.0,
        weight_decay=config.weight_decay_micros / 1_000_000.0,
    )
    effective_class_counts = {
        "none": 27 * config.real_repeat_factor + config.synthetic_replay_per_class,
        "2/4": 9 * config.real_repeat_factor + config.synthetic_replay_per_class,
        "3/4": 9 * config.real_repeat_factor + config.synthetic_replay_per_class,
        "4/4": 9 * config.real_repeat_factor + config.synthetic_replay_per_class,
    }
    class_weight_values = balanced_class_weight_values_v1(effective_class_counts)
    weights = torch.tensor(class_weight_values, dtype=torch.float32)
    history: list[dict[str, object]] = []
    best_state = None
    best_decision = AdaptationGateDecisionV1(False, ("NO_EPOCH_EVALUATED",))
    best_real = baseline_real
    best_synthetic = baseline_synthetic
    best_epoch = 0
    optimizer_steps = 0

    resume_path = root / "resume.pt"
    if resume_path.exists():
        if resume_path.is_symlink() or not resume_path.is_file():
            _fail("adaptation resume state must be a regular file")
        try:
            snapshot = torch.load(resume_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise MeterRealDomainAdaptationError("adaptation resume state cannot be loaded safely") from exc
        if not isinstance(snapshot, Mapping):
            _fail("adaptation resume state must be a mapping")
        resume_checks = {
            "role": "meter-real-domain-adaptation-resume-v1",
            "adaptation_version": METER_REAL_DOMAIN_ADAPTATION_V1,
            "repository_sha": repository_sha,
            "profile_fingerprint": profile,
            "teacher_manifest_sha256": teacher_receipt.manifest_sha256,
            "d10_manifest_sha256": d10_manifest,
            "base_checkpoint_sha256": PRESENCE_D11_SHA256,
            "base_meter_state_sha256": base_state_sha,
            "encoder_state_sha256": encoder_state_sha_before,
            "baseline_real": asdict(baseline_real),
            "baseline_synthetic": asdict(baseline_synthetic),
        }
        for name, expected in resume_checks.items():
            if snapshot.get(name) != expected:
                _fail(f"adaptation resume state {name} mismatch")
        try:
            completed_epoch_value = snapshot["completed_epoch"]
            best_epoch_value = snapshot["best_epoch"]
            optimizer_steps_value = snapshot["optimizer_steps"]
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (completed_epoch_value, best_epoch_value, optimizer_steps_value)
            ):
                _fail("adaptation resume counters must be plain integers")
            completed_epoch = completed_epoch_value
            if not 1 <= completed_epoch <= config.epochs:
                _fail("adaptation resume epoch is outside the frozen run")
            model.load_state_dict(snapshot["current_model_state"], strict=True)
            optimizer.load_state_dict(snapshot["optimizer_state_dict"])
            loaded_best = snapshot.get("best_model_state")
            best_state = None if loaded_best is None else dict(loaded_best)
            decision_payload = snapshot["best_decision"]
            if not isinstance(decision_payload, Mapping):
                _fail("adaptation resume best decision is malformed")
            accepted_value = decision_payload.get("accepted")
            reasons_value = decision_payload.get("reasons")
            if not isinstance(accepted_value, bool):
                _fail("adaptation resume best decision accepted flag is malformed")
            if not isinstance(reasons_value, Sequence) or isinstance(reasons_value, (str, bytes, bytearray)):
                _fail("adaptation resume best decision reasons are malformed")
            if any(not isinstance(value, str) for value in reasons_value):
                _fail("adaptation resume best decision reason must be a string")
            best_decision = AdaptationGateDecisionV1(
                accepted_value, tuple(reasons_value)
            )
            best_real = _evaluation_from_payload_v1(snapshot["best_real"], name="resume best real")
            best_synthetic = _evaluation_from_payload_v1(
                snapshot["best_synthetic"], name="resume best synthetic"
            )
            best_epoch = best_epoch_value
            optimizer_steps = optimizer_steps_value
            history_value = snapshot["history"]
            if not isinstance(history_value, Sequence) or isinstance(history_value, (str, bytes, bytearray)):
                _fail("adaptation resume history must be a sequence")
            history = list(history_value)
        except MeterRealDomainAdaptationError:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise MeterRealDomainAdaptationError("adaptation resume state is malformed") from exc
        assert_model_finite(model)
        if model_state_sha256(model.encoder) != encoder_state_sha_before:
            _fail("resumed adaptation mutated the frozen encoder")
        if progress is not None:
            progress(
                "resume_loaded",
                {"completed_epoch": completed_epoch, "epochs_total": config.epochs, "optimizer_steps": optimizer_steps},
            )
    else:
        completed_epoch = 0

    training_items = [("real", record) for record in real_train for _ in range(config.real_repeat_factor)]
    training_items += [("synthetic", record) for record in synthetic_replay]
    batches_per_epoch = math.ceil(len(training_items) / config.batch_size)
    if len(history) != completed_epoch:
        _fail("adaptation resume history length differs from completed epoch")
    if optimizer_steps != completed_epoch * batches_per_epoch:
        _fail("adaptation resume optimizer step count mismatch")
    if not 0 <= best_epoch <= completed_epoch:
        _fail("adaptation resume best epoch is outside completed history")
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
        total_loss, batches = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            items = [training_items[index] for index in order[start : start + config.batch_size]]
            teacher_batch = [record for kind, record in items if kind == "real"]
            synthetic_batch = [record for kind, record in items if kind == "synthetic"]
            tensors = []
            if teacher_batch:
                tensors.append(_stack_teacher(teacher_batch))
            if synthetic_batch:
                tensors.append(_stack_meter(synthetic_batch))
            images = torch.cat([item[0] for item in tensors], dim=0)
            classes = torch.cat([item[1] for item in tensors], dim=0)
            boxes = torch.cat([item[2] for item in tensors], dim=0)
            positive = torch.cat([item[3] for item in tensors], dim=0)
            total_loss += _train_meter_batch(
                model,
                images,
                classes,
                boxes,
                positive,
                weights,
                split="train",
                optimizer=optimizer,
                config=FROZEN_D11_CONFIG,
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
        if progress is not None:
            progress(
                "epoch_validation_started",
                {"epoch": epoch, "epochs_total": config.epochs, "validation_records": 18 + 1_224},
            )
        real_metrics = _evaluate_teacher(model, real_validation, config)
        synthetic_metrics = _evaluate_synthetic(model, synthetic_validation, config)
        decision = adaptation_acceptance_v1(
            baseline_real=baseline_real,
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
        if best_state is None and not decision.accepted and (
            best_epoch == 0
            or real_metrics.macro_f1 > best_real.macro_f1
            or (
                real_metrics.macro_f1 == best_real.macro_f1
                and real_metrics.loss < best_real.loss
            )
        ):
            # Preserve the strongest rejected epoch in the receipt so a HOLD
            # explains the real gate failures instead of reporting NO_EPOCH.
            best_decision = decision
            best_real = real_metrics
            best_synthetic = synthetic_metrics
            best_epoch = epoch
        better = decision.accepted and (
            best_state is None
            or real_metrics.macro_f1 > best_real.macro_f1
            or (
                real_metrics.macro_f1 == best_real.macro_f1
                and real_metrics.loss < best_real.loss
            )
        )
        if better:
            best_state = _clone_state(model)
            best_decision = decision
            best_real = real_metrics
            best_synthetic = synthetic_metrics
            best_epoch = epoch
        temporary_resume = root / "resume.tmp.pt"
        torch.save(
            {
                "role": "meter-real-domain-adaptation-resume-v1",
                "adaptation_version": METER_REAL_DOMAIN_ADAPTATION_V1,
                "repository_sha": repository_sha,
                "profile_fingerprint": profile,
                "teacher_manifest_sha256": teacher_receipt.manifest_sha256,
                "d10_manifest_sha256": d10_manifest,
                "base_checkpoint_sha256": PRESENCE_D11_SHA256,
                "base_meter_state_sha256": base_state_sha,
                "encoder_state_sha256": encoder_state_sha_before,
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
            progress(
                "epoch_checkpointed",
                {"epoch": epoch, "epochs_total": config.epochs, "resume_path": str(resume_path)},
            )
            progress("epoch_complete", event)

    if progress is not None:
        progress("phase_started", {"phase": "final_verification", "phase_index": 7, "phase_total": 7})
    ending_sha, ending_origin = verify_authoritative_repository(repository_root)
    if (ending_sha, ending_origin) != (repository_sha, repository_origin):
        _fail("repository identity changed during adaptation")
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    assert_model_finite(model)
    if model_state_sha256(model.encoder) != encoder_state_sha_before:
        _fail("frozen D11 encoder changed during pilot adaptation")
    candidate_state_sha = model_state_sha256(model) if best_state is not None else None
    replay_fingerprints: tuple[str, ...] = ()
    if best_state is not None:
        replay_fingerprints = tuple(
            _teacher_inference_fingerprint(model, real_validation) for _ in range(10)
        )
        if len(set(replay_fingerprints)) != 1:
            _fail("shadow candidate inference is not deterministic 10/10")
    status = "SHADOW_CANDIDATE_ACCEPTED" if best_state is not None else "HOLD_NO_ACCEPTED_CANDIDATE"
    run_id = _sha(
        _canonical_json(
            {
                "version": METER_REAL_DOMAIN_ADAPTATION_V1,
                "repository_sha": repository_sha,
                "profile_fingerprint": profile,
                "teacher_manifest_sha256": teacher_receipt.manifest_sha256,
                "d10_manifest_sha256": d10_manifest,
                "base_checkpoint_sha256": PRESENCE_D11_SHA256,
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
                "role": CHECKPOINT_ROLE,
                "meter_state_dict": best_state,
                "base_checkpoint_sha256": PRESENCE_D11_SHA256,
                "base_meter_state_sha256": base_state_sha,
                "candidate_meter_state_sha256": candidate_state_sha,
                "profile_fingerprint": profile,
                "teacher_manifest_sha256": teacher_receipt.manifest_sha256,
                "d10_manifest_sha256": d10_manifest,
                "best_epoch": best_epoch,
                "production_promotion_authorized": False,
            },
            temporary,
        )
        checkpoint_sha = _sha(temporary.read_bytes())
        checkpoint_path = root / f"checkpoint-{checkpoint_sha}.pt"
        temporary.rename(checkpoint_path)
        try:
            reloaded_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(reloaded_payload, Mapping):
                _fail("shadow checkpoint payload must be a mapping")
            if reloaded_payload.get("role") != CHECKPOINT_ROLE:
                _fail("shadow checkpoint role changed during reload")
            reloaded_model = build_meter_refiner(FROZEN_D11_CONFIG)
            reloaded_model.load_state_dict(reloaded_payload["meter_state_dict"], strict=True)
            assert_model_finite(reloaded_model)
        except MeterRealDomainAdaptationError:
            raise
        except Exception as exc:
            raise MeterRealDomainAdaptationError("shadow checkpoint cannot be strictly reloaded") from exc
        if model_state_sha256(reloaded_model) != candidate_state_sha:
            _fail("shadow checkpoint state hash mismatch after reload")
        checkpoint_reload_verified = True

    metrics = {
        "schema_version": METRICS_SCHEMA,
        "adaptation_version": METER_REAL_DOMAIN_ADAPTATION_V1,
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
        "training_balance": {
            "effective_class_counts": effective_class_counts,
            "class_weights": dict(zip(METER_CLASSES, class_weight_values)),
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
        "encoder_mutated": False,
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
        "schema_version": VERIFICATION_SCHEMA,
        "adaptation_version": METER_REAL_DOMAIN_ADAPTATION_V1,
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
        "encoder_mutated": False,
        "checkpoint_is_shadow_only": True,
        "candidate_replay_10_of_10": best_state is not None and len(set(replay_fingerprints)) == 1,
        "candidate_replay_fingerprint": replay_fingerprints[0] if replay_fingerprints else None,
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
    lines = [f"{verification_sha}  {verification_path.name}", f"{metrics_sha}  {metrics_path.name}"]
    if checkpoint_path is not None and checkpoint_sha is not None:
        lines.append(f"{checkpoint_sha}  {checkpoint_path.name}")
    (root / "RUN_COMPLETE").write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    if progress is not None:
        progress(
            "run_complete",
            {"status": status, "best_epoch": best_epoch, "epochs_total": config.epochs, "run_root": str(root)},
        )
    return metrics


def sealed_test_access_allowed() -> bool:
    return False


def runtime_connection_allowed() -> bool:
    return False


def resolver_connection_allowed() -> bool:
    return False


def production_promotion_allowed() -> bool:
    return False
