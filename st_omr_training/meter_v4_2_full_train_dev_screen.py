"""Meter V4-2 deterministic full-TRAIN numerator candidate and dev-only screen."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Final

from PIL import Image
import torch
from torch.nn import functional as F

from .meter_teacher_gold_admission_v1 import (
    ALLOWED_USE,
    CHOICES_SCHEMA,
    EXPECTED_TASKS,
    PILOT_SCHEMA,
    _adaptation_split_by_family,
    _bounded_ascii,
    _json_file,
    _mapping,
    _sequence,
    _sha as _teacher_sha,
    _validate_permission,
    _validate_privacy,
    _xywh,
)
from .meter_v4_0_numerator_audit_run import _derive_selected_crop
from .meter_v4_1_numerator_specialist import (
    EXPECTED_PARAMETER_COUNT_V4_1,
    FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1,
    NUMERATOR_CLASSES_V4_1,
    NumeratorRecordV4_1,
    NumeratorSpecialistV4_1,
    VerifiedParentV4_1,
    load_crop_tensor_v4_1,
    translate_ink_v4_1,
)
from .training_model import (
    assert_finite_tensor,
    assert_model_finite,
    count_trainable_parameters,
    model_state_sha256,
    set_deterministic_cpu,
)


METER_V4_2_FULL_TRAIN_DEV_SCREEN: Final[str] = "meter-v4-2-full-train-dev-screen-v1"
FINAL_SEED_V4_2: Final[int] = 812_042
EXPECTED_PILOT_SHA256: Final[str] = "7f6234d97dc9d5afb3357d4fd313c614f240bf7cc87b1dbd63c3528a125c86a8"
EXPECTED_CHOICES_SHA256: Final[str] = "fa4a46ee45b633e34b78623f54b53c92fef0f7c8935c3f29938c0f4234f2cc1e"
EXPECTED_PERMISSION_SHA256: Final[str] = "99f34e629c16c0b3961f1ae2dee35d49e71b63a041b811e268c0d9a4fef063f8"
EXPECTED_PRIVACY_SHA256: Final[str] = "e3dc68be617f1f1fdff97bdb7798dc20a747ffa3a2bd2acc9a76814e0a0322c3"
_MAX_PILOT_BYTES: Final[int] = 32 * 1024 * 1024
_MAX_CHOICES_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_EVIDENCE_BYTES: Final[int] = 64 * 1024


class MeterV4_2Error(RuntimeError):
    """Raised when V4-2 violates provenance, determinism or dev-screen bounds."""


def _fail(message: str) -> None:
    raise MeterV4_2Error(message)


@dataclass(frozen=True, slots=True)
class FullTrainResultV4_2:
    final_loss: float
    model_state_sha256: str
    optimizer_steps: int
    model: NumeratorSpecialistV4_1


@dataclass(frozen=True, slots=True)
class DevPredictionV4_2:
    record_id: str
    family_id: str
    true_class: str
    predicted_class: str
    logits: tuple[float, float, float]
    probabilities: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class DevSummaryV4_2:
    record_count: int
    accuracy: float
    macro_f1: float
    per_class_recall: dict[str, float]
    confusion: tuple[tuple[int, int, int], ...]


def build_full_train_batch_v4_2(
    parent: VerifiedParentV4_1,
    crops_by_record_id: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    records = sorted(parent.records, key=lambda row: (row.numerator_class, row.family_id, row.record_id))
    if len(records) != 27 or Counter(row.numerator_class for row in records) != Counter({"2": 9, "3": 9, "4": 9}):
        _fail("V4-2 requires exact balanced 27 V4-0 TRAIN records")
    if set(crops_by_record_id) != {row.record_id for row in records}:
        _fail("V4-2 crop tensor map must match exact 27 parent records")
    images: list[torch.Tensor] = []
    labels: list[int] = []
    origins: list[str] = []
    for row in records:
        image = crops_by_record_id[row.record_id]
        for dy in (-2, 0, 2):
            for dx in (-2, 0, 2):
                images.append(translate_ink_v4_1(image, dx=dx, dy=dy))
                labels.append(NUMERATOR_CLASSES_V4_1.index(row.numerator_class))
                origins.append(row.record_id)
    batch = torch.stack(images, dim=0)
    target = torch.tensor(labels, dtype=torch.long)
    if tuple(batch.shape) != (243, 1, 64, 64):
        _fail("V4-2 full TRAIN batch must contain exactly 243 views")
    if Counter(target.tolist()) != Counter({0: 81, 1: 81, 2: 81}) or len(set(origins)) != 27:
        _fail("V4-2 augmented TRAIN batch lost class/family balance")
    return batch, target, tuple(origins)


def _new_candidate_model() -> NumeratorSpecialistV4_1:
    set_deterministic_cpu(FINAL_SEED_V4_2)
    model = NumeratorSpecialistV4_1(FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1).cpu()
    if count_trainable_parameters(model) != EXPECTED_PARAMETER_COUNT_V4_1:
        _fail("V4-2 candidate parameter count differs from accepted V4-1 architecture")
    assert_model_finite(model)
    return model


def train_full_candidate_v4_2(
    parent: VerifiedParentV4_1,
    crops_by_record_id: Mapping[str, torch.Tensor],
) -> FullTrainResultV4_2:
    config = FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1
    model = _new_candidate_model()
    batch, target, _origins = build_full_train_batch_v4_2(parent, crops_by_record_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate_micros / 1_000_000.0,
        weight_decay=config.weight_decay_micros / 1_000_000.0,
    )
    final_loss = math.nan
    model.train()
    for _epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = F.cross_entropy(logits, target)
        assert_finite_tensor("V4-2 full TRAIN loss", loss)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_milli / 1000.0)
        if not math.isfinite(float(grad_norm)):
            _fail("V4-2 gradient norm is non-finite")
        optimizer.step()
        assert_model_finite(model)
        final_loss = float(loss.detach().cpu().item())
    if not math.isfinite(final_loss):
        _fail("V4-2 final loss is non-finite")
    return FullTrainResultV4_2(
        final_loss=final_loss,
        model_state_sha256=model_state_sha256(model),
        optimizer_steps=config.epochs,
        model=model,
    )


def select_validation_positives_v4_2(
    *,
    pilot_path: str | Path,
    choices_path: str | Path,
    permission_path: str | Path,
    privacy_path: str | Path,
) -> tuple[tuple[tuple[Mapping[str, object], Mapping[str, object], str], ...], dict[str, str]]:
    """Select exact nine positive adaptation-validation families without decoding images."""
    pilot, pilot_raw = _json_file(Path(pilot_path), maximum=_MAX_PILOT_BYTES, name="V4-2 pilot")
    choices, choices_raw = _json_file(Path(choices_path), maximum=_MAX_CHOICES_BYTES, name="V4-2 choices")
    permission, permission_raw = _json_file(Path(permission_path), maximum=_MAX_EVIDENCE_BYTES, name="V4-2 permission", canonical=True)
    privacy, privacy_raw = _json_file(Path(privacy_path), maximum=_MAX_EVIDENCE_BYTES, name="V4-2 privacy", canonical=True)
    actual_hashes = {
        "pilot_sha256": _teacher_sha(pilot_raw),
        "choices_sha256": _teacher_sha(choices_raw),
        "permission_sha256": _teacher_sha(permission_raw),
        "privacy_sha256": _teacher_sha(privacy_raw),
    }
    expected_hashes = {
        "pilot_sha256": EXPECTED_PILOT_SHA256,
        "choices_sha256": EXPECTED_CHOICES_SHA256,
        "permission_sha256": EXPECTED_PERMISSION_SHA256,
        "privacy_sha256": EXPECTED_PRIVACY_SHA256,
    }
    if actual_hashes != expected_hashes:
        _fail("V4-2 Teacher Gold source hashes differ from accepted V4-0 provenance")
    _validate_permission(permission)
    _validate_privacy(privacy)
    if permission.get("allowed_use") != ALLOWED_USE:
        _fail("V4-2 permission use differs from approved offline Meter pilot")
    if pilot.get("schema") != PILOT_SCHEMA or choices.get("schema") != CHOICES_SCHEMA:
        _fail("V4-2 Teacher Gold schema mismatch")
    if _mapping("V4-2 pilot selection", pilot.get("selection")).get("test_opened") is not False or choices.get("test_opened") is not False:
        _fail("sealed TEST evidence reached V4-2")
    tasks = [_mapping(f"V4-2 task[{i}]", row) for i, row in enumerate(_sequence("V4-2 tasks", pilot.get("tasks")))]
    answers = [_mapping(f"V4-2 answer[{i}]", row) for i, row in enumerate(_sequence("V4-2 answers", choices.get("answers")))]
    if len(tasks) != EXPECTED_TASKS or len(answers) != EXPECTED_TASKS:
        _fail("V4-2 requires exact Teacher Gold task cardinality")
    answer_by_id = {_bounded_ascii("V4-2 answer task_id", row.get("task_id")): row for row in answers}
    splits = _adaptation_split_by_family(tasks)
    selected = []
    for task in tasks:
        if task.get("kind") != "positive":
            continue
        family = _bounded_ascii("V4-2 family", task.get("family_key"))
        if splits[family] != "validation":
            continue
        task_id = _bounded_ascii("V4-2 task id", task.get("task_id"))
        answer = answer_by_id.get(task_id)
        if not isinstance(answer, Mapping):
            _fail("V4-2 validation answer missing")
        expected = task.get("expected_class")
        if expected not in ("2/4", "3/4", "4/4"):
            _fail("V4-2 validation class outside supported numerator classes")
        if answer.get("status") != "accepted" or answer.get("label_confirmed") is not True or answer.get("crop_usable") is not True:
            _fail("V4-2 validation positive is not accepted/confirmed/usable")
        if answer.get("label") != expected or answer.get("expected_class") != expected:
            _fail("V4-2 validation label mismatch")
        _xywh("V4-2 validation roi_crop_box", answer.get("roi_crop_box"))
        _xywh("V4-2 validation bbox", answer.get("bbox"))
        selected.append((task, answer, "validation"))
    if len(selected) != 9 or Counter(str(task.get("expected_class")) for task, _answer, _split in selected) != Counter({"2/4": 3, "3/4": 3, "4/4": 3}):
        _fail("V4-2 requires exactly 3/3/3 positive adaptation-validation families")
    if len({_bounded_ascii("V4-2 selected family", task.get("family_key")) for task, _answer, _split in selected}) != 9:
        _fail("V4-2 validation families must be unique")
    return tuple(sorted(selected, key=lambda row: (_bounded_ascii("family", row[0].get("family_key"))))), actual_hashes


def _crop_tensor(image: Image.Image) -> torch.Tensor:
    if image.mode != "L" or image.size != (64, 64):
        _fail("V4-2 numerator crop must be gray8 64x64")
    values = torch.tensor(list(image.tobytes()), dtype=torch.float32).reshape(1, 64, 64)
    tensor = (255.0 - values) / 255.0
    assert_finite_tensor("V4-2 dev crop tensor", tensor)
    return tensor


def evaluate_validation_positives_v4_2(
    model: NumeratorSpecialistV4_1,
    selected: Sequence[tuple[Mapping[str, object], Mapping[str, object], str]],
) -> tuple[tuple[DevPredictionV4_2, ...], DevSummaryV4_2]:
    if len(selected) != 9:
        _fail("V4-2 dev screen requires exactly nine selected positives")
    model.eval()
    predictions: list[DevPredictionV4_2] = []
    with torch.no_grad():
        for task, answer, adaptation_split in selected:
            identity, numerator, _vector, _bbox, _transform, _source_sha = _derive_selected_crop(task, answer, adaptation_split)
            true_class = identity.numerator_class
            logits = model(_crop_tensor(numerator).unsqueeze(0))[0]
            probabilities = torch.softmax(logits, dim=0)
            assert_finite_tensor("V4-2 dev probabilities", probabilities)
            pred_index = int(torch.argmax(probabilities).item())
            predictions.append(DevPredictionV4_2(
                record_id=identity.record_id,
                family_id=identity.family_id,
                true_class=true_class,
                predicted_class=NUMERATOR_CLASSES_V4_1[pred_index],
                logits=tuple(float(v) for v in logits.cpu().tolist()),
                probabilities=tuple(float(v) for v in probabilities.cpu().tolist()),
            ))
    confusion = [[0, 0, 0] for _ in range(3)]
    for row in predictions:
        confusion[NUMERATOR_CLASSES_V4_1.index(row.true_class)][NUMERATOR_CLASSES_V4_1.index(row.predicted_class)] += 1
    if any(sum(row) != 3 for row in confusion):
        _fail("V4-2 dev truth cardinality must remain three per class")
    correct = sum(confusion[i][i] for i in range(3))
    recalls = {}
    f1s = []
    for i, name in enumerate(NUMERATOR_CLASSES_V4_1):
        tp = confusion[i][i]
        fn = sum(confusion[i]) - tp
        fp = sum(confusion[r][i] for r in range(3)) - tp
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls[name] = recall
        f1s.append(f1)
    summary = DevSummaryV4_2(
        record_count=9,
        accuracy=correct / 9.0,
        macro_f1=sum(f1s) / 3.0,
        per_class_recall=recalls,
        confusion=tuple(tuple(v for v in row) for row in confusion),
    )
    return tuple(predictions), summary


def dev_decision_v4_2(summary: DevSummaryV4_2, *, deterministic_repeat_pass: bool) -> dict[str, object]:
    reasons = []
    if deterministic_repeat_pass is not True:
        reasons.append("FULL_TRAIN_DETERMINISM_FAILED")
    if summary.record_count != 9:
        reasons.append("DEV_RECORD_COUNT_NOT_9")
    if summary.accuracy != 1.0:
        reasons.append("DEV_ACCURACY_NOT_9_OF_9")
    if summary.macro_f1 != 1.0:
        reasons.append("DEV_MACRO_F1_NOT_1")
    for class_name in NUMERATOR_CLASSES_V4_1:
        if summary.per_class_recall.get(class_name) != 1.0:
            reasons.append(f"DEV_{class_name}_RECALL_NOT_3_OF_3")
    passed = not reasons
    return {
        "name": "FULL_TRAIN_DEV_SCREEN_PASS" if passed else "FULL_TRAIN_DEV_SCREEN_HOLD",
        "accepted_for_shadow_planning": passed,
        "reasons": reasons,
        "fresh_independent_holdout_required": True,
        "production_promotion_authorized": False,
    }
