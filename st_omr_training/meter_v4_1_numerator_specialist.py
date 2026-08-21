"""Deterministic Meter V4-1 learned numerator specialist.

V4-1 consumes only the exact completed V4-0 numerator crop artifact.  It trains
one tiny fixed-epoch CNN per inherited family-disjoint OOF fold.  Teacher Gold
adaptation-validation, D10, sealed TEST, runtime, Resolver and production remain
outside this module's admissible surface.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Final

from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torch.nn import functional as F

from .training_model import (
    TORCH_PINNED_VERSION,
    assert_finite_tensor,
    assert_model_finite,
    count_trainable_parameters,
    model_state_sha256,
    set_deterministic_cpu,
)


METER_V4_1_NUMERATOR_SPECIALIST: Final[str] = "meter-v4-1-learned-numerator-specialist-v1"
NUMERATOR_CLASSES_V4_1: Final[tuple[str, ...]] = ("2", "3", "4")
EXPECTED_V4_0_REPOSITORY_BINDING: Final[str] = (
    "8641fc45ae0e5613d280eee8af12ac105c765c313190660c88479e38bf6eff48"
)
EXPECTED_V4_0_RESULT_SHA256: Final[str] = (
    "422e79d7f71a1d2228e1392160d6ef4444521d8796f3c3b8fb6cd0a226c9060a"
)
EXPECTED_V4_0_RESULT_SCHEMA: Final[str] = (
    "st-omr-meter-v4-0-numerator-representation-audit-result-v1"
)
EXPECTED_V4_0_EXPERIMENT: Final[str] = "meter-v4-0-numerator-representation-audit-v1"
EXPECTED_PARAMETER_COUNT_V4_1: Final[int] = 9_571
MASTER_SEED_V4_1: Final[int] = 812_041
_MAX_JSON_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_CROP_BYTES: Final[int] = 64 * 1024


class MeterV4_1Error(RuntimeError):
    """Raised when V4-1 provenance, training or metric state fails closed."""


def _fail(message: str) -> None:
    raise MeterV4_1Error(message)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _hex64(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        _fail(f"{name} must be canonical lowercase SHA-256")
    return value


def _finite(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class NumeratorSpecialistConfigV4_1:
    output_size: int = 64
    conv1_channels: int = 8
    conv2_channels: int = 16
    pooled_size: int = 4
    hidden_dim: int = 32
    epochs: int = 160
    learning_rate_micros: int = 1_000
    weight_decay_micros: int = 100
    grad_clip_milli: int = 1_000
    shift_pixels: int = 2
    folds: int = 3
    master_seed: int = MASTER_SEED_V4_1

    def __post_init__(self) -> None:
        frozen = {
            "output_size": 64,
            "conv1_channels": 8,
            "conv2_channels": 16,
            "pooled_size": 4,
            "hidden_dim": 32,
            "epochs": 160,
            "learning_rate_micros": 1_000,
            "weight_decay_micros": 100,
            "grad_clip_milli": 1_000,
            "shift_pixels": 2,
            "folds": 3,
            "master_seed": MASTER_SEED_V4_1,
        }
        for name, expected in frozen.items():
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value != expected:
                raise ValueError(f"{name} is frozen to {expected!r}")


FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1: Final[NumeratorSpecialistConfigV4_1] = (
    NumeratorSpecialistConfigV4_1()
)


@dataclass(frozen=True, slots=True)
class NumeratorRecordV4_1:
    record_id: str
    family_id: str
    numerator_class: str
    fold: int
    crop_png_sha256: str

    def __post_init__(self) -> None:
        _hex64("record_id", self.record_id)
        _hex64("crop_png_sha256", self.crop_png_sha256)
        if not isinstance(self.family_id, str) or not self.family_id or len(self.family_id) > 256:
            raise ValueError("family_id must be bounded non-empty string")
        if self.numerator_class not in NUMERATOR_CLASSES_V4_1:
            raise ValueError("numerator_class must be 2, 3, or 4")
        if not isinstance(self.fold, int) or isinstance(self.fold, bool) or self.fold not in (0, 1, 2):
            raise ValueError("fold must be 0, 1, or 2")

    @property
    def class_index(self) -> int:
        return NUMERATOR_CLASSES_V4_1.index(self.numerator_class)


@dataclass(frozen=True, slots=True)
class PredictionV4_1:
    record_id: str
    family_id: str
    fold: int
    true_class: str
    predicted_class: str
    logits: tuple[float, float, float]
    probabilities: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SummaryV4_1:
    record_count: int
    accuracy: float
    macro_f1: float
    per_class_recall: dict[str, float]
    confusion: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class FoldTrainingResultV4_1:
    fold: int
    train_record_ids: tuple[str, ...]
    holdout_record_ids: tuple[str, ...]
    final_loss: float
    model_state_sha256: str
    optimizer_steps: int
    predictions: tuple[PredictionV4_1, ...]


@dataclass(frozen=True, slots=True)
class DecisionV4_1:
    decision: str
    strong_signal: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VerifiedParentV4_1:
    root: Path
    result_sha256: str
    repository_binding: str
    records: tuple[NumeratorRecordV4_1, ...]
    result: dict[str, object]


class NumeratorSpecialistV4_1(nn.Module):
    """Tiny fixed three-class CNN over one 64x64 numerator crop."""

    def __init__(
        self,
        config: NumeratorSpecialistConfigV4_1 = FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1,
    ) -> None:
        super().__init__()
        if config != FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1:
            raise ValueError("V4-1 model requires the frozen specialist config")
        self.config = config
        self.features = nn.Sequential(
            nn.Conv2d(1, config.conv1_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(config.conv1_channels, config.conv2_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((config.pooled_size, config.pooled_size)),
        )
        flattened = config.conv2_channels * config.pooled_size * config.pooled_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, len(NUMERATOR_CLASSES_V4_1)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            _fail("V4-1 images must be torch.Tensor")
        if images.dtype != torch.float32 or images.ndim != 4:
            _fail("V4-1 images must be float32 [batch,1,64,64]")
        if images.shape[1:] != (1, 64, 64) or images.shape[0] < 1:
            _fail("V4-1 image tensor shape differs from frozen 1x64x64 input")
        assert_finite_tensor("V4-1 input", images)
        if bool((images < 0).any()) or bool((images > 1).any()):
            _fail("V4-1 image tensor must remain in [0,1]")
        logits = self.classifier(self.features(images))
        assert_finite_tensor("V4-1 logits", logits)
        return logits


def config_fingerprint_v4_1(
    config: NumeratorSpecialistConfigV4_1 = FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1,
) -> str:
    if config != FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1:
        _fail("V4-1 config fingerprint requires frozen config")
    payload = {
        "version": METER_V4_1_NUMERATOR_SPECIALIST,
        "torch_version": TORCH_PINNED_VERSION,
        "classes": list(NUMERATOR_CLASSES_V4_1),
        "parent_result_sha256": EXPECTED_V4_0_RESULT_SHA256,
        "config": asdict(config),
        "augmentation": {"x_shifts": [-2, 0, 2], "y_shifts": [-2, 0, 2]},
    }
    return sha256(_canonical_json(payload)).hexdigest()


def build_model_v4_1(
    fold: int,
    config: NumeratorSpecialistConfigV4_1 = FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1,
) -> NumeratorSpecialistV4_1:
    if fold not in (0, 1, 2):
        _fail("V4-1 fold must be 0, 1, or 2")
    set_deterministic_cpu(config.master_seed + fold)
    model = NumeratorSpecialistV4_1(config).cpu()
    parameter_count = count_trainable_parameters(model)
    if parameter_count != EXPECTED_PARAMETER_COUNT_V4_1:
        _fail(
            f"V4-1 trainable parameter count changed: expected {EXPECTED_PARAMETER_COUNT_V4_1}, got {parameter_count}"
        )
    assert_model_finite(model)
    return model


def _read_bounded(path: Path, *, maximum: int, name: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        _fail(f"{name} must be a regular file")
    size = path.stat().st_size
    if size <= 0 or size > maximum:
        _fail(f"{name} size is outside V4-1 bounds")
    raw = path.read_bytes()
    if len(raw) != size:
        _fail(f"{name} short read")
    return raw


def verify_parent_artifact_v4_1(root: str | Path) -> VerifiedParentV4_1:
    """Verify the exact accepted V4-0 result and crop artifact before training."""
    root_path = Path(root)
    if not root_path.is_dir() or root_path.is_symlink():
        _fail("V4-1 parent root must be an existing regular directory")
    result_raw = _read_bounded(root_path / "result.json", maximum=_MAX_JSON_BYTES, name="V4-0 result.json")
    complete_raw = _read_bounded(root_path / "COMPLETE", maximum=256, name="V4-0 COMPLETE")
    result_sha = sha256(result_raw).hexdigest()
    if result_sha != EXPECTED_V4_0_RESULT_SHA256:
        _fail("V4-0 result SHA differs from accepted parent evidence")
    expected_complete = f"{result_sha}  result.json\n".encode("ascii")
    if complete_raw != expected_complete:
        _fail("V4-0 COMPLETE receipt does not bind exact result.json")
    try:
        result = json.loads(result_raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeterV4_1Error("V4-0 result.json is not canonical ASCII JSON") from exc
    if not isinstance(result, dict):
        _fail("V4-0 result root must be object")
    if result.get("schema") != EXPECTED_V4_0_RESULT_SCHEMA:
        _fail("V4-0 result schema changed")
    if result.get("experiment") != EXPECTED_V4_0_EXPERIMENT:
        _fail("V4-0 experiment identity changed")
    repository_binding = _hex64("V4-0 repository binding", result.get("repository_sha"))
    if repository_binding != EXPECTED_V4_0_REPOSITORY_BINDING:
        _fail("V4-0 repository binding differs from accepted evidence")
    decision = result.get("decision")
    if not isinstance(decision, Mapping) or decision.get("name") != "REPRESENTATION_SIGNAL_STRONG" or decision.get("strong_signal") is not True:
        _fail("V4-0 parent did not record the accepted strong representation signal")
    if decision.get("reasons") != []:
        _fail("V4-0 accepted parent unexpectedly contains decision reasons")
    surface = result.get("audit_surface")
    if not isinstance(surface, Mapping):
        _fail("V4-0 audit surface missing")
    required_safety = {
        "teacher_positive_train_records": 27,
        "teacher_positive_validation_records": 9,
        "teacher_adaptation_validation_evaluated": False,
        "teacher_adaptation_validation_images_decoded": 0,
        "none_tasks_used": 0,
        "d10_opened": False,
        "test_opened": False,
    }
    for name, expected in required_safety.items():
        if surface.get(name) != expected:
            _fail(f"V4-0 safety field {name} differs from accepted parent")
    if result.get("optimizer_steps") != 0 or result.get("d11_checkpoint_loaded") is not False or result.get("v3_checkpoint_loaded") is not False:
        _fail("V4-0 parent unexpectedly contains learned/checkpoint state")
    if result.get("runtime_connected") is not False or result.get("resolver_connected") is not False or result.get("production_promotion_authorized") is not False:
        _fail("V4-0 parent crossed runtime/production boundary")

    rows = result.get("crop_records")
    if not isinstance(rows, list) or len(rows) != 27:
        _fail("V4-0 parent must contain exactly 27 crop records")
    records: list[NumeratorRecordV4_1] = []
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            _fail(f"V4-0 crop record {index} is not object")
        records.append(
            NumeratorRecordV4_1(
                record_id=str(raw_row.get("record_id")),
                family_id=str(raw_row.get("family_id")),
                numerator_class=str(raw_row.get("numerator_class")),
                fold=int(raw_row.get("fold")) if isinstance(raw_row.get("fold"), int) and not isinstance(raw_row.get("fold"), bool) else -1,
                crop_png_sha256=str(raw_row.get("crop_png_sha256")),
            )
        )
    if len({row.record_id for row in records}) != 27 or len({row.family_id for row in records}) != 27:
        _fail("V4-1 parent records require unique record and family identities")
    if Counter(row.numerator_class for row in records) != Counter({"2": 9, "3": 9, "4": 9}):
        _fail("V4-1 parent classes must remain balanced 9/9/9")
    for fold in (0, 1, 2):
        if Counter(row.numerator_class for row in records if row.fold == fold) != Counter({"2": 3, "3": 3, "4": 3}):
            _fail("V4-1 inherited fold must contain exactly 3/3/3 held-out classes")

    crop_dir = root_path / "crops"
    if not crop_dir.is_dir() or crop_dir.is_symlink():
        _fail("V4-0 crops directory missing")
    actual_names = {path.name for path in crop_dir.iterdir() if path.is_file() and not path.is_symlink()}
    expected_names = {f"{row.record_id}.png" for row in records}
    if actual_names != expected_names:
        _fail("V4-0 crop file set differs from exact 27 parent records")
    for row in records:
        raw = _read_bounded(crop_dir / f"{row.record_id}.png", maximum=_MAX_CROP_BYTES, name="V4-0 crop PNG")
        if sha256(raw).hexdigest() != row.crop_png_sha256:
            _fail(f"V4-0 crop SHA mismatch for record {row.record_id}")
        try:
            with Image.open(BytesIO(raw)) as opened:
                opened.load()
                if opened.format != "PNG" or opened.mode != "L" or opened.size != (64, 64):
                    _fail("V4-0 parent crop must remain gray8 PNG 64x64")
        except UnidentifiedImageError as exc:
            raise MeterV4_1Error("V4-0 parent crop is not decodable PNG") from exc

    records.sort(key=lambda row: (row.fold, row.numerator_class, row.family_id, row.record_id))
    return VerifiedParentV4_1(
        root=root_path,
        result_sha256=result_sha,
        repository_binding=repository_binding,
        records=tuple(records),
        result=dict(result),
    )


def load_crop_tensor_v4_1(parent: VerifiedParentV4_1, record: NumeratorRecordV4_1) -> torch.Tensor:
    raw = _read_bounded(
        parent.root / "crops" / f"{record.record_id}.png",
        maximum=_MAX_CROP_BYTES,
        name="V4-1 crop PNG",
    )
    if sha256(raw).hexdigest() != record.crop_png_sha256:
        _fail("V4-1 crop changed after parent verification")
    with Image.open(BytesIO(raw)) as opened:
        opened.load()
        if opened.format != "PNG" or opened.mode != "L" or opened.size != (64, 64):
            _fail("V4-1 crop must remain gray8 PNG 64x64")
        values = torch.tensor(list(opened.tobytes()), dtype=torch.float32).reshape(1, 64, 64)
    tensor = (255.0 - values) / 255.0
    assert_finite_tensor("V4-1 crop tensor", tensor)
    return tensor


def translate_ink_v4_1(image: torch.Tensor, *, dx: int, dy: int) -> torch.Tensor:
    """Translate one [1,64,64] ink tensor without wrapping; exposed pixels are background zero."""
    if not isinstance(image, torch.Tensor) or image.dtype != torch.float32 or image.shape != (1, 64, 64):
        _fail("V4-1 translation input must be float32 [1,64,64]")
    if dx not in (-2, 0, 2) or dy not in (-2, 0, 2):
        _fail("V4-1 translation is frozen to shifts -2/0/+2")
    assert_finite_tensor("V4-1 translation input", image)
    output = torch.zeros_like(image)
    src_x0 = max(0, -dx)
    src_x1 = min(64, 64 - dx)
    src_y0 = max(0, -dy)
    src_y1 = min(64, 64 - dy)
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    output[:, dst_y0:dst_y1, dst_x0:dst_x1] = image[:, src_y0:src_y1, src_x0:src_x1]
    return output


def build_augmented_train_batch_v4_1(
    records: Sequence[NumeratorRecordV4_1],
    crops_by_record_id: Mapping[str, torch.Tensor],
    *,
    heldout_fold: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    if heldout_fold not in (0, 1, 2):
        _fail("V4-1 heldout fold must be 0, 1, or 2")
    train_records = sorted(
        (row for row in records if row.fold != heldout_fold),
        key=lambda row: (row.numerator_class, row.family_id, row.record_id),
    )
    if len(train_records) != 18 or Counter(row.numerator_class for row in train_records) != Counter({"2": 6, "3": 6, "4": 6}):
        _fail("V4-1 each fold must train on exact balanced 18 records")
    if set(crops_by_record_id) != {row.record_id for row in records}:
        _fail("V4-1 crop tensor map must match exact parent record ids")
    images: list[torch.Tensor] = []
    labels: list[int] = []
    origins: list[str] = []
    shifts = (-2, 0, 2)
    for row in train_records:
        crop = crops_by_record_id[row.record_id]
        for dy in shifts:
            for dx in shifts:
                images.append(translate_ink_v4_1(crop, dx=dx, dy=dy))
                labels.append(row.class_index)
                origins.append(row.record_id)
    if len(images) != 162:
        _fail("V4-1 deterministic augmentation must yield exactly 162 training views")
    batch = torch.stack(images, dim=0)
    target = torch.tensor(labels, dtype=torch.long)
    return batch, target, tuple(origins)


def train_fold_v4_1(
    records: Sequence[NumeratorRecordV4_1],
    crops_by_record_id: Mapping[str, torch.Tensor],
    *,
    heldout_fold: int,
    config: NumeratorSpecialistConfigV4_1 = FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1,
) -> FoldTrainingResultV4_1:
    if config != FROZEN_NUMERATOR_SPECIALIST_CONFIG_V4_1:
        _fail("V4-1 training requires frozen config")
    model = build_model_v4_1(heldout_fold, config)
    batch, target, _origins = build_augmented_train_batch_v4_1(
        records, crops_by_record_id, heldout_fold=heldout_fold
    )
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
        assert_finite_tensor("V4-1 fold loss", loss)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_clip_milli / 1000.0
        )
        if not math.isfinite(float(grad_norm)):
            _fail("V4-1 gradient norm is non-finite")
        optimizer.step()
        assert_model_finite(model)
        final_loss = float(loss.detach().cpu().item())
    if not math.isfinite(final_loss):
        _fail("V4-1 final loss is non-finite")

    holdout = sorted(
        (row for row in records if row.fold == heldout_fold),
        key=lambda row: (row.numerator_class, row.family_id, row.record_id),
    )
    if len(holdout) != 9 or Counter(row.numerator_class for row in holdout) != Counter({"2": 3, "3": 3, "4": 3}):
        _fail("V4-1 each fold must evaluate exact balanced 9 held-out families")
    model.eval()
    predictions: list[PredictionV4_1] = []
    with torch.no_grad():
        for row in holdout:
            image = crops_by_record_id[row.record_id].unsqueeze(0)
            logits = model(image)[0]
            probabilities = torch.softmax(logits, dim=0)
            assert_finite_tensor("V4-1 probabilities", probabilities)
            pred_index = int(torch.argmax(probabilities).item())
            predictions.append(
                PredictionV4_1(
                    record_id=row.record_id,
                    family_id=row.family_id,
                    fold=heldout_fold,
                    true_class=row.numerator_class,
                    predicted_class=NUMERATOR_CLASSES_V4_1[pred_index],
                    logits=tuple(float(value) for value in logits.cpu().tolist()),
                    probabilities=tuple(float(value) for value in probabilities.cpu().tolist()),
                )
            )
    train_ids = tuple(sorted(row.record_id for row in records if row.fold != heldout_fold))
    holdout_ids = tuple(sorted(row.record_id for row in holdout))
    return FoldTrainingResultV4_1(
        fold=heldout_fold,
        train_record_ids=train_ids,
        holdout_record_ids=holdout_ids,
        final_loss=final_loss,
        model_state_sha256=model_state_sha256(model),
        optimizer_steps=config.epochs,
        predictions=tuple(predictions),
    )


def summarize_predictions_v4_1(predictions: Sequence[PredictionV4_1]) -> SummaryV4_1:
    if len(predictions) != 27:
        _fail("V4-1 OOF summary requires exactly 27 predictions")
    if len({row.record_id for row in predictions}) != 27 or len({row.family_id for row in predictions}) != 27:
        _fail("V4-1 OOF predictions require unique record/family ids")
    confusion = [[0, 0, 0] for _ in range(3)]
    for row in predictions:
        if row.true_class not in NUMERATOR_CLASSES_V4_1 or row.predicted_class not in NUMERATOR_CLASSES_V4_1:
            _fail("V4-1 prediction class is outside 2/3/4")
        true_index = NUMERATOR_CLASSES_V4_1.index(row.true_class)
        pred_index = NUMERATOR_CLASSES_V4_1.index(row.predicted_class)
        confusion[true_index][pred_index] += 1
    if any(sum(row) != 9 for row in confusion):
        _fail("V4-1 OOF truth cardinality must remain 9 per class")
    correct = sum(confusion[index][index] for index in range(3))
    accuracy = correct / 27.0
    recalls: dict[str, float] = {}
    f1_values: list[float] = []
    for index, class_name in enumerate(NUMERATOR_CLASSES_V4_1):
        tp = confusion[index][index]
        fn = sum(confusion[index]) - tp
        fp = sum(confusion[row][index] for row in range(3)) - tp
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls[class_name] = recall
        f1_values.append(f1)
    macro_f1 = sum(f1_values) / 3.0
    for name, value in {"accuracy": accuracy, "macro_f1": macro_f1, **recalls}.items():
        _finite(f"V4-1 metric {name}", value)
    return SummaryV4_1(
        record_count=27,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_recall=recalls,
        confusion=tuple(tuple(value for value in row) for row in confusion),
    )


def decision_v4_1(summary: SummaryV4_1) -> DecisionV4_1:
    reasons: list[str] = []
    if summary.record_count != 27:
        reasons.append("OOF_RECORD_COUNT_NOT_27")
    if summary.accuracy + 1e-12 < 26.0 / 27.0:
        reasons.append("OOF_ACCURACY_BELOW_26_OF_27")
    if summary.macro_f1 + 1e-12 < 0.95:
        reasons.append("OOF_MACRO_F1_BELOW_95_PERCENT")
    for class_name in NUMERATOR_CLASSES_V4_1:
        if summary.per_class_recall.get(class_name, 0.0) + 1e-12 < 8.0 / 9.0:
            reasons.append(f"OOF_{class_name}_RECALL_BELOW_8_OF_9")
    strong = not reasons
    return DecisionV4_1(
        decision=("LEARNED_NUMERATOR_SIGNAL_STRONG" if strong else "LEARNED_NUMERATOR_SIGNAL_INSUFFICIENT"),
        strong_signal=strong,
        reasons=tuple(reasons),
    )
