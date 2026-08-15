"""Stage 7-D11 local barline + meter refiner training.

D11 consumes only the accepted Stage 7-D10 TRAIN/VALIDATION ROI derivative
bundle. The sealed TEST split is forbidden. The accepted D7 Structure core
is never loaded by this module and therefore cannot be mutated: only the new
barline and meter refiner weights are trainable.
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
import random
import re
from typing import Callable, Final

from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F

from .stage7c_execution import verify_authoritative_repository, verify_stage7c_runtime
from .stage7d10_local_roi_derivatives import (
    STAGE7D10_LABEL_SCHEMA,
    STAGE7D10_VERSION,
    Stage7D10Receipt,
    verify_stage7d10_derivatives,
)
from .stage7d9_structure_refinement_contract import (
    BARLINE_ROI,
    D9_ACCEPTANCE,
    EXPECTED_D7_STRUCTURE_STATE_SHA256,
    METER_CLASSES,
    METER_ROI,
    stage7d9_contract_fingerprint,
)
from .training_model import (
    TrainingRuntimeError,
    assert_finite_tensor,
    assert_model_finite,
    count_trainable_parameters,
    model_state_sha256,
    set_deterministic_cpu,
)

STAGE7D11_VERSION: Final[str] = "stage7d11-barline-meter-training-v1"
STAGE7D11_METRICS_SCHEMA: Final[str] = "stage7d11-refiner-metrics-v1"
STAGE7D11_VERIFICATION_SCHEMA: Final[str] = "stage7d11-refiner-verification-v1"
BARLINE_MODEL_VERSION: Final[str] = "stage7d11-barline-refiner-v1"
METER_MODEL_VERSION: Final[str] = "stage7d11-meter-refiner-v1"
EXPECTED_D10_REPOSITORY_SHA: Final[str] = "562c8fcfabf1b41573f1ef591d88ae65335ce16a"
EXPECTED_D10_ROI_RECORDS: Final[int] = 22_128
EXPECTED_D10_KIND_COUNTS: Final[dict[str, int]] = {"barline": 11_064, "meter": 11_064}
EXPECTED_D10_SPLIT_COUNTS: Final[dict[str, int]] = {"train": 19_680, "validation": 2_448}
EXPECTED_TASK_SPLIT_COUNTS: Final[dict[str, int]] = {"train": 9_840, "validation": 1_224}
BARLINE_MAX_PARAMETERS: Final[int] = 500_000
METER_MAX_PARAMETERS: Final[int] = 750_000
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_SPLITS = frozenset({"train", "validation"})
_ALLOWED_KINDS = frozenset({"barline", "meter"})
ProgressCallback = Callable[[str, Mapping[str, object]], None]


class Stage7D11TrainingError(RuntimeError):
    """Raised when D11 provenance, data, numeric state, or artifacts fail closed."""


def _fail(message: str) -> None:
    raise Stage7D11TrainingError(message)


def _canonical_json(payload: object) -> bytes:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Stage7D11TrainingError("payload is not canonical JSON serializable") from exc


def _sha256_hex(value: object, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        _fail(f"{name} must be canonical lowercase SHA-256")
    return value


def _read_canonical_json(path: Path, maximum: int, name: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{name} must be a regular file")
    size = path.stat().st_size
    if not 1 <= size <= maximum:
        _fail(f"{name} byte length is outside D11 bounds")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("ascii"), parse_constant=lambda token: _fail(f"non-finite constant in {name}: {token}"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D11TrainingError(f"{name} is not valid ASCII JSON") from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        _fail(f"{name} must be canonical JSON object bytes")
    return payload, raw


def _development_manifest_rows(rows: object) -> tuple[Mapping[str, object], ...]:
    """Reject TEST immediately after reading only ``split``."""
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        _fail("D10 manifest records must be a sequence")
    accepted: list[Mapping[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"D10 manifest record[{index}] must be an object")
        split = row.get("split")
        if split == "test":
            _fail("sealed TEST record reached Stage 7-D11")
        if split not in _ALLOWED_SPLITS:
            _fail("D11 record split must be train or validation")
        accepted.append(row)
    return tuple(accepted)


@dataclass(frozen=True, slots=True)
class Stage7D11TrainingConfig:
    batch_size: int = 32
    epochs: int = 8
    learning_rate_micros: int = 700
    weight_decay_micros: int = 100
    grad_clip_milli: int = 1000
    master_seed: int = 711_011
    barline_target_width_px: int = 3
    meter_bbox_loss_milli: int = 2000
    heartbeat_batches: int = 50
    optimizer: str = "adamw"
    barline_objective: str = "bce_plus_soft_dice_v1"
    meter_objective: str = "balanced_ce_plus_positive_smooth_l1_v1"
    checkpoint_selection: str = "min_validation_loss_per_refiner"

    def __post_init__(self) -> None:
        bounds = {
            "batch_size": (self.batch_size, 1, 64), "epochs": (self.epochs, 1, 32),
            "learning_rate_micros": (self.learning_rate_micros, 1, 100_000),
            "weight_decay_micros": (self.weight_decay_micros, 0, 100_000),
            "grad_clip_milli": (self.grad_clip_milli, 1, 100_000),
            "master_seed": (self.master_seed, 0, 2**63 - 1),
            "barline_target_width_px": (self.barline_target_width_px, 1, 9),
            "meter_bbox_loss_milli": (self.meter_bbox_loss_milli, 1, 10_000),
            "heartbeat_batches": (self.heartbeat_batches, 1, 10_000),
        }
        for name, (value, low, high) in bounds.items():
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ValueError(f"{name} is outside the Stage 7-D11 boundary")
        expected = {
            "optimizer": "adamw",
            "barline_objective": "bce_plus_soft_dice_v1",
            "meter_objective": "balanced_ce_plus_positive_smooth_l1_v1",
            "checkpoint_selection": "min_validation_loss_per_refiner",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} is frozen to {value!r}")


FROZEN_D11_CONFIG: Final[Stage7D11TrainingConfig] = Stage7D11TrainingConfig()


@dataclass(frozen=True, slots=True)
class D11Record:
    record_id: str
    kind: str
    split: str
    family_id: str
    source_sample_id: str
    measure_number: int
    image_path: Path
    image_sha256: str
    label_path: Path
    label_sha256: str


@dataclass(frozen=True, slots=True)
class BarlineMetrics:
    strict_dice: float
    tolerant_f1_2px: float


@dataclass(frozen=True, slots=True)
class MeterMetrics:
    macro_f1: float
    positive_localization_f1_2px: float


@dataclass(frozen=True, slots=True)
class RefinerTrainingResult:
    task: str
    untrained_validation_loss: float
    best_validation_loss: float
    best_epoch: int
    optimizer_steps: int
    state_sha256: str
    parameter_count: int
    model_fingerprint: str
    validation_metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class Stage7D11RunResult:
    run_id: str
    run_directory: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    metrics_path: Path
    metrics_sha256: str
    verification_path: Path
    verification_sha256: str
    barline: RefinerTrainingResult
    meter: RefinerTrainingResult
    acceptance_passed: bool


def stage7d11_profile_fingerprint(config: Stage7D11TrainingConfig = FROZEN_D11_CONFIG) -> str:
    if not isinstance(config, Stage7D11TrainingConfig):
        raise TypeError("config must be Stage7D11TrainingConfig")
    payload = {
        "version": STAGE7D11_VERSION,
        "barline_model_version": BARLINE_MODEL_VERSION,
        "meter_model_version": METER_MODEL_VERSION,
        "config": asdict(config),
        "d9_contract_fingerprint": stage7d9_contract_fingerprint(),
        "d10_repository_sha": EXPECTED_D10_REPOSITORY_SHA,
        "accepted_d7_structure_state_sha256": EXPECTED_D7_STRUCTURE_STATE_SHA256,
        "barline_roi": asdict(BARLINE_ROI), "meter_roi": asdict(METER_ROI),
        "meter_classes": METER_CLASSES, "acceptance": asdict(D9_ACCEPTANCE),
        "split_policy": "train-new-refiners-validation-readonly-test-forbidden",
        "core_policy": "accepted-d7-structure-core-not-loaded-and-frozen",
    }
    return sha256(_canonical_json(payload)).hexdigest()


class BarlineRefiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(24, 32, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 24, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(24, 12, 3, padding=1), nn.ReLU(), nn.Conv2d(12, 1, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _validate_images(images, BARLINE_ROI.output_height, BARLINE_ROI.output_width, "barline")
        logits = self.decoder(self.encoder(images)); assert_finite_tensor("D11 barline logits", logits); return logits


class MeterRefiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 8))
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(24 * 6 * 8, 64), nn.ReLU())
        self.classifier = nn.Linear(64, len(METER_CLASSES)); self.bbox = nn.Linear(64, 4)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_images(images, METER_ROI.output_height, METER_ROI.output_width, "meter")
        features = self.projection(self.pool(self.encoder(images))); class_logits = self.classifier(features)
        raw = torch.sigmoid(self.bbox(features))
        boxes = torch.stack((torch.minimum(raw[:, 0], raw[:, 2]), torch.minimum(raw[:, 1], raw[:, 3]), torch.maximum(raw[:, 0], raw[:, 2]), torch.maximum(raw[:, 1], raw[:, 3])), dim=1)
        assert_finite_tensor("D11 meter class logits", class_logits); assert_finite_tensor("D11 meter boxes", boxes)
        return class_logits, boxes


def _validate_images(images: object, height: int, width: int, task: str) -> torch.Tensor:
    if not isinstance(images, torch.Tensor) or images.dtype != torch.float32 or images.ndim != 4:
        raise TrainingRuntimeError(f"D11 {task} images must be float32 [B,1,H,W]")
    if tuple(images.shape[1:]) != (1, height, width):
        raise TrainingRuntimeError(f"D11 {task} image shape differs from frozen ROI")
    assert_finite_tensor(f"D11 {task} images", images); return images


def _build_seed(task: str, config: Stage7D11TrainingConfig) -> int:
    if task == "barline": return config.master_seed + 101
    if task == "meter": return config.master_seed + 202
    raise ValueError("D11 task must be barline or meter")


def build_barline_refiner(config: Stage7D11TrainingConfig = FROZEN_D11_CONFIG) -> BarlineRefiner:
    set_deterministic_cpu(_build_seed("barline", config)); model = BarlineRefiner().cpu(); count = count_trainable_parameters(model)
    if not 0 < count <= BARLINE_MAX_PARAMETERS: _fail("barline refiner exceeds frozen D9 parameter budget")
    assert_model_finite(model); return model


def build_meter_refiner(config: Stage7D11TrainingConfig = FROZEN_D11_CONFIG) -> MeterRefiner:
    set_deterministic_cpu(_build_seed("meter", config)); model = MeterRefiner().cpu(); count = count_trainable_parameters(model)
    if not 0 < count <= METER_MAX_PARAMETERS: _fail("meter refiner exceeds frozen D9 parameter budget")
    assert_model_finite(model); return model


def refiner_model_fingerprint(task: str, config: Stage7D11TrainingConfig = FROZEN_D11_CONFIG) -> str:
    if task not in _ALLOWED_KINDS: raise ValueError("D11 task must be barline or meter")
    payload = {"task": task, "model_version": BARLINE_MODEL_VERSION if task == "barline" else METER_MODEL_VERSION, "profile_fingerprint": stage7d11_profile_fingerprint(config), "architecture": "local-conv8-16-24-32-up24-up12-seg-v1" if task == "barline" else "local-conv8-16-24-24-pool6x8-fc64-class4-box4-v1"}
    return sha256(_canonical_json(payload)).hexdigest()


def _tensor_from_l_image(image: Image.Image) -> torch.Tensor:
    if image.mode != "L": _fail("D11 tensor conversion requires grayscale L image")
    raw = bytearray(image.tobytes()); tensor = torch.frombuffer(raw, dtype=torch.uint8).clone().reshape(image.height, image.width)
    return 1.0 - tensor.to(dtype=torch.float32) / 255.0


def _finite_xy(point: object, name: str) -> tuple[float, float]:
    if not isinstance(point, Mapping): _fail(f"{name} must be a mapping")
    try: x, y = float(point["x"]), float(point["y"])
    except (KeyError, TypeError, ValueError) as exc: raise Stage7D11TrainingError(f"{name} is malformed") from exc
    if not math.isfinite(x) or not math.isfinite(y): _fail(f"{name} must be finite")
    return x, y


def barline_target_mask(target: Mapping[str, object], config: Stage7D11TrainingConfig = FROZEN_D11_CONFIG) -> torch.Tensor:
    line = target.get("barline_segment")
    if not isinstance(line, Mapping): _fail("D11 barline target is missing")
    x0, y0 = _finite_xy(line.get("start"), "barline.start"); x1, y1 = _finite_xy(line.get("end"), "barline.end")
    if not (0 <= x0 <= BARLINE_ROI.output_width and 0 <= x1 <= BARLINE_ROI.output_width and 0 <= y0 <= BARLINE_ROI.output_height and 0 <= y1 <= BARLINE_ROI.output_height): _fail("barline target lies outside ROI")
    canvas = Image.new("L", (BARLINE_ROI.output_width, BARLINE_ROI.output_height), 0); ImageDraw.Draw(canvas).line((x0, y0, x1, y1), fill=255, width=config.barline_target_width_px)
    return _tensor_from_l_image(Image.eval(canvas, lambda value: 255 - value)).unsqueeze(0)


def meter_target(target: Mapping[str, object]) -> tuple[int, torch.Tensor, bool]:
    meter_class = target.get("meter_class")
    if meter_class not in METER_CLASSES: _fail("D11 meter target class is outside frozen D9 classes")
    class_index = METER_CLASSES.index(str(meter_class)); bbox = target.get("meter_bbox")
    if meter_class == "none":
        if bbox is not None: _fail("none meter target must not carry bbox")
        return class_index, torch.zeros(4, dtype=torch.float32), False
    if not isinstance(bbox, Mapping): _fail("visible meter target requires bbox")
    try: values = (float(bbox["x_min"]) / METER_ROI.output_width, float(bbox["y_min"]) / METER_ROI.output_height, float(bbox["x_max"]) / METER_ROI.output_width, float(bbox["y_max"]) / METER_ROI.output_height)
    except (KeyError, TypeError, ValueError) as exc: raise Stage7D11TrainingError("meter bbox is malformed") from exc
    x0, y0, x1, y1 = values
    if any(not math.isfinite(v) for v in values) or not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1): _fail("meter bbox lies outside normalized ROI")
    return class_index, torch.tensor(values, dtype=torch.float32), True


def barline_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape or logits.dtype != torch.float32 or targets.dtype != torch.float32: raise TrainingRuntimeError("D11 barline logits/targets must be same-shape float32")
    assert_finite_tensor("D11 barline logits", logits); assert_finite_tensor("D11 barline targets", targets)
    if bool((targets < 0).any()) or bool((targets > 1).any()): raise TrainingRuntimeError("D11 barline targets must be in [0,1]")
    bce = F.binary_cross_entropy_with_logits(logits, targets); probability = torch.sigmoid(logits)
    intersection = (probability * targets).sum(dim=(0, 2, 3)); denominator = probability.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3)); dice = (2 * intersection + 1.0) / (denominator + 1.0)
    result = bce + (1.0 - dice.mean()); assert_finite_tensor("D11 barline loss", result); return result


def meter_loss(class_logits: torch.Tensor, boxes: torch.Tensor, class_targets: torch.Tensor, box_targets: torch.Tensor, positive_mask: torch.Tensor, class_weights: torch.Tensor, config: Stage7D11TrainingConfig = FROZEN_D11_CONFIG) -> torch.Tensor:
    if class_logits.dtype != torch.float32 or class_logits.ndim != 2 or class_logits.shape[1] != len(METER_CLASSES): raise TrainingRuntimeError("D11 meter class logits shape is invalid")
    batch = class_logits.shape[0]
    if boxes.shape != (batch, 4) or boxes.dtype != torch.float32 or class_targets.shape != (batch,) or class_targets.dtype != torch.long or box_targets.shape != (batch, 4) or box_targets.dtype != torch.float32 or positive_mask.shape != (batch,) or positive_mask.dtype != torch.bool or class_weights.shape != (len(METER_CLASSES),) or class_weights.dtype != torch.float32: raise TrainingRuntimeError("D11 meter target/weight shape is invalid")
    for name, tensor in (("class_logits", class_logits), ("boxes", boxes), ("box_targets", box_targets), ("class_weights", class_weights)): assert_finite_tensor(f"D11 meter {name}", tensor)
    if bool((class_targets < 0).any()) or bool((class_targets >= len(METER_CLASSES)).any()): raise TrainingRuntimeError("D11 meter class target is out of range")
    if bool((boxes < 0).any()) or bool((boxes > 1).any()) or bool((box_targets < 0).any()) or bool((box_targets > 1).any()): raise TrainingRuntimeError("D11 meter boxes must stay in [0,1]")
    classification = F.cross_entropy(class_logits, class_targets, weight=class_weights); bbox = F.smooth_l1_loss(boxes[positive_mask], box_targets[positive_mask]) if bool(positive_mask.any()) else boxes.sum() * 0.0
    result = classification + (config.meter_bbox_loss_milli / 1000.0) * bbox; assert_finite_tensor("D11 meter loss", result); return result


def _binary_dice(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred, true = prediction.to(torch.bool), target.to(torch.bool); intersection = int((pred & true).sum().item()); denominator = int(pred.sum().item() + true.sum().item()); return 1.0 if denominator == 0 else 2.0 * intersection / denominator


def _tolerant_f1(prediction: torch.Tensor, target: torch.Tensor, radius: int = 2) -> float:
    pred, true = prediction.to(torch.float32), target.to(torch.float32)
    if pred.ndim != 4 or true.ndim != 4 or pred.shape != true.shape: raise TrainingRuntimeError("D11 tolerant F1 requires equal [B,1,H,W] masks")
    kernel = 2 * radius + 1; pred_d = F.max_pool2d(pred, kernel, stride=1, padding=radius) > 0; true_d = F.max_pool2d(true, kernel, stride=1, padding=radius) > 0; pred_b, true_b = pred > 0, true > 0
    pc, tc = int(pred_b.sum().item()), int(true_b.sum().item())
    if pc == 0 and tc == 0: return 1.0
    precision = 0.0 if pc == 0 else float((pred_b & true_d).sum().item()) / pc; recall = 0.0 if tc == 0 else float((true_b & pred_d).sum().item()) / tc
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def acceptance_from_metrics(barline: BarlineMetrics, meter: MeterMetrics) -> bool:
    return barline.strict_dice >= D9_ACCEPTANCE.barline_min_strict_dice_milli / 1000.0 and barline.tolerant_f1_2px >= D9_ACCEPTANCE.barline_min_tolerant_f1_2px_milli / 1000.0 and meter.macro_f1 >= D9_ACCEPTANCE.meter_min_macro_f1_milli / 1000.0 and meter.positive_localization_f1_2px >= D9_ACCEPTANCE.meter_min_positive_localization_f1_2px_milli / 1000.0


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]: return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _load_d11_label(record: D11Record) -> dict[str, object]:
    payload, raw = _read_canonical_json(record.label_path, 2 * 1024 * 1024, "D11 ROI label")
    if sha256(raw).hexdigest() != record.label_sha256: _fail("D11 ROI label SHA-256 mismatch")
    if payload.get("schema_version") != STAGE7D10_LABEL_SCHEMA or payload.get("stage7d10_version") != STAGE7D10_VERSION or payload.get("d9_contract_fingerprint") != stage7d9_contract_fingerprint(): _fail("D11 ROI label schema/version/fingerprint mismatch")
    if payload.get("record_id") != record.record_id or payload.get("kind") != record.kind or payload.get("split") != record.split or payload.get("measure_number") != record.measure_number or payload.get("roi_image_sha256") != record.image_sha256: _fail("D11 ROI label identity/binding mismatch")
    return payload


def _load_d11_image(record: D11Record) -> torch.Tensor:
    if record.image_path.is_symlink() or not record.image_path.is_file(): _fail("D11 ROI image must be regular file")
    raw = record.image_path.read_bytes()
    if sha256(raw).hexdigest() != record.image_sha256: _fail("D11 ROI image SHA-256 mismatch")
    expected = BARLINE_ROI if record.kind == "barline" else METER_ROI
    try:
        with Image.open(BytesIO(raw)) as opened:
            opened.load()
            if opened.format != "PNG" or opened.mode != "L" or opened.size != (expected.output_width, expected.output_height): _fail("D11 ROI format/mode/dimensions differ from frozen policy")
            return _tensor_from_l_image(opened).unsqueeze(0)
    except Stage7D11TrainingError: raise
    except Exception as exc: raise Stage7D11TrainingError("D11 ROI PNG cannot be decoded") from exc


def load_verified_stage7d11_records(d10_root: str | Path, *, expected_manifest_sha256: str, expected_artifact_binding_sha256: str) -> tuple[D11Record, ...]:
    expected_manifest = _sha256_hex(expected_manifest_sha256, "expected D10 manifest SHA-256"); expected_binding = _sha256_hex(expected_artifact_binding_sha256, "expected D10 artifact binding SHA-256"); root = Path(d10_root)
    receipt: Stage7D10Receipt = verify_stage7d10_derivatives(root, expected_authoritative_surface=True, expected_repository_sha=EXPECTED_D10_REPOSITORY_SHA, require_complete=True)
    checks = {"d9_contract_fingerprint": stage7d9_contract_fingerprint(), "manifest_sha256": expected_manifest, "artifact_binding_sha256": expected_binding, "roi_record_count": EXPECTED_D10_ROI_RECORDS, "split_counts": EXPECTED_D10_SPLIT_COUNTS, "kind_counts": EXPECTED_D10_KIND_COUNTS, "test_records": 0, "optimizer_steps": 0}
    for name, expected in checks.items():
        if getattr(receipt, name) != expected: _fail(f"accepted D10 receipt {name} mismatch")
    manifest, raw = _read_canonical_json(root / "manifest.json", 64 * 1024 * 1024, "D10 manifest")
    if sha256(raw).hexdigest() != expected_manifest: _fail("D10 manifest differs from explicitly accepted D10 identity")
    rows = _development_manifest_rows(manifest.get("records")); result: list[D11Record] = []; counts: Counter[tuple[str, str]] = Counter(); seen: set[str] = set()
    for row in rows:
        record_id = _sha256_hex(row.get("record_id"), "D10 record_id")
        if record_id in seen: _fail("duplicate D11 record_id")
        seen.add(record_id); kind, split = row.get("kind"), row.get("split")
        if kind not in _ALLOWED_KINDS or split not in _ALLOWED_SPLITS: _fail("D11 record kind/split is invalid")
        family_id, source_sample_id = row.get("family_id"), row.get("source_sample_id")
        if not isinstance(family_id, str) or not family_id or not isinstance(source_sample_id, str) or not source_sample_id: _fail("D11 record family/source identity is invalid")
        measure_number = row.get("measure_number")
        if not isinstance(measure_number, int) or isinstance(measure_number, bool) or measure_number <= 0: _fail("D11 measure_number must be positive integer")
        image_sha = _sha256_hex(row.get("image_sha256"), "D10 ROI image SHA-256"); label_sha = _sha256_hex(row.get("label_sha256"), "D10 ROI label SHA-256"); image_rel, label_rel = row.get("image_path"), row.get("label_path")
        if not isinstance(image_rel, str) or not isinstance(label_rel, str): _fail("D11 artifact paths must be strings")
        image_path, label_path = root / image_rel, root / label_rel
        if root.resolve() not in image_path.resolve().parents or root.resolve() not in label_path.resolve().parents: _fail("D11 artifact path escapes D10 root")
        result.append(D11Record(record_id, str(kind), str(split), family_id, source_sample_id, measure_number, image_path, image_sha, label_path, label_sha)); counts[(str(kind), str(split))] += 1
    expected_counts = {(kind, split): count for kind in ("barline", "meter") for split, count in EXPECTED_TASK_SPLIT_COUNTS.items()}
    if dict(counts) != expected_counts: _fail("D11 task/split cardinality differs from accepted D10 surface")
    return tuple(sorted(result, key=lambda item: (item.kind, item.split, item.record_id)))


def _stack_barline(records: Sequence[D11Record], config: Stage7D11TrainingConfig) -> tuple[torch.Tensor, torch.Tensor]:
    images, targets = [], []
    for record in records:
        if record.kind != "barline": _fail("meter record reached barline batch")
        label = _load_d11_label(record); target = label.get("target")
        if not isinstance(target, Mapping): _fail("D11 barline label target is missing")
        images.append(_load_d11_image(record)); targets.append(barline_target_mask(target, config))
    return torch.stack(images), torch.stack(targets)


def _stack_meter(records: Sequence[D11Record]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images, classes, boxes, positive = [], [], [], []
    for record in records:
        if record.kind != "meter": _fail("barline record reached meter batch")
        label = _load_d11_label(record); target = label.get("target")
        if not isinstance(target, Mapping): _fail("D11 meter label target is missing")
        ci, box, pos = meter_target(target); images.append(_load_d11_image(record)); classes.append(ci); boxes.append(box); positive.append(pos)
    return torch.stack(images), torch.tensor(classes, dtype=torch.long), torch.stack(boxes), torch.tensor(positive, dtype=torch.bool)


def _meter_class_weights(records: Sequence[D11Record]) -> torch.Tensor:
    counts = Counter()
    for record in records:
        target = _load_d11_label(record).get("target")
        if not isinstance(target, Mapping) or target.get("meter_class") not in METER_CLASSES: _fail("D11 meter class missing while building class weights")
        counts[str(target["meter_class"])] += 1
    if any(counts[name] <= 0 for name in METER_CLASSES): _fail("every D11 meter class must appear in TRAIN")
    total = float(sum(counts.values())); raw = torch.tensor([total / counts[name] for name in METER_CLASSES], dtype=torch.float32); raw = raw / raw.mean(); return torch.clamp(raw, min=0.25, max=4.0)


def _train_barline_batch(model: BarlineRefiner, images: torch.Tensor, targets: torch.Tensor, *, split: str, optimizer: torch.optim.Optimizer, config: Stage7D11TrainingConfig) -> float:
    if split != "train": raise TrainingRuntimeError("D11 optimizer step is allowed only for TRAIN")
    model.train(); optimizer.zero_grad(set_to_none=True); loss = barline_loss(model(images), targets); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_milli / 1000.0)
    for p in model.parameters():
        if p.grad is not None: assert_finite_tensor("D11 barline gradient", p.grad)
    optimizer.step(); assert_model_finite(model); return float(loss.detach().cpu().item())


def _train_meter_batch(model: MeterRefiner, images: torch.Tensor, classes: torch.Tensor, boxes: torch.Tensor, positive: torch.Tensor, class_weights: torch.Tensor, *, split: str, optimizer: torch.optim.Optimizer, config: Stage7D11TrainingConfig) -> float:
    if split != "train": raise TrainingRuntimeError("D11 optimizer step is allowed only for TRAIN")
    model.train(); optimizer.zero_grad(set_to_none=True); logits, predicted = model(images); loss = meter_loss(logits, predicted, classes, boxes, positive, class_weights, config); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_milli / 1000.0)
    for p in model.parameters():
        if p.grad is not None: assert_finite_tensor("D11 meter gradient", p.grad)
    optimizer.step(); assert_model_finite(model); return float(loss.detach().cpu().item())


def _evaluate_barline(model: BarlineRefiner, records: Sequence[D11Record], config: Stage7D11TrainingConfig) -> tuple[float, BarlineMetrics]:
    model.eval(); total, batches, preds, truths = 0.0, 0, [], []
    with torch.no_grad():
        for start in range(0, len(records), config.batch_size):
            images, targets = _stack_barline(records[start:start + config.batch_size], config); logits = model(images); total += float(barline_loss(logits, targets).item()); batches += 1; preds.append((torch.sigmoid(logits) >= 0.5).float()); truths.append(targets)
    if batches == 0: _fail("D11 barline validation produced no batches")
    pred, true = torch.cat(preds), torch.cat(truths); return total / batches, BarlineMetrics(_binary_dice(pred, true), _tolerant_f1(pred, true, 2))


def _bbox_mask(box: torch.Tensor, height: int, width: int) -> torch.Tensor:
    values = [float(v) for v in box.tolist()]; x0 = max(0, min(width - 1, int(math.floor(values[0] * width)))); y0 = max(0, min(height - 1, int(math.floor(values[1] * height)))); x1 = max(x0 + 1, min(width, int(math.ceil(values[2] * width)))); y1 = max(y0 + 1, min(height, int(math.ceil(values[3] * height)))); mask = torch.zeros((1, 1, height, width)); mask[:, :, y0:y1, x0:x1] = 1.0; return mask


def _evaluate_meter(model: MeterRefiner, records: Sequence[D11Record], class_weights: torch.Tensor, config: Stage7D11TrainingConfig) -> tuple[float, MeterMetrics]:
    model.eval(); total, batches = 0.0, 0; confusion = torch.zeros((4, 4), dtype=torch.int64); locations: list[float] = []
    with torch.no_grad():
        for start in range(0, len(records), config.batch_size):
            images, classes, boxes, positive = _stack_meter(records[start:start + config.batch_size]); logits, predicted_boxes = model(images); total += float(meter_loss(logits, predicted_boxes, classes, boxes, positive, class_weights, config).item()); batches += 1
            for true_class, predicted_class in zip(classes.tolist(), logits.argmax(1).tolist()): confusion[true_class, predicted_class] += 1
            for index in torch.nonzero(positive, as_tuple=False).flatten().tolist(): locations.append(_tolerant_f1(_bbox_mask(predicted_boxes[index], METER_ROI.output_height, METER_ROI.output_width), _bbox_mask(boxes[index], METER_ROI.output_height, METER_ROI.output_width), 2))
    if batches == 0 or not locations: _fail("D11 meter validation produced incomplete metrics")
    f1_values = []
    for i in range(4):
        tp = int(confusion[i, i]); fp = int(confusion[:, i].sum()) - tp; fn = int(confusion[i, :].sum()) - tp; precision = 0.0 if tp + fp == 0 else tp / (tp + fp); recall = 0.0 if tp + fn == 0 else tp / (tp + fn); f1_values.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return total / batches, MeterMetrics(sum(f1_values) / 4, sum(locations) / len(locations))


def _train_barline(records: Sequence[D11Record], config: Stage7D11TrainingConfig, progress: ProgressCallback | None):
    train = tuple(r for r in records if r.kind == "barline" and r.split == "train"); validation = tuple(r for r in records if r.kind == "barline" and r.split == "validation"); model = build_barline_refiner(config); optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate_micros / 1e6, weight_decay=config.weight_decay_micros / 1e6); untrained, _ = _evaluate_barline(model, validation, config); best, best_epoch, best_state, steps, history = untrained, 0, _clone_state(model), 0, []
    for epoch in range(1, config.epochs + 1):
        indices = list(range(len(train))); random.Random(_build_seed("barline", config) + epoch * 1_000_003).shuffle(indices); epoch_loss = 0.0; batch_count = 0
        for start in range(0, len(indices), config.batch_size):
            images, targets = _stack_barline([train[i] for i in indices[start:start + config.batch_size]], config); epoch_loss += _train_barline_batch(model, images, targets, split="train", optimizer=optimizer, config=config); steps += 1; batch_count += 1
            if progress and batch_count % config.heartbeat_batches == 0: progress("batch_progress", {"task": "barline", "epoch": epoch, "batch": batch_count, "optimizer_steps": steps})
        val_loss, metrics = _evaluate_barline(model, validation, config); history.append({"epoch": epoch, "train_loss": epoch_loss / batch_count, "validation_loss": val_loss, **asdict(metrics)})
        if val_loss < best: best, best_epoch, best_state = val_loss, epoch, _clone_state(model)
        if progress: progress("epoch_complete", {"task": "barline", "epoch": epoch, "epochs": config.epochs, "validation_loss": val_loss, **asdict(metrics)})
    if best_epoch <= 0: _fail("D11 barline refiner did not improve over untrained validation loss")
    model.load_state_dict(best_state, strict=True); final_loss, metrics = _evaluate_barline(model, validation, config)
    if abs(final_loss - best) > 1e-9: _fail("D11 barline restored best state does not reproduce best loss")
    if steps != math.ceil(9840 / config.batch_size) * config.epochs: _fail("D11 barline optimizer step count mismatch")
    return model, RefinerTrainingResult("barline", untrained, best, best_epoch, steps, model_state_sha256(model), count_trainable_parameters(model), refiner_model_fingerprint("barline", config), asdict(metrics)), history


def _train_meter(records: Sequence[D11Record], config: Stage7D11TrainingConfig, progress: ProgressCallback | None):
    train = tuple(r for r in records if r.kind == "meter" and r.split == "train"); validation = tuple(r for r in records if r.kind == "meter" and r.split == "validation"); weights = _meter_class_weights(train); model = build_meter_refiner(config); optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate_micros / 1e6, weight_decay=config.weight_decay_micros / 1e6); untrained, _ = _evaluate_meter(model, validation, weights, config); best, best_epoch, best_state, steps, history = untrained, 0, _clone_state(model), 0, []
    for epoch in range(1, config.epochs + 1):
        indices = list(range(len(train))); random.Random(_build_seed("meter", config) + epoch * 1_000_003).shuffle(indices); epoch_loss = 0.0; batch_count = 0
        for start in range(0, len(indices), config.batch_size):
            images, classes, boxes, positive = _stack_meter([train[i] for i in indices[start:start + config.batch_size]]); epoch_loss += _train_meter_batch(model, images, classes, boxes, positive, weights, split="train", optimizer=optimizer, config=config); steps += 1; batch_count += 1
            if progress and batch_count % config.heartbeat_batches == 0: progress("batch_progress", {"task": "meter", "epoch": epoch, "batch": batch_count, "optimizer_steps": steps})
        val_loss, metrics = _evaluate_meter(model, validation, weights, config); history.append({"epoch": epoch, "train_loss": epoch_loss / batch_count, "validation_loss": val_loss, **asdict(metrics)})
        if val_loss < best: best, best_epoch, best_state = val_loss, epoch, _clone_state(model)
        if progress: progress("epoch_complete", {"task": "meter", "epoch": epoch, "epochs": config.epochs, "validation_loss": val_loss, **asdict(metrics)})
    if best_epoch <= 0: _fail("D11 meter refiner did not improve over untrained validation loss")
    model.load_state_dict(best_state, strict=True); final_loss, metrics = _evaluate_meter(model, validation, weights, config)
    if abs(final_loss - best) > 1e-9: _fail("D11 meter restored best state does not reproduce best loss")
    if steps != math.ceil(9840 / config.batch_size) * config.epochs: _fail("D11 meter optimizer step count mismatch")
    return model, RefinerTrainingResult("meter", untrained, best, best_epoch, steps, model_state_sha256(model), count_trainable_parameters(model), refiner_model_fingerprint("meter", config), asdict(metrics)), history


def _fresh_run_root(path: Path, repository_root: Path) -> None:
    resolved, repo = path.resolve(), repository_root.resolve()
    if resolved == repo or repo in resolved.parents: _fail("D11 run output must stay outside repository")
    if path.exists() or path.is_symlink(): _fail("D11 run_root must be fresh")
    path.mkdir(parents=True)


def _write_new(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink(): _fail(f"D11 refuses to overwrite {path.name}")
    path.write_bytes(raw)


def run_verified_stage7d11_training(d10_root: str | Path, run_root: str | Path, repository_root: str | Path, *, expected_d10_manifest_sha256: str, expected_d10_artifact_binding_sha256: str, progress: ProgressCallback | None = None, config: Stage7D11TrainingConfig = FROZEN_D11_CONFIG) -> Stage7D11RunResult:
    if config != FROZEN_D11_CONFIG: _fail("authoritative D11 run requires frozen config")
    expected_manifest = _sha256_hex(expected_d10_manifest_sha256, "expected D10 manifest SHA-256"); expected_binding = _sha256_hex(expected_d10_artifact_binding_sha256, "expected D10 artifact binding SHA-256"); repository_sha, repository_origin = verify_authoritative_repository(repository_root); runtime = verify_stage7c_runtime(); records = load_verified_stage7d11_records(d10_root, expected_manifest_sha256=expected_manifest, expected_artifact_binding_sha256=expected_binding)
    if len(records) != EXPECTED_D10_ROI_RECORDS: _fail("D11 record count mismatch")
    total_parameters = count_trainable_parameters(build_barline_refiner(config)) + count_trainable_parameters(build_meter_refiner(config))
    if total_parameters > D9_ACCEPTANCE.max_total_new_trainable_parameters: _fail("D11 total new parameter count exceeds frozen D9 ceiling")
    root = Path(run_root); _fresh_run_root(root, Path(repository_root)); run_id = sha256(_canonical_json({"version": STAGE7D11_VERSION, "repository_sha": repository_sha, "profile_fingerprint": stage7d11_profile_fingerprint(config), "d10_manifest_sha256": expected_manifest, "d10_artifact_binding_sha256": expected_binding})).hexdigest(); run_directory = root / run_id; run_directory.mkdir()
    if progress: progress("training_started", {"run_id": run_id, "barline_train": 9840, "barline_validation": 1224, "meter_train": 9840, "meter_validation": 1224, "test": 0})
    barline_model, barline_result, barline_history = _train_barline(records, config, progress); meter_model, meter_result, meter_history = _train_meter(records, config, progress)
    ending_sha, ending_origin = verify_authoritative_repository(repository_root); ending_runtime = verify_stage7c_runtime()
    if (ending_sha, ending_origin, ending_runtime) != (repository_sha, repository_origin, runtime): _fail("repository/runtime identity changed during D11 training")
    barline_metrics = BarlineMetrics(**barline_result.validation_metrics); meter_metrics = MeterMetrics(**meter_result.validation_metrics); accepted = acceptance_from_metrics(barline_metrics, meter_metrics)
    checkpoint_payload = {"barline_state_dict": _clone_state(barline_model), "meter_state_dict": _clone_state(meter_model)}; temp = run_directory / "checkpoint.tmp.pt"; torch.save(checkpoint_payload, temp); checkpoint_sha = sha256(temp.read_bytes()).hexdigest(); checkpoint_path = run_directory / f"checkpoint-{checkpoint_sha}.pt"; temp.rename(checkpoint_path)
    metrics_payload = {"schema_version": STAGE7D11_METRICS_SCHEMA, "stage7d11_version": STAGE7D11_VERSION, "repository_sha": repository_sha, "repository_origin": repository_origin, "runtime": runtime, "profile_fingerprint": stage7d11_profile_fingerprint(config), "configuration": asdict(config), "d10": {"repository_sha": EXPECTED_D10_REPOSITORY_SHA, "manifest_sha256": expected_manifest, "artifact_binding_sha256": expected_binding, "roi_records": EXPECTED_D10_ROI_RECORDS, "test_records": 0}, "accepted_d7_structure_state_sha256": EXPECTED_D7_STRUCTURE_STATE_SHA256, "accepted_d7_structure_core_loaded": False, "barline": asdict(barline_result), "meter": asdict(meter_result), "history": {"barline": barline_history, "meter": meter_history}, "acceptance_thresholds": asdict(D9_ACCEPTANCE), "acceptance_passed": accepted, "checkpoint": {"filename": checkpoint_path.name, "sha256": checkpoint_sha}, "sealed_test_split_opened": False}
    metrics_bytes = _canonical_json(metrics_payload); metrics_sha = sha256(metrics_bytes).hexdigest(); metrics_path = run_directory / f"metrics-{metrics_sha}.json"; _write_new(metrics_path, metrics_bytes)
    try: checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc: raise Stage7D11TrainingError("D11 checkpoint cannot be safely reloaded") from exc
    reload_barline, reload_meter = build_barline_refiner(config), build_meter_refiner(config)
    try: reload_barline.load_state_dict(checkpoint["barline_state_dict"], strict=True); reload_meter.load_state_dict(checkpoint["meter_state_dict"], strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc: raise Stage7D11TrainingError("D11 checkpoint state cannot be strictly reloaded") from exc
    if model_state_sha256(reload_barline) != barline_result.state_sha256 or model_state_sha256(reload_meter) != meter_result.state_sha256: _fail("D11 checkpoint state hash mismatch")
    verification_payload = {"schema_version": STAGE7D11_VERIFICATION_SCHEMA, "stage7d11_version": STAGE7D11_VERSION, "repository_sha": repository_sha, "profile_fingerprint": stage7d11_profile_fingerprint(config), "d10_manifest_sha256": expected_manifest, "d10_artifact_binding_sha256": expected_binding, "metrics_sha256": metrics_sha, "checkpoint_sha256": checkpoint_sha, "barline_state_sha256": barline_result.state_sha256, "meter_state_sha256": meter_result.state_sha256, "barline_optimizer_steps": barline_result.optimizer_steps, "meter_optimizer_steps": meter_result.optimizer_steps, "train_records": 19_680, "validation_records": 2_448, "test_records": 0, "test_opened": False, "accepted_d7_structure_core_loaded": False, "accepted_d7_structure_core_mutated": False, "checkpoint_reload_verified": True, "repository_stable_during_run": True, "runtime_stable_during_run": True, "acceptance_passed": accepted}
    verification_bytes = _canonical_json(verification_payload); verification_sha = sha256(verification_bytes).hexdigest(); verification_path = run_directory / f"verification-{verification_sha}.json"; _write_new(verification_path, verification_bytes); _write_new(run_directory / "COMPLETE", f"{verification_sha}  {verification_path.name}\n{metrics_sha}  {metrics_path.name}\n{checkpoint_sha}  {checkpoint_path.name}\n".encode("ascii"))
    if progress: progress("training_complete", {"run_id": run_id, "acceptance_passed": accepted, **asdict(barline_metrics), **asdict(meter_metrics)})
    return Stage7D11RunResult(run_id, run_directory, checkpoint_path, checkpoint_sha, metrics_path, metrics_sha, verification_path, verification_sha, barline_result, meter_result, accepted)
