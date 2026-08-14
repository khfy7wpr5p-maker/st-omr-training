"""Stage 7-D7 StaffSet + StructureSet specialist training.

D7 is the first specialist-model training stage. It consumes only the accepted
Stage 7-D6 TRAIN/VALIDATION derivative sidecars and the already-frozen source
PNGs. The sealed TEST split has no D6 labels and is rejected before any D7
artifact path or byte access.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
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
from .stage7d6_specialist_derivatives import (
    EXPECTED_DEVELOPMENT_FAMILY_COUNTS,
    EXPECTED_DEVELOPMENT_SAMPLE_COUNTS,
    STAGE7D6_LABEL_SCHEMA,
    STAGE7D6_VERSION,
    verify_stage7d6_derivatives,
)
from .training_model import (
    MAX_TRAINABLE_PARAMETERS,
    TrainingRuntimeError,
    assert_finite_tensor,
    assert_model_finite,
    count_trainable_parameters,
    model_state_sha256,
    set_deterministic_cpu,
)


STAGE7D7_VERSION: Final[str] = "stage7d7-staff-structure-training-v1"
STAGE7D7_METRICS_SCHEMA: Final[str] = "stage7d7-specialist-metrics-v1"
STAGE7D7_VERIFICATION_SCHEMA: Final[str] = "stage7d7-specialist-verification-v1"
STAFF_MODEL_VERSION: Final[str] = "stage7d7-staff-dense-segmentation-v1"
STRUCTURE_MODEL_VERSION: Final[str] = "stage7d7-structure-dense-segmentation-v1"
EXPECTED_D6_DERIVATIVE_BUILD_ID: Final[str] = (
    "0faafe229f3497b1147cf0f0ac0ce4b7efe6fa31f360a6a33a3b82c986c8c519"
)
EXPECTED_D6_MANIFEST_SHA256: Final[str] = (
    "e8e415eb6ba9d91a1a880709c3f31d559aa20bf5149734f45b5f84ced16afee9"
)
EXPECTED_D6_ARTIFACT_BINDING_SHA256: Final[str] = (
    "3b7558f0f927ad47a61ed5afb5faa8584dca8647cf8683d4043686eb7b077ea1"
)
EXPECTED_D6_LABEL_COUNT: Final[int] = 1383
EXPECTED_D6_FAMILY_COUNT: Final[int] = 461

STAFF_CHANNELS: Final[tuple[str, ...]] = ("staff_lines", "staff_region")
STRUCTURE_CHANNELS: Final[tuple[str, ...]] = (
    "system_region",
    "measure_region",
    "barline",
    "clef_g2",
    "meter_2_4",
    "meter_3_4",
    "meter_4_4",
)
TASK_CHANNELS: Final[dict[str, tuple[str, ...]]] = {
    "staff": STAFF_CHANNELS,
    "structure": STRUCTURE_CHANNELS,
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
ProgressCallback = Callable[[str, Mapping[str, object]], None]


class Stage7D7TrainingError(RuntimeError):
    """Raised when D7 provenance, data, numeric state, or output fails closed."""


def _fail(message: str) -> None:
    raise Stage7D7TrainingError(message)


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
        raise Stage7D7TrainingError("payload is not canonical JSON serializable") from exc


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        _fail(f"{name} must be lowercase SHA-256 hex")
    return value


def _plain_positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _fail(f"{name} must be a positive integer")
    return value


def _read_canonical_json(path: Path, maximum: int, name: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{name} must be a regular file")
    size = path.stat().st_size
    if not 1 <= size <= maximum:
        _fail(f"{name} byte length is outside the D7 bound")
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda value: _fail(f"non-finite constant in {name}: {value}"),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D7TrainingError(f"{name} is not valid canonical JSON") from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        _fail(f"{name} must be canonical JSON object bytes")
    return payload, raw


@dataclass(frozen=True, slots=True)
class Stage7D7TrainingConfig:
    input_height: int = 96
    input_width: int = 512
    batch_size: int = 6
    epochs: int = 8
    learning_rate_micros: int = 700
    weight_decay_micros: int = 100
    grad_clip_milli: int = 1000
    master_seed: int = 77_007
    objective: str = "bce_plus_soft_dice_v1"
    optimizer: str = "adamw"
    checkpoint_selection: str = "min_validation_loss_per_task"

    def __post_init__(self) -> None:
        expected_ints = {
            "input_height": (self.input_height, 64, 256),
            "input_width": (self.input_width, 256, 1024),
            "batch_size": (self.batch_size, 1, 32),
            "epochs": (self.epochs, 1, 32),
            "learning_rate_micros": (self.learning_rate_micros, 1, 100_000),
            "weight_decay_micros": (self.weight_decay_micros, 0, 100_000),
            "grad_clip_milli": (self.grad_clip_milli, 1, 100_000),
            "master_seed": (self.master_seed, 0, 2**63 - 1),
        }
        for name, (value, lower, upper) in expected_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or not lower <= value <= upper:
                raise ValueError(f"{name} is outside the Stage 7-D7 boundary")
        if self.objective != "bce_plus_soft_dice_v1":
            raise ValueError("objective is frozen for Stage 7-D7")
        if self.optimizer != "adamw":
            raise ValueError("optimizer is frozen for Stage 7-D7")
        if self.checkpoint_selection != "min_validation_loss_per_task":
            raise ValueError("checkpoint_selection is frozen for Stage 7-D7")


FROZEN_D7_CONFIG: Final[Stage7D7TrainingConfig] = Stage7D7TrainingConfig()


@dataclass(frozen=True, slots=True)
class Stage7D7Record:
    sample_id: str
    family_id: str
    split: str
    png_sha256: str
    label_sha256: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True, slots=True)
class TaskTrainingResult:
    task: str
    channels: tuple[str, ...]
    untrained_validation_loss: float
    best_validation_loss: float
    best_epoch: int
    optimizer_steps: int
    channel_dice: dict[str, float]
    state_sha256: str
    parameter_count: int
    model_fingerprint: str


@dataclass(frozen=True, slots=True)
class Stage7D7RunResult:
    run_id: str
    run_directory: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    metrics_path: Path
    metrics_sha256: str
    verification_path: Path
    verification_sha256: str
    staff: TaskTrainingResult
    structure: TaskTrainingResult


def stage7d7_profile_fingerprint(config: Stage7D7TrainingConfig = FROZEN_D7_CONFIG) -> str:
    if not isinstance(config, Stage7D7TrainingConfig):
        raise TypeError("config must be Stage7D7TrainingConfig")
    payload = {
        "version": STAGE7D7_VERSION,
        "staff_model_version": STAFF_MODEL_VERSION,
        "structure_model_version": STRUCTURE_MODEL_VERSION,
        "config": asdict(config),
        "staff_channels": STAFF_CHANNELS,
        "structure_channels": STRUCTURE_CHANNELS,
        "d6_derivative_build_id": EXPECTED_D6_DERIVATIVE_BUILD_ID,
        "d6_manifest_sha256": EXPECTED_D6_MANIFEST_SHA256,
        "d6_artifact_binding_sha256": EXPECTED_D6_ARTIFACT_BINDING_SHA256,
        "split_policy": "train-optimizer-validation-readonly-test-forbidden",
    }
    return sha256(_canonical_json(payload)).hexdigest()


def _development_records(records: object) -> tuple[Mapping[str, object], ...]:
    """Reject TEST immediately after reading only split."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        _fail("D6 manifest records must be a sequence")
    accepted: list[Mapping[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            _fail(f"D6 record[{index}] is not an object")
        split = record.get("split")
        if split == "test":
            _fail("sealed TEST record reached Stage 7-D7")
        if split not in {"train", "validation"}:
            _fail("D7 record split must be train or validation")
        accepted.append(record)
    return tuple(accepted)


def load_verified_stage7d7_records(
    corpus_root: str | Path,
    derivative_root: str | Path,
) -> tuple[Stage7D7Record, ...]:
    source_root = Path(corpus_root)
    d6_root = Path(derivative_root)
    receipt = verify_stage7d6_derivatives(source_root, d6_root)
    expected = {
        "derivative_build_id": EXPECTED_D6_DERIVATIVE_BUILD_ID,
        "manifest_sha256": EXPECTED_D6_MANIFEST_SHA256,
        "artifact_binding_sha256": EXPECTED_D6_ARTIFACT_BINDING_SHA256,
        "label_count": EXPECTED_D6_LABEL_COUNT,
        "sample_count": EXPECTED_D6_LABEL_COUNT,
        "family_count": EXPECTED_D6_FAMILY_COUNT,
        "test_specialist_records": 0,
    }
    for key, value in expected.items():
        if getattr(receipt, key) != value:
            _fail(f"accepted D6 receipt {key} mismatch")
    manifest, raw_manifest = _read_canonical_json(
        d6_root / "manifest.json", 32 * 1024 * 1024, "D6 manifest"
    )
    if sha256(raw_manifest).hexdigest() != EXPECTED_D6_MANIFEST_SHA256:
        _fail("D6 manifest differs from authoritative accepted build")
    rows = _development_records(manifest.get("records"))
    result: list[Stage7D7Record] = []
    counts: Counter[str] = Counter()
    families: dict[str, set[str]] = {"train": set(), "validation": set()}
    seen: set[str] = set()
    for row in rows:
        sample_id = _sha(row.get("sample_id"), "sample_id")
        if sample_id in seen:
            _fail("duplicate D7 sample_id")
        seen.add(sample_id)
        family_id = row.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            _fail("family_id must be non-empty")
        split = str(row.get("split"))
        png_sha = _sha(row.get("png_sha256"), "png_sha256")
        label_sha = _sha(row.get("label_sha256"), "label_sha256")
        image_path = source_root / "images" / f"{png_sha}.png"
        label_path = d6_root / "labels" / f"{label_sha}.json"
        if image_path.is_symlink() or not image_path.is_file():
            _fail("D7 source image is missing or not a regular file")
        if label_path.is_symlink() or not label_path.is_file():
            _fail("D7 label is missing or not a regular file")
        result.append(Stage7D7Record(
            sample_id=sample_id,
            family_id=family_id,
            split=split,
            png_sha256=png_sha,
            label_sha256=label_sha,
            image_path=image_path,
            label_path=label_path,
        ))
        counts[split] += 1
        families[split].add(family_id)
    if dict(sorted(counts.items())) != EXPECTED_DEVELOPMENT_SAMPLE_COUNTS:
        _fail("D7 sample split counts differ from accepted D6")
    if {key: len(value) for key, value in families.items()} != EXPECTED_DEVELOPMENT_FAMILY_COUNTS:
        _fail("D7 family split counts differ from accepted D6")
    return tuple(sorted(result, key=lambda item: item.sample_id))


def _tensor_from_l_image(image: Image.Image) -> torch.Tensor:
    if image.mode != "L":
        _fail("tensor conversion requires grayscale L image")
    raw = bytearray(image.tobytes())
    tensor = torch.frombuffer(raw, dtype=torch.uint8).clone().reshape(image.height, image.width)
    return tensor.to(dtype=torch.float32) / 255.0


def _load_label(record: Stage7D7Record) -> dict[str, object]:
    payload, raw = _read_canonical_json(record.label_path, 2 * 1024 * 1024, "D7 label")
    if sha256(raw).hexdigest() != record.label_sha256:
        _fail("D7 label SHA-256 mismatch")
    if payload.get("schema_version") != STAGE7D6_LABEL_SCHEMA:
        _fail("D7 label schema is not the accepted D6 schema")
    if payload.get("stage7d6_version") != STAGE7D6_VERSION:
        _fail("D7 label version is not accepted D6")
    if payload.get("sample_id") != record.sample_id or payload.get("split") != record.split:
        _fail("D7 label identity/split mismatch")
    return payload


def _load_input_image(record: Stage7D7Record, label: Mapping[str, object], config: Stage7D7TrainingConfig) -> torch.Tensor:
    raw = record.image_path.read_bytes()
    if sha256(raw).hexdigest() != record.png_sha256:
        _fail("D7 source PNG SHA-256 mismatch")
    image_meta = label.get("image")
    if not isinstance(image_meta, Mapping):
        _fail("D7 label image metadata is missing")
    try:
        with Image.open(record.image_path) as opened:
            opened.load()
            if opened.format != "PNG" or opened.mode != "L":
                _fail("D7 source must be grayscale PNG")
            if opened.size != (
                _plain_positive_int(image_meta.get("width"), "image width"),
                _plain_positive_int(image_meta.get("height"), "image height"),
            ):
                _fail("D7 source PNG dimensions differ from label")
            resized = opened.resize(
                (config.input_width, config.input_height),
                resample=Image.Resampling.BILINEAR,
            )
    except OSError as exc:
        raise Stage7D7TrainingError("D7 source PNG cannot be decoded") from exc
    return (1.0 - _tensor_from_l_image(resized)).unsqueeze(0)


def _scaled_box(box: Mapping[str, object], sx: float, sy: float) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(box[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage7D7TrainingError("invalid D6 bbox for D7 raster target") from exc
    if any(not math.isfinite(value) for value in values):
        _fail("non-finite D6 bbox reached D7")
    x0, y0, x1, y1 = values
    return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)


def _scaled_line(line: Mapping[str, object], sx: float, sy: float) -> tuple[float, float, float, float]:
    try:
        start = line["start"]
        end = line["end"]
        if not isinstance(start, Mapping) or not isinstance(end, Mapping):
            raise TypeError
        values = (
            float(start["x"]) * sx,
            float(start["y"]) * sy,
            float(end["x"]) * sx,
            float(end["y"]) * sy,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage7D7TrainingError("invalid D6 line for D7 raster target") from exc
    if any(not math.isfinite(value) for value in values):
        _fail("non-finite D6 line reached D7")
    return values


def _mask_tensor(mask: Image.Image) -> torch.Tensor:
    return _tensor_from_l_image(mask)


def target_masks_from_label(
    label: Mapping[str, object],
    task: str,
    config: Stage7D7TrainingConfig = FROZEN_D7_CONFIG,
) -> torch.Tensor:
    """Rasterize accepted D6 geometry into fixed dense specialist targets."""
    if task not in TASK_CHANNELS:
        raise ValueError("task must be 'staff' or 'structure'")
    image_meta = label.get("image")
    geometry = label.get("geometry")
    if not isinstance(image_meta, Mapping) or not isinstance(geometry, Mapping):
        _fail("D7 label lacks image/geometry objects")
    width = _plain_positive_int(image_meta.get("width"), "image width")
    height = _plain_positive_int(image_meta.get("height"), "image height")
    sx = config.input_width / width
    sy = config.input_height / height
    channels = TASK_CHANNELS[task]
    canvases = [Image.new("L", (config.input_width, config.input_height), 0) for _ in channels]
    draws = [ImageDraw.Draw(canvas) for canvas in canvases]
    channel_index = {name: index for index, name in enumerate(channels)}
    if task == "staff":
        staffs = geometry.get("staff_instances")
        if not isinstance(staffs, list) or not staffs:
            _fail("D7 StaffSet target is empty")
        for staff in staffs:
            if not isinstance(staff, Mapping):
                _fail("D7 staff target is malformed")
            bbox = staff.get("staff_instance_bbox")
            lines = staff.get("five_staff_lines")
            if not isinstance(bbox, Mapping) or not isinstance(lines, list) or len(lines) != 5:
                _fail("D7 StaffSet target violates five-line contract")
            draws[channel_index["staff_region"]].rectangle(_scaled_box(bbox, sx, sy), fill=255)
            for line in lines:
                if not isinstance(line, Mapping):
                    _fail("D7 staff line target is malformed")
                draws[channel_index["staff_lines"]].line(_scaled_line(line, sx, sy), fill=255, width=2)
    else:
        systems = geometry.get("systems")
        measures = geometry.get("measures")
        if not isinstance(systems, list) or not systems or not isinstance(measures, list) or not measures:
            _fail("D7 StructureSet target is empty")
        for system in systems:
            if not isinstance(system, Mapping) or not isinstance(system.get("system_bbox"), Mapping):
                _fail("D7 system target is malformed")
            draws[channel_index["system_region"]].rectangle(
                _scaled_box(system["system_bbox"], sx, sy), fill=255
            )
        meter_map = {"2/4": "meter_2_4", "3/4": "meter_3_4", "4/4": "meter_4_4"}
        for measure in measures:
            if not isinstance(measure, Mapping):
                _fail("D7 measure target is malformed")
            bbox = measure.get("measure_bbox")
            barline = measure.get("barline_segment")
            if not isinstance(bbox, Mapping) or not isinstance(barline, Mapping):
                _fail("D7 measure bbox/barline target is malformed")
            draws[channel_index["measure_region"]].rectangle(_scaled_box(bbox, sx, sy), fill=255)
            draws[channel_index["barline"]].line(_scaled_line(barline, sx, sy), fill=255, width=2)
            clef = measure.get("clef_g2_bbox")
            if clef is not None:
                if not isinstance(clef, Mapping):
                    _fail("D7 clef target is malformed")
                draws[channel_index["clef_g2"]].rectangle(_scaled_box(clef, sx, sy), fill=255)
            meter = measure.get("meter_bbox")
            if meter is not None:
                if not isinstance(meter, Mapping):
                    _fail("D7 meter target is malformed")
                channel = meter_map.get(str(measure.get("meter_class")))
                if channel is None:
                    _fail("D7 meter class is outside V1")
                draws[channel_index[channel]].rectangle(_scaled_box(meter, sx, sy), fill=255)
    return torch.stack([_mask_tensor(canvas) for canvas in canvases], dim=0)


class DenseGeometrySpecialist(nn.Module):
    """Small fully-convolutional geometry specialist with task-isolated weights."""
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        if not isinstance(out_channels, int) or isinstance(out_channels, bool) or not 1 <= out_channels <= 16:
            raise ValueError("out_channels is outside D7 boundary")
        self.out_channels = out_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(24, 16, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, out_channels, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not isinstance(images, torch.Tensor) or images.dtype != torch.float32 or images.ndim != 4:
            raise TrainingRuntimeError("D7 images must be float32 [B,1,H,W]")
        if images.shape[1] != 1:
            raise TrainingRuntimeError("D7 model requires one grayscale channel")
        assert_finite_tensor("D7 model images", images)
        logits = self.decoder(self.encoder(images))
        assert_finite_tensor("D7 model logits", logits)
        return logits


def _task_seed(task: str, config: Stage7D7TrainingConfig) -> int:
    return config.master_seed + (101 if task == "staff" else 202)


def build_specialist_model(
    task: str,
    config: Stage7D7TrainingConfig = FROZEN_D7_CONFIG,
) -> DenseGeometrySpecialist:
    if task not in TASK_CHANNELS:
        raise ValueError("task must be 'staff' or 'structure'")
    set_deterministic_cpu(_task_seed(task, config))
    model = DenseGeometrySpecialist(len(TASK_CHANNELS[task])).cpu()
    count = count_trainable_parameters(model)
    if not 0 < count <= MAX_TRAINABLE_PARAMETERS:
        _fail("D7 specialist parameter count exceeds V1 ceiling")
    assert_model_finite(model)
    return model


def specialist_model_fingerprint(task: str, config: Stage7D7TrainingConfig = FROZEN_D7_CONFIG) -> str:
    if task not in TASK_CHANNELS:
        raise ValueError("task must be 'staff' or 'structure'")
    payload = {
        "model_version": STAFF_MODEL_VERSION if task == "staff" else STRUCTURE_MODEL_VERSION,
        "task": task,
        "channels": TASK_CHANNELS[task],
        "input_height": config.input_height,
        "input_width": config.input_width,
        "architecture": "conv8-16-24-24-up16-up8-v1",
    }
    return sha256(_canonical_json(payload)).hexdigest()


def dense_geometry_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape or logits.dtype != torch.float32 or targets.dtype != torch.float32:
        raise TrainingRuntimeError("D7 logits/targets must be same-shape float32 tensors")
    assert_finite_tensor("D7 logits", logits)
    assert_finite_tensor("D7 targets", targets)
    if bool((targets < 0).any()) or bool((targets > 1).any()):
        raise TrainingRuntimeError("D7 targets must be in [0,1]")
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    loss = bce + (1.0 - dice.mean())
    assert_finite_tensor("D7 loss", loss)
    return loss


def threshold_channel_dice(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise TrainingRuntimeError("D7 dice shape mismatch")
    predictions = (torch.sigmoid(logits) >= 0.5).to(torch.float32)
    dims = (0, 2, 3)
    intersection = (predictions * targets).sum(dim=dims)
    denominator = predictions.sum(dim=dims) + targets.sum(dim=dims)
    return (2.0 * intersection + 1.0) / (denominator + 1.0)


def train_specialist_batch(
    model: DenseGeometrySpecialist,
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    split: str,
    optimizer: torch.optim.Optimizer,
    config: Stage7D7TrainingConfig = FROZEN_D7_CONFIG,
) -> float:
    if split != "train":
        raise TrainingRuntimeError("D7 optimizer step is allowed only for TRAIN")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = dense_geometry_loss(logits, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_milli / 1000.0)
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert_finite_tensor("D7 model gradient", parameter.grad)
    optimizer.step()
    assert_model_finite(model)
    value = float(loss.detach().cpu().item())
    if not math.isfinite(value):
        raise TrainingRuntimeError("D7 training loss is non-finite")
    return value


def _stack_records(
    records: Sequence[Stage7D7Record],
    task: str,
    config: Stage7D7TrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    images: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for record in records:
        label = _load_label(record)
        images.append(_load_input_image(record, label, config))
        targets.append(target_masks_from_label(label, task, config))
    image_batch = torch.stack(images, dim=0)
    target_batch = torch.stack(targets, dim=0)
    assert_finite_tensor("D7 image batch", image_batch)
    assert_finite_tensor("D7 target batch", target_batch)
    return image_batch, target_batch


def _evaluate(
    model: DenseGeometrySpecialist,
    records: Sequence[Stage7D7Record],
    task: str,
    config: Stage7D7TrainingConfig,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    batch_count = 0
    dice_sum = torch.zeros(len(TASK_CHANNELS[task]), dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, len(records), config.batch_size):
            batch = records[start : start + config.batch_size]
            images, targets = _stack_records(batch, task, config)
            logits = model(images)
            loss = dense_geometry_loss(logits, targets)
            total_loss += float(loss.item())
            dice_sum += threshold_channel_dice(logits, targets).to(torch.float64)
            batch_count += 1
    if batch_count <= 0:
        _fail("D7 validation produced no batches")
    channel_dice = {
        name: float(dice_sum[index].item() / batch_count)
        for index, name in enumerate(TASK_CHANNELS[task])
    }
    return total_loss / batch_count, channel_dice


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _train_task(
    task: str,
    train_records: Sequence[Stage7D7Record],
    validation_records: Sequence[Stage7D7Record],
    config: Stage7D7TrainingConfig,
    progress: ProgressCallback | None,
) -> tuple[DenseGeometrySpecialist, TaskTrainingResult, list[dict[str, object]]]:
    model = build_specialist_model(task, config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate_micros / 1_000_000.0,
        weight_decay=config.weight_decay_micros / 1_000_000.0,
    )
    untrained_loss, _ = _evaluate(model, validation_records, task, config)
    best_loss = untrained_loss
    best_epoch = 0
    best_state = _clone_state(model)
    optimizer_steps = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, config.epochs + 1):
        indices = list(range(len(train_records)))
        random.Random(_task_seed(task, config) + epoch * 1_000_003).shuffle(indices)
        epoch_loss = 0.0
        epoch_batches = 0
        for start in range(0, len(indices), config.batch_size):
            selected = [train_records[index] for index in indices[start : start + config.batch_size]]
            images, targets = _stack_records(selected, task, config)
            epoch_loss += train_specialist_batch(
                model, images, targets, split="train", optimizer=optimizer, config=config
            )
            optimizer_steps += 1
            epoch_batches += 1
        validation_loss, channel_dice = _evaluate(model, validation_records, task, config)
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / epoch_batches,
            "validation_loss": validation_loss,
            "channel_dice": channel_dice,
        })
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _clone_state(model)
        if progress is not None:
            progress("epoch_complete", {
                "task": task,
                "epoch": epoch,
                "epochs": config.epochs,
                "validation_loss": validation_loss,
                "best_validation_loss": best_loss,
            })
    if best_epoch <= 0:
        _fail(f"D7 {task} specialist did not improve over untrained validation loss")
    model.load_state_dict(best_state, strict=True)
    assert_model_finite(model)
    final_loss, final_dice = _evaluate(model, validation_records, task, config)
    if abs(final_loss - best_loss) > 1e-9:
        _fail("D7 restored best state does not reproduce best validation loss")
    result = TaskTrainingResult(
        task=task,
        channels=TASK_CHANNELS[task],
        untrained_validation_loss=untrained_loss,
        best_validation_loss=best_loss,
        best_epoch=best_epoch,
        optimizer_steps=optimizer_steps,
        channel_dice=final_dice,
        state_sha256=model_state_sha256(model),
        parameter_count=count_trainable_parameters(model),
        model_fingerprint=specialist_model_fingerprint(task, config),
    )
    return model, result, history


def _fresh_run_root(path: Path, repository_root: Path) -> None:
    resolved = path.resolve()
    repo = repository_root.resolve()
    if resolved == repo or repo in resolved.parents:
        _fail("D7 run output must stay outside repository")
    if path.exists() or path.is_symlink():
        _fail("D7 run_root must be fresh")
    path.mkdir(parents=True)


def _write_new(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        _fail(f"D7 refuses to overwrite {path.name}")
    path.write_bytes(data)


def run_verified_stage7d7_training(
    corpus_root: str | Path,
    derivative_root: str | Path,
    run_root: str | Path,
    repository_root: str | Path,
    *,
    progress: ProgressCallback | None = None,
    config: Stage7D7TrainingConfig = FROZEN_D7_CONFIG,
) -> Stage7D7RunResult:
    if config != FROZEN_D7_CONFIG:
        _fail("authoritative D7 run requires the frozen training config")
    repository_sha, repository_origin = verify_authoritative_repository(repository_root)
    runtime = verify_stage7c_runtime()
    records = load_verified_stage7d7_records(corpus_root, derivative_root)
    train_records = tuple(record for record in records if record.split == "train")
    validation_records = tuple(record for record in records if record.split == "validation")
    if len(train_records) != 1230 or len(validation_records) != 153:
        _fail("D7 train/validation cardinality mismatch")
    root = Path(run_root)
    _fresh_run_root(root, Path(repository_root))
    run_identity = {
        "version": STAGE7D7_VERSION,
        "repository_sha": repository_sha,
        "profile_fingerprint": stage7d7_profile_fingerprint(config),
        "d6_manifest_sha256": EXPECTED_D6_MANIFEST_SHA256,
    }
    run_id = sha256(_canonical_json(run_identity)).hexdigest()
    run_directory = root / run_id
    run_directory.mkdir()
    if progress is not None:
        progress("training_started", {"run_id": run_id, "train": 1230, "validation": 153, "test": 0})
    staff_model, staff_result, staff_history = _train_task(
        "staff", train_records, validation_records, config, progress
    )
    structure_model, structure_result, structure_history = _train_task(
        "structure", train_records, validation_records, config, progress
    )
    ending_sha, ending_origin = verify_authoritative_repository(repository_root)
    ending_runtime = verify_stage7c_runtime()
    if ending_sha != repository_sha or ending_origin != repository_origin:
        _fail("repository identity changed during D7 training")
    if ending_runtime != runtime:
        _fail("runtime identity changed during D7 training")
    checkpoint_payload = {
        "schema_version": "stage7d7-specialist-checkpoint-v1",
        "repository_sha": repository_sha,
        "profile_fingerprint": stage7d7_profile_fingerprint(config),
        "staff_model_fingerprint": staff_result.model_fingerprint,
        "structure_model_fingerprint": structure_result.model_fingerprint,
        "staff_state_dict": _clone_state(staff_model),
        "structure_state_dict": _clone_state(structure_model),
        "staff_best_epoch": staff_result.best_epoch,
        "structure_best_epoch": structure_result.best_epoch,
    }
    temp_checkpoint = run_directory / "checkpoint.tmp.pt"
    torch.save(checkpoint_payload, temp_checkpoint)
    checkpoint_sha = sha256(temp_checkpoint.read_bytes()).hexdigest()
    checkpoint_path = run_directory / f"checkpoint-{checkpoint_sha}.pt"
    temp_checkpoint.rename(checkpoint_path)
    metrics_payload = {
        "schema_version": STAGE7D7_METRICS_SCHEMA,
        "stage7d7_version": STAGE7D7_VERSION,
        "repository_sha": repository_sha,
        "repository_origin": repository_origin,
        "runtime": runtime,
        "profile_fingerprint": stage7d7_profile_fingerprint(config),
        "configuration": asdict(config),
        "dataset": {
            "d6_derivative_build_id": EXPECTED_D6_DERIVATIVE_BUILD_ID,
            "d6_manifest_sha256": EXPECTED_D6_MANIFEST_SHA256,
            "d6_artifact_binding_sha256": EXPECTED_D6_ARTIFACT_BINDING_SHA256,
            "train_samples": 1230,
            "validation_samples": 153,
            "test_records": 0,
        },
        "staff": asdict(staff_result),
        "structure": asdict(structure_result),
        "history": {"staff": staff_history, "structure": structure_history},
        "checkpoint": {"filename": checkpoint_path.name, "sha256": checkpoint_sha},
        "sealed_test_split_opened": False,
    }
    metrics_bytes = _canonical_json(metrics_payload)
    metrics_sha = sha256(metrics_bytes).hexdigest()
    metrics_path = run_directory / f"metrics-{metrics_sha}.json"
    _write_new(metrics_path, metrics_bytes)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Stage7D7TrainingError("D7 checkpoint cannot be safely reloaded") from exc
    if not isinstance(checkpoint, dict):
        _fail("D7 checkpoint root is invalid")
    reload_staff = build_specialist_model("staff", config)
    reload_structure = build_specialist_model("structure", config)
    try:
        reload_staff.load_state_dict(checkpoint["staff_state_dict"], strict=True)
        reload_structure.load_state_dict(checkpoint["structure_state_dict"], strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise Stage7D7TrainingError("D7 checkpoint state cannot be strictly reloaded") from exc
    assert_model_finite(reload_staff)
    assert_model_finite(reload_structure)
    if model_state_sha256(reload_staff) != staff_result.state_sha256:
        _fail("D7 staff checkpoint state hash mismatch")
    if model_state_sha256(reload_structure) != structure_result.state_sha256:
        _fail("D7 structure checkpoint state hash mismatch")
    verification_payload = {
        "schema_version": STAGE7D7_VERIFICATION_SCHEMA,
        "stage7d7_version": STAGE7D7_VERSION,
        "repository_sha": repository_sha,
        "profile_fingerprint": stage7d7_profile_fingerprint(config),
        "d6_manifest_sha256": EXPECTED_D6_MANIFEST_SHA256,
        "d6_artifact_binding_sha256": EXPECTED_D6_ARTIFACT_BINDING_SHA256,
        "metrics_sha256": metrics_sha,
        "checkpoint_sha256": checkpoint_sha,
        "staff_state_sha256": staff_result.state_sha256,
        "structure_state_sha256": structure_result.state_sha256,
        "staff_optimizer_steps": staff_result.optimizer_steps,
        "structure_optimizer_steps": structure_result.optimizer_steps,
        "train_samples": 1230,
        "validation_samples": 153,
        "test_records": 0,
        "test_opened": False,
        "checkpoint_reload_verified": True,
        "repository_stable_during_run": True,
        "runtime_stable_during_run": True,
    }
    verification_bytes = _canonical_json(verification_payload)
    verification_sha = sha256(verification_bytes).hexdigest()
    verification_path = run_directory / f"verification-{verification_sha}.json"
    _write_new(verification_path, verification_bytes)
    complete = (
        f"{verification_sha}  {verification_path.name}\n"
        f"{metrics_sha}  {metrics_path.name}\n"
        f"{checkpoint_sha}  {checkpoint_path.name}\n"
    ).encode("ascii")
    _write_new(run_directory / "COMPLETE", complete)
    if progress is not None:
        progress("training_complete", {
            "run_id": run_id,
            "staff_best_validation_loss": staff_result.best_validation_loss,
            "structure_best_validation_loss": structure_result.best_validation_loss,
        })
    return Stage7D7RunResult(
        run_id=run_id,
        run_directory=run_directory,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        metrics_path=metrics_path,
        metrics_sha256=metrics_sha,
        verification_path=verification_path,
        verification_sha256=verification_sha,
        staff=staff_result,
        structure=structure_result,
    )
