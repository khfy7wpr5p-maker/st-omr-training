"""Real frozen-checkpoint inference boundary for the Meter Presence bridge.

This is the first runtime stage that may execute an audited real checkpoint.
It consumes only a reconstructed historical D11 Meter ROI artifact and the
exact frozen D11 checkpoint. It does not train, tune thresholds, read dataset
splits, run digit specialists, compose a meter, or invoke the Resolver.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Final, Mapping

from PIL import Image, UnidentifiedImageError
import torch
from torch import nn

from .runtime_meter_historical_roi_adapter_v1 import (
    HistoricalMeterRoiArtifactV1,
    historical_meter_roi_profile_fingerprint_v1,
)
from .runtime_meter_real_checkpoint_audit_v1 import (
    AuditedCheckpointStateV1,
    PRESENCE_D11_SHA256,
    audit_presence_d11_checkpoint_v1,
)

REAL_METER_PRESENCE_INFERENCE_V1: Final[str] = "runtime-real-meter-presence-inference-v1"
METER_CLASSES: Final[tuple[str, ...]] = ("none", "2/4", "3/4", "4/4")


class RealMeterInferenceError(RuntimeError):
    pass


class RuntimeD11PresenceBridge(nn.Module):
    """Runtime mirror of the frozen D11 MeterRefiner architecture."""

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
        self.classifier = nn.Linear(64, 4)
        self.bbox = nn.Linear(64, 4)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.dtype != torch.float32 or tuple(images.shape[1:]) != (1, 192, 256):
            raise RealMeterInferenceError("D11 Presence input must be float32 [B,1,192,256]")
        features = self.projection(self.pool(self.encoder(images)))
        logits = self.classifier(features)
        raw = torch.sigmoid(self.bbox(features))
        boxes = torch.stack(
            (
                torch.minimum(raw[:, 0], raw[:, 2]),
                torch.minimum(raw[:, 1], raw[:, 3]),
                torch.maximum(raw[:, 0], raw[:, 2]),
                torch.maximum(raw[:, 1], raw[:, 3]),
            ),
            dim=1,
        )
        if not bool(torch.isfinite(logits).all().item()) or not bool(torch.isfinite(boxes).all().item()):
            raise RealMeterInferenceError("D11 Presence inference produced non-finite output")
        return logits, boxes


def _canonical_sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return sha256(raw).hexdigest()


def runtime_d11_presence_architecture_fingerprint_v1() -> str:
    return _canonical_sha({
        "version": REAL_METER_PRESENCE_INFERENCE_V1,
        "architecture": "conv8-16-24-24-pool6x8-fc64-class4-box4-v1",
        "input": "gray8-256x192-invert-to-ink-float32",
        "classes": METER_CLASSES,
        "checkpoint_sha256": PRESENCE_D11_SHA256,
        "historical_roi_profile": historical_meter_roi_profile_fingerprint_v1(),
    })


def _tensor_from_artifact(artifact: HistoricalMeterRoiArtifactV1) -> torch.Tensor:
    if not isinstance(artifact, HistoricalMeterRoiArtifactV1):
        raise TypeError("artifact must be HistoricalMeterRoiArtifactV1")
    if sha256(artifact.png_bytes).hexdigest() != artifact.image_sha256:
        raise RealMeterInferenceError("historical ROI byte identity changed")
    if artifact.profile_fingerprint != historical_meter_roi_profile_fingerprint_v1():
        raise RealMeterInferenceError("historical ROI profile identity changed")
    try:
        with Image.open(BytesIO(artifact.png_bytes)) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (256, 192):
                raise RealMeterInferenceError("D11 Presence requires exact gray8 256x192 historical ROI PNG")
            image.load()
            raw = bytearray(image.tobytes())
    except (UnidentifiedImageError, OSError) as exc:
        raise RealMeterInferenceError("historical ROI PNG cannot be decoded") from exc
    tensor = torch.frombuffer(raw, dtype=torch.uint8).clone().reshape(192, 256)
    tensor = 1.0 - tensor.to(dtype=torch.float32) / 255.0
    return tensor.unsqueeze(0).unsqueeze(0)


@dataclass(frozen=True, slots=True)
class RealPresenceInferenceV1:
    measure_id: str
    staff_id: str
    roi_image_sha256: str
    checkpoint_sha256: str
    class_probabilities: tuple[float, float, float, float]
    presence_score: float
    predicted_bbox_normalized: tuple[float, float, float, float]
    predicted_x_center_canvas: float
    architecture_fingerprint: str
    inference_fingerprint: str

    def __post_init__(self) -> None:
        if self.checkpoint_sha256 != PRESENCE_D11_SHA256:
            raise ValueError("unexpected D11 checkpoint identity")
        if len(self.class_probabilities) != 4 or any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in self.class_probabilities):
            raise ValueError("invalid D11 class probabilities")
        if not math.isfinite(self.presence_score) or not 0.0 <= self.presence_score <= 1.0:
            raise ValueError("invalid Presence score")
        if len(self.predicted_bbox_normalized) != 4 or any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in self.predicted_bbox_normalized):
            raise ValueError("invalid D11 bbox")


def infer_presence_from_audited_state_v1(
    artifact: HistoricalMeterRoiArtifactV1,
    audited: AuditedCheckpointStateV1,
) -> RealPresenceInferenceV1:
    if audited.role != "presence-d11-bridge" or audited.checkpoint_sha256 != PRESENCE_D11_SHA256:
        raise RealMeterInferenceError("audited state is not the frozen D11 Presence bridge")
    model = RuntimeD11PresenceBridge().cpu()
    try:
        model.load_state_dict(dict(audited.model_state), strict=True)
    except Exception as exc:
        raise RealMeterInferenceError("audited D11 state cannot populate runtime architecture") from exc
    model.eval()
    tensor = _tensor_from_artifact(artifact)
    with torch.inference_mode():
        logits, boxes = model(tensor)
        probabilities = torch.softmax(logits[0], dim=0)
    probs = tuple(float(v) for v in probabilities.tolist())
    if any(not math.isfinite(v) for v in probs):
        raise RealMeterInferenceError("D11 softmax produced non-finite probability")
    presence = 1.0 - probs[0]
    box = tuple(float(v) for v in boxes[0].tolist())
    x_center = ((box[0] + box[2]) * 0.5) * 256.0
    architecture_fp = runtime_d11_presence_architecture_fingerprint_v1()
    inference_fp = _canonical_sha({
        "version": REAL_METER_PRESENCE_INFERENCE_V1,
        "measure_id": artifact.measure_id,
        "staff_id": artifact.staff_id,
        "roi_image_sha256": artifact.image_sha256,
        "checkpoint_sha256": audited.checkpoint_sha256,
        "class_probabilities": probs,
        "presence_score": presence,
        "bbox": box,
        "x_center_canvas": x_center,
        "architecture_fingerprint": architecture_fp,
    })
    return RealPresenceInferenceV1(
        measure_id=artifact.measure_id,
        staff_id=artifact.staff_id,
        roi_image_sha256=artifact.image_sha256,
        checkpoint_sha256=audited.checkpoint_sha256,
        class_probabilities=probs,
        presence_score=presence,
        predicted_bbox_normalized=box,
        predicted_x_center_canvas=x_center,
        architecture_fingerprint=architecture_fp,
        inference_fingerprint=inference_fp,
    )


def infer_presence_from_checkpoint_v1(
    artifact: HistoricalMeterRoiArtifactV1,
    checkpoint_path: Path,
) -> RealPresenceInferenceV1:
    """Audit the exact frozen checkpoint, then execute one read-only CPU inference."""
    audited = audit_presence_d11_checkpoint_v1(checkpoint_path)
    return infer_presence_from_audited_state_v1(artifact, audited)


def digit_specialist_execution_allowed_in_this_stage() -> bool:
    return False


def meter_composition_allowed_in_this_stage() -> bool:
    return False


def resolver_connection_allowed_in_this_stage() -> bool:
    return False
