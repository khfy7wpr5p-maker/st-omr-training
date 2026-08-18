"""M4-E3K-B1 frozen D7 StaffSet geometry transfer.

This module is inference-only. It strictly reloads the accepted Stage 7-D7
StaffSet checkpoint, predicts the frozen ``staff_lines`` + ``staff_region``
channels on TRAIN images, and deterministically decodes page-space five-line
staff geometry for the E3K-R2 boundary proposal stage.

B1 intentionally changes one variable relative to the accepted R2 TRAIN upper
bound: authoritative D6 staff geometry is replaced by frozen D7-predicted staff
geometry. It does not load D11, does not train, and does not tune thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Final, Mapping, Sequence

import torch

from .stage7d7_specialist_training import (
    FROZEN_D7_CONFIG,
    STAFF_CHANNELS,
    Stage7D7Record,
    Stage7D7TrainingError,
    _load_input_image,
    build_specialist_model,
    specialist_model_fingerprint,
    stage7d7_profile_fingerprint,
)
from .training_model import assert_model_finite, model_state_sha256


STAGE: Final[str] = "M4-E3K-B1-FROZEN-D7-STAFF-GEOMETRY-TRANSFER"
D7_DENSE_THRESHOLD: Final[float] = 0.50
MINIMUM_STAFF_COMPONENT_WIDTH_FRACTION: Final[float] = 0.08
MINIMUM_STAFF_COMPONENT_AREA_FRACTION: Final[float] = 0.002
EXPECTED_D7_CHECKPOINT_SHA256: Final[str] = (
    "5f009ca8ba68d38497a7dd25590d4dd98c537f20c5d5525bf66e288afbf417dc"
)
EXPECTED_D7_STAFF_STATE_SHA256: Final[str] = (
    "3131548548521229e6acd6fee8cffc66081cb54125645f9eff5a488de7603af8"
)
CHECKPOINT_SCHEMA: Final[str] = "stage7d7-specialist-checkpoint-v1"
MAX_CHECKPOINT_BYTES: Final[int] = 64 * 1024 * 1024


class M4E3KB1GeometryError(RuntimeError):
    """Raised when frozen D7 inference/geometry invariants fail closed."""


def _fail(message: str) -> None:
    raise M4E3KB1GeometryError(message)


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        _fail(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise M4E3KB1GeometryError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class _Component:
    x0: int
    y0: int
    x1: int
    y1: int
    area: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1


@dataclass(frozen=True, slots=True)
class PredictedStaffGeometry:
    """One D7-predicted staff expressed in original page coordinates."""

    staff_bbox: dict[str, float]
    five_staff_lines: tuple[dict[str, dict[str, float]], ...]
    staff_spacing: float
    model_component_bbox: tuple[int, int, int, int]
    model_staff_slope: float
    line_template_score: float

    def __post_init__(self) -> None:
        if len(self.five_staff_lines) != 5:
            raise ValueError("predicted staff must contain exactly five lines")
        if not math.isfinite(self.staff_spacing) or self.staff_spacing <= 0:
            raise ValueError("predicted staff spacing must be finite and positive")
        if not math.isfinite(self.model_staff_slope) or abs(self.model_staff_slope) > 0.35:
            raise ValueError("predicted D7 staff slope is outside B1 bound")
        if not math.isfinite(self.line_template_score) or self.line_template_score <= 0:
            raise ValueError("predicted staff line-template score must be positive")


def load_frozen_d7_staff_model(checkpoint_path: str | Path) -> torch.nn.Module:
    """Strictly reload the accepted D7 StaffSet model and verify its state hash."""

    path = Path(checkpoint_path)
    if path.is_symlink() or not path.is_file():
        _fail("D7 checkpoint must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= MAX_CHECKPOINT_BYTES:
        _fail("D7 checkpoint byte length is outside B1 bound")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != EXPECTED_D7_CHECKPOINT_SHA256:
        _fail("D7 checkpoint SHA-256 differs from the accepted checkpoint")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise M4E3KB1GeometryError("D7 checkpoint cannot be safely loaded") from exc
    if not isinstance(payload, Mapping):
        _fail("D7 checkpoint root must be a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        _fail("D7 checkpoint schema mismatch")
    if payload.get("profile_fingerprint") != stage7d7_profile_fingerprint(FROZEN_D7_CONFIG):
        _fail("D7 checkpoint profile fingerprint mismatch")
    if payload.get("staff_model_fingerprint") != specialist_model_fingerprint(
        "staff", FROZEN_D7_CONFIG
    ):
        _fail("D7 StaffSet model fingerprint mismatch")
    state = payload.get("staff_state_dict")
    if not isinstance(state, Mapping):
        _fail("D7 StaffSet state dict is missing")
    model = build_specialist_model("staff", FROZEN_D7_CONFIG)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise M4E3KB1GeometryError("D7 StaffSet state cannot be strictly loaded") from exc
    assert_model_finite(model)
    if model_state_sha256(model) != EXPECTED_D7_STAFF_STATE_SHA256:
        _fail("D7 StaffSet state SHA-256 mismatch")
    model.eval()
    return model


def _binary_components(mask: torch.Tensor) -> tuple[_Component, ...]:
    """8-connected components on a small CPU bool mask using row-run union-find."""

    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 2:
        raise TypeError("mask must be a 2D torch.bool tensor")
    if mask.device.type != "cpu":
        _fail("B1 component mask must be on CPU")
    height, width = int(mask.shape[0]), int(mask.shape[1])
    rows = mask.tolist()

    parent: list[int] = []
    run_data: list[tuple[int, int, int]] = []  # y, x0, x1 inclusive
    previous: list[tuple[int, int, int]] = []  # x0, x1, run_id

    def make_run(y: int, x0: int, x1: int) -> int:
        run_id = len(parent)
        parent.append(run_id)
        run_data.append((y, x0, x1))
        return run_id

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for y in range(height):
        current: list[tuple[int, int, int]] = []
        x = 0
        row = rows[y]
        while x < width:
            if not row[x]:
                x += 1
                continue
            start = x
            while x + 1 < width and row[x + 1]:
                x += 1
            end = x
            run_id = make_run(y, start, end)
            current.append((start, end, run_id))
            for px0, px1, previous_id in previous:
                # +/-1 admits diagonal adjacency (8-connectivity).
                if px1 + 1 < start:
                    continue
                if end + 1 < px0:
                    break
                union(run_id, previous_id)
            x += 1
        previous = current

    groups: dict[int, list[int]] = {}
    for run_id in range(len(run_data)):
        groups.setdefault(find(run_id), []).append(run_id)

    components: list[_Component] = []
    for run_ids in groups.values():
        ys: list[int] = []
        x0s: list[int] = []
        x1s: list[int] = []
        area = 0
        for run_id in run_ids:
            y, x0, x1 = run_data[run_id]
            ys.append(y)
            x0s.append(x0)
            x1s.append(x1)
            area += x1 - x0 + 1
        components.append(
            _Component(
                x0=min(x0s),
                y0=min(ys),
                x1=max(x1s),
                y1=max(ys),
                area=area,
            )
        )
    components.sort(key=lambda item: (item.y0, item.x0, item.y1, item.x1))
    return tuple(components)


def _fit_common_staff_slope(
    line_probabilities: torch.Tensor,
    component: _Component,
) -> float:
    """Estimate common staff slope from the mean active staff-line y per x."""

    binary = line_probabilities >= D7_DENSE_THRESHOLD
    xs: list[float] = []
    ys: list[float] = []
    for x in range(component.x0, component.x1 + 1):
        active_y: list[int] = []
        weights: list[float] = []
        for y in range(component.y0, component.y1 + 1):
            if bool(binary[y, x]):
                active_y.append(y)
                weights.append(float(line_probabilities[y, x].item()))
        if active_y:
            total = sum(weights)
            if total <= 0:
                continue
            xs.append(float(x))
            ys.append(sum(y * w for y, w in zip(active_y, weights)) / total)
    if len(xs) < 2:
        _fail("D7 staff-line component has insufficient x support for slope")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0:
        _fail("D7 staff-line slope denominator is degenerate")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    if not math.isfinite(slope) or abs(slope) > 0.35:
        _fail("D7 predicted staff slope exceeds frozen B1 bound")
    return slope


def _equal_spaced_line_template(
    line_probabilities: torch.Tensor,
    component: _Component,
    slope: float,
) -> tuple[float, float, float]:
    """Return (first_line_y_at_ref, spacing, score) in D7 model coordinates.

    Probabilities below the frozen 0.50 D7 dense threshold contribute zero.
    Exactly five equal-spaced lines are selected by exhaustive deterministic
    maximization of the deskewed staff-line probability profile.
    """

    reference_x = (component.x0 + component.x1) / 2.0
    profile: dict[int, float] = {}
    for x in range(component.x0, component.x1 + 1):
        for y in range(component.y0, component.y1 + 1):
            probability = float(line_probabilities[y, x].item())
            if probability < D7_DENSE_THRESHOLD:
                continue
            deskewed_y = float(y) - slope * (float(x) - reference_x)
            bucket = int(round(deskewed_y))
            profile[bucket] = profile.get(bucket, 0.0) + probability
    if len(profile) < 5:
        _fail("D7 staff-line probability profile cannot support five lines")
    minimum_y = min(profile)
    maximum_y = max(profile)
    maximum_spacing = (maximum_y - minimum_y) // 4
    if maximum_spacing < 1:
        _fail("D7 staff-line profile has no positive five-line spacing")

    best: tuple[float, float, int, int] | None = None
    # Sort key: maximize total score, then weakest-line support, then spacing,
    # then prefer the smaller top y for deterministic ties.
    for spacing in range(1, maximum_spacing + 1):
        final_start = maximum_y - 4 * spacing
        for first_y in range(minimum_y, final_start + 1):
            supports = [profile.get(first_y + index * spacing, 0.0) for index in range(5)]
            if any(value <= 0.0 for value in supports):
                continue
            total = sum(supports)
            weakest = min(supports)
            candidate = (total, weakest, spacing, -first_y)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        _fail("D7 staff-line probabilities do not resolve to five equal-spaced lines")
    total, _, spacing_int, negative_first = best
    return float(-negative_first), float(spacing_int), float(total)


def _line_x_support(
    line_probabilities: torch.Tensor,
    component: _Component,
    *,
    line_y_at_reference: float,
    slope: float,
) -> tuple[int, int]:
    """Find threshold-supported x extent for one decoded staff line."""

    reference_x = (component.x0 + component.x1) / 2.0
    supported: list[int] = []
    for x in range(component.x0, component.x1 + 1):
        expected_y = line_y_at_reference + slope * (float(x) - reference_x)
        center = int(round(expected_y))
        y0 = max(component.y0, center - 1)
        y1 = min(component.y1, center + 1)
        if any(
            float(line_probabilities[y, x].item()) >= D7_DENSE_THRESHOLD
            for y in range(y0, y1 + 1)
        ):
            supported.append(x)
    if len(supported) < 2:
        _fail("decoded D7 staff line lacks two-column threshold support")
    return min(supported), max(supported)


def decode_d7_staff_geometry(
    probabilities: torch.Tensor,
    *,
    original_width: int,
    original_height: int,
) -> tuple[PredictedStaffGeometry, ...]:
    """Decode frozen D7 StaffSet probabilities into original page geometry."""

    if not isinstance(probabilities, torch.Tensor) or probabilities.dtype != torch.float32:
        raise TypeError("probabilities must be float32 tensor")
    if probabilities.ndim != 3 or tuple(probabilities.shape) != (
        len(STAFF_CHANNELS),
        FROZEN_D7_CONFIG.input_height,
        FROZEN_D7_CONFIG.input_width,
    ):
        _fail("D7 StaffSet probability tensor shape mismatch")
    if probabilities.device.type != "cpu":
        _fail("D7 StaffSet probabilities must be on CPU")
    if not bool(torch.isfinite(probabilities).all()):
        _fail("D7 StaffSet probabilities contain non-finite values")
    if bool((probabilities < 0).any()) or bool((probabilities > 1).any()):
        _fail("D7 StaffSet probabilities must be in [0,1]")
    if not isinstance(original_width, int) or isinstance(original_width, bool) or original_width <= 0:
        _fail("original_width must be positive integer")
    if not isinstance(original_height, int) or isinstance(original_height, bool) or original_height <= 0:
        _fail("original_height must be positive integer")

    line_index = STAFF_CHANNELS.index("staff_lines")
    region_index = STAFF_CHANNELS.index("staff_region")
    line_probs = probabilities[line_index]
    region_mask = probabilities[region_index] >= D7_DENSE_THRESHOLD
    input_h = FROZEN_D7_CONFIG.input_height
    input_w = FROZEN_D7_CONFIG.input_width
    minimum_width = input_w * MINIMUM_STAFF_COMPONENT_WIDTH_FRACTION
    minimum_area = input_h * input_w * MINIMUM_STAFF_COMPONENT_AREA_FRACTION
    components = tuple(
        component
        for component in _binary_components(region_mask)
        if component.width >= minimum_width and component.area >= minimum_area
    )

    scale_x = original_width / float(input_w)
    scale_y = original_height / float(input_h)
    decoded: list[PredictedStaffGeometry] = []
    for component in components:
        try:
            slope = _fit_common_staff_slope(line_probs, component)
            first_y, spacing, template_score = _equal_spaced_line_template(
                line_probs, component, slope
            )
            reference_x = (component.x0 + component.x1) / 2.0
            lines: list[dict[str, dict[str, float]]] = []
            for index in range(5):
                line_y_ref = first_y + index * spacing
                x0, x1 = _line_x_support(
                    line_probs,
                    component,
                    line_y_at_reference=line_y_ref,
                    slope=slope,
                )
                y0 = line_y_ref + slope * (float(x0) - reference_x)
                y1 = line_y_ref + slope * (float(x1) - reference_x)
                lines.append({
                    "start": {"x": float(x0) * scale_x, "y": y0 * scale_y},
                    "end": {"x": float(x1) * scale_x, "y": y1 * scale_y},
                })
        except M4E3KB1GeometryError:
            # A staff-region component that cannot satisfy the frozen five-line
            # contract is not silently invented into geometry.
            continue

        decoded.append(
            PredictedStaffGeometry(
                staff_bbox={
                    "x_min": float(component.x0) * scale_x,
                    "y_min": float(component.y0) * scale_y,
                    "x_max": float(component.x1 + 1) * scale_x,
                    "y_max": float(component.y1 + 1) * scale_y,
                },
                five_staff_lines=tuple(lines),
                staff_spacing=spacing * scale_y,
                model_component_bbox=(component.x0, component.y0, component.x1, component.y1),
                model_staff_slope=slope,
                line_template_score=template_score,
            )
        )
    decoded.sort(
        key=lambda item: (
            item.staff_bbox["y_min"],
            item.staff_bbox["x_min"],
            item.staff_bbox["y_max"],
        )
    )
    return tuple(decoded)


def predict_d7_staff_geometry(
    model: torch.nn.Module,
    record: Stage7D7Record,
    label: Mapping[str, object],
) -> tuple[PredictedStaffGeometry, ...]:
    """Run frozen StaffSet inference for one already-selected TRAIN record."""

    if record.split != "train":
        _fail("B1 inference accepts TRAIN records only")
    image_meta = label.get("image")
    if not isinstance(image_meta, Mapping):
        _fail("D6 image metadata is missing")
    width_value = image_meta.get("width")
    height_value = image_meta.get("height")
    if not isinstance(width_value, int) or isinstance(width_value, bool) or width_value <= 0:
        _fail("D6 image width must be positive integer")
    if not isinstance(height_value, int) or isinstance(height_value, bool) or height_value <= 0:
        _fail("D6 image height must be positive integer")
    try:
        input_tensor = _load_input_image(record, label, FROZEN_D7_CONFIG)
    except Stage7D7TrainingError as exc:
        raise M4E3KB1GeometryError("D7 TRAIN image cannot be loaded for inference") from exc
    with torch.no_grad():
        logits = model(input_tensor.unsqueeze(0))
        probabilities = torch.sigmoid(logits).squeeze(0).to(dtype=torch.float32, device="cpu")
    return decode_d7_staff_geometry(
        probabilities,
        original_width=width_value,
        original_height=height_value,
    )
