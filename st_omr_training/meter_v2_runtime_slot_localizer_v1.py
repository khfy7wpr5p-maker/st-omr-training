"""TRAIN-derived deterministic Meter V2 runtime digit-slot proposal candidate.

Shadow-only. The policy is frozen from M3-C2 TRAIN-only anchor modes plus M4A
TRAIN-only positive digit geometry. It uses only the 256x192 D10 Meter ROI
pixels and measure_number at inference time. No ground-truth box, no D11 bbox,
no training, no threshold tuning, no TEST, and no runtime Resolver wiring.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Final
from PIL import Image
from .meter_v2_deterministic_composer_v1 import MeterBox

M3C2_ANCHOR_MODES_SHA256: Final[str] = "f8594e1550027ed2f69670a030ec8a6a7d8247c1b07a5000b111038208b31745"
M4A_DATASET_MANIFEST_SHA256: Final[str] = "ebda40dae10f0d6490df2c7728dab5cc2cc6f58b5420b198dfbb441a99ecebb9"
MODE_DATA_SHA256: Final[str] = "19bb1e639225d79a2c085ac1d0fd45455ad4010be1099e11f6ec0ed31477650b"
RUNTIME_ROI_SIZE: Final[tuple[int, int]] = (256, 192)
INK_BASELINE_QUANTILE_MILLI: Final[int] = 200
SEARCH_HALF_WIDTH_MULTIPLIER_MILLI: Final[int] = 1000
LOW_SUPPORT_TRAIN_COUNT: Final[int] = 30


def _load_frozen_modes() -> dict[str, list[dict[str, object]]]:
    path = Path(__file__).with_name("meter_v2_runtime_slot_modes_v1.json")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != MODE_DATA_SHA256:
        raise RuntimeError("Meter V2 frozen runtime slot mode-data SHA mismatch")
    payload = json.loads(raw.decode("utf-8"))
    if set(payload) != {str(i) for i in range(1, 9)}:
        raise RuntimeError("Meter V2 frozen runtime slot mode inventory mismatch")
    return payload


_MODE_DATA: Final[dict[str, list[dict[str, object]]]] = _load_frozen_modes()


class MeterV2RuntimeSlotError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MeterRuntimeDigitModeProposal:
    measure_number: int
    mode_index: int
    train_count: int
    low_support: bool
    refined_x_center: float
    numerator_bbox: MeterBox
    denominator_bbox: MeterBox


def meter_v2_runtime_slot_profile_fingerprint() -> str:
    payload = {
        "version": "meter-v2-runtime-slot-localizer-v1",
        "m3c2_anchor_modes_sha256": M3C2_ANCHOR_MODES_SHA256,
        "m4a_dataset_manifest_sha256": M4A_DATASET_MANIFEST_SHA256,
        "mode_data_sha256": MODE_DATA_SHA256,
        "roi_size": list(RUNTIME_ROI_SIZE),
        "ink": "sum(255-gray)-p20-column-baseline",
        "ink_baseline_quantile_milli": INK_BASELINE_QUANTILE_MILLI,
        "search_half_width_multiplier_milli": SEARCH_HALF_WIDTH_MULTIPLIER_MILLI,
        "role_box": "train-median-y-width-height",
        "mode_count_max": 2,
        "ground_truth_at_runtime": False,
        "d11_bbox_at_runtime": False,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return sha256(raw).hexdigest()


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeterV2RuntimeSlotError("runtime slot geometry must be finite numeric")
    number = float(value)
    if not math.isfinite(number):
        raise MeterV2RuntimeSlotError("runtime slot geometry must be finite numeric")
    return number


def _percentile_floor(values: list[float], milli: int) -> float:
    if not values or not 0 <= milli <= 1000:
        raise MeterV2RuntimeSlotError("runtime slot projection percentile is invalid")
    ordered = sorted(values)
    return ordered[((len(ordered) - 1) * milli) // 1000]


def _refined_x_center(gray: Image.Image, mode: dict[str, object]) -> float | None:
    zone = mode["zone"]
    if not isinstance(zone, list) or len(zone) != 4:
        raise MeterV2RuntimeSlotError("frozen runtime slot zone is malformed")
    zx0, zy0, zx1, zy1 = (_finite(v) for v in zone)
    prior = _finite(mode["prior_center_x"])
    width = _finite(mode["median_meter_width"])
    if not zx0 < zx1 or not zy0 < zy1 or width <= 0:
        raise MeterV2RuntimeSlotError("frozen runtime slot geometry is invalid")
    half = width * SEARCH_HALF_WIDTH_MULTIPLIER_MILLI / 1000.0
    x0 = max(0, min(gray.width - 1, int(math.floor(max(zx0, prior - half)))))
    x1 = max(x0 + 1, min(gray.width, int(math.ceil(min(zx1, prior + half)))))
    y0 = max(0, min(gray.height - 1, int(math.floor(zy0))))
    y1 = max(y0 + 1, min(gray.height, int(math.ceil(zy1))))
    pixels = gray.load()
    darkness = [sum(255.0 - float(pixels[x, y]) for y in range(y0, y1)) for x in range(x0, x1)]
    baseline = _percentile_floor(darkness, INK_BASELINE_QUANTILE_MILLI)
    weights = [max(0.0, value - baseline) for value in darkness]
    total = sum(weights)
    if total <= 0.0:
        return None
    return sum(float(x0 + i) * weight for i, weight in enumerate(weights)) / total


def _role_box(center_x: float, role: object, gray: Image.Image) -> MeterBox:
    if not isinstance(role, list) or len(role) != 3:
        raise MeterV2RuntimeSlotError("frozen role template is malformed")
    cy, width, height = (_finite(v) for v in role)
    if width <= 0 or height <= 0:
        raise MeterV2RuntimeSlotError("frozen role template size is invalid")
    box = MeterBox(center_x-width/2, cy-height/2, center_x+width/2, cy+height/2)
    if not (0 <= box.x0 < box.x1 <= gray.width and 0 <= box.y0 < box.y1 <= gray.height):
        raise MeterV2RuntimeSlotError("runtime slot proposal falls outside the frozen ROI")
    return box


def propose_meter_v2_runtime_digit_modes_v1(image: Image.Image, *, measure_number: int) -> tuple[MeterRuntimeDigitModeProposal, ...]:
    if not isinstance(image, Image.Image):
        raise MeterV2RuntimeSlotError("runtime slot source must be a PIL image")
    if image.size != RUNTIME_ROI_SIZE:
        raise MeterV2RuntimeSlotError("runtime slot source must be the frozen 256x192 Meter ROI")
    if isinstance(measure_number, bool) or not isinstance(measure_number, int):
        raise MeterV2RuntimeSlotError("measure_number must be an integer")
    modes = _MODE_DATA.get(str(measure_number))
    if modes is None:
        raise MeterV2RuntimeSlotError("measure_number is outside the frozen 1..8 anchor surface")
    gray = image.convert("L")
    result = []
    for mode in modes:
        cx = _refined_x_center(gray, mode)
        if cx is None:
            continue
        train_count = int(mode["train_count"])
        result.append(MeterRuntimeDigitModeProposal(
            measure_number=measure_number,
            mode_index=int(mode["mode_index"]),
            train_count=train_count,
            low_support=bool(mode["support_warning"]) or train_count < LOW_SUPPORT_TRAIN_COUNT,
            refined_x_center=cx,
            numerator_bbox=_role_box(cx, mode["numerator"], gray),
            denominator_bbox=_role_box(cx, mode["denominator"], gray),
        ))
    return tuple(result)


def runtime_slot_policy_train_derived_only() -> bool:
    return True


def runtime_digit_bbox_localization_accepted() -> bool:
    return False


def resolver_connection_allowed() -> bool:
    return False
