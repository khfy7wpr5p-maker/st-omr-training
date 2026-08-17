"""M4-E3I bounded two-candidate canonical D10 meter ROI recovery.

This module is development-only adapter infrastructure.  It does not train a
model, tune a threshold, open TEST, or promote a checkpoint.  Its two jobs are:

1. recover at most two deterministic page-space measure-start candidates from
   already-decoded five-line staff geometry; and
2. render every candidate through the *existing* Stage 7-D10 crop/resize/pad
   implementation so inference ROIs cannot silently drift from D10 training
   ROIs.

The only learned scalar used by candidate recovery is the first-measure offset
frozen from the TRAIN-only M4-E3G/V6 audit.  VALIDATION is not an input to this
module.  M3-C2 local meter-zone modes are intentionally not interpreted as
page-space measure-start anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from statistics import median
from typing import Final, Mapping, Sequence

from . import stage7d10_local_roi_derivatives as _d10
from .stage7d9_structure_refinement_contract import METER_ROI


M4_E3I_VERSION: Final[str] = "m4-e3i-bounded-two-candidate-canonical-d10-meter-roi-v1"
MAX_CANDIDATES_PER_SYSTEM: Final[int] = 2
D11_PROPOSAL_THRESHOLD: Final[float] = 0.90
SPECIALIST_THRESHOLDS: Final[tuple[tuple[str, float], ...]] = (
    ("2", 0.48),
    ("3", 0.60),
    ("4", 0.47),
)
FROZEN_D11_CHECKPOINT_SHA256: Final[str] = (
    "cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3"
)
FROZEN_D7_CHECKPOINT_SHA256: Final[str] = (
    "5f009ca8ba68d38497a7dd25590d4dd98c537f20c5d5525bf66e288afbf417dc"
)


class M4E3IAdapterError(ValueError):
    """Raised when bounded adapter geometry fails closed."""


def _finite(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise M4E3IAdapterError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise M4E3IAdapterError(f"{name} must be finite")
    return result


def _hex64(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise M4E3IAdapterError(f"{name} must be lowercase SHA-256 hex")
    return value


@dataclass(frozen=True, slots=True)
class FrozenTrainAnchorPolicy:
    """TRAIN-derived scalar provenance plus fixed bounded geometry rules."""

    source_stage: str
    source_result_sha256: str
    derivation_split: str
    first_measure_offset_staff_spaces: float
    candidate_methods: tuple[str, ...]
    max_candidates: int

    def __post_init__(self) -> None:
        if self.derivation_split != "train":
            raise M4E3IAdapterError("anchor policy must be derived from TRAIN only")
        _hex64("source_result_sha256", self.source_result_sha256)
        offset = _finite(
            "first_measure_offset_staff_spaces",
            self.first_measure_offset_staff_spaces,
        )
        if not -2.0 <= offset <= 2.0:
            raise M4E3IAdapterError("TRAIN anchor offset is outside bounded geometry range")
        if self.candidate_methods != (
            "median_five_line_left",
            "coverage_min_five_line_left",
        ):
            raise M4E3IAdapterError("candidate methods differ from frozen E3I policy")
        if self.max_candidates != MAX_CANDIDATES_PER_SYSTEM:
            raise M4E3IAdapterError("candidate count must stay bounded at two")


FROZEN_TRAIN_ANCHOR_POLICY: Final[FrozenTrainAnchorPolicy] = FrozenTrainAnchorPolicy(
    source_stage="M4-E3G-DEVELOPMENT-ONLY-FIVE-LINE-STAFF-GEOMETRY-RECOVERY",
    source_result_sha256=(
        "db9536b983c7aabee30243696fb88e8ea74016b4600a70b93ad630562b8b86ec"
    ),
    derivation_split="train",
    first_measure_offset_staff_spaces=-0.06619667590040451,
    candidate_methods=(
        "median_five_line_left",
        "coverage_min_five_line_left",
    ),
    max_candidates=MAX_CANDIDATES_PER_SYSTEM,
)


@dataclass(frozen=True, slots=True)
class MeasureStartCandidate:
    anchor_x: float
    method: str
    staff_left_x: float
    staff_spacing: float

    def __post_init__(self) -> None:
        for name, value in (
            ("anchor_x", self.anchor_x),
            ("staff_left_x", self.staff_left_x),
            ("staff_spacing", self.staff_spacing),
        ):
            _finite(name, value)
        if self.staff_spacing <= 0:
            raise M4E3IAdapterError("staff_spacing must be positive")
        if self.method not in FROZEN_TRAIN_ANCHOR_POLICY.candidate_methods:
            raise M4E3IAdapterError("unknown bounded anchor method")


@dataclass(frozen=True, slots=True)
class CanonicalMeterRoiCandidate:
    anchor: MeasureStartCandidate
    image_bytes: bytes
    image_sha256: str
    transform: object

    def __post_init__(self) -> None:
        if not isinstance(self.image_bytes, bytes) or not self.image_bytes:
            raise M4E3IAdapterError("candidate ROI image bytes must be non-empty")
        _hex64("image_sha256", self.image_sha256)
        if sha256(self.image_bytes).hexdigest() != self.image_sha256:
            raise M4E3IAdapterError("candidate ROI SHA-256 mismatch")


def _canonical_staff_line(name: str, value: object) -> dict[str, dict[str, float]]:
    """Delegate line validation to D10's canonical geometry validator."""
    try:
        return _d10._line(name, value)
    except _d10.Stage7D10DerivativeError as exc:
        raise M4E3IAdapterError(str(exc)) from exc


def _five_line_geometry(
    five_staff_lines: Sequence[object],
) -> tuple[tuple[float, ...], float]:
    if (
        not isinstance(five_staff_lines, Sequence)
        or isinstance(five_staff_lines, (str, bytes, bytearray))
        or len(five_staff_lines) != 5
    ):
        raise M4E3IAdapterError("exactly five decoded staff lines are required")

    lefts: list[float] = []
    centers_y: list[float] = []
    for index, raw_line in enumerate(five_staff_lines):
        line = _canonical_staff_line(f"five_staff_lines[{index}]", raw_line)
        start = line["start"]
        end = line["end"]
        lefts.append(min(start["x"], end["x"]))
        centers_y.append((start["y"] + end["y"]) / 2.0)

    ordered_y = sorted(centers_y)
    gaps = [right - left for left, right in zip(ordered_y, ordered_y[1:])]
    if len(gaps) != 4 or any(not math.isfinite(gap) or gap <= 0 for gap in gaps):
        raise M4E3IAdapterError("five staff lines must have strictly increasing Y centers")
    spacing = float(median(gaps))
    if not math.isfinite(spacing) or spacing <= 0:
        raise M4E3IAdapterError("decoded staff spacing must be finite and positive")
    return tuple(lefts), spacing


def recover_measure_start_candidates(
    five_staff_lines: Sequence[object],
    *,
    policy: FrozenTrainAnchorPolicy = FROZEN_TRAIN_ANCHOR_POLICY,
) -> tuple[MeasureStartCandidate, ...]:
    """Recover one or two deterministic measure-start anchors.

    Candidate 0 preserves V6's robust median five-line left edge. Candidate 1
    adds a bounded coverage fallback using the minimum real left endpoint among
    the same five decoded lines. Both receive the same TRAIN-only frozen offset.
    No D11 score, specialist score, VALIDATION label, or TEST data participates
    in candidate generation.
    """
    if not isinstance(policy, FrozenTrainAnchorPolicy):
        raise M4E3IAdapterError("policy must be FrozenTrainAnchorPolicy")

    lefts, staff_spacing = _five_line_geometry(five_staff_lines)
    staff_lefts = (
        float(median(lefts)),
        float(min(lefts)),
    )
    offset_pixels = policy.first_measure_offset_staff_spaces * staff_spacing

    result: list[MeasureStartCandidate] = []
    for method, staff_left_x in zip(policy.candidate_methods, staff_lefts):
        anchor_x = staff_left_x + offset_pixels
        candidate = MeasureStartCandidate(
            anchor_x=anchor_x,
            method=method,
            staff_left_x=staff_left_x,
            staff_spacing=staff_spacing,
        )
        if any(math.isclose(candidate.anchor_x, prior.anchor_x, rel_tol=0.0, abs_tol=1e-9) for prior in result):
            continue
        result.append(candidate)

    if not 1 <= len(result) <= policy.max_candidates:
        raise M4E3IAdapterError("bounded candidate recovery produced invalid count")
    return tuple(result)


def render_canonical_meter_roi(
    image_bytes: bytes,
    *,
    staff_bbox: Mapping[str, object],
    candidate: MeasureStartCandidate,
) -> CanonicalMeterRoiCandidate:
    """Render one candidate with the existing D10 crop and render functions.

    A minimal synthetic measure box supplies ``x_min=candidate.anchor_x`` only
    because D10's meter policy anchors on measure start. The crop transform and
    image rendering themselves are not reimplemented here: both calls delegate
    directly to Stage 7-D10.
    """
    if not isinstance(candidate, MeasureStartCandidate):
        raise M4E3IAdapterError("candidate must be MeasureStartCandidate")
    try:
        staff = _d10._box("staff_instance_bbox", staff_bbox)
        image = _d10._open_grayscale_png(image_bytes)
        measure_width = max(1.0, candidate.staff_spacing)
        measure_bbox = {
            "x_min": candidate.anchor_x,
            "y_min": staff["y_min"],
            "x_max": candidate.anchor_x + measure_width,
            "y_max": staff["y_max"],
        }
        transform = _d10._crop_transform(
            measure_bbox=measure_bbox,
            staff_bbox=staff,
            staff_spacing=candidate.staff_spacing,
            image_width=image.width,
            image_height=image.height,
            policy=METER_ROI,
        )
        roi_bytes = _d10._render_roi(image, transform)
    except _d10.Stage7D10DerivativeError as exc:
        raise M4E3IAdapterError(str(exc)) from exc

    return CanonicalMeterRoiCandidate(
        anchor=candidate,
        image_bytes=roi_bytes,
        image_sha256=sha256(roi_bytes).hexdigest(),
        transform=transform,
    )


def recover_canonical_meter_roi_candidates(
    image_bytes: bytes,
    *,
    staff_bbox: Mapping[str, object],
    five_staff_lines: Sequence[object],
    policy: FrozenTrainAnchorPolicy = FROZEN_TRAIN_ANCHOR_POLICY,
) -> tuple[CanonicalMeterRoiCandidate, ...]:
    """Recover at most two anchors and render each through canonical D10."""
    anchors = recover_measure_start_candidates(five_staff_lines, policy=policy)
    rois = tuple(
        render_canonical_meter_roi(
            image_bytes,
            staff_bbox=staff_bbox,
            candidate=candidate,
        )
        for candidate in anchors
    )
    if not 1 <= len(rois) <= MAX_CANDIDATES_PER_SYSTEM:
        raise M4E3IAdapterError("canonical ROI candidate count escaped E3I bound")
    return rois
