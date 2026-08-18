"""M4-E3K deterministic measure-boundary proposal recovery.

This development-only module generates high-recall vertical-stroke proposals
inside one known staff/system span. It does not classify a proposal as a
barline, does not load D7/D11 checkpoints, and does not touch TEST. The next
stage may feed these bounded proposals to the frozen D11 local barline refiner.

The proposal surface is intentionally geometry-first:

    grayscale page + accepted staff geometry
        -> deterministic Otsu ink threshold
        -> vertical support through the first-to-fifth staff-line band
        -> top/bottom endpoint support
        -> x clustering
        -> bounded candidate list

False-positive stems are acceptable at this stage; false-negative measure
boundaries are not. Candidate count is therefore never silently top-k pruned.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Mapping, Sequence

from PIL import Image


STAGE: Final[str] = "M4-E3K-DETERMINISTIC-MEASURE-BOUNDARY-PROPOSALS"
EVALUATION_TOLERANCES_STAFF_SPACES: Final[tuple[float, ...]] = (0.5, 1.0, 2.0)


class M4E3KBoundaryProposalError(RuntimeError):
    """Raised when E3K geometry or proposal invariants fail closed."""


@dataclass(frozen=True, slots=True)
class BoundaryProposalConfig:
    """Frozen V1 geometry policy; values are staff-spacing normalized."""

    horizontal_probe_radius_staff_spaces: float = 0.10
    endpoint_half_window_staff_spaces: float = 0.30
    minimum_vertical_coverage: float = 0.45
    cluster_gap_staff_spaces: float = 0.20
    maximum_proposals_per_system: int = 128

    def __post_init__(self) -> None:
        finite_positive = (
            self.horizontal_probe_radius_staff_spaces,
            self.endpoint_half_window_staff_spaces,
            self.cluster_gap_staff_spaces,
        )
        if any(not math.isfinite(value) or value <= 0 for value in finite_positive):
            raise ValueError("E3K normalized geometry values must be finite and positive")
        if (
            not math.isfinite(self.minimum_vertical_coverage)
            or not 0.0 < self.minimum_vertical_coverage <= 1.0
        ):
            raise ValueError("minimum_vertical_coverage must be in (0,1]")
        if (
            not isinstance(self.maximum_proposals_per_system, int)
            or isinstance(self.maximum_proposals_per_system, bool)
            or not 1 <= self.maximum_proposals_per_system <= 512
        ):
            raise ValueError("maximum_proposals_per_system is outside E3K bounds")


FROZEN_E3K_CONFIG: Final[BoundaryProposalConfig] = BoundaryProposalConfig()


@dataclass(frozen=True, slots=True)
class BoundaryProposal:
    x: float
    score: float
    vertical_coverage: float
    top_supported: bool
    bottom_supported: bool
    cluster_left: int
    cluster_right: int

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.x, self.score, self.vertical_coverage)
        ):
            raise ValueError("boundary proposal numeric values must be finite")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("boundary proposal score must be in [0,1]")
        if not 0.0 <= self.vertical_coverage <= 1.0:
            raise ValueError("vertical coverage must be in [0,1]")
        if not isinstance(self.top_supported, bool) or not isinstance(self.bottom_supported, bool):
            raise ValueError("endpoint support flags must be bool")
        if self.cluster_left > self.cluster_right:
            raise ValueError("boundary proposal cluster is inverted")


@dataclass(frozen=True, slots=True)
class BoundaryProposalResult:
    stage: str
    otsu_threshold: int
    system_left_x: float
    system_right_x: float
    staff_top_y: float
    staff_bottom_y: float
    staff_spacing: float
    proposals: tuple[BoundaryProposal, ...]


@dataclass(frozen=True, slots=True)
class BoundaryRecallMetrics:
    truth_count: int
    proposal_count: int
    nearest_error_staff_spaces: tuple[float, ...]
    recall_by_tolerance: dict[float, float]
    p50_error_staff_spaces: float
    p95_error_staff_spaces: float


def _fail(message: str) -> None:
    raise M4E3KBoundaryProposalError(message)


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool):
        _fail(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise M4E3KBoundaryProposalError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def _bbox(name: str, value: Mapping[str, object]) -> tuple[float, float, float, float]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be a mapping")
    x0 = _finite_number(f"{name}.x_min", value.get("x_min"))
    y0 = _finite_number(f"{name}.y_min", value.get("y_min"))
    x1 = _finite_number(f"{name}.x_max", value.get("x_max"))
    y1 = _finite_number(f"{name}.y_max", value.get("y_max"))
    if not x0 < x1 or not y0 < y1:
        _fail(f"{name} must have positive extent")
    return x0, y0, x1, y1


def _otsu_threshold(image: Image.Image, bounds: tuple[int, int, int, int]) -> int:
    left, top, right, bottom = bounds
    histogram = [0] * 256
    pixels = image.load()
    count = 0
    for y in range(top, bottom):
        for x in range(left, right):
            histogram[pixels[x, y]] += 1
            count += 1
    if count <= 0:
        _fail("E3K Otsu surface is empty")
    nonzero = [index for index, value in enumerate(histogram) if value]
    if len(nonzero) < 2:
        _fail("E3K Otsu surface has no grayscale contrast")

    total_sum = sum(index * value for index, value in enumerate(histogram))
    background_weight = 0
    background_sum = 0
    best_threshold = nonzero[0]
    best_variance = -1.0
    for threshold in range(256):
        value = histogram[threshold]
        background_weight += value
        background_sum += threshold * value
        foreground_weight = count - background_weight
        if background_weight == 0 or foreground_weight == 0:
            continue
        mean_background = background_sum / background_weight
        mean_foreground = (total_sum - background_sum) / foreground_weight
        variance = (
            background_weight
            * foreground_weight
            * (mean_background - mean_foreground) ** 2
        )
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return int(best_threshold)


def _any_ink(
    pixels: object,
    *,
    x0: int,
    x1: int,
    y: int,
    threshold: int,
) -> bool:
    return any(pixels[x, y] <= threshold for x in range(x0, x1 + 1))


def _column_evidence(
    image: Image.Image,
    *,
    x: int,
    x_left: int,
    x_right: int,
    staff_top: float,
    staff_bottom: float,
    staff_spacing: float,
    threshold: int,
    config: BoundaryProposalConfig,
) -> tuple[float, bool, bool, float]:
    radius = max(1, int(round(staff_spacing * config.horizontal_probe_radius_staff_spaces)))
    probe_left = max(x_left, x - radius)
    probe_right = min(x_right - 1, x + radius)
    core_top = max(0, int(math.floor(staff_top)))
    core_bottom = min(image.height, int(math.ceil(staff_bottom)) + 1)
    if core_bottom - core_top < 3:
        _fail("staff band is too short for E3K vertical evidence")

    pixels = image.load()
    supported_rows = sum(
        1
        for y in range(core_top, core_bottom)
        if _any_ink(
            pixels,
            x0=probe_left,
            x1=probe_right,
            y=y,
            threshold=threshold,
        )
    )
    coverage = supported_rows / (core_bottom - core_top)

    endpoint_half = max(1, int(round(staff_spacing * config.endpoint_half_window_staff_spaces)))
    top0 = max(0, int(round(staff_top)) - endpoint_half)
    top1 = min(image.height - 1, int(round(staff_top)) + endpoint_half)
    bottom0 = max(0, int(round(staff_bottom)) - endpoint_half)
    bottom1 = min(image.height - 1, int(round(staff_bottom)) + endpoint_half)

    top_supported = any(
        _any_ink(
            pixels,
            x0=probe_left,
            x1=probe_right,
            y=y,
            threshold=threshold,
        )
        for y in range(top0, top1 + 1)
    )
    bottom_supported = any(
        _any_ink(
            pixels,
            x0=probe_left,
            x1=probe_right,
            y=y,
            threshold=threshold,
        )
        for y in range(bottom0, bottom1 + 1)
    )
    score = min(
        1.0,
        0.8 * coverage
        + 0.1 * float(top_supported)
        + 0.1 * float(bottom_supported),
    )
    return coverage, top_supported, bottom_supported, score


def _clusters(active: Sequence[int], maximum_gap: int) -> tuple[tuple[int, ...], ...]:
    if not active:
        return ()
    result: list[list[int]] = [[active[0]]]
    for x in active[1:]:
        if x - result[-1][-1] <= maximum_gap + 1:
            result[-1].append(x)
        else:
            result.append([x])
    return tuple(tuple(cluster) for cluster in result)


def propose_measure_boundaries(
    image: Image.Image,
    *,
    staff_bbox: Mapping[str, object],
    staff_spacing: float,
    system_bbox: Mapping[str, object] | None = None,
    config: BoundaryProposalConfig = FROZEN_E3K_CONFIG,
) -> BoundaryProposalResult:
    """Generate deterministic vertical boundary proposals for one staff/system.

    `staff_bbox.y_min/y_max` are interpreted as the first/fifth staff-line y
    coordinates, matching the accepted D5/D6 aggregated staff geometry.
    `system_bbox` may narrow the x search surface; when absent, staff x bounds
    are used. No candidate is dropped by ranking.
    """

    if not isinstance(image, Image.Image) or image.mode != "L":
        _fail("E3K source image must be a PIL grayscale L image")
    if image.width < 8 or image.height < 8:
        _fail("E3K source image is too small")
    if not isinstance(config, BoundaryProposalConfig):
        raise TypeError("config must be BoundaryProposalConfig")
    spacing = _finite_number("staff_spacing", staff_spacing)
    if spacing <= 0:
        _fail("staff_spacing must be positive")

    staff_x0, staff_y0, staff_x1, staff_y1 = _bbox("staff_bbox", staff_bbox)
    if system_bbox is None:
        search_x0, search_x1 = staff_x0, staff_x1
    else:
        system_x0, _, system_x1, _ = _bbox("system_bbox", system_bbox)
        search_x0 = max(staff_x0, system_x0)
        search_x1 = min(staff_x1, system_x1)
    if not search_x0 < search_x1:
        _fail("staff/system x intersection is empty")
    if not 0 <= staff_y0 < staff_y1 <= image.height:
        _fail("staff y geometry lies outside the image")

    x_left = max(0, int(math.floor(search_x0)))
    x_right = min(image.width, int(math.ceil(search_x1)))
    if x_right - x_left < 3:
        _fail("E3K x search surface is too narrow")

    otsu_top = max(0, int(math.floor(staff_y0 - spacing * 0.5)))
    otsu_bottom = min(image.height, int(math.ceil(staff_y1 + spacing * 0.5)))
    if otsu_bottom - otsu_top < 3:
        _fail("E3K Otsu staff surface is too short")
    threshold = _otsu_threshold(image, (x_left, otsu_top, x_right, otsu_bottom))

    evidence: dict[int, tuple[float, bool, bool, float]] = {}
    active: list[int] = []
    for x in range(x_left, x_right):
        item = _column_evidence(
            image,
            x=x,
            x_left=x_left,
            x_right=x_right,
            staff_top=staff_y0,
            staff_bottom=staff_y1,
            staff_spacing=spacing,
            threshold=threshold,
            config=config,
        )
        evidence[x] = item
        coverage, top_supported, bottom_supported, _ = item
        if (
            coverage >= config.minimum_vertical_coverage
            and top_supported
            and bottom_supported
        ):
            active.append(x)

    maximum_gap = max(1, int(round(spacing * config.cluster_gap_staff_spaces)))
    clusters = _clusters(active, maximum_gap)
    proposals: list[BoundaryProposal] = []
    for cluster in clusters:
        center = (cluster[0] + cluster[-1]) / 2.0
        peak = min(
            cluster,
            key=lambda x: (-evidence[x][3], abs(x - center), x),
        )
        coverage, top_supported, bottom_supported, score = evidence[peak]
        proposals.append(
            BoundaryProposal(
                x=float(peak),
                score=score,
                vertical_coverage=coverage,
                top_supported=top_supported,
                bottom_supported=bottom_supported,
                cluster_left=cluster[0],
                cluster_right=cluster[-1],
            )
        )

    proposals.sort(key=lambda item: item.x)
    if len(proposals) > config.maximum_proposals_per_system:
        _fail(
            "E3K proposal count exceeded the frozen per-system bound; "
            "candidates are not silently top-k pruned"
        )

    return BoundaryProposalResult(
        stage=STAGE,
        otsu_threshold=threshold,
        system_left_x=float(search_x0),
        system_right_x=float(search_x1),
        staff_top_y=staff_y0,
        staff_bottom_y=staff_y1,
        staff_spacing=spacing,
        proposals=tuple(proposals),
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        _fail("cannot compute percentile of empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def evaluate_boundary_recall(
    proposals: Sequence[BoundaryProposal],
    truth_xs: Sequence[float],
    *,
    staff_spacing: float,
) -> BoundaryRecallMetrics:
    """Read-only object-level recall audit at predeclared staff-space tolerances."""

    spacing = _finite_number("staff_spacing", staff_spacing)
    if spacing <= 0:
        _fail("staff_spacing must be positive")
    if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes, bytearray)):
        _fail("proposals must be a sequence")
    if not isinstance(truth_xs, Sequence) or isinstance(truth_xs, (str, bytes, bytearray)):
        _fail("truth_xs must be a sequence")
    if not truth_xs:
        _fail("truth_xs must be non-empty")
    candidate_xs: list[float] = []
    for proposal in proposals:
        if not isinstance(proposal, BoundaryProposal):
            _fail("proposal sequence contains a non-BoundaryProposal")
        candidate_xs.append(proposal.x)
    if not candidate_xs:
        errors = [math.inf for _ in truth_xs]
    else:
        errors = []
        for index, truth in enumerate(truth_xs):
            truth_x = _finite_number(f"truth_xs[{index}]", truth)
            errors.append(min(abs(candidate - truth_x) for candidate in candidate_xs) / spacing)

    recall_by_tolerance = {
        tolerance: sum(error <= tolerance for error in errors) / len(errors)
        for tolerance in EVALUATION_TOLERANCES_STAFF_SPACES
    }
    finite_errors = [error for error in errors if math.isfinite(error)]
    p50 = math.inf if not finite_errors else _percentile(finite_errors, 0.50)
    p95 = math.inf if not finite_errors else _percentile(finite_errors, 0.95)
    return BoundaryRecallMetrics(
        truth_count=len(truth_xs),
        proposal_count=len(candidate_xs),
        nearest_error_staff_spaces=tuple(errors),
        recall_by_tolerance=recall_by_tolerance,
        p50_error_staff_spaces=p50,
        p95_error_staff_spaces=p95,
    )
