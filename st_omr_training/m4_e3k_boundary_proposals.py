"""M4-E3K deterministic measure-boundary proposal recovery.

This development-only module generates high-recall barline-like stroke proposals
inside one known staff/system span. It does not classify a proposal as a
barline, does not load D7/D11 checkpoints, and does not touch TEST. A later
stage may feed these bounded proposals to the frozen D11 local barline refiner.

The proposal surface is intentionally geometry-first:

    grayscale page + accepted five-line staff geometry
        -> deterministic Otsu ink threshold
        -> recover common staff slope from the five accepted staff lines
        -> scan along the direction perpendicular to the staff
        -> full first-to-fifth-line support + endpoint continuity
        -> x clustering in the staff-centre reference frame
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
    minimum_endpoint_coverage: float = 0.50
    cluster_gap_staff_spaces: float = 0.20
    maximum_absolute_staff_slope: float = 0.35
    maximum_staff_slope_spread: float = 0.05
    maximum_proposals_per_system: int = 128

    def __post_init__(self) -> None:
        finite_positive = (
            self.horizontal_probe_radius_staff_spaces,
            self.endpoint_half_window_staff_spaces,
            self.cluster_gap_staff_spaces,
            self.maximum_absolute_staff_slope,
            self.maximum_staff_slope_spread,
        )
        if any(not math.isfinite(value) or value <= 0 for value in finite_positive):
            raise ValueError("E3K normalized geometry values must be finite and positive")
        for name, value in (
            ("minimum_vertical_coverage", self.minimum_vertical_coverage),
            ("minimum_endpoint_coverage", self.minimum_endpoint_coverage),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0,1]")
        if (
            not isinstance(self.maximum_proposals_per_system, int)
            or isinstance(self.maximum_proposals_per_system, bool)
            or not 1 <= self.maximum_proposals_per_system <= 512
        ):
            raise ValueError("maximum_proposals_per_system is outside E3K bounds")


FROZEN_E3K_CONFIG: Final[BoundaryProposalConfig] = BoundaryProposalConfig()


@dataclass(frozen=True, slots=True)
class _StaffLine:
    x0: float
    y0: float
    x1: float
    y1: float
    slope: float

    def y_at(self, x: float) -> float:
        return self.y0 + self.slope * (x - self.x0)


@dataclass(frozen=True, slots=True)
class BoundaryProposal:
    x: float
    score: float
    vertical_coverage: float
    top_endpoint_coverage: float
    bottom_endpoint_coverage: float
    cluster_left: int
    cluster_right: int

    def __post_init__(self) -> None:
        numeric = (
            self.x,
            self.score,
            self.vertical_coverage,
            self.top_endpoint_coverage,
            self.bottom_endpoint_coverage,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("boundary proposal numeric values must be finite")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("boundary proposal score must be in [0,1]")
        for name, value in (
            ("vertical_coverage", self.vertical_coverage),
            ("top_endpoint_coverage", self.top_endpoint_coverage),
            ("bottom_endpoint_coverage", self.bottom_endpoint_coverage),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.cluster_left > self.cluster_right:
            raise ValueError("boundary proposal cluster is inverted")


@dataclass(frozen=True, slots=True)
class BoundaryProposalResult:
    stage: str
    otsu_threshold: int
    system_left_x: float
    system_right_x: float
    staff_top_y_at_reference: float
    staff_bottom_y_at_reference: float
    staff_spacing: float
    staff_slope: float
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


def _point(name: str, value: object) -> tuple[float, float]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be a mapping")
    return (
        _finite_number(f"{name}.x", value.get("x")),
        _finite_number(f"{name}.y", value.get("y")),
    )


def _staff_lines(
    value: Sequence[object],
    *,
    config: BoundaryProposalConfig,
) -> tuple[tuple[_StaffLine, ...], float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 5
    ):
        _fail("five_staff_lines must contain exactly five line mappings")
    parsed: list[_StaffLine] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            _fail("five_staff_lines entry must be a mapping")
        x0, y0 = _point(f"five_staff_lines[{index}].start", item.get("start"))
        x1, y1 = _point(f"five_staff_lines[{index}].end", item.get("end"))
        if math.isclose(x0, x1, abs_tol=1e-9):
            _fail("staff line cannot be vertical")
        if x1 < x0:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        slope = (y1 - y0) / (x1 - x0)
        if abs(slope) > config.maximum_absolute_staff_slope:
            _fail("staff slope exceeds E3K safety bound")
        parsed.append(_StaffLine(x0=x0, y0=y0, x1=x1, y1=y1, slope=slope))

    common_left = max(line.x0 for line in parsed)
    common_right = min(line.x1 for line in parsed)
    if not common_left < common_right:
        _fail("five staff lines have no common x span")
    reference_x = (common_left + common_right) / 2.0
    parsed.sort(key=lambda line: line.y_at(reference_x))
    slopes = sorted(line.slope for line in parsed)
    median_slope = slopes[len(slopes) // 2]
    if max(abs(line.slope - median_slope) for line in parsed) > config.maximum_staff_slope_spread:
        _fail("five staff lines disagree on common slope")
    y_values = [line.y_at(reference_x) for line in parsed]
    if any(y_values[index + 1] <= y_values[index] for index in range(4)):
        _fail("five staff lines are not strictly ordered")
    return tuple(parsed), median_slope, common_left, common_right


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


def _path_probe_bounds(
    *,
    anchor_x: float,
    center_y: float,
    staff_slope: float,
    y: int,
    radius: int,
    x_left: int,
    x_right: int,
) -> tuple[int, int] | None:
    # A rotated barline is perpendicular to the staff direction. If the staff
    # has dy/dx=m, the perpendicular line has dx/dy=-m.
    path_x = anchor_x - staff_slope * (float(y) - center_y)
    probe_left = max(x_left, int(math.floor(path_x - radius)))
    probe_right = min(x_right - 1, int(math.ceil(path_x + radius)))
    if probe_left > probe_right:
        return None
    return probe_left, probe_right


def _row_has_ink(
    pixels: object,
    *,
    anchor_x: float,
    center_y: float,
    staff_slope: float,
    y: int,
    radius: int,
    x_left: int,
    x_right: int,
    threshold: int,
) -> bool:
    bounds = _path_probe_bounds(
        anchor_x=anchor_x,
        center_y=center_y,
        staff_slope=staff_slope,
        y=y,
        radius=radius,
        x_left=x_left,
        x_right=x_right,
    )
    if bounds is None:
        return False
    probe_left, probe_right = bounds
    return any(pixels[x, y] <= threshold for x in range(probe_left, probe_right + 1))


def _vertical_window_coverage(
    pixels: object,
    *,
    anchor_x: float,
    center_y: float,
    staff_slope: float,
    y0: int,
    y1: int,
    radius: int,
    x_left: int,
    x_right: int,
    threshold: int,
) -> float:
    if y1 < y0:
        _fail("vertical endpoint window is inverted")
    count = y1 - y0 + 1
    supported = sum(
        1
        for y in range(y0, y1 + 1)
        if _row_has_ink(
            pixels,
            anchor_x=anchor_x,
            center_y=center_y,
            staff_slope=staff_slope,
            y=y,
            radius=radius,
            x_left=x_left,
            x_right=x_right,
            threshold=threshold,
        )
    )
    return supported / count


def _column_evidence(
    image: Image.Image,
    *,
    anchor_x: int,
    x_left: int,
    x_right: int,
    staff_lines: tuple[_StaffLine, ...],
    staff_slope: float,
    staff_spacing: float,
    threshold: int,
    config: BoundaryProposalConfig,
) -> tuple[float, float, float, float]:
    radius = max(1, int(round(staff_spacing * config.horizontal_probe_radius_staff_spaces)))
    top_y = staff_lines[0].y_at(float(anchor_x))
    bottom_y = staff_lines[-1].y_at(float(anchor_x))
    if not top_y < bottom_y:
        _fail("local staff span is inverted")
    center_y = (top_y + bottom_y) / 2.0
    core_top = max(0, int(math.floor(top_y)))
    core_bottom = min(image.height - 1, int(math.ceil(bottom_y)))
    if core_bottom - core_top < 2:
        _fail("staff band is too short for E3K vertical evidence")

    pixels = image.load()
    supported_rows = sum(
        1
        for y in range(core_top, core_bottom + 1)
        if _row_has_ink(
            pixels,
            anchor_x=float(anchor_x),
            center_y=center_y,
            staff_slope=staff_slope,
            y=y,
            radius=radius,
            x_left=x_left,
            x_right=x_right,
            threshold=threshold,
        )
    )
    coverage = supported_rows / (core_bottom - core_top + 1)

    endpoint_half = max(1, int(round(staff_spacing * config.endpoint_half_window_staff_spaces)))
    top0 = max(0, int(round(top_y)) - endpoint_half)
    top1 = min(image.height - 1, int(round(top_y)) + endpoint_half)
    bottom0 = max(0, int(round(bottom_y)) - endpoint_half)
    bottom1 = min(image.height - 1, int(round(bottom_y)) + endpoint_half)

    top_coverage = _vertical_window_coverage(
        pixels,
        anchor_x=float(anchor_x),
        center_y=center_y,
        staff_slope=staff_slope,
        y0=top0,
        y1=top1,
        radius=radius,
        x_left=x_left,
        x_right=x_right,
        threshold=threshold,
    )
    bottom_coverage = _vertical_window_coverage(
        pixels,
        anchor_x=float(anchor_x),
        center_y=center_y,
        staff_slope=staff_slope,
        y0=bottom0,
        y1=bottom1,
        radius=radius,
        x_left=x_left,
        x_right=x_right,
        threshold=threshold,
    )
    score = min(
        1.0,
        0.8 * coverage + 0.1 * top_coverage + 0.1 * bottom_coverage,
    )
    return coverage, top_coverage, bottom_coverage, score


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
    five_staff_lines: Sequence[object],
    staff_spacing: float,
    system_bbox: Mapping[str, object] | None = None,
    config: BoundaryProposalConfig = FROZEN_E3K_CONFIG,
) -> BoundaryProposalResult:
    """Generate deterministic boundary proposals for one staff/system.

    The accepted five staff lines define both local staff y positions and the
    shared staff slope. Candidate x is expressed at the local staff-centre
    reference frame; the ink probe follows the direction perpendicular to the
    staff, so the same code supports clean and rotated final-PNG geometry.
    No candidate is dropped by ranking.
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
    lines, staff_slope, common_line_left, common_line_right = _staff_lines(
        five_staff_lines,
        config=config,
    )
    if system_bbox is None:
        search_x0, search_x1 = staff_x0, staff_x1
    else:
        system_x0, _, system_x1, _ = _bbox("system_bbox", system_bbox)
        search_x0 = max(staff_x0, system_x0)
        search_x1 = min(staff_x1, system_x1)
    search_x0 = max(search_x0, common_line_left)
    search_x1 = min(search_x1, common_line_right)
    if not search_x0 < search_x1:
        _fail("staff/system/five-line x intersection is empty")
    if not 0 <= staff_y0 < staff_y1 <= image.height:
        _fail("staff bbox y geometry lies outside the image")

    x_left = max(0, int(math.floor(search_x0)))
    x_right = min(image.width, int(math.ceil(search_x1)))
    if x_right - x_left < 3:
        _fail("E3K x search surface is too narrow")

    reference_x = (search_x0 + search_x1) / 2.0
    reference_top = lines[0].y_at(reference_x)
    reference_bottom = lines[-1].y_at(reference_x)
    otsu_top = max(0, int(math.floor(min(staff_y0, reference_top) - spacing * 0.5)))
    otsu_bottom = min(
        image.height,
        int(math.ceil(max(staff_y1, reference_bottom) + spacing * 0.5)),
    )
    if otsu_bottom - otsu_top < 3:
        _fail("E3K Otsu staff surface is too short")
    threshold = _otsu_threshold(image, (x_left, otsu_top, x_right, otsu_bottom))

    evidence: dict[int, tuple[float, float, float, float]] = {}
    active: list[int] = []
    for x in range(x_left, x_right):
        item = _column_evidence(
            image,
            anchor_x=x,
            x_left=x_left,
            x_right=x_right,
            staff_lines=lines,
            staff_slope=staff_slope,
            staff_spacing=spacing,
            threshold=threshold,
            config=config,
        )
        evidence[x] = item
        coverage, top_coverage, bottom_coverage, _ = item
        if (
            coverage >= config.minimum_vertical_coverage
            and top_coverage >= config.minimum_endpoint_coverage
            and bottom_coverage >= config.minimum_endpoint_coverage
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
        coverage, top_coverage, bottom_coverage, score = evidence[peak]
        proposals.append(
            BoundaryProposal(
                x=float(peak),
                score=score,
                vertical_coverage=coverage,
                top_endpoint_coverage=top_coverage,
                bottom_endpoint_coverage=bottom_coverage,
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
        staff_top_y_at_reference=reference_top,
        staff_bottom_y_at_reference=reference_bottom,
        staff_spacing=spacing,
        staff_slope=staff_slope,
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
