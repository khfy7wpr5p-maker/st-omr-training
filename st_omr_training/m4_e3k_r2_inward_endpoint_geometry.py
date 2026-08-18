"""M4-E3K-R2 inward endpoint geometry recovery.

R2 is a one-variable recovery experiment built on the frozen E3K proposal
surface. R1 showed that the dominant TRAIN miss reason was the lower endpoint
window asking a true staff-spanning barline to continue outside the staff.

R2 changes only endpoint-window direction:

* top endpoint: sample from the top staff line inward/downward;
* bottom endpoint: sample from the bottom staff line inward/upward.

The total sampled endpoint span is preserved: the V1 symmetric window used a
half-width on both sides, so R2 uses twice that half-width on the inward side.
All frozen thresholds, Otsu logic, staff-slope recovery, probe radius,
clustering, candidate bound, and recall tolerances are reused unchanged.

This module is development-only. It does not load model weights, train, tune a
threshold, open VALIDATION/TEST, or authorize D11/promotion.
"""

from __future__ import annotations

import math
from typing import Final, Mapping, Sequence

from PIL import Image

from .m4_e3k_boundary_proposals import (
    BoundaryProposal,
    BoundaryProposalConfig,
    BoundaryProposalResult,
    FROZEN_E3K_CONFIG,
    _bbox,
    _clusters,
    _fail,
    _finite_number,
    _otsu_threshold,
    _row_has_ink,
    _staff_lines,
    _vertical_window_coverage,
)


STAGE: Final[str] = "M4-E3K-R2-INWARD-ENDPOINT-GEOMETRY-RECOVERY"


def _column_evidence_inward(
    image: Image.Image,
    *,
    anchor_x: int,
    x_left: int,
    x_right: int,
    staff_lines: tuple[object, ...],
    staff_slope: float,
    staff_spacing: float,
    threshold: int,
    config: BoundaryProposalConfig,
) -> tuple[float, float, float, float]:
    """Return frozen E3K evidence with endpoint windows directed inward.

    V1 sampled ``[endpoint-half, endpoint+half]``. R2 preserves the same
    nominal total span ``2*half`` but places all of it inside the staff. This
    isolates window direction instead of simultaneously shortening the test.
    """

    radius = max(
        1,
        int(round(staff_spacing * config.horizontal_probe_radius_staff_spaces)),
    )
    top_y = staff_lines[0].y_at(float(anchor_x))
    bottom_y = staff_lines[-1].y_at(float(anchor_x))
    if not top_y < bottom_y:
        _fail("local staff span is inverted")
    center_y = (top_y + bottom_y) / 2.0
    core_top = max(0, int(math.floor(top_y)))
    core_bottom = min(image.height - 1, int(math.ceil(bottom_y)))
    if core_bottom - core_top < 2:
        _fail("staff band is too short for E3K-R2 vertical evidence")

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

    endpoint_half = max(
        1,
        int(round(staff_spacing * config.endpoint_half_window_staff_spaces)),
    )
    endpoint_span = 2 * endpoint_half
    top_anchor = int(round(top_y))
    bottom_anchor = int(round(bottom_y))

    # One variable only: direction. Keep the old symmetric window's nominal
    # total span, but move it entirely into the staff rather than asking the
    # barline to continue above/below its true staff-line endpoints.
    top0 = max(0, top_anchor)
    top1 = min(image.height - 1, top_anchor + endpoint_span)
    bottom0 = max(0, bottom_anchor - endpoint_span)
    bottom1 = min(image.height - 1, bottom_anchor)
    if top1 < top0 or bottom1 < bottom0:
        _fail("R2 inward endpoint window is inverted")

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


def propose_measure_boundaries_r2(
    image: Image.Image,
    *,
    staff_bbox: Mapping[str, object],
    five_staff_lines: Sequence[object],
    staff_spacing: float,
    system_bbox: Mapping[str, object] | None = None,
    config: BoundaryProposalConfig = FROZEN_E3K_CONFIG,
) -> BoundaryProposalResult:
    """Generate R2 proposals while preserving all non-endpoint E3K policy."""

    if not isinstance(image, Image.Image) or image.mode != "L":
        _fail("E3K-R2 source image must be a PIL grayscale L image")
    if image.width < 8 or image.height < 8:
        _fail("E3K-R2 source image is too small")
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
        _fail("E3K-R2 x search surface is too narrow")

    reference_x = (search_x0 + search_x1) / 2.0
    reference_top = lines[0].y_at(reference_x)
    reference_bottom = lines[-1].y_at(reference_x)
    otsu_top = max(
        0,
        int(math.floor(min(staff_y0, reference_top) - spacing * 0.5)),
    )
    otsu_bottom = min(
        image.height,
        int(math.ceil(max(staff_y1, reference_bottom) + spacing * 0.5)),
    )
    if otsu_bottom - otsu_top < 3:
        _fail("E3K-R2 Otsu staff surface is too short")
    threshold = _otsu_threshold(image, (x_left, otsu_top, x_right, otsu_bottom))

    evidence: dict[int, tuple[float, float, float, float]] = {}
    active: list[int] = []
    for x in range(x_left, x_right):
        item = _column_evidence_inward(
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

    maximum_gap = max(
        1,
        int(round(spacing * config.cluster_gap_staff_spaces)),
    )
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
            "E3K-R2 proposal count exceeded the frozen per-system bound; "
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
