"""Deterministic system-membership stage for accepted staff geometry.

The upstream multi-staff detector intentionally discovers staff instances only;
its provisional ``system-1`` assignment is not treated as musical/system truth.
This module replaces that provisional membership under an explicit grouping
policy and fails closed whenever image-only membership is underdetermined.

No model, checkpoint, TRAIN/VALIDATION/TEST split, meter semantics, or resolver
state is accessed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from typing import Final

from PIL import Image, UnidentifiedImageError

from .runtime_geometry_engine_contract import (
    BoxContract,
    PageGeometryContract,
    StaffGeometryContract,
    SystemGeometryContract,
)


SYSTEM_GROUPER_V1_VERSION: Final[str] = "runtime-system-grouper-v1"
SYSTEM_GROUPER_POLICIES: Final[tuple[str, ...]] = (
    "auto-v1",
    "monostaff-v1",
    "fixed-two-staff-v1",
)

# The automatic path uses only a strict dark-column continuity observation in a
# bounded corridor around the already-detected staff-line left endpoints.  These
# constants are a frozen detection contract, not values fitted to TRAIN,
# VALIDATION, TEST, or the System Geometry evidence fixtures.
CONNECTOR_DARK_THRESHOLD: Final[int] = 128
MIN_CONNECTOR_COVERAGE_MILLI: Final[int] = 800
CONNECTOR_CORRIDOR_HALF_WIDTH_SPACINGS_MILLI: Final[int] = 1000

G01_UPSTREAM_GEOMETRY_NOT_ACCEPTED: Final[str] = "G01_UPSTREAM_GEOMETRY_NOT_ACCEPTED"
G02_MEASURE_GEOMETRY_ALREADY_PRESENT: Final[str] = "G02_MEASURE_GEOMETRY_ALREADY_PRESENT"
G03_DECLARED_POLICY_STAFF_COUNT_MISMATCH: Final[str] = "G03_DECLARED_POLICY_STAFF_COUNT_MISMATCH"
G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP: Final[str] = "G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP"
G05_INVALID_STAFF_ORDER: Final[str] = "G05_INVALID_STAFF_ORDER"
G06_RASTER_EVIDENCE_REQUIRED: Final[str] = "G06_RASTER_EVIDENCE_REQUIRED"

SYSTEM_GROUPER_REASON_PRIORITY: Final[tuple[str, ...]] = (
    G01_UPSTREAM_GEOMETRY_NOT_ACCEPTED,
    G02_MEASURE_GEOMETRY_ALREADY_PRESENT,
    G05_INVALID_STAFF_ORDER,
    G03_DECLARED_POLICY_STAFF_COUNT_MISMATCH,
    G06_RASTER_EVIDENCE_REQUIRED,
    G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP,
)


class SystemGrouperV1Error(ValueError):
    """Raised when supplied raster bytes violate the grouping input boundary."""


@dataclass(frozen=True, slots=True)
class SystemGroupingReportV1:
    status: str
    policy: str
    input_geometry_fingerprint: str
    output_geometry_fingerprint: str | None
    primary_reason: str | None = None
    secondary_reasons: tuple[str, ...] = ()
    adjacent_connector_coverages_milli: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("accepted", "ambiguous", "rejected"):
            raise ValueError("unsupported system grouping status")
        if self.policy not in SYSTEM_GROUPER_POLICIES:
            raise ValueError("unsupported system grouping policy")
        _require_sha256("input_geometry_fingerprint", self.input_geometry_fingerprint)
        if self.output_geometry_fingerprint is not None:
            _require_sha256("output_geometry_fingerprint", self.output_geometry_fingerprint)
        reasons = (() if self.primary_reason is None else (self.primary_reason,)) + self.secondary_reasons
        if len(set(reasons)) != len(reasons):
            raise ValueError("system grouping reasons must be unique")
        if any(reason not in SYSTEM_GROUPER_REASON_PRIORITY for reason in reasons):
            raise ValueError("unknown system grouping reason")
        if tuple(sorted(reasons, key=SYSTEM_GROUPER_REASON_PRIORITY.index)) != reasons:
            raise ValueError("system grouping reasons must follow canonical priority")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 1000
            for value in self.adjacent_connector_coverages_milli
        ):
            raise ValueError("connector coverages must be integer milli fractions in [0,1000]")
        if self.status == "accepted":
            if reasons or self.output_geometry_fingerprint is None:
                raise ValueError("accepted grouping requires output fingerprint and no reasons")
        else:
            if not reasons or self.output_geometry_fingerprint is not None:
                raise ValueError("non-accepted grouping requires reasons and no output fingerprint")

    @property
    def active_reasons(self) -> tuple[str, ...]:
        if self.primary_reason is None:
            return ()
        return (self.primary_reason,) + self.secondary_reasons

    def canonical_payload(self) -> dict[str, object]:
        return {
            "version": SYSTEM_GROUPER_V1_VERSION,
            "status": self.status,
            "policy": self.policy,
            "input_geometry_fingerprint": self.input_geometry_fingerprint,
            "output_geometry_fingerprint": self.output_geometry_fingerprint,
            "primary_reason": self.primary_reason,
            "secondary_reasons": list(self.secondary_reasons),
            "adjacent_connector_coverages_milli": list(self.adjacent_connector_coverages_milli),
        }

    def fingerprint(self) -> str:
        return _canonical_sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class SystemGroupingResultV1:
    page: PageGeometryContract | None
    report: SystemGroupingReportV1


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex string")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex string")


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256(raw).hexdigest()


def page_geometry_fingerprint_v1(page: PageGeometryContract) -> str:
    if not isinstance(page, PageGeometryContract):
        raise TypeError("page must be PageGeometryContract")
    payload = {
        "normalized_image_sha256": page.normalized_image_sha256,
        "geometry_config_fingerprint": page.geometry_config_fingerprint,
        "page_width": page.page_width,
        "page_height": page.page_height,
        "transform": {
            "forward": list(page.transform.forward),
            "inverse": list(page.transform.inverse),
        },
        "systems": [
            {
                "system_id": system.system_id,
                "bbox": [
                    system.system_bbox.x_min,
                    system.system_bbox.y_min,
                    system.system_bbox.x_max,
                    system.system_bbox.y_max,
                ],
                "staff_ids": list(system.staff_ids),
            }
            for system in page.systems
        ],
        "staffs": [
            {
                "staff_id": staff.staff_id,
                "system_id": staff.system_id,
                "bbox": [
                    staff.staff_bbox.x_min,
                    staff.staff_bbox.y_min,
                    staff.staff_bbox.x_max,
                    staff.staff_bbox.y_max,
                ],
                "staff_spacing": staff.staff_spacing,
                "lines": [
                    [line.start.x, line.start.y, line.end.x, line.end.y]
                    for line in staff.five_staff_lines
                ],
            }
            for staff in page.staffs
        ],
        "measure_proposals": [
            {
                "measure_id": measure.measure_id,
                "system_id": measure.system_id,
                "staff_id": measure.staff_id,
                "bbox": [measure.bbox.x_min, measure.bbox.y_min, measure.bbox.x_max, measure.bbox.y_max],
                "left": [
                    measure.left_boundary.start.x,
                    measure.left_boundary.start.y,
                    measure.left_boundary.end.x,
                    measure.left_boundary.end.y,
                ],
                "right": [
                    measure.right_boundary.start.x,
                    measure.right_boundary.start.y,
                    measure.right_boundary.end.x,
                    measure.right_boundary.end.y,
                ],
                "status": measure.status,
                "reasons": list(measure.reasons),
            }
            for measure in page.measure_proposals
        ],
        "status": page.status,
        "reasons": list(page.reasons),
    }
    return _canonical_sha256(payload)


def system_grouper_config_fingerprint_v1(*, upstream_fingerprint: str, policy: str) -> str:
    _require_sha256("upstream_fingerprint", upstream_fingerprint)
    if policy not in SYSTEM_GROUPER_POLICIES:
        raise ValueError("unsupported system grouping policy")
    return _canonical_sha256(
        {
            "version": SYSTEM_GROUPER_V1_VERSION,
            "policy": policy,
            "upstream_geometry_config_fingerprint": upstream_fingerprint,
            "ordering": "top-to-bottom-staff-bbox-then-staff-id",
            "auto_multi_staff_guessing": False,
            "connector_dark_threshold": CONNECTOR_DARK_THRESHOLD,
            "min_connector_coverage_milli": MIN_CONNECTOR_COVERAGE_MILLI,
            "connector_corridor_half_width_spacings_milli": CONNECTOR_CORRIDOR_HALF_WIDTH_SPACINGS_MILLI,
            "checkpoint_access": False,
            "optimizer_access": False,
            "train_validation_test_access": False,
            "meter_semantics": False,
        }
    )


def _canonical_reasons(codes: tuple[str, ...]) -> tuple[str, ...]:
    unknown = set(codes) - set(SYSTEM_GROUPER_REASON_PRIORITY)
    if unknown:
        raise ValueError("unknown system grouping reason")
    unique = set(codes)
    return tuple(code for code in SYSTEM_GROUPER_REASON_PRIORITY if code in unique)


def _stop(
    page: PageGeometryContract,
    *,
    policy: str,
    status: str,
    codes: tuple[str, ...],
    connector_coverages_milli: tuple[int, ...] = (),
) -> SystemGroupingResultV1:
    reasons = _canonical_reasons(codes)
    report = SystemGroupingReportV1(
        status=status,
        policy=policy,
        input_geometry_fingerprint=page_geometry_fingerprint_v1(page),
        output_geometry_fingerprint=None,
        primary_reason=reasons[0],
        secondary_reasons=reasons[1:],
        adjacent_connector_coverages_milli=connector_coverages_milli,
    )
    return SystemGroupingResultV1(page=None, report=report)


def _staff_order_is_canonical(staffs: tuple[StaffGeometryContract, ...]) -> bool:
    keys = tuple((staff.staff_bbox.y_min, staff.staff_bbox.y_max, staff.staff_id) for staff in staffs)
    return keys == tuple(sorted(keys)) and len({staff.staff_id for staff in staffs}) == len(staffs)


def _decode_page_gray_png(data: bytes, page: PageGeometryContract) -> Image.Image:
    if not isinstance(data, bytes) or not data:
        raise SystemGrouperV1Error("normalized image bytes must be non-empty bytes")
    if sha256(data).hexdigest() != page.normalized_image_sha256:
        raise SystemGrouperV1Error("normalized image SHA does not match page geometry")
    try:
        with Image.open(BytesIO(data)) as opened:
            if opened.format != "PNG":
                raise SystemGrouperV1Error("system grouper accepts normalized PNG only")
            opened.load()
            if opened.mode != "L":
                raise SystemGrouperV1Error("system grouper requires gray8 normalized PNG")
            if opened.size != (page.page_width, page.page_height):
                raise SystemGrouperV1Error("normalized PNG dimensions do not match page geometry")
            return opened.copy()
    except SystemGrouperV1Error:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise SystemGrouperV1Error("normalized PNG cannot be decoded safely") from exc


def _connector_coverage_milli(
    image: Image.Image,
    upper: StaffGeometryContract,
    lower: StaffGeometryContract,
) -> int:
    y0 = max(0, int(round(upper.staff_bbox.y_max)))
    y1 = min(image.height, int(round(lower.staff_bbox.y_min)))
    if y1 <= y0:
        return 0
    spacing = (upper.staff_spacing + lower.staff_spacing) / 2.0
    half_width = spacing * CONNECTOR_CORRIDOR_HALF_WIDTH_SPACINGS_MILLI / 1000.0
    left_anchor = (upper.staff_bbox.x_min + lower.staff_bbox.x_min) / 2.0
    x0 = max(0, int(round(left_anchor - half_width)))
    x1 = min(image.width - 1, int(round(left_anchor + half_width)))
    if x1 < x0:
        return 0
    pixels = image.load()
    height = y1 - y0
    best_dark = 0
    for x in range(x0, x1 + 1):
        dark = sum(
            1
            for y in range(y0, y1)
            if int(pixels[x, y]) <= CONNECTOR_DARK_THRESHOLD
        )
        if dark > best_dark:
            best_dark = dark
    return min(1000, best_dark * 1000 // height)


def _auto_membership_from_raster(
    page: PageGeometryContract,
    normalized_png: bytes | None,
) -> tuple[tuple[tuple[int, ...], ...] | None, tuple[int, ...], str | None]:
    if len(page.staffs) == 1:
        return ((0,),), (), None
    if normalized_png is None:
        return None, (), G06_RASTER_EVIDENCE_REQUIRED
    image = _decode_page_gray_png(normalized_png, page)
    coverages = tuple(
        _connector_coverage_milli(image, upper, lower)
        for upper, lower in zip(page.staffs, page.staffs[1:])
    )
    connected = tuple(value >= MIN_CONNECTOR_COVERAGE_MILLI for value in coverages)

    # Auto-v1 accepts only a positively observed, continuous connector chain.
    # Absence of a connector is NOT treated as proof of a system boundary:
    # a valid grouped score can omit/lose that visual cue.  Mixed or all-absent
    # patterns therefore stay AMBIGUOUS rather than being split by guesswork.
    if connected and all(connected):
        return (tuple(range(len(page.staffs))),), coverages, None
    return None, coverages, G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP


def _membership_for_declared_policy(
    staff_count: int,
    policy: str,
) -> tuple[tuple[int, ...], ...] | None:
    if policy == "monostaff-v1":
        return tuple((index,) for index in range(staff_count))
    if policy == "fixed-two-staff-v1":
        if staff_count == 0 or staff_count % 2:
            return None
        return tuple(tuple(range(index, index + 2)) for index in range(0, staff_count, 2))
    raise ValueError("unsupported declared system grouping policy")


def group_staffs_into_systems_v1(
    page: PageGeometryContract,
    *,
    normalized_png: bytes | None = None,
    policy: str = "auto-v1",
) -> SystemGroupingResultV1:
    """Rebind accepted staff geometry to deterministic system membership.

    ``auto-v1`` accepts one staff directly, and for multiple staffs accepts only
    a strict raster-observed connector chain spanning every adjacent staff gap.
    It never interprets missing connector ink as proof of a system boundary.
    Under-determined layouts therefore return AMBIGUOUS instead of guessing.

    ``monostaff-v1`` is the explicit current ScoreMosaic V1 policy: every staff
    instance is a distinct system. ``fixed-two-staff-v1`` is an explicit
    caller-declared grand-staff policy: adjacent ordered staffs are paired, and
    odd counts fail closed. Declared policies are input contracts; neither is
    inferred from SVG metadata, spacing thresholds, Meter, or model output.
    """
    if not isinstance(page, PageGeometryContract):
        raise TypeError("page must be PageGeometryContract")
    if policy not in SYSTEM_GROUPER_POLICIES:
        raise ValueError("unsupported system grouping policy")

    if page.status != "accepted" or not page.staffs:
        return _stop(
            page,
            policy=policy,
            status="rejected",
            codes=(G01_UPSTREAM_GEOMETRY_NOT_ACCEPTED,),
        )
    if page.measure_proposals:
        return _stop(
            page,
            policy=policy,
            status="rejected",
            codes=(G02_MEASURE_GEOMETRY_ALREADY_PRESENT,),
        )
    if not _staff_order_is_canonical(page.staffs):
        return _stop(
            page,
            policy=policy,
            status="ambiguous",
            codes=(G05_INVALID_STAFF_ORDER,),
        )

    connector_coverages: tuple[int, ...] = ()
    if policy == "auto-v1":
        membership, connector_coverages, reason = _auto_membership_from_raster(page, normalized_png)
        if membership is None:
            assert reason is not None
            return _stop(
                page,
                policy=policy,
                status="ambiguous",
                codes=(reason,),
                connector_coverages_milli=connector_coverages,
            )
    else:
        membership = _membership_for_declared_policy(len(page.staffs), policy)
        if membership is None:
            return _stop(
                page,
                policy=policy,
                status="ambiguous",
                codes=(G03_DECLARED_POLICY_STAFF_COUNT_MISMATCH,),
            )

    rebound_staffs: list[StaffGeometryContract] = []
    systems: list[SystemGeometryContract] = []
    for system_index, member_indices in enumerate(membership, start=1):
        system_id = f"system-{system_index}"
        members = tuple(page.staffs[index] for index in member_indices)
        rebound_members = tuple(
            StaffGeometryContract(
                staff_id=staff.staff_id,
                system_id=system_id,
                five_staff_lines=staff.five_staff_lines,
                staff_bbox=staff.staff_bbox,
                staff_spacing=staff.staff_spacing,
            )
            for staff in members
        )
        rebound_staffs.extend(rebound_members)
        systems.append(
            SystemGeometryContract(
                system_id=system_id,
                system_bbox=BoxContract(
                    min(staff.staff_bbox.x_min for staff in rebound_members),
                    min(staff.staff_bbox.y_min for staff in rebound_members),
                    max(staff.staff_bbox.x_max for staff in rebound_members),
                    max(staff.staff_bbox.y_max for staff in rebound_members),
                ),
                staff_ids=tuple(staff.staff_id for staff in rebound_members),
            )
        )

    output = PageGeometryContract(
        normalized_image_sha256=page.normalized_image_sha256,
        geometry_config_fingerprint=system_grouper_config_fingerprint_v1(
            upstream_fingerprint=page.geometry_config_fingerprint,
            policy=policy,
        ),
        page_width=page.page_width,
        page_height=page.page_height,
        transform=page.transform,
        systems=tuple(systems),
        staffs=tuple(rebound_staffs),
        measure_proposals=(),
        status="accepted",
    )
    report = SystemGroupingReportV1(
        status="accepted",
        policy=policy,
        input_geometry_fingerprint=page_geometry_fingerprint_v1(page),
        output_geometry_fingerprint=page_geometry_fingerprint_v1(output),
        adjacent_connector_coverages_milli=connector_coverages,
    )
    return SystemGroupingResultV1(page=output, report=report)
