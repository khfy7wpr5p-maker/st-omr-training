"""Deterministic system-local measure boundary construction.

This stage consumes accepted System Grouper output, never regroups staffs, and
uses only normalized raster geometry. System edges are implicit geometric
boundaries; strong internal vertical runs are geometric separator candidates,
not semantic barline recognition.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from typing import Final

from PIL import Image, UnidentifiedImageError

from .runtime_geometry_engine_contract import (
    BoxContract,
    LineSegmentContract,
    MeasureProposalContract,
    PageGeometryContract,
    Point2DContract,
    StaffGeometryContract,
)
from .runtime_system_grouper_v1 import page_geometry_fingerprint_v1


MEASURE_SYSTEM_BOUNDARIES_V2_VERSION: Final[str] = "runtime-measure-system-boundaries-v2"
VERTICAL_DARK_THRESHOLD: Final[int] = 128
MIN_VERTICAL_COVERAGE_MILLI: Final[int] = 800
VERTICAL_ENDPOINT_ANCHOR_TOLERANCE_PX: Final[int] = 1
MAX_BARLINE_CLUSTER_GAP_SPACINGS_MILLI: Final[int] = 1000
EDGE_SNAP_SPACINGS_MILLI: Final[int] = 1000
MAX_CROSS_STAFF_BOUNDARY_DELTA_SPACINGS_MILLI: Final[int] = 500
MIN_MEASURE_WIDTH_SPACINGS_MILLI: Final[int] = 2000

B01_UPSTREAM_GEOMETRY_NOT_ACCEPTED: Final[str] = "B01_UPSTREAM_GEOMETRY_NOT_ACCEPTED"
B02_MEASURE_GEOMETRY_ALREADY_PRESENT: Final[str] = "B02_MEASURE_GEOMETRY_ALREADY_PRESENT"
B03_SYSTEM_STAFF_MEMBERSHIP_INVALID: Final[str] = "B03_SYSTEM_STAFF_MEMBERSHIP_INVALID"
B04_SYSTEM_ORDER_INVALID: Final[str] = "B04_SYSTEM_ORDER_INVALID"
B05_CROSS_STAFF_BOUNDARY_MISMATCH: Final[str] = "B05_CROSS_STAFF_BOUNDARY_MISMATCH"
B06_MEASURE_TOO_NARROW: Final[str] = "B06_MEASURE_TOO_NARROW"

BOUNDARY_REASON_PRIORITY: Final[tuple[str, ...]] = (
    B01_UPSTREAM_GEOMETRY_NOT_ACCEPTED,
    B02_MEASURE_GEOMETRY_ALREADY_PRESENT,
    B03_SYSTEM_STAFF_MEMBERSHIP_INVALID,
    B04_SYSTEM_ORDER_INVALID,
    B05_CROSS_STAFF_BOUNDARY_MISMATCH,
    B06_MEASURE_TOO_NARROW,
)
BOUNDARY_KINDS: Final[tuple[str, ...]] = ("system_edge", "vertical_cluster")


class MeasureSystemBoundariesV2Error(ValueError):
    """Raised for input-integrity violations, not musical ambiguity."""


@dataclass(frozen=True, slots=True)
class StaffBoundaryEvidenceV2:
    staff_id: str
    raw_run_centers: tuple[float, ...]
    clustered_centers: tuple[float, ...]
    boundary_x: tuple[float, ...]
    boundary_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.staff_id:
            raise ValueError("staff_id must be non-empty")
        if len(self.boundary_x) < 2 or len(self.boundary_x) != len(self.boundary_kinds):
            raise ValueError("staff boundary evidence requires aligned start/end sequences")
        if any(kind not in BOUNDARY_KINDS for kind in self.boundary_kinds):
            raise ValueError("unsupported boundary kind")
        if self.boundary_kinds[0] != "system_edge" or self.boundary_kinds[-1] != "system_edge":
            raise ValueError("staff boundary evidence must start/end at system edges")
        if any(next_value <= value for value, next_value in zip(self.boundary_x, self.boundary_x[1:])):
            raise ValueError("staff boundaries must be strictly increasing")


@dataclass(frozen=True, slots=True)
class LogicalMeasureV2:
    logical_measure_id: str
    system_id: str
    measure_index: int
    left_x: float
    right_x: float
    left_kind: str
    right_kind: str
    member_measure_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.logical_measure_id or not self.system_id:
            raise ValueError("logical measure ids must be non-empty")
        if not isinstance(self.measure_index, int) or isinstance(self.measure_index, bool) or self.measure_index <= 0:
            raise ValueError("measure_index must be a positive integer")
        if not math.isfinite(self.left_x) or not math.isfinite(self.right_x) or self.right_x <= self.left_x:
            raise ValueError("logical measure x bounds must be finite and increasing")
        if self.left_kind not in BOUNDARY_KINDS or self.right_kind not in BOUNDARY_KINDS:
            raise ValueError("unsupported logical boundary kind")
        if not self.member_measure_ids or len(set(self.member_measure_ids)) != len(self.member_measure_ids):
            raise ValueError("logical measure must reference unique member measures")


@dataclass(frozen=True, slots=True)
class MeasureSystemBoundaryReportV2:
    status: str
    input_geometry_fingerprint: str
    output_geometry_fingerprint: str | None
    primary_reason: str | None = None
    secondary_reasons: tuple[str, ...] = ()
    staff_evidence: tuple[StaffBoundaryEvidenceV2, ...] = ()
    logical_measures: tuple[LogicalMeasureV2, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("accepted", "ambiguous", "rejected"):
            raise ValueError("unsupported boundary report status")
        _require_sha256("input_geometry_fingerprint", self.input_geometry_fingerprint)
        if self.output_geometry_fingerprint is not None:
            _require_sha256("output_geometry_fingerprint", self.output_geometry_fingerprint)
        reasons = (() if self.primary_reason is None else (self.primary_reason,)) + self.secondary_reasons
        if len(set(reasons)) != len(reasons):
            raise ValueError("boundary reasons must be unique")
        if any(reason not in BOUNDARY_REASON_PRIORITY for reason in reasons):
            raise ValueError("unknown boundary reason")
        if tuple(sorted(reasons, key=BOUNDARY_REASON_PRIORITY.index)) != reasons:
            raise ValueError("boundary reasons must follow canonical priority")
        if self.status == "accepted":
            if reasons or self.output_geometry_fingerprint is None or not self.logical_measures:
                raise ValueError("accepted report requires output and logical measures without reasons")
        else:
            if not reasons or self.output_geometry_fingerprint is not None or self.logical_measures:
                raise ValueError("non-accepted report requires reasons and no output logical measures")

    @property
    def active_reasons(self) -> tuple[str, ...]:
        return (() if self.primary_reason is None else (self.primary_reason,)) + self.secondary_reasons

    def canonical_payload(self) -> dict[str, object]:
        return {
            "version": MEASURE_SYSTEM_BOUNDARIES_V2_VERSION,
            "status": self.status,
            "input_geometry_fingerprint": self.input_geometry_fingerprint,
            "output_geometry_fingerprint": self.output_geometry_fingerprint,
            "primary_reason": self.primary_reason,
            "secondary_reasons": list(self.secondary_reasons),
            "staff_evidence": [
                {
                    "staff_id": item.staff_id,
                    "raw_run_centers": list(item.raw_run_centers),
                    "clustered_centers": list(item.clustered_centers),
                    "boundary_x": list(item.boundary_x),
                    "boundary_kinds": list(item.boundary_kinds),
                }
                for item in self.staff_evidence
            ],
            "logical_measures": [
                {
                    "logical_measure_id": item.logical_measure_id,
                    "system_id": item.system_id,
                    "measure_index": item.measure_index,
                    "left_x": item.left_x,
                    "right_x": item.right_x,
                    "left_kind": item.left_kind,
                    "right_kind": item.right_kind,
                    "member_measure_ids": list(item.member_measure_ids),
                }
                for item in self.logical_measures
            ],
        }

    def fingerprint(self) -> str:
        return _canonical_sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class MeasureSystemBoundariesV2Result:
    page: PageGeometryContract | None
    report: MeasureSystemBoundaryReportV2


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
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


def measure_system_boundaries_v2_config_fingerprint(parent_geometry_fingerprint: str) -> str:
    _require_sha256("parent_geometry_fingerprint", parent_geometry_fingerprint)
    return _canonical_sha256(
        {
            "version": MEASURE_SYSTEM_BOUNDARIES_V2_VERSION,
            "parent_geometry_fingerprint": parent_geometry_fingerprint,
            "vertical_dark_threshold": VERTICAL_DARK_THRESHOLD,
            "min_vertical_coverage_milli": MIN_VERTICAL_COVERAGE_MILLI,
            "vertical_endpoint_anchor_tolerance_px": VERTICAL_ENDPOINT_ANCHOR_TOLERANCE_PX,
            "max_barline_cluster_gap_spacings_milli": MAX_BARLINE_CLUSTER_GAP_SPACINGS_MILLI,
            "edge_snap_spacings_milli": EDGE_SNAP_SPACINGS_MILLI,
            "max_cross_staff_boundary_delta_spacings_milli": MAX_CROSS_STAFF_BOUNDARY_DELTA_SPACINGS_MILLI,
            "min_measure_width_spacings_milli": MIN_MEASURE_WIDTH_SPACINGS_MILLI,
            "implicit_system_edges": True,
            "semantic_barline_recognition": False,
            "meter_access": False,
            "resolver_access": False,
            "model_access": False,
            "checkpoint_access": False,
            "optimizer_access": False,
            "train_validation_test_access": False,
        }
    )


def _decode_page(normalized_png: bytes, geometry: PageGeometryContract) -> Image.Image:
    if not isinstance(normalized_png, bytes) or not normalized_png:
        raise MeasureSystemBoundariesV2Error("normalized_png must be non-empty bytes")
    if sha256(normalized_png).hexdigest() != geometry.normalized_image_sha256:
        raise MeasureSystemBoundariesV2Error("normalized PNG identity does not match grouped geometry")
    try:
        with Image.open(BytesIO(normalized_png)) as opened:
            if opened.format != "PNG" or opened.mode != "L":
                raise MeasureSystemBoundariesV2Error("measure/system boundaries require normalized gray8 PNG")
            opened.load()
            if opened.size != (geometry.page_width, geometry.page_height):
                raise MeasureSystemBoundariesV2Error("normalized PNG dimensions do not match grouped geometry")
            return opened.copy()
    except MeasureSystemBoundariesV2Error:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MeasureSystemBoundariesV2Error("normalized PNG cannot be decoded safely") from exc


def _canonical_reasons(codes: tuple[str, ...]) -> tuple[str, ...]:
    if set(codes) - set(BOUNDARY_REASON_PRIORITY):
        raise ValueError("unknown boundary reason")
    chosen = set(codes)
    return tuple(reason for reason in BOUNDARY_REASON_PRIORITY if reason in chosen)


def _stop(
    geometry: PageGeometryContract,
    *,
    status: str,
    codes: tuple[str, ...],
    evidence: tuple[StaffBoundaryEvidenceV2, ...] = (),
) -> MeasureSystemBoundariesV2Result:
    reasons = _canonical_reasons(codes)
    report = MeasureSystemBoundaryReportV2(
        status=status,
        input_geometry_fingerprint=page_geometry_fingerprint_v1(geometry),
        output_geometry_fingerprint=None,
        primary_reason=reasons[0],
        secondary_reasons=reasons[1:],
        staff_evidence=evidence,
        logical_measures=(),
    )
    return MeasureSystemBoundariesV2Result(page=None, report=report)


def _membership_is_exact(geometry: PageGeometryContract) -> bool:
    staff_ids = tuple(staff.staff_id for staff in geometry.staffs)
    if len(set(staff_ids)) != len(staff_ids):
        return False
    staff_keys = tuple(
        (staff.staff_bbox.y_min, staff.staff_bbox.y_max, staff.staff_id)
        for staff in geometry.staffs
    )
    if staff_keys != tuple(sorted(staff_keys)):
        return False
    if len({system.system_id for system in geometry.systems}) != len(geometry.systems):
        return False

    staff_by_id = {staff.staff_id: staff for staff in geometry.staffs}
    staff_order = {staff.staff_id: index for index, staff in enumerate(geometry.staffs)}
    owner: dict[str, str] = {}
    for system in geometry.systems:
        if not system.staff_ids or len(set(system.staff_ids)) != len(system.staff_ids):
            return False
        if any(staff_id not in staff_by_id for staff_id in system.staff_ids):
            return False
        expected_ids = tuple(sorted(system.staff_ids, key=staff_order.__getitem__))
        if system.staff_ids != expected_ids:
            return False
        members = tuple(staff_by_id[staff_id] for staff_id in system.staff_ids)
        expected_bbox = BoxContract(
            min(staff.staff_bbox.x_min for staff in members),
            min(staff.staff_bbox.y_min for staff in members),
            max(staff.staff_bbox.x_max for staff in members),
            max(staff.staff_bbox.y_max for staff in members),
        )
        if system.system_bbox != expected_bbox:
            return False
        for staff_id in system.staff_ids:
            if staff_id in owner:
                return False
            owner[staff_id] = system.system_id

    if set(owner) != set(staff_ids):
        return False
    return all(owner[staff.staff_id] == staff.system_id for staff in geometry.staffs)


def _system_order_is_canonical(geometry: PageGeometryContract) -> bool:
    if not geometry.systems:
        return False
    keys = tuple(
        (system.system_bbox.y_min, system.system_bbox.y_max, system.system_id)
        for system in geometry.systems
    )
    return keys == tuple(sorted(keys))


def _vertical_runs(image: Image.Image, staff: StaffGeometryContract) -> tuple[tuple[int, int, float], ...]:
    top = int(round(staff.five_staff_lines[0].start.y))
    bottom = int(round(staff.five_staff_lines[-1].start.y))
    top = max(0, min(top, image.height - 1))
    bottom = max(top, min(bottom, image.height - 1))
    span = bottom - top + 1
    minimum_dark = math.ceil(span * MIN_VERTICAL_COVERAGE_MILLI / 1000)
    left = max(0, int(math.floor(staff.staff_bbox.x_min)))
    right = min(image.width - 1, int(math.ceil(staff.staff_bbox.x_max)) - 1)
    pixels = image.load()
    columns: list[int] = []
    top_anchor_end = min(bottom, top + VERTICAL_ENDPOINT_ANCHOR_TOLERANCE_PX)
    bottom_anchor_start = max(top, bottom - VERTICAL_ENDPOINT_ANCHOR_TOLERANCE_PX)
    for x in range(left, right + 1):
        dark_count = sum(1 for y in range(top, bottom + 1) if int(pixels[x, y]) <= VERTICAL_DARK_THRESHOLD)
        touches_top = any(
            int(pixels[x, y]) <= VERTICAL_DARK_THRESHOLD
            for y in range(top, top_anchor_end + 1)
        )
        touches_bottom = any(
            int(pixels[x, y]) <= VERTICAL_DARK_THRESHOLD
            for y in range(bottom_anchor_start, bottom + 1)
        )
        if dark_count >= minimum_dark and touches_top and touches_bottom:
            columns.append(x)
    if not columns:
        return ()
    runs: list[tuple[int, int, float]] = []
    start = previous = columns[0]
    for value in columns[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append((start, previous, (start + previous) / 2.0))
        start = previous = value
    runs.append((start, previous, (start + previous) / 2.0))
    return tuple(runs)


def _cluster_vertical_runs(
    runs: tuple[tuple[int, int, float], ...],
    staff_spacing: float,
) -> tuple[float, ...]:
    if not runs:
        return ()
    maximum_gap = staff_spacing * MAX_BARLINE_CLUSTER_GAP_SPACINGS_MILLI / 1000.0
    clusters: list[tuple[int, int]] = []
    start, end, _ = runs[0]
    for next_start, next_end, _ in runs[1:]:
        ink_gap = next_start - end - 1
        if ink_gap <= maximum_gap:
            end = next_end
            continue
        clusters.append((start, end))
        start, end = next_start, next_end
    clusters.append((start, end))
    return tuple((start + end) / 2.0 for start, end in clusters)


def _staff_evidence(image: Image.Image, staff: StaffGeometryContract) -> StaffBoundaryEvidenceV2:
    runs = _vertical_runs(image, staff)
    raw_centers = tuple(round(center, 9) for _, _, center in runs)
    clusters = _cluster_vertical_runs(runs, staff.staff_spacing)
    left_edge = float(staff.staff_bbox.x_min)
    right_edge = float(staff.staff_bbox.x_max)
    snap = staff.staff_spacing * EDGE_SNAP_SPACINGS_MILLI / 1000.0
    internal = tuple(center for center in clusters if center - left_edge > snap and right_edge - center > snap)
    boundary_x = (left_edge,) + internal + (right_edge,)
    boundary_kinds = ("system_edge",) + ("vertical_cluster",) * len(internal) + ("system_edge",)
    return StaffBoundaryEvidenceV2(
        staff_id=staff.staff_id,
        raw_run_centers=raw_centers,
        clustered_centers=tuple(round(value, 9) for value in clusters),
        boundary_x=tuple(round(value, 9) for value in boundary_x),
        boundary_kinds=boundary_kinds,
    )


def _logical_system_boundaries(
    system_staffs: tuple[StaffGeometryContract, ...],
    evidence_by_staff: dict[str, StaffBoundaryEvidenceV2],
) -> tuple[tuple[float, ...], tuple[str, ...]] | None:
    sequences = tuple(evidence_by_staff[staff.staff_id] for staff in system_staffs)
    lengths = {len(item.boundary_x) for item in sequences}
    if len(lengths) != 1:
        return None
    width = next(iter(lengths))
    logical_x: list[float] = []
    logical_kind: list[str] = []
    for index in range(width):
        values = tuple(item.boundary_x[index] for item in sequences)
        kinds = tuple(item.boundary_kinds[index] for item in sequences)
        tolerance = max(staff.staff_spacing for staff in system_staffs) * (
            MAX_CROSS_STAFF_BOUNDARY_DELTA_SPACINGS_MILLI / 1000.0
        )
        if max(values) - min(values) > tolerance:
            return None
        if len(set(kinds)) != 1:
            return None
        logical_x.append(round(sum(values) / len(values), 9))
        logical_kind.append(kinds[0])
    if any(next_value <= value for value, next_value in zip(logical_x, logical_x[1:])):
        return None
    return tuple(logical_x), tuple(logical_kind)


def propose_measure_system_boundaries_v2(
    normalized_png: bytes,
    geometry: PageGeometryContract,
) -> MeasureSystemBoundariesV2Result:
    """Create deterministic system-local logical and per-staff measure geometry."""
    if not isinstance(geometry, PageGeometryContract):
        raise TypeError("geometry must be PageGeometryContract")
    if geometry.status != "accepted" or not geometry.staffs or not geometry.systems:
        return _stop(geometry, status="rejected", codes=(B01_UPSTREAM_GEOMETRY_NOT_ACCEPTED,))
    if geometry.measure_proposals:
        return _stop(geometry, status="rejected", codes=(B02_MEASURE_GEOMETRY_ALREADY_PRESENT,))
    if not _membership_is_exact(geometry):
        return _stop(geometry, status="rejected", codes=(B03_SYSTEM_STAFF_MEMBERSHIP_INVALID,))
    if not _system_order_is_canonical(geometry):
        return _stop(geometry, status="ambiguous", codes=(B04_SYSTEM_ORDER_INVALID,))

    image = _decode_page(normalized_png, geometry)
    evidence = tuple(_staff_evidence(image, staff) for staff in geometry.staffs)
    evidence_by_staff = {item.staff_id: item for item in evidence}
    staff_by_id = {staff.staff_id: staff for staff in geometry.staffs}

    measures: list[MeasureProposalContract] = []
    logical_measures: list[LogicalMeasureV2] = []
    for system in geometry.systems:
        system_staffs = tuple(staff_by_id[staff_id] for staff_id in system.staff_ids)
        logical = _logical_system_boundaries(system_staffs, evidence_by_staff)
        if logical is None:
            return _stop(
                geometry,
                status="ambiguous",
                codes=(B05_CROSS_STAFF_BOUNDARY_MISMATCH,),
                evidence=evidence,
            )
        logical_x, logical_kinds = logical
        minimum_width = max(staff.staff_spacing for staff in system_staffs) * (
            MIN_MEASURE_WIDTH_SPACINGS_MILLI / 1000.0
        )
        for measure_index, (left_x, right_x) in enumerate(zip(logical_x, logical_x[1:]), start=1):
            if right_x - left_x < minimum_width:
                return _stop(
                    geometry,
                    status="ambiguous",
                    codes=(B06_MEASURE_TOO_NARROW,),
                    evidence=evidence,
                )
            member_ids: list[str] = []
            for staff in system_staffs:
                measure_id = f"{staff.staff_id}-measure-{measure_index}"
                member_ids.append(measure_id)
                measures.append(
                    MeasureProposalContract(
                        measure_id=measure_id,
                        system_id=system.system_id,
                        staff_id=staff.staff_id,
                        bbox=BoxContract(left_x, staff.staff_bbox.y_min, right_x, staff.staff_bbox.y_max),
                        left_boundary=LineSegmentContract(
                            Point2DContract(left_x, staff.staff_bbox.y_min),
                            Point2DContract(left_x, staff.staff_bbox.y_max),
                        ),
                        right_boundary=LineSegmentContract(
                            Point2DContract(right_x, staff.staff_bbox.y_min),
                            Point2DContract(right_x, staff.staff_bbox.y_max),
                        ),
                        status="accepted",
                    )
                )
            logical_measures.append(
                LogicalMeasureV2(
                    logical_measure_id=f"{system.system_id}-measure-{measure_index}",
                    system_id=system.system_id,
                    measure_index=measure_index,
                    left_x=left_x,
                    right_x=right_x,
                    left_kind=logical_kinds[measure_index - 1],
                    right_kind=logical_kinds[measure_index],
                    member_measure_ids=tuple(member_ids),
                )
            )

    output = PageGeometryContract(
        normalized_image_sha256=geometry.normalized_image_sha256,
        geometry_config_fingerprint=measure_system_boundaries_v2_config_fingerprint(
            geometry.geometry_config_fingerprint
        ),
        page_width=geometry.page_width,
        page_height=geometry.page_height,
        transform=geometry.transform,
        systems=geometry.systems,
        staffs=geometry.staffs,
        measure_proposals=tuple(measures),
        status="accepted",
    )
    report = MeasureSystemBoundaryReportV2(
        status="accepted",
        input_geometry_fingerprint=page_geometry_fingerprint_v1(geometry),
        output_geometry_fingerprint=page_geometry_fingerprint_v1(output),
        staff_evidence=evidence,
        logical_measures=tuple(logical_measures),
    )
    return MeasureSystemBoundariesV2Result(page=output, report=report)
