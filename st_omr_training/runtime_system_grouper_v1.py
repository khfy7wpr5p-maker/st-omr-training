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
import json
from typing import Final

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

G01_UPSTREAM_GEOMETRY_NOT_ACCEPTED: Final[str] = "G01_UPSTREAM_GEOMETRY_NOT_ACCEPTED"
G02_MEASURE_GEOMETRY_ALREADY_PRESENT: Final[str] = "G02_MEASURE_GEOMETRY_ALREADY_PRESENT"
G03_DECLARED_POLICY_STAFF_COUNT_MISMATCH: Final[str] = "G03_DECLARED_POLICY_STAFF_COUNT_MISMATCH"
G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP: Final[str] = "G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP"
G05_INVALID_STAFF_ORDER: Final[str] = "G05_INVALID_STAFF_ORDER"

SYSTEM_GROUPER_REASON_PRIORITY: Final[tuple[str, ...]] = (
    G01_UPSTREAM_GEOMETRY_NOT_ACCEPTED,
    G02_MEASURE_GEOMETRY_ALREADY_PRESENT,
    G05_INVALID_STAFF_ORDER,
    G03_DECLARED_POLICY_STAFF_COUNT_MISMATCH,
    G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP,
)


@dataclass(frozen=True, slots=True)
class SystemGroupingReportV1:
    status: str
    policy: str
    input_geometry_fingerprint: str
    output_geometry_fingerprint: str | None
    primary_reason: str | None = None
    secondary_reasons: tuple[str, ...] = ()

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
) -> SystemGroupingResultV1:
    reasons = _canonical_reasons(codes)
    report = SystemGroupingReportV1(
        status=status,
        policy=policy,
        input_geometry_fingerprint=page_geometry_fingerprint_v1(page),
        output_geometry_fingerprint=None,
        primary_reason=reasons[0],
        secondary_reasons=reasons[1:],
    )
    return SystemGroupingResultV1(page=None, report=report)


def _staff_order_is_canonical(staffs: tuple[StaffGeometryContract, ...]) -> bool:
    keys = tuple((staff.staff_bbox.y_min, staff.staff_bbox.y_max, staff.staff_id) for staff in staffs)
    return keys == tuple(sorted(keys)) and len({staff.staff_id for staff in staffs}) == len(staffs)


def _membership_for_policy(
    staff_count: int,
    policy: str,
) -> tuple[tuple[int, ...], ...] | None:
    if policy == "auto-v1":
        if staff_count == 1:
            return ((0,),)
        return None
    if policy == "monostaff-v1":
        return tuple((index,) for index in range(staff_count))
    if policy == "fixed-two-staff-v1":
        if staff_count == 0 or staff_count % 2:
            return None
        return tuple(tuple(range(index, index + 2)) for index in range(0, staff_count, 2))
    raise ValueError("unsupported system grouping policy")


def group_staffs_into_systems_v1(
    page: PageGeometryContract,
    *,
    policy: str = "auto-v1",
) -> SystemGroupingResultV1:
    """Rebind accepted staff geometry to deterministic system membership.

    ``auto-v1`` deliberately accepts only the unambiguous one-staff case.
    Multi-staff image-only membership remains underdetermined by the currently
    merged evidence and therefore returns AMBIGUOUS rather than guessing.

    ``monostaff-v1`` is the explicit current ScoreMosaic V1 policy: every staff
    is a distinct system. ``fixed-two-staff-v1`` is an explicit caller-declared
    grand-staff policy: adjacent ordered staffs are paired, and odd counts fail
    closed.  Policy is an input contract, not inferred from hidden metadata.
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

    membership = _membership_for_policy(len(page.staffs), policy)
    if membership is None:
        reason = (
            G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP
            if policy == "auto-v1"
            else G03_DECLARED_POLICY_STAFF_COUNT_MISMATCH
        )
        return _stop(page, policy=policy, status="ambiguous", codes=(reason,))

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
    )
    return SystemGroupingResultV1(page=output, report=report)
