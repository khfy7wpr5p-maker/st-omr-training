"""Isolated declarative contract for the future ST Geometry Engine runtime lane.

The Geometry Engine is limited to page/system/staff/measure geometry proposals.
It does not infer musical semantics, load specialist models, train, access TEST,
or integrate with Stage 7-D10 / Stage 7-D13.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
from typing import Final

from .runtime_page_normalizer_contract import HomographyContract


GEOMETRY_ENGINE_CONTRACT_VERSION: Final[str] = "runtime-geometry-engine-contract-v1"
GEOMETRY_ENGINE_SCHEMA: Final[str] = "runtime-geometry-engine-contract-v1"
GEOMETRY_STATUSES: Final[tuple[str, ...]] = ("accepted", "ambiguous", "rejected")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
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


@dataclass(frozen=True, slots=True)
class Point2DContract:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in (self.x, self.y)
        ):
            raise ValueError("point coordinates must be finite numbers")


@dataclass(frozen=True, slots=True)
class LineSegmentContract:
    start: Point2DContract
    end: Point2DContract

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError("line segment must have nonzero length")


@dataclass(frozen=True, slots=True)
class BoxContract:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        ):
            raise ValueError("box coordinates must be finite numbers")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("box must have positive width and height")


@dataclass(frozen=True, slots=True)
class GeometryInputContract:
    """Accepted NormalizedPage metadata required by the Geometry Engine."""

    normalized_image_sha256: str
    normalizer_config_fingerprint: str
    normalized_width: int
    normalized_height: int
    transform: HomographyContract

    def __post_init__(self) -> None:
        _require_sha256("normalized_image_sha256", self.normalized_image_sha256)
        _require_sha256("normalizer_config_fingerprint", self.normalizer_config_fingerprint)
        for name, value in (("normalized_width", self.normalized_width), ("normalized_height", self.normalized_height)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class StaffGeometryContract:
    staff_id: str
    system_id: str
    five_staff_lines: tuple[LineSegmentContract, ...]
    staff_bbox: BoxContract
    staff_spacing: float

    def __post_init__(self) -> None:
        if not self.staff_id or not self.system_id:
            raise ValueError("staff/system ids must be non-empty")
        if len(self.five_staff_lines) != 5:
            raise ValueError("accepted staff geometry requires exactly five staff lines")
        if (
            not isinstance(self.staff_spacing, (int, float))
            or isinstance(self.staff_spacing, bool)
            or not math.isfinite(self.staff_spacing)
            or self.staff_spacing <= 0
        ):
            raise ValueError("staff spacing must be positive and finite")

        centers = tuple((line.start.y + line.end.y) / 2.0 for line in self.five_staff_lines)
        if any(next_value <= value for value, next_value in zip(centers, centers[1:])):
            raise ValueError("staff lines must be ordered from top to bottom")


@dataclass(frozen=True, slots=True)
class SystemGeometryContract:
    system_id: str
    system_bbox: BoxContract
    staff_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.system_id:
            raise ValueError("system_id must be non-empty")
        if not self.staff_ids or len(set(self.staff_ids)) != len(self.staff_ids):
            raise ValueError("system must reference one or more unique staffs")


@dataclass(frozen=True, slots=True)
class MeasureProposalContract:
    measure_id: str
    system_id: str
    staff_id: str
    bbox: BoxContract
    left_boundary: LineSegmentContract
    right_boundary: LineSegmentContract
    status: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.measure_id or not self.system_id or not self.staff_id:
            raise ValueError("measure/system/staff ids must be non-empty")
        if self.status not in GEOMETRY_STATUSES:
            raise ValueError("unsupported measure proposal status")
        if self.status == "accepted" and self.reasons:
            raise ValueError("accepted measure proposal cannot carry ambiguity reasons")
        if self.status != "accepted" and not self.reasons:
            raise ValueError("ambiguous/rejected proposal must explain why")


@dataclass(frozen=True, slots=True)
class PageGeometryContract:
    normalized_image_sha256: str
    geometry_config_fingerprint: str
    page_width: int
    page_height: int
    transform: HomographyContract
    systems: tuple[SystemGeometryContract, ...]
    staffs: tuple[StaffGeometryContract, ...]
    measure_proposals: tuple[MeasureProposalContract, ...]
    status: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_sha256("normalized_image_sha256", self.normalized_image_sha256)
        _require_sha256("geometry_config_fingerprint", self.geometry_config_fingerprint)
        if self.status not in GEOMETRY_STATUSES:
            raise ValueError("unsupported page geometry status")
        for name, value in (("page_width", self.page_width), ("page_height", self.page_height)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.status == "accepted" and self.reasons:
            raise ValueError("accepted page geometry cannot carry rejection reasons")
        if self.status != "accepted" and not self.reasons:
            raise ValueError("ambiguous/rejected page geometry must explain why")

        system_ids = [system.system_id for system in self.systems]
        staff_ids = [staff.staff_id for staff in self.staffs]
        measure_ids = [measure.measure_id for measure in self.measure_proposals]
        if len(set(system_ids)) != len(system_ids):
            raise ValueError("system ids must be unique")
        if len(set(staff_ids)) != len(staff_ids):
            raise ValueError("staff ids must be unique")
        if len(set(measure_ids)) != len(measure_ids):
            raise ValueError("measure ids must be unique")

        system_id_set = set(system_ids)
        staff_id_set = set(staff_ids)
        for staff in self.staffs:
            if staff.system_id not in system_id_set:
                raise ValueError("staff references an unknown system")
        for system in self.systems:
            if any(staff_id not in staff_id_set for staff_id in system.staff_ids):
                raise ValueError("system references an unknown staff")
        for measure in self.measure_proposals:
            if measure.system_id not in system_id_set or measure.staff_id not in staff_id_set:
                raise ValueError("measure proposal references unknown geometry")

        for box in [
            *(system.system_bbox for system in self.systems),
            *(staff.staff_bbox for staff in self.staffs),
            *(measure.bbox for measure in self.measure_proposals),
        ]:
            if box.x_min < 0 or box.y_min < 0 or box.x_max > self.page_width or box.y_max > self.page_height:
                raise ValueError("accepted geometry must stay inside normalized page bounds")


GEOMETRY_ALLOWED_OBSERVATIONS: Final[tuple[str, ...]] = (
    "system_bbox",
    "five_staff_lines",
    "staff_bbox",
    "staff_spacing",
    "measure_bbox_proposal",
    "measure_left_boundary_proposal",
    "measure_right_boundary_proposal",
)

GEOMETRY_FORBIDDEN_SEMANTICS: Final[tuple[str, ...]] = (
    "meter_class",
    "notehead_class",
    "rest_class",
    "accidental_class",
    "pitch",
    "duration",
    "chord",
    "voice",
    "musicxml_generation",
)


def runtime_geometry_engine_contract_payload() -> dict[str, object]:
    return {
        "schema_version": GEOMETRY_ENGINE_SCHEMA,
        "contract_version": GEOMETRY_ENGINE_CONTRACT_VERSION,
        "input": {
            "kind": "accepted-normalized-page",
            "requires_normalized_image_sha256": True,
            "requires_normalizer_fingerprint": True,
            "requires_forward_inverse_transform": True,
        },
        "output": {
            "statuses": GEOMETRY_STATUSES,
            "measure_outputs_are_proposals": True,
            "fail_closed_on_ambiguity": True,
            "coordinates_must_be_replayable_to_original_page": True,
        },
        "allowed_observations": GEOMETRY_ALLOWED_OBSERVATIONS,
        "forbidden_semantics": GEOMETRY_FORBIDDEN_SEMANTICS,
        "isolation": {
            "stage7d10_read": False,
            "stage7d10_write": False,
            "stage7d13_read": False,
            "stage7d13_write": False,
            "checkpoint_access": False,
            "optimizer_access": False,
            "test_split_access": False,
        },
    }


def runtime_geometry_engine_contract_fingerprint() -> str:
    return _canonical_sha256(runtime_geometry_engine_contract_payload())
