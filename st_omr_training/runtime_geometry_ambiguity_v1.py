"""Deterministic AMBIGUOUS reason resolver for runtime geometry.

This module contains no image detector and no learned model.  It only turns a
set of already-observed geometry ambiguity codes into one canonical report.
The fixed priority order is intentionally frozen so repeated runs cannot choose
different primary/secondary reasons for the same active code set.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Final, Iterable


GEOMETRY_AMBIGUITY_SCHEMA: Final[str] = "runtime-geometry-ambiguity-report-v1"

A04_PAGE_CROPPED: Final[str] = "A04_PAGE_CROPPED"
A03_LOW_VISIBILITY: Final[str] = "A03_LOW_VISIBILITY"
A01_INCOMPLETE_STAFF: Final[str] = "A01_INCOMPLETE_STAFF"
A05_OVERLAPPING_CANDIDATES: Final[str] = "A05_OVERLAPPING_CANDIDATES"
A02_STAFFS_TOO_CLOSE: Final[str] = "A02_STAFFS_TOO_CLOSE"
A07_EXTRA_LINE_CANDIDATES: Final[str] = "A07_EXTRA_LINE_CANDIDATES"
A06_IRREGULAR_SPACING: Final[str] = "A06_IRREGULAR_SPACING"

AMBIGUITY_PRIORITY: Final[tuple[str, ...]] = (
    A04_PAGE_CROPPED,
    A03_LOW_VISIBILITY,
    A01_INCOMPLETE_STAFF,
    A05_OVERLAPPING_CANDIDATES,
    A02_STAFFS_TOO_CLOSE,
    A07_EXTRA_LINE_CANDIDATES,
    A06_IRREGULAR_SPACING,
)
AMBIGUITY_CODE_SET: Final[frozenset[str]] = frozenset(AMBIGUITY_PRIORITY)


@dataclass(frozen=True, slots=True)
class GeometryAmbiguityReport:
    schema_version: str
    status: str
    primary_reason: str
    secondary_reasons: tuple[str, ...]
    active_reasons: tuple[str, ...]
    priority_order: tuple[str, ...] = AMBIGUITY_PRIORITY

    def __post_init__(self) -> None:
        if self.schema_version != GEOMETRY_AMBIGUITY_SCHEMA:
            raise ValueError("unsupported geometry ambiguity schema")
        if self.status != "AMBIGUOUS":
            raise ValueError("ambiguity report status must be AMBIGUOUS")
        if self.priority_order != AMBIGUITY_PRIORITY:
            raise ValueError("ambiguity priority order is frozen")
        if self.primary_reason not in AMBIGUITY_CODE_SET:
            raise ValueError("primary ambiguity reason is unknown")
        if any(code not in AMBIGUITY_CODE_SET for code in self.secondary_reasons):
            raise ValueError("secondary ambiguity reason is unknown")
        if len(set(self.secondary_reasons)) != len(self.secondary_reasons):
            raise ValueError("secondary ambiguity reasons must be unique")
        if self.primary_reason in self.secondary_reasons:
            raise ValueError("primary reason cannot be repeated as secondary")
        expected = (self.primary_reason, *self.secondary_reasons)
        if self.active_reasons != expected:
            raise ValueError("active_reasons must equal primary plus secondaries")
        if tuple(sorted(self.active_reasons, key=AMBIGUITY_PRIORITY.index)) != self.active_reasons:
            raise ValueError("ambiguity reasons must follow frozen priority order")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "primary_reason": self.primary_reason,
            "secondary_reasons": self.secondary_reasons,
            "active_reasons": self.active_reasons,
            "priority_order": self.priority_order,
            "deterministic_checks": {
                "codes_unique": len(set(self.active_reasons)) == len(self.active_reasons),
                "primary_is_highest_priority": self.primary_reason == self.active_reasons[0],
                "secondary_order_valid": tuple(
                    sorted(self.secondary_reasons, key=AMBIGUITY_PRIORITY.index)
                ) == self.secondary_reasons,
                "unknown_codes_present": False,
            },
        }

    def fingerprint(self) -> str:
        raw = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return sha256(raw).hexdigest()


def build_geometry_ambiguity_report(active_codes: Iterable[str]) -> GeometryAmbiguityReport:
    """Return the one canonical report for a non-empty set of active A01-A07 codes."""
    if isinstance(active_codes, (str, bytes)):
        raise TypeError("active_codes must be an iterable of codes, not a string")
    unique = set(active_codes)
    if not unique:
        raise ValueError("at least one ambiguity code is required")
    unknown = unique - AMBIGUITY_CODE_SET
    if unknown:
        raise ValueError(f"unknown ambiguity codes: {sorted(unknown)}")
    ordered = tuple(code for code in AMBIGUITY_PRIORITY if code in unique)
    return GeometryAmbiguityReport(
        schema_version=GEOMETRY_AMBIGUITY_SCHEMA,
        status="AMBIGUOUS",
        primary_reason=ordered[0],
        secondary_reasons=ordered[1:],
        active_reasons=ordered,
    )
