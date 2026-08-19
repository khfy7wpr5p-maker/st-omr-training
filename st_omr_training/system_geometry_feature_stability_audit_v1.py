"""Fixture/synthetic-only feature stability audit for System Geometry evidence.

This module audits candidate raw evidence signatures on a frozen fixture dataset.
It deliberately does not define or authorize a runtime system-grouping rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Final

from .system_geometry_evidence_dataset_v1 import (
    SystemGeometryEvidenceDatasetV1,
    SystemGeometryPairDatasetRecordV1,
)
from .system_geometry_evidence_v1 import StaffSystemRelation


SYSTEM_GEOMETRY_FEATURE_STABILITY_AUDIT_VERSION: Final[str] = (
    "system-geometry-feature-stability-audit-v1"
)
SYSTEM_GEOMETRY_FEATURE_STABILITY_AUDIT_CLAIM_BOUNDARY: Final[str] = (
    "FIXTURE_ONLY_NO_RUNTIME_GROUPING_RULE"
)


class SystemGeometryFeatureStabilityAuditError(ValueError):
    """Raised when the fixture surface is too weak for a stability audit."""


def _canonical_pair(left: object, right: object) -> tuple[object, object]:
    values = [left, right]
    values.sort(
        key=lambda value: json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return values[0], values[1]


def _candidate_features(
    record: SystemGeometryPairDatasetRecordV1,
) -> dict[str, object]:
    return {
        "staff_count_pair": _canonical_pair(
            record.system_a_staff_count,
            record.system_b_staff_count,
        ),
        "measure_count_pair": _canonical_pair(
            record.system_a_measure_count,
            record.system_b_measure_count,
        ),
        "grouping_token_presence_pair": _canonical_pair(
            bool(record.system_a_grouping_tokens),
            bool(record.system_b_grouping_tokens),
        ),
        "grouping_token_signature_pair": _canonical_pair(
            tuple(sorted(record.system_a_grouping_tokens)),
            tuple(sorted(record.system_b_grouping_tokens)),
        ),
        "barline_group_count_pair": _canonical_pair(
            record.system_a_barline_group_count,
            record.system_b_barline_group_count,
        ),
    }


def _encoded_feature_value(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


@dataclass(frozen=True, slots=True)
class SystemGeometryFeatureStabilityAuditV1:
    dataset_fingerprint: str
    relation_counts: dict[str, int]
    hard_positive_count: int
    hard_adjacent_negative_count: int
    feature_values_by_relation: dict[str, dict[str, tuple[str, ...]]]
    overlapping_features: tuple[str, ...]
    disjoint_on_fixture_surface: tuple[str, ...]
    version: str = SYSTEM_GEOMETRY_FEATURE_STABILITY_AUDIT_VERSION
    claim_boundary: str = SYSTEM_GEOMETRY_FEATURE_STABILITY_AUDIT_CLAIM_BOUNDARY

    def canonical_payload(self) -> dict[str, object]:
        return {
            "claim_boundary": self.claim_boundary,
            "dataset_fingerprint": self.dataset_fingerprint,
            "disjoint_on_fixture_surface": list(self.disjoint_on_fixture_surface),
            "feature_values_by_relation": self.feature_values_by_relation,
            "hard_adjacent_negative_count": self.hard_adjacent_negative_count,
            "hard_positive_count": self.hard_positive_count,
            "overlapping_features": list(self.overlapping_features),
            "relation_counts": self.relation_counts,
            "version": self.version,
        }

    def fingerprint(self) -> str:
        raw = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return sha256(raw).hexdigest()


def audit_system_geometry_feature_stability_v1(
    dataset: SystemGeometryEvidenceDatasetV1,
) -> SystemGeometryFeatureStabilityAuditV1:
    """Audit raw fixture features without producing a grouping rule."""

    if not isinstance(dataset, SystemGeometryEvidenceDatasetV1):
        raise SystemGeometryFeatureStabilityAuditError(
            "dataset must be SystemGeometryEvidenceDatasetV1"
        )

    hard_positive_count = sum(
        1
        for record in dataset.records
        if record.relation is StaffSystemRelation.SAME_SYSTEM
        and record.system_a_staff_count >= 2
        and record.system_a_measure_count >= 2
    )
    hard_adjacent_negative_count = sum(
        1
        for record in dataset.records
        if record.relation is StaffSystemRelation.DIFFERENT_SYSTEM
        and record.adjacent_different_systems
        and record.system_a_staff_count >= 2
        and record.system_b_staff_count >= 2
        and record.system_a_measure_count >= 2
        and record.system_b_measure_count >= 2
    )
    if hard_positive_count < 1:
        raise SystemGeometryFeatureStabilityAuditError(
            "expanded surface requires multi-measure multi-staff SAME_SYSTEM evidence"
        )
    if hard_adjacent_negative_count < 1:
        raise SystemGeometryFeatureStabilityAuditError(
            "expanded surface requires adjacent multi-staff DIFFERENT_SYSTEM hard negatives"
        )

    relation_names = tuple(relation.value for relation in StaffSystemRelation)
    observed: dict[str, dict[str, set[str]]] = {}
    for record in dataset.records:
        relation_name = record.relation.value
        for feature_name, feature_value in _candidate_features(record).items():
            by_relation = observed.setdefault(
                feature_name,
                {name: set() for name in relation_names},
            )
            by_relation[relation_name].add(_encoded_feature_value(feature_value))

    stable_payload: dict[str, dict[str, tuple[str, ...]]] = {}
    overlapping: list[str] = []
    disjoint: list[str] = []
    for feature_name in sorted(observed):
        by_relation = observed[feature_name]
        stable_payload[feature_name] = {
            name: tuple(sorted(by_relation[name])) for name in relation_names
        }
        same_values = by_relation[StaffSystemRelation.SAME_SYSTEM.value]
        different_values = by_relation[StaffSystemRelation.DIFFERENT_SYSTEM.value]
        if same_values.intersection(different_values):
            overlapping.append(feature_name)
        else:
            disjoint.append(feature_name)

    return SystemGeometryFeatureStabilityAuditV1(
        dataset_fingerprint=dataset.fingerprint(),
        relation_counts=dataset.relation_counts(),
        hard_positive_count=hard_positive_count,
        hard_adjacent_negative_count=hard_adjacent_negative_count,
        feature_values_by_relation=stable_payload,
        overlapping_features=tuple(overlapping),
        disjoint_on_fixture_surface=tuple(disjoint),
    )


def system_geometry_feature_stability_rule_design_allowed() -> bool:
    """Fixture stability evidence never authorizes a grouping rule by itself."""

    return False


def system_geometry_feature_stability_runtime_connection_allowed() -> bool:
    """Fixture stability audit never authorizes runtime/production connection."""

    return False
