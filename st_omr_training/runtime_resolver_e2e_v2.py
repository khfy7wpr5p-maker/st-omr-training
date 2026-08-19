"""Accepted deterministic runtime lane from raster bytes to Resolver v1.

This module intentionally replaces the legacy shadow preparation dependency on
runtime_measure_geometry_v1. Learned Meter execution remains an external
producer boundary; its supplied score records must be bound to the exact
Runtime Local ROI bytes generated in the same invocation before they can enter
Meter Integration v3.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Final

from .runtime_deterministic_resolver_v1 import (
    DeterministicResolverResult,
    resolve_specialist_evidence_v1,
)
from .runtime_geometry_engine_contract import GeometryInputContract, PageGeometryContract
from .runtime_geometry_engine_v2 import detect_multistaff_geometry_v2
from .runtime_local_roi_v1 import RuntimeRoiBatch, extract_runtime_rois_v1
from .runtime_measure_system_boundaries_v2 import (
    MeasureSystemBoundaryReportV2,
    propose_measure_system_boundaries_v2,
)
from .runtime_meter_integration_v3 import (
    MeterModelEvidenceV3,
    MeterRuntimeIntegrationV3Result,
    integrate_meter_evidence_v3,
)
from .runtime_page_normalizer_contract import NormalizedPageContract, RasterPageInputContract
from .runtime_page_normalizer_v1 import normalize_raster_page_v1
from .runtime_specialist_evidence_v1 import SpecialistEvidenceBatch
from .runtime_system_grouper_v1 import (
    SYSTEM_GROUPER_POLICIES,
    SystemGroupingReportV1,
    group_staffs_into_systems_v1,
    page_geometry_fingerprint_v1,
)


RUNTIME_RESOLVER_E2E_V2_VERSION: Final[str] = "runtime-resolver-e2e-v2"
EVIDENCE_ORIGINS: Final[tuple[str, ...]] = ("test-fixture", "external-model")


class RuntimeResolverE2EV2Error(RuntimeError):
    """Fail-closed runtime stop with an observable stage and reasons."""

    def __init__(self, stage: str, reasons: tuple[str, ...]) -> None:
        if not stage or not reasons:
            raise ValueError("E2E stop needs a stage and at least one reason")
        self.stage = stage
        self.reasons = reasons
        super().__init__(f"{stage}: {';'.join(reasons)}")


@dataclass(frozen=True, slots=True)
class RuntimeResolverPreparedV2:
    normalized_png: bytes
    normalized_page: NormalizedPageContract
    detected_geometry: PageGeometryContract
    grouped_geometry: PageGeometryContract
    grouping_report: SystemGroupingReportV1
    measure_geometry: PageGeometryContract
    boundary_report: MeasureSystemBoundaryReportV2
    roi_batch: RuntimeRoiBatch
    system_policy: str

    def __post_init__(self) -> None:
        if self.system_policy not in SYSTEM_GROUPER_POLICIES:
            raise ValueError("prepared E2E state has unsupported system policy")
        if not self.normalized_png:
            raise ValueError("prepared E2E state requires normalized PNG bytes")
        for page in (self.detected_geometry, self.grouped_geometry, self.measure_geometry):
            if page.status != "accepted":
                raise ValueError("prepared E2E state can contain only accepted geometry")
        if self.grouping_report.status != "accepted" or self.boundary_report.status != "accepted":
            raise ValueError("prepared E2E reports must be accepted")
        if self.roi_batch.source_image_sha256 != self.measure_geometry.normalized_image_sha256:
            raise ValueError("prepared ROI source identity disagrees with measure geometry")


@dataclass(frozen=True, slots=True)
class BoundMeterEvidenceRecordV2:
    source_image_sha256: str
    roi_id: str
    roi_image_sha256: str
    evidence: MeterModelEvidenceV3

    def __post_init__(self) -> None:
        _require_sha(self.source_image_sha256, "source_image_sha256")
        _require_sha(self.roi_image_sha256, "roi_image_sha256")
        if not self.roi_id:
            raise ValueError("bound Meter record roi_id must be non-empty")
        if not isinstance(self.evidence, MeterModelEvidenceV3):
            raise TypeError("bound Meter record evidence must be MeterModelEvidenceV3")
        if self.evidence.roi_id != self.roi_id:
            raise ValueError("bound Meter record ROI identity disagrees with evidence")


@dataclass(frozen=True, slots=True)
class BoundMeterEvidenceBatchV2:
    provider_fingerprint: str
    evidence_origin: str
    source_image_sha256: str
    records: tuple[BoundMeterEvidenceRecordV2, ...]

    def __post_init__(self) -> None:
        _require_sha(self.provider_fingerprint, "provider_fingerprint")
        _require_sha(self.source_image_sha256, "source_image_sha256")
        if self.evidence_origin not in EVIDENCE_ORIGINS:
            raise ValueError("unsupported Meter evidence origin")
        if not self.records:
            raise ValueError("bound Meter batch cannot be empty")
        if any(not isinstance(item, BoundMeterEvidenceRecordV2) for item in self.records):
            raise TypeError("bound Meter batch contains a non-record")
        if any(item.source_image_sha256 != self.source_image_sha256 for item in self.records):
            raise ValueError("bound Meter batch mixes source-image identities")
        roi_ids = tuple(item.roi_id for item in self.records)
        evidence_ids = tuple(item.evidence.evidence_id for item in self.records)
        measure_ids = tuple(item.evidence.measure_id for item in self.records)
        if len(set(roi_ids)) != len(roi_ids):
            raise ValueError("bound Meter batch has duplicate ROI ownership")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("bound Meter batch has duplicate evidence ids")
        if len(set(measure_ids)) != len(measure_ids):
            raise ValueError("bound Meter batch has duplicate measure ownership")


@dataclass(frozen=True, slots=True)
class RuntimeResolverE2EV2Result:
    prepared: RuntimeResolverPreparedV2
    meter_result: MeterRuntimeIntegrationV3Result
    resolver_result: DeterministicResolverResult
    evidence_origin: str
    provider_fingerprint: str
    fingerprint: str

    def __post_init__(self) -> None:
        if self.evidence_origin not in EVIDENCE_ORIGINS:
            raise ValueError("unsupported E2E evidence origin")
        _require_sha(self.provider_fingerprint, "provider_fingerprint")
        _require_sha(self.fingerprint, "fingerprint")

    @property
    def is_real_model_proof(self) -> bool:
        # Byte binding alone cannot prove the external provider's checkpoint or
        # actual inference execution. Real-model proof stays closed in V2.
        return False


def _require_sha(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be lowercase SHA-256")


def _canonical_sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    return sha256(raw).hexdigest()


def _stop(stage: str, reasons: tuple[str, ...] | list[str] | None, fallback: str) -> None:
    values = tuple(str(item) for item in (reasons or ()) if str(item))
    raise RuntimeResolverE2EV2Error(stage, values or (fallback,))


def prepare_runtime_resolver_e2e_v2(
    raster_bytes: bytes,
    raster_contract: RasterPageInputContract,
    *,
    system_policy: str,
) -> RuntimeResolverPreparedV2:
    """Run the accepted deterministic raster->ROI lane and stop before Meter evidence."""
    if system_policy not in SYSTEM_GROUPER_POLICIES:
        raise ValueError("system_policy must be an explicit System Grouper v1 policy")

    normalized = normalize_raster_page_v1(raster_bytes, raster_contract)
    if normalized.page.status != "accepted" or normalized.normalized_png is None:
        _stop("page-normalizer", normalized.page.rejection_reasons, "normalization-not-accepted")
    page = normalized.page
    if (
        page.normalized_image_sha256 is None
        or page.normalized_width is None
        or page.normalized_height is None
        or page.transform is None
    ):
        _stop("page-normalizer", (), "accepted-normalizer-missing-runtime-identity")

    geometry_input = GeometryInputContract(
        normalized_image_sha256=page.normalized_image_sha256,
        normalizer_config_fingerprint=page.normalizer_config_fingerprint,
        normalized_width=page.normalized_width,
        normalized_height=page.normalized_height,
        transform=page.transform,
    )
    detected = detect_multistaff_geometry_v2(normalized.normalized_png, geometry_input)
    if detected.page.status != "accepted":
        _stop("geometry-engine-v2", detected.page.reasons, "geometry-not-accepted")

    grouped = group_staffs_into_systems_v1(
        detected.page,
        normalized_png=normalized.normalized_png,
        policy=system_policy,
    )
    if grouped.page.status != "accepted" or grouped.report.status != "accepted":
        _stop("system-grouper-v1", grouped.page.reasons or grouped.report.active_reasons, "system-grouping-not-accepted")

    measured = propose_measure_system_boundaries_v2(normalized.normalized_png, grouped.page)
    if measured.page.status != "accepted" or measured.report.status != "accepted":
        _stop("measure-system-boundaries-v2", measured.page.reasons or measured.report.active_reasons, "measure-boundaries-not-accepted")

    roi_batch = extract_runtime_rois_v1(normalized.normalized_png, measured.page)
    return RuntimeResolverPreparedV2(
        normalized_png=normalized.normalized_png,
        normalized_page=page,
        detected_geometry=detected.page,
        grouped_geometry=grouped.page,
        grouping_report=grouped.report,
        measure_geometry=measured.page,
        boundary_report=measured.report,
        roi_batch=roi_batch,
        system_policy=system_policy,
    )


def _validate_bound_meter_batch(
    prepared: RuntimeResolverPreparedV2,
    batch: BoundMeterEvidenceBatchV2,
) -> tuple[MeterModelEvidenceV3, ...]:
    if batch.source_image_sha256 != prepared.measure_geometry.normalized_image_sha256:
        _stop("meter-evidence-binding", (), "source-image-sha-mismatch")

    start_rois = {
        item.roi_id: item
        for item in prepared.roi_batch.artifacts
        if item.kind == "measure-start"
    }
    expected_measure_ids = {item.measure_id for item in prepared.measure_geometry.measure_proposals}
    if len(start_rois) != len(expected_measure_ids):
        _stop("meter-evidence-binding", (), "measure-start-roi-cardinality-mismatch")

    records_by_measure = {item.evidence.measure_id: item for item in batch.records}
    if set(records_by_measure) != expected_measure_ids:
        _stop("meter-evidence-binding", (), "meter-record-measure-coverage-mismatch")

    ordered: list[MeterModelEvidenceV3] = []
    for measure in prepared.measure_geometry.measure_proposals:
        record = records_by_measure[measure.measure_id]
        roi = start_rois.get(record.roi_id)
        if roi is None:
            _stop("meter-evidence-binding", (), "meter-record-roi-not-found")
        if (
            roi.measure_id != measure.measure_id
            or roi.staff_id != measure.staff_id
            or roi.source_image_sha256 != record.source_image_sha256
            or roi.roi_image_sha256 != record.roi_image_sha256
            or record.evidence.roi_id != roi.roi_id
            or record.evidence.measure_id != measure.measure_id
            or record.evidence.staff_id != measure.staff_id
        ):
            _stop("meter-evidence-binding", (), "meter-record-roi-byte-or-owner-mismatch")
        ordered.append(record.evidence)
    return tuple(ordered)


def _resolver_payload(result: DeterministicResolverResult) -> dict[str, object]:
    return {
        "status": result.status,
        "config_fingerprint": result.config_fingerprint,
        "measures": [
            {
                "measure_id": item.measure_id,
                "staff_id": item.staff_id,
                "status": item.status,
                "meter_class": item.meter_class,
                "reasons": list(item.reasons),
                "notes": [
                    {
                        "id": note.notehead_observation_id,
                        "class": note.notehead_class,
                        "accidental": note.accidental_class,
                        "accidental_id": note.accidental_observation_id,
                    }
                    for note in item.notes
                ],
                "rests": [
                    {"id": rest.rest_observation_id, "class": rest.rest_class}
                    for rest in item.rests
                ],
            }
            for item in result.measures
        ],
    }


def resolve_runtime_resolver_e2e_v2(
    prepared: RuntimeResolverPreparedV2,
    bound_meter_evidence: BoundMeterEvidenceBatchV2,
    *,
    other_specialist_evidence: SpecialistEvidenceBatch | None = None,
) -> RuntimeResolverE2EV2Result:
    """Bind exact Meter ROI bytes, compose Meter and invoke the real Resolver path."""
    if not isinstance(prepared, RuntimeResolverPreparedV2):
        raise TypeError("prepared must be RuntimeResolverPreparedV2")
    if not isinstance(bound_meter_evidence, BoundMeterEvidenceBatchV2):
        raise TypeError("bound_meter_evidence must be BoundMeterEvidenceBatchV2")
    if other_specialist_evidence is not None and not isinstance(other_specialist_evidence, SpecialistEvidenceBatch):
        raise TypeError("other_specialist_evidence must be SpecialistEvidenceBatch or None")

    meter_evidence = _validate_bound_meter_batch(prepared, bound_meter_evidence)
    meter_result = integrate_meter_evidence_v3(
        prepared.measure_geometry,
        prepared.boundary_report,
        prepared.roi_batch,
        meter_evidence,
    )
    extra = () if other_specialist_evidence is None else other_specialist_evidence.observations
    combined = SpecialistEvidenceBatch(meter_result.evidence_batch.observations + extra)
    resolver = resolve_specialist_evidence_v1(prepared.measure_geometry, combined)

    fingerprint = _canonical_sha(
        {
            "version": RUNTIME_RESOLVER_E2E_V2_VERSION,
            "system_policy": prepared.system_policy,
            "normalized_image_sha256": prepared.measure_geometry.normalized_image_sha256,
            "detected_geometry": page_geometry_fingerprint_v1(prepared.detected_geometry),
            "grouped_geometry": page_geometry_fingerprint_v1(prepared.grouped_geometry),
            "measure_geometry": page_geometry_fingerprint_v1(prepared.measure_geometry),
            "roi_config_fingerprint": prepared.roi_batch.config_fingerprint,
            "roi_hashes": [[item.roi_id, item.roi_image_sha256] for item in prepared.roi_batch.artifacts],
            "meter_fingerprint": meter_result.fingerprint(),
            "provider_fingerprint": bound_meter_evidence.provider_fingerprint,
            "evidence_origin": bound_meter_evidence.evidence_origin,
            "resolver": _resolver_payload(resolver),
        }
    )
    return RuntimeResolverE2EV2Result(
        prepared=prepared,
        meter_result=meter_result,
        resolver_result=resolver,
        evidence_origin=bound_meter_evidence.evidence_origin,
        provider_fingerprint=bound_meter_evidence.provider_fingerprint,
        fingerprint=fingerprint,
    )


def old_measure_geometry_fallback_allowed() -> bool:
    return False


def sealed_test_access_allowed() -> bool:
    return False


def real_model_proof_allowed_from_test_fixture() -> bool:
    return False
