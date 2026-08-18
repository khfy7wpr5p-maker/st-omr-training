"""Isolated shadow orchestrator from raster page to deterministic resolver.

The orchestration path is deliberately model-agnostic.  It prepares normalized
page, staff geometry, measure geometry and in-memory runtime ROIs, then accepts
an explicit SpecialistEvidenceBatch for deterministic resolution.  No D10/D13
model/checkpoint is imported or loaded here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .runtime_deterministic_resolver_v1 import (
    DeterministicResolverResult,
    resolve_specialist_evidence_v1,
)
from .runtime_geometry_engine_contract import GeometryInputContract, PageGeometryContract
from .runtime_geometry_engine_v2 import detect_multistaff_geometry_v2
from .runtime_local_roi_v1 import RuntimeRoiBatch, extract_runtime_rois_v1
from .runtime_measure_geometry_v1 import propose_measure_geometry_v1
from .runtime_page_normalizer_contract import NormalizedPageContract, RasterPageInputContract
from .runtime_page_normalizer_v1 import normalize_raster_page_v1
from .runtime_specialist_evidence_v1 import SpecialistEvidenceBatch


@dataclass(frozen=True, slots=True)
class RuntimeShadowPrepared:
    normalized_png: bytes
    normalized_page: NormalizedPageContract
    staff_geometry: PageGeometryContract
    measure_geometry: PageGeometryContract
    roi_batch: RuntimeRoiBatch


def prepare_runtime_shadow_v1(
    raster_bytes: bytes,
    raster_contract: RasterPageInputContract,
) -> RuntimeShadowPrepared:
    """Run the deterministic image/geometry/ROI lane and stop before specialists."""
    normalized = normalize_raster_page_v1(raster_bytes, raster_contract)
    if normalized.page.status != "accepted" or normalized.normalized_png is None:
        raise ValueError(f"page normalization did not accept input: {normalized.page.rejection_reasons}")
    page = normalized.page
    if (
        page.normalized_image_sha256 is None
        or page.normalized_width is None
        or page.normalized_height is None
        or page.transform is None
    ):
        raise ValueError("accepted normalized page is missing required runtime identity")

    geometry_input = GeometryInputContract(
        normalized_image_sha256=page.normalized_image_sha256,
        normalizer_config_fingerprint=page.normalizer_config_fingerprint,
        normalized_width=page.normalized_width,
        normalized_height=page.normalized_height,
        transform=page.transform,
    )
    staff_result = detect_multistaff_geometry_v2(normalized.normalized_png, geometry_input)
    if staff_result.page.status != "accepted":
        raise ValueError(f"staff geometry is ambiguous: {staff_result.page.reasons}")

    measure_result = propose_measure_geometry_v1(normalized.normalized_png, staff_result.page)
    if measure_result.page.status != "accepted":
        raise ValueError(f"measure geometry is ambiguous: {measure_result.page.reasons}")

    roi_batch = extract_runtime_rois_v1(normalized.normalized_png, measure_result.page)
    return RuntimeShadowPrepared(
        normalized_png=normalized.normalized_png,
        normalized_page=page,
        staff_geometry=staff_result.page,
        measure_geometry=measure_result.page,
        roi_batch=roi_batch,
    )


def resolve_runtime_shadow_v1(
    prepared: RuntimeShadowPrepared,
    evidence: SpecialistEvidenceBatch,
) -> DeterministicResolverResult:
    """Resolve explicit specialist evidence without loading any specialist model."""
    if not isinstance(prepared, RuntimeShadowPrepared):
        raise TypeError("prepared must be RuntimeShadowPrepared")
    return resolve_specialist_evidence_v1(prepared.measure_geometry, evidence)
