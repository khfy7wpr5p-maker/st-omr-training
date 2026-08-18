"""TRAIN-only scoring for M4-E3K-R2 inward endpoint geometry recovery.

This scorer replays the same accepted D6 authoritative geometry surface used by
E3K-A, but swaps only the proposal function to the R2 inward-endpoint variant.
It is intentionally TRAIN-only. VALIDATION and TEST are not accepted entry
points, and passing R2 TRAIN does not authorize D11 or promotion.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Final
import math

from .m4_e3k_boundary_proposals import EVALUATION_TOLERANCES_STAFF_SPACES, FROZEN_E3K_CONFIG
from .m4_e3k_boundary_scoring import (
    _canonical_json,
    _line_intersection_x_at_middle_staff,
    _mapping,
    _percentile,
    _percentile_or_none,
    _positive_number,
    _proposal_errors,
    _read_image,
    _read_label,
    _sequence,
    _system_objects,
)
from .m4_e3k_r2_inward_endpoint_geometry import STAGE as PROPOSAL_STAGE
from .m4_e3k_r2_inward_endpoint_geometry import propose_measure_boundaries_r2
from .stage7d7_specialist_training import (
    EXPECTED_D6_ARTIFACT_BINDING_SHA256,
    EXPECTED_D6_DERIVATIVE_BUILD_ID,
    EXPECTED_D6_MANIFEST_SHA256,
    load_verified_stage7d7_records,
)


STAGE: Final[str] = "M4-E3K-R2-TRAIN-INWARD-ENDPOINT-RECOVERY"
REPORT_SCHEMA: Final[str] = "m4-e3k-r2-train-inward-endpoint-recovery-v1"
MINIMUM_RECALL_AT_ONE_STAFF_SPACE: Final[float] = 0.98
EXPECTED_TRAIN_RECORDS: Final[int] = 1230
EXPECTED_TRAIN_SYSTEMS: Final[int] = 2346
EXPECTED_INTERIOR_BOUNDARIES: Final[int] = 7494


class M4E3KR2ScoringError(RuntimeError):
    """Raised when the R2 TRAIN scoring contract fails closed."""


def _fail(message: str) -> None:
    raise M4E3KR2ScoringError(message)


def score_e3k_r2_train(
    corpus_root: str | Path,
    d6_root: str | Path,
) -> dict[str, object]:
    """Score the one-variable R2 recovery on accepted TRAIN only."""

    records = load_verified_stage7d7_records(corpus_root, d6_root)
    selected = tuple(record for record in records if record.split == "train")
    if len(selected) != EXPECTED_TRAIN_RECORDS:
        _fail(
            f"R2 expected {EXPECTED_TRAIN_RECORDS} TRAIN records, got {len(selected)}"
        )
    if any(record.split != "train" for record in selected):
        _fail("R2 selected surface crossed TRAIN boundary")

    total_errors: list[float] = []
    proposal_counts: list[int] = []
    reason_counts: Counter[str] = Counter()
    system_count = 0
    systems_with_interior_boundaries = 0
    boundary_count = 0
    no_proposal_systems = 0

    for record in selected:
        label = _read_label(record)
        image = _read_image(record, label)

        for bundle in _system_objects(label):
            system_count += 1
            system = _mapping("bundle.system", bundle.get("system"))
            staff = _mapping("bundle.staff", bundle.get("staff"))
            measures = tuple(_sequence("bundle.measures", bundle.get("measures")))
            five_staff_lines = _sequence(
                "staff.five_staff_lines",
                staff.get("five_staff_lines"),
            )
            spacing = _positive_number(
                "staff.staff_spacing",
                staff.get("staff_spacing"),
            )
            staff_bbox = _mapping(
                "staff.staff_instance_bbox",
                staff.get("staff_instance_bbox"),
            )
            system_bbox = _mapping("system.system_bbox", system.get("system_bbox"))

            result = propose_measure_boundaries_r2(
                image,
                staff_bbox=staff_bbox,
                five_staff_lines=five_staff_lines,
                staff_spacing=spacing,
                system_bbox=system_bbox,
            )
            proposal_counts.append(len(result.proposals))
            if not result.proposals:
                no_proposal_systems += 1

            interior = measures[:-1]
            if interior:
                systems_with_interior_boundaries += 1
            truths: list[float] = []
            for measure_obj in interior:
                measure = _mapping("interior measure", measure_obj)
                barline = _mapping(
                    "measure.barline_segment",
                    measure.get("barline_segment"),
                )
                truths.append(
                    _line_intersection_x_at_middle_staff(
                        barline,
                        five_staff_lines,
                    )
                )

            errors = _proposal_errors(
                result.proposals,
                truths,
                staff_spacing=spacing,
            ) if truths else []
            total_errors.extend(errors)
            boundary_count += len(truths)
            reason_counts[
                "NO_PROPOSAL_SYSTEM" if truths and not result.proposals else "SCORED"
            ] += len(truths)

    if system_count != EXPECTED_TRAIN_SYSTEMS:
        _fail(f"R2 expected {EXPECTED_TRAIN_SYSTEMS} systems, got {system_count}")
    if boundary_count != EXPECTED_INTERIOR_BOUNDARIES:
        _fail(
            f"R2 expected {EXPECTED_INTERIOR_BOUNDARIES} interior boundaries, "
            f"got {boundary_count}"
        )
    if len(total_errors) != boundary_count:
        _fail("R2 error cardinality differs from truth cardinality")
    if len(proposal_counts) != system_count:
        _fail("R2 proposal-count cardinality differs from system count")

    recall_by_tolerance = {
        str(tolerance): sum(error <= tolerance for error in total_errors) / boundary_count
        for tolerance in EVALUATION_TOLERANCES_STAFF_SPACES
    }
    recall_one = recall_by_tolerance["1.0"]
    gate_pass = recall_one >= MINIMUM_RECALL_AT_ONE_STAFF_SPACE

    finite_errors = [error for error in total_errors if math.isfinite(error)]
    report = {
        "schema_version": REPORT_SCHEMA,
        "stage": STAGE,
        "state": "COMPLETE_PASS_TRAIN_ONLY" if gate_pass else "COMPLETE_FAIL_PRESERVED",
        "surface": {
            "split": "train",
            "records": len(selected),
            "systems": system_count,
            "systems_with_interior_boundaries": systems_with_interior_boundaries,
            "topology_relevant_interior_boundaries": boundary_count,
        },
        "profile": {
            "proposal_stage": PROPOSAL_STAGE,
            "proposal_config": asdict(FROZEN_E3K_CONFIG),
            "single_change": "symmetric_endpoint_windows_to_inward_endpoint_windows",
            "endpoint_nominal_span_policy": "preserve_2x_frozen_half_window_on_inward_side",
            "d6_derivative_build_id": EXPECTED_D6_DERIVATIVE_BUILD_ID,
            "d6_manifest_sha256": EXPECTED_D6_MANIFEST_SHA256,
            "d6_artifact_binding_sha256": EXPECTED_D6_ARTIFACT_BINDING_SHA256,
            "minimum_recall_at_one_staff_space": MINIMUM_RECALL_AT_ONE_STAFF_SPACE,
        },
        "metrics": {
            "boundary_recall_by_tolerance_staff_spaces": recall_by_tolerance,
            "nearest_boundary_error_p50_staff_spaces": (
                _percentile_or_none(total_errors, 0.50)
            ),
            "nearest_boundary_error_p95_staff_spaces": (
                _percentile_or_none(total_errors, 0.95)
            ),
            "finite_nearest_error_count": len(finite_errors),
            "proposal_count_per_system_p50": _percentile(proposal_counts, 0.50),
            "proposal_count_per_system_p95": _percentile(proposal_counts, 0.95),
            "proposal_count_per_system_max": max(proposal_counts),
            "no_proposal_systems": no_proposal_systems,
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "gate": {
            "boundary_recall_at_1_staff_space": recall_one,
            "minimum_boundary_recall_at_1_staff_space": MINIMUM_RECALL_AT_ONE_STAFF_SPACE,
            "pass": gate_pass,
            "authorizes_next_root_cause_step": gate_pass,
            "authorizes_e3k_b": False,
            "authorizes_d11_validator": False,
        },
        "safety": {
            "train_only": True,
            "validation_opened": False,
            "test_opened": False,
            "d7_weights_loaded": False,
            "d11_weights_loaded": False,
            "training_started": False,
            "optimizer_steps": 0,
            "threshold_tuning": False,
            "production_promotion": False,
        },
    }
    _canonical_json(report)
    return report


def persist_e3k_r2_train_report(
    corpus_root: str | Path,
    d6_root: str | Path,
    *,
    report_path: str | Path,
) -> dict[str, object]:
    path = Path(report_path)
    if path.exists() or path.is_symlink():
        _fail("R2 report path must be fresh")
    if not path.parent.is_dir():
        _fail("R2 report parent must already exist")
    report = score_e3k_r2_train(corpus_root, d6_root)
    raw = _canonical_json(report) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(raw)
    except FileExistsError as exc:
        raise M4E3KR2ScoringError("R2 report path already exists") from exc
    return report
