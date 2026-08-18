"""M4-E3K-B1 TRAIN-only scoring with frozen D7-predicted staff geometry.

B1 is deliberately an attribution step, not full deployment. Frozen D7 StaffSet
predictions supply staff bbox, five-line geometry and staff spacing. D6 truth is
used only to associate a predicted staff with each known TRAIN system and to
score true interior measure boundaries. D7 ``system_region`` is not introduced
in this step, so the single changed variable relative to R2 is staff geometry.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Final

from .m4_e3k_b1_d7_staff_geometry import (
    D7_DENSE_THRESHOLD,
    EXPECTED_D7_CHECKPOINT_SHA256,
    EXPECTED_D7_STAFF_STATE_SHA256,
    MINIMUM_STAFF_COMPONENT_AREA_FRACTION,
    MINIMUM_STAFF_COMPONENT_WIDTH_FRACTION,
    PredictedStaffGeometry,
    load_frozen_d7_staff_model,
    predict_d7_staff_geometry,
)
from .m4_e3k_boundary_proposals import EVALUATION_TOLERANCES_STAFF_SPACES
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
from .m4_e3k_r2_inward_endpoint_geometry import (
    STAGE as R2_PROPOSAL_STAGE,
    propose_measure_boundaries_r2,
)
from .m4_e3k_boundary_proposals import M4E3KBoundaryProposalError
from .stage7d7_specialist_training import (
    EXPECTED_D6_ARTIFACT_BINDING_SHA256,
    EXPECTED_D6_DERIVATIVE_BUILD_ID,
    EXPECTED_D6_MANIFEST_SHA256,
    FROZEN_D7_CONFIG,
    load_verified_stage7d7_records,
)


STAGE: Final[str] = "M4-E3K-B1-TRAIN-FROZEN-D7-STAFF-GEOMETRY-SCORING"
REPORT_SCHEMA: Final[str] = "m4-e3k-b1-train-d7-staff-geometry-report-v1"
MINIMUM_SYSTEM_STAFF_MATCH_COVERAGE: Final[float] = 0.98
MINIMUM_BOUNDARY_RECALL_AT_ONE_STAFF_SPACE: Final[float] = 0.98
EXPECTED_TRAIN_RECORDS: Final[int] = 1230
EXPECTED_TRAIN_SYSTEMS: Final[int] = 2346
EXPECTED_TRAIN_INTERIOR_BOUNDARIES: Final[int] = 7494


class M4E3KB1ScoringError(RuntimeError):
    """Raised when B1 provenance, inference or scoring invariants fail closed."""


def _fail(message: str) -> None:
    raise M4E3KB1ScoringError(message)


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        _fail(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise M4E3KB1ScoringError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def _bbox(name: str, value: Mapping[str, object]) -> tuple[float, float, float, float]:
    box = _mapping(name, value)
    x0 = _finite(f"{name}.x_min", box.get("x_min"))
    y0 = _finite(f"{name}.y_min", box.get("y_min"))
    x1 = _finite(f"{name}.x_max", box.get("x_max"))
    y1 = _finite(f"{name}.y_max", box.get("y_max"))
    if not x0 < x1 or not y0 < y1:
        _fail(f"{name} must have positive extent")
    return x0, y0, x1, y1


def _bbox_iou(a: Mapping[str, object], b: Mapping[str, object]) -> float:
    ax0, ay0, ax1, ay1 = _bbox("bbox_a", a)
    bx0, by0, bx1, by1 = _bbox("bbox_b", b)
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def _predicted_center_y(item: PredictedStaffGeometry) -> float:
    return (item.staff_bbox["y_min"] + item.staff_bbox["y_max"]) / 2.0


def match_predicted_staffs_to_truth_systems(
    predicted: Sequence[PredictedStaffGeometry],
    bundles: Sequence[Mapping[str, object]],
) -> tuple[tuple[PredictedStaffGeometry | None, float | None], ...]:
    """Evaluation-only one-to-one association using truth system vertical spans.

    A predicted staff is eligible only when its center y lies inside the truth
    system bbox. Among eligible unused predictions, maximum bbox IoU with the
    truth staff wins. No minimum IoU is tuned; poor geometry remains matched and
    is allowed to hurt downstream boundary recall.
    """

    used: set[int] = set()
    result: list[tuple[PredictedStaffGeometry | None, float | None]] = []
    for bundle in bundles:
        system = _mapping("bundle.system", bundle.get("system"))
        truth_staff = _mapping("bundle.staff", bundle.get("staff"))
        system_box = _mapping("system.system_bbox", system.get("system_bbox"))
        truth_staff_box = _mapping(
            "staff.staff_instance_bbox", truth_staff.get("staff_instance_bbox")
        )
        _, sy0, _, sy1 = _bbox("system.system_bbox", system_box)
        candidates: list[tuple[float, float, int, PredictedStaffGeometry]] = []
        truth_center = sum(_bbox("truth_staff_bbox", truth_staff_box)[1::2]) / 2.0
        for index, item in enumerate(predicted):
            if index in used:
                continue
            center_y = _predicted_center_y(item)
            if not sy0 <= center_y <= sy1:
                continue
            iou = _bbox_iou(item.staff_bbox, truth_staff_box)
            center_distance = abs(center_y - truth_center)
            candidates.append((iou, -center_distance, -index, item))
        if not candidates:
            result.append((None, None))
            continue
        iou, _, negative_index, item = max(candidates)
        index = -negative_index
        used.add(index)
        result.append((item, float(iou)))
    return tuple(result)


def _profile_payload() -> dict[str, object]:
    return {
        "stage": STAGE,
        "proposal_stage": R2_PROPOSAL_STAGE,
        "surface": "TRAIN_only",
        "single_changed_variable": "D6_truth_staff_geometry_to_frozen_D7_StaffSet_geometry",
        "truth_system_usage": "evaluation_association_and_x_search_bound_only",
        "d7_system_region_used": False,
        "d7_dense_threshold": D7_DENSE_THRESHOLD,
        "staff_component_width_fraction": MINIMUM_STAFF_COMPONENT_WIDTH_FRACTION,
        "staff_component_area_fraction": MINIMUM_STAFF_COMPONENT_AREA_FRACTION,
        "d7_checkpoint_sha256": EXPECTED_D7_CHECKPOINT_SHA256,
        "d7_staff_state_sha256": EXPECTED_D7_STAFF_STATE_SHA256,
        "d7_input": [FROZEN_D7_CONFIG.input_height, FROZEN_D7_CONFIG.input_width],
        "d6_derivative_build_id": EXPECTED_D6_DERIVATIVE_BUILD_ID,
        "d6_manifest_sha256": EXPECTED_D6_MANIFEST_SHA256,
        "d6_artifact_binding_sha256": EXPECTED_D6_ARTIFACT_BINDING_SHA256,
        "minimum_system_staff_match_coverage": MINIMUM_SYSTEM_STAFF_MATCH_COVERAGE,
        "minimum_boundary_recall_at_one_staff_space": MINIMUM_BOUNDARY_RECALL_AT_ONE_STAFF_SPACE,
        "error_normalization": "authoritative_D6_truth_staff_spacing",
    }


def profile_fingerprint() -> str:
    return sha256(_canonical_json(_profile_payload())).hexdigest()


def score_e3k_b1_train(
    corpus_root: str | Path,
    d6_root: str | Path,
    d7_checkpoint_path: str | Path,
) -> dict[str, object]:
    """Run frozen D7 StaffSet -> R2 proposal scoring on TRAIN only."""

    records = load_verified_stage7d7_records(corpus_root, d6_root)
    train_records = tuple(record for record in records if record.split == "train")
    if len(train_records) != EXPECTED_TRAIN_RECORDS:
        _fail("B1 expected exactly 1230 TRAIN records")
    if any(record.split != "train" for record in train_records):
        _fail("B1 selected surface crossed TRAIN boundary")

    model = load_frozen_d7_staff_model(d7_checkpoint_path)

    system_count = 0
    matched_systems = 0
    systems_with_interior = 0
    boundary_count = 0
    total_errors: list[float] = []
    proposal_counts: list[int] = []
    predicted_staff_counts: list[int] = []
    match_ious: list[float] = []
    reason_counts: Counter[str] = Counter()
    no_proposal_systems = 0

    for record in train_records:
        label = _read_label(record)
        image = _read_image(record, label)
        bundles = tuple(_system_objects(label))
        try:
            predicted = predict_d7_staff_geometry(model, record, label)
        except Exception as exc:
            raise M4E3KB1ScoringError(
                f"frozen D7 StaffSet inference failed for {record.sample_id}: {exc}"
            ) from exc
        predicted_staff_counts.append(len(predicted))
        matches = match_predicted_staffs_to_truth_systems(predicted, bundles)
        if len(matches) != len(bundles):
            _fail("B1 staff/system association cardinality mismatch")

        for bundle, (predicted_staff, match_iou) in zip(bundles, matches):
            system_count += 1
            system = _mapping("bundle.system", bundle.get("system"))
            truth_staff = _mapping("bundle.staff", bundle.get("staff"))
            measures = tuple(_sequence("bundle.measures", bundle.get("measures")))
            truth_lines = _sequence("truth_staff.five_staff_lines", truth_staff.get("five_staff_lines"))
            truth_spacing = _positive_number("truth_staff.staff_spacing", truth_staff.get("staff_spacing"))
            interior = measures[:-1]
            if interior:
                systems_with_interior += 1
            truth_xs: list[float] = []
            for measure_obj in interior:
                measure = _mapping("interior measure", measure_obj)
                barline = _mapping("measure.barline_segment", measure.get("barline_segment"))
                truth_xs.append(_line_intersection_x_at_middle_staff(barline, truth_lines))
            boundary_count += len(truth_xs)

            if predicted_staff is None:
                proposal_counts.append(0)
                no_proposal_systems += 1
                reason_counts["UNMATCHED_D7_STAFF"] += len(truth_xs)
                total_errors.extend([math.inf] * len(truth_xs))
                continue

            matched_systems += 1
            if match_iou is None:
                _fail("matched D7 staff is missing evaluation IoU")
            match_ious.append(match_iou)
            try:
                proposal_result = propose_measure_boundaries_r2(
                    image,
                    staff_bbox=predicted_staff.staff_bbox,
                    five_staff_lines=predicted_staff.five_staff_lines,
                    staff_spacing=predicted_staff.staff_spacing,
                    system_bbox=_mapping("system.system_bbox", system.get("system_bbox")),
                )
                proposals = proposal_result.proposals
            except M4E3KBoundaryProposalError:
                proposals = ()
                reason_counts["R2_PROPOSAL_GEOMETRY_FAIL"] += len(truth_xs)
            proposal_counts.append(len(proposals))
            if not proposals:
                no_proposal_systems += 1
            errors = _proposal_errors(proposals, truth_xs, staff_spacing=truth_spacing)
            total_errors.extend(errors)
            for error in errors:
                if not math.isfinite(error):
                    reason_counts["NO_PROPOSAL_FOR_BOUNDARY"] += 1
                elif error > 2.0:
                    reason_counts["NEAREST_PROPOSAL_OVER_2_STAFF_SPACES"] += 1

    if system_count != EXPECTED_TRAIN_SYSTEMS:
        _fail(f"B1 expected {EXPECTED_TRAIN_SYSTEMS} TRAIN systems, got {system_count}")
    if boundary_count != EXPECTED_TRAIN_INTERIOR_BOUNDARIES:
        _fail(
            f"B1 expected {EXPECTED_TRAIN_INTERIOR_BOUNDARIES} TRAIN interior boundaries, "
            f"got {boundary_count}"
        )
    if len(total_errors) != boundary_count:
        _fail("B1 boundary error cardinality mismatch")
    if len(proposal_counts) != system_count:
        _fail("B1 proposal-count cardinality mismatch")

    match_coverage = matched_systems / system_count
    recall = {
        str(tolerance): sum(error <= tolerance for error in total_errors) / boundary_count
        for tolerance in EVALUATION_TOLERANCES_STAFF_SPACES
    }
    recall_one = recall["1.0"]
    gate_pass = (
        match_coverage >= MINIMUM_SYSTEM_STAFF_MATCH_COVERAGE
        and recall_one >= MINIMUM_BOUNDARY_RECALL_AT_ONE_STAFF_SPACE
    )

    return {
        "schema_version": REPORT_SCHEMA,
        "stage": STAGE,
        "state": "COMPLETE_PASS_TRAIN_ONLY" if gate_pass else "COMPLETE_FAIL_PRESERVED",
        "profile_fingerprint": profile_fingerprint(),
        "profile": _profile_payload(),
        "surface": {
            "split": "train",
            "records": len(train_records),
            "systems": system_count,
            "systems_with_interior_boundaries": systems_with_interior,
            "topology_relevant_interior_boundaries": boundary_count,
        },
        "metrics": {
            "d7_predicted_staff_count_per_page_p50": _percentile(predicted_staff_counts, 0.50),
            "d7_predicted_staff_count_per_page_p95": _percentile(predicted_staff_counts, 0.95),
            "matched_systems": matched_systems,
            "unmatched_systems": system_count - matched_systems,
            "system_staff_match_coverage": match_coverage,
            "matched_staff_bbox_iou_p50": None if not match_ious else _percentile(match_ious, 0.50),
            "matched_staff_bbox_iou_p95": None if not match_ious else _percentile(match_ious, 0.95),
            "boundary_recall_by_tolerance_staff_spaces": recall,
            "nearest_boundary_error_p50_staff_spaces": _percentile_or_none(total_errors, 0.50),
            "nearest_boundary_error_p95_staff_spaces": _percentile_or_none(total_errors, 0.95),
            "proposal_count_per_system_p50": _percentile(proposal_counts, 0.50),
            "proposal_count_per_system_p95": _percentile(proposal_counts, 0.95),
            "proposal_count_per_system_max": max(proposal_counts),
            "no_proposal_systems": no_proposal_systems,
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "gate": {
            "minimum_system_staff_match_coverage": MINIMUM_SYSTEM_STAFF_MATCH_COVERAGE,
            "system_staff_match_coverage": match_coverage,
            "minimum_boundary_recall_at_1_staff_space": MINIMUM_BOUNDARY_RECALL_AT_ONE_STAFF_SPACE,
            "boundary_recall_at_1_staff_space": recall_one,
            "pass": gate_pass,
            "authorizes_e3k_b2_system_geometry": gate_pass,
            "authorizes_d11_validator": False,
        },
        "safety": {
            "train_only": True,
            "validation_opened": False,
            "test_opened": False,
            "final_a_opened": False,
            "final_b_opened": False,
            "d7_weights_loaded": True,
            "d11_weights_loaded": False,
            "training_started": False,
            "optimizer_steps": 0,
            "threshold_tuning": False,
            "production_promotion": False,
            "truth_staff_geometry_used_for_proposals": False,
            "truth_system_bbox_used_for_evaluation_association": True,
        },
    }


def persist_e3k_b1_train_report(
    corpus_root: str | Path,
    d6_root: str | Path,
    d7_checkpoint_path: str | Path,
    *,
    report_path: str | Path,
) -> dict[str, object]:
    path = Path(report_path)
    if path.exists() or path.is_symlink():
        _fail("B1 report path must be fresh")
    if not path.parent.is_dir() or path.parent.is_symlink():
        _fail("B1 report parent must be an existing regular directory")
    report = score_e3k_b1_train(corpus_root, d6_root, d7_checkpoint_path)
    raw = _canonical_json(report)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        _fail("B1 temporary report path must be fresh")
    try:
        temporary.write_bytes(raw)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if path.read_bytes() != raw:
        _fail("persisted B1 report bytes failed verification")
    return report
