"""Authoritative development scoring for M4-E3K boundary proposals.

E3K-A is a read-only *geometry upper-bound* test. It reuses the accepted D6/D7
development-record gate, opens only the requested TRAIN or VALIDATION payloads,
uses authoritative D6 staff geometry to align the deterministic proposal probe,
and scores topology-relevant interior measure boundaries.

Passing E3K-A does not authorize D11 integration or deployment. A separate
E3K-B package must replace D6 staff geometry with frozen D7-predicted staff
geometry and pass its own development gate first.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Final

from PIL import Image

from .m4_e3k_boundary_proposals import (
    EVALUATION_TOLERANCES_STAFF_SPACES,
    FROZEN_E3K_CONFIG,
    STAGE as PROPOSAL_STAGE,
    BoundaryProposal,
    M4E3KBoundaryProposalError,
    propose_measure_boundaries,
)
from .stage7d6_specialist_derivatives import STAGE7D6_LABEL_SCHEMA, STAGE7D6_VERSION
from .stage7d7_specialist_training import (
    EXPECTED_D6_ARTIFACT_BINDING_SHA256,
    EXPECTED_D6_DERIVATIVE_BUILD_ID,
    EXPECTED_D6_MANIFEST_SHA256,
    Stage7D7Record,
    load_verified_stage7d7_records,
)


STAGE: Final[str] = "M4-E3K-A-AUTHORITATIVE-D6-GEOMETRY-FEASIBILITY"
REPORT_SCHEMA: Final[str] = "m4-e3k-a-boundary-feasibility-report-v1"
ALLOWED_SPLITS: Final[frozenset[str]] = frozenset({"train", "validation"})
MINIMUM_RECALL_AT_ONE_STAFF_SPACE: Final[float] = 0.98
MAX_LABEL_BYTES: Final[int] = 2 * 1024 * 1024
MAX_IMAGE_BYTES: Final[int] = 64 * 1024 * 1024


class M4E3KScoringError(RuntimeError):
    """Raised when E3K-A provenance, geometry, or scoring fails closed."""


def _fail(message: str) -> None:
    raise M4E3KScoringError(message)


def _canonical_json(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise M4E3KScoringError("report is not canonical JSON serializable") from exc


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        _fail(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise M4E3KScoringError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def _positive_number(name: str, value: object) -> float:
    result = _finite(name, value)
    if result <= 0:
        _fail(f"{name} must be positive")
    return result


def _positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _fail(f"{name} must be a positive integer")
    return value


def _mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be a mapping")
    return value


def _sequence(name: str, value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(f"{name} must be a sequence")
    return value


def _point(name: str, value: object) -> tuple[float, float]:
    point = _mapping(name, value)
    return (
        _finite(f"{name}.x", point.get("x")),
        _finite(f"{name}.y", point.get("y")),
    )


def _profile_payload() -> dict[str, object]:
    return {
        "stage": STAGE,
        "proposal_stage": PROPOSAL_STAGE,
        "proposal_config": asdict(FROZEN_E3K_CONFIG),
        "evaluation_tolerances_staff_spaces": list(EVALUATION_TOLERANCES_STAFF_SPACES),
        "minimum_recall_at_one_staff_space": MINIMUM_RECALL_AT_ONE_STAFF_SPACE,
        "truth_surface": "interior_measure_trailing_barlines_only",
        "staff_geometry_source": "authoritative_D6_ground_truth_upper_bound",
        "d6_derivative_build_id": EXPECTED_D6_DERIVATIVE_BUILD_ID,
        "d6_manifest_sha256": EXPECTED_D6_MANIFEST_SHA256,
        "d6_artifact_binding_sha256": EXPECTED_D6_ARTIFACT_BINDING_SHA256,
    }


def profile_fingerprint() -> str:
    return sha256(_canonical_json(_profile_payload())).hexdigest()


def _read_label(record: Stage7D7Record) -> dict[str, object]:
    path = record.label_path
    if path.is_symlink() or not path.is_file():
        _fail("D6 label must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= MAX_LABEL_BYTES:
        _fail("D6 label byte length is outside E3K-A bound")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != record.label_sha256:
        _fail("D6 label SHA-256 mismatch")
    try:
        payload = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda value: _fail(f"non-finite D6 label constant: {value}"),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise M4E3KScoringError("D6 label is not valid ASCII JSON") from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        _fail("D6 label must be canonical JSON object bytes")
    if payload.get("schema_version") != STAGE7D6_LABEL_SCHEMA:
        _fail("D6 label schema mismatch")
    if payload.get("stage7d6_version") != STAGE7D6_VERSION:
        _fail("D6 label version mismatch")
    if payload.get("sample_id") != record.sample_id:
        _fail("D6 label sample identity mismatch")
    if payload.get("split") != record.split:
        _fail("D6 label split mismatch")
    return payload


def _read_image(record: Stage7D7Record, label: Mapping[str, object]) -> Image.Image:
    path = record.image_path
    if path.is_symlink() or not path.is_file():
        _fail("source PNG must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= MAX_IMAGE_BYTES:
        _fail("source PNG byte length is outside E3K-A bound")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != record.png_sha256:
        _fail("source PNG SHA-256 mismatch")
    image_meta = _mapping("label.image", label.get("image"))
    width = _positive_int("label.image.width", image_meta.get("width"))
    height = _positive_int("label.image.height", image_meta.get("height"))
    try:
        with Image.open(BytesIO(raw)) as opened:
            opened.load()
            if opened.format != "PNG" or opened.mode != "L":
                _fail("E3K-A source must be grayscale PNG")
            if opened.size != (width, height):
                _fail("E3K-A source dimensions differ from D6 label")
            return opened.copy()
    except M4E3KScoringError:
        raise
    except Exception as exc:
        raise M4E3KScoringError("E3K-A source PNG decode failed") from exc


def _middle_staff_line(
    five_staff_lines: Sequence[object],
) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(five_staff_lines) != 5:
        _fail("E3K-A requires exactly five staff lines")
    parsed: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    for index, item in enumerate(five_staff_lines):
        line = _mapping(f"five_staff_lines[{index}]", item)
        start = _point(f"five_staff_lines[{index}].start", line.get("start"))
        end = _point(f"five_staff_lines[{index}].end", line.get("end"))
        if math.isclose(start[0], end[0], abs_tol=1e-12):
            _fail("staff line cannot be vertical")
        parsed.append((start, end, (start[1] + end[1]) / 2.0))
    parsed.sort(key=lambda item: item[2])
    return parsed[2][0], parsed[2][1]


def _line_intersection_x_at_middle_staff(
    barline: Mapping[str, object],
    five_staff_lines: Sequence[object],
) -> float:
    """Return trailing-barline x where it intersects the middle staff line."""

    b0 = _point("barline.start", barline.get("start"))
    b1 = _point("barline.end", barline.get("end"))
    s0, s1 = _middle_staff_line(five_staff_lines)
    bdx = b1[0] - b0[0]
    bdy = b1[1] - b0[1]
    sdx = s1[0] - s0[0]
    sdy = s1[1] - s0[1]
    denominator = bdx * sdy - bdy * sdx
    if abs(denominator) < 1e-9:
        _fail("barline is parallel to the middle staff line")
    qx = s0[0] - b0[0]
    qy = s0[1] - b0[1]
    t = (qx * sdy - qy * sdx) / denominator
    if not -0.10 <= t <= 1.10:
        _fail("barline does not cross the middle staff line within bounded tolerance")
    x = b0[0] + t * bdx
    if not math.isfinite(x):
        _fail("barline/staff intersection x is non-finite")
    return x


def _percentile(values: Sequence[float | int], fraction: float) -> float:
    if not values:
        _fail("cannot compute percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _percentile_or_none(values: Sequence[float], fraction: float) -> float | None:
    """Treat missing proposals as +infinity without serializing non-finite JSON."""

    if not values:
        _fail("cannot compute error percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0] if math.isfinite(ordered[0]) else None
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if not math.isfinite(ordered[lower]) or not math.isfinite(ordered[upper]):
        return None
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _proposal_errors(
    proposals: Sequence[BoundaryProposal],
    truth_xs: Sequence[float],
    *,
    staff_spacing: float,
) -> list[float]:
    spacing = _positive_number("staff_spacing", staff_spacing)
    candidate_xs = [proposal.x for proposal in proposals]
    if not candidate_xs:
        return [math.inf] * len(truth_xs)
    return [min(abs(candidate - truth) for candidate in candidate_xs) / spacing for truth in truth_xs]


def _system_objects(label: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    geometry = _mapping("label.geometry", label.get("geometry"))
    systems_raw = _sequence("geometry.systems", geometry.get("systems"))
    staffs_raw = _sequence("geometry.staff_instances", geometry.get("staff_instances"))
    measures_raw = _sequence("geometry.measures", geometry.get("measures"))

    systems: dict[str, Mapping[str, object]] = {}
    for item in systems_raw:
        system = _mapping("system", item)
        system_id = system.get("system_id")
        if not isinstance(system_id, str) or not system_id or system_id in systems:
            _fail("system ids must be non-empty and unique")
        systems[system_id] = system

    staffs: dict[str, Mapping[str, object]] = {}
    for item in staffs_raw:
        staff = _mapping("staff", item)
        system_id = staff.get("system_id")
        if not isinstance(system_id, str) or system_id not in systems or system_id in staffs:
            _fail("each system must resolve to exactly one staff instance")
        staffs[system_id] = staff
    if set(staffs) != set(systems):
        _fail("staff/system identity surfaces differ")

    measures_by_number: dict[int, Mapping[str, object]] = {}
    for item in measures_raw:
        measure = _mapping("measure", item)
        number = _positive_int("measure.measure_number", measure.get("measure_number"))
        system_id = measure.get("system_id")
        if not isinstance(system_id, str) or system_id not in systems:
            _fail("measure references unknown system")
        if number in measures_by_number:
            _fail("duplicate measure number")
        measures_by_number[number] = measure

    result: list[Mapping[str, object]] = []
    for system_id, system in systems.items():
        numbers_raw = _sequence("system.measure_numbers", system.get("measure_numbers"))
        numbers = [_positive_int("system.measure_number", value) for value in numbers_raw]
        if not numbers or len(set(numbers)) != len(numbers):
            _fail("system measure-number sequence is empty or duplicated")
        try:
            measures = [measures_by_number[number] for number in numbers]
        except KeyError as exc:
            raise M4E3KScoringError("system references missing measure") from exc
        if any(measure.get("system_id") != system_id for measure in measures):
            _fail("system measure identity mismatch")
        result.append({"system": system, "staff": staffs[system_id], "measures": tuple(measures)})
    return tuple(result)


def score_e3k_a_split(
    corpus_root: str | Path,
    d6_root: str | Path,
    *,
    split: str,
) -> dict[str, object]:
    """Score one requested development split without mutating any model/data."""

    if split not in ALLOWED_SPLITS:
        _fail("E3K-A split must be train or validation; TEST is forbidden")
    records = load_verified_stage7d7_records(corpus_root, d6_root)
    selected = tuple(record for record in records if record.split == split)
    if not selected:
        _fail("requested E3K-A split is empty")
    if any(record.split != split for record in selected):
        _fail("E3K-A selected surface crossed split boundary")

    total_errors: list[float] = []
    proposal_counts: list[int] = []
    system_count = 0
    systems_with_interior_boundaries = 0
    boundary_count = 0
    no_proposal_systems = 0
    reason_counts: Counter[str] = Counter()

    for record in selected:
        label = _read_label(record)
        image = _read_image(record, label)
        for bundle in _system_objects(label):
            system_count += 1
            system = _mapping("bundle.system", bundle.get("system"))
            staff = _mapping("bundle.staff", bundle.get("staff"))
            measures = tuple(_sequence("bundle.measures", bundle.get("measures")))
            five_staff_lines = _sequence("staff.five_staff_lines", staff.get("five_staff_lines"))
            staff_spacing = _positive_number("staff.staff_spacing", staff.get("staff_spacing"))
            try:
                proposal_result = propose_measure_boundaries(
                    image,
                    staff_bbox=_mapping("staff.staff_instance_bbox", staff.get("staff_instance_bbox")),
                    five_staff_lines=five_staff_lines,
                    staff_spacing=staff_spacing,
                    system_bbox=_mapping("system.system_bbox", system.get("system_bbox")),
                )
            except M4E3KBoundaryProposalError as exc:
                raise M4E3KScoringError(
                    f"proposal generation failed for {record.sample_id}/{system.get('system_id')}: {exc}"
                ) from exc
            proposals = proposal_result.proposals
            proposal_counts.append(len(proposals))
            if not proposals:
                no_proposal_systems += 1

            # Only a trailing barline with a following measure becomes a new
            # measure-start boundary. The final barline of a system is excluded.
            interior = measures[:-1]
            if not interior:
                continue
            systems_with_interior_boundaries += 1
            truth_xs: list[float] = []
            for measure_obj in interior:
                measure = _mapping("interior measure", measure_obj)
                barline = _mapping("measure.barline_segment", measure.get("barline_segment"))
                truth_xs.append(_line_intersection_x_at_middle_staff(barline, five_staff_lines))
            boundary_count += len(truth_xs)
            errors = _proposal_errors(proposals, truth_xs, staff_spacing=staff_spacing)
            total_errors.extend(errors)
            for error in errors:
                if not math.isfinite(error):
                    reason_counts["NO_PROPOSAL"] += 1
                elif error > 2.0:
                    reason_counts["NEAREST_GT_OVER_2_STAFF_SPACES"] += 1

    if boundary_count <= 0 or len(total_errors) != boundary_count:
        _fail("E3K-A topology boundary surface is empty or inconsistent")
    if not proposal_counts or len(proposal_counts) != system_count:
        _fail("E3K-A proposal-count surface is inconsistent")

    recall = {
        str(tolerance): sum(error <= tolerance for error in total_errors) / boundary_count
        for tolerance in EVALUATION_TOLERANCES_STAFF_SPACES
    }
    p50_error = _percentile_or_none(total_errors, 0.50)
    p95_error = _percentile_or_none(total_errors, 0.95)
    recall_one = recall["1.0"]
    gate_pass = recall_one >= MINIMUM_RECALL_AT_ONE_STAFF_SPACE

    return {
        "schema_version": REPORT_SCHEMA,
        "stage": STAGE,
        "state": "COMPLETE_PASS" if gate_pass else "COMPLETE_FAIL_PRESERVED",
        "split": split,
        "profile_fingerprint": profile_fingerprint(),
        "profile": _profile_payload(),
        "surface": {
            "records": len(selected),
            "systems": system_count,
            "systems_with_interior_boundaries": systems_with_interior_boundaries,
            "topology_relevant_interior_boundaries": boundary_count,
        },
        "metrics": {
            "boundary_recall_by_tolerance_staff_spaces": recall,
            "nearest_boundary_error_p50_staff_spaces": p50_error,
            "nearest_boundary_error_p95_staff_spaces": p95_error,
            "proposal_count_per_system_p50": _percentile(proposal_counts, 0.50),
            "proposal_count_per_system_p95": _percentile(proposal_counts, 0.95),
            "proposal_count_per_system_max": max(proposal_counts),
            "no_proposal_systems": no_proposal_systems,
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "gate": {
            "minimum_boundary_recall_at_1_staff_space": MINIMUM_RECALL_AT_ONE_STAFF_SPACE,
            "boundary_recall_at_1_staff_space": recall_one,
            "pass": gate_pass,
            "authorizes_e3k_b": gate_pass,
            "authorizes_d11_validator": False,
        },
        "safety": {
            "staff_geometry_source": "authoritative_D6_ground_truth_upper_bound_only",
            "deployment_readiness_claimed": False,
            "d7_weights_loaded": False,
            "d11_weights_loaded": False,
            "optimizer_steps": 0,
            "training_started": False,
            "threshold_tuning": False,
            "test_opened": False,
            "final_a_opened": False,
            "final_b_opened": False,
            "production_promotion": False,
        },
    }


def persist_e3k_a_report(
    corpus_root: str | Path,
    d6_root: str | Path,
    *,
    split: str,
    report_path: str | Path,
) -> dict[str, object]:
    """Score and atomically persist one fresh canonical report file."""

    path = Path(report_path)
    if path.exists() or path.is_symlink():
        _fail("E3K-A report path must be fresh")
    if not path.parent.is_dir() or path.parent.is_symlink():
        _fail("E3K-A report parent must be an existing regular directory")
    report = score_e3k_a_split(corpus_root, d6_root, split=split)
    raw = _canonical_json(report)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        _fail("E3K-A temporary report path must be fresh")
    try:
        temporary.write_bytes(raw)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if path.read_bytes() != raw:
        _fail("persisted E3K-A report bytes failed verification")
    return report
