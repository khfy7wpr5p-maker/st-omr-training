"""M4-E3K-R1 TRAIN-only root-cause audit for missed measure boundaries.

This module does not tune or change the frozen E3K proposal algorithm. It
replays the exact accepted TRAIN surface and explains *why* each true interior
measure boundary failed the frozen 1.0-staff-space proposal gate.

VALIDATION and TEST are intentionally rejected. The output is diagnostic only
and cannot authorize E3K-B, D11 integration, training, or promotion.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Final

from .degradation import DegradationConfig, sample_degradation_config
from .m4_e3k_boundary_proposals import (
    FROZEN_E3K_CONFIG,
    _bbox,
    _column_evidence,
    _otsu_threshold,
    _staff_lines,
    propose_measure_boundaries,
)
from .m4_e3k_boundary_scoring import (
    _canonical_json,
    _line_intersection_x_at_middle_staff,
    _mapping,
    _positive_int,
    _positive_number,
    _read_image,
    _read_label,
    _sequence,
    _system_objects,
)
from .stage7d7_specialist_training import load_verified_stage7d7_records


STAGE: Final[str] = "M4-E3K-R1-TRAIN-BOUNDARY-MISS-ROOT-CAUSE-AUDIT"
REPORT_SCHEMA: Final[str] = "m4-e3k-r1-train-boundary-miss-audit-v1"
MISS_TOLERANCE_STAFF_SPACES: Final[float] = 1.0
DIAGNOSTIC_WINDOW_STAFF_SPACES: Final[float] = 2.0
MAX_MANIFEST_BYTES: Final[int] = 32 * 1024 * 1024
MAX_WORST_RECORDS: Final[int] = 100


class M4E3KR1AuditError(RuntimeError):
    """Raised when R1 provenance, split, or diagnostic invariants fail closed."""


def _fail(message: str) -> None:
    raise M4E3KR1AuditError(message)


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        _fail(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise M4E3KR1AuditError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def _load_train_degradation_index(corpus_root: str | Path) -> dict[str, dict[str, object]]:
    """Read only TRAIN sample fields needed for degradation stratification.

    For non-TRAIN rows, only ``split`` is inspected before continuing. This
    keeps VALIDATION/TEST payload fields outside the R1 diagnostic surface.
    """

    path = Path(corpus_root) / "manifest.json"
    if path.is_symlink() or not path.is_file():
        _fail("source manifest must be a regular non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= MAX_MANIFEST_BYTES:
        _fail("source manifest byte length is outside R1 bound")
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda value: _fail(f"non-finite manifest constant: {value}"),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise M4E3KR1AuditError("source manifest is not valid ASCII JSON") from exc
    if not isinstance(payload, dict):
        _fail("source manifest must be an object")
    samples = payload.get("samples")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes, bytearray)):
        _fail("source manifest samples must be a sequence")

    result: dict[str, dict[str, object]] = {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            _fail(f"source sample[{index}] must be a mapping")
        split = sample.get("split")
        if split != "train":
            continue
        sample_id = sample.get("sample_id")
        config_payload = sample.get("degradation_config")
        if not isinstance(sample_id, str) or len(sample_id) != 64:
            _fail("TRAIN sample_id must be SHA-256 hex")
        if not isinstance(config_payload, Mapping):
            _fail("TRAIN degradation_config must be a mapping")
        try:
            config = DegradationConfig(**dict(config_payload))
        except (TypeError, ValueError) as exc:
            raise M4E3KR1AuditError("TRAIN degradation_config is invalid") from exc
        profile = _classify_degradation_profile(config)
        if sample_id in result:
            _fail("duplicate TRAIN sample_id in source manifest")
        result[sample_id] = {
            "profile": profile,
            "config": asdict(config),
        }
    if len(result) != 1230:
        _fail("R1 expected exactly 1230 TRAIN degradation records")
    return result


def _classify_degradation_profile(config: DegradationConfig) -> str:
    """Recover clean/light/medium by replaying the frozen Stage-4 sampler."""

    if not isinstance(config, DegradationConfig):
        raise TypeError("config must be DegradationConfig")
    observed = asdict(config)
    matches = []
    for profile in ("clean", "light", "medium"):
        expected = sample_degradation_config(
            config.seed,
            profile,
            raster_width=config.raster_width,
        )
        if asdict(expected) == observed:
            matches.append(profile)
    if len(matches) != 1:
        _fail("TRAIN degradation config does not resolve to one frozen profile")
    return matches[0]


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        _fail("cannot compute percentile of empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _barline_geometry(
    barline: Mapping[str, object],
    *,
    staff_slope: float,
    staff_spacing: float,
) -> tuple[float, float]:
    start = _mapping("barline.start", barline.get("start"))
    end = _mapping("barline.end", barline.get("end"))
    x0 = _finite("barline.start.x", start.get("x"))
    y0 = _finite("barline.start.y", start.get("y"))
    x1 = _finite("barline.end.x", end.get("x"))
    y1 = _finite("barline.end.y", end.get("y"))
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length <= 0:
        _fail("truth barline has zero length")
    length_staff = length / staff_spacing

    # Staff unit vector is (1,m). A perfect barline is perpendicular to it.
    staff_norm = math.hypot(1.0, staff_slope)
    line_norm = length
    dot = abs((dx * 1.0 + dy * staff_slope) / (line_norm * staff_norm))
    dot = min(1.0, max(0.0, dot))
    perpendicularity_error_deg = math.degrees(math.asin(dot))
    return length_staff, perpendicularity_error_deg


def _diagnose_truth_boundary(
    image,
    *,
    truth_x: float,
    staff_bbox: Mapping[str, object],
    five_staff_lines: Sequence[object],
    staff_spacing: float,
    system_bbox: Mapping[str, object],
    proposals,
    threshold: int,
) -> dict[str, object]:
    config = FROZEN_E3K_CONFIG
    spacing = _positive_number("staff_spacing", staff_spacing)
    staff_x0, _, staff_x1, _ = _bbox("staff_bbox", staff_bbox)
    system_x0, _, system_x1, _ = _bbox("system_bbox", system_bbox)
    lines, staff_slope, common_line_left, common_line_right = _staff_lines(
        five_staff_lines,
        config=config,
    )
    search_x0 = max(staff_x0, system_x0, common_line_left)
    search_x1 = min(staff_x1, system_x1, common_line_right)
    x_left = max(0, int(math.floor(search_x0)))
    x_right = min(image.width, int(math.ceil(search_x1)))

    proposal_xs = [float(item.x) for item in proposals]
    nearest_proposal_error = (
        math.inf
        if not proposal_xs
        else min(abs(value - truth_x) for value in proposal_xs) / spacing
    )
    nearest_proposal_x = None
    if proposal_xs:
        nearest_proposal_x = min(proposal_xs, key=lambda value: (abs(value - truth_x), value))

    if not search_x0 <= truth_x < search_x1:
        return {
            "reason": "TRUTH_OUTSIDE_SEARCH_X",
            "nearest_proposal_error_staff_spaces": nearest_proposal_error,
            "nearest_proposal_x": nearest_proposal_x,
            "truth_best_vertical_coverage": None,
            "truth_best_top_endpoint_coverage": None,
            "truth_best_bottom_endpoint_coverage": None,
            "truth_best_score": None,
            "truth_active_column_within_1sp": False,
        }

    window = max(1, int(math.ceil(DIAGNOSTIC_WINDOW_STAFF_SPACES * spacing)))
    start_x = max(x_left, int(math.floor(truth_x)) - window)
    end_x = min(x_right - 1, int(math.ceil(truth_x)) + window)
    evidence_rows: list[tuple[int, float, float, float, float]] = []
    active_within_one = False
    for x in range(start_x, end_x + 1):
        coverage, top_cov, bottom_cov, score = _column_evidence(
            image,
            anchor_x=x,
            x_left=x_left,
            x_right=x_right,
            staff_lines=lines,
            staff_slope=staff_slope,
            staff_spacing=spacing,
            threshold=threshold,
            config=config,
        )
        evidence_rows.append((x, coverage, top_cov, bottom_cov, score))
        is_active = (
            coverage >= config.minimum_vertical_coverage
            and top_cov >= config.minimum_endpoint_coverage
            and bottom_cov >= config.minimum_endpoint_coverage
        )
        if is_active and abs(float(x) - truth_x) / spacing <= MISS_TOLERANCE_STAFF_SPACES:
            active_within_one = True

    if not evidence_rows:
        _fail("truth diagnostic window unexpectedly empty")
    best = min(
        evidence_rows,
        key=lambda row: (-row[4], abs(float(row[0]) - truth_x), row[0]),
    )
    _, best_cov, best_top, best_bottom, best_score = best

    if nearest_proposal_error <= MISS_TOLERANCE_STAFF_SPACES:
        reason = "HIT"
    elif active_within_one:
        reason = "CLUSTER_PEAK_DISPLACEMENT"
    elif best_cov < config.minimum_vertical_coverage:
        reason = "VERTICAL_COVERAGE_FAIL"
    elif best_top < config.minimum_endpoint_coverage and best_bottom < config.minimum_endpoint_coverage:
        reason = "BOTH_ENDPOINTS_FAIL"
    elif best_top < config.minimum_endpoint_coverage:
        reason = "TOP_ENDPOINT_FAIL"
    elif best_bottom < config.minimum_endpoint_coverage:
        reason = "BOTTOM_ENDPOINT_FAIL"
    else:
        # Best score may occur outside 1sp even when all three gates pass there.
        # This means the truth-near region did not itself satisfy the active gate.
        near_rows = [
            row for row in evidence_rows
            if abs(float(row[0]) - truth_x) / spacing <= MISS_TOLERANCE_STAFF_SPACES
        ]
        near_active = any(
            row[1] >= config.minimum_vertical_coverage
            and row[2] >= config.minimum_endpoint_coverage
            and row[3] >= config.minimum_endpoint_coverage
            for row in near_rows
        )
        reason = "CLUSTER_PEAK_DISPLACEMENT" if near_active else "COMBINED_GATE_FAIL"

    return {
        "reason": reason,
        "nearest_proposal_error_staff_spaces": nearest_proposal_error,
        "nearest_proposal_x": nearest_proposal_x,
        "truth_best_vertical_coverage": best_cov,
        "truth_best_top_endpoint_coverage": best_top,
        "truth_best_bottom_endpoint_coverage": best_bottom,
        "truth_best_score": best_score,
        "truth_active_column_within_1sp": active_within_one,
    }


def audit_e3k_r1_train(corpus_root: str | Path, d6_root: str | Path) -> dict[str, object]:
    """Run the frozen R1 diagnostic on TRAIN only."""

    records = load_verified_stage7d7_records(corpus_root, d6_root)
    train_records = tuple(record for record in records if record.split == "train")
    if len(train_records) != 1230:
        _fail("R1 expected exactly 1230 verified TRAIN records")
    if any(record.split != "train" for record in train_records):
        _fail("R1 selected surface crossed TRAIN boundary")
    degradation_index = _load_train_degradation_index(corpus_root)
    if set(degradation_index) != {record.sample_id for record in train_records}:
        _fail("TRAIN degradation index does not match verified D6 record ids")

    reason_counts: Counter[str] = Counter()
    profile_total: Counter[str] = Counter()
    profile_hits: Counter[str] = Counter()
    profile_miss_reasons: dict[str, Counter[str]] = defaultdict(Counter)
    profile_errors: dict[str, list[float]] = defaultdict(list)
    profile_lengths: dict[str, list[float]] = defaultdict(list)
    profile_perp_errors: dict[str, list[float]] = defaultdict(list)
    rotation_bucket_total: Counter[str] = Counter()
    rotation_bucket_hits: Counter[str] = Counter()
    all_errors: list[float] = []
    worst: list[dict[str, object]] = []
    boundary_count = 0
    system_count = 0

    for record in train_records:
        label = _read_label(record)
        image = _read_image(record, label)
        degradation = degradation_index[record.sample_id]
        profile = str(degradation["profile"])
        config_payload = _mapping("degradation.config", degradation["config"])
        rotation_mdeg = int(config_payload.get("rotation_mdeg", 0))
        abs_rotation = abs(rotation_mdeg)
        if abs_rotation == 0:
            rotation_bucket = "0deg"
        elif abs_rotation <= 1000:
            rotation_bucket = "0-1deg"
        else:
            rotation_bucket = "1-2.5deg"

        for bundle in _system_objects(label):
            system_count += 1
            system = _mapping("bundle.system", bundle.get("system"))
            staff = _mapping("bundle.staff", bundle.get("staff"))
            measures = tuple(_sequence("bundle.measures", bundle.get("measures")))
            five_staff_lines = _sequence("staff.five_staff_lines", staff.get("five_staff_lines"))
            staff_spacing = _positive_number("staff.staff_spacing", staff.get("staff_spacing"))
            staff_bbox = _mapping("staff.staff_instance_bbox", staff.get("staff_instance_bbox"))
            system_bbox = _mapping("system.system_bbox", system.get("system_bbox"))
            proposal_result = propose_measure_boundaries(
                image,
                staff_bbox=staff_bbox,
                five_staff_lines=five_staff_lines,
                staff_spacing=staff_spacing,
                system_bbox=system_bbox,
            )

            for measure_obj in measures[:-1]:
                boundary_count += 1
                measure = _mapping("interior measure", measure_obj)
                measure_number = _positive_int("measure.measure_number", measure.get("measure_number"))
                barline = _mapping("measure.barline_segment", measure.get("barline_segment"))
                truth_x = _line_intersection_x_at_middle_staff(barline, five_staff_lines)
                diag = _diagnose_truth_boundary(
                    image,
                    truth_x=truth_x,
                    staff_bbox=staff_bbox,
                    five_staff_lines=five_staff_lines,
                    staff_spacing=staff_spacing,
                    system_bbox=system_bbox,
                    proposals=proposal_result.proposals,
                    threshold=proposal_result.otsu_threshold,
                )
                reason = str(diag["reason"])
                error = float(diag["nearest_proposal_error_staff_spaces"])
                if not math.isfinite(error):
                    _fail("R1 expected at least one proposal per scored system")
                lines, staff_slope, _, _ = _staff_lines(five_staff_lines, config=FROZEN_E3K_CONFIG)
                del lines
                length_staff, perp_error = _barline_geometry(
                    barline,
                    staff_slope=staff_slope,
                    staff_spacing=staff_spacing,
                )

                reason_counts[reason] += 1
                profile_total[profile] += 1
                rotation_bucket_total[rotation_bucket] += 1
                profile_errors[profile].append(error)
                profile_lengths[profile].append(length_staff)
                profile_perp_errors[profile].append(perp_error)
                all_errors.append(error)
                if reason == "HIT":
                    profile_hits[profile] += 1
                    rotation_bucket_hits[rotation_bucket] += 1
                else:
                    profile_miss_reasons[profile][reason] += 1

                worst.append({
                    "sample_id": record.sample_id,
                    "system_id": system.get("system_id"),
                    "measure_number": measure_number,
                    "profile": profile,
                    "rotation_mdeg": rotation_mdeg,
                    "reason": reason,
                    "truth_x": truth_x,
                    "nearest_proposal_x": diag["nearest_proposal_x"],
                    "nearest_proposal_error_staff_spaces": error,
                    "truth_best_vertical_coverage": diag["truth_best_vertical_coverage"],
                    "truth_best_top_endpoint_coverage": diag["truth_best_top_endpoint_coverage"],
                    "truth_best_bottom_endpoint_coverage": diag["truth_best_bottom_endpoint_coverage"],
                    "truth_best_score": diag["truth_best_score"],
                    "barline_length_staff_spaces": length_staff,
                    "barline_perpendicularity_error_deg": perp_error,
                    "proposal_count": len(proposal_result.proposals),
                    "otsu_threshold": proposal_result.otsu_threshold,
                })

    if boundary_count != 7494:
        _fail(f"R1 expected 7494 TRAIN interior boundaries, got {boundary_count}")
    if sum(reason_counts.values()) != boundary_count:
        _fail("R1 reason cardinality mismatch")

    profiles: dict[str, object] = {}
    for profile in ("clean", "light", "medium"):
        total = profile_total[profile]
        if total <= 0:
            _fail(f"R1 profile {profile} is empty")
        profiles[profile] = {
            "boundaries": total,
            "hits_at_1_staff_space": profile_hits[profile],
            "recall_at_1_staff_space": profile_hits[profile] / total,
            "nearest_error_p50_staff_spaces": _percentile(profile_errors[profile], 0.50),
            "nearest_error_p95_staff_spaces": _percentile(profile_errors[profile], 0.95),
            "barline_length_p50_staff_spaces": _percentile(profile_lengths[profile], 0.50),
            "barline_perpendicularity_error_p95_deg": _percentile(profile_perp_errors[profile], 0.95),
            "miss_reason_counts": dict(sorted(profile_miss_reasons[profile].items())),
        }

    rotation_buckets = {
        bucket: {
            "boundaries": rotation_bucket_total[bucket],
            "hits_at_1_staff_space": rotation_bucket_hits[bucket],
            "recall_at_1_staff_space": (
                rotation_bucket_hits[bucket] / rotation_bucket_total[bucket]
                if rotation_bucket_total[bucket]
                else None
            ),
        }
        for bucket in ("0deg", "0-1deg", "1-2.5deg")
    }

    worst.sort(
        key=lambda item: (
            -float(item["nearest_proposal_error_staff_spaces"]),
            str(item["sample_id"]),
            int(item["measure_number"]),
        )
    )
    worst = worst[:MAX_WORST_RECORDS]

    hit_count = reason_counts["HIT"]
    report = {
        "schema_version": REPORT_SCHEMA,
        "stage": STAGE,
        "state": "COMPLETE_DIAGNOSTIC_ONLY",
        "surface": {
            "split": "train",
            "records": len(train_records),
            "systems": system_count,
            "interior_boundaries": boundary_count,
        },
        "frozen_policy": {
            "miss_tolerance_staff_spaces": MISS_TOLERANCE_STAFF_SPACES,
            "diagnostic_window_staff_spaces": DIAGNOSTIC_WINDOW_STAFF_SPACES,
            "proposal_config": asdict(FROZEN_E3K_CONFIG),
            "threshold_tuning": False,
        },
        "summary": {
            "hits_at_1_staff_space": hit_count,
            "misses_at_1_staff_space": boundary_count - hit_count,
            "recall_at_1_staff_space": hit_count / boundary_count,
            "nearest_error_p50_staff_spaces": _percentile(all_errors, 0.50),
            "nearest_error_p95_staff_spaces": _percentile(all_errors, 0.95),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "by_degradation_profile": profiles,
        "by_abs_rotation_bucket": rotation_buckets,
        "worst_records": worst,
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
            "authorizes_e3k_b": False,
            "authorizes_d11_validator": False,
        },
    }
    # Prove the final payload can be persisted canonically without NaN/Infinity.
    _canonical_json(report)
    return report


def persist_e3k_r1_train_report(
    corpus_root: str | Path,
    d6_root: str | Path,
    *,
    report_path: str | Path,
) -> dict[str, object]:
    path = Path(report_path)
    if path.exists() or path.is_symlink():
        _fail("R1 report path must be fresh")
    if not path.parent.is_dir():
        _fail("R1 report parent must already exist")
    report = audit_e3k_r1_train(corpus_root, d6_root)
    raw = _canonical_json(report) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(raw)
    except FileExistsError as exc:
        raise M4E3KR1AuditError("R1 report path already exists") from exc
    return report
