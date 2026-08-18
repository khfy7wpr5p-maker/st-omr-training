"""Frozen M4-E3I development geometry preflight.

This audit is intentionally cheaper than model inference.  It evaluates the
fixed two-candidate E3I geometry against already-scored V4/V5/V6 development
records.  If even an oracle that always selects the better of the two frozen
candidates cannot satisfy the anchor-P95 gate, full D11/specialist scoring is
provably unable to pass the complete E3I gate and must not be run as a promotion
step.

Candidate generation is never tuned here.  VALIDATION labels are used only for
scoring, through the absolute anchor-error values already persisted by earlier
development audits.  TEST and Final-B are outside this module.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from itertools import combinations, product
import json
import math
from pathlib import Path
from statistics import median
from typing import Final, Mapping, Sequence


M4_E3I_GEOMETRY_PREFLIGHT_VERSION: Final[str] = "m4-e3i-v1-frozen-geometry-preflight-v1"
FROZEN_OFFSET_STAFF_SPACES: Final[float] = -0.06619667590040451
MAXIMUM_ANCHOR_P95_STAFF_SPACES: Final[float] = 2.0
EXPECTED_V4_SHA256: Final[str] = "98de47c01c4f83eef1f30c6be143c9cfb19a5c63f8f6f6b5745da6d6808ce826"
EXPECTED_V5_SHA256: Final[str] = "0b6a3229a97150597ffc54b81774a12810237ba0f40ea1f21f5f10745d3dde04"
EXPECTED_V6_SHA256: Final[str] = "7a734bc9d08369da1e0592e22a167777ef587c490e6c3489de6b32d98cf44e06"


class M4E3IGeometryPreflightError(ValueError):
    """Raised when frozen preflight evidence is malformed or inconsistent."""


def _finite(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise M4E3IGeometryPreflightError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise M4E3IGeometryPreflightError(f"{name} must be finite")
    return result


def _key(record: Mapping[str, object]) -> tuple[str, str]:
    sample_id = record.get("sample_id")
    system_id = record.get("system_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise M4E3IGeometryPreflightError("sample_id must be non-empty")
    if not isinstance(system_id, str) or not system_id:
        raise M4E3IGeometryPreflightError("system_id must be non-empty")
    return sample_id, system_id


def _records(payload: Mapping[str, object]) -> dict[tuple[str, str], Mapping[str, object]]:
    raw = payload.get("records")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise M4E3IGeometryPreflightError("records must be a sequence")
    result: dict[tuple[str, str], Mapping[str, object]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise M4E3IGeometryPreflightError("record must be a mapping")
        key = _key(item)
        if key in result:
            raise M4E3IGeometryPreflightError("duplicate sample/system record")
        result[key] = item
    return result


def _reconstruct_target_and_truth_spacing(
    observations: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """Solve |x-target| / truth_spacing = persisted absolute error.

    Earlier frozen V4/V5/V6 audits store selected anchor X and absolute error in
    ground-truth staff spaces.  Two or more distinct observations determine the
    underlying target/spacing up to sign; all sign combinations are tested and
    the solution that reproduces all persisted errors is required to be exact.
    """
    if len(observations) < 2:
        raise M4E3IGeometryPreflightError("at least two anchor observations required")
    best: tuple[float, float, float] | None = None
    for (x1, e1), (x2, e2) in combinations(observations, 2):
        for sign1, sign2 in product((-1.0, 1.0), repeat=2):
            denominator = sign1 * e1 - sign2 * e2
            if abs(denominator) <= 1e-12:
                continue
            truth_spacing = (x1 - x2) / denominator
            if not math.isfinite(truth_spacing) or truth_spacing <= 0:
                continue
            target_x = x1 - sign1 * e1 * truth_spacing
            residual = sum(
                abs(abs(x - target_x) / truth_spacing - error)
                for x, error in observations
            )
            candidate = (residual, truth_spacing, target_x)
            if best is None or candidate < best:
                best = candidate
    if best is None or best[0] > 1e-6:
        raise M4E3IGeometryPreflightError("anchor target reconstruction is inconsistent")
    return best[2], best[1]


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise M4E3IGeometryPreflightError("percentile requires non-empty values")
    if not 0.0 <= probability <= 1.0:
        raise M4E3IGeometryPreflightError("probability outside [0,1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def audit_records(
    v4_payload: Mapping[str, object],
    v5_payload: Mapping[str, object],
    v6_payload: Mapping[str, object],
) -> dict[str, object]:
    v4 = _records(v4_payload)
    v5 = _records(v5_payload)
    v6 = _records(v6_payload)
    if not (set(v4) == set(v5) == set(v6)):
        raise M4E3IGeometryPreflightError("V4/V5/V6 system surfaces differ")

    oracle_errors: list[float] = []
    first_errors: list[float] = []
    second_errors: list[float] = []
    by_class: dict[str, list[float]] = defaultdict(list)
    unsolved = 0
    second_better = 0
    first_better = 0
    equal = 0

    for key in sorted(v6):
        observations: list[tuple[float, float]] = []
        for source in (v4[key], v5[key], v6[key]):
            x = source.get("selected_anchor_x")
            error = source.get("anchor_error_staff_spaces")
            if isinstance(x, (int, float)) and not isinstance(x, bool) and isinstance(error, (int, float)) and not isinstance(error, bool):
                observations.append((_finite("selected_anchor_x", x), _finite("anchor_error_staff_spaces", error)))
        if len(observations) < 2:
            unsolved += 1
            continue

        target_x, truth_spacing = _reconstruct_target_and_truth_spacing(observations)
        current = v6[key]
        selected_x = _finite("v6.selected_anchor_x", current.get("selected_anchor_x"))
        reported_error = _finite("v6.anchor_error_staff_spaces", current.get("anchor_error_staff_spaces"))
        reconstructed_first_error = abs(selected_x - target_x) / truth_spacing
        if not math.isclose(reconstructed_first_error, reported_error, rel_tol=0.0, abs_tol=1e-6):
            raise M4E3IGeometryPreflightError("reconstructed V6 error differs from persisted V6 error")

        robust_left = _finite("v6.robust_staff_x_min", current.get("robust_staff_x_min"))
        component_left = _finite("v6.raw_component_x_min", current.get("raw_component_x_min"))
        model_spacing = (selected_x - robust_left) / FROZEN_OFFSET_STAFF_SPACES
        if not math.isfinite(model_spacing) or model_spacing <= 0:
            raise M4E3IGeometryPreflightError("V6 decoded model spacing is invalid")
        second_x = component_left + FROZEN_OFFSET_STAFF_SPACES * model_spacing
        reconstructed_second_error = abs(second_x - target_x) / truth_spacing
        oracle_error = min(reconstructed_first_error, reconstructed_second_error)

        first_errors.append(reconstructed_first_error)
        second_errors.append(reconstructed_second_error)
        oracle_errors.append(oracle_error)
        truth = current.get("truth")
        if truth not in {"none", "2/4", "3/4", "4/4"}:
            raise M4E3IGeometryPreflightError("truth class outside frozen meter surface")
        by_class[str(truth)].append(oracle_error)
        if reconstructed_second_error + 1e-12 < reconstructed_first_error:
            second_better += 1
        elif reconstructed_first_error + 1e-12 < reconstructed_second_error:
            first_better += 1
        else:
            equal += 1

    oracle_p95 = _percentile(oracle_errors, 0.95)
    class_metrics = {
        label: {
            "count": len(values),
            "oracle_anchor_p50_staff_spaces": _percentile(values, 0.50),
            "oracle_anchor_p95_staff_spaces": _percentile(values, 0.95),
            "oracle_within_2_staff_spaces_rate": sum(value <= 2.0 for value in values) / len(values),
        }
        for label, values in sorted(by_class.items())
    }
    return {
        "version": M4_E3I_GEOMETRY_PREFLIGHT_VERSION,
        "candidate_policy": {
            "candidate_1": "median_five_line_left_plus_frozen_train_offset",
            "candidate_2": "staff_region_component_left_plus_same_frozen_train_offset",
            "maximum_candidates": 2,
            "frozen_offset_staff_spaces": FROZEN_OFFSET_STAFF_SPACES,
            "oracle_selection_used_for_lower_bound": True,
        },
        "surface": {
            "systems_total": len(v6),
            "systems_scored": len(oracle_errors),
            "systems_without_anchor_observations": unsolved,
            "system_geometry_coverage": len(oracle_errors) / len(v6),
        },
        "metrics": {
            "candidate_1_anchor_p50_staff_spaces": _percentile(first_errors, 0.50),
            "candidate_1_anchor_p95_staff_spaces": _percentile(first_errors, 0.95),
            "candidate_2_anchor_p50_staff_spaces": _percentile(second_errors, 0.50),
            "candidate_2_anchor_p95_staff_spaces": _percentile(second_errors, 0.95),
            "oracle_best_of_two_anchor_p50_staff_spaces": _percentile(oracle_errors, 0.50),
            "oracle_best_of_two_anchor_p95_staff_spaces": oracle_p95,
            "oracle_best_of_two_within_2_staff_spaces_rate": sum(value <= 2.0 for value in oracle_errors) / len(oracle_errors),
            "oracle_best_of_two_over_2_staff_spaces_count": sum(value > 2.0 for value in oracle_errors),
            "oracle_best_of_two_over_10_staff_spaces_count": sum(value > 10.0 for value in oracle_errors),
            "candidate_2_better_count": second_better,
            "candidate_1_better_count": first_better,
            "candidate_tie_count": equal,
            "by_truth_class": class_metrics,
        },
        "gate": {
            "maximum_anchor_p95_staff_spaces": MAXIMUM_ANCHOR_P95_STAFF_SPACES,
            "oracle_anchor_p95_pass": oracle_p95 <= MAXIMUM_ANCHOR_P95_STAFF_SPACES,
            "full_model_scoring_required": oracle_p95 <= MAXIMUM_ANCHOR_P95_STAFF_SPACES,
        },
        "safety": {
            "candidate_generation_tuned": False,
            "threshold_tuning": False,
            "model_inference_run": False,
            "optimizer_steps_added": 0,
            "test_opened": False,
            "final_b_opened": False,
            "promotion_allowed": False,
        },
    }


def _load_bound(path: Path, expected_sha256: str) -> Mapping[str, object]:
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise M4E3IGeometryPreflightError(f"SHA-256 mismatch for {path.name}")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise M4E3IGeometryPreflightError(f"{path.name} must contain a JSON object")
    return value


def audit_files(v4_path: str | Path, v5_path: str | Path, v6_path: str | Path) -> dict[str, object]:
    return audit_records(
        _load_bound(Path(v4_path), EXPECTED_V4_SHA256),
        _load_bound(Path(v5_path), EXPECTED_V5_SHA256),
        _load_bound(Path(v6_path), EXPECTED_V6_SHA256),
    )
