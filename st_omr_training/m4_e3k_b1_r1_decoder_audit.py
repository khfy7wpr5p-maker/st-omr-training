"""M4-E3K-B1-R1 TRAIN-only frozen D7 StaffSet decoder root-cause audit.

This module is diagnostic only. It does not change the frozen B1 decoder,
thresholds, model weights, proposal logic, or any split policy. For every TRAIN
page it reuses the exact accepted B1 inference path, records where each
``staff_region`` component survives or fails, and cross-checks the diagnostic
trace against the unchanged ``decode_d7_staff_geometry`` result.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Final

import torch

from .m4_e3k_b1_d7_staff_geometry import (
    D7_DENSE_THRESHOLD,
    EXPECTED_D7_CHECKPOINT_SHA256,
    EXPECTED_D7_STAFF_STATE_SHA256,
    MINIMUM_STAFF_COMPONENT_AREA_FRACTION,
    MINIMUM_STAFF_COMPONENT_WIDTH_FRACTION,
    M4E3KB1GeometryError,
    _binary_components,
    _equal_spaced_line_template,
    _fit_common_staff_slope,
    _line_x_support,
    decode_d7_staff_geometry,
    load_frozen_d7_staff_model,
)
from .m4_e3k_boundary_scoring import (
    _canonical_json,
    _percentile,
    _read_label,
    _system_objects,
)
from .stage7d7_specialist_training import (
    FROZEN_D7_CONFIG,
    STAFF_CHANNELS,
    Stage7D7TrainingError,
    _load_input_image,
    load_verified_stage7d7_records,
)


STAGE: Final[str] = "M4-E3K-B1-R1-TRAIN-D7-STAFF-DECODER-ROOT-CAUSE-AUDIT"
REPORT_SCHEMA: Final[str] = "m4-e3k-b1-r1-d7-staff-decoder-audit-report-v1"
EXPECTED_TRAIN_RECORDS: Final[int] = 1230
EXPECTED_TRAIN_SYSTEMS: Final[int] = 2346
MAX_REPORT_BYTES: Final[int] = 16 * 1024 * 1024


class M4E3KB1R1AuditError(RuntimeError):
    """Raised when B1-R1 provenance, inference, or diagnostic parity fails closed."""


def _fail(message: str) -> None:
    raise M4E3KB1R1AuditError(message)


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        _fail(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise M4E3KB1R1AuditError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def _percentile_or_zero(values: Sequence[float | int], fraction: float) -> float:
    if not values:
        return 0.0
    return _percentile(values, fraction)


def _page_terminal_reason(
    *,
    truth_systems: int,
    region_active_pixels: int,
    raw_components: int,
    qualifying_components: int,
    slope_pass: int,
    template_pass: int,
    x_support_pass: int,
    decoded: int,
) -> str:
    if region_active_pixels == 0:
        return "NO_REGION_SUPPORT_AT_FROZEN_THRESHOLD"
    if raw_components == 0:
        return "NO_RAW_REGION_COMPONENT"
    if qualifying_components == 0:
        return "ALL_REGION_COMPONENTS_REJECTED_BY_FROZEN_SIZE_GATES"
    if slope_pass == 0:
        return "ALL_QUALIFYING_COMPONENTS_FAILED_SLOPE"
    if template_pass == 0:
        return "ALL_SLOPE_PASS_COMPONENTS_FAILED_FIVE_LINE_TEMPLATE"
    if x_support_pass == 0:
        return "ALL_TEMPLATE_PASS_COMPONENTS_FAILED_X_SUPPORT"
    if decoded < truth_systems:
        return "DECODED_STAFF_COUNT_BELOW_TRUTH_SYSTEM_COUNT"
    if decoded > truth_systems:
        return "DECODED_STAFF_COUNT_ABOVE_TRUTH_SYSTEM_COUNT"
    return "DECODED_STAFF_COUNT_EQUALS_TRUTH_SYSTEM_COUNT"


def audit_d7_staff_probabilities(
    probabilities: torch.Tensor,
    *,
    original_width: int,
    original_height: int,
    truth_system_count: int,
) -> dict[str, object]:
    """Trace the unchanged B1 decoder on one frozen D7 probability tensor."""

    expected_shape = (
        len(STAFF_CHANNELS),
        FROZEN_D7_CONFIG.input_height,
        FROZEN_D7_CONFIG.input_width,
    )
    if not isinstance(probabilities, torch.Tensor) or probabilities.dtype != torch.float32:
        raise TypeError("probabilities must be a float32 tensor")
    if probabilities.ndim != 3 or tuple(probabilities.shape) != expected_shape:
        _fail("B1-R1 probability tensor shape mismatch")
    if probabilities.device.type != "cpu":
        _fail("B1-R1 probabilities must be on CPU")
    if not bool(torch.isfinite(probabilities).all()):
        _fail("B1-R1 probabilities contain non-finite values")
    if bool((probabilities < 0).any()) or bool((probabilities > 1).any()):
        _fail("B1-R1 probabilities must lie in [0,1]")
    if not isinstance(original_width, int) or isinstance(original_width, bool) or original_width <= 0:
        _fail("original_width must be a positive integer")
    if not isinstance(original_height, int) or isinstance(original_height, bool) or original_height <= 0:
        _fail("original_height must be a positive integer")
    if not isinstance(truth_system_count, int) or isinstance(truth_system_count, bool) or truth_system_count <= 0:
        _fail("truth_system_count must be a positive integer")

    line_index = STAFF_CHANNELS.index("staff_lines")
    region_index = STAFF_CHANNELS.index("staff_region")
    line_probs = probabilities[line_index]
    region_probs = probabilities[region_index]

    line_mask = line_probs >= D7_DENSE_THRESHOLD
    region_mask = region_probs >= D7_DENSE_THRESHOLD
    line_active_pixels = int(line_mask.sum().item())
    region_active_pixels = int(region_mask.sum().item())
    total_pixels = int(region_mask.numel())

    raw_components = _binary_components(region_mask)
    input_h = FROZEN_D7_CONFIG.input_height
    input_w = FROZEN_D7_CONFIG.input_width
    minimum_width = input_w * MINIMUM_STAFF_COMPONENT_WIDTH_FRACTION
    minimum_area = input_h * input_w * MINIMUM_STAFF_COMPONENT_AREA_FRACTION

    qualifying = []
    component_failure_counts: Counter[str] = Counter()
    for component in raw_components:
        width_pass = component.width >= minimum_width
        area_pass = component.area >= minimum_area
        if not width_pass and not area_pass:
            component_failure_counts["SIZE_WIDTH_AND_AREA_FAIL"] += 1
            continue
        if not width_pass:
            component_failure_counts["SIZE_WIDTH_FAIL"] += 1
            continue
        if not area_pass:
            component_failure_counts["SIZE_AREA_FAIL"] += 1
            continue
        qualifying.append(component)

    slope_pass = 0
    slope_fail = 0
    template_pass = 0
    template_fail = 0
    x_support_pass = 0
    x_support_fail = 0
    decoded_trace = 0
    component_trace: list[dict[str, object]] = []

    for component in qualifying:
        row: dict[str, object] = {
            "bbox": [component.x0, component.y0, component.x1, component.y1],
            "width": component.width,
            "height": component.height,
            "area": component.area,
        }
        try:
            slope = _fit_common_staff_slope(line_probs, component)
        except M4E3KB1GeometryError as exc:
            slope_fail += 1
            component_failure_counts["SLOPE_FAIL"] += 1
            row["terminal"] = "SLOPE_FAIL"
            row["message"] = str(exc)
            component_trace.append(row)
            continue

        slope_pass += 1
        row["slope"] = slope
        try:
            first_y, spacing, template_score = _equal_spaced_line_template(
                line_probs, component, slope
            )
        except M4E3KB1GeometryError as exc:
            template_fail += 1
            component_failure_counts["FIVE_LINE_TEMPLATE_FAIL"] += 1
            row["terminal"] = "FIVE_LINE_TEMPLATE_FAIL"
            row["message"] = str(exc)
            component_trace.append(row)
            continue

        template_pass += 1
        row["first_line_y"] = first_y
        row["spacing"] = spacing
        row["template_score"] = template_score
        reference_x = (component.x0 + component.x1) / 2.0
        line_supports: list[list[int]] = []
        try:
            for index in range(5):
                line_y_ref = first_y + index * spacing
                x0, x1 = _line_x_support(
                    line_probs,
                    component,
                    line_y_at_reference=line_y_ref,
                    slope=slope,
                )
                line_supports.append([x0, x1])
                # Exercise the same finite page-space mapping used by B1.
                scale_x = original_width / float(input_w)
                scale_y = original_height / float(input_h)
                y0 = line_y_ref + slope * (float(x0) - reference_x)
                y1 = line_y_ref + slope * (float(x1) - reference_x)
                for value in (x0 * scale_x, x1 * scale_x, y0 * scale_y, y1 * scale_y):
                    _finite("mapped staff-line coordinate", value)
        except M4E3KB1GeometryError as exc:
            x_support_fail += 1
            component_failure_counts["X_SUPPORT_FAIL"] += 1
            row["terminal"] = "X_SUPPORT_FAIL"
            row["message"] = str(exc)
            component_trace.append(row)
            continue

        x_support_pass += 1
        decoded_trace += 1
        row["line_supports"] = line_supports
        row["terminal"] = "DECODED"
        component_trace.append(row)

    # Critical diagnostic parity check: the trace is not allowed to invent a
    # different decoder result than the unchanged B1 implementation.
    unchanged_decoded = decode_d7_staff_geometry(
        probabilities,
        original_width=original_width,
        original_height=original_height,
    )
    if decoded_trace != len(unchanged_decoded):
        _fail("B1-R1 trace diverged from unchanged B1 decoder output")

    terminal = _page_terminal_reason(
        truth_systems=truth_system_count,
        region_active_pixels=region_active_pixels,
        raw_components=len(raw_components),
        qualifying_components=len(qualifying),
        slope_pass=slope_pass,
        template_pass=template_pass,
        x_support_pass=x_support_pass,
        decoded=decoded_trace,
    )

    return {
        "truth_system_count": truth_system_count,
        "frozen_threshold": D7_DENSE_THRESHOLD,
        "region_probability_max": float(region_probs.max().item()),
        "region_probability_mean": float(region_probs.mean().item()),
        "line_probability_max": float(line_probs.max().item()),
        "line_probability_mean": float(line_probs.mean().item()),
        "region_active_pixels": region_active_pixels,
        "region_active_fraction": region_active_pixels / total_pixels,
        "line_active_pixels": line_active_pixels,
        "line_active_fraction": line_active_pixels / total_pixels,
        "raw_region_components": len(raw_components),
        "qualifying_region_components": len(qualifying),
        "slope_pass_components": slope_pass,
        "slope_fail_components": slope_fail,
        "five_line_template_pass_components": template_pass,
        "five_line_template_fail_components": template_fail,
        "x_support_pass_components": x_support_pass,
        "x_support_fail_components": x_support_fail,
        "decoded_staff_count": decoded_trace,
        "decoder_parity_pass": True,
        "terminal_reason": terminal,
        "component_failure_counts": dict(sorted(component_failure_counts.items())),
        "components": component_trace,
    }


def _profile_payload() -> dict[str, object]:
    return {
        "stage": STAGE,
        "surface": "TRAIN_only",
        "purpose": "diagnose_where_frozen_B1_D7_staff_decoder_loses_staff_instances",
        "d7_checkpoint_sha256": EXPECTED_D7_CHECKPOINT_SHA256,
        "d7_staff_state_sha256": EXPECTED_D7_STAFF_STATE_SHA256,
        "d7_input": [FROZEN_D7_CONFIG.input_height, FROZEN_D7_CONFIG.input_width],
        "d7_dense_threshold": D7_DENSE_THRESHOLD,
        "staff_component_width_fraction": MINIMUM_STAFF_COMPONENT_WIDTH_FRACTION,
        "staff_component_area_fraction": MINIMUM_STAFF_COMPONENT_AREA_FRACTION,
        "threshold_sweep": False,
        "decoder_behavior_changed": False,
    }


def profile_fingerprint() -> str:
    return sha256(_canonical_json(_profile_payload())).hexdigest()


def run_b1_r1_train_audit(
    corpus_root: str | Path,
    d6_root: str | Path,
    d7_checkpoint_path: str | Path,
) -> dict[str, object]:
    """Run frozen D7 inference and decoder tracing on all accepted TRAIN pages."""

    records = load_verified_stage7d7_records(corpus_root, d6_root)
    train_records = tuple(record for record in records if record.split == "train")
    if len(train_records) != EXPECTED_TRAIN_RECORDS:
        _fail("B1-R1 expected exactly 1230 TRAIN records")
    if any(record.split != "train" for record in train_records):
        _fail("B1-R1 selected surface crossed TRAIN boundary")

    model = load_frozen_d7_staff_model(d7_checkpoint_path)

    page_rows: list[dict[str, object]] = []
    terminal_counts: Counter[str] = Counter()
    component_failure_counts: Counter[str] = Counter()
    truth_system_total = 0

    region_peaks: list[float] = []
    region_means: list[float] = []
    line_peaks: list[float] = []
    line_means: list[float] = []
    region_active_fractions: list[float] = []
    line_active_fractions: list[float] = []
    raw_component_counts: list[int] = []
    qualifying_component_counts: list[int] = []
    decoded_counts: list[int] = []

    aggregate_component_counts: Counter[str] = Counter()

    for record in train_records:
        label = _read_label(record)
        bundles = tuple(_system_objects(label))
        truth_system_count = len(bundles)
        truth_system_total += truth_system_count
        image_meta = label.get("image")
        if not isinstance(image_meta, Mapping):
            _fail("D6 image metadata is missing")
        width = image_meta.get("width")
        height = image_meta.get("height")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            _fail("D6 image width must be positive integer")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            _fail("D6 image height must be positive integer")
        try:
            input_tensor = _load_input_image(record, label, FROZEN_D7_CONFIG)
        except Stage7D7TrainingError as exc:
            raise M4E3KB1R1AuditError(
                f"D7 TRAIN image cannot be loaded for B1-R1: {record.sample_id}"
            ) from exc
        with torch.no_grad():
            logits = model(input_tensor.unsqueeze(0))
            probabilities = torch.sigmoid(logits).squeeze(0).to(dtype=torch.float32, device="cpu")

        page = audit_d7_staff_probabilities(
            probabilities,
            original_width=width,
            original_height=height,
            truth_system_count=truth_system_count,
        )
        row = {"sample_id": record.sample_id, **page}
        page_rows.append(row)
        terminal_counts[str(page["terminal_reason"])] += 1
        for reason, count in dict(page["component_failure_counts"]).items():
            component_failure_counts[str(reason)] += int(count)

        truth_systems = int(page["truth_system_count"])
        decoded = int(page["decoded_staff_count"])
        if decoded < truth_systems:
            aggregate_component_counts["PAGES_DECODED_LT_TRUTH"] += 1
        elif decoded == truth_systems:
            aggregate_component_counts["PAGES_DECODED_EQ_TRUTH"] += 1
        else:
            aggregate_component_counts["PAGES_DECODED_GT_TRUTH"] += 1
        if decoded == 0:
            aggregate_component_counts["PAGES_DECODED_ZERO"] += 1
        if float(page["region_probability_max"]) < D7_DENSE_THRESHOLD:
            aggregate_component_counts["PAGES_REGION_PEAK_BELOW_THRESHOLD"] += 1
        if int(page["region_active_pixels"]) == 0:
            aggregate_component_counts["PAGES_REGION_NO_ACTIVE_PIXELS"] += 1

        region_peaks.append(float(page["region_probability_max"]))
        region_means.append(float(page["region_probability_mean"]))
        line_peaks.append(float(page["line_probability_max"]))
        line_means.append(float(page["line_probability_mean"]))
        region_active_fractions.append(float(page["region_active_fraction"]))
        line_active_fractions.append(float(page["line_active_fraction"]))
        raw_component_counts.append(int(page["raw_region_components"]))
        qualifying_component_counts.append(int(page["qualifying_region_components"]))
        decoded_counts.append(decoded)

    if truth_system_total != EXPECTED_TRAIN_SYSTEMS:
        _fail(
            f"B1-R1 expected {EXPECTED_TRAIN_SYSTEMS} TRAIN systems, got {truth_system_total}"
        )

    return {
        "schema_version": REPORT_SCHEMA,
        "stage": STAGE,
        "state": "COMPLETE_DIAGNOSTIC_ONLY",
        "profile_fingerprint": profile_fingerprint(),
        "profile": _profile_payload(),
        "surface": {
            "split": "train",
            "records": len(train_records),
            "truth_systems": truth_system_total,
        },
        "aggregate": {
            "page_counts": dict(sorted(aggregate_component_counts.items())),
            "terminal_page_reason_counts": dict(sorted(terminal_counts.items())),
            "component_failure_counts": dict(sorted(component_failure_counts.items())),
            "region_probability_max_p50": _percentile_or_zero(region_peaks, 0.50),
            "region_probability_max_p95": _percentile_or_zero(region_peaks, 0.95),
            "region_probability_mean_p50": _percentile_or_zero(region_means, 0.50),
            "line_probability_max_p50": _percentile_or_zero(line_peaks, 0.50),
            "line_probability_max_p95": _percentile_or_zero(line_peaks, 0.95),
            "line_probability_mean_p50": _percentile_or_zero(line_means, 0.50),
            "region_active_fraction_p50": _percentile_or_zero(region_active_fractions, 0.50),
            "region_active_fraction_p95": _percentile_or_zero(region_active_fractions, 0.95),
            "line_active_fraction_p50": _percentile_or_zero(line_active_fractions, 0.50),
            "line_active_fraction_p95": _percentile_or_zero(line_active_fractions, 0.95),
            "raw_region_components_per_page_p50": _percentile_or_zero(raw_component_counts, 0.50),
            "raw_region_components_per_page_p95": _percentile_or_zero(raw_component_counts, 0.95),
            "qualifying_region_components_per_page_p50": _percentile_or_zero(
                qualifying_component_counts, 0.50
            ),
            "qualifying_region_components_per_page_p95": _percentile_or_zero(
                qualifying_component_counts, 0.95
            ),
            "decoded_staff_count_per_page_p50": _percentile_or_zero(decoded_counts, 0.50),
            "decoded_staff_count_per_page_p95": _percentile_or_zero(decoded_counts, 0.95),
            "decoded_staff_total": sum(decoded_counts),
        },
        "page_records": page_rows,
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
            "threshold_sweep": False,
            "decoder_behavior_changed": False,
            "r2_proposal_executed": False,
            "production_promotion": False,
            "b2_authorized": False,
            "d11_authorized": False,
        },
    }


def persist_b1_r1_train_audit(
    corpus_root: str | Path,
    d6_root: str | Path,
    d7_checkpoint_path: str | Path,
    *,
    report_path: str | Path,
) -> dict[str, object]:
    """Persist one fresh canonical TRAIN-only decoder audit report."""

    path = Path(report_path)
    if path.exists() or path.is_symlink():
        _fail("B1-R1 report path must be fresh")
    if not path.parent.is_dir() or path.parent.is_symlink():
        _fail("B1-R1 report parent must be an existing regular directory")
    report = run_b1_r1_train_audit(corpus_root, d6_root, d7_checkpoint_path)
    raw = _canonical_json(report)
    if not 1 <= len(raw) <= MAX_REPORT_BYTES:
        _fail("B1-R1 report byte length is outside diagnostic bound")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        _fail("B1-R1 temporary report path must be fresh")
    try:
        temporary.write_bytes(raw)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if path.read_bytes() != raw:
        _fail("persisted B1-R1 report bytes failed verification")
    return report
