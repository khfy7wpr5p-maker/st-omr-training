"""TRAIN-only topology audit for future deterministic System Geometry v1.

This module is diagnostic only. It reads the frozen Stage 7-D6 derivative
manifest and opens only TRAIN labels. VALIDATION labels are counted by split but
never opened; TEST has no D6 surface and is not accessed. No runtime geometry,
Meter model, optimizer, threshold tuning, Resolver wiring, or production state
is mutated here.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
from typing import Callable, Final

EXPECTED_D6_MANIFEST_SHA256: Final[str] = "e8e415eb6ba9d91a1a880709c3f31d559aa20bf5149734f45b5f84ced16afee9"
EXPECTED_TRAIN_RECORDS: Final[int] = 1230
EXPECTED_VALIDATION_RECORDS: Final[int] = 153
EXPECTED_TOTAL_RECORDS: Final[int] = 1383
VERSION: Final[str] = "system-geometry-train-topology-audit-v1"
READ_WORKERS: Final[int] = 8
ProgressCallback = Callable[[int, int], None]


class SystemGeometryTopologyAuditError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise SystemGeometryTopologyAuditError(message)


def _load_json(path: Path) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        _fail(f"missing regular file: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("ascii"), parse_constant=lambda x: _fail(f"nonfinite JSON: {x}"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SystemGeometryTopologyAuditError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        _fail(f"JSON root must be object: {path}")
    return payload, raw


def _box(value: object, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    try:
        vals = tuple(float(value[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemGeometryTopologyAuditError(f"{name} malformed") from exc
    if not all(math.isfinite(v) for v in vals):
        _fail(f"{name} nonfinite")
    x0, y0, x1, y1 = vals
    if not x0 < x1 or not y0 < y1:
        _fail(f"{name} must have positive area")
    return vals


def _q(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p05": _q(values, 0.05),
        "median": float(median(values)) if values else None,
        "p95": _q(values, 0.95),
        "max": max(values) if values else None,
    }


def _horizontal_overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    denom = min(a[2] - a[0], b[2] - b[0])
    return overlap / denom if denom > 0 else 0.0


def _open_train_label(task: tuple[int, dict[str, object], Path]) -> tuple[int, dict[str, object], dict[str, object]]:
    index, record, root = task
    label_sha = record.get("label_sha256")
    if not isinstance(label_sha, str) or len(label_sha) != 64:
        _fail(f"record[{index}] label SHA malformed")
    label_path = root / "labels" / f"{label_sha}.json"
    label, raw = _load_json(label_path)
    if sha256(raw).hexdigest() != label_sha:
        _fail(f"record[{index}] label SHA mismatch")
    if label.get("split") != "train":
        _fail("TRAIN audit encountered non-TRAIN label")
    if label.get("sample_id") != record.get("sample_id"):
        _fail("record/label sample identity mismatch")
    return index, record, label


def audit_d6_train_topology(
    d6_root: str | Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    root = Path(d6_root)
    manifest, manifest_raw = _load_json(root / "manifest.json")
    if sha256(manifest_raw).hexdigest() != EXPECTED_D6_MANIFEST_SHA256:
        _fail("D6 manifest SHA-256 mismatch")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_TOTAL_RECORDS:
        _fail("D6 record inventory mismatch")

    split_counts: Counter[str] = Counter()
    train_tasks: list[tuple[int, dict[str, object], Path]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            _fail(f"record[{index}] malformed")
        split = record.get("split")
        if split not in {"train", "validation"}:
            _fail(f"record[{index}] unexpected split")
        split_counts[str(split)] += 1
        if split == "train":
            train_tasks.append((index, record, root))

    if split_counts != Counter({"train": EXPECTED_TRAIN_RECORDS, "validation": EXPECTED_VALIDATION_RECORDS}):
        _fail("D6 split counts mismatch")
    if len(train_tasks) != EXPECTED_TRAIN_RECORDS:
        _fail("TRAIN record inventory mismatch")

    system_counts_per_page: Counter[int] = Counter()
    staff_counts_per_page: Counter[int] = Counter()
    staffs_per_system: Counter[int] = Counter()
    measures_per_system: Counter[int] = Counter()
    adjacent_system_gap_spacing_units: list[float] = []
    adjacent_system_horizontal_overlap: list[float] = []
    intra_system_staff_gap_spacing_units: list[float] = []
    train_pages = 0
    systems_total = 0
    staffs_total = 0
    multi_system_pages = 0
    multi_staff_systems = 0
    labels_opened = 0

    # Mounted Google Drive is especially slow for many tiny sequential opens.
    # Bounded parallel read-only loading changes no audit semantics; executor.map
    # preserves manifest order and every label is still SHA-verified before use.
    with ThreadPoolExecutor(max_workers=READ_WORKERS) as executor:
        loaded = executor.map(_open_train_label, train_tasks)
        for completed, (_index, record, label) in enumerate(loaded, start=1):
            labels_opened += 1
            if progress is not None and (completed == 1 or completed % 100 == 0 or completed == EXPECTED_TRAIN_RECORDS):
                progress(completed, EXPECTED_TRAIN_RECORDS)

            geometry = label.get("geometry")
            if not isinstance(geometry, dict):
                _fail("geometry missing")
            systems = geometry.get("systems")
            staffs = geometry.get("staff_instances")
            measures = geometry.get("measures")
            if not isinstance(systems, list) or not isinstance(staffs, list) or not isinstance(measures, list):
                _fail("geometry topology arrays missing")
            if not systems or not staffs:
                _fail("TRAIN page must expose system/staff geometry")

            train_pages += 1
            system_counts_per_page[len(systems)] += 1
            staff_counts_per_page[len(staffs)] += 1
            systems_total += len(systems)
            staffs_total += len(staffs)
            if len(systems) > 1:
                multi_system_pages += 1

            staff_by_system: dict[str, list[dict[str, object]]] = {}
            spacing_by_system: dict[str, list[float]] = {}
            for staff in staffs:
                if not isinstance(staff, dict):
                    _fail("staff entry malformed")
                sid = staff.get("system_id")
                if not isinstance(sid, str) or not sid:
                    _fail("staff system_id malformed")
                spacing = float(staff.get("staff_spacing", 0.0))
                if not math.isfinite(spacing) or spacing <= 0:
                    _fail("staff spacing malformed")
                _box(staff.get("staff_instance_bbox"), "staff_instance_bbox")
                staff_by_system.setdefault(sid, []).append(staff)
                spacing_by_system.setdefault(sid, []).append(spacing)

            measure_by_system: Counter[str] = Counter()
            for measure_row in measures:
                if not isinstance(measure_row, dict):
                    _fail("measure entry malformed")
                sid = measure_row.get("system_id")
                if not isinstance(sid, str) or not sid:
                    _fail("measure system_id malformed")
                measure_by_system[sid] += 1

            ordered_systems: list[tuple[str, tuple[float, float, float, float], float]] = []
            seen_system_ids: set[str] = set()
            for system in systems:
                if not isinstance(system, dict):
                    _fail("system entry malformed")
                sid = system.get("system_id")
                if not isinstance(sid, str) or not sid or sid in seen_system_ids:
                    _fail("system_id malformed or duplicate")
                seen_system_ids.add(sid)
                bbox = _box(system.get("system_bbox"), "system_bbox")
                members = staff_by_system.get(sid, [])
                if not members:
                    _fail("system has no staff")
                nstaff = len(members)
                staffs_per_system[nstaff] += 1
                if nstaff > 1:
                    multi_staff_systems += 1
                nmeas = int(measure_by_system.get(sid, 0))
                if nmeas <= 0:
                    _fail("system has no measures")
                measures_per_system[nmeas] += 1
                spacing = float(median(spacing_by_system[sid]))
                ordered_systems.append((sid, bbox, spacing))

                if nstaff > 1:
                    member_boxes = sorted(
                        ((_box(s.get("staff_instance_bbox"), "staff_instance_bbox"), float(s["staff_spacing"])) for s in members),
                        key=lambda item: item[0][1],
                    )
                    for (a, sa), (b, sb) in zip(member_boxes, member_boxes[1:]):
                        gap = b[1] - a[3]
                        denom = (sa + sb) / 2.0
                        intra_system_staff_gap_spacing_units.append(gap / denom)

            if set(staff_by_system) != seen_system_ids or set(measure_by_system) != seen_system_ids:
                _fail("system/staff/measure topology identity mismatch")

            ordered_systems.sort(key=lambda item: item[1][1])
            for (_, a, sa), (_, b, sb) in zip(ordered_systems, ordered_systems[1:]):
                denom = (sa + sb) / 2.0
                adjacent_system_gap_spacing_units.append((b[1] - a[3]) / denom)
                adjacent_system_horizontal_overlap.append(_horizontal_overlap_ratio(a, b))

    if labels_opened != EXPECTED_TRAIN_RECORDS or train_pages != EXPECTED_TRAIN_RECORDS:
        _fail("TRAIN-only label access count mismatch")

    general_grouping_identifiable = multi_staff_systems > 0
    decision = (
        "TRAIN_SUPPORTS_MULTI_STAFF_SYSTEM_GROUPING_ANALYSIS"
        if general_grouping_identifiable
        else "HOLD_GENERAL_GROUPER_NOT_IDENTIFIABLE_FROM_TRAIN"
    )

    return {
        "version": VERSION,
        "d6_manifest_sha256": EXPECTED_D6_MANIFEST_SHA256,
        "decision": decision,
        "train_pages": train_pages,
        "validation_records_counted_only": EXPECTED_VALIDATION_RECORDS,
        "validation_labels_opened": 0,
        "test_accessed": False,
        "training_performed": False,
        "threshold_tuning": False,
        "read_workers": READ_WORKERS,
        "systems_total": systems_total,
        "staffs_total": staffs_total,
        "multi_system_pages": multi_system_pages,
        "multi_staff_systems": multi_staff_systems,
        "system_counts_per_page": dict(sorted(system_counts_per_page.items())),
        "staff_counts_per_page": dict(sorted(staff_counts_per_page.items())),
        "staffs_per_system": dict(sorted(staffs_per_system.items())),
        "measures_per_system": dict(sorted(measures_per_system.items())),
        "adjacent_system_gap_staff_spacing_units": _summary(adjacent_system_gap_spacing_units),
        "adjacent_system_horizontal_overlap_ratio": _summary(adjacent_system_horizontal_overlap),
        "intra_system_staff_gap_staff_spacing_units": _summary(intra_system_staff_gap_spacing_units),
        "general_multi_staff_grouping_identifiable": general_grouping_identifiable,
    }
