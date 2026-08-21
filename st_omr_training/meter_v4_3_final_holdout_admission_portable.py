"""Mount-portable candidate discovery for Meter V4-3 final holdout admission.

Google Drive permits '/' in folder display names, while a POSIX mount cannot expose
'/' as a literal path component. This module therefore discovers sample folders by
the frozen sample-folder grammar instead of assuming class container path spelling.
It remains admission-only: no checkpoint is opened and no inference is run.

The candidate pool may contain a bounded surplus above the originally planned
65 samples per class. Surplus candidates are useful because previously observed
or duplicate families are rejected before the deterministic 50/50/50 selection.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Final

from st_omr_training.meter_v4_3_final_holdout_admission import (
    CandidateV4_3,
    MeterV4_3AdmissionError,
    RESULT_SCHEMA_V4_1,
    RESULT_SCHEMA_V4_2,
    _FOLDER_RE,
    _canonical_json,
    _parse_bbox_header,
    _read_json,
    observed_families,
    select_final_holdout,
)

MIN_PER_CLASS: Final[int] = 65
MAX_PER_CLASS: Final[int] = 80
METER_BY_NUMERATOR: Final[dict[str, str]] = {"2": "2/4", "3": "3/4", "4": "4/4"}


def _fail(message: str) -> None:
    raise MeterV4_3AdmissionError(message)


def scan_candidate_pool_portable(candidate_root: str | Path) -> tuple[CandidateV4_3, ...]:
    root = Path(candidate_root)
    if not root.is_dir() or root.is_symlink():
        _fail("candidate root must be an existing regular directory")

    candidates: list[CandidateV4_3] = []
    for folder in sorted(root.rglob("*"), key=lambda p: str(p)):
        if not folder.is_dir() or folder.is_symlink():
            continue
        try:
            depth = len(folder.relative_to(root).parts)
        except ValueError:
            _fail("candidate path escaped candidate root")
        if depth > 3:
            continue
        match = _FOLDER_RE.match(folder.name)
        if not match:
            continue
        numerator_class = match.group("num")
        family_id = match.group("family")
        meter_class = METER_BY_NUMERATOR[numerator_class]
        image_path = folder / "image.png"
        bbox_path = folder / "bbox_meter.txt"
        if not image_path.is_file() or image_path.is_symlink() or image_path.stat().st_size <= 0:
            _fail(f"image.png missing or empty: {folder}")
        _parse_bbox_header(bbox_path, expected_meter=meter_class)
        candidates.append(
            CandidateV4_3(
                numerator_class=numerator_class,
                meter_class=meter_class,
                folder_name=folder.name,
                family_id=family_id,
                image_path=str(image_path),
                bbox_path=str(bbox_path),
            )
        )

    counts = Counter(row.numerator_class for row in candidates)
    expected_classes = {"2", "3", "4"}
    if set(counts) != expected_classes:
        _fail(f"candidate classes must be exactly 2/3/4, got {dict(sorted(counts.items()))}")
    for class_name in ("2", "3", "4"):
        count = counts[class_name]
        if count < MIN_PER_CLASS or count > MAX_PER_CLASS:
            _fail(
                f"class {class_name} candidate count must be within "
                f"{MIN_PER_CLASS}..{MAX_PER_CLASS}, got {count}"
            )
    total = len(candidates)
    if total < MIN_PER_CLASS * 3 or total > MAX_PER_CLASS * 3:
        _fail(
            f"V4-3 candidate pool must be within "
            f"{MIN_PER_CLASS * 3}..{MAX_PER_CLASS * 3} sample folders, got {total}"
        )
    return tuple(candidates)


def build_manifest_portable(
    *,
    candidate_root: str | Path,
    v4_1_result_path: str | Path,
    v4_2_result_path: str | Path,
) -> dict[str, object]:
    v4_1 = _read_json(Path(v4_1_result_path), schema=RESULT_SCHEMA_V4_1, name="V4-1 result")
    v4_2 = _read_json(Path(v4_2_result_path), schema=RESULT_SCHEMA_V4_2, name="V4-2 result")
    observed = observed_families(v4_1, v4_2)
    candidates = scan_candidate_pool_portable(candidate_root)
    candidate_counts = Counter(row.numerator_class for row in candidates)
    selected, excluded = select_final_holdout(candidates, observed=observed)
    manifest: dict[str, object] = {
        "schema": "st-omr-meter-v4-3-final-holdout-admission-manifest-v1",
        "experiment": "meter-v4-3-final-holdout-admission-v1",
        "candidate_root_name": Path(candidate_root).name,
        "candidate_count": len(candidates),
        "candidate_classes": {key: candidate_counts[key] for key in ("2", "3", "4")},
        "candidate_pool_bounds_per_class": {"min": MIN_PER_CLASS, "max": MAX_PER_CLASS},
        "previously_observed_family_count": len(observed),
        "selected_count": 150,
        "selected_classes": {"2": 50, "3": 50, "4": 50},
        "selected": [asdict(row) for row in selected],
        "excluded": [asdict(row) for row in excluded],
        "bbox_annotation_complete": False,
        "model_evaluated": False,
        "candidate_checkpoint_opened": False,
        "test_opened": False,
        "runtime_connected": False,
        "production_promotion_authorized": False,
        "mount_portable_discovery": True,
    }
    manifest["selection_sha256"] = sha256(_canonical_json(manifest["selected"])).hexdigest()
    return manifest
