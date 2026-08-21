"""Fail-closed admission for the Meter V4-3 independent real holdout.

This module does not evaluate the model and does not modify candidate samples.
It selects a deterministic, balanced 150-family manifest from a larger Drive
candidate pool while excluding every family already observed by V4-1 or V4-2.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Final


METER_V4_3_FINAL_HOLDOUT_ADMISSION: Final[str] = "meter-v4-3-final-holdout-admission-v1"
RESULT_SCHEMA_V4_1: Final[str] = "st-omr-meter-v4-1-learned-numerator-specialist-result-v1"
RESULT_SCHEMA_V4_2: Final[str] = "st-omr-meter-v4-2-full-train-dev-screen-result-v1"
EXPECTED_CANDIDATES_PER_CLASS: Final[int] = 65
SELECTED_PER_CLASS: Final[int] = 50
CLASS_DIRS: Final[dict[str, str]] = {"2": "2/4", "3": "3/4", "4": "4/4"}
_FOLDER_RE = re.compile(r"^(?P<num>[234])_4_[0-9a-f]{12}_(?P<family>(?:aa|ab)_\d+)-")
_MAX_JSON_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_BBOX_BYTES: Final[int] = 2048


class MeterV4_3AdmissionError(RuntimeError):
    """Raised when the final holdout admission surface is not trustworthy."""


def _fail(message: str) -> None:
    raise MeterV4_3AdmissionError(message)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _read_json(path: Path, *, schema: str, name: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        _fail(f"{name} must be a regular file")
    size = path.stat().st_size
    if size <= 0 or size > _MAX_JSON_BYTES:
        _fail(f"{name} size is outside bounds")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeterV4_3AdmissionError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != schema:
        _fail(f"{name} schema mismatch")
    return value


def _nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        _fail(f"{name} must be a bounded non-empty string")
    return value


def observed_families(v4_1_result: dict[str, object], v4_2_result: dict[str, object]) -> frozenset[str]:
    records = v4_1_result.get("records")
    if not isinstance(records, list) or len(records) != 27:
        _fail("V4-1 must expose exactly 27 training/OOF records")
    v4_1_families = {
        _nonempty_string("V4-1 family_id", row.get("family_id"))
        for row in records
        if isinstance(row, dict)
    }
    if len(v4_1_families) != 27:
        _fail("V4-1 families must be 27 unique families")

    development = v4_2_result.get("development_validation")
    if not isinstance(development, dict):
        _fail("V4-2 development_validation missing")
    predictions = development.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 9:
        _fail("V4-2 development validation must expose exactly 9 predictions")
    v4_2_families = {
        _nonempty_string("V4-2 family_id", row.get("family_id"))
        for row in predictions
        if isinstance(row, dict)
    }
    if len(v4_2_families) != 9:
        _fail("V4-2 development families must be 9 unique families")
    if v4_1_families & v4_2_families:
        _fail("V4-1 and V4-2 observed family sets unexpectedly overlap")
    return frozenset(v4_1_families | v4_2_families)


@dataclass(frozen=True, slots=True)
class CandidateV4_3:
    numerator_class: str
    meter_class: str
    folder_name: str
    family_id: str
    image_path: str
    bbox_path: str


@dataclass(frozen=True, slots=True)
class ExcludedV4_3:
    numerator_class: str
    folder_name: str
    family_id: str
    reason: str


def _parse_bbox_header(path: Path, *, expected_meter: str) -> None:
    if not path.is_file() or path.is_symlink():
        _fail(f"bbox_meter.txt missing: {path}")
    raw = path.read_bytes()
    if len(raw) <= 0 or len(raw) > _MAX_BBOX_BYTES:
        _fail(f"bbox_meter.txt size outside bounds: {path}")
    try:
        text = raw.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise MeterV4_3AdmissionError(f"bbox_meter.txt is not UTF-8: {path}") from exc
    fields: dict[str, str] = {}
    for token in text.split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    required = {"id", "meter", "split", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "admit", "notes"}
    if not required.issubset(fields):
        _fail(f"bbox_meter.txt missing required fields: {path}")
    if fields["meter"] != expected_meter:
        _fail(f"bbox_meter.txt meter mismatch: {path}")
    # V4-3 admission is intentionally before human bbox annotation.
    if any(fields[key] for key in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")):
        _fail(f"candidate bbox must still be blank at admission: {path}")


def scan_candidate_pool(candidate_root: str | Path) -> tuple[CandidateV4_3, ...]:
    root = Path(candidate_root)
    if not root.is_dir() or root.is_symlink():
        _fail("candidate root must be an existing regular directory")
    candidates: list[CandidateV4_3] = []
    for numerator_class, class_dir_name in CLASS_DIRS.items():
        class_dir = root / class_dir_name
        if not class_dir.is_dir() or class_dir.is_symlink():
            _fail(f"missing class directory {class_dir_name}")
        folders = sorted(path for path in class_dir.iterdir() if path.is_dir() and not path.is_symlink())
        stray = [path.name for path in class_dir.iterdir() if not path.is_dir()]
        if stray:
            _fail(f"class directory contains non-folder entries: {class_dir_name}")
        if len(folders) != EXPECTED_CANDIDATES_PER_CLASS:
            _fail(
                f"{class_dir_name} must contain exactly {EXPECTED_CANDIDATES_PER_CLASS} candidate folders, got {len(folders)}"
            )
        for folder in folders:
            match = _FOLDER_RE.match(folder.name)
            if not match or match.group("num") != numerator_class:
                _fail(f"candidate folder name is outside frozen grammar: {folder.name}")
            family_id = match.group("family")
            image_path = folder / "image.png"
            bbox_path = folder / "bbox_meter.txt"
            if not image_path.is_file() or image_path.is_symlink() or image_path.stat().st_size <= 0:
                _fail(f"image.png missing or empty: {folder}")
            _parse_bbox_header(bbox_path, expected_meter=class_dir_name)
            candidates.append(
                CandidateV4_3(
                    numerator_class=numerator_class,
                    meter_class=class_dir_name,
                    folder_name=folder.name,
                    family_id=family_id,
                    image_path=str(image_path),
                    bbox_path=str(bbox_path),
                )
            )
    if len(candidates) != 195:
        _fail("V4-3 candidate pool must contain exactly 195 folders")
    return tuple(candidates)


def select_final_holdout(
    candidates: tuple[CandidateV4_3, ...],
    *,
    observed: frozenset[str],
) -> tuple[tuple[CandidateV4_3, ...], tuple[ExcludedV4_3, ...]]:
    family_occurrences: defaultdict[str, list[CandidateV4_3]] = defaultdict(list)
    for row in candidates:
        family_occurrences[row.family_id].append(row)
    duplicate_families = {family for family, rows in family_occurrences.items() if len(rows) != 1}

    eligible_by_class: dict[str, list[CandidateV4_3]] = {"2": [], "3": [], "4": []}
    excluded: list[ExcludedV4_3] = []
    for row in candidates:
        reason = None
        if row.family_id in observed:
            reason = "PREVIOUSLY_OBSERVED_FAMILY"
        elif row.family_id in duplicate_families:
            reason = "DUPLICATE_FAMILY_IN_CANDIDATE_POOL"
        if reason:
            excluded.append(
                ExcludedV4_3(
                    numerator_class=row.numerator_class,
                    folder_name=row.folder_name,
                    family_id=row.family_id,
                    reason=reason,
                )
            )
        else:
            eligible_by_class[row.numerator_class].append(row)

    selected: list[CandidateV4_3] = []
    for class_name in ("2", "3", "4"):
        eligible = sorted(eligible_by_class[class_name], key=lambda row: (row.folder_name, row.family_id))
        if len(eligible) < SELECTED_PER_CLASS:
            _fail(f"class {class_name} has only {len(eligible)} eligible independent families")
        selected.extend(eligible[:SELECTED_PER_CLASS])

    selected_families = [row.family_id for row in selected]
    if len(selected) != 150 or len(set(selected_families)) != 150:
        _fail("final holdout selection must contain exactly 150 unique families")
    if set(selected_families) & set(observed):
        _fail("final holdout selection leaked a previously observed family")
    if Counter(row.numerator_class for row in selected) != Counter({"2": 50, "3": 50, "4": 50}):
        _fail("final holdout selection must remain balanced 50/50/50")
    return tuple(selected), tuple(sorted(excluded, key=lambda row: (row.numerator_class, row.folder_name)))


def build_manifest(
    *,
    candidate_root: str | Path,
    v4_1_result_path: str | Path,
    v4_2_result_path: str | Path,
) -> dict[str, object]:
    v4_1 = _read_json(Path(v4_1_result_path), schema=RESULT_SCHEMA_V4_1, name="V4-1 result")
    v4_2 = _read_json(Path(v4_2_result_path), schema=RESULT_SCHEMA_V4_2, name="V4-2 result")
    observed = observed_families(v4_1, v4_2)
    candidates = scan_candidate_pool(candidate_root)
    selected, excluded = select_final_holdout(candidates, observed=observed)
    manifest: dict[str, object] = {
        "schema": "st-omr-meter-v4-3-final-holdout-admission-manifest-v1",
        "experiment": METER_V4_3_FINAL_HOLDOUT_ADMISSION,
        "candidate_root_name": Path(candidate_root).name,
        "candidate_count": 195,
        "candidate_classes": {"2": 65, "3": 65, "4": 65},
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
    }
    manifest["selection_sha256"] = sha256(_canonical_json(manifest["selected"])).hexdigest()
    return manifest


def write_manifest_atomic(manifest: dict[str, object], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_bytes(raw)
    temp.replace(path)
