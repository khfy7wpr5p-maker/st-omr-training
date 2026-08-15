"""Full pre-optimizer safety scan for the frozen Stage 7-D13 training surface."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Final, Mapping

from .stage7d13_measure_derivatives import STAGE7D13_DERIVATIVE_VERSION, STAGE7D13_LABEL_SCHEMA
from .stage7d13_symbol_models import build_symbol_model, encode_detector_targets
from .stage7d13_symbol_training_contract import (
    EXPECTED_SOURCE_FAMILY_COUNTS,
    MAX_PARAMETERS_COMBINED,
    SPECIALIST_CLASSES,
)
from .stage7d13_verified_surface import (
    D13_DERIVATIVE_BUILD_ID,
    D13_DERIVATIVE_MANIFEST_SHA256,
    D13_IMAGE_COUNT,
    D13_LABEL_COUNT,
    D13_RECORD_COUNT,
    D13_RECORD_SPLIT_COUNTS,
)
from .training_model import count_trainable_parameters


STAGE7D13_PREFLIGHT_VERSION: Final[str] = "stage7d13-training-preflight-v1"
_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_EXPECTED_TOP: Final[frozenset[str]] = frozenset(
    {"manifest.json", "manifest.sha256", "build.json", "images", "labels"}
)


class Stage7D13PreflightError(RuntimeError):
    """Raised before optimizer creation when the frozen D13 surface is unsafe."""


def _fail(message: str) -> None:
    raise Stage7D13PreflightError(message)


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha64(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX for c in value):
        _fail(f"{name} must be lowercase SHA-256")
    return value


def _json(path: Path, name: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{name} must be regular non-symlink file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D13PreflightError(f"{name} is not valid ASCII JSON") from exc
    if not isinstance(value, dict) or _canonical(value) != raw:
        _fail(f"{name} must be canonical JSON object bytes")
    return value, raw


@dataclass(frozen=True, slots=True)
class Stage7D13PreflightReceipt:
    version: str
    record_count: int
    record_split_counts: dict[str, int]
    unique_image_count: int
    label_count: int
    family_split_counts: dict[str, int]
    parameter_counts: dict[str, int]
    parameter_count_total: int
    collision_free: bool
    test_opened: bool
    preflight_passed: bool


def verify_stage7d13_training_preflight(derivative_root: str | Path) -> Stage7D13PreflightReceipt:
    """Hash/decode-label/collision/parameter scan before any optimizer is created."""
    root = Path(derivative_root)
    if root.is_symlink() or not root.is_dir():
        _fail("D13 derivative root must be regular non-symlink directory")
    if {p.name for p in root.iterdir()} != _EXPECTED_TOP:
        _fail("D13 derivative top-level surface mismatch")
    manifest, raw = _json(root / "manifest.json", "D13 manifest")
    if sha256(raw).hexdigest() != D13_DERIVATIVE_MANIFEST_SHA256:
        _fail("D13 preflight manifest SHA mismatch")
    if manifest.get("derivative_build_id") != D13_DERIVATIVE_BUILD_ID:
        _fail("D13 preflight build id mismatch")
    rows = manifest.get("records")
    if not isinstance(rows, list) or len(rows) != D13_RECORD_COUNT:
        _fail("D13 preflight record cardinality mismatch")

    split_counts: Counter[str] = Counter()
    family_split: dict[str, str] = {}
    image_hashes: set[str] = set()
    label_hashes: set[str] = set()
    seen_records: set[str] = set()

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"D13 manifest record[{index}] must be object")
        split = row.get("split")
        if split not in ("train", "validation"):
            _fail("sealed TEST or invalid split reached D13 preflight")
        record_id = _sha64(row.get("record_id"), "record_id")
        image_sha = _sha64(row.get("image_sha256"), "image_sha256")
        label_sha = _sha64(row.get("label_sha256"), "label_sha256")
        family_id = row.get("family_id")
        if not isinstance(family_id, str) or not family_id:
            _fail("D13 family id is invalid")
        if record_id in seen_records:
            _fail("duplicate D13 record id")
        seen_records.add(record_id)
        prior = family_split.setdefault(family_id, str(split))
        if prior != split:
            _fail("D13 family crosses TRAIN/VALIDATION")
        split_counts[str(split)] += 1

        if image_sha not in image_hashes:
            image_path = root / "images" / f"{image_sha}.png"
            if image_path.is_symlink() or not image_path.is_file():
                _fail("D13 preflight image must be regular file")
            if sha256(image_path.read_bytes()).hexdigest() != image_sha:
                _fail("D13 preflight image SHA mismatch")
            image_hashes.add(image_sha)

        label_path = root / "labels" / f"{label_sha}.json"
        label, label_raw = _json(label_path, "D13 measure label")
        if sha256(label_raw).hexdigest() != label_sha:
            _fail("D13 preflight label SHA mismatch")
        if label_sha in label_hashes:
            _fail("duplicate D13 label SHA")
        label_hashes.add(label_sha)
        if label.get("schema_version") != STAGE7D13_LABEL_SCHEMA:
            _fail("D13 preflight label schema mismatch")
        if label.get("stage7d13_derivative_version") != STAGE7D13_DERIVATIVE_VERSION:
            _fail("D13 preflight label version mismatch")
        if label.get("record_id") != record_id or label.get("split") != split:
            _fail("D13 preflight label/manifest identity mismatch")
        targets = label.get("targets")
        if not isinstance(targets, Mapping):
            _fail("D13 preflight targets missing")
        for specialist in SPECIALIST_CLASSES:
            target_rows = targets.get(specialist)
            if not isinstance(target_rows, list) or any(not isinstance(value, Mapping) for value in target_rows):
                _fail(f"D13 preflight {specialist} targets malformed")
            encode_detector_targets(specialist, [target_rows])

    if dict(split_counts) != D13_RECORD_SPLIT_COUNTS:
        _fail("D13 preflight split counts mismatch")
    if len(seen_records) != D13_RECORD_COUNT or len(label_hashes) != D13_LABEL_COUNT:
        _fail("D13 preflight record/label cardinality mismatch")
    if len(image_hashes) != D13_IMAGE_COUNT:
        _fail("D13 preflight content-addressed image cardinality mismatch")
    family_counts = dict(Counter(family_split.values()))
    if family_counts != EXPECTED_SOURCE_FAMILY_COUNTS:
        _fail("D13 preflight family split counts mismatch")

    parameter_counts = {
        specialist: count_trainable_parameters(build_symbol_model(specialist))
        for specialist in SPECIALIST_CLASSES
    }
    total_parameters = sum(parameter_counts.values())
    if total_parameters > MAX_PARAMETERS_COMBINED:
        _fail("D13 combined specialist parameter cap exceeded")

    return Stage7D13PreflightReceipt(
        version=STAGE7D13_PREFLIGHT_VERSION,
        record_count=D13_RECORD_COUNT,
        record_split_counts=dict(D13_RECORD_SPLIT_COUNTS),
        unique_image_count=len(image_hashes),
        label_count=len(label_hashes),
        family_split_counts=family_counts,
        parameter_counts=parameter_counts,
        parameter_count_total=total_parameters,
        collision_free=True,
        test_opened=False,
        preflight_passed=True,
    )
