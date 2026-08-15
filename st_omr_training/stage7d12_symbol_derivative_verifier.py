"""Independent persisted-bundle verifier for Stage 7-D12.

This module does not build symbol labels and does not trust the builder's in-memory
receipt.  It independently reopens the persisted D12 bundle, the frozen source
corpus, and the accepted D6 sidecars; recomputes hashes, identities,
cardinalities, family split isolation and class inventory; and fails closed on
any drift.

D12-5 verifies an *uncompleted* bundle.  A ``COMPLETE`` marker is forbidden at
this gate.  Writing completion evidence is a later controlled closure step.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Final

from .stage7d12_symbol_derivatives import (
    STAGE7D12_BUILD_SCHEMA,
    STAGE7D12_DERIVATIVE_VERSION,
    STAGE7D12_LABEL_SCHEMA,
    STAGE7D12_MANIFEST_SCHEMA,
    stage7d12_derivative_profile_fingerprint,
)
from .stage7d12_symbol_geometry import STAGE7D12_SYMBOL_GEOMETRY_VERSION
from .stage7d12_symbol_gt_contract import (
    ACCIDENTAL_CLASSES,
    EXPECTED_DEVELOPMENT_FAMILY_COUNTS,
    EXPECTED_DEVELOPMENT_SAMPLE_COUNTS,
    NOTEHEAD_FILL_CLASSES,
    REST_CLASSES,
    stage7d12_contract_fingerprint,
    validate_canonical_event_id,
)
from .stage7d5_geometry import STAGE7D5_TRANSFORM_VERSION
from .stage7d6_specialist_derivatives import (
    STAGE7D6_LABEL_SCHEMA,
    STAGE7D6_VERSION,
    verify_stage7d6_derivatives,
)


STAGE7D12_VERIFIER_VERSION: Final[str] = "stage7d12-symbol-persisted-verifier-v1"
EXPECTED_D6_DERIVATIVE_BUILD_ID: Final[str] = (
    "0faafe229f3497b1147cf0f0ac0ce4b7efe6fa31f360a6a33a3b82c986c8c519"
)
EXPECTED_D6_MANIFEST_SHA256: Final[str] = (
    "e8e415eb6ba9d91a1a880709c3f31d559aa20bf5149734f45b5f84ced16afee9"
)
EXPECTED_D6_ARTIFACT_BINDING_SHA256: Final[str] = (
    "3b7558f0f927ad47a61ed5afb5faa8584dca8647cf8683d4043686eb7b077ea1"
)
EXPECTED_D6_LABEL_COUNT: Final[int] = 1383
EXPECTED_D6_FAMILY_COUNT: Final[int] = 461

_ALLOWED_SPLITS: Final[frozenset[str]] = frozenset({"train", "validation"})
_EXPECTED_SAMPLE_COUNT: Final[int] = sum(EXPECTED_DEVELOPMENT_SAMPLE_COUNTS.values())
_EXPECTED_FAMILY_COUNT: Final[int] = sum(EXPECTED_DEVELOPMENT_FAMILY_COUNTS.values())
_TOP_LEVEL: Final[frozenset[str]] = frozenset(
    {"manifest.json", "manifest.sha256", "build.json", "labels"}
)
_MAX_MANIFEST_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_BUILD_BYTES: Final[int] = 2 * 1024 * 1024
_MAX_LABEL_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_IMAGE_BYTES: Final[int] = 32 * 1024 * 1024
_HEX64: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


class Stage7D12VerificationError(RuntimeError):
    """Raised when persisted D12 evidence cannot be independently proven."""


def _fail(message: str) -> None:
    raise Stage7D12VerificationError(message)


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
        raise Stage7D12VerificationError("payload is not canonical JSON") from exc


def _regular_file(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        _fail(f"{name} must be a regular non-symlink file")


def _regular_directory(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_dir():
        _fail(f"{name} must be a regular non-symlink directory")


def _read_bounded(path: Path, maximum: int, name: str) -> bytes:
    _regular_file(path, name)
    size = path.stat().st_size
    if not 1 <= size <= maximum:
        _fail(f"{name} byte length is outside the D12 verifier bound")
    return path.read_bytes()


def _read_canonical_json(
    path: Path, maximum: int, name: str
) -> tuple[dict[str, object], bytes]:
    raw = _read_bounded(path, maximum, name)
    try:
        payload = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda token: _fail(
                f"non-finite JSON constant in {name}: {token}"
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D12VerificationError(
            f"{name} is not valid ASCII JSON"
        ) from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        _fail(f"{name} must be canonical JSON object bytes")
    return payload, raw


def _hex64(name: str, value: object) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        _fail(f"{name} must be lowercase SHA-256 hex")
    return value


def _identifier(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or len(value) > 256
    ):
        _fail(f"{name} must be bounded non-empty ASCII")
    return value


def _positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _fail(f"{name} must be a positive integer")
    return value


def _finite(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def development_rows(rows: object) -> tuple[Mapping[str, object], ...]:
    """Expose TRAIN/VALIDATION while touching only ``split`` on TEST rows."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        _fail("source samples must be a sequence")
    accepted: list[Mapping[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"source sample[{index}] must be a mapping")
        split = row.get("split")
        if split == "test":
            continue
        if split not in _ALLOWED_SPLITS:
            _fail(f"source sample[{index}] has invalid development split")
        accepted.append(row)
    return tuple(accepted)


def _empty_inventory() -> dict[str, dict[str, dict[str, int]]]:
    return {
        split: {
            "notehead": {name: 0 for name in NOTEHEAD_FILL_CLASSES},
            "rest": {name: 0 for name in REST_CLASSES},
            "accidental": {name: 0 for name in ACCIDENTAL_CLASSES},
        }
        for split in ("train", "validation")
    }


def _box(
    name: str,
    value: object,
    *,
    width: int,
    height: int,
    outer: dict[str, float] | None = None,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    }:
        _fail(f"{name} must be a canonical bbox")
    result = {
        key: _finite(f"{name}.{key}", value.get(key))
        for key in ("x_min", "y_min", "x_max", "y_max")
    }
    if not result["x_min"] < result["x_max"] or not result["y_min"] < result["y_max"]:
        _fail(f"{name} must have positive area")
    epsilon = 1e-5
    if (
        result["x_min"] < -epsilon
        or result["y_min"] < -epsilon
        or result["x_max"] > width + epsilon
        or result["y_max"] > height + epsilon
    ):
        _fail(f"{name} lies outside final PNG")
    if outer is not None and (
        result["x_min"] < outer["x_min"] - epsilon
        or result["y_min"] < outer["y_min"] - epsilon
        or result["x_max"] > outer["x_max"] + epsilon
        or result["y_max"] > outer["y_max"] + epsilon
    ):
        _fail(f"{name} lies outside owning measure")
    return result


def _point_inside(
    name: str,
    value: object,
    box: Mapping[str, float],
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"x", "y"}:
        _fail(f"{name} must be a canonical point")
    x = _finite(f"{name}.x", value.get("x"))
    y = _finite(f"{name}.y", value.get("y"))
    epsilon = 1e-5
    if not (
        float(box["x_min"]) - epsilon <= x <= float(box["x_max"]) + epsilon
        and float(box["y_min"]) - epsilon <= y <= float(box["y_max"]) + epsilon
    ):
        _fail(f"{name} lies outside its notehead bbox")


def _source_index(corpus_root: Path) -> dict[str, Mapping[str, object]]:
    manifest, _ = _read_canonical_json(
        corpus_root / "manifest.json", _MAX_MANIFEST_BYTES, "source manifest.json"
    )
    rows = development_rows(manifest.get("samples"))
    if len(rows) != _EXPECTED_SAMPLE_COUNT:
        _fail("source development sample cardinality mismatch")

    index: dict[str, Mapping[str, object]] = {}
    sample_counts: Counter[str] = Counter()
    family_split: dict[str, str] = {}
    for row in rows:
        sample_id = _hex64("source.sample_id", row.get("sample_id"))
        if sample_id in index:
            _fail("duplicate source development sample_id")
        family_id = _identifier("source.family_id", row.get("family_id"))
        split = row.get("split")
        if split not in _ALLOWED_SPLITS:
            _fail("source split escaped TEST seal")
        assert isinstance(split, str)
        prior = family_split.setdefault(family_id, split)
        if prior != split:
            _fail("source family crosses TRAIN/VALIDATION")
        sample_counts[split] += 1
        index[sample_id] = row

    if dict(sorted(sample_counts.items())) != EXPECTED_DEVELOPMENT_SAMPLE_COUNTS:
        _fail("source development split counts mismatch")
    family_counts = dict(sorted(Counter(family_split.values()).items()))
    if family_counts != EXPECTED_DEVELOPMENT_FAMILY_COUNTS:
        _fail("source development family split counts mismatch")
    if len(family_split) != _EXPECTED_FAMILY_COUNT:
        _fail("source development family cardinality mismatch")
    return index


def _d6_index(corpus_root: Path, d6_root: Path) -> dict[str, Mapping[str, object]]:
    receipt = verify_stage7d6_derivatives(corpus_root, d6_root)
    expected = {
        "derivative_build_id": EXPECTED_D6_DERIVATIVE_BUILD_ID,
        "manifest_sha256": EXPECTED_D6_MANIFEST_SHA256,
        "artifact_binding_sha256": EXPECTED_D6_ARTIFACT_BINDING_SHA256,
        "label_count": EXPECTED_D6_LABEL_COUNT,
        "sample_count": EXPECTED_D6_LABEL_COUNT,
        "family_count": EXPECTED_D6_FAMILY_COUNT,
        "test_specialist_records": 0,
    }
    for name, value in expected.items():
        if getattr(receipt, name) != value:
            _fail(f"accepted D6 receipt {name} mismatch")

    manifest, raw = _read_canonical_json(
        d6_root / "manifest.json", _MAX_MANIFEST_BYTES, "accepted D6 manifest.json"
    )
    if sha256(raw).hexdigest() != EXPECTED_D6_MANIFEST_SHA256:
        _fail("accepted D6 manifest SHA-256 mismatch")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != _EXPECTED_SAMPLE_COUNT:
        _fail("accepted D6 record cardinality mismatch")
    result: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            _fail("accepted D6 record must be an object")
        if record.get("split") not in _ALLOWED_SPLITS:
            _fail("accepted D6 manifest contains forbidden split")
        sample_id = _hex64("D6 record.sample_id", record.get("sample_id"))
        if sample_id in result:
            _fail("duplicate accepted D6 sample_id")
        result[sample_id] = record
    return result


def _load_d6_label(
    d6_root: Path, record: Mapping[str, object]
) -> tuple[dict[str, object], str]:
    label_sha = _hex64("D6 record.label_sha256", record.get("label_sha256"))
    label, raw = _read_canonical_json(
        d6_root / "labels" / f"{label_sha}.json",
        _MAX_LABEL_BYTES,
        "accepted D6 label",
    )
    if sha256(raw).hexdigest() != label_sha:
        _fail("accepted D6 label SHA-256 mismatch")
    if label.get("schema_version") != STAGE7D6_LABEL_SCHEMA:
        _fail("accepted D6 label schema mismatch")
    if label.get("stage7d6_version") != STAGE7D6_VERSION:
        _fail("accepted D6 label version mismatch")
    return label, label_sha


def _d6_geometry_lineage(label: Mapping[str, object]) -> tuple[str, str, str]:
    geometry = label.get("geometry")
    if not isinstance(geometry, Mapping):
        _fail("accepted D6 geometry must be an object")
    return (
        _hex64(
            "D6 geometry.geometry_instrumentation_fingerprint",
            geometry.get("geometry_instrumentation_fingerprint"),
        ),
        _hex64(
            "D6 geometry.geometry_svg_sha256",
            geometry.get("geometry_svg_sha256"),
        ),
        _hex64(
            "D6 geometry.geometry_transform_fingerprint",
            geometry.get("geometry_transform_fingerprint"),
        ),
    )


def _validate_symbol_label(
    label: Mapping[str, object],
    *,
    source: Mapping[str, object],
    d6_record: Mapping[str, object],
    d6_label_sha: str,
    d6_geometry_lineage: tuple[str, str, str],
    inventory: dict[str, dict[str, dict[str, int]]],
) -> None:
    expected_top = {
        "schema_version",
        "stage7d12_derivative_version",
        "contract_fingerprint",
        "sample_id",
        "family_id",
        "split",
        "page_number",
        "image",
        "accepted_d6",
        "lineage",
        "symbol_geometry",
    }
    if set(label) != expected_top:
        _fail("D12 symbol label top-level keys mismatch")
    if label.get("schema_version") != STAGE7D12_LABEL_SCHEMA:
        _fail("D12 symbol label schema mismatch")
    if label.get("stage7d12_derivative_version") != STAGE7D12_DERIVATIVE_VERSION:
        _fail("D12 symbol label version mismatch")
    if label.get("contract_fingerprint") != stage7d12_contract_fingerprint():
        _fail("D12 symbol label contract fingerprint mismatch")

    sample_id = _hex64("source.sample_id", source.get("sample_id"))
    family_id = _identifier("source.family_id", source.get("family_id"))
    split = source.get("split")
    if split not in _ALLOWED_SPLITS:
        _fail("source split is outside D12 development surface")
    assert isinstance(split, str)
    page_number = _positive_int("source.page_number", source.get("page_number"))
    png_sha = _hex64("source.png_sha256", source.get("png_sha256"))
    width = _positive_int("source.width", source.get("width"))
    height = _positive_int("source.height", source.get("height"))
    if (
        label.get("sample_id") != sample_id
        or label.get("family_id") != family_id
        or label.get("split") != split
        or label.get("page_number") != page_number
    ):
        _fail("D12 symbol label source identity mismatch")

    image = label.get("image")
    if not isinstance(image, Mapping) or dict(image) != {
        "png_sha256": png_sha,
        "width": width,
        "height": height,
        "mode": "L",
        "image_format": "png",
    }:
        _fail("D12 symbol label image lineage mismatch")

    if d6_record.get("sample_id") != sample_id or d6_record.get("png_sha256") != png_sha:
        _fail("accepted D6 record/source identity mismatch")
    accepted_d6 = label.get("accepted_d6")
    if not isinstance(accepted_d6, Mapping) or dict(accepted_d6) != {
        "manifest_sha256": EXPECTED_D6_MANIFEST_SHA256,
        "artifact_binding_sha256": EXPECTED_D6_ARTIFACT_BINDING_SHA256,
        "label_sha256": d6_label_sha,
    }:
        _fail("D12 symbol label accepted-D6 lineage mismatch")

    source_musicxml_sha = _hex64(
        "source.source_musicxml_sha256", source.get("source_musicxml_sha256")
    )
    source_svg_sha = _hex64("source.source_svg_sha256", source.get("source_svg_sha256"))
    renderer_fp = _hex64(
        "source.renderer_config_fingerprint", source.get("renderer_config_fingerprint")
    )
    degradation_fp = _hex64(
        "source.degradation_config_fingerprint",
        source.get("degradation_config_fingerprint"),
    )
    d6_instrumentation, d6_geometry_svg, d6_transform = d6_geometry_lineage
    lineage = label.get("lineage")
    expected_lineage = {
        "source_musicxml_sha256": source_musicxml_sha,
        "source_svg_sha256": source_svg_sha,
        "renderer_config_fingerprint": renderer_fp,
        "degradation_config_fingerprint": degradation_fp,
        "symbol_geometry_version": STAGE7D12_SYMBOL_GEOMETRY_VERSION,
        "geometry_instrumentation_fingerprint": d6_instrumentation,
        "geometry_svg_sha256": d6_geometry_svg,
        "d5_transform_version": STAGE7D5_TRANSFORM_VERSION,
        "geometry_transform_fingerprint": d6_transform,
    }
    if not isinstance(lineage, Mapping) or dict(lineage) != expected_lineage:
        _fail("D12 symbol label geometry/source lineage mismatch")

    symbol_geometry = label.get("symbol_geometry")
    if not isinstance(symbol_geometry, Mapping) or set(symbol_geometry) != {
        "coordinate_space",
        "view_box",
        "measures",
    }:
        _fail("D12 symbol_geometry shape mismatch")
    if symbol_geometry.get("coordinate_space") != "final_png_pixels":
        _fail("D12 persisted symbol geometry must be final_png_pixels")
    view_box = symbol_geometry.get("view_box")
    if not isinstance(view_box, list) or len(view_box) != 4:
        _fail("D12 symbol view_box must contain four values")
    for index, value in enumerate(view_box):
        _finite(f"symbol_geometry.view_box[{index}]", value)

    measures = symbol_geometry.get("measures")
    if not isinstance(measures, list) or not measures:
        _fail("D12 symbol label requires at least one measure")
    seen_measure_numbers: set[int] = set()
    seen_measure_renderer_ids: set[str] = set()
    seen_event_ids: dict[str, set[str]] = {
        "notehead": set(),
        "rest": set(),
        "accidental": set(),
    }
    seen_renderer_ids: dict[str, set[str]] = {
        "notehead": set(),
        "rest": set(),
        "accidental": set(),
    }

    for measure in measures:
        if not isinstance(measure, Mapping) or set(measure) != {
            "measure_number",
            "renderer_measure_id",
            "measure_bbox",
            "noteheads",
            "rests",
            "accidentals",
        }:
            _fail("D12 persisted measure shape mismatch")
        number = _positive_int("measure_number", measure.get("measure_number"))
        if number in seen_measure_numbers:
            _fail("duplicate persisted measure_number")
        seen_measure_numbers.add(number)
        renderer_measure_id = _identifier(
            "renderer_measure_id", measure.get("renderer_measure_id")
        )
        if renderer_measure_id in seen_measure_renderer_ids:
            _fail("duplicate persisted renderer_measure_id")
        seen_measure_renderer_ids.add(renderer_measure_id)
        measure_box = _box(
            "measure_bbox",
            measure.get("measure_bbox"),
            width=width,
            height=height,
        )

        noteheads = measure.get("noteheads")
        rests = measure.get("rests")
        accidentals = measure.get("accidentals")
        if not all(isinstance(rows, list) for rows in (noteheads, rests, accidentals)):
            _fail("D12 target collections must be lists")
        assert isinstance(noteheads, list)
        assert isinstance(rests, list)
        assert isinstance(accidentals, list)

        for row in noteheads:
            if not isinstance(row, Mapping) or set(row) != {
                "canonical_event_id",
                "renderer_id",
                "notehead_bbox",
                "notehead_center",
                "fill_class",
            }:
                _fail("persisted notehead target shape mismatch")
            event_id = validate_canonical_event_id(row.get("canonical_event_id"))
            renderer_id = _identifier("notehead.renderer_id", row.get("renderer_id"))
            if event_id in seen_event_ids["notehead"] or renderer_id in seen_renderer_ids["notehead"]:
                _fail("duplicate persisted NoteHeadSet identity")
            seen_event_ids["notehead"].add(event_id)
            seen_renderer_ids["notehead"].add(renderer_id)
            cls = row.get("fill_class")
            if cls not in NOTEHEAD_FILL_CLASSES:
                _fail("persisted notehead class is outside D12")
            assert isinstance(cls, str)
            box = _box(
                "notehead_bbox",
                row.get("notehead_bbox"),
                width=width,
                height=height,
                outer=measure_box,
            )
            _point_inside("notehead_center", row.get("notehead_center"), box)
            inventory[split]["notehead"][cls] += 1

        for row in rests:
            if not isinstance(row, Mapping) or set(row) != {
                "canonical_event_id",
                "renderer_id",
                "rest_bbox",
                "rest_class",
                "duration_class",
            }:
                _fail("persisted rest target shape mismatch")
            event_id = validate_canonical_event_id(row.get("canonical_event_id"))
            renderer_id = _identifier("rest.renderer_id", row.get("renderer_id"))
            if event_id in seen_event_ids["rest"] or renderer_id in seen_renderer_ids["rest"]:
                _fail("duplicate persisted RestSet identity")
            seen_event_ids["rest"].add(event_id)
            seen_renderer_ids["rest"].add(renderer_id)
            cls = row.get("rest_class")
            if cls not in REST_CLASSES or row.get("duration_class") != cls:
                _fail("persisted rest class/duration is outside D12")
            assert isinstance(cls, str)
            _box(
                "rest_bbox",
                row.get("rest_bbox"),
                width=width,
                height=height,
                outer=measure_box,
            )
            inventory[split]["rest"][cls] += 1

        for row in accidentals:
            if not isinstance(row, Mapping) or set(row) != {
                "canonical_event_id",
                "renderer_id",
                "accidental_bbox",
                "accidental_class",
            }:
                _fail("persisted accidental target shape mismatch")
            event_id = validate_canonical_event_id(row.get("canonical_event_id"))
            renderer_id = _identifier("accidental.renderer_id", row.get("renderer_id"))
            if event_id in seen_event_ids["accidental"] or renderer_id in seen_renderer_ids["accidental"]:
                _fail("duplicate persisted AccidentalSet identity")
            seen_event_ids["accidental"].add(event_id)
            seen_renderer_ids["accidental"].add(renderer_id)
            cls = row.get("accidental_class")
            if cls not in ACCIDENTAL_CLASSES:
                _fail("persisted accidental class is outside D12")
            assert isinstance(cls, str)
            _box(
                "accidental_bbox",
                row.get("accidental_bbox"),
                width=width,
                height=height,
                outer=measure_box,
            )
            inventory[split]["accidental"][cls] += 1


@dataclass(frozen=True, slots=True)
class Stage7D12VerificationReceipt:
    verifier_version: str
    derivative_build_id: str
    manifest_sha256: str
    artifact_binding_sha256: str
    sample_count: int
    family_count: int
    label_count: int
    label_bytes_total: int
    sample_split_counts: dict[str, int]
    family_split_counts: dict[str, int]
    observed_class_inventory: dict[str, dict[str, dict[str, int]]]
    test_specialist_records: int
    optimizer_steps: int
    complete_marker_present: bool
    verification_passed: bool


def verify_stage7d12_symbol_derivatives(
    corpus_root: str | Path,
    d6_root: str | Path,
    derivative_root: str | Path,
) -> Stage7D12VerificationReceipt:
    """Independently reopen and verify an uncompleted persisted D12 bundle."""

    if not all(
        isinstance(value, (str, Path))
        for value in (corpus_root, d6_root, derivative_root)
    ):
        raise TypeError(
            "corpus_root, d6_root and derivative_root must be str or pathlib.Path"
        )
    source_root = Path(corpus_root)
    accepted_d6_root = Path(d6_root)
    root = Path(derivative_root)
    _regular_directory(source_root, "source corpus root")
    _regular_directory(accepted_d6_root, "accepted D6 root")
    _regular_directory(root, "D12 derivative root")
    if {entry.name for entry in root.iterdir()} != _TOP_LEVEL:
        _fail("D12 derivative top-level layout mismatch or premature COMPLETE")
    _regular_directory(root / "labels", "D12 labels")

    source_index = _source_index(source_root)
    d6_index = _d6_index(source_root, accepted_d6_root)
    if set(source_index) != set(d6_index):
        _fail("source/D6 development sample identities do not match exactly")

    manifest, manifest_raw = _read_canonical_json(
        root / "manifest.json", _MAX_MANIFEST_BYTES, "D12 manifest.json"
    )
    manifest_sha = sha256(manifest_raw).hexdigest()
    checksum = _read_bounded(root / "manifest.sha256", 256, "D12 manifest.sha256")
    if checksum != f"{manifest_sha}  manifest.json\n".encode("ascii"):
        _fail("D12 manifest.sha256 content mismatch")

    expected_manifest_header = {
        "schema_version": STAGE7D12_MANIFEST_SCHEMA,
        "stage7d12_derivative_version": STAGE7D12_DERIVATIVE_VERSION,
        "profile_fingerprint": stage7d12_derivative_profile_fingerprint(),
        "contract_fingerprint": stage7d12_contract_fingerprint(),
        "accepted_d6": {
            "derivative_build_id": EXPECTED_D6_DERIVATIVE_BUILD_ID,
            "manifest_sha256": EXPECTED_D6_MANIFEST_SHA256,
            "artifact_binding_sha256": EXPECTED_D6_ARTIFACT_BINDING_SHA256,
        },
        "split_policy": "family-exclusive-train-validation-only-test-sealed",
        "completion_policy": "independent-verifier-required-before-COMPLETE",
    }
    if set(manifest) != set(expected_manifest_header) | {"records"}:
        _fail("D12 manifest keys mismatch")
    for key, value in expected_manifest_header.items():
        if manifest.get(key) != value:
            _fail(f"D12 manifest {key} mismatch")

    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != _EXPECTED_SAMPLE_COUNT:
        _fail("D12 manifest record cardinality mismatch")
    if records != sorted(
        records,
        key=lambda row: str(row.get("sample_id")) if isinstance(row, Mapping) else "",
    ):
        _fail("D12 manifest records must be sorted by sample_id")

    labels_on_disk = {entry.name for entry in (root / "labels").iterdir()}
    expected_label_names: set[str] = set()
    seen_samples: set[str] = set()
    sample_counts: Counter[str] = Counter()
    family_split: dict[str, str] = {}
    inventory = _empty_inventory()
    label_bytes_total = 0
    binding_rows: list[str] = []

    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {
            "sample_id",
            "family_id",
            "split",
            "page_number",
            "png_sha256",
            "d6_label_sha256",
            "symbol_label_sha256",
        }:
            _fail(f"D12 manifest record[{index}] shape mismatch")
        split = record.get("split")
        if split not in _ALLOWED_SPLITS:
            _fail("D12 manifest contains forbidden split")
        assert isinstance(split, str)
        sample_id = _hex64("record.sample_id", record.get("sample_id"))
        if sample_id in seen_samples:
            _fail("duplicate D12 manifest sample_id")
        seen_samples.add(sample_id)
        source = source_index.get(sample_id)
        d6_record = d6_index.get(sample_id)
        if source is None or d6_record is None:
            _fail("D12 record has no exact source/D6 development identity")

        family_id = _identifier("record.family_id", record.get("family_id"))
        if family_id != source.get("family_id") or split != source.get("split"):
            _fail("D12 record family/split differs from source")
        if record.get("page_number") != source.get("page_number"):
            _fail("D12 record page number differs from source")
        png_sha = _hex64("record.png_sha256", record.get("png_sha256"))
        if png_sha != source.get("png_sha256") or png_sha != d6_record.get("png_sha256"):
            _fail("D12 record image SHA differs from source/D6")
        png_raw = _read_bounded(
            source_root / "images" / f"{png_sha}.png",
            _MAX_IMAGE_BYTES,
            "source PNG",
        )
        if sha256(png_raw).hexdigest() != png_sha:
            _fail("source PNG SHA-256 mismatch during independent verification")

        d6_label, d6_label_sha = _load_d6_label(accepted_d6_root, d6_record)
        if record.get("d6_label_sha256") != d6_label_sha:
            _fail("D12 record D6 label SHA mismatch")
        d6_lineage = _d6_geometry_lineage(d6_label)

        symbol_label_sha = _hex64(
            "record.symbol_label_sha256", record.get("symbol_label_sha256")
        )
        label_name = f"{symbol_label_sha}.json"
        if label_name in expected_label_names:
            _fail("duplicate D12 symbol label reference")
        expected_label_names.add(label_name)
        label, label_raw = _read_canonical_json(
            root / "labels" / label_name,
            _MAX_LABEL_BYTES,
            f"D12 symbol label {label_name}",
        )
        if sha256(label_raw).hexdigest() != symbol_label_sha:
            _fail("D12 symbol label SHA-256 does not match filename")
        _validate_symbol_label(
            label,
            source=source,
            d6_record=d6_record,
            d6_label_sha=d6_label_sha,
            d6_geometry_lineage=d6_lineage,
            inventory=inventory,
        )

        label_bytes_total += len(label_raw)
        sample_counts[split] += 1
        prior = family_split.setdefault(family_id, split)
        if prior != split:
            _fail("D12 family crosses TRAIN/VALIDATION")
        binding_rows.append(
            f"{sample_id}:{png_sha}:{d6_label_sha}:{symbol_label_sha}:{len(label_raw)}"
        )

    if labels_on_disk != expected_label_names:
        _fail("D12 label filenames do not exactly match manifest references")
    if len(labels_on_disk) != _EXPECTED_SAMPLE_COUNT:
        _fail("D12 label cardinality mismatch")
    sample_split_counts = dict(sorted(sample_counts.items()))
    if sample_split_counts != EXPECTED_DEVELOPMENT_SAMPLE_COUNTS:
        _fail("D12 independently recomputed split counts mismatch")
    family_split_counts = dict(sorted(Counter(family_split.values()).items()))
    if family_split_counts != EXPECTED_DEVELOPMENT_FAMILY_COUNTS:
        _fail("D12 independently recomputed family split counts mismatch")
    if len(family_split) != _EXPECTED_FAMILY_COUNT:
        _fail("D12 independently recomputed family cardinality mismatch")

    artifact_binding = sha256(
        ("\n".join(sorted(binding_rows)) + "\n").encode("ascii")
    ).hexdigest()
    target_instance_counts = {
        split: {
            kind: sum(inventory[split][kind].values())
            for kind in ("notehead", "rest", "accidental")
        }
        for split in ("train", "validation")
    }
    build, _build_raw = _read_canonical_json(
        root / "build.json", _MAX_BUILD_BYTES, "D12 build.json"
    )
    expected_build = {
        "schema_version": STAGE7D12_BUILD_SCHEMA,
        "stage7d12_derivative_version": STAGE7D12_DERIVATIVE_VERSION,
        "derivative_build_id": stage7d12_derivative_profile_fingerprint(),
        "manifest_sha256": manifest_sha,
        "sample_count": _EXPECTED_SAMPLE_COUNT,
        "family_count": _EXPECTED_FAMILY_COUNT,
        "label_count": _EXPECTED_SAMPLE_COUNT,
        "label_bytes_total": label_bytes_total,
        "sample_split_counts": EXPECTED_DEVELOPMENT_SAMPLE_COUNTS,
        "family_split_counts": EXPECTED_DEVELOPMENT_FAMILY_COUNTS,
        "target_instance_counts": target_instance_counts,
        "observed_class_inventory": inventory,
        "artifact_binding_sha256": artifact_binding,
        "test_specialist_records": 0,
        "optimizer_steps": 0,
        "complete_marker_written": False,
        "completion_policy": "independent-verifier-required-before-COMPLETE",
        "layout": {
            "manifest": "manifest.json",
            "labels": "labels/<symbol_label_sha256>.json",
            "source_images": "external frozen corpus images/<png_sha256>.png",
            "accepted_d6_labels": "external accepted D6 labels/<d6_label_sha256>.json",
        },
    }
    if build != expected_build:
        _fail("D12 build.json differs from independently recomputed evidence")

    return Stage7D12VerificationReceipt(
        verifier_version=STAGE7D12_VERIFIER_VERSION,
        derivative_build_id=stage7d12_derivative_profile_fingerprint(),
        manifest_sha256=manifest_sha,
        artifact_binding_sha256=artifact_binding,
        sample_count=_EXPECTED_SAMPLE_COUNT,
        family_count=_EXPECTED_FAMILY_COUNT,
        label_count=_EXPECTED_SAMPLE_COUNT,
        label_bytes_total=label_bytes_total,
        sample_split_counts=sample_split_counts,
        family_split_counts=family_split_counts,
        observed_class_inventory=inventory,
        test_specialist_records=0,
        optimizer_steps=0,
        complete_marker_present=False,
        verification_passed=True,
    )
