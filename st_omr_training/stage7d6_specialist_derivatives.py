"""Stage 7-D6 TRAIN/VALIDATION StaffSet + StructureSet derivative sidecars.

D6 does not train a model.  It consumes the frozen Synthetic Curriculum v1 only
after the D1 byte-integrity gate has passed, skips TEST before any specialist
artifact path or label field is derived, and writes small hash-addressed JSON
labels that reference the already-frozen PNGs by SHA-256.  No PNG or MusicXML
artifact is copied into the derivative set.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
from typing import Final

from .degradation import (
    CAIROSVG_PINNED_VERSION,
    DEGRADATION_VERSION,
    PILLOW_PINNED_VERSION,
    DegradationConfig,
    degradation_config_fingerprint,
)
from .renderer import RendererConfig, render_musicxml_svg, renderer_config_fingerprint
from .stage7d5_geometry import (
    STAGE7D5_GEOMETRY_VERSION,
    STAGE7D5_TRANSFORM_VERSION,
    PageGeometry,
    extract_staff_structure_geometry,
    map_page_geometry_to_final_png,
    render_musicxml_geometry_svg,
)
from .synthetic_curriculum_acceptance import (
    EXPECTED_BUILD_ID,
    EXPECTED_CONFIG_FINGERPRINT,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SOURCE_COMMIT,
    EXPECTED_TRANSPORT_SHA256,
)
from .synthetic_curriculum_corpus_gate import (
    EXPECTED_ARCHIVE_NAME,
    EXPECTED_ARCHIVE_SIZE_BYTES,
    SyntheticCurriculumCorpusReceipt,
    verify_stage7d_corpus,
)


STAGE7D6_VERSION: Final[str] = "stage7d6-staff-structure-derivatives-v1"
STAGE7D6_LABEL_SCHEMA: Final[str] = "stage7d6-staff-structure-label-v1"
STAGE7D6_MANIFEST_SCHEMA: Final[str] = "stage7d6-specialist-manifest-v1"
STAGE7D6_BUILD_SCHEMA: Final[str] = "stage7d6-specialist-build-v1"
STAGE7D6_RECEIPT_SCHEMA: Final[str] = "stage7d6-specialist-receipt-v1"
EXPECTED_DEVELOPMENT_SAMPLE_COUNTS: Final[dict[str, int]] = {
    "train": 1230,
    "validation": 153,
}
EXPECTED_DEVELOPMENT_FAMILY_COUNTS: Final[dict[str, int]] = {
    "train": 410,
    "validation": 51,
}
EXPECTED_DEVELOPMENT_SAMPLE_COUNT: Final[int] = 1383
EXPECTED_DEVELOPMENT_FAMILY_COUNT: Final[int] = 461
MAX_MANIFEST_BYTES: Final[int] = 32 * 1024 * 1024
MAX_BUILD_BYTES: Final[int] = 256 * 1024
MAX_LABEL_BYTES: Final[int] = 2 * 1024 * 1024
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ALLOWED_SPLITS = frozenset(EXPECTED_DEVELOPMENT_SAMPLE_COUNTS)
_TOP_LEVEL = frozenset({"manifest.json", "manifest.sha256", "build.json", "labels"})


class Stage7D6DerivativeError(RuntimeError):
    """Raised when D6 cannot derive or verify specialist labels safely."""


@dataclass(frozen=True, slots=True)
class _ManifestDegradedPage:
    """Minimal trusted Stage-4 lineage view required by the D5 mapper."""

    page_number: int
    source_musicxml_sha256: str
    renderer_config_fingerprint: str
    degradation_config_fingerprint: str
    config: DegradationConfig
    clean_width: int
    clean_height: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class Stage7D6DerivativeReceipt:
    source_commit: str
    source_build_id: str
    source_config_fingerprint: str
    source_manifest_sha256: str
    source_transport_sha256: str
    d5_geometry_version: str
    d5_transform_version: str
    manifest_sha256: str
    derivative_build_id: str
    sample_count: int
    family_count: int
    sample_split_counts: dict[str, int]
    family_split_counts: dict[str, int]
    label_count: int
    label_bytes_total: int
    artifact_binding_sha256: str
    test_specialist_records: int


def _fail(message: str) -> None:
    raise Stage7D6DerivativeError(message)


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
        raise Stage7D6DerivativeError("payload is not canonical JSON serializable") from exc


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be a regular non-symlink file")


def _require_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        _fail(f"{label} must be a regular non-symlink directory")


def _read_bounded(path: Path, maximum: int, label: str) -> bytes:
    _require_regular_file(path, label)
    size = path.stat().st_size
    if not 1 <= size <= maximum:
        _fail(f"{label} byte length is outside the D6 bound")
    return path.read_bytes()


def _load_canonical_json(path: Path, maximum: int, label: str) -> tuple[dict[str, object], bytes]:
    raw = _read_bounded(path, maximum, label)
    try:
        payload = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda value: _fail(f"non-finite JSON constant in {label}: {value}"),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D6DerivativeError(f"{label} is not valid ASCII JSON") from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        _fail(f"{label} must be canonical JSON object bytes")
    return payload, raw


def _require_sha(name: str, value: object) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        _fail(f"{name} must be lowercase SHA-256 hex")
    return value


def _require_id(name: str, value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        _fail(f"{name} violates the bounded identifier contract")
    return value


def _require_positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _fail(f"{name} must be a positive integer")
    return value


def _development_rows(samples: object) -> tuple[Mapping[str, object], ...]:
    """Return only development rows; TEST is skipped before any other key access.

    The deliberately small helper is regression-tested with hostile Mapping
    objects whose non-split fields raise if touched.  It is the D6 split seal.
    """

    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes, bytearray)):
        _fail("source manifest samples must be a sequence")
    rows: list[Mapping[str, object]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            _fail(f"source manifest sample[{index}] is not an object")
        split = sample.get("split")
        if split == "test":
            continue
        if split not in _ALLOWED_SPLITS:
            _fail(f"source manifest sample[{index}] has invalid development split")
        rows.append(sample)
    return tuple(rows)


def _verified_source_manifest(corpus_root: Path) -> tuple[dict[str, object], tuple[Mapping[str, object], ...]]:
    manifest, raw = _load_canonical_json(
        corpus_root / "manifest.json", MAX_MANIFEST_BYTES, "source manifest.json"
    )
    if sha256(raw).hexdigest() != EXPECTED_MANIFEST_SHA256:
        _fail("source manifest changed after D1 acceptance")
    if manifest.get("dataset_name") != "st-omr-synthetic-curriculum-v1":
        _fail("unexpected source dataset name")
    if manifest.get("dataset_version") != "v1":
        _fail("unexpected source dataset version")
    rows = _development_rows(manifest.get("samples"))
    if len(rows) != EXPECTED_DEVELOPMENT_SAMPLE_COUNT:
        _fail("development sample cardinality mismatch")
    return manifest, rows


def _verify_d1_receipt(receipt: SyntheticCurriculumCorpusReceipt) -> None:
    expected = {
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "build_id": EXPECTED_BUILD_ID,
        "config_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "transport_sha256": EXPECTED_TRANSPORT_SHA256,
        "transport_archive": EXPECTED_ARCHIVE_NAME,
        "archive_size_bytes": EXPECTED_ARCHIVE_SIZE_BYTES,
        "sample_count": 1536,
        "target_count": 512,
        "image_count": 1536,
    }
    for key, value in expected.items():
        if getattr(receipt, key) != value:
            _fail(f"D1 receipt {key} does not match frozen Synthetic Curriculum v1")
    if receipt.sample_split_counts != {"test": 153, **EXPECTED_DEVELOPMENT_SAMPLE_COUNTS}:
        _fail("D1 receipt sample split counts mismatch")
    if receipt.family_split_counts != {"test": 51, **EXPECTED_DEVELOPMENT_FAMILY_COUNTS}:
        _fail("D1 receipt family split counts mismatch")


def _config_from_row(row: Mapping[str, object]) -> DegradationConfig:
    raw = row.get("degradation_config")
    if not isinstance(raw, Mapping):
        _fail("development degradation_config must be an object")
    expected_keys = {
        "seed",
        "raster_width",
        "rotation_mdeg",
        "blur_milli",
        "noise_level",
        "brightness_milli",
        "contrast_milli",
        "jpeg_quality",
    }
    if set(raw) != expected_keys:
        _fail("development degradation_config keys mismatch")
    try:
        config = DegradationConfig(**dict(raw))
    except (TypeError, ValueError) as exc:
        raise Stage7D6DerivativeError("development degradation_config is invalid") from exc
    expected_fingerprint = _require_sha(
        "degradation_config_fingerprint", row.get("degradation_config_fingerprint")
    )
    if degradation_config_fingerprint(config) != expected_fingerprint:
        _fail("development degradation config fingerprint mismatch")
    return config


def _manifest_page_view(row: Mapping[str, object], config: DegradationConfig) -> _ManifestDegradedPage:
    return _ManifestDegradedPage(
        page_number=_require_positive_int("page_number", row.get("page_number")),
        source_musicxml_sha256=_require_sha(
            "source_musicxml_sha256", row.get("source_musicxml_sha256")
        ),
        renderer_config_fingerprint=_require_sha(
            "renderer_config_fingerprint", row.get("renderer_config_fingerprint")
        ),
        degradation_config_fingerprint=_require_sha(
            "degradation_config_fingerprint", row.get("degradation_config_fingerprint")
        ),
        config=config,
        clean_width=_require_positive_int("clean_width", row.get("clean_width")),
        clean_height=_require_positive_int("clean_height", row.get("clean_height")),
        width=_require_positive_int("width", row.get("width")),
        height=_require_positive_int("height", row.get("height")),
    )


def _page_by_number(pages: Sequence[object], page_number: int, label: str):
    matches = [page for page in pages if getattr(page, "page_number", None) == page_number]
    if len(matches) != 1:
        _fail(f"{label} page_number must resolve exactly once")
    return matches[0]


def _finite_number(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail("geometry coordinate must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _fail("geometry coordinate must be finite")
    return result


def _validate_point_dict(point: object, width: int, height: int) -> None:
    if not isinstance(point, Mapping) or set(point) != {"x", "y"}:
        _fail("geometry point shape mismatch")
    x = _finite_number(point.get("x"))
    y = _finite_number(point.get("y"))
    epsilon = 1e-5
    if not -epsilon <= x <= width + epsilon or not -epsilon <= y <= height + epsilon:
        _fail("geometry point lies outside final PNG bounds")


def _validate_line_dict(line: object, width: int, height: int) -> None:
    if not isinstance(line, Mapping) or set(line) != {"start", "end"}:
        _fail("geometry line shape mismatch")
    _validate_point_dict(line.get("start"), width, height)
    _validate_point_dict(line.get("end"), width, height)


def _validate_box_dict(box: object, width: int, height: int) -> None:
    if not isinstance(box, Mapping) or set(box) != {"x_min", "y_min", "x_max", "y_max"}:
        _fail("geometry box shape mismatch")
    x0 = _finite_number(box.get("x_min"))
    y0 = _finite_number(box.get("y_min"))
    x1 = _finite_number(box.get("x_max"))
    y1 = _finite_number(box.get("y_max"))
    epsilon = 1e-5
    if x1 <= x0 or y1 <= y0:
        _fail("geometry box must have positive area")
    if x0 < -epsilon or y0 < -epsilon or x1 > width + epsilon or y1 > height + epsilon:
        _fail("geometry box lies outside final PNG bounds")


def _validate_geometry_payload(
    geometry: object,
    *,
    width: int,
    height: int,
    source_musicxml_sha256: str,
    source_svg_sha256: str,
    renderer_fingerprint: str,
) -> None:
    if not isinstance(geometry, Mapping):
        _fail("geometry payload must be an object")
    if geometry.get("coordinate_space") != "final_png_pixels":
        _fail("D6 geometry must be in final PNG pixel coordinates")
    if geometry.get("source_musicxml_sha256") != source_musicxml_sha256:
        _fail("geometry MusicXML lineage mismatch")
    if geometry.get("base_renderer_config_fingerprint") != renderer_fingerprint:
        _fail("geometry renderer lineage mismatch")
    _require_sha("geometry_svg_sha256", geometry.get("geometry_svg_sha256"))
    _require_sha(
        "geometry_instrumentation_fingerprint",
        geometry.get("geometry_instrumentation_fingerprint"),
    )
    _require_sha("geometry_transform_fingerprint", geometry.get("geometry_transform_fingerprint"))

    staffs = geometry.get("staff_instances")
    systems = geometry.get("systems")
    measures = geometry.get("measures")
    if not isinstance(staffs, list) or not staffs:
        _fail("StaffSet must contain at least one graphical staff instance")
    if not isinstance(systems, list) or len(systems) != len(staffs):
        _fail("StructureSet system/staff cardinality mismatch")
    if not isinstance(measures, list) or not measures:
        _fail("StructureSet must contain at least one measure")

    staff_ids: set[str] = set()
    system_ids: set[str] = set()
    measure_numbers: set[int] = set()
    for staff in staffs:
        if not isinstance(staff, Mapping):
            _fail("staff instance must be an object")
        staff_id = _require_id("staff_instance_id", staff.get("staff_instance_id"))
        if staff_id in staff_ids:
            _fail("duplicate staff_instance_id")
        staff_ids.add(staff_id)
        _require_id("staff.system_id", staff.get("system_id"))
        lines = staff.get("five_staff_lines")
        if not isinstance(lines, list) or len(lines) != 5:
            _fail("StaffSet requires exactly five staff lines")
        for line in lines:
            _validate_line_dict(line, width, height)
        _validate_box_dict(staff.get("staff_instance_bbox"), width, height)
        spacing = _finite_number(staff.get("staff_spacing"))
        if spacing <= 0:
            _fail("staff_spacing must be positive")

    for system in systems:
        if not isinstance(system, Mapping):
            _fail("system must be an object")
        system_id = _require_id("system_id", system.get("system_id"))
        if system_id in system_ids:
            _fail("duplicate system_id")
        system_ids.add(system_id)
        if system.get("staff_instance_id") not in staff_ids:
            _fail("system references unknown staff_instance_id")
        _validate_box_dict(system.get("system_bbox"), width, height)
        numbers = system.get("measure_numbers")
        if not isinstance(numbers, list) or not numbers:
            _fail("system measure_numbers must be non-empty")
        for number in numbers:
            if not isinstance(number, int) or isinstance(number, bool) or number < 1:
                _fail("measure number must be a positive integer")

    for measure in measures:
        if not isinstance(measure, Mapping):
            _fail("measure must be an object")
        number = measure.get("measure_number")
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            _fail("measure_number must be positive")
        if number in measure_numbers:
            _fail("duplicate measure_number")
        measure_numbers.add(number)
        if measure.get("system_id") not in system_ids:
            _fail("measure references unknown system_id")
        if measure.get("meter_class") not in {"2/4", "3/4", "4/4"}:
            _fail("measure meter_class is outside V1")
        _validate_box_dict(measure.get("measure_bbox"), width, height)
        _validate_line_dict(measure.get("barline_segment"), width, height)
        for name in ("clef_g2_bbox", "meter_bbox"):
            value = measure.get(name)
            if value is not None:
                _validate_box_dict(value, width, height)

    covered = {
        number
        for system in systems
        for number in system.get("measure_numbers", [])
    }
    if covered != measure_numbers:
        _fail("system measure coverage does not match measure objects")


def _label_payload(row: Mapping[str, object], mapped: PageGeometry) -> dict[str, object]:
    sample_id = _require_sha("sample_id", row.get("sample_id"))
    family_id = _require_id("family_id", row.get("family_id"))
    split = row.get("split")
    if split not in _ALLOWED_SPLITS:
        _fail("label split must be train or validation")
    image_sha = _require_sha("png_sha256", row.get("png_sha256"))
    width = _require_positive_int("width", row.get("width"))
    height = _require_positive_int("height", row.get("height"))
    source_musicxml_sha = _require_sha(
        "source_musicxml_sha256", row.get("source_musicxml_sha256")
    )
    source_svg_sha = _require_sha("source_svg_sha256", row.get("source_svg_sha256"))
    renderer_fingerprint = _require_sha(
        "renderer_config_fingerprint", row.get("renderer_config_fingerprint")
    )
    degradation_fingerprint = _require_sha(
        "degradation_config_fingerprint", row.get("degradation_config_fingerprint")
    )
    geometry = asdict(mapped)
    _validate_geometry_payload(
        geometry,
        width=width,
        height=height,
        source_musicxml_sha256=source_musicxml_sha,
        source_svg_sha256=source_svg_sha,
        renderer_fingerprint=renderer_fingerprint,
    )
    return {
        "schema_version": STAGE7D6_LABEL_SCHEMA,
        "stage7d6_version": STAGE7D6_VERSION,
        "sample_id": sample_id,
        "family_id": family_id,
        "split": split,
        "page_number": _require_positive_int("page_number", row.get("page_number")),
        "image": {
            "png_sha256": image_sha,
            "width": width,
            "height": height,
            "mode": row.get("mode"),
            "image_format": row.get("image_format"),
        },
        "lineage": {
            "source_musicxml_sha256": source_musicxml_sha,
            "source_svg_sha256": source_svg_sha,
            "renderer_config_fingerprint": renderer_fingerprint,
            "degradation_config_fingerprint": degradation_fingerprint,
            "d5_geometry_version": STAGE7D5_GEOMETRY_VERSION,
            "d5_transform_version": STAGE7D5_TRANSFORM_VERSION,
        },
        "geometry": geometry,
    }


def _development_row_metadata(row: Mapping[str, object]) -> dict[str, object]:
    split = row.get("split")
    if split not in _ALLOWED_SPLITS:
        _fail("development record split must be train or validation")
    return {
        "sample_id": _require_sha("sample_id", row.get("sample_id")),
        "family_id": _require_id("family_id", row.get("family_id")),
        "split": split,
        "page_number": _require_positive_int("page_number", row.get("page_number")),
        "source_musicxml_sha256": _require_sha(
            "source_musicxml_sha256", row.get("source_musicxml_sha256")
        ),
        "source_svg_sha256": _require_sha("source_svg_sha256", row.get("source_svg_sha256")),
        "renderer_config_fingerprint": _require_sha(
            "renderer_config_fingerprint", row.get("renderer_config_fingerprint")
        ),
        "degradation_config_fingerprint": _require_sha(
            "degradation_config_fingerprint", row.get("degradation_config_fingerprint")
        ),
        "png_sha256": _require_sha("png_sha256", row.get("png_sha256")),
        "width": _require_positive_int("width", row.get("width")),
        "height": _require_positive_int("height", row.get("height")),
    }


def _profile_payload() -> dict[str, object]:
    return {
        "version": STAGE7D6_VERSION,
        "label_schema": STAGE7D6_LABEL_SCHEMA,
        "manifest_schema": STAGE7D6_MANIFEST_SCHEMA,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_build_id": EXPECTED_BUILD_ID,
        "source_config_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_transport_sha256": EXPECTED_TRANSPORT_SHA256,
        "d5_geometry_version": STAGE7D5_GEOMETRY_VERSION,
        "d5_transform_version": STAGE7D5_TRANSFORM_VERSION,
        "sample_split_counts": EXPECTED_DEVELOPMENT_SAMPLE_COUNTS,
        "family_split_counts": EXPECTED_DEVELOPMENT_FAMILY_COUNTS,
        "image_copy_policy": "hash-reference-only",
        "test_policy": "skip-before-specialist-field-or-path-derivation",
    }


def stage7d6_profile_fingerprint() -> str:
    return sha256(_canonical_json(_profile_payload())).hexdigest()


def _write_bytes(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        _fail(f"refusing to overwrite derivative artifact: {path.name}")
    path.write_bytes(data)


def _source_row_index(rows: tuple[Mapping[str, object], ...]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    family_split: dict[str, str] = {}
    sample_counts: Counter[str] = Counter()
    families: dict[str, set[str]] = {"train": set(), "validation": set()}
    for row in rows:
        meta = _development_row_metadata(row)
        sample_id = str(meta["sample_id"])
        if sample_id in result:
            _fail("duplicate development sample_id in source manifest")
        family = str(meta["family_id"])
        split = str(meta["split"])
        prior = family_split.setdefault(family, split)
        if prior != split:
            _fail("development family crosses train/validation split")
        sample_counts[split] += 1
        families[split].add(family)
        result[sample_id] = meta
    if dict(sorted(sample_counts.items())) != EXPECTED_DEVELOPMENT_SAMPLE_COUNTS:
        _fail("source development sample split counts mismatch")
    family_counts = {split: len(values) for split, values in families.items()}
    if family_counts != EXPECTED_DEVELOPMENT_FAMILY_COUNTS:
        _fail("source development family split counts mismatch")
    if len(family_split) != EXPECTED_DEVELOPMENT_FAMILY_COUNT:
        _fail("source development family count mismatch")
    return result


def _build_into(corpus_root: Path, output_root: Path, rows: tuple[Mapping[str, object], ...]) -> None:
    labels_root = output_root / "labels"
    labels_root.mkdir(parents=True)
    renderer_config = RendererConfig()
    frozen_renderer_fingerprint = renderer_config_fingerprint(renderer_config)

    family_cache: dict[str, tuple[object, tuple[PageGeometry, ...]]] = {}
    family_lineage: dict[str, tuple[str, str, str]] = {}
    records: list[dict[str, object]] = []
    label_hashes: set[str] = set()

    for row in rows:
        meta = _development_row_metadata(row)
        family_id = str(meta["family_id"])
        target_sha = str(meta["source_musicxml_sha256"])
        source_svg_sha = str(meta["source_svg_sha256"])
        renderer_fingerprint = str(meta["renderer_config_fingerprint"])
        if renderer_fingerprint != frozen_renderer_fingerprint:
            _fail("development renderer fingerprint differs from frozen D5 renderer")

        lineage = (target_sha, source_svg_sha, renderer_fingerprint)
        prior_lineage = family_lineage.setdefault(family_id, lineage)
        if prior_lineage != lineage:
            _fail("family derivatives disagree on MusicXML/SVG/renderer lineage")

        cached = family_cache.get(family_id)
        if cached is None:
            target_path = corpus_root / "targets" / f"{target_sha}.musicxml"
            target_bytes = _read_bounded(target_path, 4 * 1024 * 1024, "development MusicXML")
            if sha256(target_bytes).hexdigest() != target_sha:
                _fail("development MusicXML hash mismatch")
            base_render = render_musicxml_svg(target_bytes, renderer_config)
            geometry_render = render_musicxml_geometry_svg(target_bytes, renderer_config)
            if base_render.source_musicxml_sha256 != target_sha:
                _fail("base renderer MusicXML lineage mismatch")
            if geometry_render.source_musicxml_sha256 != target_sha:
                _fail("geometry renderer MusicXML lineage mismatch")
            if base_render.config_fingerprint != renderer_fingerprint:
                _fail("base renderer configuration changed")
            if geometry_render.base_renderer_config_fingerprint != renderer_fingerprint:
                _fail("geometry renderer configuration changed")
            geometry_pages = extract_staff_structure_geometry(geometry_render, target_bytes)
            cached = (base_render, geometry_pages)
            family_cache[family_id] = cached

        base_render, geometry_pages = cached
        page_number = int(meta["page_number"])
        base_page = _page_by_number(base_render.pages, page_number, "base render")
        if base_page.sha256 != source_svg_sha:
            _fail("pinned Verovio source SVG hash does not match frozen corpus lineage")
        svg_geometry = _page_by_number(geometry_pages, page_number, "geometry render")

        config = _config_from_row(row)
        view = _manifest_page_view(row, config)
        mapped = map_page_geometry_to_final_png(svg_geometry, view)
        label_payload = _label_payload(row, mapped)
        label_bytes = _canonical_json(label_payload)
        if not 1 <= len(label_bytes) <= MAX_LABEL_BYTES:
            _fail("specialist label byte length is outside the D6 bound")
        label_sha = sha256(label_bytes).hexdigest()
        if label_sha in label_hashes:
            _fail("duplicate specialist label SHA-256")
        label_hashes.add(label_sha)
        _write_bytes(labels_root / f"{label_sha}.json", label_bytes)

        records.append(
            {
                "sample_id": meta["sample_id"],
                "family_id": family_id,
                "split": meta["split"],
                "page_number": page_number,
                "png_sha256": meta["png_sha256"],
                "label_sha256": label_sha,
            }
        )

    records.sort(key=lambda item: str(item["sample_id"]))
    manifest_payload = {
        "schema_version": STAGE7D6_MANIFEST_SCHEMA,
        "stage7d6_version": STAGE7D6_VERSION,
        "profile_fingerprint": stage7d6_profile_fingerprint(),
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_build_id": EXPECTED_BUILD_ID,
        "source_config_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_transport_sha256": EXPECTED_TRANSPORT_SHA256,
        "d5_geometry_version": STAGE7D5_GEOMETRY_VERSION,
        "d5_transform_version": STAGE7D5_TRANSFORM_VERSION,
        "split_policy": "family-exclusive-train-validation-only-v1",
        "image_policy": "reference-frozen-source-png-by-sha256",
        "records": records,
    }
    manifest_bytes = _canonical_json(manifest_payload)
    manifest_sha = sha256(manifest_bytes).hexdigest()
    _write_bytes(output_root / "manifest.json", manifest_bytes)
    _write_bytes(
        output_root / "manifest.sha256",
        f"{manifest_sha}  manifest.json\n".encode("ascii"),
    )

    build_payload = {
        "schema_version": STAGE7D6_BUILD_SCHEMA,
        "stage7d6_version": STAGE7D6_VERSION,
        "derivative_build_id": stage7d6_profile_fingerprint(),
        "manifest_sha256": manifest_sha,
        "sample_count": len(records),
        "family_count": len({str(record["family_id"]) for record in records}),
        "label_count": len(label_hashes),
        "sample_split_counts": dict(sorted(Counter(str(record["split"]) for record in records).items())),
        "family_split_counts": dict(
            sorted(
                Counter(
                    split
                    for split, families in (
                        (
                            candidate,
                            {
                                str(record["family_id"])
                                for record in records
                                if record["split"] == candidate
                            },
                        )
                        for candidate in ("train", "validation")
                    )
                    for _family in families
                ).items()
            )
        ),
        "layout": {
            "manifest": "manifest.json",
            "labels": "labels/<label_sha256>.json",
            "source_images": "external frozen corpus images/<png_sha256>.png",
        },
    }
    # Compute family counts explicitly; keep canonical payload simple and auditable.
    build_payload["family_split_counts"] = {
        split: len(
            {
                str(record["family_id"])
                for record in records
                if record["split"] == split
            }
        )
        for split in ("train", "validation")
    }
    _write_bytes(output_root / "build.json", _canonical_json(build_payload))


def _validate_label_against_source(
    label: dict[str, object],
    source: dict[str, object],
    record: Mapping[str, object],
) -> None:
    if set(label) != {
        "schema_version",
        "stage7d6_version",
        "sample_id",
        "family_id",
        "split",
        "page_number",
        "image",
        "lineage",
        "geometry",
    }:
        _fail("specialist label top-level keys mismatch")
    if label.get("schema_version") != STAGE7D6_LABEL_SCHEMA:
        _fail("specialist label schema mismatch")
    if label.get("stage7d6_version") != STAGE7D6_VERSION:
        _fail("specialist label version mismatch")
    for key in ("sample_id", "family_id", "split", "page_number"):
        if label.get(key) != source.get(key):
            _fail(f"specialist label {key} differs from source manifest")

    image = label.get("image")
    if not isinstance(image, Mapping):
        _fail("specialist label image lineage must be an object")
    image_expected = {
        "png_sha256": source["png_sha256"],
        "width": source["width"],
        "height": source["height"],
        "mode": "L",
        "image_format": "png",
    }
    if dict(image) != image_expected:
        _fail("specialist label image lineage mismatch")

    lineage = label.get("lineage")
    if not isinstance(lineage, Mapping):
        _fail("specialist label lineage must be an object")
    expected_lineage = {
        "source_musicxml_sha256": source["source_musicxml_sha256"],
        "source_svg_sha256": source["source_svg_sha256"],
        "renderer_config_fingerprint": source["renderer_config_fingerprint"],
        "degradation_config_fingerprint": source["degradation_config_fingerprint"],
        "d5_geometry_version": STAGE7D5_GEOMETRY_VERSION,
        "d5_transform_version": STAGE7D5_TRANSFORM_VERSION,
    }
    if dict(lineage) != expected_lineage:
        _fail("specialist label provenance mismatch")

    if record.get("sample_id") != source["sample_id"]:
        _fail("derivative record/source sample mismatch")
    if record.get("png_sha256") != source["png_sha256"]:
        _fail("derivative record/source image mismatch")
    _validate_geometry_payload(
        label.get("geometry"),
        width=int(source["width"]),
        height=int(source["height"]),
        source_musicxml_sha256=str(source["source_musicxml_sha256"]),
        source_svg_sha256=str(source["source_svg_sha256"]),
        renderer_fingerprint=str(source["renderer_config_fingerprint"]),
    )


def verify_stage7d6_derivatives(
    corpus_root: str | Path,
    derivative_root: str | Path,
) -> Stage7D6DerivativeReceipt:
    """Independently verify D6 sidecars without trusting builder Python objects."""

    if not isinstance(corpus_root, (str, Path)) or not isinstance(derivative_root, (str, Path)):
        raise TypeError("corpus_root and derivative_root must be str or pathlib.Path")
    source_root = Path(corpus_root)
    root = Path(derivative_root)
    _require_directory(root, "D6 derivative root")
    if {entry.name for entry in root.iterdir()} != _TOP_LEVEL:
        _fail("D6 derivative top-level layout mismatch")
    _require_directory(root / "labels", "D6 labels")

    _source_manifest, rows = _verified_source_manifest(source_root)
    source_index = _source_row_index(rows)

    manifest, manifest_bytes = _load_canonical_json(
        root / "manifest.json", MAX_MANIFEST_BYTES, "D6 manifest.json"
    )
    manifest_sha = sha256(manifest_bytes).hexdigest()
    checksum = _read_bounded(root / "manifest.sha256", 256, "D6 manifest.sha256")
    if checksum != f"{manifest_sha}  manifest.json\n".encode("ascii"):
        _fail("D6 manifest.sha256 content mismatch")
    expected_header = {
        "schema_version": STAGE7D6_MANIFEST_SCHEMA,
        "stage7d6_version": STAGE7D6_VERSION,
        "profile_fingerprint": stage7d6_profile_fingerprint(),
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_build_id": EXPECTED_BUILD_ID,
        "source_config_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_transport_sha256": EXPECTED_TRANSPORT_SHA256,
        "d5_geometry_version": STAGE7D5_GEOMETRY_VERSION,
        "d5_transform_version": STAGE7D5_TRANSFORM_VERSION,
        "split_policy": "family-exclusive-train-validation-only-v1",
        "image_policy": "reference-frozen-source-png-by-sha256",
    }
    if set(manifest) != set(expected_header) | {"records"}:
        _fail("D6 manifest keys mismatch")
    for key, value in expected_header.items():
        if manifest.get(key) != value:
            _fail(f"D6 manifest {key} mismatch")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_DEVELOPMENT_SAMPLE_COUNT:
        _fail("D6 record count mismatch")
    if records != sorted(records, key=lambda item: str(item.get("sample_id")) if isinstance(item, Mapping) else ""):
        _fail("D6 records must be sorted by sample_id")

    labels_on_disk = {entry.name for entry in (root / "labels").iterdir()}
    expected_label_names: set[str] = set()
    seen_samples: set[str] = set()
    sample_counts: Counter[str] = Counter()
    family_split: dict[str, str] = {}
    label_bytes_total = 0
    binding_rows: list[str] = []

    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {
            "sample_id",
            "family_id",
            "split",
            "page_number",
            "png_sha256",
            "label_sha256",
        }:
            _fail(f"D6 record[{index}] shape mismatch")
        split = record.get("split")
        if split == "test" or split not in _ALLOWED_SPLITS:
            _fail("D6 derivative manifest contains a forbidden split")
        sample_id = _require_sha("record.sample_id", record.get("sample_id"))
        if sample_id in seen_samples:
            _fail("duplicate D6 sample_id")
        seen_samples.add(sample_id)
        source = source_index.get(sample_id)
        if source is None:
            _fail("D6 record does not correspond to a development source sample")
        family = _require_id("record.family_id", record.get("family_id"))
        if family != source["family_id"] or split != source["split"]:
            _fail("D6 record family/split differs from source sample")
        if record.get("page_number") != source["page_number"]:
            _fail("D6 record page number differs from source sample")
        image_sha = _require_sha("record.png_sha256", record.get("png_sha256"))
        if image_sha != source["png_sha256"]:
            _fail("D6 record image SHA differs from source sample")
        label_sha = _require_sha("record.label_sha256", record.get("label_sha256"))
        label_name = f"{label_sha}.json"
        expected_label_names.add(label_name)
        label, raw_label = _load_canonical_json(
            root / "labels" / label_name, MAX_LABEL_BYTES, f"D6 label {label_name}"
        )
        if sha256(raw_label).hexdigest() != label_sha:
            _fail("D6 label SHA-256 does not match filename")
        _validate_label_against_source(label, source, record)
        label_bytes_total += len(raw_label)
        sample_counts[str(split)] += 1
        prior = family_split.setdefault(family, str(split))
        if prior != split:
            _fail("D6 family crosses development splits")
        binding_rows.append(f"{sample_id}:{image_sha}:{label_sha}:{len(raw_label)}")

    if labels_on_disk != expected_label_names:
        _fail("D6 label filenames do not exactly match derivative manifest")
    if len(labels_on_disk) != EXPECTED_DEVELOPMENT_SAMPLE_COUNT:
        _fail("D6 label cardinality mismatch")
    if dict(sorted(sample_counts.items())) != EXPECTED_DEVELOPMENT_SAMPLE_COUNTS:
        _fail("D6 sample split counts mismatch")
    family_counts = dict(sorted(Counter(family_split.values()).items()))
    if family_counts != EXPECTED_DEVELOPMENT_FAMILY_COUNTS:
        _fail("D6 family split counts mismatch")
    if len(family_split) != EXPECTED_DEVELOPMENT_FAMILY_COUNT:
        _fail("D6 family count mismatch")

    build, _build_bytes = _load_canonical_json(root / "build.json", MAX_BUILD_BYTES, "D6 build.json")
    expected_build = {
        "schema_version": STAGE7D6_BUILD_SCHEMA,
        "stage7d6_version": STAGE7D6_VERSION,
        "derivative_build_id": stage7d6_profile_fingerprint(),
        "manifest_sha256": manifest_sha,
        "sample_count": EXPECTED_DEVELOPMENT_SAMPLE_COUNT,
        "family_count": EXPECTED_DEVELOPMENT_FAMILY_COUNT,
        "label_count": EXPECTED_DEVELOPMENT_SAMPLE_COUNT,
        "sample_split_counts": EXPECTED_DEVELOPMENT_SAMPLE_COUNTS,
        "family_split_counts": EXPECTED_DEVELOPMENT_FAMILY_COUNTS,
        "layout": {
            "manifest": "manifest.json",
            "labels": "labels/<label_sha256>.json",
            "source_images": "external frozen corpus images/<png_sha256>.png",
        },
    }
    if build != expected_build:
        _fail("D6 build.json does not match independently derived expectations")

    artifact_binding = sha256(("\n".join(sorted(binding_rows)) + "\n").encode("ascii")).hexdigest()
    return Stage7D6DerivativeReceipt(
        source_commit=EXPECTED_SOURCE_COMMIT,
        source_build_id=EXPECTED_BUILD_ID,
        source_config_fingerprint=EXPECTED_CONFIG_FINGERPRINT,
        source_manifest_sha256=EXPECTED_MANIFEST_SHA256,
        source_transport_sha256=EXPECTED_TRANSPORT_SHA256,
        d5_geometry_version=STAGE7D5_GEOMETRY_VERSION,
        d5_transform_version=STAGE7D5_TRANSFORM_VERSION,
        manifest_sha256=manifest_sha,
        derivative_build_id=stage7d6_profile_fingerprint(),
        sample_count=EXPECTED_DEVELOPMENT_SAMPLE_COUNT,
        family_count=EXPECTED_DEVELOPMENT_FAMILY_COUNT,
        sample_split_counts=EXPECTED_DEVELOPMENT_SAMPLE_COUNTS,
        family_split_counts=EXPECTED_DEVELOPMENT_FAMILY_COUNTS,
        label_count=EXPECTED_DEVELOPMENT_SAMPLE_COUNT,
        label_bytes_total=label_bytes_total,
        artifact_binding_sha256=artifact_binding,
        test_specialist_records=0,
    )


def build_stage7d6_derivatives(
    corpus_root: str | Path,
    transport_archive: str | Path,
    output_root: str | Path,
) -> Stage7D6DerivativeReceipt:
    """Build accepted TRAIN/VALIDATION specialist sidecars and verify them."""

    if not all(isinstance(value, (str, Path)) for value in (corpus_root, transport_archive, output_root)):
        raise TypeError("corpus_root, transport_archive and output_root must be str or pathlib.Path")
    source_root = Path(corpus_root)
    final_root = Path(output_root)
    try:
        source_resolved = source_root.resolve()
        output_resolved = final_root.resolve()
        if output_resolved == source_resolved or source_resolved in output_resolved.parents:
            _fail("D6 output must be outside the frozen source corpus")
    except OSError as exc:
        raise Stage7D6DerivativeError("unable to resolve D6 source/output paths") from exc
    if final_root.exists() or final_root.is_symlink():
        _fail("D6 output root must be fresh")
    temporary = final_root.with_name(f".{final_root.name}.stage7d6-tmp")
    if temporary.exists() or temporary.is_symlink():
        _fail("D6 temporary output path already exists")

    receipt = verify_stage7d_corpus(source_root, transport_archive)
    _verify_d1_receipt(receipt)
    _source_manifest, rows = _verified_source_manifest(source_root)
    _source_row_index(rows)

    temporary.mkdir(parents=True)
    try:
        _build_into(source_root, temporary, rows)
        verified = verify_stage7d6_derivatives(source_root, temporary)
        temporary.rename(final_root)
        return verified
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def canonical_stage7d6_receipt(receipt: Stage7D6DerivativeReceipt) -> bytes:
    if not isinstance(receipt, Stage7D6DerivativeReceipt):
        raise TypeError("receipt must be Stage7D6DerivativeReceipt")
    return _canonical_json({"schema_version": STAGE7D6_RECEIPT_SCHEMA, **asdict(receipt)}) + b"\n"
