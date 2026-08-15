"""Stage 7-D12 deterministic NoteHead/Rest/Accidental derivative builder.

D12 is a data-only gate.  This module consumes the exact accepted Stage 7-D6
TRAIN/VALIDATION surface, re-hashes every referenced source PNG and D6 label,
replays pinned Verovio symbol geometry, maps it through the accepted final-PNG
transform, and persists one canonical symbol label per development image.

The builder deliberately does not write ``COMPLETE``.  D12 completion remains
blocked until a separate persisted-bundle verifier independently reopens the
output in the next controlled step.  No model, checkpoint, optimizer, backward
pass, TEST derivation, or learned prediction is used here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Final

from . import _stage7d5_geometry_v1 as _d5
from .degradation import DegradationConfig, degradation_config_fingerprint
from .renderer import RendererConfig, render_musicxml_svg, renderer_config_fingerprint
from .stage7d12_symbol_geometry import (
    STAGE7D12_SYMBOL_GEOMETRY_VERSION,
    SymbolGeometryPage,
    extract_symbol_geometry,
)
from .stage7d12_symbol_gt_contract import (
    ACCIDENTAL_CLASSES,
    EXPECTED_DEVELOPMENT_FAMILY_COUNTS,
    EXPECTED_DEVELOPMENT_SAMPLE_COUNTS,
    NOTEHEAD_FILL_CLASSES,
    OPTIMIZER_STEPS,
    REST_CLASSES,
    TEST_SPECIALIST_RECORDS,
    development_split,
    stage7d12_contract_fingerprint,
)
from .stage7d5_geometry import (
    AxisAlignedBox,
    Point2D,
    STAGE7D5_TRANSFORM_VERSION,
    render_musicxml_geometry_svg,
)
from .stage7d6_specialist_derivatives import (
    STAGE7D6_LABEL_SCHEMA,
    STAGE7D6_VERSION,
    verify_stage7d6_derivatives,
)


STAGE7D12_DERIVATIVE_VERSION: Final[str] = "stage7d12-symbol-derivatives-v1"
STAGE7D12_LABEL_SCHEMA: Final[str] = "stage7d12-symbol-label-v1"
STAGE7D12_MANIFEST_SCHEMA: Final[str] = "stage7d12-symbol-manifest-v1"
STAGE7D12_BUILD_SCHEMA: Final[str] = "stage7d12-symbol-build-v1"

# Exact accepted D6 identity already frozen by D7.  D12 copies these values into
# its neutral data package instead of importing the D7 training module (which
# would unnecessarily import torch/model code into this data-only stage).
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
_MAX_MANIFEST_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_LABEL_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_TARGET_BYTES: Final[int] = 4 * 1024 * 1024
_MAX_IMAGE_BYTES: Final[int] = 32 * 1024 * 1024
_MAX_BUILD_BYTES: Final[int] = 2 * 1024 * 1024
_EXPECTED_SAMPLE_COUNT: Final[int] = sum(EXPECTED_DEVELOPMENT_SAMPLE_COUNTS.values())
_EXPECTED_FAMILY_COUNT: Final[int] = sum(EXPECTED_DEVELOPMENT_FAMILY_COUNTS.values())


class Stage7D12DerivativeError(RuntimeError):
    """Raised when the D12 derivative boundary cannot be proven safely."""


def _fail(message: str) -> None:
    raise Stage7D12DerivativeError(message)


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
        raise Stage7D12DerivativeError("payload is not canonical JSON") from exc


def _hex64(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        _fail(f"{name} must be lowercase SHA-256 hex")
    return value


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or len(value) > 256:
        _fail(f"{name} must be bounded non-empty ASCII")
    return value


def _positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _fail(f"{name} must be a positive integer")
    return value


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
        _fail(f"{name} byte length is outside the D12 bound")
    return path.read_bytes()


def _read_canonical_json(
    path: Path,
    maximum: int,
    name: str,
) -> tuple[dict[str, object], bytes]:
    raw = _read_bounded(path, maximum, name)
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda token: _fail(
                f"non-finite JSON constant in {name}: {token}"
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage7D12DerivativeError(f"{name} is not valid ASCII JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != raw:
        _fail(f"{name} must be canonical JSON object bytes")
    return value, raw


def development_rows(rows: object) -> tuple[Mapping[str, object], ...]:
    """Expose only TRAIN/VALIDATION, touching only ``split`` on TEST rows."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        _fail("source samples must be a sequence")
    accepted: list[Mapping[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"source sample[{index}] must be a mapping")
        try:
            split = development_split(row)
        except ValueError as exc:
            raise Stage7D12DerivativeError(
                f"source sample[{index}] has invalid development split"
            ) from exc
        if split is None:
            continue
        accepted.append(row)
    return tuple(accepted)


@dataclass(frozen=True, slots=True)
class _DegradedPageView:
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
class Stage7D12DerivativeReceipt:
    derivative_build_id: str
    manifest_sha256: str
    sample_count: int
    family_count: int
    sample_split_counts: dict[str, int]
    family_split_counts: dict[str, int]
    label_count: int
    label_bytes_total: int
    artifact_binding_sha256: str
    observed_class_inventory: dict[str, dict[str, dict[str, int]]]
    test_specialist_records: int
    optimizer_steps: int
    complete_marker_written: bool


def _profile_payload() -> dict[str, object]:
    return {
        "version": STAGE7D12_DERIVATIVE_VERSION,
        "label_schema": STAGE7D12_LABEL_SCHEMA,
        "manifest_schema": STAGE7D12_MANIFEST_SCHEMA,
        "contract_fingerprint": stage7d12_contract_fingerprint(),
        "symbol_geometry_version": STAGE7D12_SYMBOL_GEOMETRY_VERSION,
        "d5_transform_version": STAGE7D5_TRANSFORM_VERSION,
        "accepted_d6": {
            "version": STAGE7D6_VERSION,
            "label_schema": STAGE7D6_LABEL_SCHEMA,
            "derivative_build_id": EXPECTED_D6_DERIVATIVE_BUILD_ID,
            "manifest_sha256": EXPECTED_D6_MANIFEST_SHA256,
            "artifact_binding_sha256": EXPECTED_D6_ARTIFACT_BINDING_SHA256,
            "label_count": EXPECTED_D6_LABEL_COUNT,
            "family_count": EXPECTED_D6_FAMILY_COUNT,
        },
        "source_sample_counts": EXPECTED_DEVELOPMENT_SAMPLE_COUNTS,
        "source_family_counts": EXPECTED_DEVELOPMENT_FAMILY_COUNTS,
        "test_specialist_records": TEST_SPECIALIST_RECORDS,
        "optimizer_steps": OPTIMIZER_STEPS,
        "completion_policy": "no-COMPLETE-until-independent-persisted-verifier",
    }


def stage7d12_derivative_profile_fingerprint() -> str:
    return sha256(_canonical_json(_profile_payload())).hexdigest()


def _source_index(
    corpus_root: Path,
) -> tuple[dict[str, Mapping[str, object]], tuple[Mapping[str, object], ...]]:
    manifest, _raw = _read_canonical_json(
        corpus_root / "manifest.json",
        _MAX_MANIFEST_BYTES,
        "source manifest.json",
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
            _fail("source development split drifted after D12 seal")
        assert isinstance(split, str)
        prior = family_split.setdefault(family_id, split)
        if prior != split:
            _fail("source family crosses TRAIN/VALIDATION")
        sample_counts[split] += 1
        index[sample_id] = row

    if dict(sorted(sample_counts.items())) != EXPECTED_DEVELOPMENT_SAMPLE_COUNTS:
        _fail("source development sample split counts mismatch")
    family_counts = dict(sorted(Counter(family_split.values()).items()))
    if family_counts != EXPECTED_DEVELOPMENT_FAMILY_COUNTS:
        _fail("source development family split counts mismatch")
    if len(family_split) != _EXPECTED_FAMILY_COUNT:
        _fail("source development family cardinality mismatch")
    return index, rows


def _accepted_d6_index(
    corpus_root: Path,
    d6_root: Path,
) -> dict[str, Mapping[str, object]]:
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
        d6_root / "manifest.json",
        _MAX_MANIFEST_BYTES,
        "accepted D6 manifest.json",
    )
    if sha256(raw).hexdigest() != EXPECTED_D6_MANIFEST_SHA256:
        _fail("accepted D6 manifest hash mismatch")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != _EXPECTED_SAMPLE_COUNT:
        _fail("accepted D6 record cardinality mismatch")
    index: dict[str, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            _fail("accepted D6 record must be an object")
        split = record.get("split")
        if split not in _ALLOWED_SPLITS:
            _fail("accepted D6 manifest contains forbidden split")
        sample_id = _hex64("D6 record.sample_id", record.get("sample_id"))
        if sample_id in index:
            _fail("duplicate accepted D6 sample_id")
        index[sample_id] = record
    return index


def _config_from_source(row: Mapping[str, object]) -> DegradationConfig:
    raw = row.get("degradation_config")
    if not isinstance(raw, Mapping):
        _fail("source degradation_config must be an object")
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
        _fail("source degradation_config keys mismatch")
    try:
        config = DegradationConfig(**dict(raw))
    except (TypeError, ValueError) as exc:
        raise Stage7D12DerivativeError("invalid source degradation config") from exc
    expected = _hex64(
        "source.degradation_config_fingerprint",
        row.get("degradation_config_fingerprint"),
    )
    if degradation_config_fingerprint(config) != expected:
        _fail("source degradation config fingerprint mismatch")
    return config


def _degraded_view(row: Mapping[str, object]) -> _DegradedPageView:
    config = _config_from_source(row)
    return _DegradedPageView(
        page_number=_positive_int("source.page_number", row.get("page_number")),
        source_musicxml_sha256=_hex64(
            "source.source_musicxml_sha256",
            row.get("source_musicxml_sha256"),
        ),
        renderer_config_fingerprint=_hex64(
            "source.renderer_config_fingerprint",
            row.get("renderer_config_fingerprint"),
        ),
        degradation_config_fingerprint=_hex64(
            "source.degradation_config_fingerprint",
            row.get("degradation_config_fingerprint"),
        ),
        config=config,
        clean_width=_positive_int("source.clean_width", row.get("clean_width")),
        clean_height=_positive_int("source.clean_height", row.get("clean_height")),
        width=_positive_int("source.width", row.get("width")),
        height=_positive_int("source.height", row.get("height")),
    )


def _page_by_number(pages: Sequence[object], page_number: int, name: str) -> object:
    matches = [
        page for page in pages if getattr(page, "page_number", None) == page_number
    ]
    if len(matches) != 1:
        _fail(f"{name} page_number must resolve exactly once")
    return matches[0]


def _box_dict(box: AxisAlignedBox) -> dict[str, float]:
    return {
        "x_min": float(box.x_min),
        "y_min": float(box.y_min),
        "x_max": float(box.x_max),
        "y_max": float(box.y_max),
    }


def _point_dict(point: Point2D) -> dict[str, float]:
    return {"x": float(point.x), "y": float(point.y)}


def _inside_image(box: AxisAlignedBox, width: int, height: int) -> bool:
    epsilon = 1e-5
    return (
        box.x_min >= -epsilon
        and box.y_min >= -epsilon
        and box.x_max <= width + epsilon
        and box.y_max <= height + epsilon
    )


def _box_inside(inner: AxisAlignedBox, outer: AxisAlignedBox) -> bool:
    epsilon = 1e-5
    return (
        inner.x_min >= outer.x_min - epsilon
        and inner.y_min >= outer.y_min - epsilon
        and inner.x_max <= outer.x_max + epsilon
        and inner.y_max <= outer.y_max + epsilon
    )


def map_symbol_page_to_final_png(
    page: SymbolGeometryPage,
    degraded_page: object,
) -> tuple[list[dict[str, object]], str]:
    """Map one D12 SVG-space symbol page through the accepted D5 transform."""

    if not isinstance(page, SymbolGeometryPage):
        raise TypeError("page must be SymbolGeometryPage")
    if page.coordinate_space != "pinned_verovio_svg":
        _fail("D12 symbol page must start in pinned Verovio SVG space")
    required = (
        "page_number",
        "source_musicxml_sha256",
        "renderer_config_fingerprint",
        "degradation_config_fingerprint",
        "config",
        "clean_width",
        "clean_height",
        "width",
        "height",
    )
    for name in required:
        if not hasattr(degraded_page, name):
            _fail(f"degraded_page is missing {name}")
    if degraded_page.page_number != page.page_number:
        _fail("symbol/degraded page_number provenance mismatch")
    if degraded_page.source_musicxml_sha256 != page.source_musicxml_sha256:
        _fail("symbol/degraded MusicXML provenance mismatch")
    if degraded_page.renderer_config_fingerprint != page.base_renderer_config_fingerprint:
        _fail("symbol/degraded renderer provenance mismatch")

    x0, y0, vb_width, vb_height = page.view_box
    clean_width = degraded_page.clean_width
    clean_height = degraded_page.clean_height
    width = degraded_page.width
    height = degraded_page.height
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (clean_width, clean_height, width, height)
    ):
        _fail("raster dimensions must be positive integers")
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in (x0, y0, vb_width, vb_height)
    ) or vb_width <= 0 or vb_height <= 0:
        _fail("symbol viewBox must be finite and positive")

    scale = clean_width / vb_width
    if abs(vb_height * scale - clean_height) > 1.0:
        _fail("clean raster dimensions drifted from symbol viewBox")

    def svg_to_clean(point: Point2D) -> Point2D:
        return Point2D((point.x - x0) * scale, (point.y - y0) * scale)

    rotation_mdeg = getattr(degraded_page.config, "rotation_mdeg", None)
    if not isinstance(rotation_mdeg, int) or isinstance(rotation_mdeg, bool):
        _fail("rotation_mdeg must be an integer")
    if rotation_mdeg:
        out_width, out_height, reverse = _d5._pillow_rotation_reverse_affine(
            clean_width,
            clean_height,
            rotation_mdeg / 1000.0,
        )
        if (out_width, out_height) != (width, height):
            _fail("Pillow rotation replay dimensions mismatch source PNG")

        def mapper(point: Point2D) -> Point2D:
            return _d5._source_to_rotated(svg_to_clean(point), reverse)

    else:
        if (clean_width, clean_height) != (width, height):
            _fail("photometric-only derivative changed geometry dimensions")
        mapper = svg_to_clean

    transform_payload = {
        "version": STAGE7D5_TRANSFORM_VERSION,
        "geometry_svg_sha256": page.geometry_svg_sha256,
        "geometry_view_box": page.view_box,
        "clean_size": [clean_width, clean_height],
        "final_size": [width, height],
        "degradation_config_fingerprint": degraded_page.degradation_config_fingerprint,
        "rotation_mdeg": rotation_mdeg,
    }
    transform_fingerprint = _d5._canonical_sha256(transform_payload)

    measures: list[dict[str, object]] = []
    seen_events: set[tuple[str, str]] = set()
    for measure in page.measures:
        measure_box = _d5._map_box(measure.measure_bbox, mapper)
        if not _inside_image(measure_box, width, height):
            _fail("mapped measure bbox lies outside final PNG")

        noteheads: list[dict[str, object]] = []
        rests: list[dict[str, object]] = []
        accidentals: list[dict[str, object]] = []
        for record in measure.noteheads:
            box = _d5._map_box(record.bbox, mapper)
            center = mapper(record.center) if record.center is not None else None
            if center is None:
                _fail("NoteHeadSet record lost required center")
            if not _inside_image(box, width, height) or not _box_inside(box, measure_box):
                _fail("mapped notehead bbox is outside image/owning measure")
            if not (
                box.x_min - 1e-5 <= center.x <= box.x_max + 1e-5
                and box.y_min - 1e-5 <= center.y <= box.y_max + 1e-5
            ):
                _fail("mapped notehead center lies outside mapped bbox")
            key = ("notehead", record.canonical_event_id)
            if key in seen_events:
                _fail("duplicate mapped NoteHeadSet canonical_event_id")
            seen_events.add(key)
            noteheads.append(
                {
                    "canonical_event_id": record.canonical_event_id,
                    "renderer_id": record.renderer_id,
                    "notehead_bbox": _box_dict(box),
                    "notehead_center": _point_dict(center),
                    "fill_class": record.class_name,
                }
            )

        for record in measure.rests:
            box = _d5._map_box(record.bbox, mapper)
            if not _inside_image(box, width, height) or not _box_inside(box, measure_box):
                _fail("mapped rest bbox is outside image/owning measure")
            key = ("rest", record.canonical_event_id)
            if key in seen_events:
                _fail("duplicate mapped RestSet canonical_event_id")
            seen_events.add(key)
            rests.append(
                {
                    "canonical_event_id": record.canonical_event_id,
                    "renderer_id": record.renderer_id,
                    "rest_bbox": _box_dict(box),
                    "rest_class": record.class_name,
                    "duration_class": record.class_name,
                }
            )

        for record in measure.accidentals:
            box = _d5._map_box(record.bbox, mapper)
            if not _inside_image(box, width, height) or not _box_inside(box, measure_box):
                _fail("mapped accidental bbox is outside image/owning measure")
            key = ("accidental", record.canonical_event_id)
            if key in seen_events:
                _fail("duplicate mapped AccidentalSet canonical_event_id")
            seen_events.add(key)
            accidentals.append(
                {
                    "canonical_event_id": record.canonical_event_id,
                    "renderer_id": record.renderer_id,
                    "accidental_bbox": _box_dict(box),
                    "accidental_class": record.class_name,
                }
            )

        measures.append(
            {
                "measure_number": measure.measure_number,
                "renderer_measure_id": measure.renderer_measure_id,
                "measure_bbox": _box_dict(measure_box),
                "noteheads": noteheads,
                "rests": rests,
                "accidentals": accidentals,
            }
        )
    return measures, transform_fingerprint


def _load_d6_label(
    d6_root: Path,
    record: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    label_sha = _hex64("D6 record.label_sha256", record.get("label_sha256"))
    label, raw = _read_canonical_json(
        d6_root / "labels" / f"{label_sha}.json",
        _MAX_LABEL_BYTES,
        "accepted D6 label",
    )
    if sha256(raw).hexdigest() != label_sha:
        _fail("accepted D6 label SHA-256 mismatch on D12 re-read")
    if label.get("schema_version") != STAGE7D6_LABEL_SCHEMA:
        _fail("accepted D6 label schema mismatch")
    if label.get("stage7d6_version") != STAGE7D6_VERSION:
        _fail("accepted D6 label version mismatch")
    return label, label_sha


def _d6_geometry_lineage(label: Mapping[str, object]) -> tuple[str, str, str]:
    geometry = label.get("geometry")
    if not isinstance(geometry, Mapping):
        _fail("accepted D6 label geometry must be an object")
    instrumentation = _hex64(
        "D6 geometry.geometry_instrumentation_fingerprint",
        geometry.get("geometry_instrumentation_fingerprint"),
    )
    geometry_svg = _hex64(
        "D6 geometry.geometry_svg_sha256",
        geometry.get("geometry_svg_sha256"),
    )
    transform = _hex64(
        "D6 geometry.geometry_transform_fingerprint",
        geometry.get("geometry_transform_fingerprint"),
    )
    return instrumentation, geometry_svg, transform


def _label_payload(
    *,
    source: Mapping[str, object],
    d6_record: Mapping[str, object],
    d6_label_sha: str,
    page: SymbolGeometryPage,
    mapped_measures: list[dict[str, object]],
    transform_fingerprint: str,
) -> dict[str, object]:
    sample_id = _hex64("source.sample_id", source.get("sample_id"))
    family_id = _identifier("source.family_id", source.get("family_id"))
    split = source.get("split")
    if split not in _ALLOWED_SPLITS:
        _fail("D12 label split must be TRAIN or VALIDATION")
    page_number = _positive_int("source.page_number", source.get("page_number"))
    png_sha = _hex64("source.png_sha256", source.get("png_sha256"))
    width = _positive_int("source.width", source.get("width"))
    height = _positive_int("source.height", source.get("height"))
    source_musicxml_sha = _hex64(
        "source.source_musicxml_sha256",
        source.get("source_musicxml_sha256"),
    )
    source_svg_sha = _hex64(
        "source.source_svg_sha256",
        source.get("source_svg_sha256"),
    )
    renderer_fp = _hex64(
        "source.renderer_config_fingerprint",
        source.get("renderer_config_fingerprint"),
    )
    degradation_fp = _hex64(
        "source.degradation_config_fingerprint",
        source.get("degradation_config_fingerprint"),
    )
    if d6_record.get("sample_id") != sample_id or d6_record.get("png_sha256") != png_sha:
        _fail("accepted D6 record disagrees with source sample identity")

    return {
        "schema_version": STAGE7D12_LABEL_SCHEMA,
        "stage7d12_derivative_version": STAGE7D12_DERIVATIVE_VERSION,
        "contract_fingerprint": stage7d12_contract_fingerprint(),
        "sample_id": sample_id,
        "family_id": family_id,
        "split": split,
        "page_number": page_number,
        "image": {
            "png_sha256": png_sha,
            "width": width,
            "height": height,
            "mode": "L",
            "image_format": "png",
        },
        "accepted_d6": {
            "manifest_sha256": EXPECTED_D6_MANIFEST_SHA256,
            "artifact_binding_sha256": EXPECTED_D6_ARTIFACT_BINDING_SHA256,
            "label_sha256": d6_label_sha,
        },
        "lineage": {
            "source_musicxml_sha256": source_musicxml_sha,
            "source_svg_sha256": source_svg_sha,
            "renderer_config_fingerprint": renderer_fp,
            "degradation_config_fingerprint": degradation_fp,
            "symbol_geometry_version": STAGE7D12_SYMBOL_GEOMETRY_VERSION,
            "geometry_instrumentation_fingerprint": (
                page.geometry_instrumentation_fingerprint
            ),
            "geometry_svg_sha256": page.geometry_svg_sha256,
            "d5_transform_version": STAGE7D5_TRANSFORM_VERSION,
            "geometry_transform_fingerprint": transform_fingerprint,
        },
        "symbol_geometry": {
            "coordinate_space": "final_png_pixels",
            "view_box": list(page.view_box),
            "measures": mapped_measures,
        },
    }


def _empty_inventory() -> dict[str, dict[str, dict[str, int]]]:
    return {
        split: {
            "notehead": {name: 0 for name in NOTEHEAD_FILL_CLASSES},
            "rest": {name: 0 for name in REST_CLASSES},
            "accidental": {name: 0 for name in ACCIDENTAL_CLASSES},
        }
        for split in ("train", "validation")
    }


def _accumulate_inventory(
    inventory: dict[str, dict[str, dict[str, int]]],
    split: str,
    measures: Sequence[Mapping[str, object]],
) -> None:
    for measure in measures:
        noteheads = measure.get("noteheads")
        rests = measure.get("rests")
        accidentals = measure.get("accidentals")
        if not all(isinstance(value, list) for value in (noteheads, rests, accidentals)):
            _fail("mapped measure target lists must be lists")
        assert isinstance(noteheads, list)
        assert isinstance(rests, list)
        assert isinstance(accidentals, list)
        for row in noteheads:
            if not isinstance(row, Mapping):
                _fail("mapped notehead target must be an object")
            cls = row.get("fill_class")
            if cls not in NOTEHEAD_FILL_CLASSES:
                _fail("mapped notehead class is outside D12")
            assert isinstance(cls, str)
            inventory[split]["notehead"][cls] += 1
        for row in rests:
            if not isinstance(row, Mapping):
                _fail("mapped rest target must be an object")
            cls = row.get("rest_class")
            if cls not in REST_CLASSES:
                _fail("mapped rest class is outside D12")
            assert isinstance(cls, str)
            inventory[split]["rest"][cls] += 1
        for row in accidentals:
            if not isinstance(row, Mapping):
                _fail("mapped accidental target must be an object")
            cls = row.get("accidental_class")
            if cls not in ACCIDENTAL_CLASSES:
                _fail("mapped accidental class is outside D12")
            assert isinstance(cls, str)
            inventory[split]["accidental"][cls] += 1


def _prepare_output_root(output_root: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    resolved = output_root.resolve()
    if resolved == repo_root or repo_root in resolved.parents:
        _fail("D12 output root must remain outside the repository")
    if output_root.exists() or output_root.is_symlink():
        _fail("D12 output root must be fresh")
    output_root.mkdir(parents=True)
    (output_root / "labels").mkdir()


def _write_new(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        _fail(f"refusing to overwrite D12 artifact: {path.name}")
    path.write_bytes(raw)


def build_stage7d12_symbol_derivatives(
    corpus_root: str | Path,
    d6_root: str | Path,
    output_root: str | Path,
) -> Stage7D12DerivativeReceipt:
    """Build the exact D12 TRAIN/VALIDATION symbol derivative surface.

    The returned receipt describes an *uncompleted* persisted bundle.  No
    ``COMPLETE`` marker is created here; independent persisted verification is a
    separate D12 gate.
    """

    if not all(isinstance(value, (str, Path)) for value in (corpus_root, d6_root, output_root)):
        raise TypeError("corpus_root, d6_root and output_root must be str or pathlib.Path")
    source_root = Path(corpus_root)
    accepted_d6_root = Path(d6_root)
    out = Path(output_root)
    _regular_directory(source_root, "source corpus root")
    _regular_directory(accepted_d6_root, "accepted D6 root")

    source_index, rows = _source_index(source_root)
    d6_index = _accepted_d6_index(source_root, accepted_d6_root)
    if set(source_index) != set(d6_index):
        _fail("source/D6 development sample identities do not match exactly")
    _prepare_output_root(out)

    renderer_config = RendererConfig()
    frozen_renderer_fp = renderer_config_fingerprint(renderer_config)
    family_cache: dict[str, tuple[object, tuple[SymbolGeometryPage, ...]]] = {}
    family_lineage: dict[str, tuple[str, str, str]] = {}
    records: list[dict[str, object]] = []
    label_hashes: set[str] = set()
    label_bytes_total = 0
    inventory = _empty_inventory()
    binding_rows: list[str] = []

    for source in rows:
        sample_id = _hex64("source.sample_id", source.get("sample_id"))
        family_id = _identifier("source.family_id", source.get("family_id"))
        split = source.get("split")
        if split not in _ALLOWED_SPLITS:
            _fail("development row escaped D12 split seal")
        assert isinstance(split, str)
        target_sha = _hex64(
            "source.source_musicxml_sha256",
            source.get("source_musicxml_sha256"),
        )
        source_svg_sha = _hex64(
            "source.source_svg_sha256",
            source.get("source_svg_sha256"),
        )
        renderer_fp = _hex64(
            "source.renderer_config_fingerprint",
            source.get("renderer_config_fingerprint"),
        )
        if renderer_fp != frozen_renderer_fp:
            _fail("source renderer fingerprint differs from pinned D12 renderer")

        png_sha = _hex64("source.png_sha256", source.get("png_sha256"))
        png_raw = _read_bounded(
            source_root / "images" / f"{png_sha}.png",
            _MAX_IMAGE_BYTES,
            "source PNG",
        )
        if sha256(png_raw).hexdigest() != png_sha:
            _fail("source PNG SHA-256 mismatch on D12 re-read")

        d6_record = d6_index[sample_id]
        d6_label, d6_label_sha = _load_d6_label(accepted_d6_root, d6_record)
        d6_instrumentation, d6_geometry_svg, d6_transform = _d6_geometry_lineage(d6_label)

        lineage = (target_sha, source_svg_sha, renderer_fp)
        prior_lineage = family_lineage.setdefault(family_id, lineage)
        if prior_lineage != lineage:
            _fail("family source lineage differs across degradation variants")

        cached = family_cache.get(family_id)
        if cached is None:
            target_raw = _read_bounded(
                source_root / "targets" / f"{target_sha}.musicxml",
                _MAX_TARGET_BYTES,
                "source MusicXML",
            )
            if sha256(target_raw).hexdigest() != target_sha:
                _fail("source MusicXML SHA-256 mismatch")
            base_render = render_musicxml_svg(target_raw, renderer_config)
            if base_render.source_musicxml_sha256 != target_sha:
                _fail("base renderer MusicXML lineage mismatch")
            if base_render.config_fingerprint != renderer_fp:
                _fail("base renderer configuration mismatch")
            geometry_render = render_musicxml_geometry_svg(target_raw, renderer_config)
            if geometry_render.source_musicxml_sha256 != target_sha:
                _fail("geometry renderer MusicXML lineage mismatch")
            if geometry_render.base_renderer_config_fingerprint != renderer_fp:
                _fail("geometry renderer configuration mismatch")
            symbol_pages = extract_symbol_geometry(geometry_render, target_raw)
            cached = (base_render, symbol_pages)
            family_cache[family_id] = cached

        base_render, symbol_pages = cached
        page_number = _positive_int("source.page_number", source.get("page_number"))
        base_page = _page_by_number(base_render.pages, page_number, "base render")
        if getattr(base_page, "sha256", None) != source_svg_sha:
            _fail("pinned source SVG SHA-256 changed from frozen corpus")
        symbol_page = _page_by_number(symbol_pages, page_number, "symbol geometry")
        if not isinstance(symbol_page, SymbolGeometryPage):
            _fail("symbol geometry page has unexpected type")
        if symbol_page.geometry_instrumentation_fingerprint != d6_instrumentation:
            _fail("D12/D6 geometry instrumentation fingerprints disagree")
        if symbol_page.geometry_svg_sha256 != d6_geometry_svg:
            _fail("D12/D6 geometry SVG hashes disagree")

        view = _degraded_view(source)
        mapped_measures, transform_fp = map_symbol_page_to_final_png(symbol_page, view)
        if transform_fp != d6_transform:
            _fail("D12 symbol mapping disagrees with accepted D6 final-PNG transform")

        label = _label_payload(
            source=source,
            d6_record=d6_record,
            d6_label_sha=d6_label_sha,
            page=symbol_page,
            mapped_measures=mapped_measures,
            transform_fingerprint=transform_fp,
        )
        raw_label = _canonical_json(label)
        if not 1 <= len(raw_label) <= _MAX_LABEL_BYTES:
            _fail("D12 symbol label byte length is outside bound")
        label_sha = sha256(raw_label).hexdigest()
        if label_sha in label_hashes:
            _fail("duplicate D12 symbol label SHA-256")
        label_hashes.add(label_sha)
        _write_new(out / "labels" / f"{label_sha}.json", raw_label)
        label_bytes_total += len(raw_label)
        _accumulate_inventory(inventory, split, mapped_measures)

        records.append(
            {
                "sample_id": sample_id,
                "family_id": family_id,
                "split": split,
                "page_number": page_number,
                "png_sha256": png_sha,
                "d6_label_sha256": d6_label_sha,
                "symbol_label_sha256": label_sha,
            }
        )
        binding_rows.append(
            f"{sample_id}:{png_sha}:{d6_label_sha}:{label_sha}:{len(raw_label)}"
        )

    records.sort(key=lambda row: str(row["sample_id"]))
    manifest_payload = {
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
        "records": records,
    }
    manifest_raw = _canonical_json(manifest_payload)
    manifest_sha = sha256(manifest_raw).hexdigest()
    _write_new(out / "manifest.json", manifest_raw)
    _write_new(
        out / "manifest.sha256",
        f"{manifest_sha}  manifest.json\n".encode("ascii"),
    )

    sample_split_counts = dict(
        sorted(Counter(str(record["split"]) for record in records).items())
    )
    family_split_counts = {
        split_name: len(
            {
                str(record["family_id"])
                for record in records
                if record["split"] == split_name
            }
        )
        for split_name in ("train", "validation")
    }
    if sample_split_counts != EXPECTED_DEVELOPMENT_SAMPLE_COUNTS:
        _fail("built D12 sample split counts mismatch")
    if family_split_counts != EXPECTED_DEVELOPMENT_FAMILY_COUNTS:
        _fail("built D12 family split counts mismatch")
    if len(records) != _EXPECTED_SAMPLE_COUNT or len(label_hashes) != _EXPECTED_SAMPLE_COUNT:
        _fail("built D12 record/label cardinality mismatch")

    artifact_binding = sha256(
        ("\n".join(sorted(binding_rows)) + "\n").encode("ascii")
    ).hexdigest()
    target_instance_counts = {
        split_name: {
            kind: sum(inventory[split_name][kind].values())
            for kind in ("notehead", "rest", "accidental")
        }
        for split_name in ("train", "validation")
    }
    build_payload = {
        "schema_version": STAGE7D12_BUILD_SCHEMA,
        "stage7d12_derivative_version": STAGE7D12_DERIVATIVE_VERSION,
        "derivative_build_id": stage7d12_derivative_profile_fingerprint(),
        "manifest_sha256": manifest_sha,
        "sample_count": len(records),
        "family_count": len({str(record["family_id"]) for record in records}),
        "label_count": len(label_hashes),
        "label_bytes_total": label_bytes_total,
        "sample_split_counts": sample_split_counts,
        "family_split_counts": family_split_counts,
        "target_instance_counts": target_instance_counts,
        "observed_class_inventory": inventory,
        "artifact_binding_sha256": artifact_binding,
        "test_specialist_records": TEST_SPECIALIST_RECORDS,
        "optimizer_steps": OPTIMIZER_STEPS,
        "complete_marker_written": False,
        "completion_policy": "independent-verifier-required-before-COMPLETE",
        "layout": {
            "manifest": "manifest.json",
            "labels": "labels/<symbol_label_sha256>.json",
            "source_images": "external frozen corpus images/<png_sha256>.png",
            "accepted_d6_labels": "external accepted D6 labels/<d6_label_sha256>.json",
        },
    }
    build_raw = _canonical_json(build_payload)
    if not 1 <= len(build_raw) <= _MAX_BUILD_BYTES:
        _fail("D12 build.json byte length is outside bound")
    _write_new(out / "build.json", build_raw)

    # Deliberately no COMPLETE marker here.  D12-5 must independently reopen the
    # persisted bundle and only then may a later controlled step write COMPLETE.
    return Stage7D12DerivativeReceipt(
        derivative_build_id=stage7d12_derivative_profile_fingerprint(),
        manifest_sha256=manifest_sha,
        sample_count=len(records),
        family_count=len({str(record["family_id"]) for record in records}),
        sample_split_counts=sample_split_counts,
        family_split_counts=family_split_counts,
        label_count=len(label_hashes),
        label_bytes_total=label_bytes_total,
        artifact_binding_sha256=artifact_binding,
        observed_class_inventory=inventory,
        test_specialist_records=TEST_SPECIALIST_RECORDS,
        optimizer_steps=OPTIMIZER_STEPS,
        complete_marker_written=False,
    )
