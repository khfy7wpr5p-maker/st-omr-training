"""Sparse D10 transport for the bounded Meter V3-A1 Colab experiment.

This module preserves the authoritative D10 manifest/binding and the exact
V3-A1 replay selection while avoiding a full 44k-artifact local materialization.
It reads the authoritative Meter labels once to recover the frozen class map,
stages only the 512 selected TRAIN images plus all 1,224 Meter VALIDATION images,
and injects the already-validated records/labels into the existing V3-A1 run.

The helper is transport-only: it does not change the model, optimizer, data
split, replay seed, acceptance gate, TEST policy, runtime connection, or
production promotion boundary.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
from typing import Final


EXPECTED_D10_RECORDS: Final[int] = 22_128
EXPECTED_METER_TRAIN: Final[int] = 9_840
EXPECTED_METER_VALIDATION: Final[int] = 1_224
SPARSE_CACHE_SCHEMA: Final[str] = "st-omr-meter-v3-a1-sparse-d10-v1"
METER_CLASSES: Final[tuple[str, ...]] = ("none", "2/4", "3/4", "4/4")


class MeterV3A1SparseD10Error(RuntimeError):
    """Raised when sparse D10 provenance or staging fails closed."""


def _fail(message: str) -> None:
    raise MeterV3A1SparseD10Error(message)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _hex64(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        _fail(f"{name} must be canonical lowercase SHA-256")
    return value


def _safe_relative(name: str, value: object) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"{name} must be a non-empty relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or any(part in {"", "."} for part in pure.parts):
        _fail(f"{name} escapes the authoritative D10 root")
    return Path(*pure.parts)


def _read_canonical_json(path: Path, *, maximum: int, name: str) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{name} must be a regular file")
    size = path.stat().st_size
    if not 1 <= size <= maximum:
        _fail(f"{name} byte length is outside sparse D10 bounds")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MeterV3A1SparseD10Error(f"{name} is not valid ASCII JSON") from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        _fail(f"{name} must be canonical JSON object bytes")
    return payload, raw


def _emit(progress: Callable[[str, Mapping[str, object]], None] | None, event: str, **payload: object) -> None:
    if progress is not None:
        progress(event, payload)


@dataclass(frozen=True, slots=True)
class SparseD10PreparationV3A1:
    source_root: Path
    cache_root: Path
    manifest_sha256: str
    artifact_binding_sha256: str
    records: tuple[object, ...]
    labels_by_record_id: Mapping[str, Mapping[str, object]]
    replay_record_ids: tuple[str, ...]
    staged_image_count: int
    meter_label_count: int


def _receipt_gate(
    source_root: Path,
    *,
    expected_manifest_sha256: str,
    expected_artifact_binding_sha256: str,
) -> None:
    receipt, _raw = _read_canonical_json(
        source_root / "receipt.json",
        maximum=2 * 1024 * 1024,
        name="D10 receipt",
    )
    if receipt.get("manifest_sha256") != expected_manifest_sha256:
        _fail("D10 receipt manifest SHA does not match the frozen authoritative manifest")
    if receipt.get("artifact_binding_sha256") != expected_artifact_binding_sha256:
        _fail("D10 receipt artifact binding does not match the frozen authoritative binding")
    if receipt.get("roi_record_count") != EXPECTED_D10_RECORDS:
        _fail("D10 receipt record count changed")
    if receipt.get("test_records") != 0:
        _fail("D10 receipt exposes TEST records")
    if receipt.get("optimizer_steps") != 0:
        _fail("D10 derivative receipt unexpectedly contains optimizer steps")


def _copy_verified_image(source: Path, destination: Path, expected_sha256: str) -> None:
    if source.is_symlink() or not source.is_file():
        _fail(f"D10 source image is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not destination.is_symlink():
        raw = destination.read_bytes()
        if _sha(raw) == expected_sha256:
            return
    raw = source.read_bytes()
    if _sha(raw) != expected_sha256:
        _fail(f"D10 source image SHA mismatch: {source}")
    temporary = destination.with_name(f".{destination.name}.part")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.write_bytes(raw)
    temporary.replace(destination)


def prepare_sparse_meter_d10_v3_a1(
    *,
    source_root: str | Path,
    cache_root: str | Path,
    expected_manifest_sha256: str,
    expected_artifact_binding_sha256: str,
    replay_per_class: int,
    replay_seed: int,
    progress: Callable[[str, Mapping[str, object]], None] | None = None,
) -> SparseD10PreparationV3A1:
    """Stage only the images actually consumed by V3-A1.

    The authoritative manifest is still read in full. All 11,064 Meter labels
    are validated once so the exact class-balanced replay IDs can be reproduced.
    Only 512 TRAIN images (128/class with the frozen V3-A1 settings) and all
    1,224 Meter VALIDATION images are copied to local SSD.
    """
    if not isinstance(replay_per_class, int) or isinstance(replay_per_class, bool) or replay_per_class <= 0:
        raise ValueError("replay_per_class must be a positive integer")
    if not isinstance(replay_seed, int) or isinstance(replay_seed, bool) or replay_seed < 0:
        raise ValueError("replay_seed must be a non-negative integer")

    source = Path(source_root)
    cache = Path(cache_root)
    manifest_sha = _hex64("expected_manifest_sha256", expected_manifest_sha256)
    binding_sha = _hex64("expected_artifact_binding_sha256", expected_artifact_binding_sha256)

    if source.is_symlink() or not source.is_dir():
        _fail("authoritative D10 source root must be a regular directory")
    if cache.exists() and (cache.is_symlink() or not cache.is_dir()):
        _fail("sparse D10 cache root must be a regular directory")
    cache.mkdir(parents=True, exist_ok=True)

    manifest, manifest_raw = _read_canonical_json(
        source / "manifest.json",
        maximum=64 * 1024 * 1024,
        name="D10 manifest",
    )
    if _sha(manifest_raw) != manifest_sha:
        _fail("D10 authoritative manifest SHA mismatch")
    sidecar = source / "manifest.sha256"
    expected_sidecar = f"{manifest_sha}  manifest.json\n".encode("ascii")
    if sidecar.is_symlink() or not sidecar.is_file() or sidecar.read_bytes() != expected_sidecar:
        _fail("D10 authoritative manifest sidecar mismatch")
    _receipt_gate(
        source,
        expected_manifest_sha256=manifest_sha,
        expected_artifact_binding_sha256=binding_sha,
    )

    rows = manifest.get("records")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        _fail("D10 manifest records must be a sequence")
    if len(rows) != EXPECTED_D10_RECORDS:
        _fail("D10 manifest record count changed")

    from .stage7d11_barline_meter_training import D11Record, _load_d11_label
    from .meter_real_domain_adaptation_v1 import deterministic_replay_ids_v1

    meter_records: list[object] = []
    relative_image_by_id: dict[str, Path] = {}
    split_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"D10 manifest record[{index}] must be an object")
        split = row.get("split")
        if split == "test":
            _fail("sealed TEST record reached sparse V3-A1 planning")
        if split not in {"train", "validation"}:
            _fail("D10 sparse transport accepts only TRAIN/VALIDATION")
        kind = row.get("kind")
        if kind not in {"barline", "meter"}:
            _fail("D10 manifest kind is outside the frozen surface")
        split_counts[str(split)] += 1
        kind_counts[str(kind)] += 1
        if kind != "meter":
            continue

        record_id = _hex64("D10 record_id", row.get("record_id"))
        image_rel = _safe_relative("D10 image_path", row.get("image_path"))
        label_rel = _safe_relative("D10 label_path", row.get("label_path"))
        family_id = row.get("family_id")
        source_sample_id = row.get("source_sample_id")
        measure_number = row.get("measure_number")
        if not isinstance(family_id, str) or not family_id:
            _fail("D10 family_id must be non-empty")
        if not isinstance(source_sample_id, str) or not source_sample_id:
            _fail("D10 source_sample_id must be non-empty")
        if not isinstance(measure_number, int) or isinstance(measure_number, bool) or measure_number <= 0:
            _fail("D10 measure_number must be positive integer")
        image_sha = _hex64("D10 image_sha256", row.get("image_sha256"))
        label_sha = _hex64("D10 label_sha256", row.get("label_sha256"))

        relative_image_by_id[record_id] = image_rel
        meter_records.append(
            D11Record(
                record_id=record_id,
                kind="meter",
                split=str(split),
                family_id=family_id,
                source_sample_id=source_sample_id,
                measure_number=measure_number,
                image_path=source / image_rel,
                image_sha256=image_sha,
                label_path=source / label_rel,
                label_sha256=label_sha,
            )
        )

    if split_counts != Counter({"train": 19_680, "validation": 2_448}):
        _fail("D10 full split counts changed")
    if kind_counts != Counter({"barline": 11_064, "meter": 11_064}):
        _fail("D10 full kind counts changed")

    meter_train = [record for record in meter_records if record.split == "train"]
    meter_validation = [record for record in meter_records if record.split == "validation"]
    if len(meter_train) != EXPECTED_METER_TRAIN or len(meter_validation) != EXPECTED_METER_VALIDATION:
        _fail("D10 Meter TRAIN/VALIDATION cardinality changed")

    labels_by_id: dict[str, Mapping[str, object]] = {}
    class_to_ids: defaultdict[str, list[str]] = defaultdict(list)
    _emit(
        progress,
        "d10_sparse_label_scan_started",
        phase="d10_sparse_meter_labels",
        phase_index=2,
        phase_total=9,
        files_completed=0,
        files_total=len(meter_records),
        records_total=len(meter_records),
        full_cache_copy=False,
    )
    for index, record in enumerate(meter_records, start=1):
        label = _load_d11_label(record)
        labels_by_id[record.record_id] = label
        if record.split == "train":
            target = label.get("target")
            if not isinstance(target, Mapping) or target.get("meter_class") not in METER_CLASSES:
                _fail("D10 TRAIN Meter class is invalid")
            class_to_ids[str(target["meter_class"])].append(record.record_id)
        if index == len(meter_records) or index % 250 == 0:
            _emit(
                progress,
                "d10_sparse_label_scan_progress",
                phase="d10_sparse_meter_labels",
                phase_index=2,
                phase_total=9,
                files_completed=index,
                files_total=len(meter_records),
                records_total=len(meter_records),
                full_cache_copy=False,
            )

    replay_ids = deterministic_replay_ids_v1(
        class_to_ids,
        per_class=replay_per_class,
        seed=replay_seed,
    )
    expected_replay = replay_per_class * len(METER_CLASSES)
    if len(replay_ids) != expected_replay or len(set(replay_ids)) != expected_replay:
        _fail("sparse replay selection is not exact and unique")

    required_ids = set(replay_ids)
    required_ids.update(record.record_id for record in meter_validation)
    if len(required_ids) != expected_replay + EXPECTED_METER_VALIDATION:
        _fail("sparse D10 required image cardinality changed")

    record_by_id = {record.record_id: record for record in meter_records}
    _emit(
        progress,
        "d10_sparse_image_stage_started",
        phase="d10_sparse_meter_images",
        phase_index=2,
        phase_total=9,
        files_completed=0,
        files_total=len(required_ids),
        records_total=len(required_ids),
        full_cache_copy=False,
    )
    for index, record_id in enumerate(sorted(required_ids), start=1):
        record = record_by_id[record_id]
        relative = relative_image_by_id[record_id]
        destination = cache / relative
        _copy_verified_image(record.image_path, destination, record.image_sha256)
        if index == len(required_ids) or index % 50 == 0:
            _emit(
                progress,
                "d10_sparse_image_stage_progress",
                phase="d10_sparse_meter_images",
                phase_index=2,
                phase_total=9,
                files_completed=index,
                files_total=len(required_ids),
                records_total=len(required_ids),
                full_cache_copy=False,
            )

    remapped_records: list[object] = []
    for record in meter_records:
        image_path = record.image_path
        if record.record_id in required_ids:
            image_path = cache / relative_image_by_id[record.record_id]
        remapped_records.append(
            D11Record(
                record_id=record.record_id,
                kind=record.kind,
                split=record.split,
                family_id=record.family_id,
                source_sample_id=record.source_sample_id,
                measure_number=record.measure_number,
                image_path=image_path,
                image_sha256=record.image_sha256,
                label_path=record.label_path,
                label_sha256=record.label_sha256,
            )
        )

    marker = {
        "schema_version": SPARSE_CACHE_SCHEMA,
        "manifest_sha256": manifest_sha,
        "artifact_binding_sha256": binding_sha,
        "meter_train_records": EXPECTED_METER_TRAIN,
        "meter_validation_records": EXPECTED_METER_VALIDATION,
        "replay_records_per_class": replay_per_class,
        "replay_record_count": len(replay_ids),
        "replay_record_ids_sha256": _sha(_canonical_json(list(replay_ids))),
        "staged_image_count": len(required_ids),
        "meter_labels_validated": len(labels_by_id),
        "full_cache_copy": False,
        "test_records": 0,
        "test_opened": False,
    }
    (cache / "SPARSE_COMPLETE.json").write_bytes(_canonical_json(marker))
    _emit(
        progress,
        "d10_sparse_complete",
        phase="d10_sparse_meter_images",
        phase_index=2,
        phase_total=9,
        files_completed=len(required_ids),
        files_total=len(required_ids),
        records_total=len(required_ids),
        full_cache_copy=False,
    )

    return SparseD10PreparationV3A1(
        source_root=source,
        cache_root=cache,
        manifest_sha256=manifest_sha,
        artifact_binding_sha256=binding_sha,
        records=tuple(remapped_records),
        labels_by_record_id=labels_by_id,
        replay_record_ids=tuple(replay_ids),
        staged_image_count=len(required_ids),
        meter_label_count=len(labels_by_id),
    )


@contextmanager
def patched_stage7d11_for_sparse_v3_a1(prepared: SparseD10PreparationV3A1):
    """Inject sparse records into the unchanged V3-A1 execution, then restore."""
    from . import stage7d11_barline_meter_training as d11

    original_loader = d11.load_verified_stage7d11_records
    original_label_loader = d11._load_d11_label

    def sparse_loader(
        root,
        *,
        expected_manifest_sha256: str,
        expected_artifact_binding_sha256: str,
    ):
        if Path(root).resolve() != prepared.source_root.resolve():
            _fail("V3-A1 sparse loader received an unexpected D10 root")
        if expected_manifest_sha256 != prepared.manifest_sha256:
            _fail("V3-A1 sparse loader manifest SHA changed")
        if expected_artifact_binding_sha256 != prepared.artifact_binding_sha256:
            _fail("V3-A1 sparse loader artifact binding changed")
        return prepared.records

    def sparse_label_loader(record):
        value = prepared.labels_by_record_id.get(record.record_id)
        if value is None:
            _fail("V3-A1 sparse label cache missed a Meter record")
        return dict(value)

    d11.load_verified_stage7d11_records = sparse_loader
    d11._load_d11_label = sparse_label_loader
    try:
        yield
    finally:
        d11.load_verified_stage7d11_records = original_loader
        d11._load_d11_label = original_label_loader
