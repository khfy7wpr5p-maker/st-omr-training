#!/usr/bin/env python3
"""Colab background runner with durable progress for Meter adaptation v1.

The process copies the accepted D10 development bundle from mounted Drive to
Colab's local SSD with exact file-count progress, then runs the shadow-only
adaptation while atomically persisting heartbeat/progress JSON to Drive.
Epoch-boundary resume state is owned by the adaptation module.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import threading
import time
import traceback
from typing import Mapping


EXPECTED_D10_RECORDS = 22_128
CACHE_SCHEMA = "st-omr-meter-d10-local-cache-v2"
LEGACY_CACHE_SCHEMA = "st-omr-meter-d10-local-cache-v1"
STATUS_SCHEMA = "st-omr-meter-background-status-v1"
D10_TOP_LEVEL = frozenset(
    {"images", "labels", "manifest.json", "manifest.sha256", "receipt.json", "COMPLETE"}
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_json(dict(payload)))
    temporary.replace(path)


class DurableStatus:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started_at = _now()
        self.started_monotonic = time.monotonic()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.state: dict[str, object] = {
            "schema_version": STATUS_SCHEMA,
            "state": "RUNNING",
            "event": "runner_started",
            "started_at": self.started_at,
            "updated_at": self.started_at,
            "elapsed_seconds": 0,
            "phase": "starting",
            "phase_index": 0,
            "phase_total": 9,
            "epoch": 0,
            "epochs_total": 8,
            "batch": 0,
            "batches_total": 0,
            "test_opened": False,
            "runtime_connected": False,
            "production_promotion_authorized": False,
        }
        self._write_locked()
        self.thread = threading.Thread(target=self._heartbeat, name="meter-status-heartbeat", daemon=True)
        self.thread.start()

    def _write_locked(self) -> None:
        self.state["updated_at"] = _now()
        self.state["elapsed_seconds"] = int(time.monotonic() - self.started_monotonic)
        _atomic_json(self.path, self.state)

    def update(self, event: str, payload: Mapping[str, object] | None = None) -> None:
        with self.lock:
            self.state["event"] = event
            if payload:
                self.state.update(payload)
            self._write_locked()
        print(json.dumps(self.state, sort_keys=True), flush=True)

    def _heartbeat(self) -> None:
        while not self.stop_event.wait(30):
            with self.lock:
                self.state["heartbeat"] = True
                self._write_locked()

    def finish(self, *, state: str, payload: Mapping[str, object] | None = None) -> None:
        with self.lock:
            self.state["state"] = state
            self.state["event"] = "runner_finished" if state == "COMPLETE" else "runner_failed"
            if payload:
                self.state.update(payload)
            self._write_locked()
        self.stop_event.set()
        self.thread.join(timeout=2)


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError("D10 artifact path must be a non-empty string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or any(part in {"", "."} for part in pure.parts):
        raise RuntimeError("D10 artifact path escapes the local cache")
    return Path(*pure.parts)


def _copy_atomic(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"D10 source artifact is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not destination.is_symlink() and destination.stat().st_size == source.stat().st_size:
        return
    temporary = destination.with_name(f".{destination.name}.part")
    if temporary.exists():
        temporary.unlink()
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _read_marker(path: Path) -> object:
    try:
        return json.loads(path.read_text("ascii"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None


def _cache_marker_path(cache_root: Path) -> Path:
    return cache_root.parent / f".{cache_root.name}.complete.json"


def materialize_d10_cache(
    *,
    source_root: Path,
    cache_root: Path,
    expected_manifest_sha256: str,
    status: DurableStatus,
) -> Path:
    """Copy exact D10 development bytes locally with resumable N/total progress."""
    source_manifest = source_root / "manifest.json"
    raw = source_manifest.read_bytes()
    if _sha(raw) != expected_manifest_sha256:
        raise RuntimeError("D10 Drive manifest SHA-256 mismatch before cache copy")
    payload = json.loads(raw.decode("ascii"))
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) != EXPECTED_D10_RECORDS:
        raise RuntimeError("D10 manifest record count differs from 22,128")
    source_manifest_sidecar = source_root / "manifest.sha256"
    expected_sidecar = f"{expected_manifest_sha256}  manifest.json\n".encode("ascii")
    if source_manifest_sidecar.read_bytes() != expected_sidecar:
        raise RuntimeError("D10 Drive manifest SHA sidecar mismatch before cache copy")
    relative_paths = [
        Path("COMPLETE"),
        Path("manifest.json"),
        Path("manifest.sha256"),
        Path("receipt.json"),
    ]
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("D10 manifest record must be an object")
        if row.get("split") == "test":
            raise RuntimeError("sealed TEST record reached D10 cache planning")
        if row.get("split") not in {"train", "validation"}:
            raise RuntimeError("D10 cache accepts only TRAIN/VALIDATION records")
        relative_paths.extend((_safe_relative(row.get("image_path")), _safe_relative(row.get("label_path"))))
    unique_paths = tuple(dict.fromkeys(relative_paths))
    marker_path = _cache_marker_path(cache_root)
    legacy_marker_path = cache_root / "CACHE_COMPLETE.json"
    marker_expected = {
        "schema_version": CACHE_SCHEMA,
        "manifest_sha256": expected_manifest_sha256,
        "record_count": EXPECTED_D10_RECORDS,
        "file_count": len(unique_paths),
        "test_records": 0,
        "test_opened": False,
    }
    if _read_marker(marker_path) == marker_expected and cache_root.is_dir():
        if {path.name for path in cache_root.iterdir()} == D10_TOP_LEVEL:
            status.update(
                "d10_cache_reused",
                {
                    "phase": "d10_local_cache",
                    "phase_index": 2,
                    "phase_total": 9,
                    "files_completed": len(unique_paths),
                    "files_total": len(unique_paths),
                    "records_total": EXPECTED_D10_RECORDS,
                },
            )
            return cache_root
    legacy_marker_expected = {
        "schema_version": LEGACY_CACHE_SCHEMA,
        "manifest_sha256": expected_manifest_sha256,
        "record_count": EXPECTED_D10_RECORDS,
        "file_count": len(unique_paths) - 1,
        "test_records": 0,
        "test_opened": False,
    }
    legacy_top_level = (D10_TOP_LEVEL - {"manifest.sha256"}) | {"CACHE_COMPLETE.json"}
    if (
        _read_marker(legacy_marker_path) == legacy_marker_expected
        and cache_root.is_dir()
        and {path.name for path in cache_root.iterdir()} == legacy_top_level
    ):
        _copy_atomic(source_manifest_sidecar, cache_root / "manifest.sha256")
        legacy_marker_path.unlink()
        _atomic_json(marker_path, marker_expected)
        status.update(
            "d10_cache_migrated",
            {
                "phase": "d10_local_cache",
                "phase_index": 2,
                "phase_total": 9,
                "files_completed": len(unique_paths),
                "files_total": len(unique_paths),
                "records_total": EXPECTED_D10_RECORDS,
            },
        )
        return cache_root
    cache_root.mkdir(parents=True, exist_ok=True)
    total = len(unique_paths)
    status.update(
        "d10_cache_started",
        {
            "phase": "d10_local_cache",
            "phase_index": 2,
            "phase_total": 9,
            "files_completed": 0,
            "files_total": total,
            "records_total": EXPECTED_D10_RECORDS,
        },
    )
    for index, relative in enumerate(unique_paths, start=1):
        source = source_root / relative
        destination = cache_root / relative
        if source_root.resolve() not in source.resolve().parents:
            raise RuntimeError("D10 source artifact escapes Drive root")
        if cache_root.resolve() not in destination.resolve().parents:
            raise RuntimeError("D10 destination artifact escapes local cache")
        _copy_atomic(source, destination)
        if index == total or index % 100 == 0:
            status.update(
                "d10_cache_progress",
                {
                    "files_completed": index,
                    "files_total": total,
                    "records_total": EXPECTED_D10_RECORDS,
                },
            )
    if legacy_marker_path.is_symlink():
        raise RuntimeError("legacy D10 cache marker must not be a symlink")
    if legacy_marker_path.is_file():
        legacy_marker_path.unlink()
    observed_top_level = {path.name for path in cache_root.iterdir()}
    if observed_top_level != D10_TOP_LEVEL:
        missing = sorted(D10_TOP_LEVEL - observed_top_level)
        unexpected = sorted(observed_top_level - D10_TOP_LEVEL)
        raise RuntimeError(
            f"D10 local cache top-level shape mismatch after copy: missing={missing}, unexpected={unexpected}"
        )
    _atomic_json(marker_path, marker_expected)
    status.update(
        "d10_cache_complete",
        {"files_completed": total, "files_total": total, "records_total": EXPECTED_D10_RECORDS},
    )
    return cache_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--choices", type=Path, required=True)
    parser.add_argument("--permission", type=Path, required=True)
    parser.add_argument("--privacy", type=Path, required=True)
    parser.add_argument("--teacher-bundle", type=Path, required=True)
    parser.add_argument("--d10-drive-root", type=Path, required=True)
    parser.add_argument("--d10-cache-root", type=Path, required=True)
    parser.add_argument("--d10-manifest-sha256", required=True)
    parser.add_argument("--d10-artifact-binding-sha256", required=True)
    parser.add_argument("--d11-drive-checkpoint", type=Path, required=True)
    parser.add_argument("--d11-local-checkpoint", type=Path, required=True)
    parser.add_argument("--d11-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    return parser


def _run(args: argparse.Namespace, status: DurableStatus) -> dict[str, object]:
    from st_omr_training.meter_real_domain_adaptation_v1 import run_meter_real_domain_adaptation_v1
    from st_omr_training.meter_teacher_gold_admission_v1 import (
        build_meter_teacher_gold_bundle_v1,
        verify_meter_teacher_gold_bundle_v1,
    )

    status.update("teacher_gold_started", {"phase": "teacher_gold", "phase_index": 1, "phase_total": 9})
    if args.teacher_bundle.is_dir() and (args.teacher_bundle / "COMPLETE").is_file():
        teacher_receipt = verify_meter_teacher_gold_bundle_v1(args.teacher_bundle)
    else:
        if args.teacher_bundle.exists():
            shutil.rmtree(args.teacher_bundle)
        teacher_receipt = build_meter_teacher_gold_bundle_v1(
            pilot_path=args.pilot,
            choices_path=args.choices,
            permission_evidence_path=args.permission,
            privacy_review_evidence_path=args.privacy,
            output_root=args.teacher_bundle,
            repository_root=args.repository_root,
        )
    status.update(
        "teacher_gold_complete",
        {"teacher_records": teacher_receipt.record_count, "teacher_manifest_sha256": teacher_receipt.manifest_sha256},
    )
    local_d10 = materialize_d10_cache(
        source_root=args.d10_drive_root,
        cache_root=args.d10_cache_root,
        expected_manifest_sha256=args.d10_manifest_sha256,
        status=status,
    )
    status.update("d11_local_copy_started", {"phase": "d11_local_copy", "phase_index": 3, "phase_total": 9})
    _copy_atomic(args.d11_drive_checkpoint, args.d11_local_checkpoint)
    observed_d11 = _sha(args.d11_local_checkpoint.read_bytes())
    if observed_d11 != args.d11_sha256:
        raise RuntimeError("local D11 checkpoint SHA-256 mismatch")
    status.update("d11_local_copy_complete", {"d11_checkpoint_sha256": observed_d11})

    def progress(event: str, payload: Mapping[str, object]) -> None:
        translated = dict(payload)
        if event == "phase_started":
            # Adaptation phases are 1..7; preparation is exposed as phases 1..3,
            # so translate the model phase into the durable 4..9 display range.
            translated["phase_index"] = min(9, int(payload.get("phase_index", 1)) + 3)
            translated["phase_total"] = 9
        status.update(event, translated)

    metrics = run_meter_real_domain_adaptation_v1(
        teacher_bundle_root=args.teacher_bundle,
        d10_root=local_d10,
        base_checkpoint_path=args.d11_local_checkpoint,
        output_root=args.output_root,
        repository_root=args.repository_root,
        expected_d10_manifest_sha256=args.d10_manifest_sha256,
        expected_d10_artifact_binding_sha256=args.d10_artifact_binding_sha256,
        progress=progress,
        resume=True,
    )
    return metrics


def main() -> int:
    args = _parser().parse_args()
    status = DurableStatus(args.status_path)
    try:
        metrics = _run(args, status)
        status.finish(
            state="COMPLETE",
            payload={
                "result": metrics["status"],
                "epoch": metrics["best"]["epoch"],
                "epochs_total": metrics["configuration"]["epochs"],
                "run_id": metrics["run_id"],
                "output_root": str(args.output_root),
                "test_opened": metrics["test_opened"],
                "runtime_connected": metrics["runtime_connected"],
                "production_promotion_authorized": metrics["production_promotion_authorized"],
            },
        )
        return 0
    except BaseException as exc:
        traceback.print_exc()
        status.finish(
            state="FAILED",
            payload={"error_type": type(exc).__name__, "error": str(exc), "output_root": str(args.output_root)},
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
