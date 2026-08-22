"""Bounded human-resolution layer for METER V5-1 bbox pilot audit flags.

This module never opens VAL/final_holdout images and never authorizes training.
It preserves the original 30-row pilot and records explicit human resolution for
only the mechanically flagged TRAIN samples.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from st_omr_training.meter_v5_1_bbox_pilot import (
    ANNOTATIONS_DIR,
    AnnotationSession,
    MeterV5_1PilotError,
    PILOT_AUDIT_NAME,
    write_pilot_audit,
)

REVIEW_SELECTION_NAME = "bbox_pilot_30_human_review_selection.json"
REVIEW_CSV_NAME = "bbox_pilot_30_human_review.csv"
REVIEW_AUDIT_NAME = "bbox_pilot_30_human_review_audit.json"
REVIEW_SCHEMA = "st-omr-meter-v5-1-bbox-human-review-v1"
ACTIONS = {"ACCEPT_AS_DRAWN", "REDRAWN_AND_ACCEPTED"}
REVIEW_COLUMNS = (
    "sample_id", "action", "bbox_binding_sha256", "image_sha256", "reviewed_utc",
)


def _fail(message: str) -> None:
    raise MeterV5_1PilotError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(tmp), str(path))


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in REVIEW_COLUMNS})
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(tmp), str(path))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != REVIEW_COLUMNS:
            _fail("human review CSV schema mismatch")
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def _bbox_binding(row: Mapping[str, str]) -> str:
    if row.get("status") != "PASS":
        _fail(f"human review requires PASS bbox: {row.get('sample_id', '')}")
    payload = {
        "sample_id": row["sample_id"],
        "meter": row["meter"],
        "split": row["split"],
        "x": int(row["x"]), "y": int(row["y"]),
        "w": int(row["w"]), "h": int(row["h"]),
        "image_sha256": row["image_sha256"],
        "image_width": int(row["image_width"]),
        "image_height": int(row["image_height"]),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _load_audit(path: Path) -> dict[str, object]:
    if not path.is_file():
        _fail(f"pilot audit missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise MeterV5_1PilotError("pilot audit JSON invalid") from exc
    if not isinstance(value, dict):
        _fail("pilot audit must be an object")
    return value


@dataclass(frozen=True)
class ReviewItem:
    review_index: int
    pilot_index: int
    sample_id: str


class HumanReviewSession:
    """Explicit human resolution for the frozen suspicious pilot set only."""

    def __init__(self, *, data_root: str | Path):
        self.data_root = Path(data_root)
        self.annotations_dir = self.data_root / ANNOTATIONS_DIR
        self.session = AnnotationSession(data_root=self.data_root)
        self.audit_path = write_pilot_audit(self.data_root)
        audit = _load_audit(self.audit_path)
        if audit.get("mechanical_gate") != "PASS" or audit.get("annotation_count") != 30:
            _fail("human review requires a complete mechanical PASS pilot")
        if audit.get("review_count") != 0:
            _fail("resolve REVIEW rows before suspicious-size human review")
        flagged = list(audit.get("suspicious_too_large_sample_ids") or [])
        flagged += [x for x in (audit.get("suspicious_too_small_sample_ids") or []) if x not in flagged]
        if not flagged:
            _fail("no suspicious pilot samples require human review")

        self.selection_path = self.annotations_dir / REVIEW_SELECTION_NAME
        audit_bytes = self.audit_path.read_bytes()
        base_audit_sha = hashlib.sha256(audit_bytes).hexdigest()
        if self.selection_path.exists():
            selection = _load_audit(self.selection_path)
            selected_ids = selection.get("sample_ids")
            if not isinstance(selected_ids, list) or not all(isinstance(x, str) for x in selected_ids):
                _fail("human review selection is malformed")
            flagged = selected_ids
        else:
            _atomic_write_json(self.selection_path, {
                "schema": REVIEW_SCHEMA,
                "base_audit_sha256": base_audit_sha,
                "sample_ids": flagged,
                "count": len(flagged),
                "scope": "train_pilot_flagged_only",
                "final_holdout_locked": True,
                "training_authorized": False,
            })

        pilot_index = {s.sample_id: s.index for s in self.session.samples}
        if any(sample_id not in pilot_index for sample_id in flagged):
            _fail("human review selection contains sample outside pilot")
        self.items = tuple(
            ReviewItem(i, pilot_index[sample_id], sample_id)
            for i, sample_id in enumerate(flagged)
        )
        self.review_path = self.annotations_dir / REVIEW_CSV_NAME
        self.resolutions = self._load_resolutions()

    def _load_resolutions(self) -> dict[str, dict[str, str]]:
        rows = _read_csv(self.review_path)
        allowed = {x.sample_id for x in self.items}
        result: dict[str, dict[str, str]] = {}
        for row in rows:
            sample_id = row["sample_id"]
            if sample_id not in allowed:
                _fail(f"human review row outside frozen selection: {sample_id}")
            if sample_id in result:
                _fail(f"duplicate human review row: {sample_id}")
            if row["action"] not in ACTIONS:
                _fail(f"invalid human review action: {sample_id}")
            current = self.session.annotations.get(sample_id)
            if current is None or current.get("status") != "PASS":
                _fail(f"resolved sample no longer has PASS bbox: {sample_id}")
            if row["image_sha256"] != current["image_sha256"]:
                _fail(f"human review image binding mismatch: {sample_id}")
            if row["bbox_binding_sha256"] != _bbox_binding(current):
                _fail(f"human review bbox changed after resolution: {sample_id}")
            result[sample_id] = row
        return result

    @property
    def resolved_count(self) -> int:
        return len(self.resolutions)

    def resume_index(self) -> int:
        for item in self.items:
            if item.sample_id not in self.resolutions:
                return item.review_index
        return max(0, len(self.items) - 1)

    def sample_payload(self, review_index: int) -> dict[str, object]:
        if type(review_index) is not int or not 0 <= review_index < len(self.items):
            _fail("human review index outside selection")
        item = self.items[review_index]
        payload = self.session.sample_payload(item.pilot_index)
        payload.update({
            "review_index": item.review_index,
            "review_total": len(self.items),
            "human_resolution": self.resolutions.get(item.sample_id, {}).get("action"),
        })
        return payload

    def _persist_resolution(self, sample_id: str, action: str) -> dict[str, object]:
        if action not in ACTIONS:
            _fail("invalid resolution action")
        current = self.session.annotations.get(sample_id)
        if current is None or current.get("status") != "PASS":
            _fail("resolution requires current PASS bbox")
        self.resolutions[sample_id] = {
            "sample_id": sample_id,
            "action": action,
            "bbox_binding_sha256": _bbox_binding(current),
            "image_sha256": current["image_sha256"],
            "reviewed_utc": _utc_now(),
        }
        ordered = [self.resolutions[x.sample_id] for x in self.items if x.sample_id in self.resolutions]
        _atomic_write_csv(self.review_path, ordered)
        return {"sample_id": sample_id, "action": action, "resolved_count": self.resolved_count}

    def accept_as_drawn(self, *, review_index: int) -> dict[str, object]:
        item = self.items[review_index]
        return self._persist_resolution(item.sample_id, "ACCEPT_AS_DRAWN")

    def redraw_and_accept(self, *, review_index: int, token: object,
                          x0: object, y0: object, x1: object, y1: object,
                          preview_width: object, preview_height: object) -> dict[str, object]:
        item = self.items[review_index]
        result = self.session.save_from_preview(
            token=token, x0=x0, y0=y0, x1=x1, y1=y1,
            preview_width=preview_width, preview_height=preview_height,
        )
        if result["sample_id"] != item.sample_id:
            _fail("redraw token does not match current human review item")
        return self._persist_resolution(item.sample_id, "REDRAWN_AND_ACCEPTED")


def write_human_review_audit(data_root: str | Path) -> Path:
    session = HumanReviewSession(data_root=data_root)
    base_audit = _load_audit(session.audit_path)
    selected = [x.sample_id for x in session.items]
    resolved = [x for x in selected if x in session.resolutions]
    unresolved = [x for x in selected if x not in session.resolutions]
    gate = (
        base_audit.get("mechanical_gate") == "PASS"
        and base_audit.get("annotation_count") == 30
        and base_audit.get("pass_count") == 30
        and base_audit.get("review_count") == 0
        and not unresolved
    )
    payload = {
        "schema": REVIEW_SCHEMA,
        "dataset": base_audit.get("dataset"),
        "selected_count": len(selected),
        "selected_sample_ids": selected,
        "resolved_count": len(resolved),
        "unresolved_count": len(unresolved),
        "unresolved_sample_ids": unresolved,
        "actions": {
            "ACCEPT_AS_DRAWN": sum(
                session.resolutions[x]["action"] == "ACCEPT_AS_DRAWN" for x in resolved
            ),
            "REDRAWN_AND_ACCEPTED": sum(
                session.resolutions[x]["action"] == "REDRAWN_AND_ACCEPTED" for x in resolved
            ),
        },
        "mechanical_gate": base_audit.get("mechanical_gate"),
        "human_review_gate": "PASS" if gate else "HOLD",
        "annotation_contract_freeze_ready": bool(gate),
        "final_holdout_locked": True,
        "training_authorized": False,
        "model_opened": False,
        "inference_count": 0,
    }
    path = Path(data_root) / ANNOTATIONS_DIR / REVIEW_AUDIT_NAME
    _atomic_write_json(path, payload)
    return path
