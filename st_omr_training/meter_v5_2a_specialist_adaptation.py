"""METER V5-2A: 300-TRAIN human full-meter BBox gate for 2/3 adaptation.

This module opens only TRAIN annotation. It does not derive digit boxes, load a
model, train, tune thresholds, read VAL/final_holdout, or mutate V5-1 seeds.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from st_omr_training import meter_v5_1_bbox_pilot as v51
from st_omr_training import meter_v5_2_train_bbox_scale as v52

TRAIN_PER_CLASS = 100
TRAIN_TOTAL = 300
SEED_TOTAL = 30
SEED_PER_CLASS = 10
REMAINING_NEW = 270
NEW_PER_CLASS = 90

EXPECTED_DATASET_FINGERPRINT = v52.EXPECTED_DATASET_FINGERPRINT
EXPECTED_SEED_SHA256 = v52.EXPECTED_SEED_SHA256

SELECTION_NAME = "bbox_adaptation_300_selection.csv"
ANNOTATION_NAME = "bbox_adaptation_300.csv"
AUDIT_NAME = "bbox_adaptation_300_audit.json"

SELECTION_COLUMNS = (
    "index", "sample_id", "family_id", "meter", "split", "folder",
    "image_relpath", "image_sha256", "image_width", "image_height",
    "seed_annotation", "selection_rank",
)


class MeterV5_2AError(v52.MeterV5_2ScaleError):
    """Fail-closed V5-2A annotation error."""


def _fail(message: str) -> None:
    raise MeterV5_2AError(message)


def _read_csv(path: Path) -> list[dict[str, str]]:
    return v51._read_csv(path)


def _write_csv(path: Path, fields, rows) -> None:
    v51._atomic_write_csv(path, fields, rows)


def _source_rank(row: Mapping[str, str]) -> str:
    return row.get("SplitRank") or row.get("SelectionRank") or ""


@dataclass(frozen=True)
class AdaptationSample:
    index: int
    sample_id: str
    family_id: str
    meter: str
    image_path: Path
    image_sha256: str
    image_width: int
    image_height: int
    seed_annotation: bool
    selection_rank: str


def verify_seed_evidence(
    data_root: str | Path,
    *,
    expected_dataset_fingerprint: str = EXPECTED_DATASET_FINGERPRINT,
    expected_seed_sha256: Mapping[str, str] = EXPECTED_SEED_SHA256,
) -> dict[str, object]:
    evidence = v52.verify_seed_evidence(
        data_root,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
        expected_seed_sha256=expected_seed_sha256,
    )
    counts = Counter(r["meter"] for r in evidence["selection_rows"])
    expected = Counter({meter: SEED_PER_CLASS for meter in v51.CLASSES})
    if counts != expected:
        _fail(f"V5-1 seed class balance must be 10/10/10, got {dict(counts)}")
    return evidence


def load_or_create_selection(
    data_root: str | Path,
    seed: Mapping[str, object],
) -> tuple[AdaptationSample, ...]:
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    ann_dir.mkdir(parents=True, exist_ok=True)
    path = ann_dir / SELECTION_NAME

    seed_rows = list(seed["selection_rows"])
    seed_ids = [r["sample_id"] for r in seed_rows]
    seed_id_set = set(seed_ids)

    if not path.exists():
        manifest_by_id: dict[str, dict[str, str]] = {}
        chosen_new: list[dict[str, str]] = []

        for meter in v51.CLASSES:
            rows = v51._validate_manifest(root / v51.MANIFEST_NAME[meter], meter)
            train_rows = [r for r in rows if r["Split"] == "train"]
            if len(train_rows) != 400:
                _fail(f"{meter}: expected 400 TRAIN rows, got {len(train_rows)}")
            for row in train_rows:
                sid = row["SampleId"]
                if sid in manifest_by_id:
                    _fail(f"duplicate TRAIN sample identity: {sid}")
                manifest_by_id[sid] = row

            eligible = [r for r in train_rows if r["SampleId"] not in seed_id_set]
            eligible.sort(key=lambda r: (_source_rank(r), r["SampleId"]))
            if len(eligible) < NEW_PER_CLASS:
                _fail(f"{meter}: fewer than {NEW_PER_CLASS} non-seed TRAIN rows")
            chosen_new.extend(eligible[:NEW_PER_CLASS])

        if not seed_id_set.issubset(manifest_by_id):
            _fail("one or more V5-1 seeds are absent from TRAIN manifests")

        ordered_ids = list(seed_ids) + [r["SampleId"] for r in chosen_new]
        if len(ordered_ids) != TRAIN_TOTAL or len(set(ordered_ids)) != TRAIN_TOTAL:
            _fail("V5-2A selection is not exactly 300 unique samples")

        output = []
        seed_sel_by_id = {r["sample_id"]: r for r in seed_rows}
        for index, sid in enumerate(ordered_ids):
            row = manifest_by_id[sid]
            meter = row["Meter"]
            image_path = root / "train" / v51.CLASS_DIR[meter] / row["Folder"] / "image.png"
            sha, width, height = v51._read_png_binding(image_path)
            if sid in seed_id_set:
                old = seed_sel_by_id[sid]
                if (
                    sha != old["image_sha256"]
                    or width != int(old["image_width"])
                    or height != int(old["image_height"])
                ):
                    _fail(f"seed image binding changed: {sid}")
            output.append({
                "index": index,
                "sample_id": sid,
                "family_id": row["FamilyId"],
                "meter": meter,
                "split": "train",
                "folder": row["Folder"],
                "image_relpath": image_path.relative_to(root).as_posix(),
                "image_sha256": sha,
                "image_width": width,
                "image_height": height,
                "seed_annotation": "1" if sid in seed_id_set else "0",
                "selection_rank": _source_rank(row),
            })
        _write_csv(path, SELECTION_COLUMNS, output)

    rows = _read_csv(path)
    if len(rows) != TRAIN_TOTAL:
        _fail(f"V5-2A selection must contain 300 rows, got {len(rows)}")
    if len({r["sample_id"] for r in rows}) != TRAIN_TOTAL:
        _fail("duplicate sample_id in V5-2A selection")
    if len({r["family_id"] for r in rows}) != TRAIN_TOTAL:
        _fail("duplicate family_id in V5-2A selection")
    if Counter(r["meter"] for r in rows) != Counter({m: TRAIN_PER_CLASS for m in v51.CLASSES}):
        _fail("V5-2A selection class balance must be 100/100/100")
    if any(r["split"] != "train" for r in rows):
        _fail("V5-2A selection must be TRAIN-only")
    if [r["sample_id"] for r in rows[:SEED_TOTAL]] != seed_ids:
        _fail("V5-2A selection does not preserve the 30 immutable seeds first")
    if any(r["seed_annotation"] != "1" for r in rows[:SEED_TOTAL]):
        _fail("V5-2A seed flags missing")
    if any(r["seed_annotation"] != "0" for r in rows[SEED_TOTAL:]):
        _fail("V5-2A non-seed row incorrectly marked seed")

    samples = []
    for expected_index, row in enumerate(rows):
        try:
            index = int(row["index"])
            width = int(row["image_width"])
            height = int(row["image_height"])
        except (KeyError, ValueError) as exc:
            raise MeterV5_2AError("invalid V5-2A selection numeric field") from exc
        if index != expected_index:
            _fail("V5-2A selection index order mismatch")
        image_path = root / row["image_relpath"]
        sha, current_width, current_height = v51._read_png_binding(image_path)
        if (sha, current_width, current_height) != (row["image_sha256"], width, height):
            _fail(f"V5-2A image binding changed: {row['sample_id']}")
        samples.append(AdaptationSample(
            index=index,
            sample_id=row["sample_id"],
            family_id=row["family_id"],
            meter=row["meter"],
            image_path=image_path,
            image_sha256=sha,
            image_width=width,
            image_height=height,
            seed_annotation=row["seed_annotation"] == "1",
            selection_rank=row["selection_rank"],
        ))
    return tuple(samples)


class AdaptationAnnotationSession:
    """Crash-safe 300-TRAIN full-meter BBox annotation session."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        expected_dataset_fingerprint: str = EXPECTED_DATASET_FINGERPRINT,
        expected_seed_sha256: Mapping[str, str] = EXPECTED_SEED_SHA256,
    ):
        self.data_root = Path(data_root)
        self.seed = verify_seed_evidence(
            self.data_root,
            expected_dataset_fingerprint=expected_dataset_fingerprint,
            expected_seed_sha256=expected_seed_sha256,
        )
        self.samples = load_or_create_selection(self.data_root, self.seed)
        self.sample_by_id = {s.sample_id: s for s in self.samples}
        self.seed_ids = set(self.seed["seed_sample_ids"])
        self.selection_path = self.data_root / v51.ANNOTATIONS_DIR / SELECTION_NAME
        self.selection_sha256 = v51._sha256_file(self.selection_path)
        self.annotation_path = self.data_root / v51.ANNOTATIONS_DIR / ANNOTATION_NAME
        self.seed_annotation_rows = {
            r["sample_id"]: dict(r) for r in self.seed["annotation_rows"]
        }
        self.annotations = self._load_or_seed_annotations()

    def _validate_annotation_row(self, row: Mapping[str, str]) -> None:
        sid = row.get("sample_id", "")
        sample = self.sample_by_id.get(sid)
        if sample is None:
            _fail(f"annotation outside V5-2A selection: {sid}")
        if row.get("meter") != sample.meter or row.get("split") != "train":
            _fail(f"annotation identity mismatch: {sid}")
        if row.get("image_sha256") != sample.image_sha256:
            _fail(f"annotation image SHA mismatch: {sid}")
        try:
            if int(row.get("image_width", "")) != sample.image_width:
                _fail(f"annotation image width mismatch: {sid}")
            if int(row.get("image_height", "")) != sample.image_height:
                _fail(f"annotation image height mismatch: {sid}")
        except ValueError as exc:
            raise MeterV5_2AError(f"invalid annotation dimensions: {sid}") from exc
        status = row.get("status")
        if status not in {"PASS", "REVIEW"}:
            _fail(f"invalid annotation status: {sid}")
        if status == "PASS":
            try:
                x, y, w, h = (int(row[k]) for k in ("x", "y", "w", "h"))
            except (KeyError, ValueError) as exc:
                raise MeterV5_2AError(f"invalid bbox integers: {sid}") from exc
            v51._validate_bbox(x, y, w, h, sample.image_width, sample.image_height)
        elif any(row.get(k, "") for k in ("x", "y", "w", "h")):
            _fail(f"REVIEW row must not contain bbox: {sid}")

    def _load_or_seed_annotations(self) -> dict[str, dict[str, str]]:
        if not self.annotation_path.exists():
            rows = []
            for sid in self.seed["seed_sample_ids"]:
                row = dict(self.seed_annotation_rows[sid])
                self._validate_annotation_row(row)
                rows.append(row)
            _write_csv(self.annotation_path, v51.ANNOTATION_COLUMNS, rows)

        result = {}
        for row in _read_csv(self.annotation_path):
            sid = row.get("sample_id", "")
            if sid in result:
                _fail(f"duplicate V5-2A annotation: {sid}")
            self._validate_annotation_row(row)
            result[sid] = dict(row)

        if not self.seed_ids.issubset(result):
            _fail("V5-2A annotation file is missing immutable seeds")
        for sid in self.seed_ids:
            if result[sid] != self.seed_annotation_rows[sid]:
                _fail(f"immutable V5-1 seed changed: {sid}")
        return result

    @property
    def handled_count(self) -> int:
        return len(self.annotations)

    @property
    def pass_count(self) -> int:
        return sum(r["status"] == "PASS" for r in self.annotations.values())

    @property
    def review_count(self) -> int:
        return sum(r["status"] == "REVIEW" for r in self.annotations.values())

    def resume_index(self) -> int:
        for sample in self.samples:
            if sample.sample_id not in self.annotations:
                return sample.index
        return TRAIN_TOTAL - 1

    def _token(self, sample: AdaptationSample) -> str:
        payload = (
            f"{self.selection_sha256}\n{sample.index}\n{sample.sample_id}\n"
            f"{sample.image_sha256}\n{sample.image_width}x{sample.image_height}\n"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _resolve_token(self, token: object) -> AdaptationSample:
        if not isinstance(token, str) or len(token) != 64:
            _fail("sample binding token malformed")
        for sample in self.samples:
            if self._token(sample) == token:
                sha, width, height = v51._read_png_binding(sample.image_path)
                if (sha, width, height) != (
                    sample.image_sha256, sample.image_width, sample.image_height
                ):
                    _fail(f"original image changed: {sample.sample_id}")
                return sample
        _fail("sample binding token does not match V5-2A selection")

    def sample_payload(self, index: int) -> dict[str, object]:
        if type(index) is not int or not 0 <= index < TRAIN_TOTAL:
            _fail("V5-2A index outside 0..299")
        sample = self.samples[index]
        sha, width, height = v51._read_png_binding(sample.image_path)
        if (sha, width, height) != (
            sample.image_sha256, sample.image_width, sample.image_height
        ):
            _fail(f"original image changed: {sample.sample_id}")
        data_uri, pw, ph = v51._preview(sample.image_path)
        existing = self.annotations.get(sample.sample_id)
        bbox = None
        status = None
        if existing:
            status = existing["status"]
            if status == "PASS":
                bbox = {k: int(existing[k]) for k in ("x", "y", "w", "h")}
        return {
            "index": sample.index,
            "total": TRAIN_TOTAL,
            "sample_id": sample.sample_id,
            "family_id": sample.family_id,
            "meter": sample.meter,
            "split": "train",
            "image_width": sample.image_width,
            "image_height": sample.image_height,
            "preview_width": pw,
            "preview_height": ph,
            "preview_data_uri": data_uri,
            "binding_token": self._token(sample),
            "bbox": bbox,
            "status": status,
            "locked_seed": sample.seed_annotation,
            "handled_count": self.handled_count,
            "pass_count": self.pass_count,
            "review_count": self.review_count,
            "remaining_unhandled": TRAIN_TOTAL - self.handled_count,
            "final_holdout_locked": True,
            "training_authorized": False,
        }

    def _persist(self) -> None:
        ordered = [
            self.annotations[s.sample_id]
            for s in self.samples
            if s.sample_id in self.annotations
        ]
        _write_csv(self.annotation_path, v51.ANNOTATION_COLUMNS, ordered)
        handled = len(ordered)
        if handled in {30, 100, 200, 300}:
            backup = self.annotation_path.with_name(
                f"bbox_adaptation_300.backup_{handled:03d}.csv"
            )
            _write_csv(backup, v51.ANNOTATION_COLUMNS, ordered)

    def _require_mutable(self, sample: AdaptationSample) -> None:
        if sample.seed_annotation or sample.sample_id in self.seed_ids:
            _fail("accepted V5-1 seed BBox is immutable in V5-2A")

    def save_from_preview(
        self, *, token: object,
        x0: object, y0: object, x1: object, y1: object,
        preview_width: object, preview_height: object,
    ) -> dict[str, object]:
        sample = self._resolve_token(token)
        self._require_mutable(sample)
        x0i = v51._parse_int(x0, "x0")
        y0i = v51._parse_int(y0, "y0")
        x1i = v51._parse_int(x1, "x1")
        y1i = v51._parse_int(y1, "y1")
        pwi = v51._parse_int(preview_width, "preview_width")
        phi = v51._parse_int(preview_height, "preview_height")
        _data, expected_pw, expected_ph = v51._preview(sample.image_path)
        if (pwi, phi) != (expected_pw, expected_ph):
            _fail("preview dimensions changed; reload before save")
        x, y, w, h = v51._rect_to_original(
            x0=x0i, y0=y0i, x1=x1i, y1=y1i,
            preview_width=pwi, preview_height=phi,
            image_width=sample.image_width, image_height=sample.image_height,
        )
        self.annotations[sample.sample_id] = {
            "sample_id": sample.sample_id,
            "meter": sample.meter,
            "split": "train",
            "x": str(x), "y": str(y), "w": str(w), "h": str(h),
            "status": "PASS",
            "image_sha256": sample.image_sha256,
            "image_width": str(sample.image_width),
            "image_height": str(sample.image_height),
            "updated_utc": v51._utc_now(),
        }
        self._persist()
        return {
            "sample_id": sample.sample_id,
            "status": "PASS",
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "handled_count": self.handled_count,
            "pass_count": self.pass_count,
            "review_count": self.review_count,
        }

    def mark_review(self, *, token: object) -> dict[str, object]:
        sample = self._resolve_token(token)
        self._require_mutable(sample)
        self.annotations[sample.sample_id] = {
            "sample_id": sample.sample_id,
            "meter": sample.meter,
            "split": "train",
            "x": "", "y": "", "w": "", "h": "",
            "status": "REVIEW",
            "image_sha256": sample.image_sha256,
            "image_width": str(sample.image_width),
            "image_height": str(sample.image_height),
            "updated_utc": v51._utc_now(),
        }
        self._persist()
        return {
            "sample_id": sample.sample_id,
            "status": "REVIEW",
            "handled_count": self.handled_count,
            "pass_count": self.pass_count,
            "review_count": self.review_count,
        }


def write_annotation_audit(
    data_root: str | Path,
    *,
    expected_dataset_fingerprint: str = EXPECTED_DATASET_FINGERPRINT,
    expected_seed_sha256: Mapping[str, str] = EXPECTED_SEED_SHA256,
) -> Path:
    session = AdaptationAnnotationSession(
        data_root=data_root,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
        expected_seed_sha256=expected_seed_sha256,
    )
    rows = list(session.annotations.values())
    pass_rows = [r for r in rows if r["status"] == "PASS"]
    review_rows = [r for r in rows if r["status"] == "REVIEW"]
    per_class = {
        meter: {
            "PASS": sum(r["meter"] == meter and r["status"] == "PASS" for r in rows),
            "REVIEW": sum(r["meter"] == meter and r["status"] == "REVIEW" for r in rows),
        }
        for meter in v51.CLASSES
    }
    seed_mutations = sum(
        session.annotations.get(sid) != session.seed_annotation_rows[sid]
        for sid in session.seed_ids
    )
    family_ids = [session.sample_by_id[r["sample_id"]].family_id for r in rows]
    mechanical_pass = (
        len(rows) == TRAIN_TOTAL
        and len(pass_rows) == TRAIN_TOTAL
        and not review_rows
        and per_class == {
            meter: {"PASS": TRAIN_PER_CLASS, "REVIEW": 0}
            for meter in v51.CLASSES
        }
        and len({r["sample_id"] for r in rows}) == TRAIN_TOTAL
        and len(set(family_ids)) == TRAIN_TOTAL
        and seed_mutations == 0
    )

    payload = {
        "schema": "st-omr-meter-v5-2a-annotation-audit-v1",
        "dataset": v51.DATASET_NAME,
        "dataset_fingerprint_sha256": expected_dataset_fingerprint,
        "train_total_expected": TRAIN_TOTAL,
        "annotation_count": len(rows),
        "pass_count": len(pass_rows),
        "review_count": len(review_rows),
        "per_class": per_class,
        "missing_annotation_count": TRAIN_TOTAL - len(rows),
        "seed_mutation_count": seed_mutations,
        "mechanical_gate": "PASS" if mechanical_pass else "HOLD",
        "human_visual_review_required": True,
        "training_authorized": False,
        "slot_derivation_authorized_after_human_qa": mechanical_pass,
        "trainable_specialists": ["2-AI", "3-AI"],
        "frozen_control_specialist": "4-AI",
        "threshold_tuning_allowed": False,
        "single_three_class_model_allowed": False,
        "validation_opened": False,
        "final_holdout_locked": True,
        "model_opened": False,
        "inference_count": 0,
    }
    path = session.data_root / v51.ANNOTATIONS_DIR / AUDIT_NAME
    v51._atomic_write_json(path, payload)
    return path
