"""METER V5-2 train-only full-meter BBox scale annotation.

Canonical continuation of V5-1. The accepted 30 human full-meter boxes are
immutable seeds. This module never derives digit boxes, never runs a model, and
never opens VAL/final_holdout images for annotation.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from st_omr_training import meter_v5_1_bbox_pilot as v51

TRAIN_PER_CLASS = 400
TRAIN_TOTAL = 1200
SEED_TOTAL = 30
REMAINING_NEW = TRAIN_TOTAL - SEED_TOTAL

EXPECTED_DATASET_FINGERPRINT = "e7e849ac6d6d7a622dc94107a4dc4074c48e1e0ab837e726857394cb4072b8f0"
EXPECTED_SEED_SHA256 = {
    "annotation_csv": "b60a953811aa136752372d7c8cea6fe7a1c1c964a62bd53c0c9d48c56c735665",
    "selection_csv": "4070f46f64efed5b12b26f7dd1d4e3f09b4abf804125140d38a212819bbcbe97",
    "audit_json": "dd254ba7c7408a73168d52da3b50de2de8eafa4748e3f4e64e4d11707eeae366",
    "holdout_lock_json": "51923443c1cc892a2f8852a52ef7658bc604b6fa948a3cc43f4f8b6dab3cc2e3",
}

SELECTION_NAME = "bbox_train_1200_selection.csv"
ANNOTATION_NAME = "bbox_train_1200.csv"
AUDIT_NAME = "bbox_train_1200_audit.json"

SELECTION_COLUMNS = (
    "index", "sample_id", "family_id", "meter", "split", "folder",
    "image_relpath", "image_sha256", "image_width", "image_height",
    "seed_annotation", "selection_rank",
)


class MeterV5_2ScaleError(v51.MeterV5_1PilotError):
    """Fail-closed V5-2 scale-annotation error."""


def _fail(message: str) -> None:
    raise MeterV5_2ScaleError(message)


def _sha256_file(path: Path) -> str:
    return v51._sha256_file(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    return v51._read_csv(path)


def _write_csv(path: Path, fields, rows) -> None:
    v51._atomic_write_csv(path, fields, rows)


def _write_json(path: Path, value: object) -> None:
    v51._atomic_write_json(path, value)


def _source_rank(row: Mapping[str, str]) -> str:
    return row.get("SplitRank") or row.get("SelectionRank") or ""


@dataclass(frozen=True)
class TrainSample:
    index: int
    sample_id: str
    family_id: str
    meter: str
    split: str
    folder: str
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
    """Verify V5-1 pilot evidence before V5-2 can open."""
    root = Path(data_root)
    gate = v51.verify_dataset_structure(root)
    if gate["dataset_fingerprint_sha256"] != expected_dataset_fingerprint:
        _fail("clean dataset fingerprint mismatch")

    ann_dir = root / v51.ANNOTATIONS_DIR
    paths = {
        "annotation_csv": ann_dir / v51.PILOT_CSV_NAME,
        "selection_csv": ann_dir / v51.PILOT_SELECTION_NAME,
        "audit_json": ann_dir / v51.PILOT_AUDIT_NAME,
        "holdout_lock_json": ann_dir / v51.FINAL_HOLDOUT_LOCK_NAME,
    }
    for key, path in paths.items():
        if not path.is_file():
            _fail(f"missing V5-1 seed evidence: {path}")
        expected = expected_seed_sha256.get(key)
        if not expected:
            _fail(f"missing expected seed hash for {key}")
        actual = _sha256_file(path)
        if actual != expected:
            _fail(f"V5-1 seed {key} SHA mismatch")

    selection_rows = _read_csv(paths["selection_csv"])
    annotation_rows = _read_csv(paths["annotation_csv"])
    if len(selection_rows) != SEED_TOTAL or len(annotation_rows) != SEED_TOTAL:
        _fail("V5-1 seed evidence must contain exactly 30 rows")
    if len({r["sample_id"] for r in selection_rows}) != SEED_TOTAL:
        _fail("duplicate sample_id in V5-1 seed selection")
    if len({r["sample_id"] for r in annotation_rows}) != SEED_TOTAL:
        _fail("duplicate sample_id in V5-1 seed annotations")
    if any(r.get("split") != "train" for r in selection_rows + annotation_rows):
        _fail("V5-1 seed evidence must be TRAIN-only")
    if any(r.get("status") != "PASS" for r in annotation_rows):
        _fail("all V5-1 seed annotations must be PASS")

    audit = json.loads(paths["audit_json"].read_text(encoding="utf-8-sig"))
    if (
        audit.get("annotation_count") != SEED_TOTAL
        or audit.get("pass_count") != SEED_TOTAL
        or audit.get("review_count") != 0
        or audit.get("annotation_contract_freeze_ready") is not True
    ):
        _fail("V5-1 pilot audit is not a completed 30/30 PASS freeze")

    lock = json.loads(paths["holdout_lock_json"].read_text(encoding="utf-8-sig"))
    if (
        lock.get("locked") is not True
        or lock.get("annotation_opened") is not False
        or lock.get("training_opened") is not False
        or lock.get("model_evaluated") is not False
    ):
        _fail("FINAL_HOLDOUT lock is not in the required closed state")

    sel_by_id = {r["sample_id"]: r for r in selection_rows}
    ann_by_id = {r["sample_id"]: r for r in annotation_rows}
    if set(sel_by_id) != set(ann_by_id):
        _fail("V5-1 selection/annotation sample identities differ")
    for sample_id, ann in ann_by_id.items():
        sel = sel_by_id[sample_id]
        if ann.get("meter") != sel.get("meter"):
            _fail(f"seed meter mismatch: {sample_id}")
        if ann.get("image_sha256") != sel.get("image_sha256"):
            _fail(f"seed image SHA mismatch: {sample_id}")
        try:
            width = int(sel["image_width"])
            height = int(sel["image_height"])
            x, y, w, h = (int(ann[k]) for k in ("x", "y", "w", "h"))
        except (KeyError, ValueError) as exc:
            raise MeterV5_2ScaleError(f"invalid seed numeric field: {sample_id}") from exc
        v51._validate_bbox(x, y, w, h, width, height)

    return {
        "gate": gate,
        "paths": paths,
        "selection_rows": selection_rows,
        "annotation_rows": annotation_rows,
        "seed_sample_ids": tuple(r["sample_id"] for r in selection_rows),
    }


def load_or_create_train_selection(
    data_root: str | Path,
    seed: Mapping[str, object],
) -> tuple[TrainSample, ...]:
    """Bind all 1,200 TRAIN images, preserving the 30 seed order first."""
    root = Path(data_root)
    ann_dir = root / v51.ANNOTATIONS_DIR
    ann_dir.mkdir(parents=True, exist_ok=True)
    path = ann_dir / SELECTION_NAME

    seed_rows = list(seed["selection_rows"])
    seed_ids = [r["sample_id"] for r in seed_rows]
    seed_id_set = set(seed_ids)

    if not path.exists():
        manifest_by_id: dict[str, dict[str, str]] = {}
        for meter in v51.CLASSES:
            rows = v51._validate_manifest(root / v51.MANIFEST_NAME[meter], meter)
            train_rows = [r for r in rows if r["Split"] == "train"]
            if len(train_rows) != TRAIN_PER_CLASS:
                _fail(f"{meter}: expected 400 TRAIN rows, got {len(train_rows)}")
            for row in train_rows:
                sample_id = row["SampleId"]
                if sample_id in manifest_by_id:
                    _fail(f"duplicate TRAIN sample identity: {sample_id}")
                manifest_by_id[sample_id] = row

        if len(manifest_by_id) != TRAIN_TOTAL:
            _fail(f"expected 1200 unique TRAIN samples, got {len(manifest_by_id)}")
        if not seed_id_set.issubset(manifest_by_id):
            _fail("one or more V5-1 seed samples are absent from TRAIN manifests")

        ordered_ids = list(seed_ids)
        remaining_rows = [
            row for sample_id, row in manifest_by_id.items()
            if sample_id not in seed_id_set
        ]
        meter_order = {meter: i for i, meter in enumerate(v51.CLASSES)}
        remaining_rows.sort(
            key=lambda row: (
                meter_order[row["Meter"]],
                _source_rank(row),
                row["SampleId"],
            )
        )
        ordered_ids.extend(row["SampleId"] for row in remaining_rows)

        if len(ordered_ids) != TRAIN_TOTAL or len(set(ordered_ids)) != TRAIN_TOTAL:
            _fail("V5-2 TRAIN ordering is not exactly 1200 unique samples")

        output = []
        seed_sel_by_id = {r["sample_id"]: r for r in seed_rows}
        for index, sample_id in enumerate(ordered_ids):
            row = manifest_by_id[sample_id]
            meter = row["Meter"]
            image_path = root / "train" / v51.CLASS_DIR[meter] / row["Folder"] / "image.png"
            sha, width, height = v51._read_png_binding(image_path)
            if sample_id in seed_id_set:
                old = seed_sel_by_id[sample_id]
                if (
                    sha != old["image_sha256"]
                    or width != int(old["image_width"])
                    or height != int(old["image_height"])
                ):
                    _fail(f"seed image binding changed: {sample_id}")
            output.append({
                "index": index,
                "sample_id": sample_id,
                "family_id": row["FamilyId"],
                "meter": meter,
                "split": "train",
                "folder": row["Folder"],
                "image_relpath": image_path.relative_to(root).as_posix(),
                "image_sha256": sha,
                "image_width": width,
                "image_height": height,
                "seed_annotation": "1" if sample_id in seed_id_set else "0",
                "selection_rank": _source_rank(row),
            })
        _write_csv(path, SELECTION_COLUMNS, output)

    rows = _read_csv(path)
    if len(rows) != TRAIN_TOTAL:
        _fail(f"V5-2 selection must contain 1200 rows, got {len(rows)}")
    if len({r["sample_id"] for r in rows}) != TRAIN_TOTAL:
        _fail("duplicate sample_id in V5-2 selection")
    if len({r["family_id"] for r in rows}) != TRAIN_TOTAL:
        _fail("duplicate family_id in V5-2 selection")
    if Counter(r["meter"] for r in rows) != Counter({m: TRAIN_PER_CLASS for m in v51.CLASSES}):
        _fail("V5-2 selection class balance must be 400/400/400")
    if any(r["split"] != "train" for r in rows):
        _fail("V5-2 selection must be TRAIN-only")
    if [r["sample_id"] for r in rows[:SEED_TOTAL]] != seed_ids:
        _fail("V5-2 selection does not preserve the 30 seed order first")
    if any(r["seed_annotation"] != "1" for r in rows[:SEED_TOTAL]):
        _fail("V5-2 seed flags missing from first 30 rows")
    if any(r["seed_annotation"] != "0" for r in rows[SEED_TOTAL:]):
        _fail("V5-2 non-seed rows incorrectly marked as seeds")

    samples = []
    for expected_index, row in enumerate(rows):
        try:
            index = int(row["index"])
            width = int(row["image_width"])
            height = int(row["image_height"])
        except (KeyError, ValueError) as exc:
            raise MeterV5_2ScaleError("invalid V5-2 selection integer field") from exc
        if index != expected_index:
            _fail("V5-2 selection index order mismatch")
        image_path = root / row["image_relpath"]
        sha, current_width, current_height = v51._read_png_binding(image_path)
        if (sha, current_width, current_height) != (row["image_sha256"], width, height):
            _fail(f"V5-2 image binding changed: {row['sample_id']}")
        samples.append(TrainSample(
            index=index,
            sample_id=row["sample_id"],
            family_id=row["family_id"],
            meter=row["meter"],
            split="train",
            folder=row["folder"],
            image_path=image_path,
            image_sha256=sha,
            image_width=width,
            image_height=height,
            seed_annotation=row["seed_annotation"] == "1",
            selection_rank=row["selection_rank"],
        ))
    return tuple(samples)


class ScaleAnnotationSession:
    """Crash-safe 1,200-TRAIN full-meter BBox annotation session."""

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
        self.samples = load_or_create_train_selection(self.data_root, self.seed)
        self.sample_by_id = {s.sample_id: s for s in self.samples}
        self.seed_ids = set(self.seed["seed_sample_ids"])
        self.selection_path = self.data_root / v51.ANNOTATIONS_DIR / SELECTION_NAME
        self.selection_sha256 = _sha256_file(self.selection_path)
        self.annotation_path = self.data_root / v51.ANNOTATIONS_DIR / ANNOTATION_NAME
        self.seed_annotation_rows = {
            r["sample_id"]: dict(r) for r in self.seed["annotation_rows"]
        }
        self.annotations = self._load_or_seed_annotations()

    def _validate_annotation_row(self, row: Mapping[str, str]) -> None:
        sample_id = row.get("sample_id", "")
        sample = self.sample_by_id.get(sample_id)
        if sample is None:
            _fail(f"annotation outside V5-2 TRAIN selection: {sample_id}")
        if row.get("meter") != sample.meter or row.get("split") != "train":
            _fail(f"annotation identity mismatch: {sample_id}")
        if row.get("image_sha256") != sample.image_sha256:
            _fail(f"annotation image SHA mismatch: {sample_id}")
        try:
            if int(row.get("image_width", "")) != sample.image_width:
                _fail(f"annotation image width mismatch: {sample_id}")
            if int(row.get("image_height", "")) != sample.image_height:
                _fail(f"annotation image height mismatch: {sample_id}")
        except ValueError as exc:
            raise MeterV5_2ScaleError(f"invalid annotation image dimensions: {sample_id}") from exc
        status = row.get("status")
        if status not in {"PASS", "REVIEW"}:
            _fail(f"invalid annotation status: {sample_id}")
        if status == "PASS":
            try:
                x, y, w, h = (int(row[k]) for k in ("x", "y", "w", "h"))
            except (KeyError, ValueError) as exc:
                raise MeterV5_2ScaleError(f"invalid bbox integers: {sample_id}") from exc
            v51._validate_bbox(x, y, w, h, sample.image_width, sample.image_height)
        elif any(row.get(k, "") for k in ("x", "y", "w", "h")):
            _fail(f"REVIEW row must not contain bbox: {sample_id}")

    def _load_or_seed_annotations(self) -> dict[str, dict[str, str]]:
        if not self.annotation_path.exists():
            seed_rows = []
            for sample_id in self.seed["seed_sample_ids"]:
                row = dict(self.seed_annotation_rows[sample_id])
                self._validate_annotation_row(row)
                seed_rows.append(row)
            _write_csv(self.annotation_path, v51.ANNOTATION_COLUMNS, seed_rows)

        rows = _read_csv(self.annotation_path)
        result = {}
        for row in rows:
            sample_id = row.get("sample_id", "")
            if sample_id in result:
                _fail(f"duplicate V5-2 annotation row: {sample_id}")
            self._validate_annotation_row(row)
            result[sample_id] = dict(row)

        if not self.seed_ids.issubset(result):
            _fail("V5-2 annotation file is missing one or more immutable seed rows")
        for sample_id in self.seed_ids:
            if result[sample_id] != self.seed_annotation_rows[sample_id]:
                _fail(f"immutable V5-1 seed annotation changed: {sample_id}")
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
        return max(0, TRAIN_TOTAL - 1)

    def _token(self, sample: TrainSample) -> str:
        payload = (
            f"{self.selection_sha256}\n{sample.index}\n{sample.sample_id}\n"
            f"{sample.image_sha256}\n{sample.image_width}x{sample.image_height}\n"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _resolve_token(self, token: object) -> TrainSample:
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
        _fail("sample binding token does not match V5-2 selection")

    def sample_payload(self, index: int) -> dict[str, object]:
        if type(index) is not int or not 0 <= index < TRAIN_TOTAL:
            _fail("V5-2 index outside 0..1199")
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
        }

    def _persist(self) -> None:
        ordered = [
            self.annotations[s.sample_id]
            for s in self.samples
            if s.sample_id in self.annotations
        ]
        _write_csv(self.annotation_path, v51.ANNOTATION_COLUMNS, ordered)
        handled = len(ordered)
        if handled == SEED_TOTAL or handled == TRAIN_TOTAL or (
            handled >= 100 and handled % 100 == 0
        ):
            backup = self.annotation_path.with_name(
                f"bbox_train_1200.backup_{handled:04d}.csv"
            )
            _write_csv(backup, v51.ANNOTATION_COLUMNS, ordered)

    def _require_mutable(self, sample: TrainSample) -> None:
        if sample.seed_annotation or sample.sample_id in self.seed_ids:
            _fail("accepted V5-1 seed BBox is immutable in V5-2")

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
            _fail("preview dimensions changed; reload sample before save")
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


def write_train_audit(
    data_root: str | Path,
    *,
    expected_dataset_fingerprint: str = EXPECTED_DATASET_FINGERPRINT,
    expected_seed_sha256: Mapping[str, str] = EXPECTED_SEED_SHA256,
) -> Path:
    """Write mechanical V5-2 QA without changing any human BBox."""
    session = ScaleAnnotationSession(
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
        session.annotations.get(sample_id) != session.seed_annotation_rows[sample_id]
        for sample_id in session.seed_ids
    )
    missing = TRAIN_TOTAL - len(rows)
    duplicate_samples = len(rows) - len({r["sample_id"] for r in rows})
    family_ids = [
        session.sample_by_id[r["sample_id"]].family_id
        for r in rows if r["sample_id"] in session.sample_by_id
    ]
    duplicate_families = len(family_ids) - len(set(family_ids))

    mechanical_pass = (
        len(rows) == TRAIN_TOTAL
        and len(pass_rows) == TRAIN_TOTAL
        and not review_rows
        and per_class == {
            meter: {"PASS": TRAIN_PER_CLASS, "REVIEW": 0}
            for meter in v51.CLASSES
        }
        and missing == 0
        and duplicate_samples == 0
        and duplicate_families == 0
        and seed_mutations == 0
    )

    payload = {
        "schema": "st-omr-meter-v5-2-train-bbox-audit-v1",
        "dataset": v51.DATASET_NAME,
        "dataset_fingerprint_sha256": EXPECTED_DATASET_FINGERPRINT,
        "train_total_expected": TRAIN_TOTAL,
        "annotation_count": len(rows),
        "pass_count": len(pass_rows),
        "review_count": len(review_rows),
        "per_class": per_class,
        "missing_annotation_count": missing,
        "duplicate_sample_count": duplicate_samples,
        "duplicate_family_count": duplicate_families,
        "seed_total": SEED_TOTAL,
        "seed_mutation_count": seed_mutations,
        "mechanical_gate": "PASS" if mechanical_pass else "HOLD",
        "human_visual_review_required": True,
        "training_authorized": False,
        "model_opened": False,
        "inference_count": 0,
        "validation_annotation_opened": False,
        "final_holdout_locked": True,
        "automatic_bbox_generation_count": 0,
        "digit_bbox_derivation_count": 0,
    }
    path = Path(data_root) / v51.ANNOTATIONS_DIR / AUDIT_NAME
    _write_json(path, payload)
    return path
