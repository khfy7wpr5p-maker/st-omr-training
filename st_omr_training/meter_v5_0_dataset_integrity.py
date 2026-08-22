"""Fail-closed integrity audit for METER_V2_1500 manifests.

This module does not train, infer, open checkpoints, or mutate datasets.
It only audits CSV manifests before bbox annotation/training is allowed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

CLASSES = ("2/4", "3/4", "4/4")
SPLITS = ("train", "val", "final_holdout")
EXPECTED_PER_CLASS = 500
EXPECTED_SPLIT_COUNTS = {"train": 400, "val": 50, "final_holdout": 50}
CANONICAL_COLUMNS = (
    "Split",
    "Meter",
    "FamilyId",
    "SampleId",
    "Folder",
    "SourceImage",
    "SourceSemantic",
    "SourceAgnostic",
    "SplitRank",
)
RANK_ALIASES = {"SplitRank", "SelectionRank"}
SOURCE_SHARE_GAP_MAX = 0.20
_PACKAGE_RE = re.compile(r"(?:^|[\\/])(package_[^\\/]+)(?:[\\/]|$)", re.IGNORECASE)


class DatasetIntegrityError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_domain(source_image: str) -> str:
    match = _PACKAGE_RE.search(source_image or "")
    return match.group(1).lower() if match else "unknown"


def _load_manifest(
    path: str | Path,
    expected_meter: str,
) -> tuple[list[dict[str, str]], str, list[str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise DatasetIntegrityError(f"{path}: missing header")
        fields = list(reader.fieldnames)
        rank_cols = [column for column in fields if column in RANK_ALIASES]
        if len(rank_cols) != 1:
            raise DatasetIntegrityError(f"{path}: expected exactly one rank column")
        rank_col = rank_cols[0]
        required_without_rank = set(CANONICAL_COLUMNS[:-1])
        missing = sorted(required_without_rank - set(fields))
        unexpected = sorted(set(fields) - required_without_rank - RANK_ALIASES)
        if missing or unexpected:
            raise DatasetIntegrityError(
                f"{path}: schema mismatch missing={missing} unexpected={unexpected}"
            )

        rows = []
        for line_number, row in enumerate(reader, start=2):
            normalized = {
                key: (row.get(key) or "").strip() for key in CANONICAL_COLUMNS[:-1]
            }
            normalized["SplitRank"] = (row.get(rank_col) or "").strip()
            if normalized["Meter"] != expected_meter:
                raise DatasetIntegrityError(
                    f"{path}:{line_number}: Meter={normalized['Meter']!r}, "
                    f"expected {expected_meter!r}"
                )
            if normalized["Split"] not in SPLITS:
                raise DatasetIntegrityError(f"{path}:{line_number}: invalid split")
            if not normalized["FamilyId"] or not normalized["SampleId"] or not normalized["Folder"]:
                raise DatasetIntegrityError(f"{path}:{line_number}: blank identity")
            if not normalized["SplitRank"]:
                raise DatasetIntegrityError(f"{path}:{line_number}: blank rank")
            normalized["_source_domain"] = _source_domain(normalized["SourceImage"])
            rows.append(normalized)
    return rows, rank_col, fields


def audit_manifests(
    manifests: Mapping[str, str | Path],
    *,
    consumed_family_ids: Iterable[str] = (),
    source_share_gap_max: float = SOURCE_SHARE_GAP_MAX,
) -> dict:
    if set(manifests) != set(CLASSES):
        raise DatasetIntegrityError(f"expected manifests for {CLASSES}")

    all_rows: list[dict[str, str]] = []
    manifest_meta = {}
    reasons: list[str] = []

    for meter in CLASSES:
        path = Path(manifests[meter])
        rows, rank_col, fields = _load_manifest(path, meter)
        manifest_meta[meter] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": len(rows),
            "rank_column": rank_col,
            "columns": fields,
        }
        if rank_col != "SplitRank":
            reasons.append(f"NON_CANONICAL_RANK_COLUMN:{meter}:{rank_col}")
        if len(rows) != EXPECTED_PER_CLASS:
            reasons.append(f"CLASS_COUNT:{meter}:{len(rows)}")

        split_counts = Counter(row["Split"] for row in rows)
        for split, expected in EXPECTED_SPLIT_COUNTS.items():
            if split_counts.get(split, 0) != expected:
                reasons.append(
                    f"SPLIT_COUNT:{meter}:{split}:{split_counts.get(split, 0)}"
                )

        for key in ("FamilyId", "SampleId", "Folder", "SplitRank"):
            if len({row[key] for row in rows}) != len(rows):
                reasons.append(f"DUPLICATE_{key.upper()}:{meter}")
        all_rows.extend(rows)

    if len(all_rows) != 1500:
        reasons.append(f"TOTAL_COUNT:{len(all_rows)}")
    if len({row["SampleId"] for row in all_rows}) != len(all_rows):
        reasons.append("GLOBAL_DUPLICATE_SAMPLE_ID")
    if len({row["Folder"] for row in all_rows}) != len(all_rows):
        reasons.append("GLOBAL_DUPLICATE_FOLDER")

    family_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        family_rows[row["FamilyId"]].append(row)

    split_leaks = []
    for family_id, rows in sorted(family_rows.items()):
        splits = sorted({row["Split"] for row in rows})
        if len(splits) > 1:
            split_leaks.append(
                {
                    "family_id": family_id,
                    "splits": splits,
                    "meters": sorted({row["Meter"] for row in rows}),
                    "folders": sorted(row["Folder"] for row in rows),
                }
            )
    if split_leaks:
        reasons.append(f"FAMILY_SPLIT_LEAKAGE:{len(split_leaks)}")

    consumed = set(consumed_family_ids)
    consumed_overlap = sorted(family for family in family_rows if family in consumed)
    if consumed_overlap:
        reasons.append(f"CONSUMED_HOLDOUT_OVERLAP:{len(consumed_overlap)}")

    source_counts: dict[str, Counter[str]] = {}
    source_domains = set()
    for meter in CLASSES:
        counts = Counter(
            row["_source_domain"] for row in all_rows if row["Meter"] == meter
        )
        source_counts[meter] = counts
        source_domains.update(counts)

    source_shares = {}
    source_gaps = {}
    for domain in sorted(source_domains):
        shares = {
            meter: source_counts[meter].get(domain, 0) / max(1, EXPECTED_PER_CLASS)
            for meter in CLASSES
        }
        gap = max(shares.values()) - min(shares.values())
        source_shares[domain] = shares
        source_gaps[domain] = gap
        if gap > source_share_gap_max:
            reasons.append(f"SOURCE_DOMAIN_SHARE_GAP:{domain}:{gap:.6f}")

    return {
        "schema": "st-omr-meter-v5-0-dataset-integrity-audit-v1",
        "status": "PASS" if not reasons else "HOLD",
        "training_authorized": not reasons,
        "bbox_annotation_authorized": not reasons,
        "expected": {
            "total": 1500,
            "per_class": EXPECTED_PER_CLASS,
            "split_per_class": EXPECTED_SPLIT_COUNTS,
            "source_share_gap_max": source_share_gap_max,
        },
        "manifest_meta": manifest_meta,
        "actual": {
            "total": len(all_rows),
            "unique_families": len(family_rows),
            "unique_samples": len({row["SampleId"] for row in all_rows}),
            "unique_folders": len({row["Folder"] for row in all_rows}),
            "class_split_counts": {
                meter: dict(
                    Counter(row["Split"] for row in all_rows if row["Meter"] == meter)
                )
                for meter in CLASSES
            },
            "source_domain_counts": {
                meter: dict(source_counts[meter]) for meter in CLASSES
            },
            "source_domain_shares": source_shares,
            "source_domain_share_gaps": source_gaps,
        },
        "split_leaks": split_leaks,
        "consumed_holdout_overlap": consumed_overlap,
        "reasons": reasons,
        "safety": {
            "training": False,
            "tuning": False,
            "model_evaluated": False,
            "checkpoint_opened": False,
            "inference_count": 0,
            "dataset_mutated": False,
        },
    }


def write_audit_receipt(result: Mapping, path: str | Path) -> str:
    payload = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    Path(path).write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
