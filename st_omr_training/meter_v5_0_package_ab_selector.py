"""Deterministic package_ab-only selector for METER V5-0.

This module is data-selection only. It does not copy datasets, annotate bbox,
train, load checkpoints, or run inference.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

CLASSES = ("2/4", "3/4", "4/4")
METER_FLAGS = {"2/4": "Meter2_4", "3/4": "Meter3_4", "4/4": "Meter4_4"}
EXPECTED_SPLIT_COUNTS = {"train": 400, "val": 50, "final_holdout": 50}
EXPECTED_PER_CLASS = 500
DEFAULT_SEED = "st-omr-meter-v5-0-package-ab-selector-v1"
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
REQUIRED_INDEX_COLUMNS = {
    "Package",
    "Sample",
    "Family",
    "PNG",
    "Semantic",
    "Agnostic",
    "Complete",
    "Meter2_4",
    "Meter3_4",
    "Meter4_4",
}


class PackageAbSelectionError(ValueError):
    pass


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_family_blacklist(paths: Iterable[str | Path]) -> set[str]:
    families: set[str] = set()
    for path in paths:
        p = Path(path)
        with p.open("r", encoding="utf-8-sig") as fh:
            for raw in fh:
                family = raw.strip()
                if family:
                    families.add(family)
    return families


def load_master_index(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise PackageAbSelectionError("MASTER_INDEX.tsv: missing header")
        missing = sorted(REQUIRED_INDEX_COLUMNS - set(reader.fieldnames))
        if missing:
            raise PackageAbSelectionError(
                f"MASTER_INDEX.tsv: missing required columns {missing}"
            )
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def _is_package_ab_path(path: str) -> bool:
    normalized = (path or "").replace("/", "\\").lower()
    return "\\package_ab\\" in normalized


def _eligible_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    blacklist: set[str],
) -> dict[str, list[dict[str, str]]]:
    by_meter = {meter: [] for meter in CLASSES}

    for row in rows:
        if row.get("Package") != "ab" or row.get("Complete") != "1":
            continue

        family_id = f"ab_{row.get('Family', '')}"
        if family_id in blacklist:
            continue

        source_paths = (
            row.get("PNG", ""),
            row.get("Semantic", ""),
            row.get("Agnostic", ""),
        )
        if not all(_is_package_ab_path(path) for path in source_paths):
            raise PackageAbSelectionError(
                f"package_ab provenance mismatch for {family_id}: {source_paths!r}"
            )

        active = [
            meter
            for meter, flag in METER_FLAGS.items()
            if row.get(flag) == "1"
        ]
        if len(active) != 1:
            continue

        meter = active[0]
        normalized = dict(row)
        normalized["FamilyId"] = family_id
        by_meter[meter].append(normalized)

    return by_meter


def build_package_ab_selection(
    rows: Iterable[Mapping[str, str]],
    *,
    blacklist: Iterable[str] = (),
    seed: str = DEFAULT_SEED,
) -> tuple[dict[str, list[dict[str, str]]], dict]:
    blacklist_set = set(blacklist)
    raw = _eligible_rows(rows, blacklist=blacklist_set)

    family_meters: dict[str, set[str]] = defaultdict(set)
    for meter, meter_rows in raw.items():
        for row in meter_rows:
            family_meters[row["FamilyId"]].add(meter)

    ambiguous_families = {
        family_id for family_id, meters in family_meters.items() if len(meters) > 1
    }

    clean_capacity = {}
    selected: dict[str, list[dict[str, str]]] = {}
    raw_capacity = {}

    for meter in CLASSES:
        meter_rows = [
            row for row in raw[meter] if row["FamilyId"] not in ambiguous_families
        ]
        raw_capacity[meter] = {
            "rows": len(raw[meter]),
            "families": len({row["FamilyId"] for row in raw[meter]}),
        }

        by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in meter_rows:
            by_family[row["FamilyId"]].append(row)

        chosen_per_family: list[dict[str, str]] = []
        for family_id, family_rows in by_family.items():
            family_rows = sorted(
                family_rows,
                key=lambda row: _sha256_text(
                    f"{seed}|sample|{meter}|{family_id}|{row['Sample']}"
                ),
            )
            chosen_per_family.append(family_rows[0])

        clean_capacity[meter] = len(chosen_per_family)
        if clean_capacity[meter] < EXPECTED_PER_CLASS:
            raise PackageAbSelectionError(
                f"{meter}: insufficient clean package_ab families "
                f"{clean_capacity[meter]} < {EXPECTED_PER_CLASS}"
            )

        ranked = sorted(
            chosen_per_family,
            key=lambda row: _sha256_text(
                f"{seed}|family|{meter}|{row['FamilyId']}"
            ),
        )[:EXPECTED_PER_CLASS]

        output_rows = []
        for index, row in enumerate(ranked):
            if index < 400:
                split = "train"
            elif index < 450:
                split = "val"
            else:
                split = "final_holdout"

            family_id = row["FamilyId"]
            sample_id = row["Sample"]
            output_rows.append(
                {
                    "Split": split,
                    "Meter": meter,
                    "FamilyId": family_id,
                    "SampleId": sample_id,
                    "Folder": (
                        f"{meter.replace('/', '_')}_{family_id}_{sample_id}"
                    ),
                    "SourceImage": row["PNG"],
                    "SourceSemantic": row["Semantic"],
                    "SourceAgnostic": row["Agnostic"],
                    "SplitRank": _sha256_text(
                        f"{seed}|family|{meter}|{family_id}"
                    ),
                }
            )
        selected[meter] = output_rows

    all_rows = [row for meter in CLASSES for row in selected[meter]]
    all_families = [row["FamilyId"] for row in all_rows]
    if len(all_rows) != 1500 or len(set(all_families)) != 1500:
        raise PackageAbSelectionError("global family-disjointness invariant failed")

    split_counts = {
        meter: {
            split: sum(row["Split"] == split for row in selected[meter])
            for split in EXPECTED_SPLIT_COUNTS
        }
        for meter in CLASSES
    }
    if any(
        split_counts[meter] != EXPECTED_SPLIT_COUNTS
        for meter in CLASSES
    ):
        raise PackageAbSelectionError("split cardinality invariant failed")

    receipt = {
        "schema": "st-omr-meter-v5-0-package-ab-selection-v1",
        "seed": seed,
        "source_domain": "package_ab",
        "blacklist_count": len(blacklist_set),
        "ambiguous_family_exclusion_count": len(ambiguous_families),
        "raw_capacity": raw_capacity,
        "clean_family_capacity": clean_capacity,
        "selected": {
            meter: {
                "total": len(selected[meter]),
                "split_counts": split_counts[meter],
                "unique_families": len(
                    {row["FamilyId"] for row in selected[meter]}
                ),
            }
            for meter in CLASSES
        },
        "global_unique_families": len(set(all_families)),
        "blacklist_overlap": len(set(all_families) & blacklist_set),
        "source_domain_share_gap": 0.0,
        "selection_ready_for_materialization_review": True,
        "bbox_annotation_authorized": False,
        "training_authorized": False,
        "safety": {
            "dataset_copied": False,
            "dataset_mutated": False,
            "bbox_annotation": False,
            "training": False,
            "tuning": False,
            "checkpoint_opened": False,
            "model_evaluated": False,
            "inference_count": 0,
        },
    }
    return selected, receipt


def write_selection_manifests(
    selected: Mapping[str, list[Mapping[str, str]]],
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    if out.exists():
        raise PackageAbSelectionError(
            f"output directory already exists: {out}"
        )
    out.mkdir(parents=True)

    paths = {}
    for meter in CLASSES:
        path = out / f"{meter.replace('/', '_')}_SELECTION_MANIFEST.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CANONICAL_COLUMNS)
            writer.writeheader()
            writer.writerows(selected[meter])
        paths[meter] = str(path)
    return paths
