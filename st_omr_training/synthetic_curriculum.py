"""Frozen reproducible synthetic curriculum corpus for ST-OMR V1.

This package reuses the already-closed Stage 1-6 synthetic pipeline. It creates
only synthetic artifacts, persists them outside the Git repository, and never
loads a model, trains, touches real/user data, or opens the Stage 9 test set.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Final

from .dataset_builder import (
    DEFAULT_DEGRADATION_PROFILES,
    DEFAULT_FAMILY_PROFILES,
    SyntheticDatasetBuild,
    SyntheticDatasetConfig,
    build_synthetic_dataset,
    plan_synthetic_families,
    synthetic_dataset_config_fingerprint,
    write_synthetic_dataset,
)


SYNTHETIC_CURRICULUM_PROFILE_VERSION: Final[str] = "st-synthetic-curriculum-v1"
SYNTHETIC_CURRICULUM_CONFIG: Final[SyntheticDatasetConfig] = SyntheticDatasetConfig(
    dataset_name="st-omr-synthetic-curriculum-v1",
    dataset_version="v1",
    family_count=512,
    seed_start=100_000,
    split_seed=8_001,
    measure_count=8,
    raster_width=1000,
    family_profiles=DEFAULT_FAMILY_PROFILES,
    degradation_profiles=DEFAULT_DEGRADATION_PROFILES,
)
SYNTHETIC_CURRICULUM_CONFIG_FINGERPRINT: Final[str] = synthetic_dataset_config_fingerprint(
    SYNTHETIC_CURRICULUM_CONFIG
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_synthetic_output_path(
    output_dir: str | Path,
    repository_root: str | Path | None = None,
) -> Path:
    if not isinstance(output_dir, (str, Path)):
        raise TypeError("output_dir must be str or pathlib.Path")
    root = _repository_root() if repository_root is None else Path(repository_root).resolve()
    target = Path(output_dir).expanduser().resolve()
    if target == root or root in target.parents:
        raise ValueError("synthetic curriculum artifacts must stay outside the Git repository")
    if target.exists():
        raise FileExistsError("synthetic curriculum output path must be fresh")
    return target


def curriculum_plan_summary() -> dict[str, object]:
    plans = plan_synthetic_families(SYNTHETIC_CURRICULUM_CONFIG)
    split_counts = Counter(plan.split.value for plan in plans)
    profile_counts = Counter(plan.profile for plan in plans)
    return {
        "schema_version": "st-synthetic-curriculum-plan-v1",
        "profile_version": SYNTHETIC_CURRICULUM_PROFILE_VERSION,
        "config_fingerprint": SYNTHETIC_CURRICULUM_CONFIG_FINGERPRINT,
        "family_count": len(plans),
        "measure_count": SYNTHETIC_CURRICULUM_CONFIG.measure_count,
        "raster_width": SYNTHETIC_CURRICULUM_CONFIG.raster_width,
        "degradation_profiles": list(SYNTHETIC_CURRICULUM_CONFIG.degradation_profiles),
        "family_split_counts": dict(sorted(split_counts.items())),
        "family_profile_counts": dict(sorted(profile_counts.items())),
    }


def build_and_persist_synthetic_curriculum(
    output_dir: str | Path,
    *,
    progress=None,
) -> SyntheticDatasetBuild:
    target = validate_synthetic_output_path(output_dir)
    build = build_synthetic_dataset(SYNTHETIC_CURRICULUM_CONFIG, progress=progress)
    if build.config_fingerprint != SYNTHETIC_CURRICULUM_CONFIG_FINGERPRINT:
        raise RuntimeError("synthetic curriculum config fingerprint drifted")
    write_synthetic_dataset(build, target, progress=progress)
    return build


def _progress(payload: dict[str, object]) -> None:
    print(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        file=sys.stderr,
        flush=True,
    )


def _build_summary(build: SyntheticDatasetBuild) -> dict[str, object]:
    plan = curriculum_plan_summary()
    return {
        "schema_version": "st-synthetic-curriculum-build-v1",
        "profile_version": SYNTHETIC_CURRICULUM_PROFILE_VERSION,
        "config_fingerprint": build.config_fingerprint,
        "build_id": build.build_id,
        "manifest_sha256": build.manifest_sha256,
        "sample_count": len(build.manifest.samples),
        "target_count": len(build.targets),
        "image_count": len(build.images),
        "family_split_counts": plan["family_split_counts"],
        "family_profile_counts": plan["family_profile_counts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the frozen ST-OMR V1 synthetic curriculum corpus.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="fresh output directory outside the Git repository",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the deterministic family/split plan without rendering artifacts",
    )
    arguments = parser.parse_args(argv)
    if arguments.plan_only:
        print(json.dumps(curriculum_plan_summary(), sort_keys=True, separators=(",", ":")))
        return 0
    build = build_and_persist_synthetic_curriculum(arguments.output, progress=_progress)
    print(json.dumps(_build_summary(build), sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
