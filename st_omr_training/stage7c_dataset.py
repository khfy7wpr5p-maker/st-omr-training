"""Frozen Stage 7-C synthetic baseline dataset profile."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from .dataset_builder import (
    DEFAULT_DEGRADATION_PROFILES,
    DEFAULT_FAMILY_PROFILES,
    SyntheticDatasetBuild,
    SyntheticDatasetConfig,
    build_synthetic_dataset,
    synthetic_dataset_config_fingerprint,
    write_synthetic_dataset,
)


STAGE7C_BASELINE_DATASET_PROFILE_VERSION: Final[str] = "stage7c-baseline-dataset-v1"
STAGE7C_BASELINE_DATASET_CONFIG: Final[SyntheticDatasetConfig] = SyntheticDatasetConfig(
    dataset_name="st-omr-stage7c-baseline-v1",
    dataset_version="v1",
    family_count=64,
    seed_start=70_000,
    split_seed=7_001,
    measure_count=8,
    raster_width=1000,
    family_profiles=DEFAULT_FAMILY_PROFILES,
    degradation_profiles=DEFAULT_DEGRADATION_PROFILES,
)
STAGE7C_BASELINE_DATASET_CONFIG_FINGERPRINT: Final[str] = synthetic_dataset_config_fingerprint(
    STAGE7C_BASELINE_DATASET_CONFIG
)


def build_and_persist_stage7c_baseline_dataset(
    dataset_root: str | Path,
) -> SyntheticDatasetBuild:
    """Build the one frozen Stage 7-C baseline dataset and persist it without overwrite."""

    if not isinstance(dataset_root, (str, Path)):
        raise TypeError("dataset_root must be str or pathlib.Path")
    root = Path(dataset_root)
    if root.exists():
        raise FileExistsError("Stage 7-C baseline dataset path must be fresh")
    build = build_synthetic_dataset(STAGE7C_BASELINE_DATASET_CONFIG)
    if build.config_fingerprint != STAGE7C_BASELINE_DATASET_CONFIG_FINGERPRINT:
        raise RuntimeError("Stage 7-C baseline dataset config fingerprint drifted")
    write_synthetic_dataset(build, root)
    return build
