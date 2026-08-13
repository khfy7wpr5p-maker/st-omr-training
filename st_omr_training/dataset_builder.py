"""Deterministic Stage 6 Synthetic Dataset v1 construction.

Stage 6 composes the already-validated Stage 1-5 pipeline. It chooses a bounded
synthetic family plan, assigns whole families to train/validation/test splits,
creates Stage 4 derivatives, submits every sample to the independent Stage 5
manifest validator, and only then permits hash-addressed local persistence.

The module never trains a model and never ingests real/user data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Callable, Final

from .dataset_manifest import (
    DatasetManifest,
    DatasetSample,
    DatasetSplit,
    DatasetValidationResult,
    canonical_manifest_bytes,
    dataset_manifest_sha256,
    sample_from_degraded_page,
    validate_dataset_manifest,
)
from .degradation import (
    MAX_RASTER_WIDTH,
    MIN_RASTER_WIDTH,
    degrade_render_result_page,
    sample_degradation_config,
)
from .generator import GeneratorConfig, generate_score
from .musicxml_validator import validate_musicxml
from .musicxml_writer import musicxml_sha256, write_musicxml
from .renderer import render_musicxml_svg


DATASET_BUILDER_VERSION: Final[str] = "st-synthetic-dataset-builder-v1"
DATASET_SPLIT_WEIGHTS: Final[tuple[int, int, int]] = (80, 10, 10)
DEFAULT_FAMILY_PROFILES: Final[tuple[str, ...]] = (
    "mixed",
    "note-only",
    "rest-only",
    "chord-only",
    "time-2-4",
    "time-3-4",
    "time-4-4",
    "no-accidentals",
)
DEFAULT_DEGRADATION_PROFILES: Final[tuple[str, ...]] = ("clean", "light", "medium")
MAX_BUILD_FAMILIES: Final[int] = 5_000
MAX_BUILD_SAMPLES: Final[int] = 50_000
MAX_BUILD_MEASURES: Final[int] = 64

_ALLOWED_FAMILY_PROFILES = frozenset(DEFAULT_FAMILY_PROFILES)
_ALLOWED_DEGRADATION_PROFILES = frozenset(DEFAULT_DEGRADATION_PROFILES)
_MAX_SEED = 2**63 - 1
_HEX = frozenset("0123456789abcdef")
DatasetProgressCallback = Callable[[dict[str, object]], None]


def _report_dataset_progress(
    progress: DatasetProgressCallback | None,
    event: str,
    **fields: object,
) -> None:
    if progress is not None:
        progress({"event": event, **fields})


class DatasetBuildInputError(ValueError):
    """Raised when a Stage 6 build request violates the bounded V1 contract."""


class DatasetBuildValidationError(RuntimeError):
    """Raised when the independent Stage 5 validator vetoes a Stage 6 build."""

    def __init__(self, result: DatasetValidationResult):
        self.result = result
        codes = ", ".join(issue.code for issue in result.issues)
        super().__init__(f"Stage 6 output failed independent Stage 5 validation: {codes}")


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_hex64(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _HEX for char in value)
    ):
        raise DatasetBuildInputError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class SyntheticDatasetConfig:
    """Bounded, deterministic Stage 6 V1 construction policy."""

    dataset_name: str = "st-omr-synthetic-v1"
    dataset_version: str = "v1"
    family_count: int = 24
    seed_start: int = 10_000
    split_seed: int = 1
    measure_count: int = 8
    raster_width: int = 1000
    family_profiles: tuple[str, ...] = DEFAULT_FAMILY_PROFILES
    degradation_profiles: tuple[str, ...] = DEFAULT_DEGRADATION_PROFILES

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_name, str) or not self.dataset_name:
            raise DatasetBuildInputError("dataset_name must be non-empty text")
        if not isinstance(self.dataset_version, str) or not self.dataset_version:
            raise DatasetBuildInputError("dataset_version must be non-empty text")

        if not _is_plain_int(self.family_count) or not 3 <= self.family_count <= MAX_BUILD_FAMILIES:
            raise DatasetBuildInputError(
                f"family_count must be an integer from 3 through {MAX_BUILD_FAMILIES}"
            )
        if not _is_plain_int(self.seed_start) or not 0 <= self.seed_start <= _MAX_SEED:
            raise DatasetBuildInputError("seed_start must be an integer in the Stage 6 seed range")
        if self.seed_start + self.family_count - 1 > _MAX_SEED:
            raise DatasetBuildInputError("family seed range exceeds the Stage 6 seed ceiling")
        if not _is_plain_int(self.split_seed) or not 0 <= self.split_seed <= _MAX_SEED:
            raise DatasetBuildInputError("split_seed must be an integer in the Stage 6 seed range")
        if not _is_plain_int(self.measure_count) or not 1 <= self.measure_count <= MAX_BUILD_MEASURES:
            raise DatasetBuildInputError(
                f"measure_count must be an integer from 1 through {MAX_BUILD_MEASURES}"
            )
        if (
            not _is_plain_int(self.raster_width)
            or not MIN_RASTER_WIDTH <= self.raster_width <= MAX_RASTER_WIDTH
        ):
            raise DatasetBuildInputError(
                f"raster_width must be an integer from {MIN_RASTER_WIDTH} through {MAX_RASTER_WIDTH}"
            )

        if not isinstance(self.family_profiles, tuple) or not self.family_profiles:
            raise DatasetBuildInputError("family_profiles must be a non-empty immutable tuple")
        if len(set(self.family_profiles)) != len(self.family_profiles):
            raise DatasetBuildInputError("family_profiles must not contain duplicates")
        if any(
            not isinstance(profile, str) or profile not in _ALLOWED_FAMILY_PROFILES
            for profile in self.family_profiles
        ):
            raise DatasetBuildInputError("family_profiles contains an unsupported Stage 6 profile")

        if not isinstance(self.degradation_profiles, tuple) or not self.degradation_profiles:
            raise DatasetBuildInputError("degradation_profiles must be a non-empty immutable tuple")
        if len(set(self.degradation_profiles)) != len(self.degradation_profiles):
            raise DatasetBuildInputError("degradation_profiles must not contain duplicates")
        if any(
            not isinstance(profile, str) or profile not in _ALLOWED_DEGRADATION_PROFILES
            for profile in self.degradation_profiles
        ):
            raise DatasetBuildInputError(
                "degradation_profiles contains an unsupported Stage 4 profile"
            )
        if self.family_count * len(self.degradation_profiles) > MAX_BUILD_SAMPLES:
            raise DatasetBuildInputError("minimum requested sample count exceeds the Stage 6 ceiling")


@dataclass(frozen=True, slots=True)
class SyntheticFamilyPlan:
    index: int
    seed: int
    profile: str
    split: DatasetSplit

    def __post_init__(self) -> None:
        if not _is_plain_int(self.index) or self.index < 0:
            raise DatasetBuildInputError("family plan index must be a non-negative integer")
        if not _is_plain_int(self.seed) or not 0 <= self.seed <= _MAX_SEED:
            raise DatasetBuildInputError("family plan seed is outside the Stage 6 range")
        if self.profile not in _ALLOWED_FAMILY_PROFILES:
            raise DatasetBuildInputError("family plan contains an unsupported profile")
        if not isinstance(self.split, DatasetSplit):
            raise DatasetBuildInputError("family plan split must be DatasetSplit")


@dataclass(frozen=True, slots=True)
class DatasetTargetArtifact:
    sha256: str
    musicxml: bytes

    def __post_init__(self) -> None:
        _require_hex64("target sha256", self.sha256)
        if not isinstance(self.musicxml, bytes) or not self.musicxml:
            raise DatasetBuildInputError("target MusicXML must be non-empty bytes")
        if sha256(self.musicxml).hexdigest() != self.sha256:
            raise DatasetBuildInputError("target MusicXML hash does not match its bytes")
        validation = validate_musicxml(self.musicxml)
        if not validation.is_valid:
            raise DatasetBuildInputError("target MusicXML failed the independent Stage 2-C gate")


@dataclass(frozen=True, slots=True)
class DatasetImageArtifact:
    sha256: str
    png: bytes

    def __post_init__(self) -> None:
        _require_hex64("image sha256", self.sha256)
        if not isinstance(self.png, bytes) or not self.png:
            raise DatasetBuildInputError("dataset image must be non-empty PNG bytes")
        if sha256(self.png).hexdigest() != self.sha256:
            raise DatasetBuildInputError("dataset image hash does not match its bytes")
        if not self.png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise DatasetBuildInputError("dataset image is not a PNG artifact")


@dataclass(frozen=True, slots=True)
class SyntheticDatasetBuild:
    config_fingerprint: str
    manifest: DatasetManifest
    manifest_sha256: str
    build_id: str
    targets: tuple[DatasetTargetArtifact, ...]
    images: tuple[DatasetImageArtifact, ...]
    builder_version: str = DATASET_BUILDER_VERSION

    def __post_init__(self) -> None:
        _require_hex64("config_fingerprint", self.config_fingerprint)
        _require_hex64("manifest_sha256", self.manifest_sha256)
        _require_hex64("build_id", self.build_id)
        if self.builder_version != DATASET_BUILDER_VERSION:
            raise DatasetBuildInputError("unsupported dataset builder version")
        if not isinstance(self.manifest, DatasetManifest):
            raise DatasetBuildInputError("manifest must be DatasetManifest")
        if not isinstance(self.targets, tuple) or any(
            not isinstance(item, DatasetTargetArtifact) for item in self.targets
        ):
            raise DatasetBuildInputError("targets must be an immutable target-artifact tuple")
        if not isinstance(self.images, tuple) or any(
            not isinstance(item, DatasetImageArtifact) for item in self.images
        ):
            raise DatasetBuildInputError("images must be an immutable image-artifact tuple")

        validation = validate_dataset_manifest(self.manifest)
        if not validation.is_valid:
            raise DatasetBuildValidationError(validation)
        if dataset_manifest_sha256(self.manifest) != self.manifest_sha256:
            raise DatasetBuildInputError("manifest_sha256 does not match canonical manifest bytes")

        target_hashes = tuple(item.sha256 for item in self.targets)
        image_hashes = tuple(item.sha256 for item in self.images)
        if len(set(target_hashes)) != len(target_hashes):
            raise DatasetBuildInputError("target artifact hashes must be unique")
        if len(set(image_hashes)) != len(image_hashes):
            raise DatasetBuildInputError("image artifact hashes must be unique")

        required_targets = {sample.source_musicxml_sha256 for sample in self.manifest.samples}
        required_images = {sample.png_sha256 for sample in self.manifest.samples}
        if set(target_hashes) != required_targets:
            raise DatasetBuildInputError("target artifact set does not exactly match the manifest")
        if set(image_hashes) != required_images:
            raise DatasetBuildInputError("image artifact set does not exactly match the manifest")

        expected_build_id = _build_id(
            config_fingerprint=self.config_fingerprint,
            manifest_sha256=self.manifest_sha256,
        )
        if self.build_id != expected_build_id:
            raise DatasetBuildInputError("build_id does not match config/manifest identity")


def synthetic_dataset_config_fingerprint(config: SyntheticDatasetConfig) -> str:
    if not isinstance(config, SyntheticDatasetConfig):
        raise TypeError("config must be SyntheticDatasetConfig")
    payload = {
        "builder_version": DATASET_BUILDER_VERSION,
        "split_weights": list(DATASET_SPLIT_WEIGHTS),
        "config": asdict(config),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def split_family_counts(family_count: int) -> tuple[int, int, int]:
    """Return deterministic 80/10/10 family counts with every split non-empty."""

    if not _is_plain_int(family_count) or family_count < 3:
        raise DatasetBuildInputError("family_count must be an integer of at least 3")

    total_weight = sum(DATASET_SPLIT_WEIGHTS)
    counts = [
        family_count * weight // total_weight
        for weight in DATASET_SPLIT_WEIGHTS
    ]
    counts = [max(1, count) for count in counts]

    while sum(counts) > family_count:
        if counts[0] > 1:
            counts[0] -= 1
            continue
        candidates = [index for index, count in enumerate(counts) if count > 1]
        if not candidates:
            raise DatasetBuildInputError("unable to form non-empty Stage 6 splits")
        index = max(candidates, key=lambda item: counts[item])
        counts[index] -= 1

    if sum(counts) < family_count:
        order = sorted(
            range(3),
            key=lambda item: (
                family_count * DATASET_SPLIT_WEIGHTS[item] % total_weight,
                DATASET_SPLIT_WEIGHTS[item],
                -item,
            ),
            reverse=True,
        )
        for index in order:
            if sum(counts) >= family_count:
                break
            counts[index] += 1

    if sum(counts) != family_count or any(count < 1 for count in counts):
        raise DatasetBuildInputError("internal Stage 6 split allocation failure")
    return counts[0], counts[1], counts[2]


def _split_rank(config: SyntheticDatasetConfig, *, index: int, seed: int, profile: str) -> bytes:
    payload = {
        "builder_version": DATASET_BUILDER_VERSION,
        "split_seed": config.split_seed,
        "family_index": index,
        "family_seed": seed,
        "family_profile": profile,
    }
    return sha256(_canonical_json_bytes(payload)).digest()


def plan_synthetic_families(config: SyntheticDatasetConfig) -> tuple[SyntheticFamilyPlan, ...]:
    """Create deterministic family plans before any rendering or degradation occurs."""

    if not isinstance(config, SyntheticDatasetConfig):
        raise TypeError("config must be SyntheticDatasetConfig")

    raw: list[tuple[int, int, str]] = []
    for index in range(config.family_count):
        raw.append(
            (
                index,
                config.seed_start + index,
                config.family_profiles[index % len(config.family_profiles)],
            )
        )

    ranked = sorted(
        raw,
        key=lambda item: (
            _split_rank(config, index=item[0], seed=item[1], profile=item[2]),
            item[0],
        ),
    )
    train_count, validation_count, test_count = split_family_counts(config.family_count)
    split_by_index: dict[int, DatasetSplit] = {}
    for position, (index, _seed, _profile) in enumerate(ranked):
        if position < train_count:
            split = DatasetSplit.TRAIN
        elif position < train_count + validation_count:
            split = DatasetSplit.VALIDATION
        else:
            split = DatasetSplit.TEST
        split_by_index[index] = split

    plans = tuple(
        SyntheticFamilyPlan(
            index=index,
            seed=seed,
            profile=profile,
            split=split_by_index[index],
        )
        for index, seed, profile in raw
    )
    observed = (
        sum(plan.split is DatasetSplit.TRAIN for plan in plans),
        sum(plan.split is DatasetSplit.VALIDATION for plan in plans),
        sum(plan.split is DatasetSplit.TEST for plan in plans),
    )
    if observed != (train_count, validation_count, test_count):
        raise DatasetBuildInputError("internal Stage 6 split planning failure")
    return plans


def _generator_config_for_profile(profile: str, measure_count: int) -> GeneratorConfig:
    if profile == "mixed":
        return GeneratorConfig(measure_count=measure_count)
    if profile == "note-only":
        return GeneratorConfig(measure_count=measure_count, event_kinds=("note",))
    if profile == "rest-only":
        return GeneratorConfig(measure_count=measure_count, event_kinds=("rest",))
    if profile == "chord-only":
        return GeneratorConfig(measure_count=measure_count, event_kinds=("chord",))
    if profile == "time-2-4":
        return GeneratorConfig(measure_count=measure_count, time_signatures=((2, 4),))
    if profile == "time-3-4":
        return GeneratorConfig(measure_count=measure_count, time_signatures=((3, 4),))
    if profile == "time-4-4":
        return GeneratorConfig(measure_count=measure_count, time_signatures=((4, 4),))
    if profile == "no-accidentals":
        return GeneratorConfig(measure_count=measure_count, allow_accidentals=False)
    raise DatasetBuildInputError("unsupported Stage 6 family profile")


def _degradation_seed(family_id: str, page_number: int, profile: str) -> int:
    payload = {
        "builder_version": DATASET_BUILDER_VERSION,
        "family_id": family_id,
        "page_number": page_number,
        "degradation_profile": profile,
    }
    raw = sha256(_canonical_json_bytes(payload)).digest()
    return int.from_bytes(raw[:8], "big") & _MAX_SEED


def _build_id(*, config_fingerprint: str, manifest_sha256: str) -> str:
    return sha256(
        _canonical_json_bytes(
            {
                "builder_version": DATASET_BUILDER_VERSION,
                "config_fingerprint": config_fingerprint,
                "manifest_sha256": manifest_sha256,
            }
        )
    ).hexdigest()


def build_synthetic_dataset(
    config: SyntheticDatasetConfig,
    *,
    progress: DatasetProgressCallback | None = None,
) -> SyntheticDatasetBuild:
    """Construct an in-memory Synthetic Dataset v1 and require Stage 5 acceptance."""

    if not isinstance(config, SyntheticDatasetConfig):
        raise TypeError("config must be SyntheticDatasetConfig")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")

    plans = plan_synthetic_families(config)
    samples: list[DatasetSample] = []
    targets: dict[str, DatasetTargetArtifact] = {}
    images: dict[str, DatasetImageArtifact] = {}

    for family_index, plan in enumerate(plans, start=1):
        generator_config = _generator_config_for_profile(plan.profile, config.measure_count)
        score = generate_score(generator_config, plan.seed)
        family_id = score.score_id
        musicxml = write_musicxml(score)
        target_hash = musicxml_sha256(musicxml)

        existing_target = targets.get(target_hash)
        target_artifact = DatasetTargetArtifact(target_hash, musicxml)
        if existing_target is not None:
            if existing_target.musicxml != musicxml:
                raise DatasetBuildInputError("MusicXML hash collision detected")
            raise DatasetBuildInputError(
                "two Stage 6 families produced an identical MusicXML target"
            )
        targets[target_hash] = target_artifact

        rendered = render_musicxml_svg(musicxml)
        for page in rendered.pages:
            page_number = page.page_number
            for degradation_profile in config.degradation_profiles:
                if len(samples) >= MAX_BUILD_SAMPLES:
                    raise DatasetBuildInputError(
                        f"Stage 6 output exceeds the {MAX_BUILD_SAMPLES} sample ceiling"
                    )
                degradation_seed = _degradation_seed(
                    family_id,
                    page_number,
                    degradation_profile,
                )
                degradation_config = sample_degradation_config(
                    degradation_seed,
                    degradation_profile,
                    raster_width=config.raster_width,
                )
                degraded = degrade_render_result_page(
                    rendered,
                    family_id=family_id,
                    page_number=page_number,
                    config=degradation_config,
                )
                sample = sample_from_degraded_page(degraded, split=plan.split)
                if sample.source_musicxml_sha256 != target_hash:
                    raise DatasetBuildInputError(
                        "Stage 6 symbolic target hash diverged across the pipeline"
                    )

                existing_image = images.get(degraded.png_sha256)
                image_artifact = DatasetImageArtifact(degraded.png_sha256, degraded.png)
                if existing_image is not None:
                    if existing_image.png != degraded.png:
                        raise DatasetBuildInputError("PNG hash collision detected")
                    raise DatasetBuildInputError(
                        "Stage 6 produced a duplicate final PNG artifact"
                    )
                images[degraded.png_sha256] = image_artifact
                samples.append(sample)

        _report_dataset_progress(
            progress,
            "dataset_family_completed",
            families_completed=family_index,
            families_total=len(plans),
            samples_built=len(samples),
        )

    manifest = DatasetManifest(
        dataset_name=config.dataset_name,
        dataset_version=config.dataset_version,
        samples=tuple(samples),
    )
    validation = validate_dataset_manifest(manifest)
    if not validation.is_valid:
        raise DatasetBuildValidationError(validation)
    _report_dataset_progress(
        progress,
        "dataset_validation_completed",
        families_total=len(plans),
        samples_total=len(samples),
    )

    manifest_hash = dataset_manifest_sha256(manifest)
    config_hash = synthetic_dataset_config_fingerprint(config)
    return SyntheticDatasetBuild(
        config_fingerprint=config_hash,
        manifest=manifest,
        manifest_sha256=manifest_hash,
        build_id=_build_id(
            config_fingerprint=config_hash,
            manifest_sha256=manifest_hash,
        ),
        targets=tuple(targets[key] for key in sorted(targets)),
        images=tuple(images[key] for key in sorted(images)),
    )


def build_metadata_bytes(build: SyntheticDatasetBuild) -> bytes:
    if not isinstance(build, SyntheticDatasetBuild):
        raise TypeError("build must be SyntheticDatasetBuild")
    split_counts = {
        split.value: sum(sample.split is split for sample in build.manifest.samples)
        for split in DatasetSplit
    }
    family_counts = {
        split.value: len(
            {
                sample.family_id
                for sample in build.manifest.samples
                if sample.split is split
            }
        )
        for split in DatasetSplit
    }
    payload = {
        "builder_version": build.builder_version,
        "build_id": build.build_id,
        "config_fingerprint": build.config_fingerprint,
        "manifest_sha256": build.manifest_sha256,
        "sample_count": len(build.manifest.samples),
        "target_count": len(build.targets),
        "image_count": len(build.images),
        "sample_split_counts": split_counts,
        "family_split_counts": family_counts,
        "layout": {
            "manifest": "manifest.json",
            "metadata": "build.json",
            "images": "images/<png_sha256>.png",
            "targets": "targets/<source_musicxml_sha256>.musicxml",
        },
    }
    return _canonical_json_bytes(payload)


def _verify_written(path: Path, expected_sha256: str) -> None:
    if sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise DatasetBuildInputError(f"persisted artifact hash mismatch: {path.name}")


def write_synthetic_dataset(
    build: SyntheticDatasetBuild,
    output_dir: str | Path,
    *,
    progress: DatasetProgressCallback | None = None,
) -> Path:
    """Persist one already-validated build using a hash-addressed, no-overwrite layout."""

    if not isinstance(build, SyntheticDatasetBuild):
        raise TypeError("build must be SyntheticDatasetBuild")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    validation = validate_dataset_manifest(build.manifest)
    if not validation.is_valid:
        raise DatasetBuildValidationError(validation)

    if not isinstance(output_dir, (str, Path)):
        raise TypeError("output_dir must be str or pathlib.Path")
    root = Path(output_dir)
    if not root.name or root.name in {".", ".."}:
        raise DatasetBuildInputError("output_dir must name a new dataset directory")
    parent = root.parent
    if not parent.exists() or not parent.is_dir():
        raise DatasetBuildInputError("output_dir parent must already exist and be a directory")
    if root.exists():
        raise DatasetBuildInputError("output_dir already exists; Stage 6 never overwrites datasets")

    temp = parent / f".{root.name}.tmp-{build.build_id[:16]}"
    if temp.exists():
        raise DatasetBuildInputError("deterministic Stage 6 staging directory already exists")

    try:
        (temp / "images").mkdir(parents=True, exist_ok=False)
        (temp / "targets").mkdir(parents=False, exist_ok=False)

        manifest_bytes = canonical_manifest_bytes(build.manifest)
        manifest_path = temp / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        _verify_written(manifest_path, build.manifest_sha256)

        metadata = build_metadata_bytes(build)
        metadata_path = temp / "build.json"
        metadata_path.write_bytes(metadata)
        _verify_written(metadata_path, sha256(metadata).hexdigest())

        checksum_path = temp / "manifest.sha256"
        checksum_path.write_text(
            f"{build.manifest_sha256}  manifest.json\n",
            encoding="ascii",
            newline="\n",
        )

        for target_index, target in enumerate(build.targets, start=1):
            path = temp / "targets" / f"{target.sha256}.musicxml"
            path.write_bytes(target.musicxml)
            _verify_written(path, target.sha256)
            if target_index % 8 == 0 or target_index == len(build.targets):
                _report_dataset_progress(
                    progress,
                    "dataset_target_written",
                    targets_written=target_index,
                    targets_total=len(build.targets),
                )

        for image_index, image in enumerate(build.images, start=1):
            path = temp / "images" / f"{image.sha256}.png"
            path.write_bytes(image.png)
            _verify_written(path, image.sha256)
            if image_index % 8 == 0 or image_index == len(build.images):
                _report_dataset_progress(
                    progress,
                    "dataset_image_written",
                    images_written=image_index,
                    images_total=len(build.images),
                )

        temp.rename(root)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise

    _report_dataset_progress(progress, "dataset_persisted")
    return root
