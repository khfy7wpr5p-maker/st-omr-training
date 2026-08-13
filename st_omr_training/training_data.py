"""Trusted Stage 7-B data adapter and deterministic grayscale preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Final

from PIL import Image
import torch

from .dataset_builder import SyntheticDatasetBuild, build_metadata_bytes
from .dataset_manifest import (
    DatasetSplit,
    MAX_IMAGE_PIXELS,
    canonical_manifest_bytes,
    validate_dataset_manifest,
)
from .training_tokens import (
    PAD_TOKEN_ID,
    VOCABULARY_SIZE,
    TokenizationError,
    tokenize_musicxml,
)

DATA_ADAPTER_VERSION: Final[str] = "st-omr-training-data-adapter-v1"
PREPROCESS_VERSION: Final[str] = "st-omr-fit-pad-grayscale-v1"
MAX_STAGE7_SEQUENCE_TOKENS: Final[int] = 4096


class TrainingDataError(ValueError):
    """Raised when data cannot cross the Stage 7-B trusted-input boundary."""


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class InputPreprocessConfig:
    """Frozen Stage 7-B fit-and-pad policy; it never crops or augments content."""

    target_height: int = 64
    target_width: int = 512

    def __post_init__(self) -> None:
        if not _is_plain_int(self.target_height) or not 32 <= self.target_height <= 512:
            raise TrainingDataError("target_height must be an integer from 32 through 512")
        if not _is_plain_int(self.target_width) or not 128 <= self.target_width <= 2048:
            raise TrainingDataError("target_width must be an integer from 128 through 2048")
        if self.target_width * self.target_height > 1_048_576:
            raise TrainingDataError("preprocessed tensor exceeds the Stage 7-B pixel ceiling")


@dataclass(frozen=True, slots=True)
class TrainingSampleRef:
    sample_id: str
    family_id: str
    split: DatasetSplit
    image_path: Path
    image_sha256: str
    target_path: Path
    target_sha256: str
    target_token_ids: tuple[int, ...]
    source_width: int
    source_height: int

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise TrainingDataError("sample_id must be non-empty text")
        if not isinstance(self.family_id, str) or not self.family_id:
            raise TrainingDataError("family_id must be non-empty text")
        if self.split not in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}:
            raise TrainingDataError("Stage 7-B sample references may not expose the sealed test split")
        if not isinstance(self.image_path, Path) or not isinstance(self.target_path, Path):
            raise TrainingDataError("artifact paths must be pathlib.Path")
        if (
            not isinstance(self.target_token_ids, tuple)
            or len(self.target_token_ids) < 2
            or len(self.target_token_ids) > MAX_STAGE7_SEQUENCE_TOKENS
        ):
            raise TrainingDataError("target token sequence length is outside the Stage 7-B boundary")
        if any(
            not _is_plain_int(item) or not 0 <= item < VOCABULARY_SIZE
            for item in self.target_token_ids
        ):
            raise TrainingDataError("target token id is outside the frozen vocabulary")
        if not _is_plain_int(self.source_width) or self.source_width <= 0:
            raise TrainingDataError("source_width must be a positive integer")
        if not _is_plain_int(self.source_height) or self.source_height <= 0:
            raise TrainingDataError("source_height must be a positive integer")


@dataclass(slots=True)
class TrainingBatch:
    images: torch.Tensor
    decoder_input_ids: torch.Tensor
    labels: torch.Tensor
    split: DatasetSplit

    def __post_init__(self) -> None:
        if self.split not in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}:
            raise TrainingDataError("sealed test data cannot enter a Stage 7-B batch")
        if not isinstance(self.images, torch.Tensor) or self.images.dtype != torch.float32:
            raise TrainingDataError("images must be a float32 torch.Tensor")
        if self.images.ndim != 4 or self.images.shape[1] != 1 or self.images.shape[0] < 1:
            raise TrainingDataError("images must have shape [batch, 1, height, width]")
        if not bool(torch.isfinite(self.images).all()):
            raise TrainingDataError("image batch contains NaN or Infinity")
        if bool((self.images < 0).any()) or bool((self.images > 1).any()):
            raise TrainingDataError("image batch must stay in normalized [0, 1] range")
        if (
            not isinstance(self.decoder_input_ids, torch.Tensor)
            or self.decoder_input_ids.dtype != torch.long
            or not isinstance(self.labels, torch.Tensor)
            or self.labels.dtype != torch.long
        ):
            raise TrainingDataError("decoder ids and labels must be torch.long tensors")
        if (
            self.decoder_input_ids.ndim != 2
            or self.labels.ndim != 2
            or self.decoder_input_ids.shape != self.labels.shape
            or self.decoder_input_ids.shape[0] != self.images.shape[0]
            or self.decoder_input_ids.shape[1] < 1
        ):
            raise TrainingDataError("decoder ids/labels have an invalid batch shape")
        for tensor, label in (
            (self.decoder_input_ids, "decoder input"),
            (self.labels, "label"),
        ):
            if bool((tensor < 0).any()) or bool((tensor >= VOCABULARY_SIZE).any()):
                raise TrainingDataError(f"{label} token id is outside the frozen vocabulary")
        if not bool((self.labels != PAD_TOKEN_ID).any()):
            raise TrainingDataError("batch contains no unmasked target labels")


def preprocess_config_fingerprint(config: InputPreprocessConfig) -> str:
    if not isinstance(config, InputPreprocessConfig):
        raise TypeError("config must be InputPreprocessConfig")
    payload = {
        "preprocess_version": PREPROCESS_VERSION,
        "policy": "fit-inside-fixed-canvas-no-upscale-center-white-pad",
        "resampling": "Pillow-BILINEAR",
        "config": asdict(config),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _fit_dimensions(
    width: int,
    height: int,
    config: InputPreprocessConfig,
) -> tuple[int, int]:
    if width <= config.target_width and height <= config.target_height:
        return width, height
    # Compare exact integer products rather than floating-point aspect ratios.
    if config.target_width * height <= config.target_height * width:
        new_width = config.target_width
        new_height = max(1, (height * config.target_width + width // 2) // width)
    else:
        new_height = config.target_height
        new_width = max(1, (width * config.target_height + height // 2) // height)
    if new_width > config.target_width or new_height > config.target_height:
        raise TrainingDataError("internal fit policy exceeded the fixed canvas")
    return new_width, new_height


def preprocess_grayscale_png(
    data: object,
    config: InputPreprocessConfig = InputPreprocessConfig(),
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> torch.Tensor:
    """Decode exact PNG/L input, fit without cropping, then center-pad with white."""

    if not isinstance(data, bytes) or not data:
        raise TrainingDataError("image artifact must be non-empty bytes")
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise TrainingDataError("image artifact is not PNG")
    if not isinstance(config, InputPreprocessConfig):
        raise TypeError("config must be InputPreprocessConfig")

    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG":
                raise TrainingDataError("decoded artifact format is not PNG")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise TrainingDataError("source image dimensions exceed Stage 5 limits")
            if expected_width is not None and width != expected_width:
                raise TrainingDataError("source image width differs from manifest metadata")
            if expected_height is not None and height != expected_height:
                raise TrainingDataError("source image height differs from manifest metadata")
            if image.mode != "L":
                raise TrainingDataError("Stage 7-B requires exact grayscale mode L")
            image.load()
            new_width, new_height = _fit_dimensions(width, height, config)
            if (new_width, new_height) != (width, height):
                fitted = image.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
            else:
                fitted = image.copy()
    except TrainingDataError:
        raise
    except Exception as exc:
        raise TrainingDataError("PNG decode/preprocess failed") from exc

    canvas = Image.new("L", (config.target_width, config.target_height), color=255)
    offset_x = (config.target_width - new_width) // 2
    offset_y = (config.target_height - new_height) // 2
    canvas.paste(fitted, (offset_x, offset_y))
    pixels = torch.tensor(list(canvas.tobytes()), dtype=torch.float32)
    tensor = pixels.reshape(1, config.target_height, config.target_width) / 255.0
    if not bool(torch.isfinite(tensor).all()):
        raise TrainingDataError("preprocessed tensor contains NaN or Infinity")
    return tensor.contiguous()


def _verify_dataset_root(build: SyntheticDatasetBuild, root: Path) -> None:
    if not root.exists() or not root.is_dir():
        raise TrainingDataError("persisted Stage 6 dataset directory does not exist")
    validation = validate_dataset_manifest(build.manifest)
    if not validation.is_valid:
        codes = ", ".join(issue.code for issue in validation.issues)
        raise TrainingDataError(f"Stage 5 manifest vetoed the build: {codes}")

    expected_manifest = canonical_manifest_bytes(build.manifest)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.read_bytes() != expected_manifest:
        raise TrainingDataError("persisted manifest bytes do not match the validated build")
    if _sha256_bytes(expected_manifest) != build.manifest_sha256:
        raise TrainingDataError("build manifest identity is inconsistent")

    expected_metadata = build_metadata_bytes(build)
    metadata_path = root / "build.json"
    if not metadata_path.is_file() or metadata_path.read_bytes() != expected_metadata:
        raise TrainingDataError("persisted build metadata does not match the validated build")

    checksum_path = root / "manifest.sha256"
    expected_checksum = f"{build.manifest_sha256}  manifest.json\n".encode("ascii")
    if not checksum_path.is_file() or checksum_path.read_bytes() != expected_checksum:
        raise TrainingDataError("persisted manifest checksum record is invalid")


def load_training_samples(
    build: object,
    dataset_root: str | Path,
    split: DatasetSplit,
    *,
    max_samples: int | None = None,
) -> tuple[TrainingSampleRef, ...]:
    """Load only train/validation references; the sealed test split is rejected first."""

    if split is DatasetSplit.TEST:
        raise TrainingDataError("Stage 6 test split is sealed until Stage 9")
    if split not in {DatasetSplit.TRAIN, DatasetSplit.VALIDATION}:
        raise TrainingDataError("split must be DatasetSplit.TRAIN or DatasetSplit.VALIDATION")
    if not isinstance(build, SyntheticDatasetBuild):
        raise TypeError("build must be SyntheticDatasetBuild")
    if not isinstance(dataset_root, (str, Path)):
        raise TypeError("dataset_root must be str or pathlib.Path")
    if max_samples is not None and (
        not _is_plain_int(max_samples) or not 1 <= max_samples <= 1024
    ):
        raise TrainingDataError("max_samples must be an integer from 1 through 1024 for Stage 7-B")

    root = Path(dataset_root)
    _verify_dataset_root(build, root)

    selected = tuple(
        sorted(
            (sample for sample in build.manifest.samples if sample.split is split),
            key=lambda sample: sample.sample_id,
        )
    )
    if max_samples is not None:
        selected = selected[:max_samples]
    if not selected:
        raise TrainingDataError(f"validated build contains no {split.value} samples")

    target_cache: dict[str, tuple[int, ...]] = {}
    refs: list[TrainingSampleRef] = []
    for sample in selected:
        target_path = root / "targets" / f"{sample.source_musicxml_sha256}.musicxml"
        image_path = root / "images" / f"{sample.png_sha256}.png"
        if not target_path.is_file() or not image_path.is_file():
            raise TrainingDataError("required persisted Stage 6 artifact is missing")

        token_ids = target_cache.get(sample.source_musicxml_sha256)
        if token_ids is None:
            target_bytes = target_path.read_bytes()
            if _sha256_bytes(target_bytes) != sample.source_musicxml_sha256:
                raise TrainingDataError("persisted MusicXML target hash mismatch")
            try:
                tokenized = tokenize_musicxml(target_bytes)
            except TokenizationError as exc:
                raise TrainingDataError("MusicXML target failed Stage 7 token round trip") from exc
            token_ids = tokenized.token_ids
            if len(token_ids) > MAX_STAGE7_SEQUENCE_TOKENS:
                raise TrainingDataError("target token sequence exceeds the Stage 7-B ceiling")
            target_cache[sample.source_musicxml_sha256] = token_ids

        image_bytes = image_path.read_bytes()
        if _sha256_bytes(image_bytes) != sample.png_sha256:
            raise TrainingDataError("persisted PNG image hash mismatch")
        # Decode once at admission so mode/dimensions cannot be trusted from metadata alone.
        preprocess_grayscale_png(
            image_bytes,
            expected_width=sample.width,
            expected_height=sample.height,
        )

        refs.append(
            TrainingSampleRef(
                sample_id=sample.sample_id,
                family_id=sample.family_id,
                split=sample.split,
                image_path=image_path,
                image_sha256=sample.png_sha256,
                target_path=target_path,
                target_sha256=sample.source_musicxml_sha256,
                target_token_ids=token_ids,
                source_width=sample.width,
                source_height=sample.height,
            )
        )
    return tuple(refs)


def load_image_tensor(
    sample: TrainingSampleRef,
    config: InputPreprocessConfig = InputPreprocessConfig(),
) -> torch.Tensor:
    if not isinstance(sample, TrainingSampleRef):
        raise TypeError("sample must be TrainingSampleRef")
    data = sample.image_path.read_bytes()
    if _sha256_bytes(data) != sample.image_sha256:
        raise TrainingDataError("image changed after Stage 7-B admission")
    return preprocess_grayscale_png(
        data,
        config,
        expected_width=sample.source_width,
        expected_height=sample.source_height,
    )


def make_training_batch(
    samples: object,
    config: InputPreprocessConfig = InputPreprocessConfig(),
) -> TrainingBatch:
    if not isinstance(samples, tuple) or not samples:
        raise TrainingDataError("samples must be a non-empty immutable tuple")
    if any(not isinstance(sample, TrainingSampleRef) for sample in samples):
        raise TrainingDataError("samples contains a non-TrainingSampleRef value")
    split = samples[0].split
    if any(sample.split is not split for sample in samples):
        raise TrainingDataError("one batch may not mix dataset splits")
    if split is DatasetSplit.TEST:
        raise TrainingDataError("sealed test data cannot enter Stage 7-B")

    images = torch.stack(tuple(load_image_tensor(sample, config) for sample in samples), dim=0)
    max_steps = max(len(sample.target_token_ids) - 1 for sample in samples)
    decoder = torch.full(
        (len(samples), max_steps),
        PAD_TOKEN_ID,
        dtype=torch.long,
    )
    labels = torch.full(
        (len(samples), max_steps),
        PAD_TOKEN_ID,
        dtype=torch.long,
    )
    for row, sample in enumerate(samples):
        source = torch.tensor(sample.target_token_ids, dtype=torch.long)
        steps = len(sample.target_token_ids) - 1
        decoder[row, :steps] = source[:-1]
        labels[row, :steps] = source[1:]

    return TrainingBatch(
        images=images,
        decoder_input_ids=decoder,
        labels=labels,
        split=split,
    )
