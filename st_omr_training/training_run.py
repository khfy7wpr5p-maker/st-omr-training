"""Bounded Stage 7-C baseline training run and auditable evidence generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata
import json
import math
import platform
from pathlib import Path
import sys
from typing import Final

import torch

from .core import (
    ChordEvent,
    NoteEvent,
    NotationIntent,
    Pitch,
    RationalDuration,
    RestEvent,
)
from .dataset_builder import SyntheticDatasetBuild
from .dataset_manifest import DatasetSplit
from .musicxml_validator import validate_musicxml
from .musicxml_writer import MusicXMLWriteError, write_musicxml
from .structure import Measure, Part, Score, TimeSignature, Voice
from .structure_validator import validate_score
from .training_data import (
    InputPreprocessConfig,
    TrainingSampleRef,
    load_image_tensor,
    load_training_samples,
    make_training_batch,
    preprocess_config_fingerprint,
)
from .training_model import (
    BaselineModelConfig,
    BaselineSTOMRModel,
    TrainerConfig,
    assert_model_finite,
    build_baseline_model,
    count_trainable_parameters,
    model_config_fingerprint,
    model_state_sha256,
    train_one_smoke_step,
    trainer_config_fingerprint,
    validation_loss,
    verify_torch_runtime,
)
from .training_tokens import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    TokenizationError,
    decode_token_ids,
    detokenize_tokens,
    tokenizer_fingerprint,
)


BASELINE_RUN_VERSION: Final[str] = "st-omr-baseline-run-v1"
MAX_BASELINE_EPOCHS: Final[int] = 100
MAX_RETAINED_CHECKPOINTS: Final[int] = 10
_MAX_SEED: Final[int] = 2**63 - 1
_HEX = frozenset("0123456789abcdef")


class BaselineRunConfigError(ValueError):
    """Raised when a Stage 7-C run request violates the frozen bounded contract."""


class BaselineRunError(RuntimeError):
    """Raised when a Stage 7-C run cannot produce acceptable auditable evidence."""


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_hex(name: str, value: object, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(char not in _HEX for char in value)
    ):
        raise BaselineRunConfigError(
            f"{name} must be a lowercase {length}-character hexadecimal digest"
        )
    return value


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BaselineRunConfig:
    """Frozen bounded execution policy for the first real synthetic-only baseline."""

    epochs: int = 40
    batch_size: int = 4
    max_train_samples: int = 1024
    max_validation_samples: int = 256
    max_decode_tokens: int = 512
    retained_checkpoints: int = 1

    def __post_init__(self) -> None:
        bounds = (
            ("epochs", self.epochs, 1, MAX_BASELINE_EPOCHS),
            ("batch_size", self.batch_size, 1, 64),
            ("max_train_samples", self.max_train_samples, 1, 1024),
            ("max_validation_samples", self.max_validation_samples, 1, 1024),
            ("max_decode_tokens", self.max_decode_tokens, 8, 4096),
        )
        for name, value, lower, upper in bounds:
            if not _is_plain_int(value) or not lower <= value <= upper:
                raise BaselineRunConfigError(
                    f"{name} must be an integer from {lower} through {upper}"
                )
        if self.retained_checkpoints != 1:
            raise BaselineRunConfigError(
                "retained_checkpoints is frozen to 1 for the first Stage 7-C baseline"
            )


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    token_error_rate: float
    exact_sequence_accuracy: float
    detokenization_success_rate: float
    semantic_validity_rate: float
    musicxml_regeneration_validity_rate: float
    validation_samples: int
    valid_semantic_predictions: int

    def __post_init__(self) -> None:
        if not _is_plain_int(self.validation_samples) or self.validation_samples <= 0:
            raise BaselineRunError("validation_samples must be positive")
        if (
            not _is_plain_int(self.valid_semantic_predictions)
            or not 0 <= self.valid_semantic_predictions <= self.validation_samples
        ):
            raise BaselineRunError("valid_semantic_predictions is outside its valid range")
        for name in (
            "token_error_rate",
            "exact_sequence_accuracy",
            "detokenization_success_rate",
            "semantic_validity_rate",
            "musicxml_regeneration_validity_rate",
        ):
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value) or value < 0:
                raise BaselineRunError(f"{name} must be finite and non-negative")
        for name in (
            "exact_sequence_accuracy",
            "detokenization_success_rate",
            "semantic_validity_rate",
            "musicxml_regeneration_validity_rate",
        ):
            if getattr(self, name) > 1:
                raise BaselineRunError(f"{name} must not exceed 1")


@dataclass(frozen=True, slots=True)
class BaselineRunResult:
    run_id: str
    run_directory: Path
    repository_sha: str
    dataset_build_id: str
    manifest_sha256: str
    untrained_validation_loss: float
    best_validation_loss: float
    best_epoch: int
    training_steps: int
    checkpoint_sha256: str
    metrics_sha256: str
    prediction_metrics: PredictionMetrics

    def __post_init__(self) -> None:
        _require_hex("run_id", self.run_id, 64)
        _require_hex("repository_sha", self.repository_sha, 40)
        _require_hex("dataset_build_id", self.dataset_build_id, 64)
        _require_hex("manifest_sha256", self.manifest_sha256, 64)
        _require_hex("checkpoint_sha256", self.checkpoint_sha256, 64)
        _require_hex("metrics_sha256", self.metrics_sha256, 64)
        if not isinstance(self.run_directory, Path):
            raise BaselineRunError("run_directory must be pathlib.Path")
        if (
            not math.isfinite(self.untrained_validation_loss)
            or not math.isfinite(self.best_validation_loss)
        ):
            raise BaselineRunError("validation losses must be finite")
        if self.best_validation_loss >= self.untrained_validation_loss:
            raise BaselineRunError(
                "best validation loss must strictly improve on the untrained baseline"
            )
        if not _is_plain_int(self.best_epoch) or self.best_epoch <= 0:
            raise BaselineRunError("best_epoch must be positive")
        if not _is_plain_int(self.training_steps) or self.training_steps <= 0:
            raise BaselineRunError("training_steps must be positive")
        if self.prediction_metrics.valid_semantic_predictions < 1:
            raise BaselineRunError(
                "Stage 7-C requires at least one semantically valid validation prediction"
            )


def baseline_run_config_fingerprint(
    run_config: BaselineRunConfig,
    model_config: BaselineModelConfig,
    trainer_config: TrainerConfig,
    preprocess_config: InputPreprocessConfig,
) -> str:
    if not isinstance(run_config, BaselineRunConfig):
        raise TypeError("run_config must be BaselineRunConfig")
    payload = {
        "run_version": BASELINE_RUN_VERSION,
        "run_config": asdict(run_config),
        "model_fingerprint": model_config_fingerprint(model_config),
        "trainer_fingerprint": trainer_config_fingerprint(trainer_config),
        "preprocess_fingerprint": preprocess_config_fingerprint(preprocess_config),
        "tokenizer_fingerprint": tokenizer_fingerprint(),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _batch_groups(
    samples: tuple[TrainingSampleRef, ...],
    batch_size: int,
) -> tuple[tuple[TrainingSampleRef, ...], ...]:
    if not samples:
        raise BaselineRunError("cannot batch an empty sample set")
    return tuple(
        samples[index : index + batch_size]
        for index in range(0, len(samples), batch_size)
    )


def _mean_validation_loss(
    model: BaselineSTOMRModel,
    samples: tuple[TrainingSampleRef, ...],
    *,
    batch_size: int,
    preprocess_config: InputPreprocessConfig,
) -> float:
    weighted_loss = 0.0
    token_count = 0
    for group in _batch_groups(samples, batch_size):
        batch = make_training_batch(group, preprocess_config)
        value = validation_loss(model, batch)
        count = int((batch.labels != PAD_TOKEN_ID).sum().item())
        if count <= 0:
            raise BaselineRunError("validation batch contains no unmasked tokens")
        weighted_loss += value * count
        token_count += count
    result = weighted_loss / token_count
    if not math.isfinite(result):
        raise BaselineRunError("validation loss is NaN or Infinity")
    return result


def _levenshtein_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_item != right_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _greedy_decode_sample(
    model: BaselineSTOMRModel,
    sample: TrainingSampleRef,
    *,
    preprocess_config: InputPreprocessConfig,
    max_decode_tokens: int,
) -> tuple[int, ...]:
    image = load_image_tensor(sample, preprocess_config).unsqueeze(0).cpu()
    predicted = [BOS_TOKEN_ID]
    model.eval()
    with torch.no_grad():
        for _ in range(max_decode_tokens - 1):
            decoder = torch.tensor([predicted], dtype=torch.long)
            logits = model(image, decoder)
            next_id = int(torch.argmax(logits[0, -1]).item())
            predicted.append(next_id)
            if next_id == EOS_TOKEN_ID:
                break
    return tuple(predicted)


def _score_from_projection(projection: object, *, score_id: str) -> Score:
    from .musicxml_roundtrip import SemanticScoreProjection

    if not isinstance(projection, SemanticScoreProjection):
        raise BaselineRunError("prediction did not produce a semantic score projection")

    measures: list[Measure] = []
    for measure_projection in projection.parts[0].measures:
        events = []
        for event_projection in measure_projection.voices[0].events:
            duration = RationalDuration(
                event_projection.duration.numerator,
                event_projection.duration.denominator,
            )
            if event_projection.event_type == "rest":
                events.append(
                    RestEvent(
                        onset=event_projection.onset,
                        duration=duration,
                        voice=1,
                        staff=1,
                    )
                )
                continue

            notes = tuple(
                NoteEvent(
                    onset=event_projection.onset,
                    duration=duration,
                    pitch=Pitch(pitch.step, pitch.alter, pitch.octave),
                    notation_intent=NotationIntent(pitch.display_accidental),
                    voice=1,
                    staff=1,
                )
                for pitch in event_projection.pitches
            )
            if event_projection.event_type == "note":
                if len(notes) != 1:
                    raise BaselineRunError("note projection has the wrong pitch count")
                events.append(notes[0])
            elif event_projection.event_type == "chord":
                events.append(
                    ChordEvent(
                        onset=event_projection.onset,
                        duration=duration,
                        notes=notes,
                        voice=1,
                        staff=1,
                    )
                )
            else:
                raise BaselineRunError("unsupported predicted event type")

        measures.append(
            Measure(
                number=measure_projection.number,
                time_signature=TimeSignature(*measure_projection.time_signature),
                voices=(Voice(voice_id=1, events=tuple(events)),),
                key_signature=0,
            )
        )

    score = Score(
        score_id=score_id,
        schema_version="st-canonical-1",
        generator_version="stage7c-reconstruction-v1",
        seed=0,
        provenance=(("source", "stage7c-greedy-prediction"),),
        parts=(Part(part_id="P1", measures=tuple(measures)),),
    )
    validation = validate_score(score)
    if not validation.is_valid:
        raise BaselineRunError("predicted semantic projection failed canonical validation")
    return score


def _evaluate_predictions(
    model: BaselineSTOMRModel,
    validation_samples: tuple[TrainingSampleRef, ...],
    *,
    preprocess_config: InputPreprocessConfig,
    max_decode_tokens: int,
) -> PredictionMetrics:
    total_edits = 0
    total_reference_tokens = 0
    exact = 0
    detokenized = 0
    semantic_valid = 0
    musicxml_valid = 0

    for sample in validation_samples:
        predicted = _greedy_decode_sample(
            model,
            sample,
            preprocess_config=preprocess_config,
            max_decode_tokens=max_decode_tokens,
        )
        target = sample.target_token_ids
        predicted_surface = predicted[1:]
        target_surface = target[1:]
        total_edits += _levenshtein_distance(predicted_surface, target_surface)
        total_reference_tokens += len(target_surface)
        if predicted == target:
            exact += 1

        try:
            predicted_tokens = decode_token_ids(predicted)
            projection = detokenize_tokens(predicted_tokens)
        except TokenizationError:
            continue
        detokenized += 1

        try:
            score = _score_from_projection(
                projection,
                score_id=f"stage7c-{sample.sample_id[:16]}",
            )
        except (BaselineRunError, TypeError, ValueError):
            continue
        semantic_valid += 1

        try:
            musicxml = write_musicxml(score)
        except MusicXMLWriteError:
            continue
        validation = validate_musicxml(musicxml)
        if validation.is_valid:
            musicxml_valid += 1

    count = len(validation_samples)
    if count <= 0 or total_reference_tokens <= 0:
        raise BaselineRunError("validation evaluation has no reference tokens")
    return PredictionMetrics(
        token_error_rate=float(total_edits / total_reference_tokens),
        exact_sequence_accuracy=float(exact / count),
        detokenization_success_rate=float(detokenized / count),
        semantic_validity_rate=float(semantic_valid / count),
        musicxml_regeneration_validity_rate=float(musicxml_valid / count),
        validation_samples=count,
        valid_semantic_predictions=semantic_valid,
    )


def _dependency_versions() -> dict[str, str]:
    result = {"torch": verify_torch_runtime()}
    for distribution in ("Pillow", "lxml", "verovio", "CairoSVG"):
        try:
            result[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            result[distribution] = "not-installed"
    return result


def _write_checkpoint(
    model: BaselineSTOMRModel,
    run_directory: Path,
    *,
    epoch: int,
    model_fingerprint: str,
) -> tuple[Path, str]:
    temporary = run_directory / "checkpoint.tmp"
    if temporary.exists():
        raise BaselineRunError("temporary checkpoint path already exists")
    torch.save(
        {
            "epoch": epoch,
            "model_fingerprint": model_fingerprint,
            "model_state_dict": model.state_dict(),
        },
        temporary,
    )
    digest = _sha256_file(temporary)
    final = run_directory / f"checkpoint-{digest}.pt"
    if final.exists():
        raise BaselineRunError("checkpoint hash path already exists")
    temporary.rename(final)
    return final, digest


def run_baseline_training(
    build: object,
    dataset_root: str | Path,
    run_root: str | Path,
    *,
    repository_sha: str,
    run_config: BaselineRunConfig = BaselineRunConfig(),
    model_config: BaselineModelConfig = BaselineModelConfig(),
    trainer_config: TrainerConfig = TrainerConfig(),
    preprocess_config: InputPreprocessConfig = InputPreprocessConfig(),
) -> BaselineRunResult:
    """Execute one strict Stage 7-C CPU baseline run without opening the test split."""

    if not isinstance(build, SyntheticDatasetBuild):
        raise TypeError("build must be SyntheticDatasetBuild")
    _require_hex("repository_sha", repository_sha, 40)
    if not isinstance(dataset_root, (str, Path)) or not isinstance(run_root, (str, Path)):
        raise TypeError("dataset_root and run_root must be str or pathlib.Path")
    if not isinstance(run_config, BaselineRunConfig):
        raise TypeError("run_config must be BaselineRunConfig")
    if not isinstance(model_config, BaselineModelConfig):
        raise TypeError("model_config must be BaselineModelConfig")
    if not isinstance(trainer_config, TrainerConfig):
        raise TypeError("trainer_config must be TrainerConfig")
    if not isinstance(preprocess_config, InputPreprocessConfig):
        raise TypeError("preprocess_config must be InputPreprocessConfig")
    if (
        model_config.input_height != preprocess_config.target_height
        or model_config.input_width != preprocess_config.target_width
    ):
        raise BaselineRunConfigError(
            "model input dimensions must exactly match the frozen preprocess canvas"
        )
    if not 0 <= trainer_config.master_seed <= _MAX_SEED:
        raise BaselineRunConfigError("trainer master seed is outside the Stage 7-C range")

    verify_torch_runtime()
    run_fingerprint = baseline_run_config_fingerprint(
        run_config,
        model_config,
        trainer_config,
        preprocess_config,
    )
    identity_payload = {
        "run_version": BASELINE_RUN_VERSION,
        "repository_sha": repository_sha,
        "dataset_build_id": build.build_id,
        "manifest_sha256": build.manifest_sha256,
        "run_fingerprint": run_fingerprint,
    }
    run_id = sha256(_canonical_json_bytes(identity_payload)).hexdigest()

    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    run_directory = root / run_id
    if run_directory.exists():
        raise BaselineRunError("run directory already exists; silent resume is forbidden")
    run_directory.mkdir()
    incomplete = run_directory / "INCOMPLETE"
    incomplete.write_bytes(_canonical_json_bytes(identity_payload))

    train_samples = load_training_samples(
        build,
        dataset_root,
        DatasetSplit.TRAIN,
        max_samples=run_config.max_train_samples,
    )
    validation_samples = load_training_samples(
        build,
        dataset_root,
        DatasetSplit.VALIDATION,
        max_samples=run_config.max_validation_samples,
    )
    if any(sample.split is DatasetSplit.TEST for sample in train_samples + validation_samples):
        raise BaselineRunError("sealed test split crossed the Stage 7-C data boundary")
    if any(
        len(sample.target_token_ids) > run_config.max_decode_tokens
        for sample in validation_samples
    ):
        raise BaselineRunConfigError(
            "max_decode_tokens is shorter than an admitted validation target"
        )

    model = build_baseline_model(
        model_config,
        seed=trainer_config.master_seed,
    )
    parameter_count = count_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer_config.learning_rate_micros / 1_000_000,
        weight_decay=trainer_config.weight_decay_micros / 1_000_000,
        foreach=False,
        fused=False,
    )

    untrained_validation_loss = _mean_validation_loss(
        model,
        validation_samples,
        batch_size=run_config.batch_size,
        preprocess_config=preprocess_config,
    )
    best_validation_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_checkpoint: Path | None = None
    best_checkpoint_sha = ""
    training_steps = 0
    epoch_records: list[dict[str, object]] = []

    for epoch in range(1, run_config.epochs + 1):
        train_loss_sum = 0.0
        train_step_count = 0
        for group in _batch_groups(train_samples, run_config.batch_size):
            batch = make_training_batch(group, preprocess_config)
            value = train_one_smoke_step(model, batch, optimizer, trainer_config)
            train_loss_sum += value
            train_step_count += 1
            training_steps += 1

        if train_step_count <= 0:
            raise BaselineRunError("training epoch executed no optimizer steps")
        mean_train_loss = train_loss_sum / train_step_count
        current_validation_loss = _mean_validation_loss(
            model,
            validation_samples,
            batch_size=run_config.batch_size,
            preprocess_config=preprocess_config,
        )
        epoch_records.append(
            {
                "epoch": epoch,
                "mean_train_loss": mean_train_loss,
                "validation_loss": current_validation_loss,
            }
        )
        if current_validation_loss < best_validation_loss:
            previous_checkpoint = best_checkpoint
            best_validation_loss = current_validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            best_checkpoint, best_checkpoint_sha = _write_checkpoint(
                model,
                run_directory,
                epoch=epoch,
                model_fingerprint=model_config_fingerprint(model_config),
            )
            if previous_checkpoint is not None and previous_checkpoint != best_checkpoint:
                previous_checkpoint.unlink()

    if best_state is None or best_checkpoint is None:
        raise BaselineRunError("training did not produce a selected checkpoint")
    if not best_validation_loss < untrained_validation_loss:
        raise BaselineRunError(
            "best validation loss did not improve on the deterministic untrained baseline"
        )

    model.load_state_dict(best_state)
    assert_model_finite(model)
    if _sha256_file(best_checkpoint) != best_checkpoint_sha:
        raise BaselineRunError("selected checkpoint changed after hashing")

    prediction_metrics = _evaluate_predictions(
        model,
        validation_samples,
        preprocess_config=preprocess_config,
        max_decode_tokens=run_config.max_decode_tokens,
    )
    if prediction_metrics.valid_semantic_predictions < 1:
        raise BaselineRunError(
            "no validation prediction successfully crossed the semantic gate"
        )

    evidence = {
        "schema_version": "stage7c-evidence-v1",
        "run_version": BASELINE_RUN_VERSION,
        "run_id": run_id,
        "repository_sha": repository_sha,
        "dataset": {
            "build_id": build.build_id,
            "manifest_sha256": build.manifest_sha256,
            "config_fingerprint": build.config_fingerprint,
            "immutable_reference": "validated Stage 6 build manifest",
        },
        "fingerprints": {
            "run": run_fingerprint,
            "tokenizer": tokenizer_fingerprint(),
            "preprocess": preprocess_config_fingerprint(preprocess_config),
            "model": model_config_fingerprint(model_config),
            "trainer": trainer_config_fingerprint(trainer_config),
        },
        "configuration": {
            "run": asdict(run_config),
            "model": asdict(model_config),
            "trainer": asdict(trainer_config),
            "preprocess": asdict(preprocess_config),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "device": "cpu",
            "dependencies": _dependency_versions(),
        },
        "seeds": {
            "master": trainer_config.master_seed,
            "model_initialization": trainer_config.master_seed,
        },
        "training": {
            "parameter_count": parameter_count,
            "data_ordering": "sample_id-ascending-fixed-each-epoch",
            "epochs_completed": run_config.epochs,
            "training_steps": training_steps,
            "untrained_validation_loss": untrained_validation_loss,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "epoch_records": epoch_records,
        },
        "prediction_metrics": asdict(prediction_metrics),
        "checkpoint": {
            "sha256": best_checkpoint_sha,
            "state_sha256": model_state_sha256(model),
            "filename": best_checkpoint.name,
        },
        "sealed_test_split_opened": False,
    }
    evidence_bytes = _canonical_json_bytes(evidence)
    metrics_sha = sha256(evidence_bytes).hexdigest()
    metrics_path = run_directory / f"metrics-{metrics_sha}.json"
    metrics_path.write_bytes(evidence_bytes)
    if _sha256_file(metrics_path) != metrics_sha:
        raise BaselineRunError("metrics evidence changed after hashing")

    (run_directory / "COMPLETE").write_text(
        f"{metrics_sha}  {metrics_path.name}\n",
        encoding="ascii",
    )
    incomplete.unlink()
    return BaselineRunResult(
        run_id=run_id,
        run_directory=run_directory,
        repository_sha=repository_sha,
        dataset_build_id=build.build_id,
        manifest_sha256=build.manifest_sha256,
        untrained_validation_loss=float(untrained_validation_loss),
        best_validation_loss=float(best_validation_loss),
        best_epoch=best_epoch,
        training_steps=training_steps,
        checkpoint_sha256=best_checkpoint_sha,
        metrics_sha256=metrics_sha,
        prediction_metrics=prediction_metrics,
    )
