"""Stage 8-2 paired real-data experiment profile freeze.

Configuration and hash-binding only: no real-data IO, checkpoint loading,
training, sealed-test access, or ScoreMosaic integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Final

from .real_data_contract import (
    RealDataManifest,
    RealDataSplit,
    STAGE7C_CHECKPOINT_SHA256,
    STAGE7C_MODEL_STATE_SHA256,
    real_data_manifest_sha256,
    validate_real_data_manifest,
)
from .real_data_intake import intake_policy_fingerprint
from .training_data import InputPreprocessConfig, preprocess_config_fingerprint
from .training_model import (
    BaselineModelConfig,
    TrainerConfig,
    model_config_fingerprint,
    trainer_config_fingerprint,
)
from .training_tokens import tokenizer_fingerprint

STAGE8_PAIRED_PROFILE_VERSION: Final[str] = "st-stage8-paired-pilot-v1"
PILOT_TOTAL_SAMPLES: Final[int] = 50
PILOT_TRAIN_SAMPLES: Final[int] = 40
PILOT_VALIDATION_SAMPLES: Final[int] = 10
PILOT_EPOCHS: Final[int] = 40
PILOT_BATCH_SIZE: Final[int] = 4
PILOT_MAX_DECODE_TOKENS: Final[int] = 1536
PILOT_DECODE_MEASURES: Final[int] = 8
PILOT_RETAINED_CHECKPOINTS: Final[int] = 1
PILOT_MAX_SECONDS_PER_CANDIDATE: Final[int] = 1800
PILOT_MAX_TOTAL_SECONDS: Final[int] = 3600
PILOT_METRICS: Final[tuple[str, ...]] = (
    "validation_loss",
    "token_error_rate",
    "exact_sequence_accuracy",
    "detokenization_success_rate",
    "semantic_validity_rate",
    "musicxml_regeneration_validity_rate",
)
_HEX = frozenset("0123456789abcdef")


class Stage8ProfileError(ValueError):
    """Raised when the frozen Stage 8-2 profile or binding is violated."""


class Stage8Candidate(str, Enum):
    CHECKPOINT_FINE_TUNE = "candidate-a-stage7c-fine-tune"
    FROM_SCRATCH = "candidate-b-from-scratch"


def _hex64(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in _HEX for ch in value):
        raise Stage8ProfileError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


@dataclass(frozen=True, slots=True)
class Stage8PairedRunProfile:
    total_samples: int = PILOT_TOTAL_SAMPLES
    train_samples: int = PILOT_TRAIN_SAMPLES
    validation_samples: int = PILOT_VALIDATION_SAMPLES
    epochs: int = PILOT_EPOCHS
    batch_size: int = PILOT_BATCH_SIZE
    max_decode_tokens: int = PILOT_MAX_DECODE_TOKENS
    decode_measure_count: int = PILOT_DECODE_MEASURES
    retained_checkpoints: int = PILOT_RETAINED_CHECKPOINTS
    max_seconds_per_candidate: int = PILOT_MAX_SECONDS_PER_CANDIDATE
    max_total_seconds: int = PILOT_MAX_TOTAL_SECONDS
    cpu_threads: int = 1
    device: str = "cpu"
    split_policy: str = "exact-40-train-10-validation-family-exclusive-v1"
    checkpoint_selection: str = "min_validation_loss"
    data_order: str = "canonical-sample-id-order-v1"
    metrics: tuple[str, ...] = PILOT_METRICS
    model_config: BaselineModelConfig = BaselineModelConfig()
    trainer_config: TrainerConfig = TrainerConfig()
    preprocess_config: InputPreprocessConfig = InputPreprocessConfig()
    profile_version: str = STAGE8_PAIRED_PROFILE_VERSION

    def __post_init__(self) -> None:
        expected_ints = {
            "total_samples": PILOT_TOTAL_SAMPLES,
            "train_samples": PILOT_TRAIN_SAMPLES,
            "validation_samples": PILOT_VALIDATION_SAMPLES,
            "epochs": PILOT_EPOCHS,
            "batch_size": PILOT_BATCH_SIZE,
            "max_decode_tokens": PILOT_MAX_DECODE_TOKENS,
            "decode_measure_count": PILOT_DECODE_MEASURES,
            "retained_checkpoints": PILOT_RETAINED_CHECKPOINTS,
            "max_seconds_per_candidate": PILOT_MAX_SECONDS_PER_CANDIDATE,
            "max_total_seconds": PILOT_MAX_TOTAL_SECONDS,
            "cpu_threads": 1,
        }
        for name, expected in expected_ints.items():
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value != expected:
                raise Stage8ProfileError(f"{name} is frozen to {expected}")
        if self.train_samples + self.validation_samples != self.total_samples:
            raise Stage8ProfileError("train/validation counts must sum to total_samples")
        if self.device != "cpu":
            raise Stage8ProfileError("first Stage 8 paired pilot is CPU-only")
        if self.split_policy != "exact-40-train-10-validation-family-exclusive-v1":
            raise Stage8ProfileError("split_policy is frozen")
        if self.checkpoint_selection != "min_validation_loss":
            raise Stage8ProfileError("checkpoint_selection is frozen")
        if self.data_order != "canonical-sample-id-order-v1":
            raise Stage8ProfileError("data_order is frozen")
        if self.metrics != PILOT_METRICS:
            raise Stage8ProfileError("metric family is frozen")
        if self.model_config != BaselineModelConfig():
            raise Stage8ProfileError("model config must equal the Stage 7-C baseline")
        if self.trainer_config != TrainerConfig():
            raise Stage8ProfileError("trainer config must equal the Stage 7-C baseline")
        if self.preprocess_config != InputPreprocessConfig():
            raise Stage8ProfileError("preprocess config must equal the Stage 7-C baseline")
        if self.profile_version != STAGE8_PAIRED_PROFILE_VERSION:
            raise Stage8ProfileError("profile_version is frozen")


@dataclass(frozen=True, slots=True)
class Stage8ManifestSummary:
    manifest_sha256: str
    train_samples: int
    validation_samples: int
    train_families: int
    validation_families: int


@dataclass(frozen=True, slots=True)
class Stage8ExperimentBinding:
    candidate: Stage8Candidate
    profile_fingerprint: str
    development_manifest_sha256: str
    receipt_set_sha256: str
    sealed_test_manifest_sha256: str
    initialization_checkpoint_sha256: str | None
    initialization_model_state_sha256: str | None
    test_accessed: bool = False
    online_learning: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, Stage8Candidate):
            raise Stage8ProfileError("candidate must be Stage8Candidate")
        for name in ("profile_fingerprint", "development_manifest_sha256", "receipt_set_sha256", "sealed_test_manifest_sha256"):
            _hex64(name, getattr(self, name))
        if self.test_accessed is not False:
            raise Stage8ProfileError("sealed test access is prohibited")
        if self.online_learning is not False:
            raise Stage8ProfileError("online/automatic learning is prohibited")
        if self.candidate is Stage8Candidate.CHECKPOINT_FINE_TUNE:
            if self.initialization_checkpoint_sha256 != STAGE7C_CHECKPOINT_SHA256:
                raise Stage8ProfileError("Candidate A must use the exact accepted Stage 7-C checkpoint")
            if self.initialization_model_state_sha256 != STAGE7C_MODEL_STATE_SHA256:
                raise Stage8ProfileError("Candidate A must use the exact accepted Stage 7-C model state")
        elif self.initialization_checkpoint_sha256 is not None or self.initialization_model_state_sha256 is not None:
            raise Stage8ProfileError("Candidate B must start from deterministic initialization")


def stage8_paired_profile_fingerprint(profile: Stage8PairedRunProfile = Stage8PairedRunProfile()) -> str:
    if not isinstance(profile, Stage8PairedRunProfile):
        raise TypeError("profile must be Stage8PairedRunProfile")
    payload = {
        "profile_version": profile.profile_version,
        "total_samples": profile.total_samples,
        "train_samples": profile.train_samples,
        "validation_samples": profile.validation_samples,
        "epochs": profile.epochs,
        "batch_size": profile.batch_size,
        "max_decode_tokens": profile.max_decode_tokens,
        "decode_measure_count": profile.decode_measure_count,
        "retained_checkpoints": profile.retained_checkpoints,
        "max_seconds_per_candidate": profile.max_seconds_per_candidate,
        "max_total_seconds": profile.max_total_seconds,
        "cpu_threads": profile.cpu_threads,
        "device": profile.device,
        "split_policy": profile.split_policy,
        "checkpoint_selection": profile.checkpoint_selection,
        "data_order": profile.data_order,
        "metrics": list(profile.metrics),
        "model_fingerprint": model_config_fingerprint(profile.model_config),
        "trainer_fingerprint": trainer_config_fingerprint(profile.trainer_config),
        "preprocess_fingerprint": preprocess_config_fingerprint(profile.preprocess_config),
        "tokenizer_fingerprint": tokenizer_fingerprint(),
        "intake_policy_fingerprint": intake_policy_fingerprint(),
        "candidate_a_checkpoint_sha256": STAGE7C_CHECKPOINT_SHA256,
        "candidate_a_model_state_sha256": STAGE7C_MODEL_STATE_SHA256,
        "candidate_a_fallback": "forbidden",
        "candidate_b_initialization": "deterministic-from-scratch",
        "candidate_b_master_seed": profile.trainer_config.master_seed,
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def summarize_stage8_pilot_manifest(manifest: object, profile: Stage8PairedRunProfile = Stage8PairedRunProfile()) -> Stage8ManifestSummary:
    if not isinstance(manifest, RealDataManifest):
        raise Stage8ProfileError("manifest must be RealDataManifest")
    result = validate_real_data_manifest(manifest)
    if not result.is_valid:
        first = result.issues[0]
        raise Stage8ProfileError(f"manifest invalid: {first.code} at {first.path}: {first.message}")
    train = tuple(s for s in manifest.samples if s.split is RealDataSplit.TRAIN)
    validation = tuple(s for s in manifest.samples if s.split is RealDataSplit.VALIDATION)
    if len(manifest.samples) != profile.total_samples:
        raise Stage8ProfileError("pilot requires exactly 50 admitted development samples")
    if len(train) != profile.train_samples or len(validation) != profile.validation_samples:
        raise Stage8ProfileError("pilot requires exactly 40 train and 10 validation samples")
    return Stage8ManifestSummary(
        manifest_sha256=real_data_manifest_sha256(manifest),
        train_samples=len(train),
        validation_samples=len(validation),
        train_families=len({s.family_id for s in train}),
        validation_families=len({s.family_id for s in validation}),
    )


def stage8_receipt_set_sha256(receipt_sha256s: object) -> str:
    if not isinstance(receipt_sha256s, tuple) or len(receipt_sha256s) != PILOT_TOTAL_SAMPLES:
        raise Stage8ProfileError("receipt set must contain exactly 50 immutable receipt hashes")
    checked = tuple(_hex64("receipt_sha256", item) for item in receipt_sha256s)
    if len(set(checked)) != len(checked):
        raise Stage8ProfileError("receipt set contains duplicate identities")
    payload = {"version": "st-stage8-receipt-set-v1", "receipt_sha256s": sorted(checked)}
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def make_stage8_candidate_binding(candidate: Stage8Candidate, *, development_manifest_sha256: str, receipt_set_sha256: str, sealed_test_manifest_sha256: str, profile: Stage8PairedRunProfile = Stage8PairedRunProfile()) -> Stage8ExperimentBinding:
    if candidate is Stage8Candidate.CHECKPOINT_FINE_TUNE:
        checkpoint = STAGE7C_CHECKPOINT_SHA256
        model_state = STAGE7C_MODEL_STATE_SHA256
    elif candidate is Stage8Candidate.FROM_SCRATCH:
        checkpoint = None
        model_state = None
    else:
        raise Stage8ProfileError("candidate must be Stage8Candidate")
    return Stage8ExperimentBinding(
        candidate=candidate,
        profile_fingerprint=stage8_paired_profile_fingerprint(profile),
        development_manifest_sha256=_hex64("development_manifest_sha256", development_manifest_sha256),
        receipt_set_sha256=_hex64("receipt_set_sha256", receipt_set_sha256),
        sealed_test_manifest_sha256=_hex64("sealed_test_manifest_sha256", sealed_test_manifest_sha256),
        initialization_checkpoint_sha256=checkpoint,
        initialization_model_state_sha256=model_state,
    )


def validate_paired_experiment_bindings(left: object, right: object) -> tuple[Stage8ExperimentBinding, Stage8ExperimentBinding]:
    if not isinstance(left, Stage8ExperimentBinding) or not isinstance(right, Stage8ExperimentBinding):
        raise Stage8ProfileError("paired bindings must be Stage8ExperimentBinding values")
    by_candidate = {left.candidate: left, right.candidate: right}
    required = {Stage8Candidate.CHECKPOINT_FINE_TUNE, Stage8Candidate.FROM_SCRATCH}
    if set(by_candidate) != required:
        raise Stage8ProfileError("paired experiment requires exactly Candidate A and Candidate B")
    candidate_a = by_candidate[Stage8Candidate.CHECKPOINT_FINE_TUNE]
    candidate_b = by_candidate[Stage8Candidate.FROM_SCRATCH]
    for name in ("profile_fingerprint", "development_manifest_sha256", "receipt_set_sha256", "sealed_test_manifest_sha256", "test_accessed", "online_learning"):
        if getattr(candidate_a, name) != getattr(candidate_b, name):
            raise Stage8ProfileError(f"paired candidates differ at forbidden field {name}")
    return candidate_a, candidate_b
