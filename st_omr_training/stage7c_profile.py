"""Exact immutable configuration objects for the authoritative Stage 7-C run."""

from __future__ import annotations

from typing import Final

from .training_data import InputPreprocessConfig
from .training_model import BaselineModelConfig, TrainerConfig
from .training_run import BaselineRunConfig, baseline_run_config_fingerprint


STAGE7C_FROZEN_RUN_CONFIG: Final[BaselineRunConfig] = BaselineRunConfig(
    epochs=40,
    batch_size=4,
    max_train_samples=1024,
    max_validation_samples=256,
    max_decode_tokens=1536,
    decode_measure_count=8,
    retained_checkpoints=1,
)
STAGE7C_FROZEN_MODEL_CONFIG: Final[BaselineModelConfig] = BaselineModelConfig()
STAGE7C_FROZEN_TRAINER_CONFIG: Final[TrainerConfig] = TrainerConfig()
STAGE7C_FROZEN_PREPROCESS_CONFIG: Final[InputPreprocessConfig] = InputPreprocessConfig()
STAGE7C_FROZEN_RUN_FINGERPRINT: Final[str] = baseline_run_config_fingerprint(
    STAGE7C_FROZEN_RUN_CONFIG,
    STAGE7C_FROZEN_MODEL_CONFIG,
    STAGE7C_FROZEN_TRAINER_CONFIG,
    STAGE7C_FROZEN_PREPROCESS_CONFIG,
)
