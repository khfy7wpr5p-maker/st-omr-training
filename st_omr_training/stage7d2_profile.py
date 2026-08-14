"""Frozen Stage 7-D2 Synthetic Curriculum v1 training profile."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Final

from .stage7c_profile import (
    STAGE7C_FROZEN_MODEL_CONFIG,
    STAGE7C_FROZEN_PREPROCESS_CONFIG,
    STAGE7C_FROZEN_TRAINER_CONFIG,
)
from .synthetic_curriculum_acceptance import (
    EXPECTED_BUILD_ID,
    EXPECTED_CONFIG_FINGERPRINT,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SOURCE_COMMIT,
    EXPECTED_TRANSPORT_SHA256,
)
from .synthetic_curriculum_corpus_gate import (
    EXPECTED_ARCHIVE_NAME,
    EXPECTED_ARCHIVE_SIZE_BYTES,
)
from .training_data import preprocess_config_fingerprint
from .training_model import model_config_fingerprint, trainer_config_fingerprint
from .training_tokens import tokenizer_fingerprint


STAGE7D2_RUN_VERSION: Final[str] = "stage7d2-synthetic-v1-run-v1"
STAGE7D2_EVIDENCE_SCHEMA: Final[str] = "stage7d2-evidence-v1"
STAGE7D2_VERIFICATION_SCHEMA: Final[str] = "stage7d2-authoritative-verification-v1"

EXPECTED_D1_ARTIFACT_BINDING_SHA256: Final[str] = (
    "e603b945c6dc60cf7e618ae28a7734dee97cf0e05a81891479107b18a87af540"
)
EXPECTED_D1_TARGET_BYTES_TOTAL: Final[int] = 3_506_839
EXPECTED_D1_IMAGE_BYTES_TOTAL: Final[int] = 494_937_881


@dataclass(frozen=True, slots=True)
class Stage7D2RunConfig:
    epochs: int = 40
    batch_size: int = 4
    train_samples: int = 1230
    validation_samples: int = 153
    train_families: int = 410
    validation_families: int = 51
    max_decode_tokens: int = 1536
    decode_measure_count: int = 8
    retained_checkpoints: int = 1

    def __post_init__(self) -> None:
        expected = {
            "epochs": 40,
            "batch_size": 4,
            "train_samples": 1230,
            "validation_samples": 153,
            "train_families": 410,
            "validation_families": 51,
            "max_decode_tokens": 1536,
            "decode_measure_count": 8,
            "retained_checkpoints": 1,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"Stage 7-D2 {name} is frozen to {value}")


STAGE7D2_FROZEN_RUN_CONFIG: Final[Stage7D2RunConfig] = Stage7D2RunConfig()
STAGE7D2_FROZEN_MODEL_CONFIG = STAGE7C_FROZEN_MODEL_CONFIG
STAGE7D2_FROZEN_TRAINER_CONFIG = STAGE7C_FROZEN_TRAINER_CONFIG
STAGE7D2_FROZEN_PREPROCESS_CONFIG = STAGE7C_FROZEN_PREPROCESS_CONFIG


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def stage7d2_run_fingerprint() -> str:
    payload = {
        "run_version": STAGE7D2_RUN_VERSION,
        "dataset": {
            "source_commit": EXPECTED_SOURCE_COMMIT,
            "build_id": EXPECTED_BUILD_ID,
            "config_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "transport_sha256": EXPECTED_TRANSPORT_SHA256,
            "transport_archive": EXPECTED_ARCHIVE_NAME,
            "transport_size": EXPECTED_ARCHIVE_SIZE_BYTES,
            "artifact_binding_sha256": EXPECTED_D1_ARTIFACT_BINDING_SHA256,
        },
        "run_config": asdict(STAGE7D2_FROZEN_RUN_CONFIG),
        "tokenizer_fingerprint": tokenizer_fingerprint(),
        "preprocess_fingerprint": preprocess_config_fingerprint(
            STAGE7D2_FROZEN_PREPROCESS_CONFIG
        ),
        "model_fingerprint": model_config_fingerprint(STAGE7D2_FROZEN_MODEL_CONFIG),
        "trainer_fingerprint": trainer_config_fingerprint(STAGE7D2_FROZEN_TRAINER_CONFIG),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


STAGE7D2_FROZEN_RUN_FINGERPRINT: Final[str] = stage7d2_run_fingerprint()
