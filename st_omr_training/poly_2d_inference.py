"""Bounded free-running inference for the Polyphonic V2 2D Transformer.

TR-POLY-09B1 provides the first image -> free-running V2 prediction surface for
the existing tiny 2D Transformer. It intentionally freezes deterministic greedy
argmax decoding first. It does not open TEST, tune a checkpoint, run a common
benchmark, change the tokenizer, or grant ScoreMosaic/production authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Final

import torch

from .poly_2d_checkpoint import load_and_verify_poly_2d_checkpoint
from .poly_2d_transformer import (
    TinyPoly2DTransformer,
    poly_2d_config_fingerprint,
)
from .polyphonic_representation import (
    POLYPHONIC_REPRESENTATION_VERSION,
    PolyScore,
    PolyphonicRepresentationError,
)
from .polyphonic_serialization import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    POLYPHONIC_TOKENIZER_VERSION,
    PolyphonicSerializationError,
    VOCABULARY_SIZE,
    detokenize_polyphonic_ids,
    tokenizer_fingerprint,
)
from .training_model import TORCH_PINNED_VERSION, model_state_sha256


POLY_2D_GREEDY_INFERENCE_VERSION: Final[str] = "st-omr-poly-2d-greedy-inference-v1"
POLY_2D_GREEDY_POLICY: Final[str] = "argmax-lowest-token-id-tie-v1"
POLY_2D_CONTROL_TOKEN_POLICY: Final[str] = "generated-pad-or-bos-fails-closed-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA40 = re.compile(r"^[0-9a-f]{40}$")


class Poly2DInferenceError(RuntimeError):
    """Raised when the bounded inference contract itself cannot be satisfied."""


class Poly2DInferenceStatus(str, Enum):
    EOS_VALID = "eos_valid"
    EOS_SEMANTIC_INVALID = "eos_semantic_invalid"
    INVALID_CONTROL_TOKEN = "invalid_control_token"
    MAX_STEPS = "max_steps"


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Poly2DInferenceError("inference evidence is not canonical-JSON serializable") from exc


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _optional_sha256(value: str | None, name: str) -> None:
    if value is not None and (not isinstance(value, str) or _SHA256.fullmatch(value) is None):
        raise Poly2DInferenceError(f"{name} must be lowercase SHA-256 hex when present")


@dataclass(frozen=True, slots=True)
class Poly2DInferenceIdentity:
    """Exact model/runtime identity attached to one free-running prediction."""

    model_state_sha256: str
    model_profile_sha256: str
    inference_profile_sha256: str
    tokenizer_fingerprint_sha256: str
    max_decode_steps: int
    representation_version: str = POLYPHONIC_REPRESENTATION_VERSION
    tokenizer_version: str = POLYPHONIC_TOKENIZER_VERSION
    torch_version: str = TORCH_PINNED_VERSION
    inference_version: str = POLY_2D_GREEDY_INFERENCE_VERSION
    checkpoint_sha256: str | None = None
    checkpoint_metadata_file_sha256: str | None = None
    checkpoint_receipt_sha256: str | None = None
    checkpoint_metadata_fingerprint_sha256: str | None = None
    dataset_manifest_sha256: str | None = None
    preprocess_fingerprint_sha256: str | None = None
    trainer_profile_sha256: str | None = None
    provenance_sha256: str | None = None
    registry_record_fingerprint_sha256: str | None = None
    repository_sha: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "model_state_sha256",
            "model_profile_sha256",
            "inference_profile_sha256",
            "tokenizer_fingerprint_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise Poly2DInferenceError(f"{name} must be lowercase SHA-256 hex")
        for name in (
            "checkpoint_sha256",
            "checkpoint_metadata_file_sha256",
            "checkpoint_receipt_sha256",
            "checkpoint_metadata_fingerprint_sha256",
            "dataset_manifest_sha256",
            "preprocess_fingerprint_sha256",
            "trainer_profile_sha256",
            "provenance_sha256",
            "registry_record_fingerprint_sha256",
        ):
            _optional_sha256(getattr(self, name), name)
        if self.repository_sha is not None and (
            not isinstance(self.repository_sha, str) or _GIT_SHA40.fullmatch(self.repository_sha) is None
        ):
            raise Poly2DInferenceError("repository_sha must be lowercase git SHA-40 hex when present")
        if not _plain_int(self.max_decode_steps) or self.max_decode_steps < 1:
            raise Poly2DInferenceError("max_decode_steps must be a positive plain integer")
        if self.representation_version != POLYPHONIC_REPRESENTATION_VERSION:
            raise Poly2DInferenceError("inference representation version mismatch")
        if self.tokenizer_version != POLYPHONIC_TOKENIZER_VERSION:
            raise Poly2DInferenceError("inference tokenizer version mismatch")
        if self.torch_version != TORCH_PINNED_VERSION:
            raise Poly2DInferenceError("inference torch version mismatch")
        if self.inference_version != POLY_2D_GREEDY_INFERENCE_VERSION:
            raise Poly2DInferenceError("inference version mismatch")
        if self.tokenizer_fingerprint_sha256 != tokenizer_fingerprint():
            raise Poly2DInferenceError("inference tokenizer fingerprint mismatch")

    @property
    def checkpoint_bound(self) -> bool:
        return self.checkpoint_sha256 is not None

    def fingerprint(self) -> str:
        return sha256(_canonical_json_bytes(asdict(self))).hexdigest()


@dataclass(frozen=True, slots=True)
class Poly2DInferenceResult:
    token_ids: tuple[int, ...]
    status: Poly2DInferenceStatus
    identity: Poly2DInferenceIdentity
    semantic_valid: bool
    prediction: PolyScore | None
    prediction_sha256: str | None
    error_code: str | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.token_ids, tuple)
            or not self.token_ids
            or any(not _plain_int(item) or not 0 <= item < VOCABULARY_SIZE for item in self.token_ids)
        ):
            raise Poly2DInferenceError("token_ids must be a non-empty immutable V2 id tuple")
        if self.token_ids[0] != BOS_TOKEN_ID:
            raise Poly2DInferenceError("free-running inference must start from BOS")
        if not isinstance(self.status, Poly2DInferenceStatus):
            raise Poly2DInferenceError("status must be Poly2DInferenceStatus")
        if not isinstance(self.identity, Poly2DInferenceIdentity):
            raise Poly2DInferenceError("identity must be Poly2DInferenceIdentity")
        if not isinstance(self.semantic_valid, bool):
            raise Poly2DInferenceError("semantic_valid must be bool")

        if self.semantic_valid:
            if self.status is not Poly2DInferenceStatus.EOS_VALID:
                raise Poly2DInferenceError("semantic-valid result must terminate as EOS_VALID")
            if self.prediction is None or self.prediction_sha256 is None or self.error_code is not None:
                raise Poly2DInferenceError("semantic-valid result requires prediction/hash and no error")
            if self.token_ids[-1] != EOS_TOKEN_ID:
                raise Poly2DInferenceError("semantic-valid result must end in EOS")
            if self.prediction.canonical_sha256() != self.prediction_sha256:
                raise Poly2DInferenceError("prediction SHA differs from canonical V2 score")
        else:
            if self.prediction is not None or self.prediction_sha256 is not None or self.error_code is None:
                raise Poly2DInferenceError("invalid/abstain result must contain only an explicit error code")
            if self.status is Poly2DInferenceStatus.EOS_SEMANTIC_INVALID and self.token_ids[-1] != EOS_TOKEN_ID:
                raise Poly2DInferenceError("EOS semantic-invalid result must end in EOS")
            if self.status is Poly2DInferenceStatus.INVALID_CONTROL_TOKEN and self.token_ids[-1] not in (
                PAD_TOKEN_ID,
                BOS_TOKEN_ID,
            ):
                raise Poly2DInferenceError("invalid-control result must record generated PAD or BOS")

    @property
    def generated_steps(self) -> int:
        return len(self.token_ids) - 1

    def evidence_fingerprint(self) -> str:
        payload = {
            "status": self.status.value,
            "token_ids": list(self.token_ids),
            "identity_fingerprint_sha256": self.identity.fingerprint(),
            "semantic_valid": self.semantic_valid,
            "prediction_sha256": self.prediction_sha256,
            "error_code": self.error_code,
        }
        return sha256(_canonical_json_bytes(payload)).hexdigest()


def poly_2d_inference_profile_fingerprint(
    model: TinyPoly2DTransformer,
    *,
    max_decode_steps: int,
) -> str:
    if not isinstance(model, TinyPoly2DTransformer):
        raise TypeError("model must be TinyPoly2DTransformer")
    if not _plain_int(max_decode_steps) or not 1 <= max_decode_steps <= model.config.max_target_tokens - 1:
        raise Poly2DInferenceError("max_decode_steps is outside the model target boundary")
    payload = {
        "inference_version": POLY_2D_GREEDY_INFERENCE_VERSION,
        "greedy_policy": POLY_2D_GREEDY_POLICY,
        "control_token_policy": POLY_2D_CONTROL_TOKEN_POLICY,
        "max_decode_steps": max_decode_steps,
        "model_profile_sha256": poly_2d_config_fingerprint(model.config),
        "representation_version": POLYPHONIC_REPRESENTATION_VERSION,
        "tokenizer_version": POLYPHONIC_TOKENIZER_VERSION,
        "tokenizer_fingerprint_sha256": tokenizer_fingerprint(),
        "torch_version": TORCH_PINNED_VERSION,
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _assert_cpu_model_and_image(model: TinyPoly2DTransformer, image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor):
        raise Poly2DInferenceError("image must be a torch tensor")
    if image.device.type != "cpu":
        raise Poly2DInferenceError("TR-POLY-09B1 inference is frozen to CPU")
    if image.ndim != 4 or image.shape[0] != 1:
        raise Poly2DInferenceError("TR-POLY-09B1 inference accepts exactly one image per call")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise Poly2DInferenceError("TR-POLY-09B1 model parameters must be on CPU")


def run_poly_2d_greedy_inference(
    model: TinyPoly2DTransformer,
    image: torch.Tensor,
    *,
    max_decode_steps: int | None = None,
) -> Poly2DInferenceResult:
    """Run bounded BOS-only greedy inference without any gold target prefix."""

    if not isinstance(model, TinyPoly2DTransformer):
        raise TypeError("model must be TinyPoly2DTransformer")
    _assert_cpu_model_and_image(model, image)
    if max_decode_steps is None:
        max_decode_steps = model.config.max_target_tokens - 1
    if not _plain_int(max_decode_steps) or not 1 <= max_decode_steps <= model.config.max_target_tokens - 1:
        raise Poly2DInferenceError("max_decode_steps is outside the model target boundary")

    state_before = model_state_sha256(model)
    previous_training = model.training
    token_ids: list[int] = [BOS_TOKEN_ID]
    status: Poly2DInferenceStatus | None = None
    prediction: PolyScore | None = None
    prediction_sha256: str | None = None
    error_code: str | None = None

    try:
        model.eval()
        with torch.inference_mode():
            memory = model.encode_images(image)
            if memory.shape[0] != 1:
                raise Poly2DInferenceError("visual memory batch changed during inference")
            for _step in range(max_decode_steps):
                prefix = torch.tensor([token_ids], dtype=torch.long, device=memory.device)
                logits = model.decode_from_memory(memory, prefix)
                expected = (1, len(token_ids), VOCABULARY_SIZE)
                if tuple(logits.shape) != expected:
                    raise Poly2DInferenceError("decoder returned an invalid inference logit shape")
                last_logits = logits[0, -1]
                if not bool(torch.isfinite(last_logits).all()):
                    raise Poly2DInferenceError("decoder returned non-finite next-token logits")
                next_token = int(torch.argmax(last_logits).item())
                token_ids.append(next_token)

                if next_token == PAD_TOKEN_ID:
                    status = Poly2DInferenceStatus.INVALID_CONTROL_TOKEN
                    error_code = "generated_pad"
                    break
                if next_token == BOS_TOKEN_ID:
                    status = Poly2DInferenceStatus.INVALID_CONTROL_TOKEN
                    error_code = "generated_bos"
                    break
                if next_token == EOS_TOKEN_ID:
                    try:
                        prediction = detokenize_polyphonic_ids(tuple(token_ids))
                    except (PolyphonicSerializationError, PolyphonicRepresentationError):
                        status = Poly2DInferenceStatus.EOS_SEMANTIC_INVALID
                        error_code = "strict_v2_parse_failed"
                    else:
                        status = Poly2DInferenceStatus.EOS_VALID
                        prediction_sha256 = prediction.canonical_sha256()
                    break

            if status is None:
                status = Poly2DInferenceStatus.MAX_STEPS
                error_code = "decode_limit_exhausted"
    finally:
        model.train(previous_training)

    state_after = model_state_sha256(model)
    if state_after != state_before:
        raise Poly2DInferenceError("free-running inference mutated model state")
    if status is None:
        raise Poly2DInferenceError("inference terminated without a status")

    identity = Poly2DInferenceIdentity(
        model_state_sha256=state_before,
        model_profile_sha256=poly_2d_config_fingerprint(model.config),
        inference_profile_sha256=poly_2d_inference_profile_fingerprint(
            model,
            max_decode_steps=max_decode_steps,
        ),
        tokenizer_fingerprint_sha256=tokenizer_fingerprint(),
        max_decode_steps=max_decode_steps,
    )
    semantic_valid = status is Poly2DInferenceStatus.EOS_VALID
    return Poly2DInferenceResult(
        token_ids=tuple(token_ids),
        status=status,
        identity=identity,
        semantic_valid=semantic_valid,
        prediction=prediction if semantic_valid else None,
        prediction_sha256=prediction_sha256 if semantic_valid else None,
        error_code=None if semantic_valid else error_code,
    )


def run_verified_poly_2d_checkpoint_inference(
    checkpoint_directory: Path,
    image: torch.Tensor,
    *,
    max_decode_steps: int | None = None,
) -> Poly2DInferenceResult:
    """Verify an exact TR-POLY-08B artifact, then bind inference to that artifact."""

    if not isinstance(checkpoint_directory, Path):
        raise TypeError("checkpoint_directory must be pathlib.Path")
    loaded = load_and_verify_poly_2d_checkpoint(checkpoint_directory)
    result = run_poly_2d_greedy_inference(
        loaded.model,
        image,
        max_decode_steps=max_decode_steps,
    )
    metadata = loaded.metadata
    if result.identity.model_state_sha256 != metadata.final_state_sha256:
        raise Poly2DInferenceError("verified checkpoint state differs from inference model state")
    if result.identity.model_profile_sha256 != metadata.model_profile_sha256:
        raise Poly2DInferenceError("verified checkpoint model profile differs from inference profile")

    bound_identity = replace(
        result.identity,
        checkpoint_sha256=loaded.checkpoint_sha256,
        checkpoint_metadata_file_sha256=loaded.metadata_sha256,
        checkpoint_receipt_sha256=loaded.receipt_sha256,
        checkpoint_metadata_fingerprint_sha256=metadata.fingerprint(),
        dataset_manifest_sha256=metadata.dataset_manifest_sha256,
        preprocess_fingerprint_sha256=metadata.preprocess_fingerprint_sha256,
        trainer_profile_sha256=metadata.trainer_profile_sha256,
        provenance_sha256=metadata.provenance_sha256,
        registry_record_fingerprint_sha256=metadata.registry_record_fingerprint_sha256,
        repository_sha=metadata.repository_sha,
    )
    return replace(result, identity=bound_identity)
