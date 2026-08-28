"""Tiny 2D Transformer research prototype for Polyphonic Representation V2.

TR-POLY-08 intentionally implements only a bounded teacher-forced forward path.
It preserves the full patch-row x patch-column visual memory and cross-attends
from the V2 token decoder to that memory. It contains no dataset loader,
optimizer, checkpoint persistence, benchmark claim, or production wiring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Final

import torch
from torch import nn

from .polyphonic_representation import POLYPHONIC_REPRESENTATION_VERSION
from .polyphonic_serialization import (
    PAD_TOKEN_ID,
    POLYPHONIC_TOKENIZER_VERSION,
    VOCABULARY_SIZE,
    tokenizer_fingerprint,
)
from .training_model import (
    TORCH_PINNED_VERSION,
    TrainingRuntimeError,
    assert_finite_tensor,
    assert_model_finite,
    count_trainable_parameters,
    set_deterministic_cpu,
)


POLY_2D_TRANSFORMER_VERSION: Final[str] = "st-omr-poly-2d-transformer-v1"
POLY_2D_TRANSFORMER_PROFILE_VERSION: Final[str] = "st-omr-poly-2d-transformer-profile-v1"
MAX_POLY_2D_PARAMETERS: Final[int] = 5_000_000
_MAX_SEED: Final[int] = 2**63 - 1


class Poly2DTransformerError(TrainingRuntimeError):
    """Raised when the bounded TR-POLY-08 model surface fails closed."""


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class Poly2DTransformerConfig:
    input_height: int = 96
    input_width: int = 512
    patch_height: int = 16
    patch_width: int = 16
    model_dim: int = 64
    encoder_layers: int = 2
    decoder_layers: int = 2
    attention_heads: int = 4
    feedforward_dim: int = 128
    max_target_tokens: int = 2048
    dropout_milli: int = 0

    def __post_init__(self) -> None:
        bounds = {
            "input_height": (self.input_height, 32, 512),
            "input_width": (self.input_width, 128, 2048),
            "patch_height": (self.patch_height, 4, 64),
            "patch_width": (self.patch_width, 4, 64),
            "model_dim": (self.model_dim, 32, 256),
            "encoder_layers": (self.encoder_layers, 1, 4),
            "decoder_layers": (self.decoder_layers, 1, 4),
            "attention_heads": (self.attention_heads, 1, 8),
            "feedforward_dim": (self.feedforward_dim, 64, 1024),
            "max_target_tokens": (self.max_target_tokens, 8, 8192),
            "dropout_milli": (self.dropout_milli, 0, 500),
        }
        for name, (value, lower, upper) in bounds.items():
            if not _plain_int(value) or not lower <= value <= upper:
                raise ValueError(f"{name} is outside the TR-POLY-08 boundary")
        if self.input_height % self.patch_height != 0:
            raise ValueError("input_height must be exactly divisible by patch_height")
        if self.input_width % self.patch_width != 0:
            raise ValueError("input_width must be exactly divisible by patch_width")
        if self.model_dim % self.attention_heads != 0:
            raise ValueError("model_dim must be divisible by attention_heads")
        if self.feedforward_dim < self.model_dim:
            raise ValueError("feedforward_dim must be at least model_dim")

    @property
    def patch_rows(self) -> int:
        return self.input_height // self.patch_height

    @property
    def patch_columns(self) -> int:
        return self.input_width // self.patch_width

    @property
    def visual_token_count(self) -> int:
        return self.patch_rows * self.patch_columns


FROZEN_POLY_2D_CONFIG: Final[Poly2DTransformerConfig] = Poly2DTransformerConfig()


class TinyPoly2DTransformer(nn.Module):
    """Patch-grid encoder + autoregressive cross-attention decoder.

    Unlike the V1 CNN-GRU baseline, the model never averages the vertical
    visual dimension into one global context vector. Every row/column patch is
    retained as an independent encoder-memory token with separable 2D position
    embeddings.
    """

    def __init__(self, config: Poly2DTransformerConfig) -> None:
        super().__init__()
        if not isinstance(config, Poly2DTransformerConfig):
            raise TypeError("config must be Poly2DTransformerConfig")
        self.config = config
        self.patch_embedding = nn.Conv2d(
            1,
            config.model_dim,
            kernel_size=(config.patch_height, config.patch_width),
            stride=(config.patch_height, config.patch_width),
            bias=True,
        )
        self.row_position = nn.Embedding(config.patch_rows, config.model_dim)
        self.column_position = nn.Embedding(config.patch_columns, config.model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout_milli / 1000.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.model_dim),
            enable_nested_tensor=False,
        )

        self.token_embedding = nn.Embedding(
            VOCABULARY_SIZE,
            config.model_dim,
            padding_idx=PAD_TOKEN_ID,
        )
        self.target_position = nn.Embedding(config.max_target_tokens, config.model_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.model_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout_milli / 1000.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.model_dim),
        )
        self.output_projection = nn.Linear(config.model_dim, VOCABULARY_SIZE)

    @property
    def visual_grid_shape(self) -> tuple[int, int]:
        return (self.config.patch_rows, self.config.patch_columns)

    def _validate_images(self, images: object) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            raise Poly2DTransformerError("images must be a torch tensor")
        if images.dtype != torch.float32 or images.ndim != 4:
            raise Poly2DTransformerError("images must be float32 [batch,1,height,width]")
        if tuple(images.shape[1:]) != (1, self.config.input_height, self.config.input_width):
            raise Poly2DTransformerError("image shape differs from the frozen TR-POLY-08 config")
        if images.shape[0] < 1 or images.shape[0] > 32:
            raise Poly2DTransformerError("image batch size is outside the TR-POLY-08 bound")
        assert_finite_tensor("TR-POLY-08 images", images)
        return images

    def _validate_decoder_ids(self, decoder_input_ids: object, batch_size: int) -> torch.Tensor:
        if not isinstance(decoder_input_ids, torch.Tensor):
            raise Poly2DTransformerError("decoder_input_ids must be a torch tensor")
        if decoder_input_ids.dtype != torch.long or decoder_input_ids.ndim != 2:
            raise Poly2DTransformerError("decoder_input_ids must be rank-2 torch.long")
        if decoder_input_ids.shape[0] != batch_size:
            raise Poly2DTransformerError("decoder batch does not match image batch")
        length = decoder_input_ids.shape[1]
        if not 1 <= length <= self.config.max_target_tokens:
            raise Poly2DTransformerError("decoder sequence length is outside the TR-POLY-08 bound")
        if bool((decoder_input_ids < 0).any()) or bool((decoder_input_ids >= VOCABULARY_SIZE).any()):
            raise Poly2DTransformerError("decoder token id is outside the frozen V2 vocabulary")
        return decoder_input_ids

    def _two_dimensional_positions(self, *, device: torch.device) -> torch.Tensor:
        rows = torch.arange(self.config.patch_rows, device=device, dtype=torch.long)
        columns = torch.arange(self.config.patch_columns, device=device, dtype=torch.long)
        row_values = self.row_position(rows)[:, None, :]
        column_values = self.column_position(columns)[None, :, :]
        positions = row_values + column_values
        return positions.reshape(1, self.config.visual_token_count, self.config.model_dim)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        checked = self._validate_images(images)
        patches = self.patch_embedding(checked)
        expected = (
            checked.shape[0],
            self.config.model_dim,
            self.config.patch_rows,
            self.config.patch_columns,
        )
        if tuple(patches.shape) != expected:
            raise Poly2DTransformerError("patch embedding produced an unexpected 2D grid")
        memory = patches.permute(0, 2, 3, 1).reshape(
            checked.shape[0],
            self.config.visual_token_count,
            self.config.model_dim,
        )
        memory = memory + self._two_dimensional_positions(device=checked.device)
        memory = self.encoder(memory)
        assert_finite_tensor("TR-POLY-08 visual memory", memory)
        if memory.shape[1] != self.config.patch_rows * self.config.patch_columns:
            raise Poly2DTransformerError("vertical or horizontal patch structure was collapsed")
        return memory

    @staticmethod
    def causal_mask(length: int, *, device: torch.device | None = None) -> torch.Tensor:
        if not _plain_int(length) or length < 1:
            raise ValueError("causal mask length must be a positive integer")
        return torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, images: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        checked_images = self._validate_images(images)
        checked_ids = self._validate_decoder_ids(decoder_input_ids, checked_images.shape[0])
        memory = self.encode_images(checked_images)

        length = checked_ids.shape[1]
        target_positions = torch.arange(length, device=checked_ids.device, dtype=torch.long)
        target = self.token_embedding(checked_ids) + self.target_position(target_positions)[None, :, :]
        target_padding_mask = checked_ids.eq(PAD_TOKEN_ID)
        decoded = self.decoder(
            target,
            memory,
            tgt_mask=self.causal_mask(length, device=checked_ids.device),
            tgt_key_padding_mask=target_padding_mask,
        )
        logits = self.output_projection(decoded)
        assert_finite_tensor("TR-POLY-08 logits", logits)
        expected = (checked_ids.shape[0], length, VOCABULARY_SIZE)
        if tuple(logits.shape) != expected:
            raise Poly2DTransformerError("decoder logits have unexpected shape")
        return logits


def poly_2d_config_fingerprint(config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG) -> str:
    if not isinstance(config, Poly2DTransformerConfig):
        raise TypeError("config must be Poly2DTransformerConfig")
    payload = {
        "model_version": POLY_2D_TRANSFORMER_VERSION,
        "profile_version": POLY_2D_TRANSFORMER_PROFILE_VERSION,
        "torch_version": TORCH_PINNED_VERSION,
        "representation_version": POLYPHONIC_REPRESENTATION_VERSION,
        "tokenizer_version": POLYPHONIC_TOKENIZER_VERSION,
        "tokenizer_fingerprint": tokenizer_fingerprint(),
        "vocabulary_size": VOCABULARY_SIZE,
        "config": asdict(config),
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def build_tiny_poly_2d_transformer(
    config: Poly2DTransformerConfig = FROZEN_POLY_2D_CONFIG,
    *,
    seed: int = 82_008,
) -> TinyPoly2DTransformer:
    if not _plain_int(seed) or not 0 <= seed <= _MAX_SEED:
        raise ValueError("seed is outside the TR-POLY-08 range")
    set_deterministic_cpu(seed)
    model = TinyPoly2DTransformer(config).cpu()
    parameter_count = count_trainable_parameters(model)
    if not 0 < parameter_count <= MAX_POLY_2D_PARAMETERS:
        raise Poly2DTransformerError("trainable parameter count exceeds the TR-POLY-08 prototype ceiling")
    assert_model_finite(model)
    return model
