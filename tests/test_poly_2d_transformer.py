from __future__ import annotations

import unittest

import torch

from st_omr_training.poly_2d_transformer import (
    FROZEN_POLY_2D_CONFIG,
    MAX_POLY_2D_PARAMETERS,
    POLY_2D_TRANSFORMER_PROFILE_VERSION,
    POLY_2D_TRANSFORMER_VERSION,
    Poly2DTransformerConfig,
    Poly2DTransformerError,
    build_tiny_poly_2d_transformer,
    poly_2d_config_fingerprint,
)
from st_omr_training.polyphonic_representation import POLYPHONIC_REPRESENTATION_VERSION
from st_omr_training.polyphonic_serialization import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    POLYPHONIC_TOKENIZER_VERSION,
    VOCABULARY_SIZE,
)
from st_omr_training.training_model import count_trainable_parameters, model_state_sha256


class Poly2DTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = build_tiny_poly_2d_transformer(seed=82_008).eval()

    def _images(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(
            (batch, 1, FROZEN_POLY_2D_CONFIG.input_height, FROZEN_POLY_2D_CONFIG.input_width),
            dtype=torch.float32,
        )

    def test_versions_and_frozen_target_contract_are_explicit(self) -> None:
        self.assertEqual(POLY_2D_TRANSFORMER_VERSION, "st-omr-poly-2d-transformer-v1")
        self.assertEqual(POLY_2D_TRANSFORMER_PROFILE_VERSION, "st-omr-poly-2d-transformer-profile-v1")
        self.assertEqual(POLYPHONIC_REPRESENTATION_VERSION, "st-omr-polyphonic-representation-v2")
        self.assertEqual(POLYPHONIC_TOKENIZER_VERSION, "st-omr-polyphonic-tokenizer-v1")
        self.assertGreater(VOCABULARY_SIZE, 0)

    def test_frozen_config_preserves_both_patch_axes(self) -> None:
        config = FROZEN_POLY_2D_CONFIG
        self.assertGreater(config.patch_rows, 1)
        self.assertGreater(config.patch_columns, 1)
        self.assertEqual(config.visual_token_count, config.patch_rows * config.patch_columns)
        self.assertEqual(self.model.visual_grid_shape, (config.patch_rows, config.patch_columns))

    def test_encoder_retains_every_2d_patch_as_memory_token(self) -> None:
        memory = self.model.encode_images(self._images())
        self.assertEqual(
            tuple(memory.shape),
            (
                1,
                FROZEN_POLY_2D_CONFIG.patch_rows * FROZEN_POLY_2D_CONFIG.patch_columns,
                FROZEN_POLY_2D_CONFIG.model_dim,
            ),
        )
        self.assertTrue(bool(torch.isfinite(memory).all()))

    def test_row_and_column_positions_are_independent_not_vertical_average(self) -> None:
        row_weights = self.model.row_position.weight.detach()
        column_weights = self.model.column_position.weight.detach()
        self.assertGreater(row_weights.shape[0], 1)
        self.assertGreater(column_weights.shape[0], 1)
        self.assertFalse(torch.equal(row_weights[0], row_weights[1]))
        self.assertFalse(torch.equal(column_weights[0], column_weights[1]))

    def test_local_ink_moved_between_rows_changes_full_2d_memory(self) -> None:
        config = FROZEN_POLY_2D_CONFIG
        upper = self._images()
        lower = self._images()
        upper[:, :, 0 : config.patch_height, 0 : config.patch_width] = 1.0
        lower[:, :, config.patch_height : 2 * config.patch_height, 0 : config.patch_width] = 1.0
        with torch.no_grad():
            upper_memory = self.model.encode_images(upper)
            lower_memory = self.model.encode_images(lower)
        self.assertFalse(torch.equal(upper_memory, lower_memory))

    def test_decoder_cross_attends_and_emits_frozen_v2_vocabulary(self) -> None:
        ids = torch.tensor([[BOS_TOKEN_ID, EOS_TOKEN_ID]], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(self._images(), ids)
        self.assertEqual(tuple(logits.shape), (1, 2, VOCABULARY_SIZE))
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_causal_mask_prevents_future_attention(self) -> None:
        mask = self.model.causal_mask(4)
        expected = torch.tensor(
            [
                [False, True, True, True],
                [False, False, True, True],
                [False, False, False, True],
                [False, False, False, False],
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_parameter_budget_and_initialization_are_bounded_and_deterministic(self) -> None:
        first = build_tiny_poly_2d_transformer(seed=1234)
        second = build_tiny_poly_2d_transformer(seed=1234)
        count = count_trainable_parameters(first)
        self.assertGreater(count, 0)
        self.assertLessEqual(count, MAX_POLY_2D_PARAMETERS)
        self.assertEqual(model_state_sha256(first), model_state_sha256(second))

    def test_config_fingerprint_is_deterministic_and_target_bound(self) -> None:
        self.assertEqual(poly_2d_config_fingerprint(), poly_2d_config_fingerprint())
        self.assertEqual(len(poly_2d_config_fingerprint()), 64)
        changed = Poly2DTransformerConfig(model_dim=128, attention_heads=4, feedforward_dim=256)
        self.assertNotEqual(poly_2d_config_fingerprint(), poly_2d_config_fingerprint(changed))

    def test_invalid_config_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            Poly2DTransformerConfig(input_height=95)
        with self.assertRaisesRegex(ValueError, "model_dim must be divisible"):
            Poly2DTransformerConfig(model_dim=66, attention_heads=4)
        with self.assertRaisesRegex(ValueError, "outside"):
            Poly2DTransformerConfig(max_target_tokens=1)

    def test_invalid_image_surface_fails_closed(self) -> None:
        ids = torch.tensor([[BOS_TOKEN_ID]], dtype=torch.long)
        with self.assertRaisesRegex(Poly2DTransformerError, "float32"):
            self.model(self._images().to(torch.float64), ids)
        with self.assertRaisesRegex(Poly2DTransformerError, "shape differs"):
            self.model(torch.zeros((1, 1, 64, 512), dtype=torch.float32), ids)
        bad = self._images()
        bad[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(Exception, "NaN or Infinity"):
            self.model(bad, ids)

    def test_invalid_decoder_surface_fails_closed(self) -> None:
        image = self._images()
        with self.assertRaisesRegex(Poly2DTransformerError, "rank-2 torch.long"):
            self.model(image, torch.tensor([BOS_TOKEN_ID], dtype=torch.long))
        with self.assertRaisesRegex(Poly2DTransformerError, "outside the frozen V2 vocabulary"):
            self.model(image, torch.tensor([[VOCABULARY_SIZE]], dtype=torch.long))
        too_long = torch.full(
            (1, FROZEN_POLY_2D_CONFIG.max_target_tokens + 1),
            BOS_TOKEN_ID,
            dtype=torch.long,
        )
        with self.assertRaisesRegex(Poly2DTransformerError, "sequence length"):
            self.model(image, too_long)

    def test_padding_id_is_frozen_v2_padding_not_reinterpreted(self) -> None:
        self.assertEqual(PAD_TOKEN_ID, 0)
        ids = torch.tensor([[BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID]], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(self._images(), ids)
        self.assertEqual(tuple(logits.shape), (1, 3, VOCABULARY_SIZE))


if __name__ == "__main__":
    unittest.main()
