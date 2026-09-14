from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from st_omr_training.poly_2d_inference import (
    POLY_2D_GREEDY_INFERENCE_VERSION,
    Poly2DInferenceError,
    Poly2DInferenceStatus,
    poly_2d_inference_profile_fingerprint,
    run_poly_2d_greedy_inference,
    run_verified_poly_2d_checkpoint_inference,
)
from st_omr_training.poly_2d_transformer import (
    FROZEN_POLY_2D_CONFIG,
    TinyPoly2DTransformer,
    build_tiny_poly_2d_transformer,
    poly_2d_config_fingerprint,
)
from st_omr_training.polyphonic_representation import (
    Barline,
    BarlineLocation,
    BarlineStyle,
    ClefAssignment,
    EventKind,
    ExactRational,
    KeySignature,
    NoteAtom,
    NoteType,
    PitchSpelling,
    PolyEvent,
    PolyMeasure,
    PolyPart,
    PolyScore,
    TimeSignature,
)
from st_omr_training.polyphonic_serialization import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    TOKEN_TO_ID,
    VOCABULARY_SIZE,
    tokenize_polyphonic_score,
)
from st_omr_training.training_model import model_state_sha256


def _score() -> PolyScore:
    note = PolyEvent(
        event_id="e1",
        kind=EventKind.NOTE,
        onset=ExactRational(0, 1),
        duration=ExactRational(1, 4),
        voice=1,
        staff=1,
        note_type=NoteType.QUARTER,
        noteheads=(
            NoteAtom(
                atom_id="a1",
                pitch=PitchSpelling(step="C", alter=0, octave=4),
            ),
        ),
    )
    measure = PolyMeasure(
        measure_index=1,
        source_number="1",
        time_signature=TimeSignature((4,), 4),
        key_signature=KeySignature(0, "major"),
        clefs=(ClefAssignment(staff=1, sign="G", line=2),),
        events=(note,),
        barlines=(Barline(BarlineLocation.RIGHT, BarlineStyle.REGULAR),),
    )
    return PolyScore((PolyPart("P1", 1, (measure,)),))


class _ScriptedPoly2DModel(TinyPoly2DTransformer):
    """Uses real model state but deterministic scripted next-token logits."""

    def __init__(self, scripted_ids: tuple[int, ...], *, fallback_id: int | None = None) -> None:
        super().__init__(FROZEN_POLY_2D_CONFIG)
        self._scripted_ids = scripted_ids
        self._fallback_id = fallback_id if fallback_id is not None else EOS_TOKEN_ID

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        checked = self._validate_images(images)
        return torch.zeros(
            (
                checked.shape[0],
                self.config.visual_token_count,
                self.config.model_dim,
            ),
            dtype=torch.float32,
            device=checked.device,
        )

    def decode_from_memory(
        self,
        memory: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        checked_memory = self._validate_memory(memory)
        checked_ids = self._validate_decoder_ids(decoder_input_ids, checked_memory.shape[0])
        length = checked_ids.shape[1]
        next_token = (
            self._scripted_ids[length]
            if length < len(self._scripted_ids)
            else self._fallback_id
        )
        logits = torch.full(
            (checked_ids.shape[0], length, VOCABULARY_SIZE),
            -100.0,
            dtype=torch.float32,
        )
        logits[:, -1, next_token] = 100.0
        return logits


class Poly2DInferenceTests(unittest.TestCase):
    def _image(self) -> torch.Tensor:
        return torch.zeros(
            (1, 1, FROZEN_POLY_2D_CONFIG.input_height, FROZEN_POLY_2D_CONFIG.input_width),
            dtype=torch.float32,
        )

    def test_decode_from_precomputed_memory_matches_forward(self) -> None:
        model = build_tiny_poly_2d_transformer(seed=9_001).eval()
        image = self._image()
        ids = torch.tensor([[BOS_TOKEN_ID, EOS_TOKEN_ID]], dtype=torch.long)
        with torch.inference_mode():
            memory = model.encode_images(image)
            direct = model.decode_from_memory(memory, ids)
            through_forward = model(image, ids)
        self.assertTrue(torch.equal(direct, through_forward))

    def test_decode_from_memory_rejects_wrong_grid(self) -> None:
        model = build_tiny_poly_2d_transformer(seed=9_002).eval()
        ids = torch.tensor([[BOS_TOKEN_ID]], dtype=torch.long)
        bad_memory = torch.zeros((1, 1, FROZEN_POLY_2D_CONFIG.model_dim), dtype=torch.float32)
        with self.assertRaisesRegex(Exception, "visual memory shape"):
            model.decode_from_memory(bad_memory, ids)

    def test_scripted_free_running_sequence_reconstructs_exact_v2_score(self) -> None:
        score = _score()
        target = tokenize_polyphonic_score(score)
        model = _ScriptedPoly2DModel(target.token_ids).eval()
        before = model_state_sha256(model)

        result = run_poly_2d_greedy_inference(
            model,
            self._image(),
            max_decode_steps=len(target.token_ids) - 1,
        )

        self.assertEqual(result.status, Poly2DInferenceStatus.EOS_VALID)
        self.assertTrue(result.semantic_valid)
        self.assertEqual(result.token_ids, target.token_ids)
        self.assertEqual(result.prediction, score)
        self.assertEqual(result.prediction_sha256, score.canonical_sha256())
        self.assertEqual(result.error_code, None)
        self.assertEqual(model_state_sha256(model), before)
        self.assertEqual(result.identity.model_profile_sha256, poly_2d_config_fingerprint(model.config))
        self.assertEqual(len(result.identity.fingerprint()), 64)
        self.assertEqual(len(result.evidence_fingerprint()), 64)

    def test_generated_pad_or_bos_fails_closed_without_silent_masking(self) -> None:
        for invalid_id, code in ((PAD_TOKEN_ID, "generated_pad"), (BOS_TOKEN_ID, "generated_bos")):
            with self.subTest(invalid_id=invalid_id):
                model = _ScriptedPoly2DModel((BOS_TOKEN_ID, invalid_id)).eval()
                result = run_poly_2d_greedy_inference(model, self._image(), max_decode_steps=2)
                self.assertEqual(result.status, Poly2DInferenceStatus.INVALID_CONTROL_TOKEN)
                self.assertFalse(result.semantic_valid)
                self.assertEqual(result.token_ids[-1], invalid_id)
                self.assertEqual(result.error_code, code)

    def test_eos_with_malformed_payload_is_explicit_semantic_failure(self) -> None:
        model = _ScriptedPoly2DModel((BOS_TOKEN_ID, EOS_TOKEN_ID)).eval()
        result = run_poly_2d_greedy_inference(model, self._image(), max_decode_steps=1)
        self.assertEqual(result.status, Poly2DInferenceStatus.EOS_SEMANTIC_INVALID)
        self.assertFalse(result.semantic_valid)
        self.assertEqual(result.error_code, "strict_v2_parse_failed")
        self.assertIsNone(result.prediction)

    def test_decode_limit_abstains_without_fabricating_eos(self) -> None:
        object_start = TOKEN_TO_ID["OBJ_START"]
        model = _ScriptedPoly2DModel(
            (BOS_TOKEN_ID, object_start),
            fallback_id=object_start,
        ).eval()
        result = run_poly_2d_greedy_inference(model, self._image(), max_decode_steps=3)
        self.assertEqual(result.status, Poly2DInferenceStatus.MAX_STEPS)
        self.assertEqual(result.generated_steps, 3)
        self.assertNotEqual(result.token_ids[-1], EOS_TOKEN_ID)
        self.assertEqual(result.error_code, "decode_limit_exhausted")

    def test_same_state_input_and_profile_are_deterministic(self) -> None:
        target = tokenize_polyphonic_score(_score())
        model = _ScriptedPoly2DModel(target.token_ids).eval()
        kwargs = {"max_decode_steps": len(target.token_ids) - 1}
        first = run_poly_2d_greedy_inference(model, self._image(), **kwargs)
        second = run_poly_2d_greedy_inference(model, self._image(), **kwargs)
        self.assertEqual(first.token_ids, second.token_ids)
        self.assertEqual(first.identity.fingerprint(), second.identity.fingerprint())
        self.assertEqual(first.evidence_fingerprint(), second.evidence_fingerprint())

    def test_profile_fingerprint_binds_decode_limit(self) -> None:
        model = build_tiny_poly_2d_transformer(seed=9_003)
        short = poly_2d_inference_profile_fingerprint(model, max_decode_steps=4)
        longer = poly_2d_inference_profile_fingerprint(model, max_decode_steps=5)
        self.assertEqual(len(short), 64)
        self.assertNotEqual(short, longer)
        self.assertEqual(POLY_2D_GREEDY_INFERENCE_VERSION, "st-omr-poly-2d-greedy-inference-v1")

    def test_batch_or_invalid_decode_limit_fails_closed(self) -> None:
        model = build_tiny_poly_2d_transformer(seed=9_004)
        batch = torch.zeros(
            (2, 1, FROZEN_POLY_2D_CONFIG.input_height, FROZEN_POLY_2D_CONFIG.input_width),
            dtype=torch.float32,
        )
        with self.assertRaisesRegex(Poly2DInferenceError, "exactly one image"):
            run_poly_2d_greedy_inference(model, batch, max_decode_steps=1)
        with self.assertRaisesRegex(Poly2DInferenceError, "outside"):
            run_poly_2d_greedy_inference(model, self._image(), max_decode_steps=0)

    def test_verified_checkpoint_wrapper_binds_checkpoint_and_dataset_identity(self) -> None:
        target = tokenize_polyphonic_score(_score())
        model = _ScriptedPoly2DModel(target.token_ids).eval()
        state_sha = model_state_sha256(model)
        model_profile = poly_2d_config_fingerprint(model.config)
        metadata = SimpleNamespace(
            final_state_sha256=state_sha,
            model_profile_sha256=model_profile,
            dataset_manifest_sha256="1" * 64,
            preprocess_fingerprint_sha256="2" * 64,
            trainer_profile_sha256="3" * 64,
            provenance_sha256="4" * 64,
            registry_record_fingerprint_sha256="5" * 64,
            repository_sha="6" * 40,
            fingerprint=lambda: "7" * 64,
        )
        loaded = SimpleNamespace(
            model=model,
            metadata=metadata,
            checkpoint_sha256="8" * 64,
            metadata_sha256="9" * 64,
            receipt_sha256="a" * 64,
        )
        with patch(
            "st_omr_training.poly_2d_inference.load_and_verify_poly_2d_checkpoint",
            return_value=loaded,
        ):
            result = run_verified_poly_2d_checkpoint_inference(
                Path("/verified/checkpoint"),
                self._image(),
                max_decode_steps=len(target.token_ids) - 1,
            )
        self.assertTrue(result.semantic_valid)
        self.assertTrue(result.identity.checkpoint_bound)
        self.assertEqual(result.identity.checkpoint_sha256, "8" * 64)
        self.assertEqual(result.identity.dataset_manifest_sha256, "1" * 64)
        self.assertEqual(result.identity.repository_sha, "6" * 40)


if __name__ == "__main__":
    unittest.main()
