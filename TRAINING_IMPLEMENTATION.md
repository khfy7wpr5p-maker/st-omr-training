# ST-OMR Stage 7-B Training Implementation Profile

Status: **closed — main CI verified**. This document records the concrete implementation choices permitted by the frozen `TRAINING_CONTRACT.md`. It does **not** authorize or execute the Stage 7-C baseline training run.

Stage 7-B merged through PR #21 at exact `main` commit `d02dce4ee17dfccf6f05519ab0970fdc188d0147`. Post-merge GitHub Actions run #31 (`31679810478`) succeeded on that exact commit with **336/336 tests**, exact pinned runtime verification, `pip check`, the missing-`EOS` regression, deterministic CPU smoke evidence, and `compileall`.

## Framework/runtime selection

Stage 7-B selects **PyTorch 2.13.0 CPU** as the first training runtime.

Pinned package:

```text
torch==2.13.0+cpu
```

The pin is isolated in `requirements-training.txt` and CI installs it from the official PyTorch CPU wheel index. The existing symbolic/rendering dependencies remain in `requirements.txt`.

Selection evidence reviewed on 2026-08-13:

- official PyTorch release `v2.13.0` is the stable release line reviewed for this package;
- the official CPU wheel index publishes a CPython 3.13 / manylinux x86-64 `2.13.0+cpu` wheel;
- the repository CI runtime remains Ubuntu 24.04 / CPython 3.13.

No CUDA, ROCm, pretrained model, torchvision model, external OCR/OMR engine, network label source, or model download is part of Stage 7-B.

## Frozen semantic tokenizer

Implementation: `st_omr_training/training_tokens.py`

The tokenizer uses the exact finite 35-token Stage 7-A vocabulary. `PAD` is batching-only and receives id 0. MusicXML is first accepted by the existing Stage 2-C/2-D semantic boundary, then converted to tokens, independently detokenized back to a `SemanticScoreProjection`, and compared field-for-field with the source projection.

Tokenization preserves:

- measure boundaries and effective 2/4, 3/4, or 4/4 meter;
- event order and note/rest/chord type;
- whole/half/quarter/eighth duration class;
- chord size and member order;
- pitch step, alter, octave, and visible accidental intent.

A mismatch, malformed sequence, PAD inside a semantic target, under/overfilled measure, invalid pitch/accidental relation, unsupported token, trailing token after `EOS`, or end-of-file without consuming a real `EOS` is a hard failure.

The missing-`EOS` fail-closed rule was added after automated review identified that a sequence ending immediately after a valid `MEASURE_END` could otherwise be accepted. The regression `test_missing_eos_is_rejected` is part of the final Stage 7-B evidence.

## Trusted data adapter

Implementation: `st_omr_training/training_data.py`

Stage 7-B accepts only a validated Stage 6 `SyntheticDatasetBuild` together with its exact persisted no-overwrite dataset directory. It re-checks:

- Stage 5 manifest validity;
- canonical `manifest.json` bytes and manifest SHA-256;
- exact `build.json` bytes;
- `manifest.sha256` record;
- selected MusicXML target hashes;
- selected PNG image hashes;
- target semantic token round-trip;
- actual PNG format, grayscale `L` mode, width, height, and Stage 5 pixel ceiling.

`DatasetSplit.TEST` is rejected before dataset-root or build access. Only `train` and `validation` can produce Stage 7-B sample references or batches.

## Deterministic preprocessing

Frozen preprocessing profile: `st-omr-fit-pad-grayscale-v1`.

Default tensor surface:

```text
1 x 64 x 512 float32
```

Policy:

1. exact PNG only;
2. exact grayscale `L` only;
3. verify manifest dimensions;
4. fit the complete page inside the fixed canvas with aspect ratio preserved;
5. never crop;
6. never upscale;
7. when downscaling is required, use pinned Pillow bilinear resampling;
8. center-pad unused canvas area with white;
9. normalize bytes deterministically to `[0, 1]`;
10. reject NaN/Infinity.

The policy is fingerprinted and adds no hidden/random Stage 7 augmentation.

## Baseline model

Implementation: `st_omr_training/training_model.py`

Model version: `st-omr-cnn-gru-baseline-v1`.

The one allowed Stage 7-B model is a from-scratch CPU baseline:

```text
grayscale 1x64x512 page
        ↓
2-layer bounded CNN visual encoder
        ↓
fixed-width adaptive visual sequence pool
        ↓
visual context projection
        ↓
frozen-vocabulary token embedding
        ↓
context-conditioned GRU decoder
        ↓
logits over 35 frozen tokens
```

Default model configuration:

- CNN channels: 8 then 16;
- encoder steps: 32;
- token embedding: 32;
- GRU hidden dimension: 64;
- parameter count is checked at construction and must remain at or below the Stage 7-A ceiling of 25,000,000.

No ensemble, attention service, pretrained weights, external feature extractor, OCR/OMR teacher, LLM, or network inference is used.

## Smoke trainer

Trainer version: `st-omr-smoke-trainer-v1`.

Frozen Stage 7-B defaults:

- objective: token cross-entropy with `PAD` masked;
- optimizer: AdamW;
- learning rate: 0.001;
- weight decay: 0;
- scheduler: none;
- gradient clipping: max norm 1.0;
- smoke steps: 2;
- Stage 7-C checkpoint-selection rule: minimum validation loss;
- deterministic CPU thread/RNG policy with explicit master seed.

Only `train` batches may call the gradient-update path. Only `validation` batches may call the validation-loss path. Validation is required not to mutate model state.

NaN or Infinity in inputs, logits, loss, gradients, model parameters, optimizer tensor/scalar state, gradient norm, or reported loss is a hard failure.

## Reproducibility surface

Stage 7-B fingerprints:

- tokenizer version + exact vocabulary;
- preprocessing version + fixed canvas policy;
- model version + architecture config + frozen vocabulary size + PyTorch pin;
- trainer version + optimizer/loss/clip/checkpoint-selection config + PyTorch pin.

CPU smoke construction seeds PyTorch before model initialization, enables deterministic algorithms, fixes CPU thread count to one, and hashes model state from sorted state names plus exact tensor bytes. Same seed/config/input smoke replay must produce the same initial state hash, losses, and final state hash.

This evidence is deliberately limited to the exact verified CPU runtime. It is not a CUDA/ROCm or cross-platform bit-identity claim.

## Completed Stage 7-B verification

The final Stage 7-B source head was `a8ad8bc9f14953f0ed35ef5a5a8275be69af5ebd` against exact base `6a13760d9d17130ea86636f4828ff1bff035f30d`. GitHub Actions run #30 (`31679413312`) succeeded on GitHub-generated merge candidate `3d5ee4e2ca479614e2a3322e0339ca21f25e5cae`. PR #21 was then separately approved and squash-merged to exact `main` commit `d02dce4ee17dfccf6f05519ab0970fdc188d0147`; post-merge run #31 (`31679810478`) succeeded on that exact `main` SHA.

Completed gate:

```text
exact framework/runtime pin
        ↓
frozen 35-token vocabulary tests
        ↓
MusicXML → tokens → projection exact semantic round trip
        ↓
mandatory EOS consumption + missing-EOS regression
        ↓
Stage 5/6 persisted-artifact revalidation
        ↓
test-split sealed-path tests
        ↓
deterministic no-crop preprocessing tests
        ↓
bounded model + parameter ceiling tests
        ↓
CPU forward/backward/update smoke
        ↓
NaN/Infinity fail-closed tests
        ↓
validation-no-mutation tests
        ↓
same-seed exact CPU smoke replay
        ↓
real Stage 6 persisted dataset → Stage 7-B smoke bridge
        ↓
full repository regression: 336/336
        ↓
compile validation
        ↓
GitHub-hosted PR CI
        ↓
separate merge approval
        ↓
post-merge exact-main CI
        ↓
CLOSED
```

## Explicitly out of scope

Stage 7-B does not perform the real Stage 7-C baseline training run, does not open the sealed test split, does not retain production checkpoints, does not tune against test data, does not ingest real/user/copyright-unclear material, and does not implement Stage 8, Stage 9, Stage 10, Guitar TAB training, cloud training, or ScoreMosaic integration.

Stage 7-C was implemented and accepted separately through PR #23. Its exact run evidence and limitations are recorded in `STAGE7C_EVIDENCE.md`; this Stage 7-B profile remains unchanged and closed.
