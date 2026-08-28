# TR-POLY-08 — Tiny 2D Transformer Prototype

## Purpose

TR-POLY-08 introduces the first bounded Polyphonic Representation V2 model candidate.

It is a **research prototype**, not a winning model, not a production recognizer, and not a ScoreMosaic runtime component.

No benchmark result is claimed in this package. Common comparison remains TR-POLY-09.

## Model identity

`st-omr-poly-2d-transformer-v1`

Profile:

`st-omr-poly-2d-transformer-profile-v1`

Target surface:

- representation: `st-omr-polyphonic-representation-v2`
- tokenizer: `st-omr-polyphonic-tokenizer-v1`
- PyTorch: repository-pinned CPU runtime

## Why this differs from the V1 CNN-GRU baseline

The V1 sequence baseline eventually averages visual encoder steps into one global conditioning vector. That loses the vertical relationships needed for polyphony, multi-staff reading and simultaneous voices.

The TR-POLY-08 encoder does not perform that collapse.

Pipeline:

`grayscale image`
→ `non-overlapping 2D patch embedding`
→ `patch_rows × patch_columns grid`
→ `independent row + column position embeddings`
→ `full 2D Transformer encoder memory`
→ `autoregressive V2 token decoder with cross-attention`

Every visual patch remains an independent memory token. The invariant is:

`visual_memory_tokens == patch_rows * patch_columns`

No vertical mean or global visual context vector is permitted.

## Frozen prototype profile

- input: `1 × 96 × 512` grayscale float32
- patch: `16 × 16`
- visual grid: `6 × 32 = 192` memory tokens
- model dimension: `64`
- encoder layers: `2`
- decoder layers: `2`
- attention heads: `4`
- feed-forward dimension: `128`
- dropout: `0`
- bounded target prefix: max `2048` tokens
- trainable parameter ceiling: `5,000,000`

The target-length ceiling is a prototype compute boundary, not a new tokenizer limit. The frozen V2 tokenizer itself remains unchanged.

## Decoder

The decoder consumes V2 token IDs and uses:

- frozen V2 vocabulary size;
- V2 PAD identity;
- learned target positions;
- causal self-attention;
- cross-attention to the complete 2D visual memory.

The model only exposes a teacher-forced forward surface in TR-POLY-08. Greedy/beam decoding, constrained semantic decoding, training orchestration and checkpoint selection are separate stages.

## Registry transition

TR-POLY-07 registered `candidate.poly-2d-transformer.v1` as `planned`.

TR-POLY-08 changes the same research identity to:

- lifecycle: `training_implemented`;
- authority: `experimental`;
- source module: `st_omr_training.poly_2d_transformer`;
- source version: `st-omr-poly-2d-transformer-v1`.

This does **not** mean a trained checkpoint exists. Any later checkpoint evidence must still provide the exact registry-row fingerprint, repository SHA, checkpoint/model/profile/dataset/runtime hashes, V2 tokenizer fingerprint and V2 representation identity.

Because the registry-row fingerprint changes, a hypothetical old artifact binding to the former planned row cannot be silently reused.

## Tests / architecture guards

The focused contract tests verify:

- exact version bindings;
- both patch axes contain multiple positions;
- encoder memory retains every row × column patch;
- row and column positional embeddings are independent;
- moving local ink vertically changes visual memory;
- decoder logits use the exact V2 vocabulary;
- causal masking;
- deterministic same-seed initialization;
- bounded parameter count;
- deterministic profile fingerprint;
- fail-closed image/token/config inputs;
- V2 PAD identity is retained.

## Explicit non-goals

TR-POLY-08 does not:

- train on TRAIN/VALIDATION/TEST;
- access sealed TEST;
- create, load or select a checkpoint;
- claim improved accuracy;
- run OLiMPiC/GrandStaff benchmarking;
- implement MusicXML → V2 import;
- replace the specialist architecture;
- replace deterministic musical validation;
- change V1 core/tokenizer/baseline;
- change Meter/Rest runtime decisions;
- wire into ScoreMosaic;
- grant production authority.

## Next gate

TR-POLY-09 must place the frozen CNN-GRU baseline and this 2D candidate behind the same benchmark identity and evaluation contract before any comparative claim is made.
