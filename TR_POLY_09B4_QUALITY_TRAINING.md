# TR-POLY-09B4 — Deterministic Quality-Training Regime

Status: implementation package; quality checkpoint persistence and real benchmark execution remain separate gates.

## Why this package exists

TR-POLY-08A intentionally caps training at two optimizer steps. That contract is useful for deterministic smoke verification but cannot produce a meaningful trained-model quality claim.

TR-POLY-09B4 adds a separate, versioned multi-epoch TRAIN-only regime without weakening or rewriting the frozen smoke contract.

## Execution boundary

```text
prevalidated native V2 TRAIN batches
        +
prevalidated native V2 VALIDATION batches
        +
exact quality-training config
        +
quality provenance
        ↓
new 2D Transformer from frozen seed
        ↓
fixed ordered TRAIN pass × N epochs
        ↓
read-only full VALIDATION each epoch
        ↓
minimum mean validation-loss selection
        ↓
selected model state + exact epoch evidence
```

TEST is not an accepted batch type anywhere in this path.

## Reuse instead of duplication

The quality trainer reuses the hardened TR-POLY-08A primitives for:

- AdamW construction;
- one-step TRAIN update;
- PAD-aware cross-entropy;
- finite gradient/state checks;
- gradient clipping;
- read-only VALIDATION loss;
- model-state hashing.

The quality package maps its optimizer/loss fields into a one-step `Poly2DTrainingConfig(smoke_steps=1)`. The multi-epoch control plane is separate and receives its own version/fingerprint.

## Frozen v1 policies

- batch order: `caller-frozen-tuple-order-v1`;
- scheduler: none;
- validation: every epoch, over every supplied VALIDATION batch;
- validation aggregation: arithmetic mean of batch losses;
- candidate selection: minimum mean VALIDATION loss;
- tie policy: earliest epoch wins;
- TEST access: prohibited;
- production authority: prohibited.

The ordered TRAIN/VALIDATION batch plan receives its own SHA-256 fingerprint. Reordering batches changes the training-plan identity.

## Resource bounds

The v1 control plane supports:

- at most 128 epochs;
- at most 100,000 optimizer steps;
- at most 100,000 supplied batches per split;
- the existing per-batch image/token/resource limits from TR-POLY-08A.

The exact run must satisfy `epochs × number_of_train_batches <= max_optimizer_steps` before the first gradient update.

These are safety ceilings, not recommended hyperparameters.

## Evidence emitted per epoch

Each epoch records:

- epoch number;
- cumulative optimizer steps;
- mean TRAIN loss;
- mean read-only VALIDATION loss;
- exact model-state SHA-256.

The final run result binds:

- initial model state;
- selected model state;
- selected epoch and validation loss;
- all epoch evidence;
- ordered training-plan SHA-256;
- model profile;
- quality trainer profile;
- underlying one-step primitive profile;
- quality-training provenance;
- dataset manifest;
- tokenizer fingerprint;
- repository SHA;
- pinned PyTorch runtime.

The returned model is reloaded to the selected epoch and its exact state hash is reverified.

## Determinism and selection

The model starts from the exact configured seed and follows the exact caller-supplied batch tuple order. No shuffle, sampling, scheduler or hidden early-stop rule is introduced in v1.

The fixed epoch budget is completed and the best VALIDATION epoch is selected afterward. This prevents a hidden or implementation-dependent stopping rule from changing the candidate identity.

A later shuffle/scheduler/early-stop regime must receive a new version and cannot silently inherit this trainer fingerprint.

## Data separation

- only TRAIN batches may call the gradient update primitive;
- only VALIDATION batches may enter epoch selection;
- duplicate sample IDs across the supplied TRAIN/VALIDATION plan are rejected;
- every batch must bind the exact same dataset manifest as the quality provenance;
- TEST cannot be materialized into the existing `Poly2DTrainingBatch` and is rejected again by the quality boundary.

## What B4 proves

A green B4 merge proves the repository can execute a deterministic multi-epoch training/VALIDATION-selection regime while preserving the existing safety boundaries.

It does **not** prove:

- that a quality checkpoint has been persisted;
- that a real external/native V2 training corpus has been run;
- that the default eight-epoch recipe is optimal;
- any OMR accuracy;
- TEST performance;
- ScoreMosaic or production readiness.

## Next gate — quality checkpoint

The existing TR-POLY-08B checkpoint schema is intentionally frozen to at most two optimizer steps. It must not be relaxed in place.

The next package should therefore introduce a separate quality-checkpoint artifact contract that:

- persists the selected B4 model state;
- embeds the exact B4 config/provenance/result identity;
- verifies hashes before deserialization;
- reloads with `weights_only=True`;
- binds a dedicated research registry record;
- carries no benchmark or production authority;
- remains compatible with B1 free-running inference through a separately verified wrapper.

Only after that quality checkpoint exists should the first genuine quality VALIDATION B1→B2→B3 run be executed.
