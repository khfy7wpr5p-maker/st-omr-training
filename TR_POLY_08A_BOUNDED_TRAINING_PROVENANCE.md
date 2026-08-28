# TR-POLY-08A — Bounded Training & Provenance Harness

## Purpose

TR-POLY-08 introduced a tiny 2D Transformer forward architecture for the frozen Polyphonic Representation V2 target surface. Review correctly kept that candidate out of the `training_implemented` registry lifecycle because no loss, optimizer, split enforcement, validation non-mutation, or training orchestration existed.

TR-POLY-08A fills that narrow gap without running an authoritative experiment.

## Added surface

`st_omr_training/poly_2d_training.py` defines:

- `st-omr-poly-2d-trainer-v1`;
- a bounded V2 tensor-batch contract;
- a fixed AdamW smoke-training profile;
- V2 PAD-aware cross-entropy;
- TRAIN-only optimizer updates;
- read-only VALIDATION evaluation with before/after model-state hashing;
- deterministic model/trainer profile fingerprints;
- repository/dataset/preprocess/model/trainer/tokenizer/runtime provenance binding;
- a bounded in-memory smoke-training orchestration path;
- a result receipt that explicitly forbids checkpoint, authoritative-dataset, and TEST-access claims.

## Split safety

`DatasetSplit.TEST` is rejected when a `Poly2DTrainingBatch` is constructed.

Gradient updates require exactly `DatasetSplit.TRAIN`.

Read-only evaluation requires exactly `DatasetSplit.VALIDATION` and verifies that the model state SHA-256 is unchanged.

No function in this package accepts a TEST batch.

## Teacher-forcing contract

A batch must use the frozen V2 vocabulary and:

- rank-2 `torch.long` decoder IDs and labels;
- identical decoder/label shapes;
- V2 BOS at the first decoder position of every row;
- contiguous right padding only;
- matching decoder/label padding masks;
- at least one non-PAD target;
- token IDs inside the frozen V2 vocabulary.

Images must be finite normalized float32 tensors in `[0, 1]`.

The model performs the final exact geometry and target-length check against its bound `Poly2DTransformerConfig`.

## Fixed training profile

Default bounded profile:

- seed: `82081`;
- optimizer: AdamW;
- learning rate: `5e-4`;
- weight decay: `1e-4`;
- global gradient clip: `1.0`;
- objective: cross entropy ignoring V2 PAD;
- scheduler: none;
- smoke steps: exactly 2 maximum.

This is a CI/research execution harness, not a selected production recipe.

## Provenance

Each bounded run must bind:

- exact repository SHA-40;
- exact dataset-manifest SHA-256;
- exact preprocess fingerprint SHA-256;
- exact 2D model-profile SHA-256;
- exact trainer-profile SHA-256;
- exact V2 tokenizer fingerprint;
- frozen V2 representation version;
- frozen V2 tokenizer version;
- pinned PyTorch runtime.

The provenance object has its own deterministic SHA-256 fingerprint.

A batch whose dataset identity differs from the provenance fails closed.

A changed model or trainer configuration fails closed against the provenance.

## Registry boundary remains closed

This package deliberately does **not** change `candidate.poly-2d-transformer.v1` from `architecture_only`.

Reason: TR-POLY-08A still writes no checkpoint and provides no persistence/reload verifier. Keeping the registry row closed prevents an in-memory CI smoke run from being misrepresented as a durable model artifact.

A later TR-POLY-08B package may advance the lifecycle only if it adds a bounded checkpoint artifact contract with:

- one exact state payload;
- canonical metadata;
- checkpoint SHA-256;
- reload + state-hash verification;
- exact model/trainer/dataset/runtime/tokenizer/provenance binding;
- non-overwrite behavior;
- no TEST access;
- no performance/promotion claim.

## Explicit non-goals

TR-POLY-08A does not:

- load repository TRAIN/VALIDATION data;
- execute an authoritative training run;
- create or persist a checkpoint;
- open the sealed TEST split;
- select a checkpoint by validation score;
- tune thresholds or hyperparameters;
- run a benchmark;
- claim improvement over CNN-GRU or specialists;
- change Polyphonic Representation V2;
- change the V2 tokenizer;
- change V1 models or specialist chains;
- wire anything into ScoreMosaic;
- grant production authority.

## Next gate

TR-POLY-08B: exact checkpoint persistence/reload contract for this already-fixed bounded trainer.

Only after a conforming exact artifact exists should `candidate.poly-2d-transformer.v1` move to `training_implemented`, and only after that should TR-POLY-09 compare it under the frozen common benchmark contract.
