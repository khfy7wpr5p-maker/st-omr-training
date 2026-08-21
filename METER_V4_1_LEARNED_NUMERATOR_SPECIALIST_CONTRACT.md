# Meter V4-1 — Learned Numerator Specialist Contract

## Purpose

V4-1 is the first learned follow-up to the accepted V4-0 representation audit. It tests whether a deliberately small numerator-only CNN can improve family-generalizing separation of the positive Meter classes `2/4`, `3/4`, and `4/4` without returning to whole-ROI Meter classification or to V3 logit/margin tuning.

V4-1 is development/shadow evidence only. It does not replace D11/V3, does not classify `none`, does not connect runtime/Resolver, and does not authorize production promotion.

## Accepted parent evidence

The only admissible image surface is the exact completed V4-0 artifact bound to:

- V4-0 Git head: `634750a701e934926247065fea45cfbc15f6ec53`;
- V4-0 repository SHA-256 binding: `8641fc45ae0e5613d280eee8af12ac105c765c313190660c88479e38bf6eff48`;
- V4-0 `result.json` SHA-256: `422e79d7f71a1d2228e1392160d6ef4444521d8796f3c3b8fb6cd0a226c9060a`;
- V4-0 decision: `REPRESENTATION_SIGNAL_STRONG`;
- V4-0 OOF: `25/27`, macro-F1 `0.9259259259`, recalls `2=8/9`, `3=8/9`, `4=9/9`.

V4-1 must verify the V4-0 `COMPLETE` receipt, canonical result hash, exact repository binding, safety fields, exact 27 crop identities, fold assignments and each crop PNG SHA-256 before training.

## Data boundary

V4-1 uses exactly the 27 V4-0 numerator crops:

- `2`: 9 families;
- `3`: 9 families;
- `4`: 9 families.

The frozen V4-0 fold assignment is reused exactly. Every OOF fold therefore trains on 18 families (6 per class) and predicts 9 unseen families (3 per class).

Forbidden data:

- Teacher Gold adaptation-validation images or labels;
- D10;
- sealed TEST;
- V3/D11 checkpoint outputs as classifier inputs;
- validation-ID or family-ID special cases;
- extra hand-edited crops.

## Input representation

The V4-0 crop contract is immutable for V4-1:

- gray8 PNG;
- exact `64x64`;
- white background / dark ink;
- model tensor = `(255 - pixel) / 255`, float32 in `[0,1]`.

Training-only augmentation is a fixed integer translation bank over each TRAIN crop:

```text
x shift ∈ {-2, 0, +2}
y shift ∈ {-2, 0, +2}
```

This yields exactly 9 deterministic views per training crop. Wrapped pixels are forbidden; exposed pixels are filled with zero ink/background. No rotation, blur, morphology, font synthesis, random augmentation or held-out-derived transform is allowed.

Held-out crops are evaluated unaugmented.

## Model

The frozen specialist is intentionally small:

```text
64x64x1
 -> Conv 1->8, 3x3, padding=1
 -> ReLU
 -> MaxPool 2x2
 -> Conv 8->16, 3x3, padding=1
 -> ReLU
 -> MaxPool 2x2
 -> AdaptiveAvgPool 4x4
 -> Linear 256->32
 -> ReLU
 -> Linear 32->3
```

Class order is frozen to `2, 3, 4`. No `none` output exists in V4-1.

## Training

- runtime: exact repository-pinned CPU PyTorch;
- master seed: `812041`;
- fold seed: `master_seed + fold`;
- optimizer: AdamW;
- learning rate: `0.001`;
- weight decay: `0.0001`;
- objective: cross entropy;
- epochs: exactly `160`;
- batch policy: one deterministic full augmented TRAIN batch per optimizer step;
- optimizer steps: exactly `160` per fold;
- gradient clipping: L2 norm `1.0`;
- scheduler: none;
- early stopping: forbidden;
- held-out checkpoint selection: forbidden.

The final fixed-epoch state predicts that fold. No OOF metric may alter learning rate, epochs, augmentation, architecture, loss, threshold or seed.

## Determinism

V4-1 must run CPU-only with deterministic PyTorch algorithms and one Torch thread. Model initialization, training tensor order and augmentation order are fixed. The result records per-fold final model-state SHA-256 values.

The runner must fail closed on non-finite loss/logits/parameters, wrong Torch runtime, unexpected parameter count, changed fold cardinality, duplicate family/record identity, or any parent-artifact mismatch.

## Decision gate

V4-1 is required to **improve** on the V4-0 centroid baseline rather than merely match it.

`LEARNED_NUMERATOR_SIGNAL_STRONG` requires all of:

- OOF accuracy >= `26/27`;
- OOF macro-F1 >= `0.95`;
- recall(`2`) >= `8/9`;
- recall(`3`) >= `8/9`;
- recall(`4`) >= `8/9`;
- no data-boundary or determinism failure.

Otherwise the decision is `LEARNED_NUMERATOR_SIGNAL_INSUFFICIENT` and the next action is data/crop/domain-diversity work, not another unconstrained hyperparameter sweep.

Passing V4-1 authorizes only planning the next numerator-specialist stage. It does not authorize runtime or production promotion.

## Required result evidence

The bounded Colab run writes a canonical result JSON containing:

- exact V4-0 parent hashes and safety fields;
- V4-1 repository SHA and frozen config fingerprint;
- exact 27 records and inherited fold assignments;
- per-fold TRAIN/HOLDOUT family identities;
- per-fold final model state SHA-256;
- per-record logits/probabilities/prediction;
- aggregate 3x3 confusion, accuracy, macro-F1 and per-class recall;
- decision and reasons;
- `teacher_adaptation_validation_evaluated=false`;
- `d10_opened=false`;
- `test_opened=false`;
- `runtime_connected=false`;
- `resolver_connected=false`;
- `production_promotion_authorized=false`.

No model checkpoint is promoted or exposed as a production candidate by V4-1.