# Meter V5-2D — Positive-Collapse Forensic Audit V1

## Purpose

V5-2C retention V2 proved that the historical pixel path is reproducible while the V5-2B adapted 2-AI and 3-AI candidates collapse toward positive predictions on the historical domain. V5-2D diagnoses that learning/decision path before any new BBox work or new gradient step.

This stage is read-only and inference-only. It is not a repair stage.

## Questions V5-2D may answer

The audit is designed to distinguish evidence for these hypotheses without changing the model:

1. label polarity or label-manifest inversion;
2. missing/incorrect negative examples in the V5 adaptation TRAIN construction;
3. class-weight / loss-balance behavior;
4. wrong source checkpoint or candidate-state binding;
5. candidate-specific preprocessing divergence;
6. output-head bias drift versus deeper feature drift;
7. historical-domain score/logit displacement after V5 adaptation.

It must not claim a root cause that the measured evidence does not support.

## Bound evidence

The audit requires the already-written V5-2C retention V2 report to remain `HOLD`, with historical pixel-path reproduction `true`, V5 validation-BBox authorization `false`, and the exact V5-2B candidate SHA identities.

Candidate identities remain:

- 2-AI: `61e4ed5c595d66214ab863f53094998e5cc5167094dc8a9b5934470e3188d4f2`
- 3-AI: `5d8dd8ea3aed5c2aaa383d2a494e762276afa952f5da6d37fc1dc214900f1c62`

Frozen source checkpoint identities remain unchanged.

## Data surfaces

V5-2D uses only:

- the existing 540 V5 `adaptation_train` slot crops derived from the approved 270 TRAIN samples;
- the historical M4A **TRAIN** split replayed from D10 source pixels.

Historical M4A TRAIN is frozen at exactly 26,964 records:

- digit `2`: 1,527
- digit `3`: 1,587
- digit `4`: 6,396
- `NONE`: 17,454

M4A validation is not used for root-cause measurements in V5-2D. V5 VAL and FINAL_HOLDOUT are not opened.

## Measurements

### A. Label-polarity and construction audit

For each trainable specialist, the 540 V5 adaptation slots must contain exactly:

- 90 positive labels;
- 450 negative labels;
- all denominator slots negative for 2-AI and 3-AI;
- a positive only when the slot is the numerator and the meter numerator equals the specialist digit.

The deterministic 12-epoch shuffle is reconstructed without gradients. Each epoch must contain all 540 rows exactly once. Batch-level positive/negative counts are reported.

### B. Loss contract audit

The frozen V5-2B objective is reconstructed without backward/optimizer execution:

- BCE-with-logits;
- positive weight = 450 / 90 = 5.0;
- same 540 inputs and same labels for frozen-source and candidate score evaluation.

Per-class weighted loss sums/means are reported for the frozen source and candidate. This is diagnostic only and does not authorize changing the loss.

### C. Checkpoint/state provenance

The audit re-verifies:

- exact frozen checkpoint SHA;
- exact candidate SHA;
- candidate source-checkpoint metadata;
- candidate state fingerprint;
- slot-manifest SHA binding.

### D. Parameter drift

For every parameter tensor, V5-2D reports:

- frozen L2 norm;
- candidate L2 norm;
- delta L2 norm;
- relative delta;
- cosine similarity where defined.

`head.bias` and `head.weight` are highlighted separately, but the audit must not infer that the head alone caused collapse unless deeper-layer measurements support that conclusion.

### E. Shared-input logit audit

Frozen and candidate models are fed the exact same tensor in each comparison. There is no candidate-only preprocessing branch.

For V5 adaptation TRAIN and historical M4A TRAIN, the audit reports by true label:

- logit distribution;
- probability distribution;
- positive prediction rate at the unchanged frozen threshold.

This directly measures where positive score displacement appears.

## Safety boundary

V5-2D:

- performs zero optimizer steps;
- performs no backward pass;
- writes no checkpoint;
- changes no threshold;
- adds no crop/BBox/staff/localization rule;
- does not use the 900 reserve V5 TRAIN samples;
- does not open V5 VAL;
- does not open FINAL_HOLDOUT;
- keeps 4-AI frozen and outside this forensic comparison;
- does not wire Resolver or production authority.

A V5-2D result may authorize only a root-cause decision. Any new training semantics, replay mixture, layer freezing, loss change, sampling change, or new spatial rule requires a separate explicit decision before gradients.