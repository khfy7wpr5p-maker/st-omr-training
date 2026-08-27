# Meter V5-3K — TRAIN-only Feature / Domain Shift Forensics v1

## Purpose

V5-3K is a read-only diagnostic stage opened after the completed V5-3J rescue-failure forensic report. It asks a narrower question:

> Where do the V5 and historical TRAIN domains diverge inside the frozen 64D representation and inside the fixed rescue network's 8D hidden representation?

This stage does not select a repair recipe. It exists to decide whether the next separately staged hypothesis should concern hard-negative handling, domain normalization, representation/crop behavior, or another explicitly supported mechanism.

## Immutable prerequisite evidence

- V5-3J final wrapper HEAD: `08b2458cf6fa4aee3e5f32d1aefbe637cdbd01ec`
- V5-3J implementation HEAD: `c978b14fba23f91c60f06d2166bb23e87856d8d6`
- V5-3J module blob: `092a32504ffee9b9aafa74ddefea1c2aeb831e56`
- V5-3J report SHA-256: `7a49d29e0d7257be7c59d499ab3d9ab575d369a7473b0b5298ea62aa80c7d37f`

The V5-3J report must still reproduce the V5-3I HOLD and the exact failure signatures:

- 2-AI: `V5_RECOVERED_HISTORICAL_TN_COLLAPSE`, V5 positive recovery `1.0`, historical TN regressions `5307`.
- 3-AI: `V5_POSITIVE_NOT_RECOVERED_HISTORICAL_TN_COLLAPSE`, V5 positive recovery `0.0`, historical TN regressions `15775`.

Any receipt or group-identity drift fails closed.

## Allowed surfaces

V5-3K may read only:

- V5 adaptation TRAIN;
- historical TRAIN;
- exact frozen 2-AI / 3-AI checkpoints;
- exact V5-3G rescue artifacts;
- exact V5-3G report and V5-3H execution envelope;
- exact V5-3J forensic report.

4-AI is not loaded.

Historical Validation, immutable First-30, V5 reserve, V5 validation, and FINAL_HOLDOUT remain closed.

## Fixed descriptive comparisons

For each 2-AI and 3-AI specialist, V5-3K reports four fixed comparisons:

1. same-label negative domain shift: V5 frozen TN vs historical frozen TN;
2. same-label positive domain shift: V5 frozen FN-positive vs historical frozen FN-positive;
3. critical conflict: V5 positive vs historical TN;
4. historical fixed-threshold hard TN vs preserved historical TN.

The historical hard-TN split uses only the already-fixed rescue threshold `0.50`. It is not a threshold sweep and does not evaluate alternative thresholds.

## 64D frozen-feature evidence

For every comparison V5-3K emits:

- group counts;
- per-dimension means and population standard deviations;
- centroid L2 distance;
- within-group centroid RMS radii;
- centroid-distance / summed-within-RMS ratio;
- centroid cosine similarity;
- standardized mean shift per dimension;
- the ten largest absolute standardized feature shifts.

No classifier is fitted and no PCA or learned domain projection is fitted.

## 8D rescue-hidden evidence

The already-trained rescue artifact remains frozen. V5-3K reads the fixed hidden activation

`Linear(64, 8) -> tanh`

and reports the same descriptive geometry/shift statistics over the 8D activations.

For each fixed comparison it also decomposes the difference in mean output logit into the eight fixed hidden-dimension contributions. This is an algebraic read-only decomposition of the existing output layer; it does not alter any parameter.

## Safety boundary

V5-3K explicitly forbids:

- training or fitting;
- autograd/backward;
- optimizer steps;
- checkpoint or rescue-artifact writes;
- threshold tuning or sweeps;
- hyperparameter sweeps;
- automatic second configurations;
- architecture changes;
- repair-recipe selection;
- retraining authorization;
- resolver/runtime wiring;
- production promotion;
- protected validation access.

Frozen and rescue state fingerprints are checked before and after diagnostics and must remain bit-identical.

## Interpretation policy

V5-3K produces evidence, not an automatic decision. In particular:

- a large same-label V5-vs-historical shift may support a later domain-normalization or representation hypothesis;
- a concentrated hard-negative subpopulation may support a later hard-negative-aware hypothesis;
- neither observation by itself authorizes a new training run;
- 2-AI and 3-AI may receive different later hypotheses if the evidence supports different mechanisms.

## Future gate order

1. V5-3K TRAIN-only feature/domain-shift forensics;
2. separately preregister one digit-specific repair hypothesis if supported;
3. separately authorize one fixed repair execution if approved;
4. run a new TRAIN acceptance gate;
5. only after TRAIN acceptance PASS may historical-validation retention become eligible.

A V5-3K report cannot authorize retraining by itself.
