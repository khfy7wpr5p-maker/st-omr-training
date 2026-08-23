# Meter V5-2R — TRAIN-only Class / Margin / Gradient Contribution Audit V1

## Status

Read-only evidence audit only. No training or repair objective is authorized by this stage.

V5-2P remains closed as **HOLD**. V5-2Q completed successfully as a TRAIN-only margin-geometry audit and did not select a new repair. V5-2R exists only to separate three candidate mechanisms using already-open TRAIN evidence:

1. class-frequency / gradient contribution imbalance;
2. uncontrolled head-weight growth or rotation;
3. a mixed mechanism.

No mechanism is selected by threshold in this stage. The report remains descriptive.

## Allowed evidence surfaces

Only:

- V5 adaptation TRAIN: the already-open 540 specialist slots;
- historical M4A TRAIN: the already-open 26,964 records;
- exact frozen 2-AI and 3-AI checkpoints;
- exact retained V5-2P HOLD candidate checkpoints;
- V5-2P training/numerical-integrity evidence required only to bind those exact candidates.

The V5-2P historical retention report is not read as a model-design input.

## Leakage boundary

Forbidden:

- historical M4A VALIDATION examples;
- validation error identities, images, logits, margins, labels, or transition classes;
- First-30 examples;
- V5 VALIDATION;
- FINAL_HOLDOUT.

No per-example row or identifier is emitted. Historical transition evidence is aggregate TRAIN-only counts.

## Required evidence

For each of 2-AI and 3-AI:

### Historical TRAIN positive transition matrix

At the unchanged frozen runtime threshold:

- correct → correct;
- correct → wrong;
- wrong → correct;
- wrong → wrong.

This matrix is computed only over historical TRAIN positives.

### Negative classification-margin distributions

For historical TRAIN negatives and V5 TRAIN negatives, frozen and candidate:

`negative_margin = threshold_logit - logit`

Report:

- count;
- mean;
- population standard deviation;
- min;
- p01;
- p05;
- p10;
- p25;
- p50;
- p75;
- p90;
- p95;
- p99;
- max.

### Head geometry

Report:

- frozen `head.weight` L2 norm;
- candidate `head.weight` L2 norm;
- delta-weight L2 norm;
- `||delta||2 / ||frozen||2`;
- `||candidate||2 / ||frozen||2`;
- frozen/candidate cosine;
- head angle change in degrees;
- delta-weight max absolute component.

No norm threshold is used to select a repair.

### Four-group BCE and analytic head-weight gradient

Groups:

- V5 positive;
- V5 negative;
- historical positive;
- historical negative.

Evaluate at both:

- exact frozen head;
- exact retained V5-2P candidate head.

For each group report:

- group count;
- empirical frequency within its domain;
- carried-forward V5-2P domain weight (`0.5` for V5, `0.5` for historical) for evidence reconstruction only;
- exact coefficient in the existing V5-2P objective;
- mean BCE-with-logits;
- weighted objective BCE contribution;
- analytic mean head-weight gradient L2 norm;
- analytic mean head-weight gradient infinity norm;
- analytic mean absolute gradient;
- weighted-objective gradient L2 and infinity norms.

The analytic gradient is computed in closed form:

`g = X^T(sigmoid(Xw+b) - y) / N`

No autograd or backward call is permitted.

### Gradient conflict matrices

At frozen and candidate heads, report all pairwise gradient cosines among the four groups.

Negative cosine means opposed local directions. No conflict threshold is used to choose an objective or repair.

### Exact objective reconstruction

The four empirical-frequency coefficients must sum to exactly one within tolerance:

- V5 positive + V5 negative = `0.5`;
- historical positive + historical negative = `0.5`.

Report the reconstructed existing V5-2P total weighted BCE and total weighted gradient norms at frozen and candidate heads.

This is evidence about the already-run V5-2P objective. It does not modify that objective.

## State-integrity requirements

Before evidence is accepted:

- candidate SHA256 must match the retained V5-2P HOLD artifact;
- state keys must match the frozen model;
- `features.*` must be bit-identical;
- `head.bias` must be bit-identical;
- only `head.weight` may differ;
- `head.weight` must remain exactly 64 parameters;
- thresholds remain frozen.

Any violation fails closed.

## Prohibited actions and interpretations

V5-2R must not:

- train;
- call autograd or backward;
- take optimizer steps;
- mutate or replace checkpoints;
- tune threshold or bias;
- modify domain weights;
- change LBFGS settings;
- introduce another solver;
- select a minimum-change penalty;
- select a historical-positive constraint;
- select a margin floor or constraint coefficient;
- select a new architecture;
- open validation/First-30/final surfaces;
- claim that class imbalance, weight growth, or a mixed mechanism is proven by a post-hoc threshold not preregistered here.

The output should provide enough TRAIN-only evidence for a later scientific interpretation. If a new constrained head repair is proposed after V5-2R, it is a new objective/new experiment and requires a separate preregistration before any gradient execution.

## Safety boundary

- training: **False**
- autograd/backward: **False**
- optimizer steps: **0**
- checkpoint writes/mutations: **False**
- historical VALIDATION: **closed**
- First-30: **closed**
- V5 VALIDATION: **closed**
- FINAL_HOLDOUT: **locked**
- 4-AI: **frozen**
- repair selected: **False**
- new objective selected: **False**
- production promotion: **False**
