# Meter V5-2Q — Historical-Positive Margin Audit V1

## Status

Read-only preregistered audit only. No training is authorized by this stage.

V5-2P is closed as **HOLD**. Its retained candidate checkpoints are evidence artifacts only:

- 2-AI SHA256 `369b7f610b1d9785368422f62669419868a8b86975a89f13b03c169fa6161616`
- 3-AI SHA256 `491602359a74c4829a205d60c194a00e63b54553823873f814980457667ac785`

The V5-2P HOLD result establishes that the 64-parameter fixed-bias head repair learned the open V5 TRAIN surface but did not satisfy the preregistered historical retention gate, especially for 3-AI recall. This audit does not reopen that validation evidence for model design.

## Scientific question

Using only already-open TRAIN surfaces, determine how the exact V5-2P head rotation changed historical-positive decision margins and whether the geometry is consistent with:

1. broad positive-margin compression;
2. concentration of loss in the low-margin historical-positive tail;
3. an opposed direction between V5 margin gain and historical-positive margin change;
4. a stronger systematic narrowing pattern for 3-AI than 2-AI.

The audit is descriptive. It does not select a new objective, constraint, solver, threshold, architecture, or training recipe.

## Allowed data surfaces

Only:

- V5 adaptation TRAIN: the already-open 540 specialist slots;
- historical M4A TRAIN: the already-open 26,964 records;
- exact frozen 2-AI and 3-AI source checkpoints;
- exact retained V5-2P HOLD candidate checkpoints;
- V5-2P training and numerical-integrity evidence needed to bind those candidates.

## Explicit leakage boundary

The following are forbidden inputs to this audit:

- historical M4A VALIDATION examples;
- identities, images, logits, margins, labels, or transition classes of individual historical VALIDATION errors;
- the V5-2P historical-retention report as a model-design input;
- First-30 examples;
- V5 VALIDATION;
- FINAL_HOLDOUT.

No per-example row or identifier is emitted by the audit report. All reported geometry is aggregate TRAIN-only evidence.

## Exact margin definition

For specialist threshold `t`, define the frozen runtime boundary in logit space as:

`b = log(t / (1 - t))`

For a positive TRAIN feature vector `x`, head weight `w`, and frozen bias `c`:

`positive_margin(x, w) = x · w + c - b`

The exact head-rotation effect is:

`margin_delta(x) = x · (w_candidate - w_frozen)`

because `head.bias` and the runtime threshold are unchanged.

For V5 TRAIN negatives, classification margin is measured with the opposite sign so positive values always mean movement farther into the correct side of the frozen boundary.

## Required aggregate evidence

For each of 2-AI and 3-AI report:

### Historical TRAIN positives

- frozen positive-margin quantiles: min, p01, p05, p10, p25, p50, p75, p90, p95, p99, max;
- candidate positive-margin quantiles at the unchanged threshold;
- candidate-minus-frozen margin-shift quantiles;
- mean and population standard deviation;
- fraction of positive margins decreased/increased;
- count below the frozen threshold before and after the head rotation;
- rank-binned shift summaries for bottom 10%, p10–p25, p25–p50, p50–p75, p75–p90, and top 10% of the frozen-margin distribution.

### V5 TRAIN

- V5-positive frozen and candidate margin quantiles;
- V5-positive candidate-minus-frozen margin-shift quantiles;
- all-class V5 classification-margin change quantiles;
- fraction of V5 TRAIN rows whose classification margin improved.

### Directional relation

- L2 and max-absolute norm of `delta_weight`;
- cosine of `delta_weight` with historical-positive and V5-positive mean feature vectors when defined;
- historical-positive mean logit shift;
- V5-positive mean logit shift;
- V5 all-class mean classification-margin change;
- whether V5-positive gain and historical-positive loss have opposite mean signs.

## State-integrity requirements

Before any geometry is accepted:

- candidate SHA256 must match the exact retained V5-2P HOLD artifact;
- candidate state keys must equal the frozen model state keys;
- `features.*` must be bit-identical;
- `head.bias` must be bit-identical;
- the only allowed changed state key is `head.weight`;
- `head.weight` must remain exactly 64 parameters;
- frozen thresholds remain unchanged.

Any violation fails closed.

## Prohibited interpretation

This audit must not claim that a new constrained repair is valid or selected. In particular it must not:

- infer a constraint coefficient;
- tune a positive-margin floor;
- tune threshold or bias;
- alter V5/historical domain weights;
- change LBFGS or choose another solver;
- start a second V5-2P configuration;
- train a backbone or residual layer;
- use historical VALIDATION failures to choose model behavior.

The scientific hypothesis that a TRAIN-only constrained head repair may preserve historical-positive margin is allowed to remain a hypothesis only. Any such objective is a new experiment and requires a separate preregistration before gradient execution.

## Safety boundary

- training: **False**
- autograd/backward: **False**
- optimizer steps: **0**
- checkpoint writes/mutations: **False**
- First-30: **closed**
- V5 VALIDATION: **closed**
- FINAL_HOLDOUT: **locked**
- 4-AI: **frozen**
- production promotion: **False**