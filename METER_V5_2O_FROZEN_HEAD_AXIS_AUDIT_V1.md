# Meter V5-2O — Frozen Head-Axis Audit V1

## Purpose

V5-2N established two different transfer regimes on the exact frozen 64D digit-specialist representations:

- 2-AI: V5 classes are almost perfectly assigned to the corresponding historical binary centroids even though the frozen head predicts no V5 positives.
- 3-AI: the historical and V5 class-separation vectors remain strongly directionally aligned, but V5 positives sit on the historical-negative side of the source-centroid geometry and the V5 class-separation magnitude is compressed.

Before any repair topology or training recipe is selected, V5-2O asks a narrower read-only question:

> Does the existing frozen linear-head direction still rank/separate the V5 positive and negative TRAIN slots, and how much source↔V5 midpoint translation/compression occurs along that already-existing head axis?

This distinguishes a primarily intercept/domain-offset problem from a failure of the frozen head direction itself. V5-2O does not fit a classifier, select a bias, tune a threshold, select a residual topology, or authorize training.

## Exact surfaces

Read-only surfaces only:

- the existing 540 V5 `adaptation_train` slots from the approved V5-2B slot manifest;
- historical M4A TRAIN, exactly 26,964 rows;
- the exact frozen 2-AI and 3-AI checkpoints;
- the exact V5-2N frozen feature extraction path;
- historical pixels only through the already-frozen M4A/D10 crop/preprocess helper;
- V5 pixels only from the already-existing approved 64x64 slot crops.

The 900 reserved V5 TRAIN examples remain closed. V5 validation and FINAL_HOLDOUT remain closed. 4-AI remains frozen.

## Frozen-head score contract

For each specialist independently, the existing frozen head is used exactly as stored:

`logit = frozen_head_weight · frozen_64D_feature + frozen_head_bias`

The frozen probability threshold is converted once to its equivalent logit decision boundary only for descriptive comparison:

`boundary_logit = log(threshold / (1 - threshold))`

No alternative threshold or alternative bias is evaluated or selected.

## Preregistered descriptive metrics

For both historical M4A TRAIN and V5 adaptation TRAIN, report:

1. positive and negative frozen-logit distributions;
2. positive-vs-negative rank AUC under the unchanged frozen head direction;
3. strict separation gap `min(positive_logit) - max(negative_logit)`;
4. positive and negative means and their logit gap;
5. the binary-class centroid midpoint projected onto the normalized frozen-head axis;
6. source↔V5 midpoint shift along the frozen-head axis;
7. source and V5 class-separation magnitude along the frozen-head axis;
8. V5/source class-gap ratio along that axis;
9. source↔V5 midpoint-shift magnitude divided by the V5 class-gap magnitude when finite;
10. distances of positive/negative logit extremes from the unchanged frozen decision boundary;
11. whether all V5 positives and/or all V5 negatives lie below/above the unchanged frozen boundary.

These values are descriptive. No numeric PASS threshold is preregistered.

## Interpretation boundary

Allowed interpretation:

- If V5 rank AUC is high and the V5 strict separation gap is positive, the existing frozen head direction contains useful V5 class ordering. A large midpoint shift relative to the V5 class gap is then evidence that absolute domain placement/intercept is a major part of the failure.
- If V5 rank AUC is weak or the strict separation gap is non-positive, a scalar intercept-only explanation is insufficient and future repair may need head rotation and/or a larger frozen-base residual.

Not allowed:

- selecting a bias value;
- changing the frozen runtime threshold;
- fitting a new classifier;
- selecting a residual topology;
- training a repair candidate;
- opening validation/final/reserve surfaces.

## Safety boundary

V5-2O authorizes no:

- training;
- autograd gradient computation;
- backward pass;
- optimizer step;
- checkpoint write;
- runtime threshold tuning;
- bias parameter selection;
- new BBox;
- new crop geometry;
- new spatial heuristic;
- old D11 glyph/window reuse;
- 900 reserve TRAIN opening;
- V5 validation opening;
- FINAL_HOLDOUT opening;
- 4-AI mutation;
- Resolver wiring;
- production promotion.

The only output is a V5-2O JSON evidence report under the existing annotations directory. Existing V5-2O evidence must never be silently overwritten.
