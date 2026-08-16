# Stage 7-D13-R2.3A — Rest exact-cell peak alignment

Status: **implementation candidate — external TRAIN-only A/B diagnostic pending**.

## Why this experiment exists

R2-2A corrected the D13-R1 Rest all-zero heatmap collapse by replacing global
negative accumulation with separately normalized positive and negative focal
terms. The resulting detector learned a strong Rest signal, but localization
remained poor because the heatmap maximum often moved away from the exact cell
whose regression outputs were supervised.

The external TRAIN-only diagnostic chain established:

```text
R2-2A epoch 3
GT confidence mean                 0.844445
background confidence mean         0.149145
GT-background separation           0.695300

R2-2C
same-class <=4 px candidate       391 / 563  (69.45%)
same-class <=8 px candidate       540 / 563  (95.91%)
same-class <=12 px candidate      562 / 563  (99.82%)
same-class <=16 px candidate      563 / 563 (100.00%)

R2-2D missed @4 px                172
  same-class peak 4-8 px          149
  same-class peak 8-12 px          22
  same-class peak 12-16 px          1
  wrong-class candidate <=4 px      0

R2-2E exact GT-cell regression
center error <=4 px               563 / 563 (100.00%)
bbox IoU >=0.50                   554 / 563  (98.40%)
exact GT cell is local max        198 / 563  (35.17%)

R2-2E missed-target peak-cell shift
same GT cell                         0
one-cell shift                     109
two-cell shift                      61
three-plus-cell shift                2
misses representable by [0,1) offset 0 / 172
```

The evidence therefore selects peak/regression-cell alignment as the next single
variable. It does not support changing Rest type classes, the center-offset
activation, bbox representation, scheduler, input scale, or model decomposition
before this narrower hypothesis is tested.

## Frozen R2-3A change

R2-3A starts again from the exact R1 epoch-0 Rest model and keeps the R2-2A
training surface unchanged:

```text
model                 compact stride-4 detector
input                 1x128x512 grayscale measure
Rest classes          half | quarter | eighth
TRAIN optimization    same deterministic 2048-record R2-2A partition
TRAIN diagnostic eval same deterministic 512-record R2-2A partition
epochs                 3
batch size            16
max optimizer steps   384
optimizer             AdamW
learning rate         0.0007 fixed
weight decay          0.0001
grad clip             1.0
scheduler             none
balanced focal        unchanged from R2-2A
bbox Smooth-L1        unchanged
center-offset Smooth-L1 unchanged
TEST                   sealed
production promotion   false
```

The only added objective term is:

```text
+ 1.0 * local_gt_cell_peak_alignment_loss
```

For each exact positive Rest heatmap cell, the loss compares that logit against
same-class non-target cells in a radius-2 / 5x5 window. Other true positive cells
are excluded from the competitor set. Cross-class competition is intentionally
not introduced because R2-2D found zero wrong-class candidates within 4 px among
the 172 misses.

Radius 2 is evidence-bound rather than tuned: 170 of 172 localization misses
were one or two output cells away from the exact regression-supervised cell.

## A/B interpretation

R2-3A is compared with the already completed R2-2A run, not trained from the
R2-2A epoch-3 state. Both start from the same epoch-0 model identity.

Primary evidence:

- exact GT-cell local-max rate;
- class-aware <=4 px candidate coverage / center F1;
- bbox F1;
- macro class F1;
- GT/background confidence separation;
- prediction count at the unchanged 0.25 decoder threshold;
- exact GT-cell center regression and bbox quality must not regress materially;
- optimizer steps must equal 384;
- TEST remains unopened.

A useful result should move the heatmap peak toward the regression-supervised GT
cell while preserving the already healthy regression heads. This short run is a
failure-attribution experiment, not a final Rest checkpoint-selection event.

## Decision boundary

If exact-cell local-max and <=4 px localization improve substantially without
collapsing confidence or regression, retain the current Rest detector and proceed
to a separate precision/hard-negative experiment before scheduler A/B.

If peak alignment does not materially improve localization, revisit the heatmap
target/decoder contract or a broader localization representation before adding
scheduler or splitting the specialist.

No R2-3A result changes the sealed TEST boundary automatically.
