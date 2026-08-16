# Stage 7-D13-R2.2A — Rest balanced-loss diagnostic

Status: **implementation candidate — external TRAIN-only diagnostic pending**.

## Evidence that selects this experiment

D13-R2 Rest diagnosis has now separated three failure classes without opening TEST.

### R2-1A geometry audit

The frozen D13-R1 derivative passed its identity and geometry audit.  Observed Rest targets did not show a broad stride-4 collapse:

- zero Rest targets with minimum dimension below 4 px in TRAIN or VALIDATION;
- only a small minority below two output strides (8 px);
- zero stride-4 center-cell collision records.

The evidence therefore did not justify changing the 512×128 measure representation or output stride as the first recovery action.

### R2-1B decoder threshold audit

The exact R1 Rest epoch-10 checkpoint was reproduced under the pinned runtime.  A diagnostic threshold sweep produced:

```text
threshold  detections  max score   center F1  bbox F1  macro F1
0.01       56          0.023700    0          0        0
0.05       0           0           0          0        0
0.10       0           0           0          0        0
0.15       0           0           0          0        0
0.20       0           0           0          0        0
0.25       0           0           0          0        0
```

The failure is therefore not explained by the frozen production threshold of 0.25.

### R2-2 loss-collapse audit

On the 1,224 frozen VALIDATION records the R1 Rest heatmap contained 969 positive cells and 15,039,543 negative cells: roughly 15,520 negatives per positive.

Before training:

```text
GT confidence mean          0.5093255193
background confidence mean  0.5069914783
negative / positive loss    17314.6739
```

After epoch 10:

```text
GT confidence mean          6.925282659e-12
GT confidence median        7.739352745e-15
image max confidence        0.0081499204
background confidence mean  8.313175794e-06
```

This is direct evidence that R1 training drove the Rest heatmap toward an almost-all-zero solution.  The R1 heatmap objective sums all negative focal terms and divides the combined heatmap loss only by positive count.  R2 therefore tests loss normalization before adding model complexity.

## Single-variable experiment

R2-2A keeps the following R1 choices unchanged:

```text
model             compact stride-4 center detector
input             1×128×512 grayscale measure
Rest classes      half | quarter | eighth
batch size        16
optimizer         AdamW
learning rate     0.0007 fixed
weight decay      0.0001
grad clip         1.0
focal gamma       2.0
positive weights  exact R1 TRAIN-only inverse-sqrt weights
decoder threshold 0.25
scheduler         none
```

The only intended objective change is the heatmap normalization:

```text
R1:
(sum positive focal + sum negative focal) / positive_count

R2-2A:
mean weighted-positive focal
+ 1.0 * mean negative focal
```

BBox-size and center-offset Smooth-L1 terms remain the R1 terms in the external diagnostic runner.

Zero-Rest measures are retained.  An all-negative batch receives a graph-safe zero positive term plus the mean negative focal term.

## Frozen TRAIN-only diagnostic budget

Authoritative VALIDATION is not used to iterate on this candidate.  The diagnostic partition is selected only from TRAIN records by salted SHA-256 rank of `record_id`; label content and model outcomes do not affect partition membership.

```text
optimization records   2048 TRAIN-only
diagnostic eval records 512 TRAIN-only
epochs                     3
batch size                 16
max optimizer steps       384
VALIDATION optimizer use    0
TEST opened              false
```

The optimization and diagnostic-evaluation sets are disjoint.  Encountering a TEST manifest row fails closed.

## Diagnostic evidence to persist

Evaluate before optimization and after each diagnostic epoch:

- GT Rest heatmap confidence mean/median;
- background heatmap confidence mean;
- GT-background confidence separation;
- prediction count at the frozen 0.25 decoder threshold;
- class-aware center F1 at 4 px;
- class-aware bbox F1 at IoU 0.50;
- macro Rest class F1;
- per-class prediction/target counts;
- balanced positive and negative heatmap terms;
- exact optimizer-step count.

The short run is a failure-attribution gate, not a production checkpoint-selection event.  Its model state must not be promoted to the final Rest specialist merely because the diagnostic improves.

## Decision after the external run

If the balanced objective produces clear GT/background separation and non-degenerate Rest localization, R2 keeps the current visual architecture for the next controlled training experiment and then evaluates scheduler/checkpoint policy separately.

If the heatmap still collapses or localization remains unusable despite healthy confidence separation, the next root-cause branch may inspect regression coupling or admit the previously approved detector→type-classifier decomposition.

No result from R2-2A changes the sealed TEST boundary.
