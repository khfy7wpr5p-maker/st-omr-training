# Meter V5-2E — No-Gradient Pressure Audit V1

## Purpose

V5-2D proved that the V5-2B candidate collapse is not explained by label polarity, missing negatives, deterministic batch omission, or a candidate-only preprocessing path. The next safe step is to quantify the optimization pressure that the frozen source model sees on the already-approved V5 adaptation TRAIN surface before any repair training is designed.

This stage is diagnostic only. It performs no backward pass, no optimizer step, no checkpoint write, no threshold tuning, and no spatial/BBox change.

## Evidence carried forward from V5-2D

The following observations are treated as fixed evidence and must be re-verified before V5-2E interpretation:

- V5 adaptation TRAIN has exactly 540 slots per specialist view: 90 positive and 450 negative for 2-AI and 3-AI.
- Label polarity is manifest-consistent.
- Denominator positive count is zero for 2-AI and 3-AI.
- Every deterministic epoch sees all 540 rows exactly once.
- There are no zero-positive or zero-negative batches.
- `pos_weight=5.0`, giving nominal class-weight totals of 450 positive-weight units and 450 negative-weight units.
- Frozen 2-AI and 3-AI reject all 90 V5 positives at the frozen thresholds while already rejecting all 450 V5 negatives.
- Candidate 2-AI fits V5 TRAIN perfectly; candidate 3-AI nearly perfectly.
- Historical TRAIN negative->positive rates move from 0.007194 -> 0.421119 for 2-AI and from 0.000512 -> 1.0 for 3-AI.
- Parameter drift is not head-only; feature tensors move as well, especially in 3-AI.

## Main question

Does the original V5-only loss create overwhelmingly one-sided optimization pressure from target-domain positives at the frozen checkpoint, such that a small parameter displacement can satisfy V5 TRAIN while leaving historical negatives unconstrained?

V5-2E must measure this question without taking a gradient step.

## Analytical gradient contract

For BCE-with-logits with binary target `y`, logit `z`, probability `p=sigmoid(z)`, and positive weight `w=5.0`, the derivative with respect to the logit is computed analytically only:

- positive (`y=1`): `dL/dz = w * (p - 1)`
- negative (`y=0`): `dL/dz = p`

No autograd graph, `.backward()`, optimizer, or parameter mutation is permitted.

For each specialist and each allowed data surface, report:

- count;
- probability distribution;
- logit distribution;
- signed `dL/dz` distribution;
- absolute `|dL/dz|` distribution;
- sum and mean absolute gradient pressure;
- positive-pressure total;
- negative-pressure total;
- positive/negative pressure ratio.

## Allowed surfaces

Only:

1. the existing 540 V5 `adaptation_train` slots;
2. historical M4A TRAIN replay from the exact D10 pixels;
3. exact frozen 2-AI and 3-AI source checkpoints;
4. exact V5-2B candidate checkpoints for comparison only.

V5 VAL and FINAL_HOLDOUT remain closed. The 900 reserve V5 TRAIN examples remain closed.

## Counterfactual diagnostics

V5-2E may compute loss/gradient-pressure statistics for hypothetical scalar positive weights such as `w=1.0` and the frozen `w=5.0` because no model parameters are updated. These are diagnostics, not authorized training settings.

It may also report what historical TRAIN replay pressure would exist at the frozen source checkpoint. It must not select or authorize a replay ratio automatically.

## Preregistered dominance indicator

For reporting only, V5-2E marks target-positive pressure as "two-orders dominant" when the absolute positive-to-negative pressure ratio is at least `100x`. If negative pressure is exactly zero while positive pressure is nonzero, the indicator is also true and the ratio is represented as a non-finite state rather than written as JSON infinity.

This threshold is a diagnostic evidence flag only. It is not a training gate, does not tune any model threshold, and cannot authorize a repair.

## Interpretation boundary

V5-2E may conclude that one-sided loss pressure is strongly supported if:

- V5 positive pressure at the frozen checkpoint dominates V5 negative pressure by at least two orders of magnitude; and
- historical negative pressure that would resist positive collapse is absent from the V5-only training objective.

It must not claim that `pos_weight=5.0` alone is the unique root cause unless the evidence isolates that effect from the underlying V5/source-domain mismatch. The same pressure audit at the diagnostic counterfactual `w=1.0` is used specifically to test whether the one-sided pressure exists even without the 5x positive scaling.

The conservative root-cause class may be:

`ROOT_CAUSE_CLASS=V5_ONLY_DOMAIN_ADAPTATION_WITH_UNCONSTRAINED_SOURCE_FORGETTING`

while preserving:

`DOMINANT_MECHANISM=UNRESOLVED`

if loss scaling versus representation drift cannot be uniquely separated.

## Safety boundary

V5-2E:

- TRAINING=False
- BACKWARD=False
- OPTIMIZER_STEPS=0
- CHECKPOINT_WRITE=False
- THRESHOLD_TUNING=False
- NEW_BBOX=False
- NEW_CROP_GEOMETRY=False
- NEW_SPATIAL_HEURISTIC=False
- RESERVE_V5_TRAIN=CLOSED
- V5_VAL=CLOSED
- FINAL_HOLDOUT=LOCKED
- 4-AI=FROZEN
- RESOLVER_WIRING=False
- PRODUCTION_PROMOTION=False

A repair-training design remains a separate stage and requires explicit review of V5-2E evidence before any new gradient run.