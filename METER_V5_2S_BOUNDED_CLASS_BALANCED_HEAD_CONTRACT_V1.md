# Meter V5-2S — Bounded Class-Balanced Head Repair Contract V1

## Decision

V5-2S preregisters one later repair objective. This stage does not train a
model, write a checkpoint, authorize Colab gradients, or open any validation
surface.

V5-2P remains rejected. Its candidates are evidence artifacts only.

## Exact prerequisite evidence

The contract is bound to the completed V5-2R execution:

- implementation HEAD: `85c0b0083792e8b9ec60ee632cfc7015e885d548`;
- report SHA256: `2c374189c285232eb79c7a8ca331d9a53b60286b36dde18d5dec559b14f58dc7`;
- execution-envelope SHA256: `6a80aa1536722720f6a3a85d93d363e356830eb6ff711d0b41c4af5c45080226`.

V5-2R showed that equal domain weighting was not class balancing. Approximately
89 percent of the empirical V5-2P objective coefficient came from negatives.
The resulting 2-AI and 3-AI heads grew by about 107x and 175x and rotated by
about 86 degrees. Positive gradients agreed across domains; positive and
negative gradients strongly opposed one another.

The new contract therefore addresses both observed mechanisms: group imbalance
and uncontrolled head movement.

## Fixed four-group objective

Only the already-open V5 TRAIN and historical TRAIN frozen features are used.
For each specialist, split them into exactly four non-empty groups:

1. V5 positive;
2. V5 negative;
3. historical positive;
4. historical negative.

Each group receives exactly `0.25` of the BCE objective:

`L_balanced = 0.25 BCE(V5+) + 0.25 BCE(V5-) + 0.25 BCE(HIST+) + 0.25 BCE(HIST-)`

This keeps the domains at 0.5/0.5 while making positive and negative influence
0.5/0.5. Raw record counts cannot silently change these coefficients.

## Frozen-centred proximal bound

Let `w0` be the exact frozen 64D head weight. The maximum policy angle is fixed
before training at 15 degrees:

`R = sin(15 degrees) * ||w0||2`

Let `L0 = L_balanced(w0)`. The proximal coefficient is derived once from TRAIN
only, without a search:

`lambda = 2 * L0 / R^2`

The complete objective is:

`L = L_balanced + 0.5 * lambda * ||w - w0||2^2`

At distance `R`, the proximal penalty alone equals `L0`. BCE is non-negative.
Therefore any finite final state whose total objective does not exceed the
initial objective cannot lie outside `R`. The ball also implies a maximum
15-degree direction change and confines the candidate norm ratio to
approximately `0.7412..1.2588`. This directly excludes the 107x/175x V5-2P
failure mode.

The coefficient is analytic. There is no lambda sweep, angle sweep, validation
selection, second configuration, or fallback solver.

## Frozen runtime contract

If a later exact-SHA stage separately authorizes one execution:

- only the 64 values of `head.weight` may change;
- `features.*` and `head.bias` must remain bit-identical;
- thresholds remain exactly 2-AI `0.48`, 3-AI `0.60`, and 4-AI `0.47`;
- 4-AI remains frozen;
- no BBox, crop, spatial, resolver, or runtime-routing change is allowed;
- the existing deterministic full-batch LBFGS configuration is carried forward;
- only one final solver state may be produced.

This contract stage contains no optimizer and grants no training authority.

## Gate order

1. Numerical integrity and geometry at the single final candidate.
2. Historical retention at unchanged thresholds.
3. Immutable V5 First-30 only after retention PASS.
4. V5 validation only under a separate exact-SHA authorization.
5. FINAL_HOLDOUT only under a later separate authorization.

Any non-finite value, increased objective, changed frozen tensor, geometry HOLD,
or historical retention HOLD stops the experiment. No automatic second attempt
is permitted.

## Closed surfaces

Historical validation examples are not used to choose this objective. First-30,
V5 validation, the reserve TRAIN set, and FINAL_HOLDOUT remain closed. Production
promotion remains false.
