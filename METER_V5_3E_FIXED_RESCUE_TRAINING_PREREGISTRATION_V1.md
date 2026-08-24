# Meter V5-3E — Fixed Rescue-Training Preregistration V1

## Decision

V5-3D fixed the rescue architecture and authority boundary. V5-3E now freezes
exactly one training recipe for that architecture before any gradient is run.
This stage remains review-only: it does not construct or train a model, write a
checkpoint, run retention, open First-30/V5 validation, or inspect
FINAL_HOLDOUT.

V5-3E is bound to V5-3D exact HEAD
`7d50fbec4d730aa46c69f7dfa3a20917a3478ef8`. The inherited architecture is
unchanged:

`frozen 64D feature -> Linear(64, 8) -> tanh -> Linear(8, 1) -> sigmoid`

Only independent 2-AI and 3-AI rescue parameters may ever become trainable.
The original backbone, original head weight/bias, all frozen thresholds, and
4-AI remain frozen. The rescue threshold remains exactly `0.50`.

## Exact TRAIN objective

The eligible rows and counts are inherited unchanged from V5-3D. Only rows on
which the same specialist's frozen decision is negative are eligible, from the
already-open V5 TRAIN and historical TRAIN surfaces.

For each specialist the objective contains exactly four groups:

1. V5 frozen false-negative positives;
2. V5 frozen true negatives;
3. historical frozen false-negative positives;
4. historical frozen true negatives.

Exact counts remain:

- 2-AI: `90`, `450`, `14`, `25254`;
- 3-AI: `90`, `450`, `12`, `25364`.

Every optimizer step uses the complete four-group objective. Binary
cross-entropy with logits is reduced to a mean inside each group, then the four
means are summed with fixed weights `0.25` each. This prevents record count from
giving the large historical-negative group extra authority. `pos_weight=1.0`
and label smoothing is `0.0`.

Rows must be consumed in canonical stable-manifest identity order. There is no
shuffle, resampling, validation-driven sampling, or batch-composition search.

## One fixed optimization recipe

The recipe is selected a priori. There is exactly one configuration:

- initialization: Xavier-uniform weights, gain `1.0`, zero biases;
- deterministic seed: `52023`;
- the seed is reset so 2-AI and 3-AI start from the same parameter realization;
- device: CPU;
- dtype: float32;
- deterministic algorithms: required;
- AMP: disabled;
- optimizer: AdamW;
- learning rate: `1e-3`;
- weight decay: `1e-4`;
- betas: `(0.9, 0.999)`;
- epsilon: `1e-8`;
- optimizer steps: exactly `110`;
- objective per step: complete four-group weighted objective;
- gradient global-norm clip: `1.0`;
- scheduler: none;
- warmup: none;
- early stopping: forbidden;
- selected candidate: fixed final step only.

The `110`-step budget retains the previously preregistered V5 repair optimizer
step budget as a conservative fixed compute budget, while the new rescue
parameters start from an explicit independent initialization. No result from
V5 validation, First-30, historical validation, reserve data, or FINAL_HOLDOUT
was used to choose these values.

There is no hyperparameter sweep, threshold search, architecture search,
automatic second configuration, alternate optimizer, fallback solver, or
best-checkpoint search.

## Numerical and state-isolation requirements for the later execution stage

A later, separately authorized execution harness must fail closed if any input
feature, initialized parameter, loss, gradient, or post-step rescue parameter
is non-finite. Gradient global norm is clipped at `1.0`.

Before and after the one fixed training execution, all original frozen
backbone/head tensors must be bit-identical. Only the separate rescue namespace
may change. Any abort writes no checkpoint.

These are requirements for the later execution harness; V5-3E itself executes
zero optimizer steps.

## Protected surfaces remain closed

V5-3E does not open or use:

- historical validation;
- immutable First-30;
- V5 reserve;
- V5 validation;
- FINAL_HOLDOUT;
- BBoxes or new crop geometry;
- new spatial heuristics;
- threshold tuning;
- resolver/runtime wiring;
- production promotion.

## Mandatory next gate order

1. V5-3E exact CI-green SHA.
2. Separate single-fixed-recipe execution harness.
3. One candidate execution with numerical and state-isolation checks.
4. TRAIN gate: V5 F1 `1.0`, no frozen-correct historical example becomes
   wrong, frozen tensors bit-identical, only rescue parameters changed.
5. Historical validation retention at unchanged frozen thresholds. HOLD stops.
6. Immutable First-30 only after retention PASS. HOLD stops.
7. V5 validation only under separate authorization.
8. FINAL_HOLDOUT only under later separate authorization.

A failure is HOLD. It does not authorize a second recipe, threshold adjustment,
sweep, fallback optimizer, or production promotion.

## Current safety state

- training implementation: absent;
- training authorization: false;
- training executed: false;
- optimizer steps executed: `0`;
- checkpoint/rescue artifact write: false;
- retention: not run;
- First-30: closed;
- V5 reserve: closed;
- V5 validation: closed;
- FINAL_HOLDOUT: locked;
- 4-AI: frozen;
- runtime rescue authority: disabled;
- production promotion: false.
