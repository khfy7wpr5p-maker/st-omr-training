# Meter V5-3F — Single Fixed Rescue Training Execution Harness V1

## Decision

V5-3E preregistered exactly one rescue-training recipe. V5-3F implements only
that recipe as a narrow in-memory tensor execution harness.

This stage does **not** run the authoritative V5/historical TRAIN execution.
It also does not create a Colab wrapper, discover datasets, read checkpoints,
write checkpoints, open retention/First-30/V5 validation/FINAL_HOLDOUT, change
thresholds, wire runtime rescue decisions, or promote production state.

V5-3F is bound to the exact CI-green V5-3E HEAD:

`f27d2334d9dfbdd8c6c70d3e214573765cee15c6`

and to the exact V5-3E preregistration module/document blobs.

## Implemented training surface

The harness constructs a fresh rescue specialist only:

`64D frozen feature -> Linear(64, 8) -> tanh -> Linear(8, 1)`

The parameter count is exactly `529` per rescue.

Only 2-AI and 3-AI rescue specialists are valid. The harness does not accept a
frozen 2-AI, 3-AI, or 4-AI model object by reference, so it has no code path to
mutate original backbone/head tensors. 4-AI remains frozen.

The model initialization is reset to seed `52023` for every specialist so 2-AI
and 3-AI start from the same parameter realization. Linear weights use
Xavier-uniform initialization with gain `1.0`; biases are zero.

## Exact tensor input contract

The harness accepts only a mapping containing the four V5-3E groups, in the
preregistered order:

1. `v5_frozen_false_negative_positive`
2. `v5_frozen_true_negative`
3. `historical_frozen_false_negative_positive`
4. `historical_frozen_true_negative`

Each group must be a finite CPU-convertible float32 tensor of shape `[N, 64]`.
Targets are not supplied by the caller: they are derived from the frozen group
identity, so positive/negative labels cannot be silently changed.

For a later authoritative execution, exact preregistered counts must be
enforced:

- 2-AI: `90`, `450`, `14`, `25254`
- 3-AI: `90`, `450`, `12`, `25364`

CI tests may use smaller synthetic tensors only by explicitly disabling the
authoritative count guard. Such runs are test evidence only and are never
labelled as authoritative dataset execution.

## Exact optimizer implementation

The implementation follows V5-3E without modification:

- optimizer: AdamW
- learning rate: `1e-3`
- weight decay: `1e-4`
- betas: `(0.9, 0.999)`
- epsilon: `1e-8`
- optimizer steps: exactly `110`
- full four-group objective at every step
- BCE-with-logits mean inside each group
- group coefficients: `0.25`, `0.25`, `0.25`, `0.25`
- global gradient-norm clip: `1.0`
- CPU float32
- deterministic algorithms enabled
- no AMP
- no scheduler
- no warmup
- no early stopping
- no checkpoint selection
- no sweep
- no threshold search
- no automatic second configuration
- no fallback optimizer

## Numerical fail-closed guards

The tensor harness aborts on:

- non-finite input features
- non-finite initialized rescue parameters
- non-finite logits
- non-finite group or total loss
- missing gradients
- non-finite gradients
- non-finite gradient norm
- non-finite post-step rescue parameters
- wrong group surface/order
- wrong feature dimension
- wrong specialist
- wrong approval token
- authoritative count mismatch when the exact-count guard is enabled

An abort writes no checkpoint because V5-3F contains no persistence path.

## State isolation

State isolation is structural in this stage:

- frozen models are not accepted by reference
- no checkpoint is loaded
- no `load_state_dict`/parameter copy-back path exists
- input feature tensors are detached, cloned, and never mutated
- only the newly created rescue model is passed to AdamW
- 4-AI is never constructed or opened
- runtime frozen thresholds remain unchanged
- production authority remains frozen-specialist-only

CI verifies deterministic initialization, deterministic synthetic 110-step
execution, input tensor immutability, equal group weighting, duplication
invariance, non-finite rejection, and the absence of dataset/checkpoint mutation
paths.

## Protected surfaces remain closed

V5-3F does not open or use:

- historical validation
- immutable First-30
- V5 reserve
- V5 validation
- FINAL_HOLDOUT
- BBoxes or new crop geometry
- new spatial heuristics
- threshold tuning
- resolver/runtime wiring
- production promotion

## Mandatory next gate order

1. V5-3F exact CI-green SHA.
2. Separately authorized exact TRAIN tensor materialization and **one** execution
   through the already-fixed V5-3F harness.
3. Verify numerical evidence and state isolation.
4. TRAIN gate: V5 F1 `1.0`; no frozen-correct historical TRAIN example becomes
   wrong; frozen tensors remain bit-identical; only rescue parameters changed.
5. Historical validation retention at unchanged frozen thresholds. HOLD stops.
6. Immutable First-30 only after retention PASS. HOLD stops.
7. V5 validation only under separate authorization.
8. FINAL_HOLDOUT only under later separate authorization.

A HOLD never authorizes another optimizer, threshold, architecture, seed,
training budget, sweep, or fallback path.

## Current safety state

- V5-3F tensor training implementation: present
- authoritative dataset execution: not performed
- Colab execution wrapper: absent
- checkpoint load/write: absent
- rescue artifact write: absent
- frozen-model mutation surface: absent
- CI execution: synthetic tensors only
- retention: not run
- First-30: closed
- V5 reserve: closed
- V5 validation: closed
- FINAL_HOLDOUT: locked
- 4-AI: frozen
- runtime rescue authority: disabled
- production promotion: false
