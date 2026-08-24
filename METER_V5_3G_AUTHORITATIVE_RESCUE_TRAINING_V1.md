# Meter V5-3G — Authoritative Rescue TRAIN Materialization and Single Execution V1

## Decision

V5-3G binds the CI-green V5-3F tensor harness to the exact TRAIN-only data surfaces.
It authorizes exactly one 2-AI rescue fit and one 3-AI rescue fit using the already
preregistered V5-3E recipe. No alternate optimizer, threshold, seed, topology,
training budget, sweep, retry configuration, or automatic second candidate exists.

Exact prerequisite:

- V5-3F HEAD: `7ed41f2872058ac5e3e52df756b9098a1d60052d`
- V5-3F module blob: `908b5b7f83fc5a5358261b7dc04ab606ee66e063`
- V5-3F document blob: `164f84b3dc89230024c8a62ef189204adfe4ebed`

This stage implements the authoritative execution entry, but external execution
must still use a later exact-SHA wrapper pinned to the CI-green V5-3G commit.

## Exact materialization path

V5-3G does not define new pixel or crop semantics. It reuses V5-2N exactly:

1. load the SHA-pinned frozen 2-AI and 3-AI checkpoints;
2. use the already frozen V5 TRAIN slot manifest and 64x64 crops;
3. use the historical M4A TRAIN records and existing D10 historical canvas path;
4. extract the same frozen 64D backbone features;
5. derive each frozen decision at the unchanged threshold;
6. admit only rows where that same specialist is frozen-negative.

No 4-AI checkpoint is loaded.

The four groups remain:

1. `v5_frozen_false_negative_positive`
2. `v5_frozen_true_negative`
3. `historical_frozen_false_negative_positive`
4. `historical_frozen_true_negative`

Authoritative counts must match exactly before optimizer execution:

- 2-AI: `90`, `450`, `14`, `25254`
- 3-AI: `90`, `450`, `12`, `25364`

Any count drift, target drift, feature-dimension drift, non-finite value, or
threshold drift fails closed before rescue optimization.

## Exact training recipe

V5-3G calls the V5-3F harness directly. It does not construct another optimizer.

For each specialist the only trainable state is a newly created rescue model:

`64 -> 8 -> tanh -> 1`

The V5-3E recipe remains unchanged:

- seed `52023`
- Xavier-uniform weights, zero biases
- CPU float32
- AdamW
- learning rate `1e-3`
- weight decay `1e-4`
- betas `(0.9, 0.999)`
- epsilon `1e-8`
- exactly `110` optimizer steps
- full four-group objective every step
- group coefficients `0.25` each
- BCE-with-logits mean inside each group
- global gradient norm clip `1.0`
- no scheduler, warmup, AMP, early stopping, or checkpoint selection
- no threshold search or architecture search

## Frozen-state isolation

Before materialization, V5-3G fingerprints the complete frozen 2-AI and 3-AI
state dictionaries. After each rescue fit and again after both fits, those
fingerprints must remain bit-identical.

The original specialists are never passed to the rescue optimizer. Only the
new rescue model returned by V5-3F is trainable.

4-AI is not loaded and remains frozen.

## Rescue artifact boundary

After both rescue fits succeed and frozen-state isolation passes, V5-3G writes
two separate rescue artifacts into a fresh temporary directory and atomically
renames that directory into place.

Each artifact contains only:

- metadata;
- the rescue `state_dict`.

It does not contain or replace any original specialist checkpoint.

Artifact metadata binds:

- V5-3F exact HEAD;
- recipe identity;
- source frozen-checkpoint SHA;
- V5 slot-manifest SHA;
- frozen and rescue thresholds;
- exact group counts;
- SHA256 fingerprints of the four materialized feature groups;
- exact 110-step execution evidence;
- rescue state fingerprint;
- protected-surface lock state.

Artifacts are reloaded with `weights_only=True` and their state fingerprints
are independently reverified before the authoritative report may be written.

The execution is one-shot and non-overwriting. Existing report, final artifact
directory, or temporary artifact directory blocks rerun.

## Evidence emitted

The authoritative report records:

- exact prerequisite identity;
- source frozen checkpoint hashes;
- V5 slot-manifest hash;
- aggregate group counts and feature fingerprints;
- numerical execution evidence from V5-3F;
- frozen-state before/after fingerprints;
- rescue artifact hashes and reload verification;
- numerical integrity PASS;
- frozen-state isolation PASS.

It intentionally does **not** run the next performance gate.

## Still closed

V5-3G does not open or execute:

- TRAIN candidate F1/retention acceptance gate;
- historical validation retention;
- immutable First-30;
- V5 reserve;
- V5 validation;
- FINAL_HOLDOUT;
- BBox changes;
- crop geometry changes;
- spatial heuristic changes;
- threshold tuning;
- runtime/resolver wiring;
- production promotion.

## Gate order after implementation

1. exact V5-3G CI-green SHA;
2. create an exact-SHA external execution wrapper;
3. perform the single authorized TRAIN execution and bind its receipt;
4. evaluate TRAIN V5 F1 and frozen-correct historical TRAIN retention;
5. historical validation retention at unchanged thresholds;
6. immutable First-30 diagnostic;
7. separately authorized V5 validation;
8. separately authorized FINAL_HOLDOUT.

A HOLD does not authorize retraining or a second configuration.
