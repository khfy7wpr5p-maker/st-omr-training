# Meter V5-3I — Read-only TRAIN Acceptance Gate V1

## Decision

V5-3I is the first acceptance stage after the completed single V5-3G authoritative rescue TRAIN execution and the V5-3H execution receipt.

It is **evaluation only**. It does not train, fine-tune, retry, sweep, alter thresholds, replace frozen specialists, or open any protected validation surface.

## Bound completed execution evidence

V5-3I is bound to the already-completed execution evidence:

- V5-3G executable HEAD: `b36a9d2f5daade2c3568cac8cbc736ca75ca435f`
- V5-3H wrapper HEAD: `aa426442efdef97e3323906096087dabffa1171b`
- V5-3G report SHA-256: `682c2d405287051fef18b803e2597777cb7fc55c6ba0814ea3b2d4df0fa35b9d`
- V5-3H envelope SHA-256: `f41b0fddb9d139018e0ddd16c9765d9415031e6308efd67e16aef3a05d205bf7`
- 2-AI rescue artifact SHA-256: `a27cef8d4ff89565cfe4a15e0e429a21e60daa2656324ed0380fde8674a022e6`
- 3-AI rescue artifact SHA-256: `b8a4f379c33d3aa0df77b54821996a799251abb0e7cbd8de9764b09c5efd3d65`

The gate fails closed if any of these bytes or receipt fields change.

## Read-only data surfaces

V5-3I may read only:

1. the existing V5 `adaptation_train` surface;
2. the historical M4A TRAIN surface through the unchanged D10 rendering/crop path;
3. the exact frozen 2-AI and 3-AI checkpoints;
4. the exact two V5-3G rescue artifacts;
5. the V5-3G report and V5-3H envelope.

It does **not** read:

- historical validation;
- immutable V5 First-30;
- V5 reserve;
- V5 validation;
- FINAL_HOLDOUT.

4-AI is not loaded.

## Fixed inference rule

For each 2-AI/3-AI specialist:

1. compute the unchanged frozen specialist probability;
2. if the frozen probability is at or above its unchanged threshold, keep the frozen decision;
3. only when the same specialist is frozen-negative, evaluate its rescue model;
4. use the frozen rescue threshold `0.50`;
5. do not alter any threshold after observing results.

Frozen thresholds remain:

- 2-AI: `0.48`
- 3-AI: `0.60`

This exactly preserves the rescue topology: the rescue can repair frozen-negative rows but cannot override a frozen-positive decision.

## Preregistered acceptance criteria

V5-3I returns `PASS` only if all conditions hold.

### V5 TRAIN exact recovery

For **both** 2-AI and 3-AI:

- combined V5 TRAIN F1 = `1.0`;
- V5 TRAIN false positives = `0`;
- V5 TRAIN false negatives = `0`.

### Frozen-correct retention

For **both** specialists:

- V5 TRAIN frozen-correct regression count = `0`;
- historical TRAIN frozen-correct regression count = `0`.

Historical TRAIN is used here only as an already-authorized TRAIN retention surface. This is **not** historical validation.

### State / namespace isolation

- original frozen specialist state is bit-identical before/after the gate;
- rescue artifact state is bit-identical before/after the gate;
- V5-3G receipt still proves original frozen state was bit-identical across training;
- artifact metadata still says `trainable_surface = new-rescue-parameters-only`;
- only the separate rescue namespace is accepted as changed by the completed training.

## PASS versus HOLD

`PASS` authorizes only the next separately staged gate:

`historical validation retention`

`HOLD` does **not** authorize:

- retraining;
- a second configuration;
- threshold changes;
- a different optimizer;
- more optimizer steps;
- opening historical validation;
- opening First-30;
- opening V5 validation;
- opening FINAL_HOLDOUT.

Any remediation after HOLD requires a new explicit architecture decision and a separately preregistered stage.

## Evidence output

The external V5-3I wrapper may write one atomic gate report:

`v5_3i_train_acceptance_gate_v1.json`

That report is evidence only. Dataset bytes, frozen checkpoints and rescue artifacts are read-only.

## Future order

1. V5-3I read-only TRAIN acceptance;
2. separately staged historical validation retention;
3. immutable V5 First-30 diagnostic;
4. separately authorized V5 validation;
5. separately authorized FINAL_HOLDOUT.
