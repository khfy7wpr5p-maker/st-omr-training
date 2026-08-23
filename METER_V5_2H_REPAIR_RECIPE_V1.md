# Meter V5-2H — Exact Repair Recipe V1

## Purpose

V5-2F showed that the shared one-pass source/V5 zero-crossing floor at `pos_weight=1.0` is `9.468108115675737`, while a full historical M4A TRAIN pass corresponds to `49.93333333333333` source examples per V5 example. V5-2H selects one conservative, reproducible repair pilot recipe without executing gradients.

## Selected replay ratio

The replay ratio is selected by a fixed rule, not a sweep:

`ceil(1.25 * max(shared pos_weight=1 zero-crossing))`

`ceil(1.25 * 9.468108115675737) = 12`

Therefore:

- source/V5 replay ratio = `12:1`;
- V5 adaptation TRAIN slots = `540`;
- historical M4A TRAIN replay examples = `6480`;
- combined examples in the single pilot epoch = `7020`.

The 25% safety margin is applied once to the cross-specialist analytical floor. No ratio sweep is authorized.

## Historical replay sampling

Historical replay is sampled without replacement from M4A TRAIN with deterministic stratification that preserves the frozen four-label distribution as closely as integer counts allow (Hamilton/largest-remainder allocation):

- label `2`: `367`;
- label `3`: `381`;
- label `4`: `1537`;
- label `NONE`: `4195`;
- total: `6480`.

The same fixed source replay manifest is used for 2-AI and 3-AI. V5 uses all existing `540` adaptation_train slots exactly once. The first-30 V5 diagnostic seed surface remains zero-gradient.

## Optimizer recipe

To isolate the repair objective from unnecessary optimizer changes, the prior optimizer family and hyperparameters are retained while the objective/data mixture changes:

- initialization: exact frozen source checkpoint for each specialist;
- specialists trained: 2-AI and 3-AI only;
- 4-AI: frozen;
- positive weight: `1.0`;
- optimizer: AdamW;
- learning rate: `1e-4`;
- weight decay: `1e-4`;
- batch size: `64`;
- epoch count: `1` combined epoch;
- deterministic seed: `52023`;
- shuffle: deterministic combined-manifest shuffle;
- checkpoint selection: fixed final epoch only;
- threshold sweep/tuning: forbidden.

A single combined epoch gives `ceil(7020/64) = 110` optimizer steps, intentionally close to the previous 12-epoch V5-only run (`12 * ceil(540/64) = 108` steps). This reduces optimizer-step count as a confound while changing the domain-retention objective.

## Evaluation order

No V5 validation or FINAL_HOLDOUT is opened.

After the fixed pilot candidate is produced, gates are evaluated in this order:

1. Historical M4A VALIDATION retention gate at unchanged thresholds.
   - exact frozen baseline reproduction must succeed first;
   - absolute F1 drop <= `0.005`;
   - absolute recall drop <= `0.005`;
   - precision >= `0.98`;
   - recall >= `0.98`;
   - probabilities finite and in `[0,1]`.
2. Existing first-30 V5 zero-gradient diagnostic gate.
   - 2/4 >= `8/10`;
   - 3/4 >= `8/10`;
   - 4/4 >= `9/10` using the unchanged frozen 4-AI;
   - denominator exact-4 >= `26/30`.

Any gate failure is HOLD. No threshold tuning or second repair configuration follows automatically.

## Closed surfaces

This preregistration does not authorize gradient execution yet. It does not open or alter:

- 900 reserved V5 TRAIN examples;
- V5 validation;
- FINAL_HOLDOUT;
- BBoxes;
- crop geometry;
- spatial heuristics;
- thresholds;
- 4-AI;
- Resolver wiring;
- production authority.

## Safety state

- RECIPE_SELECTED=True
- POSITIVE_WEIGHT_SELECTED=1.0
- REPLAY_RATIO_SELECTED=12
- HISTORICAL_REPLAY_COUNT=6480
- EPOCHS_SELECTED=1
- BATCH_SIZE_SELECTED=64
- LEARNING_RATE_SELECTED=1e-4
- WEIGHT_DECAY_SELECTED=1e-4
- SEED_SELECTED=52023
- REPAIR_TRAINING_AUTHORIZED=False
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
