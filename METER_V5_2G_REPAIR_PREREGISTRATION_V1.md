# Meter V5-2G — Repair Training Preregistration Boundary V1

## Purpose

V5-2F established an analytical replay-balance boundary at the frozen source checkpoints. V5-2G freezes what that evidence does and does not authorize before any new gradient run.

This stage is declarative only. It does not train, choose a replay ratio, choose a positive weight, choose a sampling strategy, mutate a checkpoint, open V5 validation, or change any spatial/BBox rule.

## Evidence carried forward

Full historical M4A TRAIN replay corresponds to a source/V5 ratio of `49.93333333333333` for the existing 540 V5 adaptation TRAIN slots.

At diagnostic `pos_weight=1.0`:

- 2-AI zero crossing: `7.289268597494998` source examples per V5 example;
- 3-AI zero crossing: `9.468108115675737` source examples per V5 example;
- both zero crossings are inside one complete historical TRAIN pass.

At frozen V5-2B `pos_weight=5.0`:

- 2-AI zero crossing: `173.9494597881494`;
- 3-AI zero crossing: `100.36788462067109`;
- neither zero crossing is reachable within one complete historical TRAIN pass.

Therefore a shared, one-pass-bounded repair design using the diagnostic `pos_weight=1.0` must, before any safety margin is added, exceed the cross-specialist floor `9.468108115675737` and cannot exceed `49.93333333333333` without reusing historical examples.

This interval is a feasibility boundary, not a selected replay ratio.

## Scientific interpretation

The evidence strongly supports that `pos_weight=5.0` amplified target-domain positive pressure enough that a single full historical replay pass could not counterbalance it at the frozen source checkpoint.

The evidence does **not** prove that positive weighting alone is the unique root cause. At `pos_weight=1.0`, V5 still exerts one-sided adaptation pressure and requires material historical replay. The conservative root-cause class remains:

`V5_ONLY_DOMAIN_ADAPTATION_WITH_UNCONSTRAINED_SOURCE_FORGETTING`

with the dominant mechanism not uniquely isolated.

## Frozen data boundary for any later repair proposal

A later repair proposal may only consider:

- the existing 540 V5 `adaptation_train` slots for gradients;
- the existing first-30 diagnostic seed surface as zero-gradient diagnostics only;
- historical M4A TRAIN replay from the exact recovered D10 pixel path;
- frozen source 2-AI and 3-AI checkpoints as initialization;
- 4-AI unchanged and frozen.

It may not use:

- the 900 reserved V5 TRAIN examples;
- V5 validation;
- FINAL_HOLDOUT;
- new BBoxes;
- new crop geometry;
- new spatial heuristics;
- threshold tuning;
- Resolver/production authority.

## Unresolved choices requiring explicit review before gradients

V5-2G deliberately leaves the following unset:

- exact positive weight;
- exact replay ratio within any feasible interval;
- replay sampling policy;
- learning rate;
- epoch count;
- checkpoint-retention rule for a repair candidate;
- ordering of target-domain and historical-retention gates.

No optimizer run may begin until those choices are preregistered and explicitly reviewed.

## Safety boundary

- TRAINING=False
- BACKWARD=False
- OPTIMIZER_STEPS=0
- CHECKPOINT_WRITE=False
- THRESHOLD_TUNING=False
- REPLAY_RATIO_SELECTED=False
- POSITIVE_WEIGHT_SELECTED=False
- SAMPLING_STRATEGY_SELECTED=False
- REPAIR_TRAINING_AUTHORIZED=False
- NEW_BBOX=False
- NEW_CROP_GEOMETRY=False
- NEW_SPATIAL_HEURISTIC=False
- RESERVE_V5_TRAIN=CLOSED
- V5_VAL=CLOSED
- FINAL_HOLDOUT=LOCKED
- 4-AI=FROZEN
- RESOLVER_WIRING=False
- PRODUCTION_PROMOTION=False
