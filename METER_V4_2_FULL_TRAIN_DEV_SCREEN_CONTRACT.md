# Meter V4-2 — Full-Train Numerator Candidate + Development Screen

## Purpose

V4-2 converts the accepted V4-1 family-disjoint OOF result into one deterministic full-TRAIN numerator specialist candidate, then performs one bounded development-only screen on the nine pre-existing Teacher Gold adaptation-validation positive families.

V4-2 is not a production validation. The nine adaptation-validation families have been observed during earlier Meter development and therefore are explicitly treated as development evidence only. A new independent family-disjoint real holdout remains mandatory before any production promotion.

## Frozen parent evidence

V4-2 inherits the exact V4-0 crop artifact and the accepted V4-1 evidence:

- V4-0 repository binding: `8641fc45ae0e5613d280eee8af12ac105c765c313190660c88479e38bf6eff48`;
- V4-0 result SHA-256: `422e79d7f71a1d2228e1392160d6ef4444521d8796f3c3b8fb6cd0a226c9060a`;
- V4-1 OOF: `27/27`, macro-F1 `1.0`, recalls `2=9/9`, `3=9/9`, `4=9/9`;
- V4-1 decision: `LEARNED_NUMERATOR_SIGNAL_STRONG`;
- V4-1 deterministic repeat: PASS.

The full-train candidate uses the unchanged V4-1 architecture, tensor representation, nine integer-translation views, optimizer, learning rate, weight decay, gradient clipping and exactly 160 epochs.

## Contamination barrier

The candidate must be completely fixed before any adaptation-validation image is decoded.

Execution order is frozen:

1. verify exact V4-0 parent result, COMPLETE receipt and 27 crop hashes;
2. load only the 27 accepted V4-0 TRAIN numerator crops;
3. train the full 27-family candidate for exactly 160 optimizer steps;
4. independently repeat the same full training and require identical canonical model-state SHA-256 and final loss;
5. only after the repeat passes, open the exact immutable Teacher Gold pilot/choices/evidence inputs;
6. select exactly nine positive adaptation-validation families (3 each `2/4`, `3/4`, `4/4`);
7. derive their numerator crops with the exact frozen V4-0 crop contract;
8. evaluate once with the already-fixed candidate.

Development labels cannot affect initialization, epochs, optimizer, augmentation, loss, architecture, threshold or checkpoint selection.

## Full-TRAIN candidate

- classes: `2 | 3 | 4` only;
- records/families: 27;
- class balance: 9/9/9;
- deterministic views per crop: 9;
- training batch: 243 views, 81/class;
- model: unchanged 9,571-parameter V4-1 CNN;
- seed: `812042`;
- epochs/optimizer steps: 160;
- AdamW, LR `0.001`, weight decay `0.0001`;
- gradient clip L2 `1.0`;
- no scheduler, early stopping or checkpoint sweep.

The final epoch is the only candidate state. It is trained twice; mismatch fails closed.

## Development screen

The screen contains exactly nine positive adaptation-validation families:

- numerator `2`: 3;
- numerator `3`: 3;
- numerator `4`: 3.

`FULL_TRAIN_DEV_SCREEN_PASS` requires:

- candidate deterministic repeat PASS;
- accuracy = `9/9`;
- macro-F1 = `1.0`;
- recall(`2`) = `3/3`;
- recall(`3`) = `3/3`;
- recall(`4`) = `3/3`.

Anything less is `FULL_TRAIN_DEV_SCREEN_HOLD` and blocks integration planning.

Passing this screen authorizes only the next shadow-integration/holdout-planning stage. It does not establish an unbiased final real-world accuracy estimate.

## Safety boundary

Forbidden:

- any `none` task in training or this numerator-only screen;
- D10 access;
- sealed TEST access;
- D11/V3 weights as numerator classifier inputs;
- development-validation-driven tuning;
- runtime/Resolver connection;
- production promotion.

Required result flags:

- `d10_opened=false`;
- `test_opened=false`;
- `none_tasks_used=0`;
- `development_validation_used=true`;
- `development_validation_used_for_training=false`;
- `fresh_independent_holdout_required=true`;
- `runtime_connected=false`;
- `resolver_connected=false`;
- `production_promotion_authorized=false`.

A passing V4-2 result may emit a bounded development candidate checkpoint artifact, but that checkpoint is not a production candidate until a fresh independent real holdout and later integration gates pass.
