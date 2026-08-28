# Meter V5-4 — Unified Numerator Shadow Contract v1

## Purpose

V5-4 freezes the user-approved architectural direction for Meter recognition without opening a new training or validation surface.

The target recognition topology is exactly one shared numerator specialist that classifies the numerator digit as one of `2`, `3`, or `4`. It replaces neither the historical 2-AI / 3-AI / 4-AI path nor any production/runtime authority in this contract stage.

## Prerequisite boundary

V5-4 is stacked on the CI-green V5-3K branch, but execution remains blocked until a completed V5-3K Drive forensic report exists and is hash-bound in a later preregistration stage.

Required entering state:

- V5-3J evidence remains immutable;
- V5-3K implementation/runner CI is green;
- V5-3K external forensic result is still required before any V5-4 training recipe may be selected;
- historical Validation, immutable First-30, V5 reserve, V5 validation and FINAL_HOLDOUT remain closed;
- current runtime/Resolver authority is unchanged.

## Frozen architecture direction

```text
meter numerator crop
        ↓
ONE Unified Numerator AI
        ↓
class scores: 2 | 3 | 4
        ↓
shadow adapter
        ↓
2/4 | 3/4 | 4/4 candidate
        ↓
future Meter Validator
        ↓
CONFIRMED | CONFLICT | REVIEW_REQUIRED
```

Rules:

1. Exactly one trainable numerator classifier is allowed for this lane.
2. Its class vocabulary is exactly `2`, `3`, `4`.
3. Three independent binary digit specialists are not the target V5-4 architecture.
4. The denominator remains outside the numerator classifier. A later adapter may compose a meter candidate only from independently admitted denominator evidence.
5. The new model is shadow-only until explicit later promotion gates pass.
6. Existing 2-AI / 3-AI / 4-AI checkpoints remain immutable controls during comparison.
7. No silent meter correction is allowed.
8. Low confidence, disagreement or unsupported composition must abstain / request review.

## Shadow comparison

A later execution stage must compare, on the same allowed non-final evidence surface:

- legacy specialist result;
- unified numerator result;
- class-wise recall for 2, 3, 4;
- macro-F1 / accuracy;
- 2↔3 confusion;
- abstention/review rate;
- regression against historical retained behavior.

The unified model cannot gain runtime authority merely by outperforming the legacy path on TRAIN.

## Adapter contract

The adapter is deterministic and non-trainable.

- numerator `2` + independently accepted denominator `4` -> shadow `2/4` candidate;
- numerator `3` + independently accepted denominator `4` -> shadow `3/4` candidate;
- numerator `4` + independently accepted denominator `4` -> shadow `4/4` candidate;
- any other/ambiguous denominator -> no composed meter candidate;
- conflicting evidence -> `REVIEW_REQUIRED`, never silent correction.

## Meter Validator boundary

The future Meter Validator is a separate deterministic layer. It may use rhythm, exact rational measure capacity, barline/measure structure, neighboring-measure continuity and MusicXML time evidence to confirm or challenge a shadow meter candidate.

It must not treat the unified AI prediction as ground truth and must not silently rewrite the meter from measure-sum evidence alone.

## Safety state

This contract authorizes only architecture declaration and static tests.

- training: false
- fitting: false
- optimizer steps: 0
- checkpoint write: false
- model mutation: false
- threshold tuning: false
- crop/BBox tuning: false
- validation access: false
- FINAL_HOLDOUT access: false
- Resolver wiring: false
- production promotion: false
- existing specialists removed: false
- V5-4 execution recipe selected: false

## Next gates

1. Complete and hash-bind V5-3K external forensic report.
2. Use that evidence to preregister one fixed unified-numerator representation/training recipe.
3. Train exactly one shared `2|3|4` classifier on TRAIN-only surfaces.
4. Run TRAIN acceptance and historical-retention checks.
5. Open bounded validation only after those gates pass.
6. Add deterministic shadow adapter and Meter Validator.
7. Freeze model/config/adapter/validator.
8. Only then open the untouched 150-example FINAL_HOLDOUT once.

No step in this contract authorizes step 2 or later automatically.
