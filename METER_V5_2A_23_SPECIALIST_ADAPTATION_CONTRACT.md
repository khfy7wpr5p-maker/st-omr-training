# Meter V5-2A — 2-AI / 3-AI Specialist Adaptation Contract

## Status

Shadow/train-only adaptation lane. This contract is stacked on PR #95 only to reuse the accepted V5-1 30 full-meter seed BBoxes and the fail-closed annotation infrastructure. PR #95's 1,200-BBox scale-up remains **HOLD**.

## Evidence entering V5-2A

The same package_ab V5 30-sample pilot, same deterministic preprocessing, same frozen checkpoint audit, and same staff-relative slot replay produced:

- 2/4: 0/10 meter PASS;
- 3/4: 0/10 meter PASS;
- 4/4: 9/10 meter PASS;
- denominator-4: 26/30 correct;
- true-2 numerator 2-AI score: median ~0.0704, max ~0.1508 vs frozen threshold 0.48;
- true-3 numerator 3-AI score: effectively ~0 vs frozen threshold 0.60;
- true-4 numerator 4-AI score: median ~0.7379, mean ~0.6628 vs frozen threshold 0.47.

Decision: this is class-specific V5/package_ab transfer failure in 2-AI and 3-AI, not a threshold-tuning problem. 4-AI remains the frozen control specialist.

## Dataset / annotation target

Authoritative dataset root:

`/content/drive/MyDrive/TEST/METER_V2_1500_PACKAGE_AB_CLEAN`

V5-2A opens only TRAIN and selects exactly 300 unique TRAIN families:

- 100 x 2/4;
- 100 x 3/4;
- 100 x 4/4.

The accepted V5-1 30 human full-meter BBoxes count toward this total (10/class) and are immutable. Therefore exactly 270 new human full-meter BBoxes remain.

One annotation means exactly one human BBox containing the complete visible upper + lower meter numerals together. Clef, key signature, and first following note remain outside when reasonably possible.

Forbidden during annotation:

- midpoint split;
- human tight numerator/denominator digit boxes;
- model-generated GT;
- automatic BBox shrink/expand/move;
- VAL or FINAL_HOLDOUT annotation;
- training or model inference.

## Approved training-input semantics

After, and only after, the 300/300 TRAIN full-meter BBox gate and human visual QA pass, V5-2A may derive specialist training slots with the already evidenced staff-relative geometry contract:

- horizontal anchor: center-x of the human full-meter BBox;
- numerator vertical center: second accepted staff line;
- denominator vertical center: fourth accepted staff line;
- slot width: `1.5960569245912566 * local_staff_spacing`;
- slot height: `2.0 * local_staff_spacing`;
- existing frozen historical 64x64 digit preprocessing is preserved;
- any unresolved or non-accepted geometry fails closed; no replacement heuristic is allowed.

This is the only approved automatic spatial derivation in V5-2A. It must not be replaced by midpoint, tight-digit, prediction-driven, or validation-tuned geometry.

## Model policy

Trainable:

- 2-AI adaptation candidate;
- 3-AI adaptation candidate.

Frozen/read-only controls:

- existing 4-AI checkpoint and threshold;
- historical frozen 2-AI / 3-AI checkpoints as baselines;
- deterministic geometry/preprocessing contracts.

No single 3-class numerator model is promoted in this lane. Historical D10/M4A evidence must be retained as replay/retention evidence when adaptation training opens; source-domain regression is a rejection condition.

## Gates

### Annotation gate

Must have all of:

- 300/300 PASS;
- 100/class;
- REVIEW=0;
- unique sample/family identities;
- zero invalid/outside BBoxes;
- zero seed mutations;
- original image SHA/dimension bindings unchanged;
- human visual QA/contact-sheet review PASS.

Until then: `TRAINING_AUTHORIZED=false`.

### 30-sample candidate pilot gate

A future adapted 2/3 candidate must be replayed on the same immutable 30 pilot records. Minimum bounded pilot gate:

- 2/4 >= 8/10;
- 3/4 >= 8/10;
- 4/4 frozen-control path >= 9/10;
- denominator-4 must not materially regress from the 26/30 baseline;
- no threshold lowering to rescue a candidate.

This is a pilot gate only, not a generalization claim.

### Validation / holdout

VALIDATION remains closed until an adaptation candidate passes TRAIN-only retention and the 30-sample pilot gate. FINAL_HOLDOUT remains locked until all earlier gates are explicitly satisfied. Production/Resolver promotion remains closed.

## Safety invariants

`TRAIN -> human annotation -> mechanical audit -> human visual QA -> approved staff-relative slot derivation -> 2/3 adaptation -> source retention -> 30-pilot replay -> later validation`

No step may infer authority from model confidence. Any provenance, geometry, checkpoint, split, or dataset mismatch fails closed.