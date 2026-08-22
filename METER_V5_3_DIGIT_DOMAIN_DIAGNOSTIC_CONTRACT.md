# Meter V5-3 — Digit/Domain Diagnostic and Recovery Contract

Status: **TRAIN-ONLY DIAGNOSTIC / V5-2 SCALE-UP HOLD / FINAL_HOLDOUT LOCKED**

## Decision

Do **not** continue the remaining 1,170 manual TRAIN full-meter BBoxes yet.
The current evidence is sufficient to treat Meter as a representation/input-domain problem, not a simple data-volume problem.

This stage exists to separate and close three causes before annotation scale-up:

1. full-meter -> digit slot geometry failure;
2. historical digit-specialist cross-domain transfer failure;
3. class representation failure between 2/4, 3/4 and 4/4.

No production/runtime/Resolver behavior is changed by this contract.

## Entering repository evidence

### A. Historical frozen specialist contract

The audited historical 2-AI / 3-AI / 4-AI checkpoints consume tight grayscale 64x64 digit crops, with aspect-preserving LANCZOS thumbnailing, no upscaling and white centering. Their frozen probability thresholds are 0.48 / 0.60 / 0.47 respectively.

These checkpoints were validated on historical ST-OMR synthetic `stomr-*` families, not on the current PrIMuS `package_ab` family distribution.

### B. V5-1R1 geometry finding

The V5-1R1 audit derived numerator/denominator slots by splitting one approved full-meter bbox at `floor(h/2)` with the same x extent and no padding search.
That diagnostic was closed after the hard midpoint split was falsified.

### C. V5-1R2 causal split

The V5-1R2 micro-pilot correctly identified two independent hypotheses that must not be conflated:

- bad slot/crop geometry;
- cross-domain transfer failure of the historical 2/3/4 specialists.

Its intended human GT uses independent numerator and denominator boxes; vertical overlap is allowed.

### D. Historical final-holdout pattern

The V4-5 independent final holdout rejected the prior numerator candidate at 0.7467 accuracy / 0.7444 macro-F1, with recall(2)=0.58, recall(3)=0.66 and recall(4)=1.00. This is a systematic 2/3 separation failure with 4 preserved, not random symmetric noise.

### E. Current data surface

The accepted clean dataset remains:

`METER_V2_1500_PACKAGE_AB_CLEAN`

- 2/4 = 500
- 3/4 = 500
- 4/4 = 500
- TRAIN = 1,200 (400/class)
- VAL = 150 (50/class)
- FINAL_HOLDOUT = 150 (50/class)
- 1,500 globally unique families
- package_ab only

The accepted 30 TRAIN full-meter human BBoxes remain immutable seed evidence.

## Scientific recovery architecture

Meter recognition is split into three explicit layers:

```text
Meter localization
    -> digit recognition
    -> deterministic meter composition / abstention
```

### Layer 1 — Meter localization

Locate the visual time-signature region. The locator does not decide 2/4, 3/4 or 4/4.

### Layer 2 — Digit recognition

Replace the three historical one-vs-rest binary specialists as the primary recovery path with one shared multiclass numerator recognizer over classes `2|3|4`.

The denominator `4` is treated as an independent structural validator, not as the class selector.

Reason: all target classes share denominator 4; therefore the numerator is the discriminative symbol, while the denominator is evidence that the crop is a valid stacked time signature.

Historical 2-AI / 3-AI / 4-AI remain read-only baselines only. Their thresholds are not tuned against V5 data.

### Layer 3 — Deterministic composer

The learned model may produce evidence only.
A deterministic composer accepts a meter only when:

- numerator is one of 2, 3, 4;
- denominator evidence is 4;
- geometry is compatible with a stacked time signature;
- ambiguity/conflict policy passes.

Otherwise it abstains. No AI output mutates authoritative score state directly.

## Phase 1 — 30-sample exact spatial diagnostic

Use only the existing 30 TRAIN pilot images: 10/class.

For each image obtain model-blind human GT for:

- one tight numerator box;
- one tight denominator box.

The approved full-meter bbox remains reference-only.
The two role boxes may overlap vertically.

Persist atomically with image SHA/dimension binding.
No model inference is shown to the annotator before the boxes are frozen.

### Phase 1A — Historical specialist transfer test

After the 60 role boxes are frozen, run the historical 2-AI / 3-AI / 4-AI exactly once on the exact human tight crops using the frozen 64x64 pixel contract.

No threshold, crop, padding, resize or score transport may be changed after scores are seen.

Report per class:

- expected specialist probability distribution;
- wrong-specialist distributions;
- threshold hit rate;
- no-hit / conflict / wrong-unique counts;
- deterministic replay.

If expected 2 or 3 remains systematically below its frozen threshold on exact human crops, the old specialist path is formally rejected for V5 and may not be rescued by threshold tuning.

## Phase 2 — Annotation-economy experiment before 1,170 manual boxes

Do not assume that all remaining full-meter boxes must be drawn manually.

PrIMuS images were produced from symbolic sources through a notation-rendering pipeline. Therefore test a bounded, TRAIN-only reconstruction/proposal path on the same 30 human-gold examples:

1. use the source MEI/render metadata available for the exact selected sample;
2. attempt deterministic meter-region reconstruction/proposal;
3. compare proposal with human full-meter and tight-digit GT;
4. measure IoU and failure modes;
5. never overwrite human GT.

A proposal path may be used for scale-up only if its gate is preregistered and passes on the 30 human-gold samples. Otherwise it is rejected and annotation remains human-in-the-loop.

If an automatic proposal path is admitted, it remains proposal-only:

`proposal -> human accept/correct -> gold annotation`

No automatic GT promotion.

## Phase 3 — New package_ab numerator model

Only after a sufficient TRAIN crop surface exists, train a new package_ab-native multiclass numerator model.

Required properties:

- classes exactly `2|3|4`;
- class-balanced TRAIN surface;
- family-disjoint split policy preserved;
- fixed preprocessing before validation;
- deterministic seed/replay evidence;
- no FINAL_HOLDOUT access;
- no threshold search against final data;
- no runtime/Resolver connection.

The old V4 tiny 3-class CNN may be used only as an architectural baseline. It must be retrained from scratch on the new clean package_ab TRAIN surface; old learned weights are not promoted.

## Phase 4 — Development validation

Use only the 150-family VAL surface after the model, preprocessing and acceptance rule are frozen.

Report at minimum:

- accuracy;
- macro-F1;
- recall for 2, 3, 4;
- full confusion matrix;
- denominator-4 structural recall;
- abstention rate;
- localization IoU if a learned/proposed locator is used;
- family-disjoint replay/determinism evidence.

A class-asymmetric failure such as strong 4 with weak 2/3 is a HOLD, not a threshold-tuning invitation.

## Final holdout rule

`final_holdout` remains locked throughout V5-3 development.

A future final evaluation requires:

- all model/config/crop/composer rules frozen;
- no unresolved TRAIN/VAL review;
- no further threshold search;
- one-time 150-family final run;
- no rerun or post-result tuning.

## Immediate gate

Before any 1,170-sample annotation scale-up:

1. freeze 30-sample tight numerator/denominator human GT;
2. execute the historical-specialist transfer diagnostic on exact human crops;
3. run the 30-sample annotation-economy proposal/reconstruction experiment;
4. choose the recovery path from evidence, not from the consumed final holdout.

Until these gates close, V5-2 1,200 full-meter BBox scale-up remains **HOLD**.
