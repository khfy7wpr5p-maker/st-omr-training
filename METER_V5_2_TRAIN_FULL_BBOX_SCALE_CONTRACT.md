# METER V5-2 — TRAIN Full-Meter BBox Scale Contract

Status: **PREREGISTERED / ANNOTATION-ONLY / TRAIN-ONLY**

## Canonical parent

V5-2 starts only from the completed V5-1 30-sample human full-meter BBox pilot on exact parent head:

`4f744d0bf8a2f6180a62f0f08abb96b83cfb5da8`

Accepted V5-1 evidence:

- 30/30 human annotations PASS;
- 10 each for 2/4, 3/4, 4/4;
- REVIEW = 0;
- invalid/outside-image bbox = 0;
- original image binding preserved;
- annotation contract freeze ready = true;
- final holdout locked;
- training/tuning/model/inference closed.

The 30 accepted red boxes are authoritative full-meter annotations and MUST NOT be reinterpreted as digit-level boxes.

## Human annotation meaning

Each sample has exactly **one** human BBox.

The box contains the complete visible meter pair:

- upper numeral;
- lower numeral;
- both inside the same rectangle.

The human should continue the V5-1 rule: keep clef, key signature and the first following note outside the box when reasonably possible.

The following are explicitly forbidden in V5-2:

- splitting a full-meter BBox at its midpoint;
- deriving numerator/denominator ground truth from a full-meter BBox;
- automatically shrinking, expanding or moving a human BBox;
- using a model prediction to create or correct annotation ground truth;
- changing the semantic meaning of the accepted 30 seed BBoxes.

## Dataset scope

Authoritative dataset:

`/content/drive/MyDrive/TEST/METER_V2_1500_PACKAGE_AB_CLEAN`

The frozen dataset contains exactly:

- 2/4: 400 TRAIN + 50 VAL + 50 FINAL_HOLDOUT;
- 3/4: 400 TRAIN + 50 VAL + 50 FINAL_HOLDOUT;
- 4/4: 400 TRAIN + 50 VAL + 50 FINAL_HOLDOUT.

V5-2 opens **TRAIN annotation only**:

- 400 × 2/4;
- 400 × 3/4;
- 400 × 4/4;
- total = 1200 TRAIN samples.

The accepted V5-1 30 annotations are immutable seed annotations and count toward this 1200 total. Therefore the remaining human work is exactly **1170 new full-meter BBoxes**.

VAL and FINAL_HOLDOUT annotation remain closed.

## Seed preservation gate

Before any V5-2 annotation session may open, implementation must verify the V5-1 canonical evidence and exact seed bindings. At minimum it must verify:

- dataset fingerprint = `e7e849ac6d6d7a622dc94107a4dc4074c48e1e0ab837e726857394cb4072b8f0`;
- V5-1 annotation CSV SHA-256 = `b60a953811aa136752372d7c8cea6fe7a1c1c964a62bd53c0c9d48c56c735665`;
- V5-1 pilot selection CSV SHA-256 = `4070f46f64efed5b12b26f7dd1d4e3f09b4abf804125140d38a212819bbcbe97`;
- V5-1 pilot audit JSON SHA-256 = `dd254ba7c7408a73168d52da3b50de2de8eafa4748e3f4e64e4d11707eeae366`;
- FINAL_HOLDOUT lock JSON SHA-256 = `51923443c1cc892a2f8852a52ef7658bc604b6fa948a3cc43f4f8b6dab3cc2e3`.

All 30 seed rows must remain byte/field-equivalent in their identity, image binding, BBox and PASS status. A V5-2 implementation must fail closed rather than silently overwrite a seed BBox.

## Required execution behavior

The implementation must:

1. verify the exact clean dataset structure before annotation;
2. expose only TRAIN images in the annotation session;
3. bind every sample to unique sample/family identity plus exact source-image SHA and dimensions;
4. persist annotation progress atomically;
5. resume by sample identity;
6. allow an unresolved sample to be marked REVIEW without fabricating a BBox;
7. never modify original `image.png` files;
8. keep the red rectangle as annotation/preview state only;
9. preserve the 30 seed annotations unchanged;
10. refuse any VAL or FINAL_HOLDOUT sample presented to the annotation UI.

## Completion gate

V5-2 annotation completion requires all of the following:

- 1200 / 1200 TRAIN annotations PASS;
- 400 / 400 / 400 PASS by meter class;
- missing annotation = 0;
- unresolved REVIEW = 0;
- duplicate sample identity = 0;
- duplicate family identity = 0;
- zero/negative BBox = 0;
- BBox outside source image = 0;
- image SHA/dimension binding mismatch = 0;
- V5-1 seed mutation = 0;
- automatic/model-generated BBox count = 0.

After mechanical completion, human visual QA/contact-sheet review is required before any training gate may be proposed.

## Safety boundary

During V5-2:

- training = false;
- optimizer steps = 0;
- tuning/calibration = false;
- model/checkpoint opening = false;
- inference count = 0;
- VAL annotation = false;
- FINAL_HOLDOUT annotation = false;
- runtime connection = false;
- Resolver connection = false;
- production promotion = false.

V5-2 authorizes full-meter TRAIN annotation only. It does not authorize digit-slot derivation, specialist-model evaluation, training, or production use.

## Superseded diagnostics

PR #93 midpoint-derived slot experiment and PR #94 tight-digit micro-pilot are not canonical parents of V5-2. Their diagnostic branches must not modify or reinterpret V5-1 annotations.
