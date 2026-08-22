# METER V5-1R2 — Human Tight-Digit Micro-Pilot

Status: **PREREGISTERED / TRAIN-ONLY / HUMAN SPATIAL GT / NO MODEL IN ANNOTATION LOOP**

## Why this stage exists

V5-1R1 falsified one specific conversion rule:

```text
human full-meter bbox -> hard midpoint split -> 2-AI / 3-AI / 4-AI
```

The first frozen V5-1 TRAIN sample produced deterministic `NO_HIT` for both derived slots, so the strict 60/60 midpoint gate cannot pass.

That failure must **not** be misread as proof that the frozen 2-AI / 3-AI / 4-AI architecture is weak. Their accepted validation evidence came from the historical M4A/D10 ST-OMR synthetic surface (`stomr-*` families), whereas V5 uses PrIMuS `package_ab` (`ab_*` families). The observed V5 failure can therefore contain two independent causes:

1. wrong digit localization / crop geometry;
2. cross-domain transfer failure.

V5-1R2 separates those causes with the smallest safe human-GT experiment.

## Parent bindings

Parent branch/head:

`agent/meter-v5-1r1-specialist-input-audit`

`52f813af15818658d53756d5567765694b2e14a7`

Frozen V5-1 files:

- `bbox_pilot_30_selection.csv` SHA-256: `4070f46f64efed5b12b26f7dd1d4e3f09b4abf804125140d38a212819bbcbe97`
- `bbox_pilot_30.csv` SHA-256: `b60a953811aa136752372d7c8cea6fe7a1c1c964a62bd53c0c9d48c56c735665`
- dataset: `METER_V2_1500_PACKAGE_AB_CLEAN`

V5-1R1 early-falsification evidence remains immutable and is not rewritten.

## Admitted surface

Exactly **9 existing V5-1 TRAIN pilot samples**:

- `2/4`: first 3 rows for that class in the frozen 30-sample selection order;
- `3/4`: first 3 rows for that class;
- `4/4`: first 3 rows for that class.

Total human role boxes when complete: **18**.

No new dataset sample is selected. No validation or final-holdout image is enumerated, decoded, hashed, displayed, or modeled.

Selection is persisted as:

`annotations/digit_bbox_pilot_9_selection.csv`

and must bind the original source image SHA/dimensions plus the already-approved full-meter bbox.

## Human annotation rule

For every selected score image, the annotator draws **two independent tight boxes**:

- `numerator`
- `denominator`

The full-meter V5-1 box is shown only as a reference boundary.

### Tight-box meaning

A role box should contain the visible ink of that one meter digit as completely as practical while excluding:

- the other meter digit;
- clef;
- key signature;
- neighboring note/rest material.

Staff lines naturally cross through time-signature digits and do not need to be removed.

### Important: vertical overlap is allowed

Numerator and denominator glyph extents may overlap vertically. Therefore:

- the two role boxes may overlap;
- no midpoint split is imposed;
- no non-overlap gate is imposed.

Both role boxes must remain inside the already-approved full-meter box and source image.

## Ground-truth authority

These real spatial labels are human-verified evidence.

Forbidden:

- model-predicted box becoming GT;
- D11 bbox becoming GT;
- 2/3/4 specialist score changing a box during annotation;
- threshold/crop search in the annotation UI;
- automatic box adjustment.

The annotation UI is deliberately **model-blind**.

## Persistence

Human output:

`annotations/digit_bbox_pilot_9.csv`

Canonical rows are keyed by `(sample_id, role)`.

Required columns:

`sample_id,meter,role,x,y,w,h,status,image_sha256,image_width,image_height,updated_utc`

Allowed statuses:

- `PASS` — valid human tight box;
- `REVIEW` — unresolved sample/role; coordinates blank.

Writes are atomic. Resume is by sample identity. Original images and V5-1 full-meter annotations are read-only.

## Mechanical audit

After all 9 samples are handled, write:

`annotations/digit_bbox_pilot_9_audit.json`

Mechanical checks include:

- exactly 9 selected samples;
- exactly 3 samples per meter class;
- exactly two role rows per sample;
- exactly one numerator and one denominator role;
- image SHA/dimension binding preserved;
- positive integer boxes;
- role boxes inside source image;
- role boxes inside approved full-meter bbox;
- numerator center above denominator center;
- duplicate role rows = 0;
- unresolved REVIEW count;
- overlap count/area reported diagnostically, never rejected solely because of overlap.

`annotation_contract_ready=true` requires 18/18 PASS and zero invalid/out-of-bound boxes.

## What happens after human annotation

Only after `annotation_contract_ready=true` may the separate V5-1R2-B transfer diagnostic execute the frozen 2-AI / 3-AI / 4-AI models on these exact human tight boxes.

That diagnostic will use:

- the already-frozen historical 64x64 pixel transform;
- exact frozen checkpoint SHA identities;
- thresholds `0.48 / 0.60 / 0.47`;
- exactly-one arbitration;
- 10/10 replay determinism.

No threshold or crop changes are allowed after seeing scores.

### Interpretation matrix

If human-tight boxes are read correctly, localization/crop geometry is the dominant proven blocker on this pilot.

If human-tight boxes still produce material `NO_HIT`, `CONFLICT`, or wrong-unique results, cross-domain transfer is not established and the old D10 specialist metrics must not be used as V5 readiness evidence. A new V5-domain specialist training plan may then be designed, still TRAIN-only and behind a separate frozen contract.

## Safety invariants

- V5 TRAIN pilot only;
- validation closed;
- final holdout closed;
- model inference during annotation: 0;
- optimizer steps: 0;
- threshold tuning: 0;
- crop search: 0;
- D11 execution: 0;
- Resolver connection: false;
- production promotion: false;
- 1,200-sample annotation scale-up: blocked until this diagnostic closes.
