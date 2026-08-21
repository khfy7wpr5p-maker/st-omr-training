# Meter V4-0 — Numerator Representation Audit

## Purpose

V4-0 is a bounded diagnostic experiment to test whether the current Meter bottleneck is primarily a representation problem rather than another loss/logit-calibration problem.

The experiment uses only the 27 positive Teacher Gold TRAIN families (9 each of 2/4, 3/4, and 4/4), derives a deterministic numerator-only crop from the accepted Teacher Gold meter bbox, and evaluates the representation with a zero-training normalized class-centroid classifier under family-disjoint out-of-fold (OOF) evaluation.

V4-0 is not a production candidate, does not replace D11/V3, does not access D10, does not evaluate the 18 Teacher Gold adaptation-validation records, performs zero optimizer steps, and never opens sealed TEST.

## Scientific question

The three supported positive classes share denominator 4. The discriminative information is therefore concentrated in the numerator glyph:

```text
2/4 -> 2
3/4 -> 3
4/4 -> 4
```

V4-0 asks whether isolating this numerator produces a substantially cleaner family-generalizing signal than whole-ROI Meter classification.

## Frozen input surface

- exact Teacher Gold pilot/choices/evidence and admission split policy v1;
- source split remains the approved Teacher Gold source TRAIN surface;
- the adaptation split is reproduced deterministically from family/class/package metadata;
- only positive records assigned to adaptation `train` are decoded and transformed;
- expected audit cardinality: exactly 27 records, 9 per positive class;
- one positive record per family;
- the 9 positive adaptation-validation families are never decoded, trained on, selected on, or evaluated by V4-0;
- the paired `none` tasks are not used by the representation audit;
- D10 is not read;
- sealed TEST is not enumerated or opened.

The source pilot JSON necessarily contains metadata/data-URI strings for the approved 36 source families, but V4-0 decodes image bytes only for the 27 positive TRAIN families selected by the frozen adaptation split policy.

## Numerator crop contract

The accepted Teacher Gold positive answer contains the full Meter bbox and ROI crop box. V4-0 replays the existing admission transform to the canonical 256x192 Teacher Gold ROI, maps the full Meter bbox into that ROI, and then derives the numerator crop deterministically:

1. validate the mapped full Meter bbox is finite, positive-area, and fully inside 256x192;
2. use the full bbox width plus 15% horizontal padding on each side;
3. use the top half of the bbox as the numerator vertical extent;
4. add 5% full-bbox-height padding above and below the numerator half;
5. clamp to the 256x192 ROI;
6. resize aspect-preserving into a 64x64 grayscale canvas with white padding;
7. convert pixels to an ink vector where white=0 and black=1;
8. L2-normalize the 4096-element vector.

No OCR, external OMR engine, heuristic class correction, validation-ID special case, or manual per-record crop tweak is permitted.

## Family-disjoint fold plan

OOF evaluation uses exactly 3 folds.

Within each class, the nine positive TRAIN families are deterministically ranked by SHA-256 over the V4-0 version, class, and family id. Rank modulo 3 assigns the fold.

Therefore each held-out fold contains exactly:

```text
2/4: 3 families
3/4: 3 families
4/4: 3 families
TOTAL: 9 records
```

Each fold builds class centroids from the other 18 records and predicts 9 previously unseen families. Every one of the 27 records receives exactly one prediction from centroids that did not consume that record or family.

The existing 18 Teacher Gold adaptation-validation records are not used for model selection, threshold choice, or metric computation.

## Zero-training centroid probe

V4-0 deliberately avoids another learned classifier. It tests the representation itself.

For each fold:

1. L2-normalize each 64x64 numerator ink vector;
2. average the six TRAIN vectors per numerator class (`2`, `3`, `4`);
3. L2-normalize each class mean to obtain three deterministic centroids;
4. compute cosine similarity from each held-out vector to the three centroids;
5. choose the highest-similarity class, with fixed class order `2 < 3 < 4` as the exact tie-break.

There are no trainable parameters, epochs, gradients, checkpoints, random augmentations, or optimizer steps. This makes the result a direct test of whether the isolated numerator pixels carry a simple family-generalizing class signal.

## Audit decision

V4-0 emits a diagnostic decision, not a promotion decision.

`REPRESENTATION_SIGNAL_STRONG` requires both:

- OOF accuracy >= 25/27 (0.9259...); and
- per-class recall >= 8/9 for each of 2, 3, and 4.

`REPRESENTATION_SIGNAL_WEAK_OR_DATA_LIMITED` is emitted otherwise.

Interpretation:

- strong signal supports moving to a dedicated learned Meter numerator specialist in V4-1;
- weak signal means the isolated glyph representation alone is insufficient with the current real-data diversity, so the next action should prioritize crop review and/or more real font/domain diversity rather than more V3 logit tuning.

The decision does not authorize production, checkpoint replacement, TEST access, or ScoreMosaic integration.

## Required evidence

The Colab audit must write a compact result JSON containing:

- repository SHA and V4-0 contract version;
- exact 27 family/task identities and deterministic fold assignment;
- replayed full Meter bbox and numerator crop geometry per record;
- crop SHA-256 per record;
- fold-by-fold cosine scores and predictions;
- aggregate 3x3 confusion matrix, accuracy, macro-F1, per-class recall;
- audit decision and reasons;
- `optimizer_steps=0`;
- `d10_opened=false`;
- `teacher_adaptation_validation_evaluated=false`;
- `teacher_adaptation_validation_images_decoded=0`;
- `test_opened=false`;
- `runtime_connected=false`;
- `production_promotion_authorized=false`.

A contact sheet of the 27 numerator crops is also produced for human crop-quality review, but visual inspection must not be used to relabel or special-case audit records.

## Safety boundary

V4-0 is an isolated training-lab diagnostic only. It may not:

- open sealed TEST;
- read D10;
- use Teacher Gold adaptation-validation outcomes;
- train or fine-tune a model;
- mutate D11/V3 checkpoints;
- connect Resolver/runtime;
- authorize production promotion;
- merge solely because CI passes.
