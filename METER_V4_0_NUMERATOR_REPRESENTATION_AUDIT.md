# Meter V4-0 — Numerator Representation Audit

## Purpose

V4-0 is a bounded diagnostic experiment to test whether the current Meter bottleneck is primarily a representation problem rather than another loss/logit-calibration problem.

The experiment uses only the 27 positive Teacher Gold TRAIN records (9 each of 2/4, 3/4, and 4/4), derives a deterministic numerator-only crop from the accepted Teacher Gold meter bbox, and evaluates a tiny 3-class specialist with family-disjoint out-of-fold (OOF) evaluation.

V4-0 is not a production candidate, does not replace D11/V3, does not access D10, does not evaluate the 18 Teacher Gold adaptation-validation records, and never opens sealed TEST.

## Scientific question

The three supported positive classes share denominator 4. The discriminative information is therefore concentrated in the numerator glyph:

```text
2/4 -> 2
3/4 -> 3
4/4 -> 4
```

V4-0 asks whether isolating this numerator produces a substantially cleaner family-generalizing signal than whole-ROI Meter classification.

## Frozen input surface

- exact Teacher Gold admission contract v1;
- source split remains the approved Teacher Gold source TRAIN surface;
- audit uses only records whose adaptation split is `train` and target class is one of `2/4`, `3/4`, `4/4`;
- expected audit cardinality: exactly 27 records, 9 per positive class;
- one positive record per family;
- Teacher Gold adaptation-validation records may be enumerated only as manifest metadata by the existing verifier, but their label/image tensors are not opened, trained on, selected on, or evaluated by V4-0;
- D10 is not read;
- sealed TEST is not enumerated or opened.

## Numerator crop contract

The accepted Teacher Gold label already contains the mapped full Meter bbox in the canonical 256x192 ROI. V4-0 derives the numerator crop deterministically:

1. validate the bbox is finite, positive-area, and fully inside 256x192;
2. use the full bbox width plus 15% horizontal padding on each side;
3. use the top half of the bbox as the numerator vertical extent;
4. add 5% full-bbox-height padding above and below the numerator half;
5. clamp to the 256x192 ROI;
6. extract from the already verified Teacher Gold image tensor;
7. resize aspect-preserving into a 64x64 canvas with white/background padding;
8. represent ink as float32 in `[0,1]` with background zero, matching the existing inverted grayscale convention.

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

Each fold trains on 18 records and predicts 9 previously unseen families. Every one of the 27 records receives exactly one prediction from a model that did not train on that record or family.

The existing 18 Teacher Gold adaptation-validation records are not used for model selection, early stopping, threshold choice, or metric computation.

## Tiny specialist

The audit model is deliberately small and from scratch:

```text
64x64 numerator crop
 -> Conv 1->8 + ReLU + MaxPool
 -> Conv 8->16 + ReLU + MaxPool
 -> AdaptiveAvgPool 4x4
 -> FC 256->32 + ReLU
 -> FC 32->3
```

Only classes `2`, `3`, `4` exist in this specialist.

Training is fixed-length and deterministic. No held-out fold is used for checkpoint selection. Training uses only deterministic integer-pixel shifts of the 18 training crops; no synthetic renderer corpus, D10 replay, V3 adapter, bbox loss, presence loss, or margin loss is used.

## Audit decision

V4-0 emits a diagnostic decision, not a promotion decision.

`REPRESENTATION_SIGNAL_STRONG` requires both:

- OOF accuracy >= 25/27 (0.9259...); and
- per-class recall >= 8/9 for each of 2, 3, and 4.

`REPRESENTATION_SIGNAL_WEAK_OR_DATA_LIMITED` is emitted otherwise.

Interpretation:

- strong signal supports moving to a dedicated Meter numerator specialist architecture;
- weak signal means the isolated glyph representation alone is insufficient with the current real-data diversity, so the next action should prioritize crop review and/or more real font/domain diversity rather than more V3 logit tuning.

The decision does not authorize production, checkpoint replacement, TEST access, or ScoreMosaic integration.

## Required evidence

The Colab audit must write a compact result JSON containing:

- repository SHA and contract version;
- exact 27 record/family ids and deterministic fold assignment;
- crop geometry and crop SHA-256 per record;
- fold-by-fold predictions;
- aggregate 3x3 confusion matrix, accuracy, macro-F1, per-class recall;
- audit decision and reasons;
- `d10_opened=false`;
- `teacher_adaptation_validation_evaluated=false`;
- `test_opened=false`;
- `runtime_connected=false`;
- `production_promotion_authorized=false`.

A contact sheet of the 27 numerator crops is also produced for human crop-quality review, but visual inspection must not be used to relabel or special-case audit records.

## Safety boundary

V4-0 is an isolated training-lab diagnostic only. It may not:

- open sealed TEST;
- read D10;
- use Teacher Gold adaptation-validation outcomes;
- mutate D11/V3 checkpoints;
- connect Resolver/runtime;
- authorize production promotion;
- merge solely because CI passes.
