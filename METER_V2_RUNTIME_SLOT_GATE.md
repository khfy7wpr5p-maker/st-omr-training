# Meter V2 — Runtime Digit-Slot Gate

Status: **PIXEL CROP FROZEN / POSITIVE SLOT LOCALIZATION NOT YET FROZEN / SHADOW-ONLY**

## What is now frozen

The historical M4 training worker was recovered and the exact `64x64` digit pixel transform is now represented by `meter_v2_digit_crop_adapter_v1.py`:

1. accept pixel coordinates or normalized coordinates;
2. normalized coordinates are scaled by source width/height;
3. floor left/top, ceil right/bottom, clip to the image;
4. crop and convert to grayscale `L`;
5. `thumbnail((64,64), LANCZOS)` preserving aspect ratio and **without upscaling**;
6. center on a white `64x64` canvas;
7. historical model tensor semantics are grayscale `uint8 / 255.0`.

This closes the preprocessing ambiguity: future shadow inference must feed the digit specialists the same pixel transform used in training.

## Presence gate now fully measured on VALIDATION

The frozen M3-B cache contains all `1,224` authoritative D10 Meter VALIDATION records and no TEST records. With the already-selected development threshold `presence_score >= 0.90`:

- TP: `590`
- FP: `8`
- FN: `1`
- TN: `625`
- recall: `590/591 = 0.9983079526`
- precision: `590/598 = 0.9866220736`
- F1: `0.9924306140`
- accuracy: `1215/1224 = 0.9926470588`

This remains a **temporary shadow Presence bridge**, not final product-quality Presence acceptance.

## Critical remaining boundary: bbox source

The recovered M4 training transform answers:

> Given a candidate digit bbox, how must its pixels be transformed before 2-AI / 3-AI / 4-AI?

It does **not** answer:

> How does runtime obtain the candidate digit bbox from a measure-start image without ground truth?

The historical evidence surfaces must not be conflated:

- M2 positive digit boxes are GT-derived and are valid for data construction / upper-bound evaluation, not runtime localization.
- M3-C2 deterministic anchor zones provide high-recall meter-zone candidates, but the whole anchor half is broader than the tight positive digit boxes used by M4 training.
- M4 hard-negative rows deliberately used those broad anchor halves as `NONE` examples; that does not prove that broad halves are valid positive digit slots.
- D11 predicted Meter bbox localization was known to be weaker than the desired product-quality digit-localization boundary.

Therefore `runtime_digit_bbox_localization_frozen()` remains `False` and PR #65 must not claim a runtime slot PASS yet.

## Required next validation order

Before production/runtime wiring:

1. **GT-slot upper-bound on all 1,224 VALIDATION records**
   - use M2/M4A GT digit boxes only for positive meters;
   - use frozen negative anchor slots for `none` diagnostics;
   - run the exact frozen 2-AI / 3-AI / 4-AI checkpoints jointly;
   - use fail-closed slot arbitration and deterministic composer;
   - measure exact-meter accuracy plus AMBIGUOUS/REJECTED rates;
   - explicitly label this as an upper bound, not runtime performance.
2. If the upper bound is acceptable, define a **TRAIN-derived deterministic runtime digit proposal/localizer** inside the M3-C2 meter zone.
3. Evaluate that proposal/localizer on VALIDATION without tuning from VALIDATION and with TEST closed.
4. Only then run true runtime end-to-end `Presence -> runtime slots -> digit AIs -> composer` comparison against D11.

If the GT-slot upper bound itself has material joint-specialist conflicts, fix that evidence boundary before spending effort on runtime localization.

## Safety

- training: NO
- optimizer steps added: 0
- sealed TEST: CLOSED
- private checkpoint binaries in GitHub/CI: NO
- runtime Resolver wiring: NO
- production promotion: NO
- merge authorization: NO
