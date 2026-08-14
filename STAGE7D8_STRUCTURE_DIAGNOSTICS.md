# Stage 7-D8 — Structure validation diagnostics

## Purpose

Stage 7-D8 is a **validation-only diagnostic package** for the accepted D7 Structure specialist. It does not retrain, fine-tune, calibrate, or mutate either D7 model.

D7 established that the Structure model learned useful system/measure/clef geometry, but its thin/sparse channels remained much weaker:

```text
system_region   Dice 0.93046746804164
measure_region  Dice 0.8445145579484793
clef_g2         Dice 0.8228637140530807
barline         Dice 0.2667824041384917
meter_2_4       Dice 0.34398488560691476
meter_3_4       Dice 0.34151152062874574
meter_4_4       Dice 0.3092358358777486
```

D8 measures **why** those channels are weak before any training-policy change is allowed.

## Accepted D7 artifact binding

D8 accepts exactly one D7 external run:

```text
run ID              4ce2903206c7965471bb9569d379d8d9d1022d9248d80886638acfe0bd822598
D7 repository head  25bdf2b3146faba54a93c00f05537f522c75b532
profile fingerprint 7b7fbc79c748da0f1195bc9273fe012e0b1128b3a1e491bb484653d47cb5201a
checkpoint SHA-256  5f009ca8ba68d38497a7dd25590d4dd98c537f20c5d5525bf66e288afbf417dc
metrics SHA-256     43cd98a75c2db740b4af6ee3c8826122fa387347820d2e7d2c639ac2fe30f792
verification SHA    cdc0733af1bd6c7336f5bd2a0cb12fcae269120d8b5a9a564f08db860ee21a0a
Structure state SHA 0d11b2ae414959b678ccc22a6b8cfcc1edc1ecadc3c73ed6ab5a0cda6e593907
```

The checkpoint is loaded only with `weights_only=True`, both Staff and Structure state dictionaries are strict-loaded, and their exact accepted state hashes must reproduce before diagnostics continue.

## Data boundary

D8 reuses the accepted D6/D7 data-verification path, but only VALIDATION records are tensorized for diagnostic inference.

```text
TRAIN       1,230 records → no D8 tensors / no optimizer
VALIDATION    153 records → read-only inference + diagnostics
TEST              0 D7 records → sealed
optimizer steps    0
```

A TEST record continues to fail at the inherited D7 split boundary. D8 contains no optimizer construction, backward pass, or optimizer step.

## Diagnostic questions

D8 is designed to separate three broad failure modes without automatically changing architecture:

1. **Threshold/calibration issue** — does a different fixed probability threshold materially improve Dice?
2. **Thin-object localization issue** — does 1–2 pixel tolerant F1 improve strongly over exact overlap?
3. **Sparse/representation limitation** — are positive pixels/positive records extremely rare while threshold/tolerance gains remain small?

The final interpretation is made from evidence after the authoritative D8 run; D8 itself does not silently select a new loss, target representation, crop policy, or model.

## Frozen threshold sweep

Every Structure channel is measured at:

```text
0.05, 0.10, 0.15, ... 0.90, 0.95
```

For each threshold D8 accumulates global VALIDATION:

- true positive pixels;
- false positive pixels;
- false negative pixels;
- precision;
- recall;
- Dice/F1;
- predicted-positive pixel fraction.

The best threshold is selected deterministically by:

1. highest Dice;
2. closest threshold to `0.50` on ties;
3. lower threshold on any remaining tie.

The accepted D7 `0.50` threshold remains explicitly reported; D8 does **not** replace it.

## Sparsity / confidence evidence

For every channel D8 reports:

- number of validation records containing at least one positive target pixel;
- positive pixel count;
- total pixel count;
- positive pixel fraction;
- mean predicted probability on positive target pixels;
- mean predicted probability on negative target pixels.

This shows whether low Dice is associated with severe target sparsity or poor probability separation.

## Tolerant localization evidence

D8 also reports bounded morphology-based precision/recall/F1 at pixel radii:

```text
1 pixel
2 pixels
```

for both:

- threshold `0.50`;
- each channel's diagnostic best threshold.

A large tolerant-F1 increase is evidence that exact overlap is penalizing near-miss localization, especially for thin barlines. It is diagnostic evidence only; D8 does not widen GT masks or alter D7 labels.

## Baseline reproduction gate

Before threshold/sparsity/tolerance diagnostics, D8 must reproduce the accepted D7 Structure result exactly under the pinned runtime:

```text
validation loss  0.49127569106908947
```

and all seven accepted D7 channel Dice values. If reproduction drifts, D8 fails closed instead of producing a new interpretation.

## Output

D8 writes outside normal Git into a fresh output directory:

```text
structure-diagnostic-<sha256>.json
COMPLETE
```

The canonical JSON report binds:

- current clean D8 repository identity;
- pinned runtime;
- exact accepted D7 artifact hashes/state;
- validation cardinality;
- baseline reproduction;
- threshold sweep;
- sparsity/confidence statistics;
- 1–2 pixel tolerant localization statistics;
- explicit `optimizer_steps = 0`;
- explicit `sealed_test_split_opened = false`;
- explicit `model_mutated = false`.

## Closure rule

D8 closes only after:

1. focused tests pass;
2. full regression and exact-head CI pass;
3. authoritative external D8 VALIDATION diagnostic runs against the exact accepted D7 bundle;
4. report hash is persisted outside Git;
5. results are interpreted before selecting any Structure refinement package.

No TEST evaluation is allowed to choose the refinement.
