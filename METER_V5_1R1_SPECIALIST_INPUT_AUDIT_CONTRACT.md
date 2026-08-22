# METER V5-1R1 — Specialist Input Contract Audit

Status: **PREREGISTERED / TRAIN-PILOT-30 ONLY / INFERENCE-ONLY**

## Purpose

V5-1R1 answers one narrow question before any 1,200-sample annotation scale-up:

> Can the already-human-approved full-meter BBox be converted by one fixed deterministic rule into numerator/denominator crops that the already-frozen 2-AI / 3-AI / 4-AI specialists read correctly?

This stage does **not** train a model, tune a threshold, open validation/final-holdout images, change runtime geometry, connect the Resolver, or authorize production promotion.

The canonical Meter architecture remains:

```text
visible Meter
  -> deterministic digit slots
  -> frozen 2-AI / 3-AI / 4-AI
  -> exactly-one arbitration per slot
  -> deterministic 2/4 | 3/4 | 4/4 composition
  -> fail closed on no-hit / conflict / unsupported evidence
```

V5-1R1 is only a bounded input-contract audit for the 30 already-annotated TRAIN pilot samples.

## Frozen parent evidence

The audit is stacked on exact V5-1 head:

`4f744d0bf8a2f6180a62f0f08abb96b83cfb5da8`

Required repository evidence:

`evidence/METER_V5_1_BBOX_PILOT_RESULT.json`

The evidence must state:

- dataset = `METER_V2_1500_PACKAGE_AB_CLEAN`;
- pilot result = `PASS`;
- annotation count = 30;
- PASS count = 30;
- REVIEW count = 0;
- 10 samples each for `2/4`, `3/4`, `4/4`;
- original pilot image binding preserved;
- final holdout locked;
- training not authorized;
- model not opened in V5-1.

The current pilot selection CSV and annotation CSV must hash exactly to the SHA-256 values bound by that evidence before any image or checkpoint inference is used.

## Admitted data surface

Only the persisted V5-1 pilot selection is admitted:

- `30` TRAIN samples total;
- `10` `2/4`;
- `10` `3/4`;
- `10` `4/4`;
- all annotation rows must be `PASS`;
- all selected image SHA-256 and dimensions must match the frozen selection and annotation records.

The audit must not enumerate, decode, hash, display, or infer on `val` or `final_holdout` images.

## Frozen full-meter -> digit-slot policy

No parameter may be tuned after specialist results are seen.

For an approved full-meter integer BBox `(x, y, w, h)`:

```text
x0 = x
x1 = x + w
y0 = y
y1 = y + h
split_y = y + floor(h / 2)

numerator   = (x0, y0,      x1, split_y)
denominator = (x0, split_y, x1, y1)
```

Rules:

- no learned localizer;
- no D11 bbox;
- no staff-line redetection;
- no threshold/morphology/contrast cleanup;
- no per-class or per-sample crop adjustment;
- no padding search;
- no alternate split search;
- no retry using model predictions.

If either derived slot has non-positive area or lies outside the exact source image, the sample fails closed.

## Frozen 64x64 specialist preprocessing

Each derived slot uses the historical 2-AI / 3-AI / 4-AI pixel contract already fingerprinted by `meter_v2_digit_crop_profile_v1`:

1. source coordinates are integer pixel bounds;
2. crop is clipped only by the already-validated source bounds;
3. crop is converted to grayscale `L`;
4. `thumbnail((64,64), LANCZOS)` preserves aspect ratio and does not upscale;
5. result is centered on a white `64x64` canvas;
6. model tensor is `float32(gray_uint8 / 255.0)`, shape `[1,1,64,64]`.

No image preprocessing variant is admissible in this audit.

## Frozen specialist identities and thresholds

Private checkpoint bytes remain outside GitHub/CI. External execution must first pass the merged strict checkpoint audit.

Frozen identities:

- 2-AI SHA-256: `92b985d989e4338e3ae39b0a984879f4188be32c0d281390839117e1e9a715fa`
- 3-AI SHA-256: `5ee45faf2efe0e2c83dbad716736d7ae16ad7251730431d368c10c4574836485`
- 4-AI SHA-256: `dcd582b60b39e65798aa77aacea3cc797cd7513b7925151f0573be4aec6af43f`

Frozen decision thresholds:

- 2-AI: `0.48` / `480` milli;
- 3-AI: `0.60` / `600` milli;
- 4-AI: `0.47` / `470` milli.

Probability-to-milli transport must use the already-merged conservative floor quantizer. Ordinary rounding is forbidden because it can promote a sub-threshold score.

## Per-slot arbitration

For each numerator and denominator crop, all three frozen specialists execute.

```text
exactly one specialist >= its threshold -> UNIQUE
zero specialists >= threshold           -> NO_HIT
more than one >= threshold               -> CONFLICT
```

Confidence is never allowed to break a conflict.

Expected visual digits are fixed from the human meter label:

```text
2/4 -> numerator 2, denominator 4
3/4 -> numerator 3, denominator 4
4/4 -> numerator 4, denominator 4
```

A `UNIQUE` result is correct only when the selected digit equals the expected digit.

## Determinism gate

Every admitted 64x64 crop must be inferred ten times with the same audited CPU model states.

For each specialist and crop, all ten probability outputs must be exactly identical in the authorized runtime. Any replay difference fails the sample and the stage.

CI may use synthetic model states only to test implementation semantics. CI synthetic inference is not real-checkpoint evidence.

## Strict pilot decision gate

V5-1R1 returns `PASS_SCALE_ANNOTATION` only if **all** conditions hold:

- 30/30 admitted samples remain image/bbox identity-valid;
- 60/60 derived digit slots are valid;
- 60/60 slots produce deterministic 10/10 replay;
- 60/60 slots have exactly one passing specialist;
- 60/60 unique specialists equal the expected digit;
- conflicts = 0;
- no-hit = 0;
- wrong-unique = 0;
- sample-level correct composition = 30/30.

Any failure returns `HOLD_INPUT_CONTRACT`.

A HOLD must not trigger threshold tuning, crop-search, per-sample exceptions, or final-holdout access. It authorizes only a separate TRAIN-only root-cause review of the failed slot crops.

## Meaning of PASS

`PASS_SCALE_ANNOTATION` means only:

- the existing full-meter annotation contract is compatible with this one frozen top/bottom slot derivation on the 30 TRAIN pilot examples;
- scaling the same human full-meter BBox annotation contract may be considered as the next bounded data step.

It does **not** prove:

- runtime locator/geometry parity;
- real-page end-to-end Meter accuracy;
- validation/final-holdout performance;
- training readiness by itself;
- Resolver readiness;
- production readiness.

Runtime geometry -> slot parity remains a later independent gate.

## Immutable output

External execution writes a fresh output directory containing:

- `result.json` — canonical audit result with per-sample/per-slot scores, crop SHA-256, arbitration, expected digit, decision and bindings;
- `COMPLETE` — SHA-256 of `result.json`.

Existing or partial output is refused. Source images, pilot CSVs, annotations and checkpoint files are read-only.

## Safety invariants

- TRAIN pilot 30 only;
- validation closed;
- final holdout closed;
- no optimizer;
- no backward;
- no model mutation;
- no threshold tuning;
- no crop-policy search;
- no D11 execution;
- no Resolver wiring;
- no production promotion;
- no automatic learning from audit output.
