# Meter V5-2B — 2-AI / 3-AI Specialist Adaptation Contract

Status: **TRAIN-only candidate lane**. Runtime, Resolver, VAL, FINAL_HOLDOUT and production promotion remain closed.

## Entering evidence

The immutable 30-sample V5/package_ab diagnostic established a class-specific transfer failure under one common preprocessing/geometry path:

- 2/4: 0/10 full-meter PASS; true-2 2-AI score median ~0.0704, max ~0.1508 versus frozen threshold 0.48.
- 3/4: 0/10 full-meter PASS; true-3 3-AI scores effectively ~0 versus frozen threshold 0.60.
- 4/4: 9/10 full-meter PASS with the frozen 4-AI threshold 0.47.
- denominator exact-4: 26/30.

Therefore V5-2B adapts **only 2-AI and 3-AI**. 4-AI is a read-only control. Threshold tuning is forbidden.

## Human evidence gate

V5-2A produced exactly 300 human full-meter TRAIN BBoxes (100/class). Mechanical audit passed 300/300 with no REVIEW and zero seed mutation. The user then reviewed all 15 contact sheets and explicitly reported no visual errors.

V5-2B must bind a human-QA attestation to the exact selection CSV, annotation CSV and mechanical audit JSON before any slot derivation or training can run.

## Data roles

The first immutable 30 V5-1 seed samples are **diagnostic-only** in V5-2B and MUST NOT enter gradient updates. This preserves the existing 10/class baseline as an unseen development gate.

The remaining 270 human-annotated TRAIN samples are adaptation training data:

- 90 x 2/4
- 90 x 3/4
- 90 x 4/4

VAL is not read. FINAL_HOLDOUT is not read.

## Approved slot derivation

The approved deterministic rule is reused without modification:

- full-meter human BBox center supplies x;
- exactly one accepted staff is required, otherwise fail closed;
- numerator center = second staff line;
- denominator center = fourth staff line;
- slot width = `1.5960569245912566 * staff_spacing`;
- slot height = `2.0 * staff_spacing`;
- no midpoint split;
- no tight-digit human GT;
- no model-generated GT;
- no automatic BBox correction.

Source `P` palette PNGs may be deterministically decoded to RGB **inside this TRAIN-only adapter**, with explicit provenance `P_TO_RGB_TRAINING_V1`; source files remain read-only. This is not a runtime-normalizer promotion.

Each derived slot is converted to the historical specialist input contract: grayscale crop -> aspect-preserving LANCZOS thumbnail without upscaling -> centered white 64x64 canvas -> uint8/255 at model input.

## Specialist training

Historical frozen checkpoint identity is required before initialization:

- 2-AI SHA-256 `92b985d989e4338e3ae39b0a984879f4188be32c0d281390839117e1e9a715fa`
- 3-AI SHA-256 `5ee45faf2efe0e2c83dbad716736d7ae16ad7251730431d368c10c4574836485`
- 4-AI SHA-256 `dcd582b60b39e65798aa77aacea3cc797cd7513b7925151f0573be4aec6af43f`

2-AI and 3-AI keep the historical binary architecture. One fixed CPU-only deterministic fine-tuning configuration is allowed; there is no hyperparameter sweep and no threshold tuning. Every 270 TRAIN sample contributes a numerator and denominator slot. For each specialist, the target digit numerator is positive and every other slot is negative.

4-AI is never optimized or rewritten.

## Diagnostic gate after training

Candidate 2-AI / 3-AI and frozen 4-AI are evaluated on the 30 diagnostic seeds using unchanged thresholds and exactly-one arbitration on numerator and denominator slots.

Minimum development gate:

- 2/4 full-meter PASS >= 8/10
- 3/4 full-meter PASS >= 8/10
- 4/4 full-meter PASS >= 9/10
- denominator exact-4 >= 26/30

A PASS does **not** open production. It only authorizes the next isolated human-validation-BBox stage. VAL remains closed until that stage is explicitly created. FINAL_HOLDOUT remains locked throughout.
