# Meter V4-4 — Final Holdout BBox Annotation Contract

## Purpose

V4-4 is the **human annotation-only** gate between the frozen V4-3 final-holdout selection and the later V4-5 one-time independent evaluation.

Frozen V4-3 selection:

- selected count: `150`
- class balance: `2=50`, `3=50`, `4=50`
- frozen selection SHA-256: `4335a48a091912ba422c16d8fcbaaa7bbf5f7a0a43f088146a50a3e02e3ed7dc`

This stage must not open the V4-2 candidate checkpoint, run inference, inspect final predictions, tune thresholds, retrain, connect runtime/Resolver, or authorize production promotion.

## BBox semantic contract

Each bbox covers the **complete Meter sign** in the original `image.png` pixel coordinate system:

- numerator digit;
- denominator digit;
- the full vertical Meter glyph region needed to contain both.

It is not a numerator-only bbox.

Fields written to `bbox_meter.txt`:

- `bbox_x`
- `bbox_y`
- `bbox_w`
- `bbox_h`

Required geometry:

- integer pixels only;
- `x >= 0`;
- `y >= 0`;
- `w > 0`;
- `h > 0`;
- `x + w <= image_width`;
- `y + h <= image_height`.

Normalized, floating, 64x64, D11 256x192, or resized-preview coordinates are not persisted.

The Colab UI may display a resized preview. Browser pointer coordinates are rounded to integer preview pixels, then mapped deterministically back to the original image using floor for the left/top edge and ceil for the right/bottom edge so the original region is not silently cropped.

## Frozen-selection boundary

The tool accepts only the V4-3 manifest schema:

`st-omr-meter-v4-3-final-holdout-admission-manifest-v1`

Before any write it requires:

- `selected_count = 150`;
- class counts `50/50/50`;
- 150 unique `family_id` values;
- 150 unique selected folder names;
- frozen selected payload reproduces the exact selection SHA;
- V4-3 safety flags remain closed.

Manifest `image_path` and `bbox_path` strings are **not trusted write targets**. The selected sample folders are rediscovered under the supplied final-holdout root by the frozen folder-name binding. Path traversal, symlinks, missing files, duplicate selected folders, and root escape fail closed.

No non-selected sample is writable through the V4-4 session.

## Image binding before first annotation

V4-3 did not include image content hashes in the frozen selection SHA. V4-4 therefore creates a separate, immutable first-read image binding before the first bbox write:

`FINAL_HOLDOUT_150_V4_4_IMAGE_BINDING.json`

It binds every selected sample to:

- selection SHA;
- index;
- numerator/meter class;
- folder name;
- family id;
- image SHA-256;
- image byte size;
- original width/height.

The initial binding may be created only while all selected bbox fields are still blank. On every later resume the current images must reproduce that binding exactly.

Unexpected image formats fail closed. Selected images must be regular non-symlink PNG files inside bounded size/dimension limits.

## `bbox_meter.txt` compatibility

The existing text contract is preserved. V4-4 requires at least:

- `id`
- `meter`
- `split`
- `bbox_x`
- `bbox_y`
- `bbox_w`
- `bbox_h`
- `admit`
- `notes`

`meter` must agree with the frozen manifest sample.

V4-4 changes only the four bbox values. It verifies that protected fields (`id`, `meter`, `split`, `admit`, `notes`) are unchanged after write.

Malformed/partial bbox values, duplicate fields, float/NaN/string injection, wrong meter, out-of-bounds values, invalid UTF-8, or oversized files fail closed.

## Save, resume, and crash recovery

`SAVE BBOX` is the only operation that updates a sample `bbox_meter.txt`.

Each save:

1. verifies the browser sample-binding token;
2. rechecks the selected image SHA/dimensions;
3. validates preview coordinates;
4. maps to original integer pixels;
5. re-reads and validates `bbox_meter.txt`;
6. detects a concurrent target change;
7. atomically replaces the bbox file;
8. re-reads the written file and verifies protected fields;
9. atomically updates progress.

Progress file:

`FINAL_HOLDOUT_150_BBOX_PROGRESS.json`

It stores:

- frozen selection SHA;
- image-binding SHA;
- current index;
- review flags;
- saved annotation bindings;
- all model/TEST/runtime/production safety flags closed.

Write order is bbox first, progress second. If the runtime dies after bbox replacement but before progress replacement, restart reconciliation imports the valid on-disk bbox into progress. The opposite direction (progress says annotated but bbox file is blank/different) fails closed.

## Browser sample binding

Every UI sample has a deterministic token bound to:

- frozen selection SHA;
- image-binding SHA;
- selected index;
- folder name;
- family id;
- image SHA.

A stale token cannot redirect a write to another selected sample.

## Mechanical QA gate

After all 150 bboxes are complete, V4-4 mechanical QA requires:

- annotated count `150`;
- missing bbox `0`;
- invalid bbox `0`;
- unique families `150`;
- class counts `50/50/50`;
- frozen selection SHA unchanged;
- image binding unchanged;
- every bbox inside original image bounds;
- unresolved review flags `0`;
- model evaluated `false`;
- inference count `0`;
- candidate checkpoint opened `false`;
- TEST opened `false`;
- runtime connected `false`;
- production promotion authorized `false`.

Mechanical QA creates:

`FINAL_HOLDOUT_150_BBOX_COMPLETE.json`

The deterministic `bbox_manifest_sha256` binds the 150 image/bbox records.

## Human visual review gate

Mechanical QA does not authorize V4-5.

After mechanical QA, V4-4 generates contact sheets under:

`FINAL_HOLDOUT_150_BBOX_REVIEW/`

A human must visually verify that every bbox:

- contains both Meter digits;
- does not cut numerator/denominator;
- is attached to the correct Meter;
- does not drift onto another symbol;
- corresponds to a visible Meter sign.

The completion receipt intentionally records:

`human_visual_review_passed = false`

until a separate explicit human-review record is created in a later bounded step.

## Security / contamination invariants

V4-4 never:

- imports V4-2 model code for inference;
- opens candidate checkpoint bytes;
- uses holdout labels for training, augmentation, calibration, checkpoint selection, threshold tuning, early stopping, or hyperparameter search;
- changes the frozen 150-family selection;
- replaces failed/ambiguous holdout samples;
- reads sealed TEST surfaces;
- connects runtime or Resolver;
- promotes production.

Any selection mismatch, image mutation, malformed bbox, path/symlink escape, incomplete state, or deterministic-reconciliation failure stops the stage.

## Human handoff

The user runs the bounded Colab notebook on the V4-4 branch, draws one full-Meter bbox per selected image, and presses `SAVE BBOX`.

The UI shows:

- Meter class;
- family id;
- original image dimensions;
- `completed / 150`;
- previous / next;
- review/skip flag;
- deterministic resume after restart.

When annotation reaches `150/150`, run the mechanical QA cell. The next allowed action is human contact-sheet review, **not V4-5 inference**.
