# Stage 7-D9 — Structure refinement contract

Status: architecture/contract only. No new training is permitted by this package.

## Evidence selecting D9

D9 is selected from the accepted Stage 7-D8 validation-only report:

- D8 repository head: `e0e721bf5a6d13025546fdf5eeb755647eef383f`
- D8 report SHA-256: `46de5f6766f78bb567f70794a364ccd44835d09af94ef29c3f1eab5cd13ce968`
- TEST opened: false
- optimizer steps in D8: 0

D8 showed that threshold calibration does not explain the weak Structure channels. Best-threshold Dice improved by only about 0.0–0.003, and 2-pixel tolerant F1 remained weak for barline and all meter channels. The weak channels also occupy an extremely small fraction of the whole-page raster.

## Problem in the D7 representation

D7 feeds the entire score page through one dense Structure segmentation model at `96 x 512`. Its encoder downsamples spatial resolution twice before decoding. This is adequate for large regions such as systems/measures and for the G2 clef, but it creates representation pressure for very thin or very small objects:

- trailing barlines;
- visible 2/4, 3/4, and 4/4 meter glyph regions.

The accepted D7 core is therefore not indiscriminately retrained.

## External contract remains unchanged

D9 does **not** change the frozen D4 `StructureSet` inference boundary. The external Structure task still emits:

- `system_bboxes`
- `measure_bboxes`
- `barline_positions`
- `clef_g2_candidate`
- `meter_candidate`
- `confidence`

Only the internal implementation is decomposed.

## Internal D9 decomposition

### 1. `structure_core`

The accepted D7 Structure path remains authoritative for its already-strong outputs:

- system region;
- measure region;
- G2 clef.

The accepted D7 core weights are frozen during the first refinement optimizer run. This prevents barline/meter work from degrading already-strong channels.

### 2. `barline_refiner`

Barline is moved out of the shared whole-page mask and into a dedicated local specialist.

Inputs:

- native-image measure-end ROI;
- five staff lines / staff spacing;
- measure bbox from the frozen core.

Frozen ROI policy:

- anchored to measure end;
- 5 staff spacings before the end and 1.5 after;
- 3 staff spacings vertical margin above/below the staff;
- fit/pad preserving aspect ratio;
- model surface `192 x 128`.

Output:

- `barline_segment` plus confidence.

This makes the thin barline a materially larger fraction of the model input instead of approximately one-tenth of one percent of the whole-page raster.

### 3. `meter_refiner`

Meter is moved to a measure-start ROI specialist.

Inputs:

- native-image measure-start ROI;
- five staff lines / staff spacing;
- measure bbox from the frozen core.

Frozen ROI policy:

- anchored to measure start;
- 0.5 staff spacing before and 12 after;
- 3 staff spacings vertical margin above/below the staff;
- fit/pad preserving aspect ratio;
- model surface `192 x 256`.

Outputs:

- meter class: `none | 2/4 | 3/4 | 4/4`;
- bounded meter bbox when visible;
- confidence.

The explicit `none` class is required because many measures carry an active meter semantically without displaying a new meter glyph. Courtesy signatures must not be silently rebound to the wrong current measure; the D6 semantic/geometry binding remains the source of training ground truth.

### 4. deterministic fusion

The frozen core and local refiners are fused deterministically into the unchanged external Structure outputs. Low-confidence or incoherent refinement results are unresolved/fail-closed; fusion must not invent a barline or meter candidate.

## Split and ground-truth policy

The existing safety boundary is unchanged:

- TRAIN: may update only the new barline/meter refiner weights;
- VALIDATION: read-only model selection and diagnostics;
- TEST: forbidden during D9 development;
- accepted D7 Structure core: frozen;
- no ScoreMosaic upload or teacher correction enters training automatically.

Synthetic local targets remain derived only from canonical music plus pinned renderer geometry plus deterministic final-PNG transforms. No learned prediction may become ground truth.

## Frozen pre-training acceptance gates

The gates are fixed before the next optimizer run:

- TEST records: `0`;
- accepted D7 core mutation: forbidden;
- total new trainable parameters: at most `1,250,000`;
- barline strict Dice: at least `0.500`;
- barline 2px tolerant F1: at least `0.700`;
- meter four-class macro F1 (`none/2/4/3/4/4/4`): at least `0.800`;
- meter positive-record localization 2px tolerant F1: at least `0.600`.

Failure to meet these validation gates does not authorize opening TEST or changing the thresholds after seeing results. A failed refinement requires another development package.

## Package closure

D9 contract closes only when:

1. contract/fingerprint is deterministic;
2. D4 external Structure outputs remain unchanged;
3. ROI and parameter bounds fail closed;
4. core freeze and TEST sealing are regression-tested;
5. focused tests + full regression + CI pass;
6. no optimizer/trainer/checkpoint execution path exists in the D9 contract module.

The next package, after D9 contract merge, will implement the derivative/ROI builder and training runner under this frozen contract. It must not train until its own code/tests/CI gates pass.
