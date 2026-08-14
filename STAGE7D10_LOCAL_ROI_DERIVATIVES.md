# Stage 7-D10 — Deterministic Local ROI Derivatives

## Purpose

Stage 7-D10 materializes the local data surface selected by the closed D9 Structure refinement contract. It does **not** train a model. The accepted D7 Structure core remains untouched and TEST remains sealed.

D10 converts accepted D6 final-PNG geometry into two higher-resolution local derivative families:

```text
measure end   -> barline ROI 192 x 128 -> barline_segment
measure start -> meter ROI   192 x 256 -> none|2/4|3/4|4/4 + meter_bbox
```

## Source authority

The authoritative runner does not accept an arbitrary subset. It reuses `load_verified_stage7d7_records`, which independently verifies the accepted D6 derivative bundle before exposing TRAIN/VALIDATION records.

The authoritative D10 source surface is therefore frozen to:

```text
TRAIN        1,230 images / 410 families
VALIDATION     153 images /  51 families
TOTAL        1,383 images / 461 families
TEST             0 records
optimizer         0 steps
```

D10 verifies the D6 label SHA, source PNG SHA, D6 schema/version, sample/family/split identity, grayscale PNG mode and source dimensions again before deriving an ROI.

## Frozen D9 ROI policies

Barline:

```text
anchor           measure end
x before         5.0 staff spacings
x after          1.5 staff spacings
vertical margin  3.0 staff spacings above/below
output           192 x 128
resize           aspect-preserving fit/pad
```

Meter:

```text
anchor           measure start
x before         0.5 staff spacing
x after          12.0 staff spacings
vertical margin  3.0 staff spacings above/below
output           192 x 256
resize           aspect-preserving fit/pad
classes          none | 2/4 | 3/4 | 4/4
```

The crop is computed in the native accepted final-PNG coordinate space from `measure_bbox`, the owning staff bbox and exact D6 `staff_spacing`. The same deterministic crop/resize/pad transform is replayed on target geometry.

## Meter semantic rule

`meter_class` in D6 represents the active canonical meter, while `meter_bbox` represents a visible current-measure meter glyph. D10 follows the visible-glyph task required by D9:

- `meter_bbox != None` -> target class must be `2/4`, `3/4` or `4/4` and the bbox is mapped into the ROI;
- `meter_bbox == None` -> target class is `none`, even when a meter remains active semantically.

This preserves the accepted courtesy/anticipatory meter correction from D6 and prevents a post-barline courtesy signature from becoming current-measure ground truth.

## Artifact identity

Each local record is hash-addressed and bound to:

- D10 version;
- D9 contract fingerprint;
- source sample id;
- source PNG SHA-256;
- accepted D6 label SHA-256;
- split;
- local task kind;
- measure number;
- frozen ROI policy id.

Each persisted label additionally records the exact ROI transform and the generated ROI PNG SHA-256.

Normal Git stores no ROI image corpus. The authoritative output must be created in a fresh external directory.

## Independent persisted-output gate

D10 does not trust successful writes. Before `COMPLETE` is emitted, `verify_stage7d10_derivatives` independently reopens the output and checks:

- canonical manifest and receipt bytes;
- manifest SHA sidecar;
- D9 fingerprint and exact repository SHA when authoritative;
- TRAIN/VALIDATION source/family cardinality;
- family split exclusivity;
- TEST=0 and optimizer=0;
- canonical hash-addressed artifact paths;
- every ROI PNG hash, grayscale mode and frozen D9 dimensions;
- every ROI label hash/schema/identity/source binding;
- recomputed record id provenance;
- ROI transform bounds and target bounds;
- exactly one barline and one meter record for every source measure;
- meter class/bbox coherence;
- no unexpected image/label artifacts;
- independently recomputed artifact-binding SHA;
- independently reconstructed receipt.

Only after this pass is the `COMPLETE` marker written; the completed bundle is then verified a second time.

## Safety boundary

D10 contains no model, optimizer, backward pass, checkpoint loader, DataLoader or TEST evaluation path. It does not mutate D7 weights and it does not manufacture ground truth.

A D10 success only means that the barline/meter local derivative dataset is complete, deterministic and auditable. It does **not** mean the local refiners have been trained or passed the D9 validation thresholds.

## Closure gate

D10 closes only after:

1. exact-head code/tests/CI pass;
2. independent P1/P2 review has no blocker;
3. authoritative external build consumes all 1,383 accepted D6 development records;
4. persisted-output verification passes;
5. external manifest/artifact-binding/receipt evidence is recorded;
6. TEST remains 0 and optimizer remains 0;
7. explicit merge approval is obtained.

The next package after closed D10 may implement/train the bounded `barline_refiner` and `meter_refiner` under the already-frozen D9 parameter and validation gates.