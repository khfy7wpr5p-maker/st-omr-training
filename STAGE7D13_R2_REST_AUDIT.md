# Stage 7-D13-R2.1 — Rest root-cause audit

Status: **implementation candidate — read-only derivative audit; external evidence pending**.

## Purpose

D13-R1 is not accepted as the final symbol-specialist stage. The Rest specialist
ended its observed ten-epoch profile with zero center F1, zero bbox F1 and zero
macro F1. D13-R2 therefore forbids a blind repeat of the same full Rest run.

The first R2 implementation gate reopens the frozen R1 derivative and asks a
narrower question before any optimizer is authorized:

> Did the existing 512×128 measure representation or Rest ground-truth geometry
> make the Rest task structurally difficult or invalid?

This package does not train a model and does not change the R1 derivative.

## Frozen R1 input identity

The audit accepts only the persisted D13-R1 measure derivative with:

```text
derivative build id
44f1932532fb511dfa59a164f94be6b899f3aa0594c0ac0a6f499a38e5fb5697

manifest SHA-256
8cfb87b5c6135be14b4c9ad488868c0edb0d37bb3bb18ad1b5e79d04fdf24f7b

artifact binding SHA-256
c42c1f69e21d61d3eefdacfc40dabf2f0fcd6ac2ceb4d5cf88d8e158246dd33e

TRAIN records        9840
VALIDATION records   1224
TOTAL records       11064
images              11062
labels              11064
TEST                    0

Rest targets
TRAIN               10602
VALIDATION            969
```

Any identity/cardinality drift fails closed.

## Audit surface

`st_omr_training.stage7d13_r2_rest_audit` performs a read-only pass over the
frozen manifest and all 11,064 canonical labels. It validates:

- exact build / manifest / artifact-binding identity;
- exact TRAIN/VALIDATION record counts;
- TEST absence before specialist payload use;
- canonical label bytes and label SHA-256;
- label/manifest record and image identity;
- frozen `512×128` grayscale measure dimensions;
- `half|quarter|eighth` Rest class boundary;
- finite in-bounds bbox and center geometry;
- center-inside-bbox invariant;
- class-agnostic stride-4 regression-cell collisions.

The audit deliberately does not open or decode image bytes in this first pass.
The representation question is measured from the already persisted transform and
transformed target geometry. Visual inspection and R1 checkpoint inference are a
separate follow-on diagnostic after this data gate passes.

## Geometry report

For TRAIN and VALIDATION separately the report records:

- Rest-positive and Rest-zero measure counts;
- per-class Rest counts;
- letterbox scale distribution;
- bbox width, height, area and minimum-dimension distributions;
- targets narrower/shorter than the frozen output stride (`4 px`);
- targets whose minimum dimension is below two output strides (`8 px`);
- per-class width/height/area distributions;
- deterministic smallest-area examples for `half`, `quarter`, and `eighth`.

The `4 px` boundary is not a newly tuned threshold. It is the frozen R1 output
stride and is used only to identify potential representation pressure.

## Safety

```text
TEST opened       false
optimizer steps   0
model loaded      false
R1 mutation       forbidden
report            fresh external JSON only
```

No output from this audit may become ground truth automatically.

## Decision gate

The external report is evidence for the next decision, not the decision itself.
After the report exists, R2-1 continues with visual and checkpoint/decoder
attribution to distinguish:

1. label/transform defect;
2. destructive measure-scale representation;
3. decoder/metric cliff;
4. localization/type-classification coupling;
5. optimizer/training instability;
6. mixed cause.

Only then may R2-2 freeze one Rest architecture: keep/refine, higher-resolution
ROI, detector→classifier split, or derivative repair.

## External execution example

```python
from st_omr_training.stage7d13_r2_rest_audit import (
    run_stage7d13_r2_rest_derivative_audit,
)

receipt = run_stage7d13_r2_rest_derivative_audit(
    "/path/to/stage7d13-measure-derivatives-d5fe4d2c120202ec7f962ef6d849b6e36af224ef",
    report_path="/external/path/d13-r2-rest-derivative-audit.json",
)
print(receipt)
```

The report path must be fresh. The authoritative derivative remains read-only.
