# M4-E3K — Deterministic Measure-Boundary Proposal Recovery

Status: **development-only implementation candidate; external TRAIN/VALIDATION scoring pending**.

## Root cause carried forward

M4-E3J established that D10 meter inference is measure-level while the failed
V1–V6 adapter family attempted to recover one meter anchor from staff/system
left geometry. That is structurally insufficient because visible time-signature
changes may occur at later measures inside the same system.

D7 `measure_region` is also not an instance mask: all measure rectangles are
rasterized into one dense channel and can merge. The accepted global D7
`barline` channel is too weak to promote directly to a boundary decoder. The
accepted D11 local barline refiner is strong, but it requires a measure-end ROI
and therefore needs a high-recall proposal source before it can act as a
validator/localizer.

## E3K V1 scope

E3K V1 implements only the deterministic proposal source:

```text
grayscale page
+ accepted staff/system geometry
        ↓
deterministic Otsu threshold inside staff band
        ↓
vertical support through first→fifth staff-line span
        ↓
outer-line endpoint support
        ↓
x clustering
        ↓
bounded vertical-stroke proposals
```

The proposal stage intentionally favors recall. Note stems or other vertical
strokes may survive and are not silently reclassified as barlines. A later
package may crop each proposal with the frozen D9/D10 measure-end geometry and
use the frozen D11 barline refiner to validate/refine it.

## Frozen V1 geometry policy

```text
horizontal probe radius       0.10 staff-space
endpoint half-window          0.30 staff-space
minimum vertical coverage     0.45
cluster gap                   0.20 staff-space
maximum proposals/system      128
```

These values are frozen before external E3K validation scoring. The grayscale
ink threshold is not tuned from labels; it is computed deterministically with
Otsu on the local staff surface.

Candidate lists are never top-k pruned. If the candidate count exceeds the
frozen bound, the package fails closed so that recall is not silently destroyed.

## Read-only scoring surface

Object-level boundary recall is reported at all three predeclared tolerances:

```text
0.5 staff-space
1.0 staff-space
2.0 staff-spaces
```

The initial development gate for deciding whether to proceed to the frozen D11
barline validator is:

```text
VALIDATION measure-boundary recall @ 1.0 staff-space >= 0.98
TEST opened                                      = false
training / optimizer steps                       = 0
threshold tuning after VALIDATION                = false
```

Proposal count distribution and nearest-boundary P50/P95 must also be persisted;
meeting recall by producing an unbounded candidate explosion is not acceptable.
No production promotion or meter scoring is authorized by this gate.

## Safety

- no D7 or D11 weight mutation;
- no new model or optimizer;
- no TEST access;
- no Final-A/Final-B access;
- no change to D11 proposal threshold or 2/3/4 specialist thresholds;
- no merge/promotion from proposal-unit tests alone;
- E3I PR #57 remains a separate fail-preserved experiment.

## Next evidence step

Run the E3K V1 proposal generator against the accepted development images and
D6 measure/barline geometry. Persist TRAIN and VALIDATION metrics separately.
If VALIDATION recall @ 1.0 staff-space is below 0.98, stop before D11 inference
and return to deterministic proposal root-cause analysis. If it passes without
candidate explosion, only then open the frozen-D11 validator package.
