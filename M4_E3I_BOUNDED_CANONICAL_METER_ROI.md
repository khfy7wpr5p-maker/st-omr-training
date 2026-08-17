# M4-E3I — Bounded Two-Candidate Canonical D10 ROI Recovery

Status: **DEVELOPMENT IMPLEMENTATION — NOT PROMOTED**

M4-E3I repairs the meter adapter boundary identified by the E3H parity audit.
It does not retrain 2/4, 3/4, 4/4 specialists and does not modify frozen model
thresholds.

## Frozen safety boundary

- maximum page-space measure-start candidates per system: **2**
- D11 proposal threshold: **0.90**
- specialist thresholds: **2=0.48, 3=0.60, 4=0.47**
- D11 checkpoint SHA-256: `cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3`
- D7 checkpoint SHA-256: `5f009ca8ba68d38497a7dd25590d4dd98c537f20c5d5525bf66e288afbf417dc`
- optimizer steps added: **0**
- TEST records opened: **0**
- Final-A / Final-B: **sealed**
- production promotion: **forbidden in this package**

## TRAIN-only provenance

The measure-start offset remains the E3G/V6 TRAIN-only frozen p50:

`-0.06619667590040451 staff-spaces`

The source result is hash-bound as:

`db9536b983c7aabee30243696fb88e8ea74016b4600a70b93ad630562b8b86ec`

VALIDATION labels are not accepted by the anchor-policy object. TEST is not an
input surface.

## Candidate recovery

E3G/V6 used one candidate derived from the median left endpoint of five decoded
staff lines. E3I preserves that candidate and adds one bounded geometry-only
coverage fallback from the minimum real left endpoint among the same five
lines. Both receive the identical frozen TRAIN offset. Duplicate anchors are
collapsed, so each system emits one or two candidates only.

This fallback addresses the E3H failure mode where some decoded staff-line
fragments begin far to the right of the actual staff start. It introduces no
score search, threshold sweep, validation-derived scalar, or unbounded anchor
scan.

M3-C2's two local meter-zone modes are **not** reused as page-space measure
anchors; those coordinate spaces have different semantics.

## Canonical D10 parity

M4-E3I does not copy the D10 crop algorithm. Candidate ROI rendering delegates
directly to the existing D10 `_crop_transform` and `_render_roi` functions with
`METER_ROI` (`measure-start-meter-roi-v1`). Therefore floor/ceil clipping,
aspect-preserving bilinear resize, centered white padding, and 256x192 output
remain on the same code path as canonical D10.

A regression test constructs the same page-space measure-start anchor through
D10 and E3I and requires byte-identical ROI PNG bytes, identical SHA-256, and
identical ROI transform metadata.

## Acceptance gate

Implementation is not considered adapter success until frozen development
scoring independently satisfies all of the following without tuning:

- system match coverage >= 98%
- anchor error P95 <= 2.0 staff-spaces
- positive proposal recall >= 98%
- none proposal false-positive rate <= 5%
- exact accuracy >= 95%
- macro F1 >= 95%
- unknown rate <= 5%
- recall for each class >= 90%
- `ADAPTER PASS = TRUE`

Failure preserves the frozen models and returns to diagnosis. It does not open
TEST and does not trigger specialist retraining automatically.
