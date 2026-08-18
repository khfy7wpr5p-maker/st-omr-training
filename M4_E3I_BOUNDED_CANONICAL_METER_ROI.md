# M4-E3I — Bounded Two-Candidate Canonical D10 ROI Recovery

Status: **DEVELOPMENT IMPLEMENTATION — GEOMETRY PREFLIGHT FAIL — NOT PROMOTED**

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
staff lines. E3I preserves that candidate and adds one bounded fallback from
the independent D7 staff-region component left edge. Both receive the identical
frozen TRAIN offset. Duplicate anchors are collapsed, so each system emits one
or two candidates only.

The staff-region fallback is deliberately different from taking the minimum of
the five decoded line endpoints. E3H/V6 evidence contains failure cases where
all five decoded line fragments begin at the same late X position while the D7
staff-region component still starts far to the left. In that case a min-line
fallback would be identical to the failed median candidate; the independent
staff-region edge preserves the missing coverage signal.

This introduces no score search, threshold sweep, validation-derived scalar, or
unbounded anchor scan.

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

## Frozen development geometry preflight

Before paying for D11 + specialist inference, E3I runs a stricter optimistic
preflight: for every development system it computes the error of both frozen
candidate rules and lets an oracle choose the better candidate. This is an
optimistic lower bound. A real selector cannot have lower anchor error than the
oracle best-of-two surface.

The audit is hash-bound to the already-persisted V4/V5/V6 development records:

- V4: `98de47c01c4f83eef1f30c6be143c9cfb19a5c63f8f6f6b5745da6d6808ce826`
- V5: `0b6a3229a97150597ffc54b81774a12810237ba0f40ea1f21f5f10745d3dde04`
- V6: `7a734bc9d08369da1e0592e22a167777ef587c490e6c3489de6b32d98cf44e06`

Result over 285 validation systems:

- geometry-scored systems: **283 / 285 = 99.298%**
- candidate-1 anchor P95: **21.619 staff-spaces**
- candidate-2 anchor P95: **17.916 staff-spaces**
- oracle best-of-two anchor P95: **16.293 staff-spaces**
- frozen gate: **<= 2.0 staff-spaces**
- oracle systems still above 2.0: **29 / 283**
- oracle systems still above 10.0: **26 / 283**

Therefore `ORACLE_ANCHOR_P95_PASS = FALSE`.

Because even an oracle selector cannot satisfy the frozen anchor gate, full
D11/specialist inference cannot make this E3I candidate surface pass the full
adapter gate. Model scoring is intentionally stopped before execution. No
threshold is tuned and no checkpoint is changed.

Detailed machine-readable evidence is frozen in
`M4_E3I_V1_GEOMETRY_PREFLIGHT.json` and the computation is implemented by
`st_omr_training/m4_e3i_geometry_preflight.py`.

## Acceptance gate

The original adapter gate remains unchanged:

- system match coverage >= 98%
- anchor error P95 <= 2.0 staff-spaces
- positive proposal recall >= 98%
- none proposal false-positive rate <= 5%
- exact accuracy >= 95%
- macro F1 >= 95%
- unknown rate <= 5%
- recall for each class >= 90%
- `ADAPTER PASS = TRUE`

Current state: **EARLY FAIL PRESERVED** at the anchor geometry gate. TEST remains
closed, specialist weights remain frozen, and PR #57 must not be promoted or
merged as an accepted adapter implementation.
