# M4-E3K-B1-R1 — D7 Staff Decoder Root-Cause Audit

## Why this audit exists

B1 replaced authoritative D6 staff geometry with frozen D7 StaffSet predictions.
Its TRAIN result failed strongly:

- systems: 2,346
- matched systems: 622
- unmatched systems: 1,724
- system/staff match coverage: 0.2651321398124467
- decoded D7 staff count/page P50/P95: 0 / 2
- boundary recall @ 1.0 staff-space: 0.00040032025620496394
- B1 PASS: false

R2 had already shown that the unchanged inward-endpoint boundary proposal reaches
0.9818521483853749 recall @ 1.0 staff-space when authoritative D6 staff geometry
is supplied. Therefore B1-R1 stops before boundary proposal scoring and asks only:

> At which exact frozen D7 -> explicit five-line staff decoding stage are staff
> instances lost?

## Frozen invariants

B1-R1 changes no behavior:

- D7 checkpoint SHA-256:
  `5f009ca8ba68d38497a7dd25590d4dd98c537f20c5d5525bf66e288afbf417dc`
- D7 StaffSet state SHA-256:
  `3131548548521229e6acd6fee8cffc66081cb54125645f9eff5a488de7603af8`
- D7 input: 96 x 512
- D7 dense threshold: 0.50
- staff-region component width gate: >= 8% input width
- staff-region component area gate: >= 0.2% input area
- exact five-line template contract: unchanged
- line x-support contract: unchanged
- no threshold sweep
- no R2 proposal execution
- no D11
- no model training

## Diagnostic trace

For every accepted TRAIN page the audit records:

1. raw `staff_region` probability max/mean
2. raw `staff_lines` probability max/mean
3. number/fraction of pixels >= frozen 0.50 threshold
4. raw 8-connected `staff_region` component count
5. component size-gate rejection counts
6. qualifying component count
7. common-slope pass/fail count
8. five-equal-line-template pass/fail count
9. five-line x-support pass/fail count
10. final decoded staff count
11. truth system count for evaluation-only count comparison
12. a terminal page-level root-cause category

The audit also runs the unchanged B1 `decode_d7_staff_geometry()` on the same
probability tensor and requires its decoded staff count to equal the diagnostic
trace count. Any disagreement fails closed.

## Root-cause categories

The report can distinguish, without changing thresholds:

- `NO_REGION_SUPPORT_AT_FROZEN_THRESHOLD`
- `NO_RAW_REGION_COMPONENT`
- `ALL_REGION_COMPONENTS_REJECTED_BY_FROZEN_SIZE_GATES`
- `ALL_QUALIFYING_COMPONENTS_FAILED_SLOPE`
- `ALL_SLOPE_PASS_COMPONENTS_FAILED_FIVE_LINE_TEMPLATE`
- `ALL_TEMPLATE_PASS_COMPONENTS_FAILED_X_SUPPORT`
- `DECODED_STAFF_COUNT_BELOW_TRUTH_SYSTEM_COUNT`
- `DECODED_STAFF_COUNT_EQUALS_TRUTH_SYSTEM_COUNT`
- `DECODED_STAFF_COUNT_ABOVE_TRUTH_SYSTEM_COUNT`

## Decision rule after the audit

This audit has no PASS gate and authorizes nothing by itself.

- If raw D7 `staff_region` / `staff_lines` support is already absent, investigate
  the D7 StaffSet model/inference surface before changing decoder rules.
- If raw support is present but components are lost at one deterministic decoder
  stage, isolate that stage in a separate recovery experiment.
- Do not tune the 0.50 threshold from this report.
- Do not open B2 or D11 from this report.

## Safety

- TRAIN only: 1,230 records
- expected truth systems: 2,346
- VALIDATION closed
- TEST closed
- Final-A/B closed
- optimizer steps: 0
- training: false
- threshold tuning/sweep: false
- production promotion: false
- merge requested: false
