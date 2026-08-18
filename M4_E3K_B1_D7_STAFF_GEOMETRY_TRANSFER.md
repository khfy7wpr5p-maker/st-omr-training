# M4-E3K-B1 — Frozen D7 StaffSet Geometry Transfer

## Purpose

R2 proved that the inward-endpoint deterministic proposal rule reaches the
TRAIN upper-bound boundary gate when authoritative D6 staff geometry is given:

- 7,494 interior TRAIN boundaries
- recall @ 1.0 staff-space: `0.9818521483853749`
- gate: `>= 0.98`

B1 changes exactly one variable: proposal geometry now comes from the accepted
frozen Stage 7-D7 **StaffSet** model (`staff_lines` + `staff_region`).

## Frozen D7 identity

- checkpoint SHA-256:
  `5f009ca8ba68d38497a7dd25590d4dd98c537f20c5d5525bf66e288afbf417dc`
- StaffSet state SHA-256:
  `3131548548521229e6acd6fee8cffc66081cb54125645f9eff5a488de7603af8`
- input: `96 x 512`
- channels: `staff_lines`, `staff_region`
- dense threshold: `0.50`

No threshold sweep is permitted.

## Frozen geometry decoder

1. Threshold D7 `staff_region` at 0.50.
2. Decode 8-connected components.
3. Preserve the V6 component gates:
   - width >= 8% of D7 input width
   - area >= 0.2% of D7 input area
4. Inside each accepted component, estimate common staff slope from the mean
   threshold-supported `staff_lines` y-position per x.
5. Deskew the frozen `staff_lines` probabilities.
6. Decode **exactly five equal-spaced lines** by deterministic exhaustive
   maximization of probability support at the frozen 0.50 threshold.
7. Recover each line's threshold-supported x extent.
8. Map bbox, five lines and spacing back to original page coordinates.
9. Feed that predicted geometry to the already-frozen R2 inward-endpoint
   proposal algorithm.

A component that cannot satisfy the five-line contract is dropped; geometry is
never invented.

## Why this is B1, not full deployment

To keep one major variable at a time, D6 truth system bboxes are still used
**only for evaluation association and x-search bounding**. They do not provide
staff bbox, staff lines or staff spacing to the proposal algorithm.

This isolates the question:

> Does the R2 boundary recall survive when D6 truth staff geometry is replaced
> by actual frozen D7 StaffSet geometry?

D7 `system_region` is intentionally not introduced yet. If B1 passes, a later
B2 step may replace the truth system association with predicted system geometry.

## TRAIN-only gates

- system/staff association coverage >= `0.98`
- boundary recall @ 1.0 authoritative staff-space >= `0.98`

Both must pass.

## Safety

- TRAIN only: 1,230 records
- expected systems: 2,346
- expected interior boundaries: 7,494
- VALIDATION closed
- TEST closed
- Final-A / Final-B closed
- D7 weights: read-only frozen load
- D11 weights: not loaded
- no optimizer
- no training
- no threshold tuning
- no promotion
- no merge requested

A B1 pass may authorize only a B2 predicted-system-geometry development step.
It does not authorize D11 integration or production promotion.
