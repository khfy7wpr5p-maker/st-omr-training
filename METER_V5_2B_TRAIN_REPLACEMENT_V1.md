# Meter V5-2B — bounded adaptation-TRAIN replacement v1

## Trigger evidence

The approved center-y -> unique-containing-staff preflight completed 298/300 PASS and 2 HOLD. The two HOLDs are:

- `150201200-1_1_1` — `2/4`, selection index `63`, non-seed TRAIN, `A04_PAGE_CROPPED`;
- `110003725-1_1_1` — `3/4`, selection index `125`, non-seed TRAIN, `A04_PAGE_CROPPED`.

The first 30 diagnostic seeds are unaffected.

## Approved replacement rule

For each HOLD, replace it only with the deterministic first unused TRAIN row in the same meter class under the original V5-2A ordering:

`(selection_rank, sample_id)`.

The candidate must also have an unused family identity. The selector does not use model inference, geometry quality, confidence, or validation evidence to skip ahead to another candidate.

After deterministic selection, the chosen next-unused candidate receives a page-geometry safety screen. If that exact candidate is not geometry-accepted, the replacement plan is HOLD; the implementation does not silently choose a later candidate.

## Mutation boundary

Only the two approved non-seed rows may change. Their existing annotations are removed. The other 298 annotations remain byte-for-byte row-preserved. The new two samples remain unannotated until the user draws new full-meter BBoxes.

Before mutation, the current selection, annotation, mechanical audit, human-QA attestation, and staff-preflight evidence are archived. After mutation, stale active audit/QA/preflight/slot/training evidence is invalidated so old hashes cannot authorize the new selection.

## Gates after replacement

1. 300 selection rows remain exactly 100/class and 30 diagnostic seeds remain unchanged.
2. Annotation count becomes 298 until the two new human BBoxes are supplied.
3. Mechanical audit and human visual QA must be re-established on the new 300/300 set.
4. The 300-sample staff-association preflight must return 300/300 PASS.
5. Only then may 600-slot derivation and the frozen 2-AI/3-AI adaptation training lane open.

Throughout this replacement stage:

- `TRAINING=false`;
- `VAL=CLOSED`;
- `FINAL_HOLDOUT=LOCKED`;
- `4-AI=FROZEN`;
- threshold tuning remains forbidden;
- no midpoint, tight-digit GT, model-generated GT, automatic BBox correction, nearest-staff fallback, or staff tolerance expansion is introduced.
