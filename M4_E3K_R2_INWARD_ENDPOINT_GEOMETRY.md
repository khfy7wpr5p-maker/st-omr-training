# M4-E3K-R2 — Inward Endpoint Geometry Recovery

## Why R2 exists

Accepted TRAIN R1 evidence on 7,494 interior boundaries found 4,630 misses at the frozen 1.0 staff-space tolerance. The dominant miss reason was:

- `BOTTOM_ENDPOINT_FAIL`: 3,721
- `COMBINED_GATE_FAIL`: 789
- `CLUSTER_PEAK_DISPLACEMENT`: 56
- `BOTH_ENDPOINTS_FAIL`: 42
- `VERTICAL_COVERAGE_FAIL`: 17
- `TOP_ENDPOINT_FAIL`: 5

The same R1 report showed clean-image recall only about 0.2046 and zero-rotation recall about 0.2064, while the median true barline length remained 4.0 staff-spaces. This supports a geometry-contract failure rather than a model or degradation-first explanation.

## One-variable change

V1 endpoint evidence used symmetric windows around the top and bottom staff lines. That asks a true barline ending at the bottom staff line to continue below its actual endpoint.

R2 changes only endpoint-window direction:

- top endpoint window: from the top staff line inward/downward;
- bottom endpoint window: from the bottom staff line inward/upward.

To isolate direction rather than window length, R2 preserves the V1 symmetric window's nominal total span. With frozen `endpoint_half_window_staff_spaces = 0.30`, R2 samples `2 * 0.30 = 0.60` staff-spaces on the inward side.

## Frozen values retained

No threshold or proposal-policy tuning is allowed in R2:

- horizontal probe radius: 0.10 staff-space
- endpoint half-window source value: 0.30 staff-space
- minimum vertical coverage: 0.45
- minimum endpoint coverage: 0.50
- cluster gap: 0.20 staff-space
- maximum proposals/system: 128
- recall gate @ 1.0 staff-space: >= 0.98

Otsu thresholding, staff-slope recovery, perpendicular probing, clustering, candidate selection, and candidate bounds remain unchanged.

## Scoring surface

R2 first replays only the accepted TRAIN surface:

- records: 1,230
- systems: 2,346
- topology-relevant interior boundaries: 7,494
- authoritative D6 ground-truth staff geometry only

A TRAIN pass does not itself authorize D11, promotion, TEST, or merge. VALIDATION remains unopened until a later explicitly approved stage.

## Safety

- no model training
- optimizer steps = 0
- no D7/D11 weights loaded
- no threshold tuning
- VALIDATION closed
- TEST closed
- Final-A / Final-B closed
- no production promotion
- PR #58 remains FAIL-preserved
- PR #59 remains diagnostic and unchanged
- R2 is a separate stacked draft branch
