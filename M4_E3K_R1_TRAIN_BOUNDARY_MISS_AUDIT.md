# M4-E3K-R1 — TRAIN Boundary Miss Root-Cause Audit

Status: **diagnostic-only; TRAIN only; no promotion authority**.

## Trigger

Frozen E3K-A TRAIN scoring failed:

- records: 1230
- systems: 2346
- topology-relevant interior boundaries: 7494
- recall @ 1.0 staff-space: 0.38217240459033897
- target: >= 0.98
- E3K-B authorized: false
- D11 authorized: false

The miss is too large for validation-driven threshold tuning. R1 therefore keeps
the E3K proposal configuration unchanged and diagnoses the TRAIN misses.

## What R1 measures

For every true interior trailing barline, R1 replays the exact frozen E3K
proposal path and records:

1. nearest proposal error in staff spaces;
2. whether the true boundary lies outside the E3K x-search surface;
3. best frozen vertical-coverage evidence near the truth;
4. top and bottom endpoint coverage near the truth;
5. whether an active column existed within 1 staff-space but clustering selected
   a peak farther away;
6. true barline length normalized by staff spacing;
7. true barline perpendicularity error relative to the accepted five-line staff;
8. clean/light/medium degradation profile;
9. absolute rotation bucket;
10. the 100 worst misses for later visual inspection.

Frozen miss reasons are:

- `HIT`
- `TRUTH_OUTSIDE_SEARCH_X`
- `VERTICAL_COVERAGE_FAIL`
- `TOP_ENDPOINT_FAIL`
- `BOTTOM_ENDPOINT_FAIL`
- `BOTH_ENDPOINTS_FAIL`
- `COMBINED_GATE_FAIL`
- `CLUSTER_PEAK_DISPLACEMENT`

## Safety

- TRAIN only;
- VALIDATION unopened;
- TEST unopened;
- no model/checkpoint load;
- no optimizer or training;
- no threshold sweep/tuning;
- no mutation of the E3K proposal algorithm;
- no E3K-B or D11 authorization;
- no production promotion.

R1 exists only to choose the next **single controlled geometry correction** from
TRAIN evidence. After that correction is frozen, TRAIN feasibility must be rerun
before any VALIDATION scoring is considered.
