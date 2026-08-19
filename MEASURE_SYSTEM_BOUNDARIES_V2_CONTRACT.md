# Measure / System Boundaries v2 Contract

Status: implementation candidate; merge requires exact-head CI and independent shadow review.

## Responsibility

This stage runs only after accepted System Grouper output. It determines, per already-established system, the deterministic geometric measure spans and binds every per-staff measure proposal to one logical system-local measure.

It is geometry only. It does not recognize barline semantics, infer meter, access a model/checkpoint, read TRAIN/VALIDATION/TEST, or invoke the Resolver.

## Input boundary

Required:

- accepted `PageGeometryContract` with canonical systems/staff memberships;
- no existing measure proposals;
- exact normalized grayscale PNG bytes whose SHA-256, dimensions and mode match the geometry contract.

System membership is input truth from the System Grouper. This layer must never regroup staffs.

## Boundary construction

For every system independently:

1. detect strong vertical raster runs spanning each staff under the frozen gray/coverage contract;
2. merge contiguous dark columns into one run;
3. cluster neighbouring vertical runs whose ink gap is at most one staff spacing. This treats double/final barline strokes as one *geometric boundary candidate* rather than inventing a tiny measure between strokes;
4. use the detected staff-line x extents as deterministic implicit system start/end edges;
5. snap any explicit vertical candidate within one staff spacing of an implicit system edge to that edge;
6. for a multi-staff system, require the same boundary cardinality and cross-staff alignment within 0.5 staff spacing;
7. derive one logical system boundary sequence as the deterministic arithmetic mean of aligned member-staff boundaries;
8. create one logical measure between each adjacent boundary pair and one per-staff `MeasureProposalContract` member for that logical measure.

System edges are geometry, not claims that a visible barline exists. Therefore a one-measure system with no internal vertical barline is valid. A missing internal barline becomes fail-closed when corroborating staffs in the same system disagree about that boundary.

## Frozen constants

- vertical dark threshold: `128`;
- minimum vertical coverage: `800/1000`;
- maximum double/final-barline stroke gap for one geometric cluster: `1000/1000` of staff spacing;
- edge snap distance: `1000/1000` of staff spacing;
- maximum cross-staff aligned-boundary delta: `500/1000` of staff spacing;
- minimum logical measure width: `2000/1000` of staff spacing.

These are versioned geometry-contract constants. They are not fitted to TRAIN, VALIDATION, TEST, Meter metrics, or model predictions.

## Canonical reasons

Priority order:

1. `B01_UPSTREAM_GEOMETRY_NOT_ACCEPTED`
2. `B02_MEASURE_GEOMETRY_ALREADY_PRESENT`
3. `B03_SYSTEM_STAFF_MEMBERSHIP_INVALID`
4. `B04_SYSTEM_ORDER_INVALID`
5. `B05_CROSS_STAFF_BOUNDARY_MISMATCH`
6. `B06_MEASURE_TOO_NARROW`

Exact PNG identity/mode/shape corruption is an input-integrity exception, not a musical ambiguity and must never be silently converted into a geometry result.

## Output

Accepted output contains:

- unchanged systems and staffs;
- deterministic per-staff measure proposals;
- logical system-local measure records with:
  - logical measure id;
  - system id;
  - 1-based system-local measure index;
  - left/right x;
  - left/right boundary kind (`system_edge` or `vertical_cluster`);
  - member per-staff measure ids;
- raw/clustered boundary evidence per staff;
- canonical input/output/report fingerprints.

An ambiguous result contains no measure proposals and one canonical reason report.

## Required edge cases

The closure suite must cover:

- system start/end without requiring visible edge barlines;
- first and last measures;
- pickup/anacrusis-shaped short first measure above the minimum-width contract;
- double/final barline stroke clustering;
- missing internal barline on one staff of a multi-staff system;
- cross-staff mismatched barline positions;
- multiple staffs sharing the same logical measure;
- independent measure layouts across a system break;
- page-edge system extent;
- no-internal-barline one-measure system;
- exact 10/10 deterministic replay;
- detector -> System Grouper -> Measure/System v2 integration.

## Fail-closed rules

This stage must not:

- regroup systems;
- use Meter or symbol classes to repair geometry;
- silently drop an unmatched staff boundary;
- turn double-bar strokes into a tiny measure;
- invent an expected measure count;
- weaken a mismatch because a later specialist could compensate;
- access sealed TEST or any training/checkpoint path.
