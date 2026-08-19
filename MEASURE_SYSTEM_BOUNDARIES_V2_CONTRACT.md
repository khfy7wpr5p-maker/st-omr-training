# Measure / System Boundaries v2 Contract

Status: implementation candidate; merge requires exact-head CI and independent shadow review.

## Responsibility

This stage runs only after accepted System Grouper output. It determines, per already-established system, deterministic geometric measure spans and binds every per-staff measure proposal to one logical system-local measure. It is geometry only: no Meter semantics, model/checkpoint, TRAIN/VALIDATION/TEST, or Resolver access.

## Input boundary

Required:

- accepted `PageGeometryContract` with canonical systems/staff memberships;
- no existing measure proposals;
- exact normalized grayscale PNG bytes whose SHA-256, dimensions and mode match the geometry contract.

System membership is predecessor truth and must never be changed here. Global staff order must be top-to-bottom; every staff must have exactly one system owner; each system member list must follow global staff order; each system bbox must equal the exact union of its member staff bboxes.

## Boundary construction

For every system independently:

1. inspect each staff for strong vertical raster columns;
2. a qualifying column must meet all three structural conditions across the outer staff-line span: at least `800/1000` dark coverage under gray threshold `128`, both first/last span rows dark, and no internal white gap longer than `1 px`;
3. merge contiguous qualifying columns into vertical runs;
4. cluster neighbouring runs whose ink gap is at most one staff spacing so double/final-barline strokes form one *geometric boundary candidate* instead of a tiny false measure;
5. use staff-line x extents as implicit system start/end geometry and exclude explicit candidates within one staff spacing of those edges from the internal sequence;
6. for a multi-staff system, require equal boundary cardinality, equal boundary kinds and x alignment within `0.5` staff spacing;
7. derive each logical system boundary by deterministic arithmetic mean of aligned member-staff boundaries;
8. create one logical measure between adjacent logical boundaries and one per-staff `MeasureProposalContract` member for that logical measure.

System edges are geometry, not a claim that visible edge barlines exist. A one-measure system with no internal vertical separator is valid. Missing/contradictory internal evidence across corroborating staffs is fail-closed.

## Frozen constants

- vertical dark threshold: `128`;
- minimum vertical coverage: `800/1000`;
- maximum internal white gap inside a vertical separator column: `1 px`;
- maximum double/final-barline stroke gap for one geometric cluster: `1000/1000` staff spacing;
- edge snap/exclusion distance: `1000/1000` staff spacing;
- maximum cross-staff aligned-boundary delta: `500/1000` staff spacing;
- minimum logical measure width: `2000/1000` staff spacing.

These are versioned geometry-contract constants, not fitted to TRAIN, VALIDATION, TEST, Meter metrics, or model predictions.

## Canonical reasons

Priority:

1. `B01_UPSTREAM_GEOMETRY_NOT_ACCEPTED`
2. `B02_MEASURE_GEOMETRY_ALREADY_PRESENT`
3. `B03_SYSTEM_STAFF_MEMBERSHIP_INVALID`
4. `B04_SYSTEM_ORDER_INVALID`
5. `B05_CROSS_STAFF_BOUNDARY_MISMATCH`
6. `B06_MEASURE_TOO_NARROW`

Exact PNG identity/mode/shape corruption is an input-integrity exception, not musical ambiguity.

## Output

Accepted output preserves systems/staffs and adds deterministic per-staff measure proposals plus logical system-local measures containing logical id, system id, 1-based index, left/right x, boundary kinds and member measure ids. Reports preserve staff raw/clustered evidence and canonical input/output/report fingerprints.

An ambiguous result contains no measure proposals and one canonical reason report.

## Required edge cases

Closure coverage must include system start/end, first/last measure, pickup-shaped short first measure above the minimum width, double/final barline clustering, near-full note-stem-like vertical ink with internal white gaps, missing internal separator on one member of a multi-staff system, cross-staff mismatch, shared logical measure membership, independent layouts across system breaks, page-edge extent, one-measure/no-internal-separator systems, corrupted predecessor staff/member/system-bbox ordering, 10/10 deterministic replay, and detector → System Grouper → Measure/System integration.

## Fail-closed rules

This stage must not regroup systems, use Meter/symbol classes to repair geometry, silently drop unmatched staff boundaries, turn double strokes into a tiny measure, promote a high-coverage but discontinuous note-stem-like column, invent expected measure count, weaken mismatches because a later specialist may compensate, or access sealed TEST/training/checkpoints.
