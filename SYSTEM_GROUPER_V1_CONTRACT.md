# Runtime System Grouper v1 Contract

Status: implementation candidate under PR #67; merge requires exact-head CI and shadow review.

## Responsibility

The System Grouper runs after accepted multi-staff detection and before measure geometry. The upstream detector's provisional `system-1` membership is not musical truth.

Input:

- accepted `PageGeometryContract` with staff observations and no measure proposals;
- optional exact normalized grayscale PNG bytes for `auto-v1`;
- one explicit grouping policy.

Output:

- accepted page with deterministic system membership; or
- `page=None` plus one canonical ambiguous/rejected report.

No model, checkpoint, optimizer, TRAIN, VALIDATION, TEST, Meter, or Resolver state is used.

## Policies

### `auto-v1`

- one detected staff: accept one system;
- multiple staffs: exact normalized raster bytes are required and hash/shape/mode bound to the page;
- inspect only a bounded corridor around adjacent staff-line left endpoints;
- a pair is positively connected only when one dark column covers at least 800/1000 of the inter-staff gap under the frozen gray threshold 128;
- multiple staffs are automatically accepted as one system only when **every adjacent pair** has positive connector evidence;
- missing connector evidence is never interpreted as proof of a system boundary;
- partial/all-absent connector patterns return `AMBIGUOUS` rather than being split by spacing, x-alignment, SVG metadata, or a fitted threshold.

This deliberately does not claim generic image-only multi-system/multi-staff partitioning when the raster is underdetermined.

### `monostaff-v1`

Current ScoreMosaic V1 declared music scope. Every detected graphical staff instance is one system, ordered top-to-bottom. This handles multi-system single-staff pages without guessing from layout distance.

### `fixed-two-staff-v1`

Explicit caller-declared grand-staff policy. Adjacent ordered staff pairs form systems. Odd staff count is ambiguous. This is not inferred from raster, SVG grouping metadata, Meter, or model output.

## Canonical reason priority

1. `G01_UPSTREAM_GEOMETRY_NOT_ACCEPTED`
2. `G02_MEASURE_GEOMETRY_ALREADY_PRESENT`
3. `G05_INVALID_STAFF_ORDER`
4. `G03_DECLARED_POLICY_STAFF_COUNT_MISMATCH`
5. `G06_RASTER_EVIDENCE_REQUIRED`
6. `G04_UNDERDETERMINED_MULTISTAFF_MEMBERSHIP`

Reports preserve one primary reason, canonical secondary ordering, adjacent connector coverages in integer milli fractions, input geometry fingerprint, and accepted output geometry fingerprint.

## Determinism and fail-closed requirements

For identical page geometry, normalized raster bytes, and policy:

- system count and membership must be identical;
- system/staff order must be identical;
- report reason ordering must be identical;
- connector evidence must be identical;
- page/report fingerprints must be identical.

The implementation must not silently reorder noncanonical input, regroup after measure proposals exist, trust provisional upstream membership, use absence of a connector as a boundary, or fall back to another policy.

## Evidence boundary

Merged System Geometry evidence already falsified standalone spacing, x-overlap, measure-boundary alignment, renderer grouping metadata, broad left darkness, and the initial topology corridor as generic rules. PR #67 also preserves an adversarial repeated-system fixture showing staff-line left endpoint alignment can occur across `DIFFERENT_SYSTEM` systems. Those cues remain rejected as standalone automatic grouping rules.

## Closure gate

System Grouper v1 may be marked complete for its declared supported policies only when:

- this contract and implementation agree;
- unit, integration, regression, edge-case and 10/10 deterministic tests pass;
- raster identity mismatch fails closed;
- grand-staff positive connector and partial/no-connector ambiguity tests pass;
- previous System Geometry adversarial regressions remain green;
- independent shadow review is PASS or PASS WITH RESIDUAL RISK with no critical blocker;
- exact-head required CI and compile pass;
- base is fresh and conflict-free.

Residual unsupported generic layouts must remain explicit `AMBIGUOUS`; they are not grounds to fabricate a system assignment.
