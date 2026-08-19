# Runtime Deterministic Resolver E2E V2 Contract

## Purpose

V2 replaces the legacy shadow preparation lane with the accepted runtime sequence:

```text
raster bytes
-> Page Normalizer v1
-> Geometry Engine v2
-> System Grouper v1
-> Measure/System Boundaries v2
-> Runtime Local ROI v1
-> byte-bound Meter evidence input
-> Meter Runtime Integration v3
-> Deterministic Resolver v1
-> structured deterministic result
```

The old `runtime_measure_geometry_v1` path is forbidden in this stage.

## System topology policy

System membership is an explicit caller contract. V2 accepts only an existing System Grouper v1 policy (`auto-v1`, `monostaff-v1`, `fixed-two-staff-v1`) and never infers a product topology from Meter evidence or measure count. Any ambiguous grouping stops before Measure/System, ROI, Meter or Resolver work.

## Bound Meter evidence

The resolver E2E stage does not run a model itself. A specialist producer returns a `BoundMeterEvidenceBatchV2` whose records bind:

- exact normalized source-image SHA-256;
- exact canonical `measure-start` ROI ID;
- exact `roi_image_sha256` from `RuntimeRoiArtifact`;
- exact Meter evidence measure/staff/system/logical-measure ownership;
- a provider fingerprint;
- an evidence origin label: `test-fixture` or `external-model`.

The E2E runtime independently matches every bound record against the exact ROI artifacts generated in the same invocation. Missing, extra, duplicate or hash-mismatched records fail closed before Meter composition.

`test-fixture` is allowed only as deterministic CI evidence. It may never be reported as real-model or real-image model proof. `external-model` records are byte-bound but the external provider remains responsible for proving its model/checkpoint/profile identity; this stage does not load that checkpoint.

## Other specialist evidence

NoteHead/Rest/Accidental observations may be supplied as an explicit `SpecialistEvidenceBatch`. They remain model-agnostic and are validated by the Resolver against exact measure/staff geometry. This stage does not manufacture missing symbol evidence.

## Ordering

All output follows geometry order, not lexical string ordering. In particular, `system-10` must not sort before `system-2` merely because of its identifier text.

## Failure semantics

An upstream normalization/geometry/grouping/measure failure stops the lane. A Meter association ambiguity is propagated into Resolver ambiguity through `SpecialistObservation`; it cannot disappear via fallback. Unknown or wrong-owner specialist evidence remains a hard validation failure.

## Determinism

For identical raster bytes, contracts, topology policy and evidence batches:

- normalized image identity;
- system grouping;
- measure geometry;
- ROI hashes;
- Meter decisions;
- Resolver ordering/reasons;
- final E2E fingerprint

must be bit-stable across 10/10 replay.

## Safety boundary

- no TRAIN/VALIDATION/TEST access;
- sealed TEST remains unopened;
- no model/checkpoint loading;
- no optimizer;
- no threshold fitting;
- no old measure-geometry fallback;
- no silent System Grouper fallback;
- no production claim from `test-fixture` Meter evidence;
- no real-image PASS without a real image and externally produced byte-bound model evidence.

## Closure gate

Implementation, unit/integration/regression tests, real raster-byte fixture E2E, 10/10 deterministic replay, no-old-lane inspection, independent shadow review and exact-head CI must pass before merge.