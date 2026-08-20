# Meter V2 Slot Geometry Adapter v1

Status: `DESIGN-FROZEN CANDIDATE / SHADOW-ONLY / NOT PRODUCTION-ACCEPTED`

## Purpose

Eliminate duplicate staff detection inside Meter V2.  Meter slot geometry must consume the already-accepted runtime geometry produced upstream:

```text
Multi-Staff Geometry v2
    -> staff_id + five_staff_lines + staff_spacing
Measure Geometry v1
    -> accepted measure proposal
Runtime Local ROI v1
    -> measure-start ROI + source_to_roi transform
Meter Slot Geometry Adapter v1
    -> numerator bbox + denominator bbox
2-AI / 3-AI / 4-AI
    -> visual digit evidence
Deterministic Meter Composer
    -> none | 2/4 | 3/4 | 4/4 | fail closed
```

The adapter does **not** infer meter class, detect staff lines from ROI pixels, load a model, train, read sealed TEST, or connect to the runtime Deterministic Resolver.

## Frozen input boundary

The adapter accepts:

- accepted `PageGeometryContract` from the existing runtime geometry lane;
- one `RuntimeRoiArtifact(kind="measure-start")`;
- the existing TRAIN-derived deterministic horizontal refinement `refined_x_center_roi`.

It requires exact identity agreement for `normalized_image_sha256`, `staff_id`, `measure_id`, and the Runtime Local ROI crop/transform.

`runtime_local_roi_v1` currently emits a translation-only `source_to_roi` transform.  Adapter v1 intentionally supports only that exact transform.  A different transform is rejected rather than guessed.

## Staff-relative slot rule

No measure-index special case exists.

- numerator vertical center = second upstream staff line at the refined x position;
- denominator vertical center = fourth upstream staff line at the refined x position;
- local staff spacing = mean of the four upstream line gaps at that x position;
- slot width = `1.5960569245912566 * local_staff_spacing`;
- slot height = `2.0 * local_staff_spacing`.

The width/height ratios are the unchanged TRAIN-derived candidate values already used by the staff-relative shadow experiment.  They were not altered from the later VALIDATION outcome.

## Fail-closed rules

The adapter returns no boxes unless the contract is accepted.

- upstream page geometry not accepted -> `AMBIGUOUS`;
- non-finite/out-of-range x -> `REJECTED`;
- non-measure-start ROI -> `REJECTED`;
- source identity mismatch -> `REJECTED`;
- missing staff/measure identity -> `REJECTED`;
- unsupported ROI transform -> `REJECTED`;
- refined x outside upstream staff-line support -> `AMBIGUOUS`;
- non-ordered/non-positive local line gaps -> `AMBIGUOUS`;
- computed numerator/denominator box outside ROI -> `AMBIGUOUS`.

Confidence is never used to override geometry conflicts.

## Safety boundary

Frozen for this shadow step:

- no modification to `Multi-Staff Geometry v2`;
- no modification to `Measure Geometry v1`;
- no modification to `Runtime Local ROI v1`;
- no local staff re-detection inside Meter;
- no validation-derived parameter tuning;
- no model/checkpoint access in the adapter;
- no training or optimizer;
- no sealed TEST access;
- no Resolver wiring;
- no production promotion;
- no merge authorization.

## Evidence before any production claim

The next evidence gate is a shadow replay that feeds **actual upstream runtime staff geometry and Runtime Local ROI transforms** into this adapter, then runs the already-frozen 2-AI / 3-AI / 4-AI and deterministic composer.

Acceptance must be compared against the existing shadow evidence without changing constants from VALIDATION:

- previous runtime-slot v1: `1100/1224` exact;
- experimental ROI-local staff-relative v2: `1155/1224` exact, one wrong accepted;
- GT-slot upper bound: `1202/1224` exact.

The adapter is not production-accepted merely because unit tests pass.  Real upstream-geometry shadow replay, focused/full CI, and a separate explicit promotion/wiring decision remain required.
