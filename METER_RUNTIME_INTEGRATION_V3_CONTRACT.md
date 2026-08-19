# Meter Runtime Integration V3 Contract

## Purpose

This stage binds Meter specialist evidence to the deterministic runtime geometry that is already accepted by System Grouper v1 and Measure/System Boundaries v2.

Runtime boundary:

```text
accepted Measure/System v2 geometry
+ exact Measure/System v2 logical-measure report
+ exact measure-start Runtime Local ROI
+ externally computed frozen Meter presence/digit evidence
-> deterministic Meter association/composition
-> model-agnostic SpecialistEvidenceBatch
```

This stage does **not** load a model/checkpoint, train, access TRAIN/VALIDATION/TEST, tune thresholds, or connect the Deterministic Resolver. Model execution remains an external specialist boundary; this module validates identity, geometry, ownership and deterministic composition only.

## Frozen specialist evidence

No threshold is fitted in this stage. Existing frozen shadow evidence is reused unchanged:

- visual Meter presence threshold: `0.90` from the prior M3-B development rule;
- digit-2 threshold: `480/1000`;
- digit-3 threshold: `600/1000`;
- digit-4 threshold: `470/1000`.

The digit specialists remain independent. A slot is accepted only when exactly one of the 2/3/4 specialists passes its frozen threshold. Multiple passing specialists are ambiguous. No passing specialist means that visual digit was not found.

## Ownership and identity

For every accepted per-staff measure proposal, exactly one Meter evidence record is required. It must bind all of:

- `system_id`;
- `logical_measure_id`;
- `measure_id`;
- `staff_id`;
- exact canonical `measure-start` `roi_id`;
- normalized source-image SHA through the ROI batch;
- exact Runtime Local ROI producer configuration;
- exact producer crop geometry and source-to-ROI translation.

`logical_measure_id` is not inferred from measure numbering. It is taken from the accepted Measure/System v2 report. The report is independently checked against the supplied geometry: measure ownership, member staffs, logical x extents, measure indices, system edges, and complete non-duplicated measure coverage must all agree.

A wrong system, measure, staff, logical measure, ROI, source image, ROI producer configuration, or non-measure-start presence region fails closed. Extra unowned ROI/evidence records and duplicate evidence identities also fail closed.

### External model-evidence provenance boundary

V3 deliberately receives already-computed presence/digit scores rather than model binaries. `MeterModelEvidenceV3` therefore proves logical ownership by IDs but does not itself cryptographically prove which ROI bytes a model evaluated. The `RuntimeRoiArtifact` bytes are independently hash-checked by the ROI contract, but the supplied score object is a trusted external specialist-output boundary in this PR.

This is acceptable only for the deterministic association stage. **Production Resolver/real-image promotion is forbidden until a later trusted specialist producer constructs Meter evidence directly from the exact `RuntimeRoiArtifact` and binds at least its `roi_image_sha256` and source-image identity.** Injected score objects used by CI are test evidence, not real-model runtime proof.

## Slot geometry

Meter digit slots do not re-detect staff lines inside the ROI. The caller supplies only the deterministic horizontal anchor `refined_x_center_roi`; vertical placement is derived from the five accepted upstream staff lines:

- numerator center: second staff line;
- denominator center: fourth staff line;
- slot width / staff spacing: `1.5960569245912566`;
- slot height / staff spacing: `2.0`.

These ratios are the unchanged TRAIN-derived values already frozen by the prior Meter shadow work. They are not retuned here.

The Runtime Local ROI transform and integer crop must exactly reproduce `runtime_local_roi_v1`, including floor-min/ceil-max rounding and page clipping. Derived source-space Meter boxes must remain inside both the produced measure-start ROI and the true owning measure.

## Composition

Presence absent + no passing digit evidence -> accepted `none`.

Presence present requires one numerator and one denominator digit. Supported V3 visible compositions are exactly:

```text
2/4
3/4
4/4
```

The denominator must therefore resolve to `4`. Unsupported combinations fail ambiguous; no musical correction is invented.

For multi-staff logical measures, accepted Meter classes must agree across all member measures. A cross-staff disagreement marks the logical measure ambiguous. V3 does not silently support polymeter; differing simultaneous staff meters remain explicit ambiguity.

## Canonical reason priority

```text
M01_UPSTREAM_GEOMETRY_NOT_ACCEPTED
M02_BOUNDARY_REPORT_MISMATCH
M03_METER_EVIDENCE_MISSING
M04_WRONG_PRESENCE_REGION
M05_IDENTITY_MISMATCH
M06_PRESENCE_AMBIGUOUS
M07_PRESENCE_REJECTED
M08_NO_DIGIT
M09_UPPER_DIGIT_NOT_FOUND
M10_LOWER_DIGIT_NOT_FOUND
M11_DIGIT_SPECIALIST_CONFLICT
M12_PRESENCE_DIGIT_CONFLICT
M13_SLOT_GEOMETRY_AMBIGUOUS
M14_UNSUPPORTED_COMPOSITION
M15_CROSS_STAFF_METER_MISMATCH
```

Reason lists are unique and sorted only by this priority. The same input must reproduce the same decisions, ordering, reason codes, output evidence and report fingerprint.

## Safety boundary

- no model/checkpoint loading;
- no optimizer;
- no threshold fitting;
- no TRAIN/VALIDATION/TEST access;
- no sealed TEST access;
- no System Grouper or Measure/System mutation;
- no duplicate Meter-local staff detector;
- no semantic barline overwrite;
- no Resolver wiring in this stage;
- no fallback to the old `runtime_measure_geometry_v1` lane;
- ambiguous/malformed evidence is never converted to an accepted Meter class;
- externally supplied score objects are not evidence of real-model execution and cannot authorize production/real-image promotion.

## Closure gate

V3 is merge-eligible only after contract/implementation review, unit + integration + regression tests, 10/10 deterministic replay, cross-staff and wrong-ownership adversarial fixtures, shadow review, and exact-head CI PASS.

Merge of this association layer does not itself authorize Resolver production wiring. The external specialist-output byte-binding risk must be closed by the later E2E specialist producer before production or real-image PASS can be claimed.