# Runtime Deterministic Resolver E2E V2 — Shadow Review

## Method

Treat the implementation as incorrect until the accepted contracts, runtime producer identities and adversarial fixtures make the intended behavior unavoidable. A green test suite alone is not proof.

## Round 1 — FAIL

Fresh inspection of the existing Deterministic Resolver found lexical measure ordering:

`system_id -> staff_id -> x -> measure_id`

This can place `system-10` before `system-2` and violates geometry order.

Correction:

- Resolver now derives system order from `geometry.systems` and member-staff order from each system contract;
- x order is applied within a system before staff-member order;
- dedicated 10-system regression proves `system-10` remains after `system-9` even when input measure proposals are reversed;
- ordering policy is included in the Resolver configuration fingerprint.

## Round 2 — FAIL

The PageGeometry contract guarantees referenced IDs exist but does not by itself guarantee bidirectional system/staff ownership. A structurally accepted page can therefore be forged so a staff declares `system-1` while `SystemGeometryContract` places it in `system-2`. It can also structurally contain a non-accepted measure proposal.

Correction:

- Resolver independently requires every system member staff to declare that exact system;
- every staff must have unique exact system ownership;
- every measure must reference a staff declaring the same system;
- every measure entering Resolver must be `accepted`;
- dedicated regressions require malformed membership and non-accepted measures to fail closed;
- membership validation is included in the Resolver configuration fingerprint.

## Round 3 — FAIL

E2E failure reporting referenced non-existent generic `.reasons` attributes on System Grouper and Measure/System report objects. Normal page-level reasons usually short-circuited the expression, but a valid fail-closed implementation must not depend on that incidental condition.

Correction:

- E2E now reads the canonical `active_reasons` report properties.

## Round 4 — PASS WITH RESIDUAL RISK

Re-review confirms:

- actual raster bytes enter Page Normalizer v1 and Geometry Engine v2;
- System Grouper v1 is called with an explicit caller policy; there is no silent topology fallback;
- Measure/System Boundaries v2 is the only measure-boundary producer in the new lane;
- Runtime Local ROI v1 output is carried forward with exact source and ROI hashes;
- Meter evidence cannot enter composition unless its measure-start ROI ownership and `roi_image_sha256` match the ROI produced in the same invocation;
- missing/extra/duplicate/wrong-owner Meter batches fail closed;
- Meter ambiguity propagates into Resolver ambiguity rather than disappearing;
- Deterministic Resolver uses canonical geometry order and independently revalidates system/staff/measure ownership;
- 10/10 full-lane replay is required and covered by a raster-byte integration regression;
- CI test-fixture Meter scores are explicitly incapable of claiming real-model proof;
- source inspection tests check for actual old-lane imports/calls rather than merely matching documentation text;
- no model/checkpoint, optimizer, TRAIN/VALIDATION/TEST, Stage 7-D10/D13 artifact or sealed TEST access is introduced.

### Residual risk 1 — external model execution is not yet proven

The E2E lane validates a byte-bound Meter evidence batch, but the current PR does not itself execute the accepted Meter checkpoints. `evidence_origin=external-model` identifies provider intent; it does not prove the provider loaded the expected checkpoint/profile. `RuntimeResolverE2EV2Result.is_real_model_proof` therefore remains always `False`.

This is **non-blocking for deterministic E2E merge** but **blocking for real-model/real-image production PASS**. The next smoke stage needs a trusted producer that consumes the exact `RuntimeRoiArtifact` bytes and records checkpoint/model/profile identity with the ROI/source hashes.

### Residual risk 2 — symbol specialists remain external

NoteHead/Rest/Accidental evidence can enter the actual Resolver via `other_specialist_evidence`, but this PR does not execute those specialist models. The deterministic Resolver path is real; a full musical recognition claim is not yet justified.

### Residual risk 3 — no active-Meter carry-forward semantics

Meter Integration v3 represents no visible current-measure glyph as class `none`. Resolver v1 preserves that observation but does not yet perform musical active-meter propagation across subsequent measures. This is acceptable for evidence-level E2E but must be addressed by the later composer/musical-validation layer before complete structured-score production.

## Decision

**SHADOW REVIEW: PASS WITH RESIDUAL RISK**

No critical blocker remains for merging the deterministic E2E plumbing after exact-head CI passes and main/base remain unchanged. Real-model/real-image PASS remains locked until the external specialist producer provenance boundary is closed.