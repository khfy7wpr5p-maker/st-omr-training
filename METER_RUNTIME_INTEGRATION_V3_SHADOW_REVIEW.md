# Meter Runtime Integration V3 — Shadow Review

## Review method

The review treats the implementation as untrusted and attempts to falsify it against the V3 contract, accepted runtime producer contracts and adversarial fixture ownership. Passing unit tests are not treated as proof by themselves.

## Round 1 — FAIL

The first CI-clean implementation was rejected by shadow review because it could still accept structurally plausible but untrusted ownership data.

Findings:

1. Runtime ROI batch config identity was not checked against the owning geometry.
2. A non-canonical ROI ID/shape could masquerade as `measure-start`.
3. Measure/System logical membership was trusted too shallowly; a forged logical report could keep the geometry fingerprint while changing logical bounds.
4. Duplicate Meter evidence IDs were not rejected.
5. Output order used lexical IDs rather than canonical geometry order.

Corrections:

- exact Runtime ROI config fingerprint validation;
- canonical `measure_id:measure-start` identity;
- deep logical-report revalidation against every accepted measure/system/staff;
- duplicate evidence-ID rejection;
- geometry-order output.

## Round 2 — FAIL

The hardened implementation was again treated as untrusted.

Findings:

1. A strict ROI-inside-float-measure check disagreed with the actual `runtime_local_roi_v1` producer, which intentionally floors minima and ceils maxima. Legitimate fractional geometry could therefore be rejected.
2. An accepted page contract can structurally contain a non-accepted measure proposal; Meter association must not attach a semantic Meter to such a measure.
3. Extra unowned Meter evidence or ROI artifacts could otherwise be ignored rather than invalidating the supplied association batch.

Corrections:

- independently reproduce the exact producer crop rule: 12 staff spacings, floor-min/ceil-max and page clipping;
- require every Measure/System v2 proposal used by the accepted report to be accepted;
- reject ghost/unowned evidence and ROI artifacts;
- independently revalidate measure→staff→system ownership.

Dedicated regressions cover fractional producer rounding, non-accepted measures, ghost evidence and ghost ROI artifacts.

## Round 3 — PASS WITH RESIDUAL RISK

Re-review confirms:

- contract ↔ implementation ownership semantics agree;
- output ordering follows canonical geometry order, including non-lexical system IDs;
- wrong system/logical-measure/measure/staff/ROI ownership fails closed;
- wrong presence region does not fall back to another ROI;
- malformed or missing Meter evidence stays explicit and propagates as ambiguous specialist evidence;
- `none` cannot bypass exact ROI identity checks;
- cross-staff disagreement marks the entire logical measure ambiguous;
- digit-specialist conflicts remain ambiguous;
- unsupported compositions are not musically repaired;
- 10/10 replay fingerprint is stable on the deterministic fixture surface;
- no model/checkpoint/optimizer/TRAIN/VALIDATION/TEST/Resolver or old-measure-lane dependency is introduced.

### Residual risk: external model-output byte provenance

`MeterModelEvidenceV3` is an externally supplied score object. V3 checks its logical IDs against the exact accepted runtime ROI, but the score object itself does not carry a cryptographic `roi_image_sha256` proving the model evaluated those exact bytes.

This does **not** invalidate the association-only stage because model execution is explicitly outside this PR. It **does block production/real-image promotion** until the later E2E specialist producer constructs evidence from the exact `RuntimeRoiArtifact` and binds the source/ROI byte identities. CI-injected scores must never be described as real-model E2E evidence.

### Residual risk: polymeter

V3 requires simultaneous staff members of one logical measure to agree on Meter. True polymeter is therefore represented as ambiguity rather than accepted asymmetric Meter. This is fail-closed and non-blocking for the current V1 scope.

## Decision

**SHADOW REVIEW: PASS WITH RESIDUAL RISK**

No blocker remains for merging the deterministic Meter association layer after exact-head CI passes and the base remains current. The external specialist-output byte provenance risk remains an explicit blocker for production Resolver/real-image promotion, not for this association-only PR.