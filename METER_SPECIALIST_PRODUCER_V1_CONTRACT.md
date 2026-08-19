# Meter Specialist Producer V1 Contract

## Purpose

This stage closes the specific byte-provenance gap left after Meter Runtime Integration V3.

The accepted runtime chain is:

```text
exact RuntimeRoiArtifact (measure-start)
+ immutable ROI PNG bytes
+ source_image_sha256
+ roi_image_sha256
+ frozen Meter specialist checkpoint identities
-> audited Meter inference runner
-> provenance-bound Meter evidence
```

The producer passes only `RuntimeRoiArtifact.png_bytes` to the inference runner. It independently verifies the ROI byte hash before and after inference and binds the resulting evidence to source-image, ROI-image, specialist-profile, request and output fingerprints.

## Scope boundary

This PR does **not** load any checkpoint itself. It introduces the trusted runtime producer contract and byte-binding adapter only.

Forbidden in this stage:

- TRAIN access;
- VALIDATION access;
- TEST access;
- threshold tuning;
- optimizer or backward pass;
- D10/D13 derivative loading;
- old measure-geometry fallback;
- Resolver wiring;
- production promotion based on CI fixtures alone.

A later stage may provide an audited real checkpoint runner, but that runner must implement this exact byte-input contract and must not obtain pixels from any alternate path.

## Required identities

Every produced Meter evidence object is bound to:

- `system_id`;
- `logical_measure_id`;
- `measure_id`;
- `staff_id`;
- canonical `measure-start` `roi_id`;
- normalized `source_image_sha256`;
- exact `roi_image_sha256`;
- presence checkpoint SHA-256;
- digit-2 checkpoint SHA-256;
- digit-3 checkpoint SHA-256;
- digit-4 checkpoint SHA-256;
- specialist-profile fingerprint;
- inference-request fingerprint;
- inference-output fingerprint.

Any missing, malformed or changed identity fails closed.

## Byte ownership rule

The producer accepts only `RuntimeRoiArtifact` with:

```text
kind == measure-start
roi_id == <measure_id>:measure-start
sha256(png_bytes) == roi_image_sha256
```

The runner receives exactly one pixel payload: `roi.png_bytes`.

The producer hashes the same bytes again after the runner returns. Any mutation or mismatch aborts evidence production.

## Specialist profile

V1 treats Meter as four frozen specialist identities:

```text
presence
2-digit
3-digit
4-digit
```

Their checkpoint SHA-256 values are part of one canonical specialist-profile fingerprint. Changing any checkpoint identity changes the profile fingerprint and therefore changes the inference request identity.

## Inference output

The audited runner returns only the raw Meter contract:

- presence status and score;
- optional refined x-center inside ROI;
- numerator independent 2/3/4 scores;
- denominator independent 2/3/4 scores;
- explicit reasons for non-accepted presence.

The producer does not apply Meter thresholds or musical composition. Those remain owned by `runtime_meter_integration_v3`.

## Determinism

Same:

```text
ROI bytes
+ source/ROI identities
+ specialist profile
+ raw inference output
```

must produce the same request fingerprint, output fingerprint, evidence ID and provenance record in 10/10 repeats.

## Fail-closed policy

The producer must refuse to create evidence when:

- ROI is not canonical `measure-start`;
- ROI bytes do not match `roi_image_sha256`;
- source/ROI/checkpoint SHA identity is malformed;
- system/logical-measure identity is empty;
- runner returns an object outside the frozen raw inference contract;
- ROI byte identity changes during inference.

No failed producer request may be silently converted into accepted Meter evidence.

## Security interpretation

This stage proves which immutable ROI byte payload was passed into the audited runner and which frozen specialist profile was declared for that request. It does not by itself prove that an arbitrary third-party callback internally honored its declared checkpoint identities.

Therefore **production promotion remains closed until the real checkpoint runner is itself audited and tested against the exact checkpoint files/SHAs**. CI fake runners are contract tests only, not real-model E2E evidence.

## Closure gate

Merge eligibility requires:

- contract/implementation agreement;
- canonical measure-start enforcement;
- byte hash verification before and after inference;
- checkpoint/profile identity binding;
- malformed/wrong runner fail-closed tests;
- 10/10 deterministic replay;
- full repository CI PASS;
- shadow review of the trust boundary.

Merge of V1 authorizes only the provenance-bound producer contract. It does not authorize real checkpoint loading, Resolver wiring, sealed TEST access or production promotion.
