# Meter Real Specialist Runner V1 — Safety Contract

Status: **CHECKPOINTS VERIFIED / RUNTIME PIXEL CONTRACT HOLD / RESOLVER CLOSED**

## Purpose

This stage is the approved read-only bridge from the already-frozen Meter checkpoints to the provenance-bound producer introduced by PR #71.

The intended path is:

```text
exact runtime measure-start ROI bytes
+ accepted geometry ownership
+ frozen Presence checkpoint
+ frozen 2/3/4 checkpoints
-> audited inference-only runner
-> provenance-bound raw Meter evidence
-> Meter Runtime Integration v3
```

This stage must not train, tune thresholds, open TEST, mutate checkpoints, wire the Deterministic Resolver, or promote production behavior.

## Frozen checkpoint identities

The authoritative read-only identities are:

- D11 Meter technical baseline / temporary Presence bridge: `cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3`
- 2-AI: `92b985d989e4338e3ae39b0a984879f4188be32c0d281390839117e1e9a715fa`
- 3-AI: `5ee45faf2efe0e2c83dbad716736d7ae16ad7251730431d368c10c4574836485`
- 4-AI: `dcd582b60b39e65798aa77aacea3cc797cd7513b7925151f0573be4aec6af43f`

All four files were re-read from the private Drive evidence surface in this stage. Their bytes matched these SHA-256 identities before any model-state inspection. No checkpoint bytes are committed to GitHub.

## Read-only checkpoint rule

A runtime checkpoint loader is admissible only when all of the following hold:

1. path is a regular non-symlink file;
2. byte length is inside the frozen bound;
3. SHA-256 exactly equals the frozen identity;
4. `torch.load(..., map_location="cpu", weights_only=True)` succeeds;
5. the required model state exists;
6. state keys and tensor shapes exactly match the frozen architecture;
7. every model tensor is finite;
8. no optimizer state is applied and no parameter is mutated by training.

The historical digit checkpoint may contain optimizer/RNG fields. The runtime audit reads only `model_state_dict`; those other fields never authorize optimizer construction or resume.

## Verified frozen architectures

### 2-AI / 3-AI / 4-AI

All three verified checkpoints expose the same model state:

```text
Conv2d 1->16, k3, padding1
ReLU
MaxPool2d 2
Conv2d 16->32, k3, padding1
ReLU
MaxPool2d 2
Conv2d 32->64, k3, padding1
ReLU
AdaptiveAvgPool2d(1,1)
Linear 64->1
```

Historical digit preprocessing remains frozen as grayscale `64x64`, aspect-preserving LANCZOS thumbnail without upscaling, centered on white, tensor semantics `uint8 / 255.0`.

### D11 Meter / temporary Presence bridge

The verified D11 checkpoint contains `meter_state_dict` matching the frozen D11 MeterRefiner architecture: convolutional encoder, `6x8` adaptive pool, FC64, four-class classifier and bbox head. Presence is the already-frozen shadow bridge `1 - P(none)`; this stage does not invent a new Presence threshold.

## Shadow-review finding: direct runtime inference is not yet trustworthy

Checkpoint verification alone is not enough to call the real runner complete. Fresh contract review found two independent runtime-input gaps.

### H01 — Presence pixel-contract mismatch

The D11 Meter checkpoint was trained on the D9/D10 `measure-start-meter-roi-v1` policy, whose frozen output is `192x256` and whose crop includes geometry-conditioned margins (`0.5` staff spacing before measure start and vertical margins of `3` staff spacings).

The current `RuntimeRoiArtifact(kind="measure-start")` is a different runtime crop contract: it begins at the accepted measure left edge, extends at most `12` staff spacings to the right, and uses the accepted measure bbox vertically.

The missing outside-ROI pixels cannot be reconstructed from `RuntimeRoiArtifact.png_bytes`. Resizing the new ROI and pretending it is the historical D11 input would therefore be an unproven pixel adapter and is forbidden.

**Decision:** D11 Presence checkpoint is SHA/architecture verified, but real Presence inference from the new runtime measure-start ROI is `HOLD_PIXEL_ADAPTER_REQUIRED`.

### H02 — Digit-slot context is not bound by the PR #71 request fingerprint

The digit specialists consume tight 64x64 candidate digit crops. Their pixel transform is frozen, but the candidate numerator/denominator boxes must be derived from admitted geometry/localization.

PR #71 currently fingerprints ROI identity plus specialist-profile identity, while its opaque runner receives only ROI bytes. A runner could close over different slot boxes without those boxes changing the inference-request fingerprint. That is a provenance hole.

Also, the current Meter v3 evidence carries `refined_x_center_roi`; when that x anchor comes from Presence/D11 output, slot creation is inherently two-phase:

```text
Presence inference -> x anchor -> deterministic slot geometry -> 2/3/4 inference
```

A one-call opaque `runner(bytes)` must not hide that dependency.

**Decision:** real digit inference is `HOLD_CONTEXT_BINDING_REQUIRED` until exact slot boxes/crop-profile identity are included in the provenance request.

## Score boundary

Historical digit thresholds were defined on exact sigmoid probabilities. Meter Integration v3 currently transports digit scores as integer milli values. Any real adapter must use a conservative quantization rule that never promotes a sub-threshold float into an accepted milli threshold, or the contract must be extended to preserve exact score semantics. Ordinary rounding is forbidden because it can promote a score below `0.48`, `0.60`, or `0.47` across the frozen boundary.

## What this stage authorizes now

Authorized:

- exact SHA verification of the four frozen checkpoints;
- strict model-state/schema inspection;
- CPU inference-only loader implementation;
- deterministic synthetic/smoke tests of the loader boundary;
- design/fix of the missing Presence pixel adapter and digit-slot provenance binding.

Not authorized:

- calling resized runtime ROI bytes equivalent to D11 training input;
- inventing digit boxes from fixture-specific constants;
- threshold tuning;
- TRAIN/VALIDATION/TEST data access for development decisions;
- Resolver wiring;
- production promotion;
- merge based only on checkpoint-load success.

## Closure gate

Real Meter Specialist Runner v1 can move from HOLD to PASS only when:

1. exact new-runtime -> Presence pixel semantics are proven without alternate/unbound pixels, or a separately admitted Presence specialist/input contract replaces the temporary D11 bridge;
2. numerator/denominator slot geometry and digit-crop profile are cryptographically bound into the inference request;
3. exact checkpoint SHA + strict state schema are enforced before inference;
4. output shapes/finiteness and fail-closed behavior are tested;
5. score transport preserves the frozen decision boundary;
6. 10/10 replay is deterministic;
7. adversarial/shadow review passes;
8. full CI passes.

Until those conditions close, status remains **HOLD** and no real-model E2E/Resolver claim is permitted.
