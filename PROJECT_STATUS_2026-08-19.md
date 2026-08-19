# ST-OMR Project Status — 2026-08-19

## Runtime architecture status

```text
Raster / Page Normalizer                    ✅
Geometry Engine v2                         ✅
Multi-staff geometry                       ✅
System Grouper v1                          ✅
Measure/System Boundaries v2               ✅
Logical measure ownership                  ✅
Runtime measure-start ROI                  ✅
Meter association v3                       ✅
Provenance-bound Meter producer            ✅ merged via PR #71
Real checkpoint audit                      ✅ merged via PR #72
Meter runtime input provenance             ✅ merged via PR #73
Historical D11 ROI reconstruction           ✅ merged via PR #75
Real D11 Presence inference implementation ✅ merged via PR #76
Actual private D11 checkpoint execution     🟡 next authorized runtime gate
Digit 2/3/4 real inference                 🔒 after D11 real execution evidence
Deterministic Meter composition            🔒 after real digit inference
Resolver real-model E2E                    🔒 later gate
Production promotion                       🔒 not authorized
```

## Merged safety foundation

- PR #71 binds exact ROI/source/checkpoint identities and deterministic request/output provenance.
- PR #72 audits the frozen private D11 + digit 2/3/4 checkpoint identities read-only and preserves conservative probability-to-milli transport.
- PR #73 binds digit slot geometry/crop identity and prevents the narrow runtime ROI from masquerading as the historical D11 crop.
- PR #75 reconstructs the historical D11 256x192 input deterministically from verified normalized source pixels and accepted geometry.
- PR #76 adds the fail-closed real D11 Presence inference boundary and passed exact-head CI #434 before merge.

## Current stage

`runtime_meter_real_inference_v1.py` is now on `main`. It is allowed to audit and execute the exact frozen D11 checkpoint in an authorized runtime against an exact reconstructed historical ROI.

GitHub CI intentionally does not contain private checkpoint binaries. Therefore CI proves the architecture/input/output/determinism contract with synthetic audited state, but it does not constitute actual private-checkpoint execution evidence.

The next gate is a read-only authorized-runtime invocation of `infer_presence_from_checkpoint_v1` using:

1. the exact D11 checkpoint whose SHA-256 is `cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3`; and
2. a real `HistoricalMeterRoiArtifactV1` reconstructed from a verified normalized page and accepted runtime geometry.

Only after that execution is recorded deterministically should digit 2/3/4 real inference be opened.

## Safety invariants still active

- TEST remains sealed for development/tuning.
- No thresholds are tuned in runtime work.
- No checkpoint is modified.
- No private checkpoint binary is committed to GitHub.
- Unsupported or unverifiable identity/provenance fails closed.
- Resolver is not allowed to consume real Meter output until real digit inference and end-to-end provenance are closed.
