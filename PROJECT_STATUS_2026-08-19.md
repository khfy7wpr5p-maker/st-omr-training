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
Real D11 Presence inference implementation 🟡 current stage
Digit 2/3/4 real inference                 🔒 next stage
Deterministic Meter composition            🔒 after real digit inference
Resolver real-model E2E                    🔒 later gate
Production promotion                       🔒 not authorized
```

## Merged safety foundation

- PR #71 binds exact ROI/source/checkpoint identities and deterministic request/output provenance.
- PR #72 audits the frozen private D11 + digit 2/3/4 checkpoint identities read-only and preserves conservative probability-to-milli transport.
- PR #73 binds digit slot geometry/crop identity and prevents the narrow runtime ROI from masquerading as the historical D11 crop.
- PR #75 reconstructs the historical D11 256x192 input deterministically from verified normalized source pixels and accepted geometry.

## Current stage

`runtime_meter_real_inference_v1.py` is the first runtime implementation that may execute the audited real D11 checkpoint. It keeps digit specialists, Meter composition, Resolver wiring and production promotion closed.

CI tests use synthetic audited state only because private checkpoint binaries are intentionally not committed to GitHub. Actual real-checkpoint execution must occur in an authorized runtime with the exact local checkpoint path and exact reconstructed historical ROI.

## Safety invariants still active

- TEST remains sealed for development/tuning.
- No thresholds are tuned in runtime work.
- No checkpoint is modified.
- No private checkpoint binary is committed to GitHub.
- Unsupported or unverifiable identity/provenance fails closed.
- Resolver is not allowed to consume real Meter output until real digit inference and end-to-end provenance are closed.
