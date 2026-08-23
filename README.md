# st-omr-training

Safe training and synthetic-data laboratory for ST-OMR: canonical music generation, MusicXML serialization and validation, deterministic notation rendering, controlled degradation, dataset validation/construction, training, specialist experiments, and evaluation.

This repository is isolated from the ScoreMosaic production runtime. Its purpose is to build traceable OMR training data, train candidate ST-OMR models, and evaluate them before any later integration decision.

## Project documents

- [Long-form architecture](ARCHITECTURE.md)
- [Current architecture overlay](ARCHITECTURE_CURRENT.md)
- [2026-08-23 architecture compatibility audit](ARCHITECTURE_COMPATIBILITY_AUDIT_2026-08-23.md)
- [Canonical data contract](DATA_CONTRACT.md)
- [MusicXML contract](MUSICXML_CONTRACT.md)
- [Renderer contract](RENDERER_CONTRACT.md)
- [Controlled degradation contract](DEGRADATION_CONTRACT.md)
- [Synthetic dataset manifest contract](DATASET_CONTRACT.md)
- [Synthetic Dataset v1 construction contract](DATASET_BUILD_CONTRACT.md)
- [Baseline ST-OMR training contract](TRAINING_CONTRACT.md)
- [Stage 7-B training implementation profile](TRAINING_IMPLEMENTATION.md)
- [Stage 7-C bounded run profile](STAGE7C_RUNBOOK.md)
- [Stage 7-C accepted evidence](STAGE7C_EVIDENCE.md)
- [Stage 8-0 real-data and fine-tuning contract](STAGE8_REAL_DATA_CONTRACT.md)
- [Stage 8-1 quarantine/intake byte-validation contract](STAGE8_1_INTAKE_CONTRACT.md)
- [Stage 8-2 paired experiment run profile](STAGE8_2_RUN_PROFILE.md)
- [Stage 8-3A historical pilot architecture delta](ARCHITECTURE_STAGE8_3A.md)
- [Future real-test sealing boundary](STAGE8_TEST_SEALING_BOUNDARY.md)
- [Verovio runtime evidence](VEROVIO_RUNTIME_EVIDENCE.md)
- [Safety and verification rules](SAFETY.md)
- [Current project status](STATUS.md)

## Current phase

Status date: **2026-08-23**.

Merged `main` baseline used by the current architecture synchronization:

```text
a6a40b218a95c72349984ee2aee7262f467021fc
```

The merged repository contains the deterministic/synthetic foundation, Staff/Structure specialist work, runtime geometry/measure ownership, provenance-bound Meter association/producer infrastructure, historical D11 ROI reconstruction, and the real D11 Presence inference implementation.

Later specialist and Meter results exist in open/draft branches and must not be confused with merged runtime authority.

### Current specialist evidence

| Specialist | Current evidence | Runtime/production authority |
|---|---|---|
| NoteHead | shadow PASS | not production-wired |
| Rest R4 `half|quarter|eighth` | shadow PASS; deterministic class arbitration PASS | Resolver connection closed |
| Accidental `sharp|flat|natural` | shadow PASS | production connection closed |
| Meter | merged runtime foundation + active V5 draft recovery work | real composition/Resolver promotion closed |
| Rhythm | not opened | locked |
| StaffPosition | not opened | locked |
| Chord | not opened | locked |

Rest's old D13-R1 zero-F1 result remains historical failure evidence. The current Rest R4 value-specific path has PASS shadow evidence and should move next toward real-image/shadow E2E validation rather than a blind repeat of the failed R1 profile.

### Meter current development boundary

The V4-5 candidate failed its one-time independent final holdout and is rejected. That consumed holdout is not reusable for tuning.

The current V5 recovery direction separates clean `package_ab` data, human spatial GT, domain-transfer diagnostics, and bounded specialist adaptation. The newest inspected lane is Meter V5-2B in draft PR #98:

```text
300 TRAIN human full-meter BBoxes target
298/300 staff-containment preflight PASS
2 deterministic replacement-required HOLD rows
        ↓
new human BBoxes + mechanical QA + visual QA
        ↓
require 300/300 preflight PASS
        ↓
only then bounded 2-AI / 3-AI adaptation may open
```

The first 30 seeds remain diagnostic-only. VAL is closed. FINAL_HOLDOUT is locked. 4-AI remains frozen control. Threshold tuning, Resolver wiring and production promotion are not authorized.

## Architecture authority rule

Current-status interpretation is explicit:

1. frozen contracts define invariants;
2. merged `main` defines executable repository authority;
3. `ARCHITECTURE_CURRENT.md` records the current merged + shadow/experimental overlay;
4. open draft PR evidence is non-main/non-production until separately merged and accepted;
5. historical stage delta documents preserve their original stage and do not override the current overlay.

The 2026-08-23 compatibility audit found no P1/P2 project-level architecture break. The material failure found was documentation drift, which this synchronization package corrects.

## Core development rule

Each stage stays isolated behind explicit contracts and validation gates. Symbolic musical ground truth remains authoritative; rendering and visual degradation are derived artifacts and must never silently modify target notation semantics.

All derivatives of one symbolic or real-source family remain in one dataset family and one train/validation/test split. Builders never validate themselves by assertion; independent validation remains a veto gate.

AI/model predictions do not become ground truth. Synthetic spatial labels require pinned-renderer geometry lineage; real spatial labels require explicit human verification and admission.

Both synthetic and real held-out test partitions remain sealed until the Stage 9 benchmark gate. A shadow specialist PASS does not authorize Resolver production wiring or ScoreMosaic promotion.

ScoreMosaic user uploads and teacher corrections are not automatic training data. User-derived material requires separate explicit training permission and the full quarantine/admission process. There is no online or automatic learning path.

Large datasets, model checkpoints, real user documents, private material, and rights-unclear score collections are not normal Git repository content.
