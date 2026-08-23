# ST-OMR Training Lab Status

Status date: **2026-08-23**.

This file is the current stage-status source for the repository. Detailed closed-stage history remains in `ARCHITECTURE.md`; the current merged + shadow/experimental architecture overlay is in `ARCHITECTURE_CURRENT.md`.

## Current authoritative baseline

Merged `main` inspected for this synchronization:

```text
main = a6a40b218a95c72349984ee2aee7262f467021fc
```

That merged baseline includes the deterministic runtime foundation through real D11 Meter Presence inference implementation and the post-PR #76 documentation synchronization.

Later NoteHead/Rest/Accidental shadow acceptance and Meter V3/V4/V5 work remain open/draft evidence unless separately merged. They must not be represented as merged runtime authority.

## Current state summary

| Area | Current status |
|---|---|
| deterministic canonical music / MusicXML / renderer / degradation / dataset pipeline | ✅ Closed / verified foundation |
| Staff / Structure specialist foundation | ✅ Closed / verified |
| NoteHead specialist | ✅ Shadow PASS evidence / not production-wired |
| Rest R4 half/quarter/eighth | ✅ Shadow PASS evidence / deterministic arbitration PASS / Resolver closed |
| Accidental specialist | ✅ Shadow PASS evidence / key-local deterministic contract preserved |
| merged Meter runtime foundation | ✅ through D11 Presence inference implementation |
| Meter V4-5 candidate | ⛔ FINAL_HOLDOUT_FAIL / rejected / holdout consumed |
| Meter V5 clean-domain recovery | 🔄 active draft experiment stack |
| Meter V5-2B | 🔄 current inspected bounded adaptation lane; 298/300 preflight before replacements |
| Rhythm specialist | 🔒 not opened |
| StaffPosition specialist | 🔒 not opened |
| Chord specialist | 🔒 not opened |
| deterministic full musical fusion | 🔒 later gate |
| real specialist Resolver E2E | 🔒 later gate |
| Stage 9 sealed benchmark | 🔒 TEST sealed |
| production / ScoreMosaic promotion | 🔒 not authorized |

## Specialist shadow evidence

### NoteHead

Current shadow evidence:

```text
Center F1  0.9882845985
BBox F1    0.9882845985
Macro F1   0.9855884316
Epochs      10/10
Decision    PASS for shadow acceptance
```

### Rest R4

The old D13-R1 Rest zero-F1 outcome is historical failure evidence, not the current Rest state.

Current value-specific Rest shadow evidence:

| Class | Recall | False-positive reduction | Decision |
|---|---:|---:|---|
| half | 1.000000 | 0.812500 | PASS |
| quarter | 0.989071 | 0.752369 | PASS |
| eighth | 0.985612 | 0.792714 | PASS |

Current Rest architecture:

```text
high-recall proposals
        ↓
half / quarter / eighth verifiers
        ↓
deterministic arbitration
        ↓
exactly one class -> accepted
none / multiple -> AMBIGUOUS
malformed / non-finite -> REJECTED
```

Resolver connection remains CLOSED. Shadow PASS is not production PASS.

### Accidental

Current shadow evidence:

```text
Center F1   0.973754
BBox F1     0.973754
Macro F1    0.956454
Steps        6150/6150
Epochs       10/10
Decision     PASS for shadow acceptance
```

The current real-model class surface remains `sharp|flat|natural`.

## Meter status

### Merged `main`

Merged runtime authority includes:

```text
Page Normalizer
→ Geometry Engine v2
→ Multi-staff geometry
→ System Grouper v1
→ Measure/System Boundaries v2
→ logical measure ownership
→ Runtime measure-start ROI
→ Meter association v3
→ provenance-bound Meter producer
→ real checkpoint audit
→ runtime input provenance
→ historical D11 ROI reconstruction
→ real D11 Presence inference implementation
→ STOP
```

Real digit composition and production Resolver authority remain closed on merged `main`.

### V4-5 historical candidate result

The one-time independent final holdout for the V4-2 candidate is consumed and failed:

```text
accuracy    0.7466666667
macro-F1    0.7444420022
recall(2)   0.58
recall(3)   0.66
recall(4)   1.00
decision    FINAL_HOLDOUT_FAIL
```

The candidate is rejected. The consumed holdout may not be rerun or reused for crop, threshold, calibration, model-selection or parameter tuning.

### V5 recovery lane

V5-0 rejects the contaminated/mixed old Meter dataset surface and moves to a clean `package_ab`-only design with globally unique families and explicit train/val/final-holdout separation.

V5-1 created the bounded TRAIN-only 30-sample human full-meter BBox pilot.

V5-2 scale-up is not the current active optimizer path because domain/transfer evidence required a smaller causal lane.

Current inspected lane: **Meter V5-2B — deterministic 2/3 specialist adaptation**.

Current gate:

```text
300 TRAIN BBoxes target
298/300 staff-containment preflight PASS
2 non-seed TRAIN HOLD rows
        ↓
deterministic same-class replacement
        ↓
new human BBoxes
        ↓
mechanical QA + human visual QA
        ↓
require 300/300 preflight PASS
        ↓
only then may bounded adaptation training open
```

The first 30 seeds are diagnostic-only. VAL is closed. FINAL_HOLDOUT locked. 4-AI is a frozen control. Threshold tuning and hyperparameter sweeps remain forbidden.

## Frozen synthetic foundation

The accepted synthetic curriculum, family isolation and sealed-test rules remain authoritative. Historical D6/D7/D8/D9/D10 identities and metrics remain preserved in their original evidence/contract files and are not rewritten by this status synchronization.

Key enduring rules:

- every derivative inherits family split identity;
- AI does not generate ground truth;
- real spatial labels require human verification and explicit admission;
- independent validators retain veto authority;
- TEST remains sealed for development/tuning;
- checkpoints/datasets stay outside normal Git where required;
- ScoreMosaic uploads and teacher corrections are not automatic training data;
- no online/automatic learning path is enabled.

## Architecture compatibility audit

The 2026-08-23 compatibility audit found no P1/P2 project-level architecture break.

PASS:

- specialist decomposition;
- deterministic arbitration/fusion authority;
- GT authority and provenance;
- family/split isolation;
- sealed TEST policy;
- Rest R4 compatibility with frozen `half|quarter|eighth` V1 scope;
- separation of shadow evidence from Resolver/production authority;
- Meter V5 separation from merged `main` runtime authority;
- consumed final-holdout protection.

FAIL before synchronization:

- current-state documentation drift.

See `ARCHITECTURE_COMPATIBILITY_AUDIT_2026-08-23.md`.

## Safety boundaries

- No direct commits to `main`; changes use branch/PR packages.
- Large datasets/checkpoints remain outside normal Git.
- TEST remains sealed until Stage 9.
- No threshold is relaxed after outcome inspection without a separately frozen development contract.
- Consumed final holdouts are not recycled into tuning.
- Private checkpoint binaries are not committed.
- Resolver production wiring cannot be inferred from shadow PASS.
- Production/ScoreMosaic promotion requires a separate later gate.

## Next gates

1. Meter V5-2B: replace the two HOLD TRAIN rows deterministically, obtain new human BBoxes, rerun mechanical + visual QA, and require 300/300 preflight PASS before training.
2. Rest R4: preserve current PASS shadow evidence; next improvement should be real-score/real-scan shadow validation and Resolver-shadow E2E rather than blind R1-style retraining.
3. Rhythm / StaffPosition / Chord remain separate future specialist packages.
4. Deterministic musical fusion and Stage 9 sealed benchmark remain mandatory before any production decision.
