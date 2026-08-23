# ST-OMR Architecture Compatibility Audit — 2026-08-23

Status: **PASS WITH DOCUMENTATION DRIFT CORRECTIONS**.

This audit compares the merged `main` architecture, frozen specialist contracts, current shadow specialist evidence, and the active Meter experiment stack. It does not open sealed TEST, modify checkpoints, train a model, or authorize production/runtime promotion.

## Baseline

- authoritative merged `main`: `a6a40b218a95c72349984ee2aee7262f467021fc`
- merged runtime Meter boundary on `main`: real D11 Presence inference implementation through PR #76, status synchronization through PR #77
- Rest/NoteHead/Accidental shadow acceptance evidence: draft PR #64
- current Meter experimental stack: draft PRs #90–#98, with V5-2B in PR #98 as the newest adaptation lane inspected by this audit

## Compatibility verdict

| Boundary | Result | Evidence / interpretation |
|---|---|---|
| Specialist decomposition vs current work | PASS | NoteHead, Rest, Accidental and Meter remain isolated specialist observations rather than one authoritative monolithic predictor. |
| Deterministic authority | PASS | Association, arbitration, pitch/fusion policy and unsupported/ambiguous handling remain deterministic/fail-closed. |
| Ground-truth authority | PASS | Synthetic GT remains canonical-music/renderer derived; real spatial GT remains human-verified; model predictions do not become GT. |
| Split isolation | PASS | TRAIN/VALIDATION roles remain explicit; sealed TEST remains unavailable to development/tuning. |
| Rest R4 vs frozen RestSet scope | PASS | R4 still recognizes the frozen V1 `half|quarter|eighth` Rest surface and uses deterministic class arbitration. |
| Rest R4 vs Resolver authority | PASS | Shadow acceptance does not wire Rest into Resolver and does not grant production authority. |
| Meter V5 vs deterministic Meter architecture | PASS | Current V5 work changes data/domain/adaptation evidence only; thresholds, final-holdout protections and Resolver/production gates remain closed. |
| Meter V4-5 consumed final holdout | PASS | Failed one-shot final-holdout evidence is preserved and explicitly non-reusable for tuning. |
| Current documentation consistency | FAIL before this sync | `STATUS.md`, `ARCHITECTURE_CURRENT.md`, and Stage 8 delta documents described older active stages and Rest R1/R2 recovery state. |
| Production readiness claim | PASS / blocked | No current shadow result is promoted to production OMR evidence. |

## Rest compatibility result

Current Rest shadow architecture:

```text
high-recall Rest proposals
        ↓
half verifier
quarter verifier
eighth verifier
        ↓
deterministic arbitration
        ↓
Rest SpecialistObservation
        ↓
Resolver connection remains CLOSED
```

Frozen shadow evidence from PR #64:

| Class | Recall | False-positive reduction | Decision |
|---|---:|---:|---|
| half | 1.000000 | 0.812500 | PASS |
| quarter | 0.989071 | 0.752369 | PASS |
| eighth | 0.985612 | 0.792714 | PASS |

Arbitration remains fail-closed:

- exactly one accepted class -> accepted Rest class;
- none accepted -> `AMBIGUOUS`;
- multiple accepted -> `AMBIGUOUS`;
- malformed/non-finite evidence -> `REJECTED`;
- threshold boundary and output ordering are deterministic.

This is compatible with the frozen D4 rule that learned models answer bounded visual questions while deterministic code owns composition/validation. Rest is therefore **not an active architecture failure**. The remaining Rest work is runtime/shadow integration and real-image/E2E validation, not a blind repeat of the failed R1 profile.

## Meter compatibility result

The merged `main` runtime remains conservative: real D11 Presence inference is implemented, but real digit composition/Resolver production authority is not granted.

The active V5 experiment stack is development-only. The latest inspected V5-2B state preserves:

- first 30 TRAIN seeds as immutable diagnostic-only examples;
- remaining adaptation data as TRAIN only;
- VAL closed;
- FINAL_HOLDOUT locked;
- 4-AI frozen as control;
- historical thresholds unchanged;
- no hyperparameter sweep;
- no Resolver wiring;
- no production promotion.

The V5-2B preflight had 298/300 accepted rows and exactly two deterministic replacement-required HOLD rows. Training must remain blocked until replacement human boxes, mechanical QA, human visual QA, and 300/300 preflight PASS are re-established.

## Documentation drift found

The audit found four material documentation problems:

1. `STATUS.md` still named D10 as the active lane although `main` and later draft work had advanced far beyond it.
2. `ARCHITECTURE_CURRENT.md` still described Rest as blocked at the R1 failure/root-cause-audit stage even though R4 shadow evidence passed.
3. `ARCHITECTURE.md` contained historical Stage 8 status labels that could be read as current operational status.
4. `ARCHITECTURE_STAGE8_3A.md` and `ARCHITECTURE_STAGE8_3A_ADAPTER.md` described Stage 8-3A as current/active without a clear historical-snapshot warning.

These are documentation-state incompatibilities, not model/runtime contract failures.

## Authority order after synchronization

Use the documents in this order:

1. frozen machine-readable contracts and sealed stage contracts define invariants;
2. merged `main` defines executable repository authority;
3. `ARCHITECTURE_CURRENT.md` defines the current merged + shadow/experimental overlay;
4. open draft PR evidence is explicitly non-main and non-production;
5. historical architecture delta documents remain evidence for their original stage and must not silently override the current overlay.

## Safety result

The audit found no evidence that current Rest or Meter work violates these project-level invariants:

- AI does not create ground truth;
- deterministic validators retain veto authority;
- sealed TEST is not a tuning loop;
- family/split leakage is prohibited;
- private checkpoints remain outside GitHub;
- unresolved provenance fails closed;
- shadow PASS is not production PASS;
- production/ScoreMosaic integration remains a separate later gate.

## Final verdict

**Architecture compatibility: PASS.**

**Documentation consistency before this synchronization: FAIL.**

**Required correction:** synchronize the live architecture/status documents while preserving frozen historical contracts and explicitly distinguishing merged `main` from draft/shadow evidence.
