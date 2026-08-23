# ST-OMR Training — Current Architecture Overlay

Status date: **2026-08-23**.

This file is the operational current-state overlay. It does not rewrite frozen historical contracts in `ARCHITECTURE.md` or stage-specific contract files.

## Authority model

Use architecture evidence in this order:

1. frozen machine-readable/stage contracts define invariants;
2. merged `main` defines executable repository authority;
3. this file records the current merged + shadow/experimental overlay;
4. open draft PR evidence is non-main, non-production evidence only;
5. historical stage-delta documents remain evidence for their original stage and cannot override this current overlay.

## Merged `main` authority

Exact merged baseline inspected for this synchronization:

```text
main = a6a40b218a95c72349984ee2aee7262f467021fc
```

Merged runtime architecture on `main`:

```text
Raster / page input
        ↓
Page Normalizer
        ↓
Geometry Engine v2
        ↓
Multi-staff geometry
        ↓
System Grouper v1
        ↓
Measure/System Boundaries v2
        ↓
Logical measure ownership
        ↓
Runtime measure-start ROI
        ↓
Meter association v3
        ↓
Provenance-bound Meter producer
        ↓
Real checkpoint audit
        ↓
Historical D11 ROI reconstruction
        ↓
Real D11 Presence inference implementation
        ↓
STOP — real digit composition / Resolver production authority not granted
```

`main` therefore contains a verified runtime foundation, but it does not contain the later draft Meter V4/V5 experiment stack as merged authority.

## Specialist architecture

The frozen specialist principle remains unchanged:

> Ask learned models the smallest visual question that is useful; use deterministic composition and musical validation for structure that does not need to be learned.

Current high-level specialist graph:

```text
accepted page / staff / measure geometry
        ↓
learned bounded observations
        ├─ NoteHead
        ├─ Rest
        ├─ Accidental
        └─ Meter
        ↓
deterministic association / arbitration / composition
        ↓
fail-closed musical validation
        ↓
future canonical candidate music
```

AI does not create ground truth. Deterministic validators retain veto authority.

## Shadow / experimental overlay

The following evidence is newer than the merged `main` baseline and must not be confused with merged runtime authority.

### NoteHead

Current shadow evidence records the real epoch-10 NoteHead artifact as PASS:

```text
Center F1  0.9882845985
BBox F1    0.9882845985
Macro F1   0.9855884316
Epochs      10/10
```

This is specialist/shadow evidence, not end-to-end PDF-reading accuracy.

### Rest R4

The old D13-R1 Rest failure is historical baseline evidence only. The current Rest path is the later value-specific R4 shadow architecture:

```text
high-recall Rest proposals
        ↓
half verifier
quarter verifier
eighth verifier
        ↓
deterministic class arbitration
        ↓
Rest SpecialistObservation
        ↓
Resolver connection remains CLOSED
```

Frozen shadow evidence from draft PR #64:

| Rest class | Recall | False-positive reduction | Shadow decision |
|---|---:|---:|---|
| half | 1.000000 | 0.812500 | PASS |
| quarter | 0.989071 | 0.752369 | PASS |
| eighth | 0.985612 | 0.792714 | PASS |

Deterministic arbitration:

```text
exactly one class accepted -> accepted Rest class
none accepted              -> AMBIGUOUS
multiple accepted          -> AMBIGUOUS
malformed / NaN / Inf       -> REJECTED
```

The arbitration contract is deterministic and repeatable. Rest is therefore no longer an active architecture-recovery failure.

Important boundary: **shadow PASS is not production PASS**. Rest remains unwired to the production Resolver, sealed TEST remains closed, and real-image/end-to-end validation is still required before any promotion decision.

### Accidental

Current shadow evidence records the authoritative epoch-10 Accidental resume as PASS:

```text
Center F1   0.973754
BBox F1     0.973754
Macro F1    0.956454
Steps        6150/6150
Epochs       10/10
```

The key-signature/local deterministic association contract also remains separate from glyph recognition. Current real-model classes are `sharp|flat|natural`; double-sharp/double-flat are not silently claimed.

### Meter

Meter has two different status layers that must remain separate.

**Merged main:** runtime D11 Presence inference boundary exists, while real digit composition and Resolver promotion remain closed.

**Draft experiment stack:** later V3/V4/V5 work explores real-domain adaptation and package/domain transfer without changing production authority.

Important preserved evidence:

- V4-5 one-time final holdout was consumed and failed; that candidate is rejected and the consumed holdout cannot be reused for tuning.
- V5-0 rejects the contaminated/mixed old `METER_V2_1500` surface and preregisters a clean `package_ab`-only replacement.
- V5-1 freezes a 30-sample TRAIN-only human full-meter BBox pilot.
- V5-2 scale-up is HOLD while the causal transfer problem is isolated.
- current inspected adaptation lane is **Meter V5-2B** in draft PR #98.

V5-2B current gate:

```text
300 TRAIN human full-meter BBoxes target
        ↓
center-y unique accepted-staff containment
        ↓
298/300 preflight PASS
        ↓
2 non-seed TRAIN rows HOLD
        ↓
deterministic same-class replacements + new human BBoxes
        ↓
mechanical QA + human visual QA
        ↓
require 300/300 preflight PASS
        ↓
only then may bounded 2-AI / 3-AI adaptation training open
```

The first 30 seeds remain diagnostic-only with zero gradient updates. VAL is closed. FINAL_HOLDOUT locked. 4-AI remains frozen control. Threshold tuning and hyperparameter sweeps are forbidden. Resolver and production promotion remain closed.

## Compatibility result

The 2026-08-23 project-level incompatibility audit found:

| Invariant | Result |
|---|---|
| specialist decomposition preserved | PASS |
| deterministic fusion/arbitration authority preserved | PASS |
| AI-generated ground truth forbidden | PASS |
| family/split leakage controls preserved | PASS |
| sealed TEST excluded from development tuning | PASS |
| Rest R4 compatible with frozen `half|quarter|eighth` V1 scope | PASS |
| Rest shadow acceptance separated from Resolver authority | PASS |
| Meter V5 experiment separated from merged runtime authority | PASS |
| consumed final-holdout protection preserved | PASS |
| documentation current-state consistency before sync | FAIL -> corrected by this package |

See `ARCHITECTURE_COMPATIBILITY_AUDIT_2026-08-23.md` for the full evidence map.

## Current project pipeline

```text
Canonical ST Music + validators                         ✅
        ↓
Deterministic MusicXML + pinned renderer               ✅
        ↓
Controlled degradation + validated synthetic data      ✅
        ↓
Staff / Structure geometry foundation                  ✅
        ↓
NoteHead shadow specialist                             ✅ PASS evidence
Rest R4 value-specific specialists                     ✅ PASS evidence
Accidental shadow specialist                           ✅ PASS evidence
Meter merged runtime foundation                        ✅
Meter V5 real-domain adaptation lane                   🔄 DRAFT / TRAIN-SAFE ONLY
        ↓
Rhythm specialist                                      🔒
StaffPosition specialist                               🔒
Chord grouping specialist                              🔒
        ↓
Deterministic duration / pitch / association fusion    🔒
        ↓
Real specialist Resolver E2E                           🔒
        ↓
Stage 9 sealed benchmark                               🔒 TEST SEALED
        ↓
Production / ScoreMosaic promotion                     🔒 NOT AUTHORIZED
```

The green specialist rows above mean accepted current **shadow evidence**, not merged production authority unless explicitly stated.

## Rest next-quality direction

Rest should no longer be developed by blindly repeating the failed R1 10-epoch profile. Future Rest quality work should proceed only through isolated evidence gates:

```text
R4 shadow artifacts
        ↓
real-score / real-scan shadow validation
        ↓
false-negative / false-positive error buckets
        ↓
small evidence-selected data/ROI/model correction only if required
        ↓
deterministic duration + meter consistency validation
        ↓
Resolver shadow E2E
        ↓
sealed benchmark
```

Additional rest semantics such as full/multi-measure rests remain outside frozen V1 and require a separately approved scope expansion.

## Meter next gate

Do not train from the current V5-2B surface until:

1. the two HOLD rows are replaced by the deterministic same-class rule;
2. new human full-meter BBoxes are supplied for those replacements;
3. mechanical audit passes;
4. human visual QA passes;
5. the staff-containment preflight returns 300/300 PASS.

No VAL/final-holdout opening is implied by that gate.

## Ground-truth authority

Synthetic ground truth remains split into deterministic authorities:

- symbolic GT: canonical ST music / deterministic MusicXML;
- spatial GT: pinned renderer geometry replayed through the exact accepted image transform.

Real spatial ground truth requires separately admitted human verification. ScoreMosaic uploads and teacher corrections remain excluded from automatic training.

## Safety boundary

This current overlay does **not** authorize:

- sealed TEST access;
- reuse of consumed final holdout evidence for tuning;
- threshold relaxation after observing outcomes;
- automatic ScoreMosaic/teacher-correction learning;
- private checkpoint publication;
- Resolver production wiring from shadow evidence;
- production promotion;
- deleting or rewriting historical failed-run evidence.

## Current next gates

1. finish the bounded Meter V5-2B 300/300 preflight before any adaptation optimizer work;
2. keep Rest R4 frozen as PASS shadow evidence and move its next work to real-image/shadow E2E validation rather than blind retraining;
3. later open Rhythm / StaffPosition / Chord work only under separate contracts;
4. preserve deterministic musical fusion and Stage 9 sealed-test authority before any ScoreMosaic promotion.
