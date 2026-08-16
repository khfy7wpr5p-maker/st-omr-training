# ST-OMR Training — Current Architecture Overlay

This file records the current active lane without replacing the repository's long-form closed-stage history in `ARCHITECTURE.md`.

## Active pipeline

```text
Canonical ST Music
        ↓
Deterministic MusicXML writer + independent validators
        ↓
Pinned Verovio renderer
        ↓
Controlled degradation
        ↓
Stage 5/6 validated synthetic persistence
        ↓
Stage 7-D0..D9 specialist decomposition history      ✅ CLOSED
        ↓
Stage 7-D10 local ROI derivatives                    ✅ CLOSED
        ↓
Stage 7-D11 barline + meter refiners                  ✅ CLOSED / technical baseline
        ↓
Stage 7-D12 NoteHead/Rest/Accidental GT gate          ✅ CLOSED / TEST SEALED
        ↓
Stage 7-D13-R1 first symbol-specialist training       ⛔ NOT ACCEPTED / BASELINE EVIDENCE
        ↓
Stage 7-D13-R2 specialist refinement                  🔄 ACTIVE ARCHITECTURE
        ↓
Rhythm / StaffPosition / Chord specialists            🔒
        ↓
Deterministic fusion + musical validators              🔒
        ↓
Stage 9 sealed benchmark/candidate gate                🔒 TEST SEALED
```

## Current decision

D13-R1 is retained as experimental evidence, not accepted as the final NoteHead/Rest/Accidental stage.

The available run evidence is enough to reject a blind rerun of the same profile:

- **NoteHead:** strong learning; visual architecture is provisionally kept. The main refinement question is optimizer/checkpoint stability because the minimum-validation-loss epoch and the strongest F1 epoch were not always identical.
- **Rest:** failed materially. The observed ten-epoch run ended with zero center F1, zero bbox F1 and zero macro F1. Rest therefore requires root-cause analysis before another full optimizer run.
- **Accidental:** meaningful learning was visible before interruption, but macro-class performance lagged localization. It is refined before being split automatically.

D13-R2 is governed by `STAGE7D13_R2_REFINEMENT_PLAN.md`.

## R2 architecture principle

The specialist system follows one rule:

> Ask learned models the smallest visual question that is useful; use deterministic composition and musical validation for structure that does not need to be learned.

The target graph is therefore allowed to differ by symbol family rather than forcing every specialist into one identical detector shape.

```text
Score / accepted measure geometry
        ↓
learned local observations
        ├─ NoteHead detection + open|filled
        ├─ Rest presence/localization → Rest type classification
        └─ Accidental presence/localization → optional type classification
        ↓
deterministic association/composition
        ↓
musical validation
```

## D13-R2 approved sequence

```text
R2-0  Freeze D13-R1 evidence
        ↓
R2-1  Rest root-cause audit
        ↓
R2-2  Rest architecture decision
        ↓
R2-3  Specialist-specific ROI / visual-scale decision
        ↓
R2-4  Short TRAIN-only diagnostic gate
        ↓
R2-5  Learning-rate scheduler A/B
        ↓
R2-6  Checkpoint-selection refinement
        ↓
R2-7  Accidental refinement
        ↓
R2-8  NoteHead optimizer/checkpoint refinement
        ↓
R2-9  Safe fast-resume contract
        ↓
R2-10 Product-oriented validation preparation
```

The first implementation package is **Rest Root-Cause Audit**. It is diagnostic/read-only with respect to model parameters. A second full Rest training run is blocked until the audit distinguishes data/label, representation/scale, decoder-threshold, localization and class-separation failure modes.

## Rest R2 candidate graph

The primary candidate, subject to R2-1 evidence, is:

```text
accepted measure / local high-resolution ROI
        ↓
RestPresence + RestLocalization
        ↓
normalized Rest crop
        ↓
RestTypeClassifier
half | quarter | eighth
        ↓
deterministic Rest validator
```

This split is not automatic. If R2-1 finds incorrect labels or destructive scaling, that root cause is fixed before adding model complexity.

## NoteHead R2 policy

R1 demonstrated that the current detector can learn NoteHead strongly. Therefore the default R2 policy is:

```text
current NoteHead detector         KEEP
learning-rate behavior            REFINE / controlled A/B
checkpoint selection              REFINE
open|filled class stability       REVIEW
visual architecture redesign      BLOCKED without new evidence
```

A high specialist F1 remains a specialist metric only; it is not end-to-end PDF-reading accuracy.

## Accidental R2 policy

Accidental is refined before it is split because R1 showed substantial learning.

First candidate:

```text
current Accidental detector
+ frozen scheduler candidate
+ refined checkpoint ranking
+ per-class metrics
```

If class-specific evidence remains weak, especially for the rare class, evaluate:

```text
AccidentalPresence/Localization
        ↓
normalized crop
        ↓
sharp | flat | natural classifier
        ↓
deterministic note-association validator
```

Glyph recognition and association to a note remain separate responsibilities.

## Training-system refinements

### Learning-rate scheduler

R2 treats scheduler policy as part of the reproducible training contract. Candidate schedules are compared under a frozen A/B protocol. A scheduler is admitted only if it improves task stability/quality, not merely scalar validation loss.

Any admitted scheduler must be checkpointed and independently verified together with model state, optimizer state, completed epoch, optimizer-step count, current LR and LR history.

### Checkpoint selection

R1 showed that minimum validation loss can disagree with the strongest F1 epoch. R2 therefore freezes a task-aware ranking rule before authoritative validation.

Candidate order:

1. specialist technical gates must pass;
2. prefer higher macro-class F1;
3. use center/bbox F1 as secondary quality evidence;
4. use validation loss as tie-break/regularization evidence rather than the sole product proxy.

The exact ranking formula must be fixed and tested before the result is known.

### Short diagnostic gate

R2 avoids another hours-long degenerate run. Before full training, a deterministic TRAIN-only diagnostic/dev partition may run a small fixed budget. A model that remains degenerate fails fast and cannot enter the authoritative full run.

The authoritative validation split and sealed TEST must not become architecture-tuning loops.

## Resume architecture

R1 recovery was safe but expensive because restart repeated the complete derivative preflight. R2 should add a fail-closed fast-resume path for immutable artifacts.

Fast resume may bypass the record-by-record scan only when a persisted preflight receipt remains bound to all of:

```text
derivative build id
manifest SHA-256
artifact-binding SHA-256
preflight receipt SHA-256
repository SHA
runtime/dependency fingerprint
training-profile fingerprint
resume checkpoint SHA-256
```

Any mismatch falls back to the full preflight. TEST sealing, family isolation, exact optimizer-step accounting and checkpoint integrity are never weakened.

## Ground-truth authority

Synthetic ground truth remains split into deterministic authorities:

- **symbolic GT:** canonical ST music / deterministic MusicXML;
- **spatial GT:** pinned Verovio geometry replayed through the exact accepted image transform.

R2 does not manufacture labels from model predictions. Real geometry later requires separately admitted, human-verified annotation. ScoreMosaic uploads and teacher corrections remain excluded from automatic training.

## D13 development surface

The accepted D13 derivative surface used by R1 remains evidence, not disposable scratch data:

```text
TRAIN records        9840
VALIDATION records   1224
TOTAL records       11064
persisted images    11062
persisted labels    11064
TEST records            0
```

Target instances:

```text
TRAIN
  NoteHeadSet     38334
  RestSet         10602
  AccidentalSet   22392

VALIDATION
  NoteHeadSet      5232
  RestSet           969
  AccidentalSet     3330
```

R2 may reuse this surface only after the root-cause audit confirms that the relevant labels/transforms are valid for the selected R2 representation. Any new ROI derivative receives a new deterministic identity and independent verification.

## Meter status

D11 Meter remains a technical baseline, not the product-quality endpoint. Meter V2 is intentionally outside D13-R2 implementation scope. After D13-R2 closes, Meter returns to product-quality review and may be decomposed into presence/digit/composer stages if the frozen product thresholds are not met.

## Product-quality direction

The user-visible question is not "is F1 high?" but "did the PDF become correct music?" Later evaluation therefore needs to connect specialist observations to:

```text
symbol detection correctness
symbol class correctness
staff-position / pitch correctness
duration correctness
accidental association correctness
exact note-event correctness
exact measure correctness
critical errors per 10,000 symbols
UNKNOWN / abstain behavior
```

No current specialist validation F1 is to be presented as end-to-end PDF-reading accuracy.

## Safety boundary

D13-R2 does **not** authorize:

- sealed TEST access;
- automatic ScoreMosaic/teacher-correction learning;
- RhythmSet/PitchSet/ChordSet training;
- Meter V2 implementation;
- production integration;
- deletion or rewriting of D13-R1 evidence;
- architecture tuning from sealed-test outcomes.

## Next gate

Implement **Rest Root-Cause Audit** as the first R2 package. Produce read-only diagnostic evidence for label correctness, target scale, decoder behavior, localization and class separation. Only after that evidence selects the failure mode may the next package freeze the Rest R2 architecture and authorize a bounded diagnostic optimizer run.
